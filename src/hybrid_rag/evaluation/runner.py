"""Offline, deterministic benchmark execution over ``RetrievalService``."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from time import perf_counter_ns
from uuid import uuid4

from hybrid_rag.evaluation.contracts import (
    BenchmarkCase,
    BlindJudgment,
    BlindLabel,
    BlindWinner,
    ComparisonSummary,
    CostDisclosure,
    CostStatus,
    EvaluationBenchmark,
    EvaluationOptions,
    EvaluationReport,
    EvaluationRun,
    ExpectedEvidence,
    IndexProvenance,
    JudgeProvenance,
    ModeSummary,
    PairwiseJudgment,
    RetrievalEvaluation,
)
from hybrid_rag.evaluation.judge import BlindJudge, DeterministicBlindJudge, blind_comparison
from hybrid_rag.ids import canonical_json_hash
from hybrid_rag.retrieval.models import ContextItem, RetrievalMode
from hybrid_rag.retrieval.query import EvidenceItem, deterministic_answer
from hybrid_rag.retrieval.service import RetrievalOptions, RetrievalService
from hybrid_rag.storage.retrieval_repository import StoredIndexProfile


class EvaluationRunner:
    """Compare fixed benchmark cases using only deterministic local operations.

    The active index is resolved once, then its immutable profile ID is passed
    to every retrieval.  Each retrieval persists a replayable ``rtr_`` trace;
    evaluation results therefore remain inspectable even after the active
    profile changes.  The offline answerer deliberately returns selected source
    text verbatim, making citation checks a reproducible proxy rather than a
    claim of LLM faithfulness.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        *,
        judge: BlindJudge | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.judge = judge
        self._fallback_judge = DeterministicBlindJudge()

    def run(
        self,
        benchmark: EvaluationBenchmark,
        *,
        options: EvaluationOptions | None = None,
        cost_disclosure: CostDisclosure | None = None,
        profile_ref: str | None = None,
    ) -> EvaluationReport:
        """Evaluate selected cases across modes and return a serializable report."""

        effective = options or EvaluationOptions()
        cases = _selected_cases(benchmark, effective)
        profile = self.retrieval_service.resolve_profile(profile_ref)
        provenance = _index_provenance(profile)
        if benchmark.expected_source_corpus_hash != provenance.corpus_content_hash:
            raise ValueError(
                "benchmark expected_source_corpus_hash does not match the pinned profile's "
                f"graph-independent corpus content hash {provenance.profile_id}: expected "
                f"{benchmark.expected_source_corpus_hash}, observed "
                f"{provenance.corpus_content_hash}"
            )
        stable_run_id = _run_id(benchmark, effective, cases, provenance)
        execution_id = f"evx_{uuid4().hex}"
        retrieval_options = RetrievalOptions(
            top_k=effective.top_k,
            candidate_multiplier=effective.candidate_multiplier,
            context_token_budget=effective.context_token_budget,
            graph_max_hops=effective.graph_max_hops,
            naive_weight=effective.naive_weight,
            local_weight=effective.local_weight,
            global_weight=effective.global_weight,
            naive_dense_weight=effective.naive_dense_weight,
            naive_bm25_weight=effective.naive_bm25_weight,
            bm25_k1=effective.bm25_k1,
            bm25_b=effective.bm25_b,
            reranker_provider=effective.reranker_provider,
            reranker_model=effective.reranker_model,
            rerank_candidate_multiplier=effective.rerank_candidate_multiplier,
        )
        evaluations: list[RetrievalEvaluation] = []
        by_case: dict[str, dict[RetrievalMode, RetrievalEvaluation]] = defaultdict(dict)
        for case in cases:
            for mode in effective.modes:
                measurement = self._evaluate_case(
                    case,
                    mode=mode,
                    retrieval_options=retrieval_options,
                    profile_id=provenance.profile_id,
                    model_info={
                        "evaluation_run_id": stable_run_id,
                        "evaluation_execution_id": execution_id,
                        "evaluation_benchmark_id": benchmark.id,
                    },
                )
                evaluations.append(measurement)
                by_case[case.id][mode] = measurement

        pairwise_judgments = tuple(
            self._judge_case(
                benchmark_id=benchmark.id,
                case=case,
                naive=by_case[case.id][RetrievalMode.NAIVE],
                hybrid=by_case[case.id][RetrievalMode.HYBRID],
            )
            for case in cases
        )
        run = EvaluationRun(
            id=stable_run_id,
            benchmark_id=benchmark.id,
            options=effective,
            case_ids=tuple(case.id for case in cases),
            index_provenance=provenance,
            execution_id=execution_id,
        )
        effective_cost = self._effective_cost_disclosure(
            supplied=cost_disclosure,
            evaluations=evaluations,
            pairwise_judgments=pairwise_judgments,
            index_provenance=provenance,
        )
        return EvaluationReport(
            run=run,
            evaluations=tuple(evaluations),
            pairwise_judgments=pairwise_judgments,
            summaries=_summaries(evaluations, effective.modes),
            comparison_summary=_comparison_summary(pairwise_judgments),
            cost_disclosure=effective_cost,
            judge_provenance=self._judge_provenance(),
        )

    def _evaluate_case(
        self,
        case: BenchmarkCase,
        *,
        mode: RetrievalMode,
        retrieval_options: RetrievalOptions,
        profile_id: str,
        model_info: dict[str, str],
    ) -> RetrievalEvaluation:
        started_ns = perf_counter_ns()
        result = self.retrieval_service.retrieve(
            case.question,
            mode=mode,
            options=retrieval_options,
            profile_ref=profile_id,
            keywords=case.keywords,
            persist=True,
            model_info=model_info,
        )
        if result.trace_id is None:
            raise RuntimeError("evaluation retrieval did not return a persisted replay trace ID")
        retrieval_finished_ns = perf_counter_ns()
        evidence = tuple(
            EvidenceItem(
                citation_id=item.citation_id,
                text=item.text,
                source_chunk_ids=(item.chunk_id,),
            )
            for item in result.context_items
        )
        answer = deterministic_answer(case.question, evidence)
        finished_ns = perf_counter_ns()

        matched = _matching_expected(case.expected_evidence, result.context_items)
        citation_context = tuple(
            item for item in result.context_items if item.citation_id in set(answer.citations)
        )
        cited = _matching_expected(case.expected_evidence, citation_context)
        allowed_citations = {item.citation_id for item in result.context_items}
        citation_allowlist_valid = set(answer.citations).issubset(allowed_citations)
        answer_supported = _answer_is_supported(answer.answer, citation_context)
        faithfulness = (
            citation_allowlist_valid
            and (
                answer.insufficient_evidence
                or (not answer.insufficient_evidence and answer_supported)
            )
        )
        expected_ids = tuple(anchor.id for anchor in case.expected_evidence)
        return RetrievalEvaluation(
            case_id=case.id,
            mode=mode,
            retrieval=result.trace,
            retrieval_trace_id=result.trace_id,
            answer=answer,
            retrieval_latency_ms=_elapsed_ms(started_ns, retrieval_finished_ns),
            total_latency_ms=_elapsed_ms(started_ns, finished_ns),
            expected_evidence_ids=expected_ids,
            matched_evidence_ids=matched,
            cited_evidence_ids=cited,
            evidence_hit_rate=len(matched) / len(expected_ids),
            cited_evidence_hit_rate=len(cited) / len(expected_ids),
            citation_allowlist_valid=citation_allowlist_valid,
            answer_supported_by_citation=answer_supported,
            citation_grounded_faithfulness=faithfulness,
            abstained=answer.insufficient_evidence,
        )

    def _judge_case(
        self,
        *,
        benchmark_id: str,
        case: BenchmarkCase,
        naive: RetrievalEvaluation,
        hybrid: RetrievalEvaluation,
    ) -> PairwiseJudgment:
        comparison, label_to_mode = blind_comparison(
            benchmark_id=benchmark_id,
            case_id=case.id,
            question=case.question,
            naive=naive,
            hybrid=hybrid,
        )
        used_fallback = False
        fallback_reason: str | None = None
        if self.judge is None:
            judgment = self._fallback_judge.judge(comparison)
            protocol = self._fallback_judge.protocol
            used_fallback = True
            fallback_reason = "no external blind judge configured"
        else:
            try:
                judgment = self.judge.judge(comparison)
                if not isinstance(judgment, BlindJudgment):
                    raise TypeError("blind judge must return BlindJudgment")
                protocol = self.judge.protocol
                if not isinstance(protocol, str) or not protocol.strip():
                    raise TypeError("blind judge protocol must be a non-blank string")
            except Exception as error:
                judgment = self._fallback_judge.judge(comparison)
                protocol = self._fallback_judge.protocol
                used_fallback = True
                fallback_reason = _fallback_reason(error)
        winner_mode = None
        if judgment.winner is not BlindWinner.TIE:
            winner_label = BlindLabel(judgment.winner.value)
            winner_mode = label_to_mode[winner_label]
        return PairwiseJudgment(
            case_id=case.id,
            comparison=comparison,
            judgment=judgment,
            label_to_mode=label_to_mode,
            winner_mode=winner_mode,
            protocol=protocol,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
        )

    def _effective_cost_disclosure(
        self,
        *,
        supplied: CostDisclosure | None,
        evaluations: list[RetrievalEvaluation],
        pairwise_judgments: tuple[PairwiseJudgment, ...],
        index_provenance: IndexProvenance,
    ) -> CostDisclosure:
        """Apply conservative cost rules after all external work is observable."""

        has_external_judge_fallback = self.judge is not None and any(
            judgment.used_fallback for judgment in pairwise_judgments
        )
        if has_external_judge_fallback:
            return CostDisclosure.unknown_judge_fallback(
                retrieval_model_calls=(
                    supplied.retrieval_model_calls
                    if supplied is not None
                    else _retrieval_model_calls(index_provenance, evaluations)
                ),
                judge_model_calls=(
                    supplied.judge_model_calls if supplied is not None else None
                ),
            )
        if _uses_external_embedding(index_provenance):
            if supplied is not None and supplied.status is CostStatus.VERIFIED:
                return supplied
            return CostDisclosure.unknown_external_embedding(
                provider=index_provenance.embedding_provider,
                retrieval_model_calls=len(evaluations),
                judge_model_calls=None if self.judge is not None else 0,
            )
        if supplied is not None:
            return supplied
        if self.judge is None:
            return CostDisclosure.offline()
        return CostDisclosure.unknown_external_judge()

    def _judge_provenance(self) -> JudgeProvenance:
        if self.judge is None:
            return JudgeProvenance(
                provider="deterministic",
                protocol=self._fallback_judge.protocol,
                external=False,
            )
        configured = getattr(self.judge, "provenance", None)
        if isinstance(configured, JudgeProvenance):
            return configured
        protocol = getattr(self.judge, "protocol", None)
        if not isinstance(protocol, str) or not protocol.strip():
            protocol = type(self.judge).__name__
        return JudgeProvenance(
            provider=type(self.judge).__name__,
            protocol=protocol,
            external=True,
        )


def _selected_cases(
    benchmark: EvaluationBenchmark,
    options: EvaluationOptions,
) -> tuple[BenchmarkCase, ...]:
    if not options.case_ids:
        return benchmark.cases
    by_id = {case.id: case for case in benchmark.cases}
    missing = [case_id for case_id in options.case_ids if case_id not in by_id]
    if missing:
        missing_ids = ", ".join(missing)
        raise ValueError(f"evaluation options reference unknown benchmark cases: {missing_ids}")
    return tuple(by_id[case_id] for case_id in options.case_ids)


def _matching_expected(
    expected: tuple[ExpectedEvidence, ...],
    context: tuple[ContextItem, ...],
) -> tuple[str, ...]:
    matched: list[str] = []
    for anchor in expected:
        if any(_matches_anchor(anchor, item) for item in context):
            matched.append(anchor.id)
    return tuple(matched)


def _matches_anchor(anchor: ExpectedEvidence, item: ContextItem) -> bool:
    if anchor.document_title is not None and _normalize(anchor.document_title) != _normalize(
        item.document_title
    ):
        return False
    return _normalize(anchor.text) in _normalize(item.text)


def _answer_is_supported(answer: str, cited_context: tuple[ContextItem, ...]) -> bool:
    normalized_answer = _normalize(answer)
    return bool(normalized_answer) and any(
        normalized_answer in _normalize(item.text) for item in cited_context
    )


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _elapsed_ms(started_ns: int, finished_ns: int) -> float:
    return (finished_ns - started_ns) / 1_000_000


def _run_id(
    benchmark: EvaluationBenchmark,
    options: EvaluationOptions,
    cases: tuple[BenchmarkCase, ...],
    index_provenance: IndexProvenance,
) -> str:
    digest = canonical_json_hash(
        {
            "benchmark": benchmark.model_dump(mode="json"),
            "options_hash": options.config_hash,
            "case_ids": [case.id for case in cases],
            "index_provenance": index_provenance.model_dump(mode="json"),
        }
    )
    return f"evr_{digest[:16]}"


def _index_provenance(profile: StoredIndexProfile) -> IndexProvenance:
    """Project stored-profile data into a serializable evaluation contract."""

    metadata = profile.metadata
    corpus_content_hash = metadata.get("corpus_content_hash")
    if not isinstance(corpus_content_hash, str):
        raise ValueError(
            f"pinned index profile {profile.id} lacks corpus_content_hash; rebuild the index"
        )
    graph_hash = metadata.get("graph_corpus_hash")
    return IndexProvenance(
        profile_id=profile.id,
        index_config_hash=profile.config_hash,
        embedding_provider=profile.provider,
        embedding_model=profile.model,
        embedding_dimensions=profile.dimensions,
        index_schema_version=profile.schema_version,
        corpus_content_hash=corpus_content_hash,
        source_corpus_hash=profile.source_corpus_hash,
        source_graph_run_id=profile.source_graph_run_id,
        source_graph_corpus_hash=graph_hash,
    )


def _uses_external_embedding(index_provenance: IndexProvenance) -> bool:
    """The current adapter boundary has one explicitly local provider: ``hash``."""

    return index_provenance.embedding_provider.casefold() != "hash"


def _retrieval_model_calls(
    index_provenance: IndexProvenance,
    evaluations: list[RetrievalEvaluation],
) -> int:
    return len(evaluations) if _uses_external_embedding(index_provenance) else 0


def _summaries(
    evaluations: list[RetrievalEvaluation],
    modes: tuple[RetrievalMode, ...],
) -> tuple[ModeSummary, ...]:
    output: list[ModeSummary] = []
    for mode in modes:
        values = [item for item in evaluations if item.mode is mode]
        output.append(
            ModeSummary(
                mode=mode,
                cases=len(values),
                mean_evidence_hit_rate=_mean(item.evidence_hit_rate for item in values),
                mean_cited_evidence_hit_rate=_mean(
                    item.cited_evidence_hit_rate for item in values
                ),
                citation_grounded_faithfulness_rate=_mean(
                    float(item.citation_grounded_faithfulness) for item in values
                ),
                mean_retrieval_latency_ms=_mean(item.retrieval_latency_ms for item in values),
                median_retrieval_latency_ms=float(
                    median(item.retrieval_latency_ms for item in values)
                ),
                mean_total_latency_ms=_mean(item.total_latency_ms for item in values),
            )
        )
    return tuple(output)


def _comparison_summary(judgments: tuple[PairwiseJudgment, ...]) -> ComparisonSummary:
    naive_wins = sum(item.winner_mode is RetrievalMode.NAIVE for item in judgments)
    hybrid_wins = sum(item.winner_mode is RetrievalMode.HYBRID for item in judgments)
    ties = sum(item.winner_mode is None for item in judgments)
    total = len(judgments)
    return ComparisonSummary(
        naive_wins=naive_wins,
        hybrid_wins=hybrid_wins,
        ties=ties,
        naive_win_rate=naive_wins / total,
        hybrid_win_rate=hybrid_wins / total,
        tie_rate=ties / total,
    )


def _mean(values: object) -> float:
    numbers = tuple(float(value) for value in values)  # type: ignore[union-attr]
    if not numbers:
        raise ValueError("cannot calculate an evaluation summary from zero values")
    return sum(numbers) / len(numbers)


def _fallback_reason(error: Exception) -> str:
    message = " ".join(str(error).split())
    detail = message if message else type(error).__name__
    return f"external blind judge failed: {detail}"[:1_000]
