"""Strict, serializable contracts for reproducible retrieval evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from hybrid_rag.ids import canonical_json_hash
from hybrid_rag.retrieval.models import RetrievalMode, RetrievalTrace
from hybrid_rag.retrieval.query import GroundedAnswer

BENCHMARK_SCHEMA_VERSION = "1"
EVALUATION_SCHEMA_VERSION = "1"

NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
BenchmarkId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=128,
        pattern=r"^[a-z][a-z0-9-]*$",
    ),
]
CaseId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=128,
        pattern=r"^[a-z][a-z0-9-]*$",
    ),
]
EvidenceId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=128,
        pattern=r"^[a-z][a-z0-9-]*$",
    ),
]
CorpusHash = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=64,
        max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    ),
]
EvaluationExecutionId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=16,
        max_length=96,
        pattern=r"^evx_[a-f0-9]{12,64}$",
    ),
]


class EvaluationCategory(StrEnum):
    """Question families required by the fixed benchmark."""

    FACT = "fact"
    COMPARISON = "comparison"
    RELATION = "relation"
    CROSS_DOCUMENT = "cross_document"


class BlindLabel(StrEnum):
    """Mode-free labels shown to a pairwise judge."""

    A = "A"
    B = "B"


class BlindWinner(StrEnum):
    """The only legal outcomes of a blind pairwise comparison."""

    A = "A"
    B = "B"
    TIE = "tie"


class CostStatus(StrEnum):
    """Whether a report's dollar value is usable for comparison."""

    NOT_APPLICABLE = "not_applicable"
    VERIFIED = "verified"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class _StrictEvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class ExpectedEvidence(_StrictEvaluationModel):
    """One required, source-text anchor for an evaluation case.

    Anchors deliberately avoid generated chunk IDs.  This makes the benchmark
    portable across deterministic chunk configurations while still measuring
    whether the returned context contains the required source fact.
    """

    id: EvidenceId
    text: NonBlankText = Field(max_length=2_000)
    document_title: str | None = Field(default=None, min_length=1, max_length=512)


class BenchmarkCase(_StrictEvaluationModel):
    """A fixed question with source-text evidence expectations."""

    id: CaseId
    question: NonBlankText = Field(max_length=2_000)
    category: EvaluationCategory
    expected_evidence: tuple[ExpectedEvidence, ...] = Field(min_length=1, max_length=8)
    reference_answer: NonBlankText = Field(max_length=10_000)
    keywords: tuple[NonBlankText, ...] = Field(default=(), max_length=12)
    tags: tuple[NonBlankText, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def validate_unique_members(self) -> BenchmarkCase:
        _reject_duplicate_ids(self.expected_evidence, field_name="expected_evidence")
        _reject_casefold_duplicates(self.keywords, field_name="keywords")
        _reject_casefold_duplicates(self.tags, field_name="tags")
        return self


class EvaluationBenchmark(_StrictEvaluationModel):
    """A versioned fixed benchmark containing 20--30 evaluation cases."""

    schema_version: Literal["1"] = BENCHMARK_SCHEMA_VERSION
    id: BenchmarkId
    title: NonBlankText = Field(max_length=512)
    corpus_id: NonBlankText = Field(max_length=512)
    expected_source_corpus_hash: CorpusHash
    description: NonBlankText = Field(max_length=4_000)
    cases: tuple[BenchmarkCase, ...] = Field(min_length=20, max_length=30)

    @model_validator(mode="after")
    def validate_cases(self) -> EvaluationBenchmark:
        _reject_duplicate_ids(self.cases, field_name="cases")
        categories = {case.category for case in self.cases}
        required = set(EvaluationCategory)
        missing = sorted(category.value for category in required - categories)
        if missing:
            raise ValueError(f"benchmark is missing required categories: {', '.join(missing)}")
        return self


class EvaluationOptions(_StrictEvaluationModel):
    """Execution options whose hash identifies comparable evaluation runs."""

    modes: tuple[RetrievalMode, ...] = (RetrievalMode.NAIVE, RetrievalMode.HYBRID)
    top_k: int = Field(default=8, ge=1, le=100)
    candidate_multiplier: int = Field(default=4, ge=1, le=20)
    context_token_budget: int = Field(default=2_400, ge=1, le=100_000)
    graph_max_hops: int = Field(default=2, ge=1, le=4)
    naive_weight: float = Field(default=1.0, ge=0.0)
    local_weight: float = Field(default=1.0, ge=0.0)
    global_weight: float = Field(default=1.0, ge=0.0)
    naive_dense_weight: float = Field(default=1.0, ge=0.0)
    naive_bm25_weight: float = Field(default=1.0, ge=0.0)
    bm25_k1: float = Field(default=1.2, gt=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    reranker_provider: str = Field(default="lexical", pattern=r"^(none|lexical)$")
    reranker_model: str = Field(default="lexical-coverage-v1", min_length=1)
    rerank_candidate_multiplier: int = Field(default=4, ge=1, le=32)
    case_ids: tuple[CaseId, ...] = ()

    @model_validator(mode="after")
    def validate_modes_and_cases(self) -> EvaluationOptions:
        if len(self.modes) < 2:
            raise ValueError("evaluation requires at least two retrieval modes")
        if len(self.modes) != len(set(self.modes)):
            raise ValueError("modes must not contain duplicates")
        missing = {RetrievalMode.NAIVE, RetrievalMode.HYBRID} - set(self.modes)
        if missing:
            missing_names = ", ".join(sorted(mode.value for mode in missing))
            raise ValueError(f"evaluation must include naive and hybrid modes: {missing_names}")
        if self.naive_weight + self.local_weight + self.global_weight <= 0:
            raise ValueError("at least one fusion weight must be positive")
        if self.naive_dense_weight + self.naive_bm25_weight <= 0:
            raise ValueError("at least one naive subroute weight must be positive")
        _reject_casefold_duplicates(self.case_ids, field_name="case_ids")
        return self

    @property
    def config_hash(self) -> str:
        return canonical_json_hash(self.model_dump(mode="json"))


class IndexProvenance(_StrictEvaluationModel):
    """The immutable index snapshot pinned for one evaluation execution."""

    profile_id: NonBlankText = Field(max_length=128)
    index_config_hash: NonBlankText = Field(max_length=128)
    embedding_provider: NonBlankText = Field(max_length=128)
    embedding_model: NonBlankText = Field(max_length=512)
    embedding_dimensions: int = Field(ge=1, le=1_000_000)
    index_schema_version: NonBlankText = Field(max_length=128)
    corpus_content_hash: CorpusHash
    source_corpus_hash: CorpusHash
    source_graph_run_id: str | None = Field(default=None, min_length=1, max_length=128)
    source_graph_corpus_hash: CorpusHash | None = None


class EvaluationRun(_StrictEvaluationModel):
    """The immutable identity and scope of one benchmark execution."""

    id: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=8,
            max_length=96,
            pattern=r"^evr_[a-f0-9]{12,64}$",
        ),
    ]
    schema_version: Literal["1"] = EVALUATION_SCHEMA_VERSION
    benchmark_id: BenchmarkId
    benchmark_schema_version: Literal["1"] = BENCHMARK_SCHEMA_VERSION
    options: EvaluationOptions
    case_ids: tuple[CaseId, ...] = Field(min_length=1, max_length=30)
    index_provenance: IndexProvenance
    execution_id: EvaluationExecutionId

    @model_validator(mode="after")
    def validate_case_ids(self) -> EvaluationRun:
        _reject_casefold_duplicates(self.case_ids, field_name="case_ids")
        return self


class RetrievalEvaluation(_StrictEvaluationModel):
    """One mode's observable result for one benchmark case."""

    case_id: CaseId
    mode: RetrievalMode
    retrieval: RetrievalTrace
    retrieval_trace_id: NonBlankText = Field(max_length=128)
    answer: GroundedAnswer
    retrieval_latency_ms: float = Field(ge=0.0)
    total_latency_ms: float = Field(ge=0.0)
    expected_evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1, max_length=8)
    matched_evidence_ids: tuple[EvidenceId, ...] = ()
    cited_evidence_ids: tuple[EvidenceId, ...] = ()
    evidence_hit_rate: float = Field(ge=0.0, le=1.0)
    cited_evidence_hit_rate: float = Field(ge=0.0, le=1.0)
    citation_allowlist_valid: bool
    answer_supported_by_citation: bool
    citation_grounded_faithfulness: bool
    abstained: bool

    @model_validator(mode="after")
    def validate_measurement(self) -> RetrievalEvaluation:
        _reject_casefold_duplicates(self.expected_evidence_ids, field_name="expected_evidence_ids")
        _reject_casefold_duplicates(self.matched_evidence_ids, field_name="matched_evidence_ids")
        _reject_casefold_duplicates(self.cited_evidence_ids, field_name="cited_evidence_ids")
        expected = set(self.expected_evidence_ids)
        if not set(self.matched_evidence_ids).issubset(expected):
            raise ValueError("matched_evidence_ids must be part of expected_evidence_ids")
        if not set(self.cited_evidence_ids).issubset(expected):
            raise ValueError("cited_evidence_ids must be part of expected_evidence_ids")
        expected_count = len(self.expected_evidence_ids)
        if self.evidence_hit_rate != len(self.matched_evidence_ids) / expected_count:
            raise ValueError("evidence_hit_rate must match matched_evidence_ids")
        if self.cited_evidence_hit_rate != len(self.cited_evidence_ids) / expected_count:
            raise ValueError("cited_evidence_hit_rate must match cited_evidence_ids")
        if self.abstained != self.answer.insufficient_evidence:
            raise ValueError("abstained must match answer.insufficient_evidence")
        if self.abstained and self.answer_supported_by_citation:
            raise ValueError("an abstention cannot claim citation support")
        if self.citation_grounded_faithfulness and not self.citation_allowlist_valid:
            raise ValueError("faithfulness requires an allowlist-valid citation")
        return self


class BlindCandidate(_StrictEvaluationModel):
    """A mode-hidden candidate made available to a pairwise judge."""

    label: BlindLabel
    evidence_hit_rate: float = Field(ge=0.0, le=1.0)
    cited_evidence_hit_rate: float = Field(ge=0.0, le=1.0)
    citation_grounded_faithfulness: bool
    abstained: bool
    answer: NonBlankText = Field(max_length=20_000)
    citation_ids: tuple[NonBlankText, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_citations(self) -> BlindCandidate:
        _reject_casefold_duplicates(self.citation_ids, field_name="citation_ids")
        return self


class BlindComparison(_StrictEvaluationModel):
    """A mode-hidden request for an answer-quality comparison."""

    case_id: CaseId
    question: NonBlankText = Field(max_length=2_000)
    candidates: tuple[BlindCandidate, BlindCandidate]

    @model_validator(mode="after")
    def validate_labels(self) -> BlindComparison:
        labels = {candidate.label for candidate in self.candidates}
        if labels != {BlindLabel.A, BlindLabel.B}:
            raise ValueError("blind comparison requires exactly one A and one B candidate")
        return self


class BlindJudgment(_StrictEvaluationModel):
    """A blind protocol response before labels are mapped back to modes."""

    winner: BlindWinner
    rationale: NonBlankText = Field(max_length=2_000)


class PairwiseJudgment(_StrictEvaluationModel):
    """A stored pairwise outcome with its blind assignment made auditable."""

    case_id: CaseId
    comparison: BlindComparison
    judgment: BlindJudgment
    label_to_mode: dict[BlindLabel, RetrievalMode]
    winner_mode: RetrievalMode | None = None
    protocol: NonBlankText = Field(max_length=256)
    used_fallback: bool
    fallback_reason: str | None = Field(default=None, min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_pairwise_outcome(self) -> PairwiseJudgment:
        if set(self.label_to_mode) != {BlindLabel.A, BlindLabel.B}:
            raise ValueError("label_to_mode requires both blind labels")
        if len(set(self.label_to_mode.values())) != 2:
            raise ValueError("each blind label must map to a distinct mode")
        if self.judgment.winner is BlindWinner.TIE:
            if self.winner_mode is not None:
                raise ValueError("a tie cannot have a winner_mode")
        elif self.winner_mode != self.label_to_mode[BlindLabel(self.judgment.winner.value)]:
            raise ValueError("winner_mode must resolve from the blind winner label")
        if not self.used_fallback and self.fallback_reason is not None:
            raise ValueError("fallback_reason is only valid when a fallback was used")
        return self


class ModeSummary(_StrictEvaluationModel):
    """Aggregate metrics for one retrieval mode."""

    mode: RetrievalMode
    cases: int = Field(ge=1)
    mean_evidence_hit_rate: float = Field(ge=0.0, le=1.0)
    mean_cited_evidence_hit_rate: float = Field(ge=0.0, le=1.0)
    citation_grounded_faithfulness_rate: float = Field(ge=0.0, le=1.0)
    mean_retrieval_latency_ms: float = Field(ge=0.0)
    median_retrieval_latency_ms: float = Field(ge=0.0)
    mean_total_latency_ms: float = Field(ge=0.0)


class ComparisonSummary(_StrictEvaluationModel):
    """Naive-vs-hybrid wins only; latency never breaks an answer-quality tie."""

    naive_wins: int = Field(ge=0)
    hybrid_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    naive_win_rate: float = Field(ge=0.0, le=1.0)
    hybrid_win_rate: float = Field(ge=0.0, le=1.0)
    tie_rate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_rates(self) -> ComparisonSummary:
        total = self.naive_wins + self.hybrid_wins + self.ties
        if total < 1:
            raise ValueError("comparison summary requires at least one judgment")
        if self.naive_win_rate != self.naive_wins / total:
            raise ValueError("naive_win_rate must match naive_wins")
        if self.hybrid_win_rate != self.hybrid_wins / total:
            raise ValueError("hybrid_win_rate must match hybrid_wins")
        if self.tie_rate != self.ties / total:
            raise ValueError("tie_rate must match ties")
        return self


class CostDisclosure(_StrictEvaluationModel):
    """Explicit cost metadata; never silently invent a provider price.

    Callers that configure every external model price can pass a verified or
    estimated price assumption into :meth:`EvaluationRunner.run`.  A zero
    dollar value is reserved for fully local hash/deterministic execution.
    """

    status: CostStatus
    retrieval_model_calls: int | None = Field(default=0, ge=0)
    judge_model_calls: int | None = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=0.0, ge=0.0)
    price_assumption: str | None = Field(default=None, min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_cost_status(self) -> CostDisclosure:
        calls = (self.retrieval_model_calls, self.judge_model_calls)
        if self.status is CostStatus.NOT_APPLICABLE:
            if calls != (0, 0) or self.cost_usd != 0.0:
                raise ValueError("not_applicable cost requires zero known calls and zero cost")
        elif self.status in {CostStatus.VERIFIED, CostStatus.ESTIMATED}:
            if self.cost_usd is None or self.price_assumption is None:
                raise ValueError("verified or estimated cost requires amount and price_assumption")
            if any(value is None for value in calls):
                raise ValueError("verified or estimated cost requires known model call counts")
        elif self.status is CostStatus.UNKNOWN and self.cost_usd is not None:
            raise ValueError("unknown cost must not provide a dollar value")
        return self

    @classmethod
    def offline(cls) -> CostDisclosure:
        """Return the standard disclosure for local hash/deterministic evaluation."""

        return cls(
            status=CostStatus.NOT_APPLICABLE,
            retrieval_model_calls=0,
            judge_model_calls=0,
            cost_usd=0.0,
            price_assumption="offline deterministic retrieval, answer, and judge",
        )

    @classmethod
    def unknown_external_embedding(
        cls,
        *,
        provider: str,
        retrieval_model_calls: int,
        judge_model_calls: int | None = 0,
    ) -> CostDisclosure:
        """Disclose observed query calls without fabricating embedding prices."""

        return cls(
            status=CostStatus.UNKNOWN,
            retrieval_model_calls=retrieval_model_calls,
            judge_model_calls=judge_model_calls,
            cost_usd=None,
            price_assumption=(
                f"external embedding provider {provider!r} has no verified evaluation "
                "price/usage disclosure"
            ),
        )

    @classmethod
    def unknown_external_judge(cls) -> CostDisclosure:
        """Avoid claiming a cost for a caller-supplied judge implementation."""

        return cls(
            status=CostStatus.UNKNOWN,
            retrieval_model_calls=0,
            judge_model_calls=None,
            cost_usd=None,
            price_assumption="caller-supplied blind judge did not disclose model usage or price",
        )

    @classmethod
    def unknown_judge_fallback(
        cls,
        *,
        retrieval_model_calls: int | None,
        judge_model_calls: int | None,
    ) -> CostDisclosure:
        """Invalidate a dollar estimate after any external judge fallback."""

        return cls(
            status=CostStatus.UNKNOWN,
            retrieval_model_calls=retrieval_model_calls,
            judge_model_calls=judge_model_calls,
            cost_usd=None,
            price_assumption=(
                "one or more external blind-judge calls failed and used the deterministic "
                "fallback; total model cost is not comparable"
            ),
        )


class JudgeProvenance(_StrictEvaluationModel):
    """Configured blind-judge identity, independent of per-case fallbacks."""

    provider: NonBlankText = Field(max_length=128)
    protocol: NonBlankText = Field(max_length=256)
    external: bool
    model: str | None = Field(default=None, min_length=1, max_length=512)
    base_url: str | None = Field(default=None, min_length=1, max_length=2_000)
    response_format: str | None = Field(default=None, min_length=1, max_length=128)
    thinking: str | None = Field(default=None, min_length=1, max_length=128)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)


class EvaluationReport(_StrictEvaluationModel):
    """Fully serializable output for a benchmark run."""

    run: EvaluationRun
    evaluations: tuple[RetrievalEvaluation, ...] = Field(min_length=2)
    pairwise_judgments: tuple[PairwiseJudgment, ...]
    summaries: tuple[ModeSummary, ...] = Field(min_length=2)
    comparison_summary: ComparisonSummary
    cost_disclosure: CostDisclosure
    judge_provenance: JudgeProvenance

    @model_validator(mode="after")
    def validate_report(self) -> EvaluationReport:
        expected_count = len(self.run.case_ids) * len(self.run.options.modes)
        if len(self.evaluations) != expected_count:
            raise ValueError("evaluations must contain one entry per selected case and mode")
        observed = {(item.case_id, item.mode) for item in self.evaluations}
        if len(observed) != len(self.evaluations):
            raise ValueError("evaluations must not contain duplicate case/mode pairs")
        if {item.case_id for item in self.evaluations} != set(self.run.case_ids):
            raise ValueError("evaluations must cover exactly the run case_ids")
        if {item.mode for item in self.evaluations} != set(self.run.options.modes):
            raise ValueError("evaluations must cover exactly the configured modes")
        if {summary.mode for summary in self.summaries} != set(self.run.options.modes):
            raise ValueError("summaries must cover exactly the configured modes")
        if len(self.pairwise_judgments) != len(self.run.case_ids):
            raise ValueError("pairwise_judgments must contain one naive-vs-hybrid result per case")
        if {item.case_id for item in self.pairwise_judgments} != set(self.run.case_ids):
            raise ValueError("pairwise_judgments must cover exactly the run case_ids")
        comparison_total = (
            self.comparison_summary.naive_wins
            + self.comparison_summary.hybrid_wins
            + self.comparison_summary.ties
        )
        if comparison_total != len(self.pairwise_judgments):
            raise ValueError("comparison summary total must equal pairwise judgment count")
        return self

    def to_json(self) -> str:
        """Return canonical report data as human-readable JSON."""

        return self.model_dump_json(indent=2)


def _reject_duplicate_ids(
    values: tuple[ExpectedEvidence | BenchmarkCase, ...], *, field_name: str
) -> None:
    ids = [str(value.id) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{field_name} must not contain duplicate IDs")


def _reject_casefold_duplicates(values: tuple[str, ...], *, field_name: str) -> None:
    folded = [value.casefold() for value in values]
    if len(folded) != len(set(folded)):
        raise ValueError(f"{field_name} must not contain duplicates ignoring case")
