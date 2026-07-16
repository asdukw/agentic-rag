"""Ragas-backed evaluation over the project's existing retrieval and answer pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

from hybrid_rag.deepseek_costs import DeepSeekCostStatus, DeepSeekCostSummary
from hybrid_rag.evaluation.agentic_metrics import aggregate_agentic_scores, score_agentic_events
from hybrid_rag.evaluation.evidence import evidence_ids
from hybrid_rag.evaluation.retrieval_metrics import (
    RankedEvidence,
    RetrievalMetricScores,
    aggregate_retrieval_scores,
    score_retrieval,
)
from hybrid_rag.evaluation.testset_contract import (
    EVALUATION_TESTSET_SCHEMA_VERSION,
    SUPPORTED_TESTSET_SCHEMA_VERSIONS,
    validate_corpus_content_hash,
    validate_testset_sources,
)
from hybrid_rag.retrieval.models import RetrievalMode
from hybrid_rag.retrieval.query import QueryClient
from hybrid_rag.retrieval.service import RetrievalOptions, RetrievalService
from hybrid_rag.storage.retrieval_repository import StoredIndexProfile


class RagasCase(TypedDict):
    user_input: str
    reference: str
    reference_contexts: list[str]
    evidence_ids: NotRequired[list[str]]
    context_evidence_ids: NotRequired[list[str]]
    document_ids: NotRequired[list[str]]
    question_type: NotRequired[str]
    answerable: NotRequired[bool]
    evidence_quotes: NotRequired[list[str]]
    generator_model: NotRequired[str]
    prompt_version: NotRequired[str]
    review_status: NotRequired[str]


_SMOKE_CASE_QUOTAS: tuple[tuple[str, int], ...] = (
    ("single_hop", 3),
    ("summary_reasoning", 1),
    ("multi_context", 1),
    ("unanswerable", 1),
)


@dataclass(frozen=True, slots=True)
class RagasTestset:
    """A versioned golden test set bound to one corpus content hash."""

    schema_version: str
    corpus_content_hash: str
    file_sha256: str
    cases: tuple[RagasCase, ...]
    sources: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class RagasEvaluationReport:
    """Serializable Ragas scores and per-sample details grouped by retrieval mode."""

    testset_path: str
    provenance: dict[str, object]
    cost: dict[str, dict[str, object]]
    modes: dict[str, dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return {
            "testset_path": self.testset_path,
            "provenance": self.provenance,
            "cost": self.cost,
            "modes": self.modes,
        }


class RagasEvaluationRunner:
    """Run golden cases through retrieval/answer modes, then use Ragas for scoring."""

    def __init__(self, retrieval_service: RetrievalService, query_client: QueryClient) -> None:
        self.retrieval_service = retrieval_service
        self.query_client = query_client

    async def run(
        self,
        testset_path: Path,
        *,
        modes: Sequence[RetrievalMode],
        retrieval_options: RetrievalOptions,
        profile_ref: str | None,
        judge_model: str,
        judge_api_key: str,
        judge_base_url: str,
        judge_max_output_tokens: int = 1_024,
        judge_timeout_seconds: float = 180.0,
        query_client_provenance: dict[str, object] | None = None,
        agentic_runner: object | None = None,
        smoke: bool = False,
    ) -> RagasEvaluationReport:
        if judge_max_output_tokens < 1:
            raise ValueError("Ragas judge_max_output_tokens must be positive")
        if judge_timeout_seconds <= 0:
            raise ValueError("Ragas judge_timeout_seconds must be positive")
        testset = _load_testset(testset_path)
        profile = self.retrieval_service.resolve_profile(profile_ref)
        profile_corpus_content_hash = _profile_corpus_content_hash(profile)
        if testset.corpus_content_hash != profile_corpus_content_hash:
            raise ValueError(
                "Ragas test set corpus_content_hash does not match the pinned index profile "
                f"({testset.corpus_content_hash} != {profile_corpus_content_hash})"
            )
        case_entries = _evaluation_case_entries(testset.cases, smoke=smoke)

        output: dict[str, dict[str, object]] = {}
        cost_observations: list[RetrievalCostObservation] = []
        for mode in modes:
            samples: list[object] = []
            scored_case_indexes: list[int] = []
            case_details: list[dict[str, object]] = []
            retrieval_scores: list[RetrievalMetricScores] = []
            for case_index, case in case_entries:
                answer = await self.retrieval_service.ask(
                    case["user_input"],
                    query_client=self.query_client,
                    mode=mode,
                    options=retrieval_options,
                    profile_ref=profile.id,
                )
                if answer.retrieval.profile_id != profile.id:
                    raise RuntimeError(
                        "retrieval returned a profile different from the profile pinned "
                        "for Ragas evaluation"
                    )
                trace_id = answer.retrieval.trace_id
                if not isinstance(trace_id, str) or not trace_id:
                    raise RuntimeError(
                        "Ragas evaluation requires every retrieval result to have "
                        "a persisted trace ID"
                    )
                if case.get("answerable", True):
                    samples.append(
                        _sample(
                            case,
                            response=answer.answer.answer,
                            retrieved_contexts=[
                                item.text for item in answer.retrieval.context_items
                            ],
                        )
                    )
                    scored_case_indexes.append(case_index)
                ranked_evidence = [
                    RankedEvidence(
                        evidence_ids=evidence_ids(
                            item.document_id,
                            page_start=item.page_start,
                            page_end=item.page_end,
                            section_path=item.section_path,
                        ),
                        text=item.text,
                    )
                    for item in answer.retrieval.context_items
                ]
                retrieval_score = score_retrieval(
                    ranked_evidence,
                    k=retrieval_options.top_k,
                    evidence_ids=case.get("evidence_ids"),
                    reference_contexts=case["reference_contexts"],
                )
                retrieval_scores.append(retrieval_score)
                refusal_correct = answer.answer.insufficient_evidence == (
                    not case.get("answerable", True)
                )
                case_details.append(
                    {
                        "case_index": case_index,
                        "retrieval_trace_id": trace_id,
                        "question_type": case.get("question_type", "legacy"),
                        "answerable": case.get("answerable", True),
                        "retrieval_metrics": retrieval_score.as_dict(),
                        "insufficient_evidence": answer.answer.insufficient_evidence,
                        "refusal_correct": refusal_correct,
                        "citations": list(answer.answer.citations),
                    }
                )
                cost_observations.append(
                    RetrievalCostObservation(
                        retrieval_trace_id=trace_id,
                        summary=answer.retrieval.trace.deepseek_cost,
                    )
                )
            result = (
                await asyncio.to_thread(
                    _evaluate,
                    samples,
                    judge_model=judge_model,
                    judge_api_key=judge_api_key,
                    judge_base_url=judge_base_url,
                    judge_max_output_tokens=judge_max_output_tokens,
                    judge_timeout_seconds=judge_timeout_seconds,
                )
                if samples
                else {"scores": [], "means": {}}
            )
            output[mode.value] = _mode_report(
                result,
                case_details,
                scored_case_indexes=scored_case_indexes,
                retrieval=aggregate_retrieval_scores(retrieval_scores),
            )
        if agentic_runner is not None:
            output["agentic"] = await self._run_agentic_mode(
                agentic_runner,
                case_entries,
                profile_id=profile.id,
                retrieval_options=retrieval_options,
                judge_model=judge_model,
                judge_api_key=judge_api_key,
                judge_base_url=judge_base_url,
                judge_max_output_tokens=judge_max_output_tokens,
                judge_timeout_seconds=judge_timeout_seconds,
            )
        return RagasEvaluationReport(
            testset_path=str(testset_path),
            provenance={
                "testset": {
                    "schema_version": testset.schema_version,
                    "corpus_content_hash": testset.corpus_content_hash,
                    "file_sha256": testset.file_sha256,
                    "case_count": len(testset.cases),
                    "sources": list(testset.sources),
                },
                "profile": _profile_provenance(profile, profile_corpus_content_hash),
                "runtime": {
                    "smoke": smoke,
                    "evaluated_case_count": len(case_entries),
                    "evaluated_case_indexes": [index for index, _case in case_entries],
                    "modes": [mode.value for mode in modes]
                    + (["agentic"] if agentic_runner is not None else []),
                    "retrieval_options": asdict(retrieval_options),
                    "query_client": dict(query_client_provenance or {}),
                    "judge": {
                        "model": judge_model,
                        "base_url": judge_base_url,
                        "max_output_tokens": judge_max_output_tokens,
                        "timeout_seconds": judge_timeout_seconds,
                        "max_retries": 0,
                        "temperature": 0.0,
                    },
                },
            },
            cost=_cost_report(cost_observations),
            modes=output,
        )

    async def _run_agentic_mode(
        self,
        runner: object,
        case_entries: Sequence[tuple[int, RagasCase]],
        *,
        profile_id: str,
        retrieval_options: RetrievalOptions,
        judge_model: str,
        judge_api_key: str,
        judge_base_url: str,
        judge_max_output_tokens: int,
        judge_timeout_seconds: float,
    ) -> dict[str, object]:
        from hybrid_rag.agentic.models import AgentEvent
        from hybrid_rag.agentic.runner import AgentRunRequest
        from hybrid_rag.retrieval.models import ContextItem
        from hybrid_rag.retrieval.query import GroundedAnswer

        run_method = getattr(runner, "run", None)
        if not callable(run_method):
            raise TypeError("agentic_runner must expose an async run() iterator")
        samples: list[object] = []
        scored_case_indexes: list[int] = []
        case_details: list[dict[str, object]] = []
        retrieval_scores: list[RetrievalMetricScores] = []
        agentic_scores = []
        for case_index, case in case_entries:
            raw_events = [
                event
                async for event in run_method(
                    AgentRunRequest(question=case["user_input"], profile_id=profile_id)
                )
            ]
            events = [
                event if isinstance(event, AgentEvent) else AgentEvent.model_validate(event)
                for event in raw_events
            ]
            failed = next((event for event in events if event.event == "failed"), None)
            if failed is not None:
                raise RuntimeError(
                    f"agentic evaluation case {case_index} failed: {failed.data.get('error')}"
                )
            answer_event = next(
                (event for event in reversed(events) if event.event == "answer"), None
            )
            if answer_event is None:
                raise RuntimeError(f"agentic evaluation case {case_index} produced no answer")
            answer = GroundedAnswer.model_validate(answer_event.data.get("answer"))
            raw_evidence = answer_event.data.get("evidence", [])
            if not isinstance(raw_evidence, list):
                raise RuntimeError(f"agentic evaluation case {case_index} has invalid evidence")
            evidence = [ContextItem.model_validate(item) for item in raw_evidence]
            duration = _agent_duration(events)
            metrics = score_agentic_events(
                events,
                reference_evidence_ids=case.get("evidence_ids"),
                answerable=case.get("answerable", True),
                duration_seconds=duration,
            )
            agentic_scores.append(metrics)
            retrieval_score = score_retrieval(
                [
                    RankedEvidence(
                        evidence_ids=evidence_ids(
                            item.document_id,
                            page_start=item.page_start,
                            page_end=item.page_end,
                            section_path=item.section_path,
                        ),
                        text=item.text,
                    )
                    for item in evidence
                ],
                k=retrieval_options.top_k,
                evidence_ids=case.get("evidence_ids"),
                reference_contexts=case["reference_contexts"],
            )
            retrieval_scores.append(retrieval_score)
            if case.get("answerable", True):
                samples.append(
                    _sample(
                        case,
                        response=answer.answer,
                        retrieved_contexts=[item.text for item in evidence],
                    )
                )
                scored_case_indexes.append(case_index)
            case_details.append(
                {
                    "case_index": case_index,
                    "agent_run_id": events[0].run_id if events else None,
                    "retrieval_trace_ids": _agent_trace_ids(events),
                    "question_type": case.get("question_type", "legacy"),
                    "answerable": case.get("answerable", True),
                    "retrieval_metrics": retrieval_score.as_dict(),
                    "agentic_metrics": metrics.as_dict(),
                    "insufficient_evidence": answer.insufficient_evidence,
                    "refusal_correct": metrics.refusal_correct,
                    "citations": list(answer.citations),
                }
            )
        result = (
            await asyncio.to_thread(
                _evaluate,
                samples,
                judge_model=judge_model,
                judge_api_key=judge_api_key,
                judge_base_url=judge_base_url,
                judge_max_output_tokens=judge_max_output_tokens,
                judge_timeout_seconds=judge_timeout_seconds,
            )
            if samples
            else {"scores": [], "means": {}}
        )
        retrieval = aggregate_retrieval_scores(retrieval_scores)
        agentic = aggregate_agentic_scores(agentic_scores)
        report = _mode_report(
            result,
            case_details,
            scored_case_indexes=scored_case_indexes,
            retrieval=retrieval,
            trace_id_field="retrieval_trace_ids",
        )
        report["agentic"] = agentic
        means = cast(dict[str, object], report["means"])
        means.update(cast(dict[str, object], agentic["means"]))
        return report


def _evaluation_case_entries(
    cases: Sequence[RagasCase], *, smoke: bool
) -> tuple[tuple[int, RagasCase], ...]:
    entries = tuple(enumerate(cases, start=1))
    if not smoke:
        return entries

    target_size = min(sum(quota for _question_type, quota in _SMOKE_CASE_QUOTAS), len(entries))
    selected: list[tuple[int, RagasCase]] = []
    selected_indexes: set[int] = set()
    for question_type, quota in _SMOKE_CASE_QUOTAS:
        matches = (
            entry for entry in entries if entry[1].get("question_type", "legacy") == question_type
        )
        for entry in matches:
            if (
                len([case for case in selected if case[1].get("question_type") == question_type])
                >= quota
            ):
                break
            selected.append(entry)
            selected_indexes.add(entry[0])

    for entry in entries:
        if len(selected) >= target_size:
            break
        if entry[0] not in selected_indexes:
            selected.append(entry)
            selected_indexes.add(entry[0])
    return tuple(sorted(selected, key=lambda entry: entry[0]))


def _load_testset(path: Path) -> RagasTestset:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read Ragas test set {path}: {error}") from error
    if isinstance(value, list):
        raise ValueError(
            "Ragas test set must use a JSON envelope with schema_version, corpus_content_hash, "
            "and cases; bare JSON arrays are unsupported"
        )
    if not isinstance(value, dict):
        raise ValueError("Ragas test set must be a JSON object envelope")

    schema_version = value.get("schema_version")
    if schema_version not in SUPPORTED_TESTSET_SCHEMA_VERSIONS:
        raise ValueError(
            "unsupported evaluation test set schema_version "
            f"{schema_version!r}; expected one of {sorted(SUPPORTED_TESTSET_SCHEMA_VERSIONS)!r}"
        )
    corpus_content_hash = validate_corpus_content_hash(
        value.get("corpus_content_hash"),
        field="Ragas test set corpus_content_hash",
    )
    values = value.get("cases")
    if not isinstance(values, list) or not values:
        raise ValueError("Ragas test set envelope cases must be a non-empty JSON array")
    raw_sources = value.get("sources")
    sources = validate_testset_sources(raw_sources) if raw_sources is not None else []

    cases: list[RagasCase] = []
    for index, item in enumerate(values, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Ragas test set item {index} must be an object")
        question = item.get("user_input")
        reference = item.get("reference")
        contexts = item.get("reference_contexts")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Ragas test set item {index} has no user_input")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"Ragas test set item {index} has no reference")
        case: RagasCase = {
            "user_input": question,
            "reference": reference,
            "reference_contexts": _string_list(
                contexts,
                field="reference_contexts",
                index=index,
            ),
        }
        if schema_version == EVALUATION_TESTSET_SCHEMA_VERSION:
            case.update(
                {
                    "evidence_ids": _string_list(
                        item.get("evidence_ids"), field="evidence_ids", index=index
                    ),
                    "context_evidence_ids": _string_list(
                        item.get("context_evidence_ids"),
                        field="context_evidence_ids",
                        index=index,
                    ),
                    "document_ids": _string_list(
                        item.get("document_ids"), field="document_ids", index=index
                    ),
                    "question_type": _required_string(item, "question_type", index),
                    "answerable": _required_bool(item, "answerable", index),
                    "evidence_quotes": _string_list(
                        item.get("evidence_quotes"), field="evidence_quotes", index=index
                    ),
                    "generator_model": _required_string(item, "generator_model", index),
                    "prompt_version": _required_string(item, "prompt_version", index),
                    "review_status": _required_string(item, "review_status", index),
                }
            )
        cases.append(case)
    return RagasTestset(
        schema_version=schema_version,
        corpus_content_hash=corpus_content_hash,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        cases=tuple(cases),
        sources=tuple(sources),
    )


def _profile_corpus_content_hash(profile: StoredIndexProfile) -> str:
    return validate_corpus_content_hash(
        profile.metadata.get("corpus_content_hash"),
        field="pinned index profile metadata.corpus_content_hash",
    )


def _profile_provenance(
    profile: StoredIndexProfile,
    corpus_content_hash: str,
) -> dict[str, object]:
    return {
        "id": profile.id,
        "config_hash": profile.config_hash,
        "provider": profile.provider,
        "model": profile.model,
        "dimensions": profile.dimensions,
        "schema_version": profile.schema_version,
        "source_corpus_hash": profile.source_corpus_hash,
        "source_graph_run_id": profile.source_graph_run_id,
        "corpus_content_hash": corpus_content_hash,
    }


def _mode_report(
    result: dict[str, object],
    case_details: list[dict[str, object]],
    *,
    scored_case_indexes: Sequence[int],
    retrieval: dict[str, object],
    trace_id_field: str = "retrieval_trace_id",
) -> dict[str, object]:
    """Attach replayable retrieval trace IDs to the Ragas result in case order."""

    scores = result.get("scores")
    if not isinstance(scores, list):
        raise RuntimeError("Ragas evaluation result did not contain per-case scores")
    if len(scores) != len(scored_case_indexes):
        raise RuntimeError(
            "Ragas evaluation returned a score count different from answerable cases"
        )
    scores_by_index = dict(zip(scored_case_indexes, scores, strict=True))
    cases = [
        {**detail, "ragas_scores": scores_by_index.get(cast(int, detail["case_index"]))}
        for detail in case_details
    ]
    ragas_means = result.get("means")
    if not isinstance(ragas_means, dict):
        raise RuntimeError("Ragas evaluation result did not contain means")
    retrieval_means = retrieval.get("means")
    combined_means = dict(ragas_means)
    if isinstance(retrieval_means, dict):
        combined_means.update(retrieval_means)
    behavior = _behavior_report(case_details)
    combined_means.update(cast(dict[str, object], behavior["means"]))
    trace_ids: list[str] = []
    for detail in case_details:
        value = detail.get(trace_id_field)
        if isinstance(value, str):
            trace_ids.append(value)
        elif isinstance(value, list):
            trace_ids.extend(item for item in value if isinstance(item, str))
    return {
        "ragas": result,
        "retrieval": retrieval,
        "behavior": behavior,
        "means": combined_means,
        "retrieval_trace_ids": trace_ids,
        "cases": cases,
    }


def _behavior_report(case_details: Sequence[dict[str, object]]) -> dict[str, object]:
    answerable = [detail for detail in case_details if detail.get("answerable") is True]
    unanswerable = [detail for detail in case_details if detail.get("answerable") is False]
    correctness = [
        detail["refusal_correct"]
        for detail in case_details
        if isinstance(detail.get("refusal_correct"), bool)
    ]
    return {
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "means": {
            "refusal_accuracy": (
                sum(value is True for value in correctness) / len(correctness)
                if correctness
                else None
            ),
            "unanswerable_refusal_rate": (
                sum(detail.get("insufficient_evidence") is True for detail in unanswerable)
                / len(unanswerable)
                if unanswerable
                else None
            ),
            "answerable_response_rate": (
                sum(detail.get("insufficient_evidence") is False for detail in answerable)
                / len(answerable)
                if answerable
                else None
            ),
        },
    }


def _agent_trace_ids(events: Sequence[object]) -> list[str]:
    trace_ids: list[str] = []
    for event in events:
        if getattr(event, "event", None) != "tool_result":
            continue
        data = getattr(event, "data", None)
        if not isinstance(data, dict):
            continue
        outcome_data = data.get("data")
        trace_id = outcome_data.get("trace_id") if isinstance(outcome_data, dict) else None
        if isinstance(trace_id, str) and trace_id and trace_id not in trace_ids:
            trace_ids.append(trace_id)
    return trace_ids


def _agent_duration(events: Sequence[object]) -> float | None:
    for event in reversed(events):
        if getattr(event, "event", None) not in {"completed", "failed"}:
            continue
        data = getattr(event, "data", None)
        if isinstance(data, dict):
            value = data.get("duration_seconds")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                return float(value)
    return None


@dataclass(frozen=True, slots=True)
class RetrievalCostObservation:
    """The observed DeepSeek cost snapshot retained on one retrieval trace."""

    retrieval_trace_id: str
    summary: DeepSeekCostSummary | None


def _cost_report(observations: Sequence[RetrievalCostObservation]) -> dict[str, dict[str, object]]:
    """Report retrieval observations without inventing Ragas judge or total costs.

    The project query client exposes cumulative usage snapshots, so we use the
    final complete trace snapshot instead of summing each trace and risking
    double-counting. A missing snapshot makes retrieval cost unknown.
    """

    serialized_observations = [
        {
            "retrieval_trace_id": observation.retrieval_trace_id,
            "cost": (
                observation.summary.model_dump(mode="json")
                if observation.summary is not None
                else None
            ),
        }
        for observation in observations
    ]
    missing = [
        observation.retrieval_trace_id
        for observation in observations
        if observation.summary is None
    ]
    latest = observations[-1].summary if observations else None
    if latest is None or missing:
        retrieval: dict[str, object] = _unknown_cost(
            "Not every retrieval trace exposed observed DeepSeek response usage; "
            "retrieval cost is not estimated."
        )
    else:
        retrieval = latest.model_dump(mode="json")
        retrieval["aggregation"] = "latest_trace_snapshot"
    retrieval["observations"] = serialized_observations
    retrieval["missing_trace_ids"] = missing

    judge = _unknown_cost(
        "Ragas judge usage is not exposed by the current API; judge cost is not estimated."
    )
    total = _unknown_cost(
        "Ragas judge cost is unknown because usage is not exposed; total cost is not estimated."
    )
    return {
        "retrieval": retrieval,
        "judge": judge,
        "total": total,
    }


def _unknown_cost(reason: str) -> dict[str, object]:
    return {
        "status": DeepSeekCostStatus.UNKNOWN.value,
        "currency": "CNY",
        "cost_cny": None,
        "price_assumption": reason,
    }


def _string_list(value: object, *, field: str, index: int) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"evaluation test set item {index} has invalid {field}")
    return [item.strip() for item in value]


def _required_string(value: dict[str, object], field: str, index: int) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"evaluation test set item {index} has invalid {field}")
    return item.strip()


def _required_bool(value: dict[str, object], field: str, index: int) -> bool:
    item = value.get(field)
    if not isinstance(item, bool):
        raise ValueError(f"evaluation test set item {index} has invalid {field}")
    return item


def _sample(case: RagasCase, *, response: str, retrieved_contexts: list[str]) -> object:
    from ragas import SingleTurnSample

    return SingleTurnSample(
        user_input=case["user_input"],
        reference=case["reference"],
        reference_contexts=case["reference_contexts"],
        response=response,
        retrieved_contexts=retrieved_contexts,
    )


def _evaluate(
    samples: list[object],
    *,
    judge_model: str,
    judge_api_key: str,
    judge_base_url: str,
    judge_max_output_tokens: int,
    judge_timeout_seconds: float,
) -> dict[str, object]:
    from openai import OpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import EvaluationResult
    from ragas.llms import llm_factory
    from ragas.metrics.base import Metric
    from ragas.metrics.collections import (
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )

    judge_llm = llm_factory(
        judge_model,
        provider="openai",
        client=OpenAI(
            api_key=judge_api_key,
            base_url=judge_base_url,
            timeout=judge_timeout_seconds,
            max_retries=0,
        ),
        max_tokens=judge_max_output_tokens,
        temperature=0.0,
    )
    metrics = cast(
        Sequence[Metric],
        [
            Faithfulness(llm=judge_llm),
            FactualCorrectness(llm=judge_llm),
            ContextPrecision(llm=judge_llm),
            ContextRecall(llm=judge_llm),
        ],
    )
    result = cast(
        EvaluationResult,
        evaluate(
            dataset=EvaluationDataset(samples=samples),  # type: ignore[arg-type]
            metrics=metrics,
            show_progress=True,
            raise_exceptions=True,
            return_executor=False,
        ),
    )
    scores = result.scores
    return {"scores": scores, "means": _means(scores)}


def _means(scores: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in scores:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(key, []).append(float(value))
    return {key: sum(items) / len(items) for key, items in values.items() if items}
