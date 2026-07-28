from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Required, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from hybrid_rag.deepseek_costs import DeepSeekUsage
from hybrid_rag.extraction.client import (
    CompletionResult,
    ExtractionClient,
    ProviderError,
    RetryableProviderError,
)
from hybrid_rag.extraction.graph import (
    GraphStats,
    build_networkx_graph,
    node_link_json,
    summarize_graph,
)
from hybrid_rag.extraction.normalization import (
    merge_relations,
    normalize_entities,
    normalize_entity_alias,
    normalize_predicate,
)
from hybrid_rag.extraction.prompts import (
    build_extraction_messages,
    build_gleaning_messages,
    build_repair_messages,
)
from hybrid_rag.extraction.reports import (
    AttemptSummary,
    BuildFailure,
    ChunkProgress,
    ExtractionQualitySummary,
    GraphBuildReport,
    GraphSummary,
    TopEntitySummary,
    UsageSummary,
)
from hybrid_rag.extraction.schemas import (
    EntityNormalizationResult,
    RelationMergeResult,
    ValidatedChunkExtraction,
)
from hybrid_rag.extraction.validation import ExtractionValidationError, validate_completion
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.graph_repository import ExtractionClaim, GraphRepository

WORKFLOW_VERSION = "3"


@dataclass(frozen=True, slots=True)
class WorkflowOptions:
    max_concurrency: int = 8
    max_attempts: int = 2
    limit: int | None = None
    retry_failed: bool = True
    review_required: bool = False
    top_k: int = 10
    retry_backoff_seconds: float = 1.0
    lease_seconds: float = 300.0
    output_path: Path | None = None

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit must not be negative")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative")
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")


class BuildState(TypedDict, total=False):
    run_id: Required[str]
    max_concurrency: Required[int]
    max_attempts: Required[int]
    limit: Required[int | None]
    retry_failed: Required[bool]
    review_required: Required[bool]
    top_k: Required[int]
    retry_backoff_seconds: Required[float]
    lease_seconds: Required[float]
    output_path: Required[str | None]
    scheduled_ids: list[str]
    processed_ids: list[str]
    normalized_entities: int
    merged_relations: int
    graph_metrics: dict[str, Any]
    report: dict[str, Any]


class ChunkState(TypedDict, total=False):
    run_id: Required[str]
    job: Required[dict[str, Any]]
    review_required: Required[bool]
    max_attempts: Required[int]
    repair_max_attempts: Required[int]
    gleaning_max_attempts: Required[int]
    retry_backoff_seconds: Required[float]
    lease_seconds: Required[float]
    stage: str
    local_attempt: Required[int]
    messages: Sequence[Mapping[str, Any]]
    claim: ExtractionClaim | None
    completion: CompletionResult | None
    provider_error: ProviderError | None
    validated: ValidatedChunkExtraction | None
    baseline_validated: ValidatedChunkExtraction | None
    validation_error: ExtractionValidationError | None
    latency_seconds: float
    invalid_response: str | None
    issues: tuple[str, ...]
    outcome: str
    stage_attempt_counts: dict[str, int]
    extraction_stage_attempt_counts: dict[str, int]


class ReviewStillPendingError(RuntimeError):
    pass


class MissingExtractionCredentialError(RuntimeError):
    pass


class GraphBuildWorkflow:
    """Explicit LangGraph orchestration over project-owned extraction algorithms."""

    def __init__(
        self,
        database: Database,
        client: ExtractionClient | None,
        repository: GraphRepository | None = None,
        *,
        repair_max_attempts: int = 1,
        gleaning_max_attempts: int = 0,
    ) -> None:
        if repair_max_attempts not in {0, 1}:
            raise ValueError("repair_max_attempts must be zero or one")
        if gleaning_max_attempts not in {0, 1}:
            raise ValueError("gleaning_max_attempts must be zero or one")
        self.database = database
        self.client = client
        self.repository = repository or GraphRepository()
        self.repair_max_attempts = repair_max_attempts
        self.gleaning_max_attempts = gleaning_max_attempts
        self._normalizations: dict[str, EntityNormalizationResult] = {}
        self._relations: dict[str, RelationMergeResult] = {}
        self._graph_stats: dict[str, GraphStats] = {}

    def compile(self, *, checkpointer: Any = None) -> Any:
        builder = StateGraph(BuildState)
        builder.add_node("load_pending_chunks", self._load_pending_chunks)
        builder.add_node("parallel_extract", self._parallel_extract)
        builder.add_node("human_review", self._human_review)
        builder.add_node("normalize_entities", self._normalize_entities)
        builder.add_node("merge_relations", self._merge_relations)
        builder.add_node("build_networkx", self._build_networkx)
        builder.add_node("persist_graph", self._persist_graph)
        builder.add_node("finalize_run", self._finalize_run)
        builder.add_edge(START, "load_pending_chunks")
        builder.add_edge("load_pending_chunks", "parallel_extract")
        builder.add_edge("parallel_extract", "human_review")
        builder.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "extract": "parallel_extract",
                "normalize": "normalize_entities",
                "finalize": "finalize_run",
            },
        )
        builder.add_edge("normalize_entities", "merge_relations")
        builder.add_edge("merge_relations", "build_networkx")
        builder.add_edge("build_networkx", "persist_graph")
        builder.add_edge("persist_graph", "finalize_run")
        builder.add_edge("finalize_run", END)
        return builder.compile(checkpointer=checkpointer, name="graph_build")

    def _chunk_graph(self) -> Any:
        builder = StateGraph(ChunkState)
        builder.add_node("claim_and_call", self._claim_and_call)
        builder.add_node("validate", self._validate)
        builder.add_node("record_attempt", self._record_attempt)
        builder.add_node("prepare_gleaning", self._prepare_gleaning)
        builder.add_node("prepare_retry", self._prepare_retry)
        builder.add_node("human_review", self._complete_or_review)
        builder.add_node("record_failure", self._record_failure)
        builder.add_edge(START, "claim_and_call")
        builder.add_conditional_edges(
            "claim_and_call",
            self._route_after_claim,
            {"validate": "validate", "done": END},
        )
        builder.add_edge("validate", "record_attempt")
        builder.add_conditional_edges(
            "record_attempt",
            self._route_after_attempt,
            {
                "complete": "human_review",
                "glean": "prepare_gleaning",
                "retry": "prepare_retry",
                "fail": "record_failure",
            },
        )
        builder.add_edge("prepare_gleaning", "claim_and_call")
        builder.add_edge("prepare_retry", "claim_and_call")
        builder.add_edge("human_review", END)
        builder.add_edge("record_failure", END)
        return builder.compile(name="chunk_extraction")

    async def _load_pending_chunks(self, state: BuildState) -> dict[str, Any]:
        with self.database.session_factory.begin() as session:
            self.repository.prepare_jobs(
                session,
                state["run_id"],
                limit=state.get("limit"),
                retry_failed=state.get("retry_failed", True),
            )
            jobs = self.repository.list_pending_jobs(session, state["run_id"])
        return {"scheduled_ids": [str(job["id"]) for job in jobs]}

    async def _parallel_extract(self, state: BuildState) -> dict[str, Any]:
        processed_ids: set[str] = set()
        # A reclaimed SQLite lease can lose a claim race between the eligibility
        # SELECT and atomic UPDATE. Re-read a few times before checkpointing this
        # parent node as complete; already-successful jobs are never selected again.
        for _ in range(3):
            with self.database.session_factory() as session:
                jobs = self.repository.list_pending_jobs(session, state["run_id"])
                run = self.repository.get_run(session, state["run_id"])
            if run is None:
                raise RuntimeError(f"graph build run disappeared: {state['run_id']}")
            if not jobs:
                break

            inputs: list[ChunkState] = []
            for job in jobs:
                stage = str(job.get("stage", "extract"))
                stage_attempt_counts = {
                    str(key): int(value)
                    for key, value in dict(job.get("stage_attempt_counts", {})).items()
                }
                extraction_stage_attempt_counts = {
                    str(key): int(value)
                    for key, value in dict(job.get("extraction_stage_attempt_counts", {})).items()
                }
                gleaning_exhausted = (
                    stage == "glean"
                    and extraction_stage_attempt_counts.get("glean", 0)
                    >= self.gleaning_max_attempts
                )
                if int(job["run_attempt_count"]) >= state["max_attempts"] or gleaning_exhausted:
                    with self.database.session_factory.begin() as session:
                        if job.get("baseline_result") is not None:
                            self.repository.complete_provisional_extraction(
                                session,
                                str(job["id"]),
                                run_id=state["run_id"],
                                needs_review=run.review_required,
                            )
                        else:
                            self.repository.fail_exhausted_extraction(
                                session,
                                str(job["id"]),
                                run_id=state["run_id"],
                                max_attempts=state["max_attempts"],
                            )
                    processed_ids.add(str(job["id"]))
                    continue
                baseline_payload = job.get("baseline_result")
                baseline = (
                    self._validated_from_payload(baseline_payload)
                    if baseline_payload is not None
                    else None
                )
                inputs.append(
                    {
                        "run_id": state["run_id"],
                        "job": job,
                        "review_required": run.review_required,
                        "max_attempts": state["max_attempts"],
                        "repair_max_attempts": self.repair_max_attempts,
                        "gleaning_max_attempts": self.gleaning_max_attempts,
                        "retry_backoff_seconds": state["retry_backoff_seconds"],
                        "lease_seconds": state["lease_seconds"],
                        "stage": stage,
                        "stage_attempt_counts": stage_attempt_counts,
                        "extraction_stage_attempt_counts": extraction_stage_attempt_counts,
                        "baseline_validated": baseline,
                        "local_attempt": 0,
                        "issues": (),
                    }
                )
            if not inputs:
                await asyncio.sleep(0)
                continue
            if self.client is None:
                raise MissingExtractionCredentialError(
                    "DEEPSEEK_API_KEY is required for uncached chunk extractions"
                )
            results = await self._chunk_graph().abatch(
                inputs,
                config={"max_concurrency": state["max_concurrency"]},
                return_exceptions=True,
            )
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                summary = "; ".join(f"{type(error).__name__}: {error}" for error in errors[:5])
                raise RuntimeError(f"chunk workers failed; resume run to retry: {summary}")
            processed_ids.update(str(job["id"]) for job in jobs)
            await asyncio.sleep(0)

        with self.database.session_factory() as session:
            final_run = self.repository.get_run(session, state["run_id"])
        if final_run is None:
            raise RuntimeError(f"graph build run disappeared: {state['run_id']}")
        remaining = final_run.total_chunks - (
            final_run.succeeded_chunks + final_run.needs_review_chunks + final_run.failed_chunks
        )
        if remaining:
            raise RuntimeError(
                f"{remaining} chunk extraction(s) still hold active leases; resume later"
            )
        return {"processed_ids": sorted(processed_ids)}

    async def _human_review(self, state: BuildState) -> dict[str, Any]:
        with self.database.session_factory.begin() as session:
            run = self.repository.get_run(session, state["run_id"])
            if run is None:
                raise RuntimeError(f"graph build run disappeared: {state['run_id']}")
            needs_review = run.needs_review_chunks
            if needs_review:
                self.repository.finalize_run(session, state["run_id"], status="awaiting_review")
        if not needs_review:
            return {}

        interrupt(
            {
                "run_id": state["run_id"],
                "needs_review": needs_review,
                "instruction": "Review every xtr_ item, then resume this run.",
            }
        )
        with self.database.session_factory.begin() as session:
            resumed = self.repository.get_run(session, state["run_id"])
            if resumed is None:
                raise RuntimeError(f"graph build run disappeared: {state['run_id']}")
            if resumed.needs_review_chunks:
                raise ReviewStillPendingError(
                    f"{resumed.needs_review_chunks} extraction(s) still need review"
                )
            self.repository.finalize_run(session, state["run_id"], status="running")
        return {}

    def _route_after_review(self, state: BuildState) -> str:
        with self.database.session_factory() as session:
            run = self.repository.get_run(session, state["run_id"])
        if run is None:
            raise RuntimeError(f"graph build run disappeared: {state['run_id']}")
        remaining = run.total_chunks - (
            run.succeeded_chunks + run.needs_review_chunks + run.failed_chunks
        )
        if remaining:
            return "extract"
        if run.total_chunks and not run.succeeded_chunks:
            return "finalize"
        return "normalize"

    async def _normalize_entities(self, state: BuildState) -> dict[str, Any]:
        extractions = self._validated_results(state["run_id"])
        normalized = normalize_entities(
            entity for extraction in extractions for entity in extraction.entities
        )
        self._normalizations[state["run_id"]] = normalized
        return {"normalized_entities": len(normalized.entities)}

    async def _merge_relations(self, state: BuildState) -> dict[str, Any]:
        extractions = self._validated_results(state["run_id"])
        normalized = self._normalizations.get(state["run_id"])
        if normalized is None:
            normalized = normalize_entities(
                entity for extraction in extractions for entity in extraction.entities
            )
            self._normalizations[state["run_id"]] = normalized
        merged = merge_relations(
            (relation for extraction in extractions for relation in extraction.relations),
            normalized.mention_to_entity,
        )
        self._relations[state["run_id"]] = merged
        return {"merged_relations": len(merged.relations)}

    async def _build_networkx(self, state: BuildState) -> dict[str, Any]:
        normalized, merged = self._domain_snapshot(state["run_id"])
        graph = build_networkx_graph(normalized.entities, merged.relations)
        stats = summarize_graph(graph, top_k=state["top_k"])
        self._graph_stats[state["run_id"]] = stats
        return {"graph_metrics": stats.model_dump(mode="json")}

    async def _persist_graph(self, state: BuildState) -> dict[str, Any]:
        normalized, merged = self._domain_snapshot(state["run_id"])
        stats = self._graph_stats.get(state["run_id"])
        if stats is None:
            graph = build_networkx_graph(normalized.entities, merged.relations)
            stats = summarize_graph(graph, top_k=state["top_k"])
            self._graph_stats[state["run_id"]] = stats
        largest = stats.weak_component_sizes[0] if stats.weak_component_sizes else 0
        with self.database.session_factory.begin() as session:
            self.repository.replace_snapshot(
                session,
                state["run_id"],
                normalized.entities,
                merged.relations,
                component_count=stats.weak_components,
                largest_component_nodes=largest,
                isolated_entity_count=len(stats.isolate_ids),
            )
        output_path = state.get("output_path")
        if output_path:
            graph = build_networkx_graph(normalized.entities, merged.relations)
            self._write_graph(Path(output_path), state["run_id"], node_link_json(graph, indent=2))
        return {"graph_metrics": stats.model_dump(mode="json")}

    async def _finalize_run(self, state: BuildState) -> dict[str, Any]:
        with self.database.session_factory.begin() as session:
            self.repository.finalize_run(session, state["run_id"])
        report = self.report(state["run_id"], top_k=state["top_k"])
        with self.database.session_factory.begin() as session:
            run = self.repository.get_run(session, state["run_id"])
            if run is None:
                raise RuntimeError(f"graph build run disappeared: {state['run_id']}")
            envelope = dict(run.report)
            envelope["final_report"] = report.model_dump(mode="json")
            self.repository.finalize_run(session, state["run_id"], report=envelope)
        return {"report": report.model_dump(mode="json")}

    async def _prepare_attempt(self, state: ChunkState) -> dict[str, Any]:
        job = state["job"]
        stage = state.get("stage", "extract")
        if stage == "glean":
            baseline = state.get("baseline_validated")
            if baseline is None:
                raise RuntimeError("cannot glean without a durable validated baseline")
            messages = build_gleaning_messages(str(job["text"]), baseline)
        elif stage == "repair":
            messages = build_repair_messages(
                str(job["text"]),
                state.get("invalid_response"),
                state.get("issues", ()),
            )
        else:
            messages = build_extraction_messages(str(job["text"]))
        with self.database.session_factory.begin() as session:
            claim = self.repository.claim_extraction(
                session,
                str(job["id"]),
                run_id=state["run_id"],
                stage=stage,
                messages=messages,
                lease_seconds=state["lease_seconds"],
                max_attempts=state["max_attempts"],
                max_stage_attempts=(state["gleaning_max_attempts"] if stage == "glean" else None),
            )
        stage_attempt_counts = dict(state.get("stage_attempt_counts", {}))
        extraction_stage_attempt_counts = dict(state.get("extraction_stage_attempt_counts", {}))
        if claim is not None:
            stage_attempt_counts[stage] = stage_attempt_counts.get(stage, 0) + 1
            extraction_stage_attempt_counts[stage] = (
                extraction_stage_attempt_counts.get(stage, 0) + 1
            )
        return {
            "messages": messages,
            "claim": claim,
            "stage_attempt_counts": stage_attempt_counts,
            "extraction_stage_attempt_counts": extraction_stage_attempt_counts,
            "local_attempt": state.get("local_attempt", 0) + (1 if claim else 0),
            "completion": None,
            "provider_error": None,
            "validated": None,
            "validation_error": None,
        }

    async def _claim_and_call(self, state: ChunkState) -> dict[str, Any]:
        prepared = await self._prepare_attempt(state)
        if prepared.get("claim") is None:
            return prepared
        call_state = dict(state)
        call_state.update(prepared)
        called = await self._call_model(call_state)  # type: ignore[arg-type]
        return {**prepared, **called}

    @staticmethod
    def _route_after_claim(state: ChunkState) -> str:
        return "validate" if state.get("claim") is not None else "done"

    async def _call_model(self, state: ChunkState) -> dict[str, Any]:
        if self.client is None:
            raise MissingExtractionCredentialError(
                "DEEPSEEK_API_KEY is required for uncached chunk extractions"
            )
        job = state["job"]
        started = time.perf_counter()
        try:
            if state.get("stage") == "glean":
                baseline = state.get("baseline_validated")
                if baseline is None:
                    raise RuntimeError("cannot glean without a durable validated baseline")
                completion = await self.client.glean(str(job["text"]), baseline)
            elif state.get("stage") == "repair":
                completion = await self.client.repair(
                    str(job["text"]),
                    state.get("invalid_response"),
                    state.get("issues", ()),
                )
            else:
                completion = await self.client.extract(str(job["text"]))
        except ProviderError as error:
            return {
                "provider_error": error,
                "completion": None,
                "latency_seconds": time.perf_counter() - started,
            }
        return {
            "completion": completion,
            "provider_error": None,
            "latency_seconds": time.perf_counter() - started,
        }

    async def _validate(self, state: ChunkState) -> dict[str, Any]:
        provider_error = state.get("provider_error")
        if provider_error is not None:
            return {"validated": None, "validation_error": None}
        completion = state.get("completion")
        if completion is None:
            raise RuntimeError("model call produced neither a completion nor a provider error")
        try:
            validated = validate_completion(
                extraction_id=str(state["job"]["id"]),
                source_chunk_id=str(state["job"]["chunk_id"]),
                chunk_text=str(state["job"]["text"]),
                content=completion.content,
                finish_reason=completion.finish_reason,
            )
        except ExtractionValidationError as error:
            return {"validated": None, "validation_error": error}
        return {"validated": validated, "validation_error": None}

    async def _record_attempt(self, state: ChunkState) -> dict[str, Any]:
        claim = state.get("claim")
        if claim is None:
            raise RuntimeError("cannot persist an unclaimed extraction attempt")
        completion = state.get("completion")
        provider_error = state.get("provider_error")
        validation_error = state.get("validation_error")
        validated = state.get("validated")
        baseline = state.get("baseline_validated")
        if validated is not None:
            if (
                state.get("stage") == "glean"
                and baseline is not None
                and not self._covers_baseline(validated, baseline)
            ):
                outcome = "glean_regressed"
                error_text = "gleaning candidate omitted an accepted first-pass fact"
            else:
                outcome = "succeeded"
                error_text = None
        elif provider_error is not None:
            outcome = (
                "provider_retryable"
                if isinstance(provider_error, RetryableProviderError)
                else "provider_terminal"
            )
            error_text = str(provider_error)
        elif validation_error is not None:
            outcome = validation_error.kind.value
            error_text = str(validation_error)
        else:
            raise RuntimeError("attempt has no validation outcome")
        metadata = self._completion_metadata(completion)
        if state.get("stage") == "glean" and baseline is not None:
            accepted = validated is not None and self._covers_baseline(validated, baseline)
            baseline_entity_facts, baseline_relation_facts = self._fact_signatures(baseline)
            candidate_entity_facts, candidate_relation_facts = (
                self._fact_signatures(validated) if validated is not None else (set(), set())
            )
            metadata.update(
                {
                    "gleaning_accepted": accepted,
                    "baseline_entities": len(baseline.entities),
                    "baseline_relations": len(baseline.relations),
                    "candidate_entities": len(validated.entities) if validated else 0,
                    "candidate_relations": len(validated.relations) if validated else 0,
                    "added_entities": (
                        len(candidate_entity_facts - baseline_entity_facts) if accepted else 0
                    ),
                    "added_relations": (
                        len(candidate_relation_facts - baseline_relation_facts) if accepted else 0
                    ),
                }
            )
        if provider_error is not None:
            metadata.update(
                {
                    "provider_request_id": provider_error.provider_request_id,
                    "status_code": provider_error.status_code,
                }
            )
        with self.database.session_factory.begin() as session:
            self.repository.record_attempt(
                session,
                claim,
                outcome=outcome,
                raw_response=completion.content if completion else None,
                response_metadata=metadata,
                error=error_text,
                prompt_tokens=completion.prompt_tokens if completion else 0,
                completion_tokens=completion.completion_tokens if completion else 0,
                latency_seconds=state.get("latency_seconds"),
            )
            if state.get("stage") == "glean":
                self.repository.persist_provisional_result(
                    session,
                    claim,
                    self._accepted_result(state),
                )
            elif self._should_glean(state):
                claim = state.get("claim")
                validated = state.get("validated")
                if claim is None or validated is None:
                    raise RuntimeError("cannot stage gleaning without an accepted extraction")
                self.repository.stage_gleaning(session, claim, validated)
        return {"outcome": outcome}

    @classmethod
    def _route_after_attempt(cls, state: ChunkState) -> str:
        if state.get("stage") == "glean":
            return "complete"
        if state.get("validated") is not None:
            return "glean" if cls._should_glean(state) else "complete"
        claim = state.get("claim")
        if claim is None or claim.run_attempt_number >= state["max_attempts"]:
            return "fail"
        stage_attempt_counts = state.get("stage_attempt_counts", {})
        repair_max_attempts = state.get("repair_max_attempts", 1)
        if (
            state.get("stage") == "repair"
            and stage_attempt_counts.get("repair", 0) >= repair_max_attempts
        ):
            return "fail"
        provider_error = state.get("provider_error")
        if provider_error is not None:
            return "retry" if isinstance(provider_error, RetryableProviderError) else "fail"
        validation_error = state.get("validation_error")
        if validation_error is None:
            return "fail"
        if (
            validation_error.repairable
            and stage_attempt_counts.get("repair", 0) >= repair_max_attempts
        ):
            return "fail"
        retryable = validation_error.repairable or validation_error.retryable_provider
        return "retry" if retryable else "fail"

    @staticmethod
    def _should_glean(state: ChunkState) -> bool:
        claim = state.get("claim")
        return bool(
            state.get("stage", "extract") == "extract"
            and state.get("validated") is not None
            and state.get("gleaning_max_attempts", 0) > 0
            and state.get("extraction_stage_attempt_counts", {}).get("glean", 0)
            < state["gleaning_max_attempts"]
            and state.get("stage_attempt_counts", {}).get("repair", 0) == 0
            and claim is not None
            and claim.run_attempt_number < state["max_attempts"]
        )

    async def _prepare_gleaning(self, state: ChunkState) -> dict[str, Any]:
        validated = state.get("validated")
        if validated is None:
            raise RuntimeError("cannot prepare gleaning without an accepted extraction")
        return {
            "stage": "glean",
            "baseline_validated": validated,
            "claim": None,
            "completion": None,
            "provider_error": None,
            "validated": None,
            "validation_error": None,
            "invalid_response": None,
            "issues": (),
        }

    async def _prepare_retry(self, state: ChunkState) -> dict[str, Any]:
        claim = state.get("claim")
        if claim is None:
            raise RuntimeError("cannot retry an unclaimed extraction")
        outcome = state.get("outcome")
        if outcome is None:
            raise RuntimeError("retry requested before an attempt outcome was recorded")
        validation_error = state.get("validation_error")
        provider_error = state.get("provider_error")
        error = validation_error or provider_error
        if error is None:
            raise RuntimeError("retry requested without an error")
        with self.database.session_factory.begin() as session:
            self.repository.requeue_extraction(
                session,
                claim,
                error=str(error),
                outcome=outcome,
            )
        delay = state["retry_backoff_seconds"] * (2 ** max(state["local_attempt"] - 1, 0))
        if delay:
            await asyncio.sleep(min(delay, 30.0))
        if validation_error is not None and validation_error.repairable:
            stage = "repair"
            completion = state.get("completion")
            invalid_response = completion.content if completion else None
            issues = validation_error.repair_messages
        else:
            stage = state.get("stage", "extract")
            invalid_response = state.get("invalid_response")
            issues = state.get("issues", ())
        return {
            "stage": stage,
            "invalid_response": invalid_response,
            "issues": issues,
            "claim": None,
        }

    async def _complete_or_review(self, state: ChunkState) -> dict[str, Any]:
        claim = state.get("claim")
        if claim is None:
            raise RuntimeError("cannot complete an extraction without a validated result")
        validated = self._accepted_result(state)
        with self.database.session_factory.begin() as session:
            result = self.repository.complete_extraction(
                session,
                claim,
                validated,
                needs_review=state["review_required"],
            )
        return {"outcome": str(result["status"])}

    @classmethod
    def _accepted_result(cls, state: ChunkState) -> ValidatedChunkExtraction:
        candidate = state.get("validated")
        if state.get("stage") != "glean":
            if candidate is None:
                raise RuntimeError("cannot complete an extraction without a validated result")
            return candidate

        baseline = state.get("baseline_validated")
        if baseline is None:
            raise RuntimeError("cannot complete gleaning without its validated baseline")
        if candidate is None:
            return cls._with_validation_warning(
                baseline,
                "Gleaning failed; retained the accepted first-pass extraction.",
            )
        if not cls._covers_baseline(candidate, baseline):
            return cls._with_validation_warning(
                baseline,
                "Gleaning omitted an accepted first-pass fact; retained the baseline.",
            )
        return candidate

    @staticmethod
    def _with_validation_warning(
        extraction: ValidatedChunkExtraction,
        warning: str,
    ) -> ValidatedChunkExtraction:
        warnings = tuple(dict.fromkeys((*extraction.validation_warnings, warning)))
        return extraction.model_copy(update={"validation_warnings": warnings})

    @classmethod
    def _covers_baseline(
        cls,
        candidate: ValidatedChunkExtraction,
        baseline: ValidatedChunkExtraction,
    ) -> bool:
        baseline_entities, baseline_relations = cls._fact_signatures(baseline)
        candidate_entities, candidate_relations = cls._fact_signatures(candidate)
        return baseline_entities <= candidate_entities and baseline_relations <= candidate_relations

    @staticmethod
    def _fact_signatures(
        extraction: ValidatedChunkExtraction,
    ) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
        mention_keys = {
            entity.id: (normalize_entity_alias(entity.name), entity.entity_type)
            for entity in extraction.entities
        }
        entities = {
            (
                *mention_keys[entity.id],
                entity.description,
                tuple(sorted(entity.aliases, key=lambda value: (value.casefold(), value))),
                tuple(sorted(span.quote for span in entity.evidence)),
            )
            for entity in extraction.entities
        }
        relations = {
            (
                mention_keys[relation.source_mention_id],
                normalize_predicate(relation.predicate),
                mention_keys[relation.target_mention_id],
                relation.description,
                tuple(sorted(span.quote for span in relation.evidence)),
            )
            for relation in extraction.relations
        }
        return entities, relations

    async def _record_failure(self, state: ChunkState) -> dict[str, Any]:
        claim = state.get("claim")
        if claim is None:
            return {"outcome": "skipped"}
        error = state.get("validation_error") or state.get("provider_error")
        message = str(error or "extraction failed without a classified error")
        with self.database.session_factory.begin() as session:
            self.repository.fail_extraction(
                session,
                claim,
                error=message,
                outcome=state.get("outcome", "failed"),
            )
        return {"outcome": "failed"}

    def _validated_results(self, run_id: str) -> tuple[ValidatedChunkExtraction, ...]:
        with self.database.session_factory() as session:
            rows = self.repository.accepted_results(session, run_id)
        values = []
        for row in rows:
            payload = row.get("result")
            if payload is None:
                raise RuntimeError(f"successful extraction has no result: {row['id']}")
            values.append(self._validated_from_payload(payload))
        return tuple(sorted(values, key=lambda item: item.source_chunk_id))

    @staticmethod
    def _validated_from_payload(payload: Any) -> ValidatedChunkExtraction:
        return ValidatedChunkExtraction.model_validate_json(json.dumps(payload, ensure_ascii=False))

    def _domain_snapshot(
        self, run_id: str
    ) -> tuple[EntityNormalizationResult, RelationMergeResult]:
        normalized = self._normalizations.get(run_id)
        merged = self._relations.get(run_id)
        if normalized is not None and merged is not None:
            return normalized, merged
        extractions = self._validated_results(run_id)
        normalized = normalize_entities(
            entity for extraction in extractions for entity in extraction.entities
        )
        merged = merge_relations(
            (relation for extraction in extractions for relation in extraction.relations),
            normalized.mention_to_entity,
        )
        self._normalizations[run_id] = normalized
        self._relations[run_id] = merged
        return normalized, merged

    def report(self, run_id: str, *, top_k: int = 10) -> GraphBuildReport:
        with self.database.session_factory() as session:
            run = self.repository.get_run(session, run_id)
            if run is None:
                raise RuntimeError(f"graph build run not found: {run_id}")
            usage = self._usage_summary(
                run,
                self.repository.deepseek_usage(session, run_id),
            )
            persisted_report = run.report.get("final_report")
            if (
                isinstance(persisted_report, dict)
                and persisted_report.get("run_id") == run.id
                and persisted_report.get("status") == run.status
                and "extraction_quality" in persisted_report
            ):
                return GraphBuildReport.model_validate(persisted_report).model_copy(
                    update={"usage": usage, "deepseek_cost": None}
                )
            stats = self.repository.stats(session, run_id=run_id, top_k=top_k)
            inspected = self.repository.inspect(session, run_id) or {}
            failures = self._failure_summaries(session, inspected)
        extraction_quality = self._extraction_quality(self._validated_results(run_id))
        remaining = max(
            run.total_chunks - run.succeeded_chunks - run.needs_review_chunks - run.failed_chunks,
            0,
        )
        duration = self._duration_seconds(run.started_at, run.finished_at)
        return GraphBuildReport(
            run_id=run.id,
            status=run.status,
            model=run.model,
            extraction_config_hash=run.extraction_config_hash,
            graph_config_hash=run.graph_config_hash,
            corpus_hash=run.corpus_hash,
            started_at=run.started_at,
            finished_at=run.finished_at,
            duration_seconds=duration,
            chunks=ChunkProgress(
                total=run.total_chunks,
                cached=run.cached_chunks,
                scheduled=run.scheduled_chunks,
                succeeded=run.succeeded_chunks,
                needs_review=run.needs_review_chunks,
                failed=run.failed_chunks,
                remaining=remaining,
            ),
            attempts=AttemptSummary(
                total=run.attempt_count,
                extract=run.extract_attempt_count,
                repair=run.repair_attempt_count,
                glean=max(
                    run.attempt_count - run.extract_attempt_count - run.repair_attempt_count,
                    0,
                ),
            ),
            usage=UsageSummary(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cache_hit_tokens=usage.cache_hit_tokens,
                cache_miss_tokens=usage.cache_miss_tokens,
                cache_breakdown_complete=usage.cache_breakdown_complete,
                by_operation_and_model=usage.by_operation_and_model,
            ),
            extraction_quality=extraction_quality,
            graph=GraphSummary(
                nodes=int(stats["nodes"]),
                edges=int(stats["edges"]),
                weakly_connected_components=int(stats["weakly_connected_components"]),
                largest_component_nodes=int(stats["largest_component_nodes"]),
                isolated_nodes=int(stats["isolated_nodes"]),
                top_entities=tuple(
                    TopEntitySummary.model_validate(item) for item in stats["top_entities"]
                ),
            ),
            failures=failures,
        )

    @staticmethod
    def _extraction_quality(
        extractions: Sequence[ValidatedChunkExtraction],
    ) -> ExtractionQualitySummary:
        return ExtractionQualitySummary(
            raw_entities=sum(item.raw_entity_count or len(item.entities) for item in extractions),
            accepted_entities=sum(len(item.entities) for item in extractions),
            dropped_entities=sum(item.dropped_entity_count for item in extractions),
            raw_relations=sum(
                item.raw_relation_count or len(item.relations) for item in extractions
            ),
            accepted_relations=sum(len(item.relations) for item in extractions),
            dropped_relations=sum(item.dropped_relation_count for item in extractions),
            sanitized_relation_records=sum(item.sanitized_relation_records for item in extractions),
            chunks_with_drops=sum(
                item.dropped_entity_count > 0 or item.dropped_relation_count > 0
                for item in extractions
            ),
        )

    @staticmethod
    def _usage_summary(run: Any, records: tuple[DeepSeekUsage, ...]) -> UsageSummary:
        """Summarize persisted response usage without guessing missing cache data."""

        prompt_tokens = int(run.prompt_tokens)
        completion_tokens = int(run.completion_tokens)
        records_prompt = sum(record.prompt_tokens for record in records)
        complete = records_prompt == prompt_tokens and all(
            record.cache_breakdown_complete for record in records
        )
        if not records and prompt_tokens == 0:
            complete = True
        return UsageSummary(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(run.total_tokens),
            cache_hit_tokens=(
                sum(record.cache_hit_tokens or 0 for record in records) if complete else None
            ),
            cache_miss_tokens=(
                sum(record.cache_miss_tokens or 0 for record in records) if complete else None
            ),
            cache_breakdown_complete=complete,
            by_operation_and_model=records,
        )

    def _failure_summaries(
        self, session: Any, inspected_run: Mapping[str, Any]
    ) -> tuple[BuildFailure, ...]:
        failures: list[BuildFailure] = []
        for item in inspected_run.get("items", []):
            if item.get("status") != "failed":
                continue
            extraction = self.repository.inspect(session, str(item["extraction_id"])) or {}
            attempts = extraction.get("attempts", [])
            last = attempts[-1] if attempts else {}
            review_rejected = item.get("review_status") == "rejected"
            failures.append(
                BuildFailure(
                    extraction_id=str(item["extraction_id"]),
                    chunk_id=str(item["chunk_id"]),
                    attempt_id=str(last["id"]) if last.get("id") else None,
                    attempt=int(last.get("ordinal", 0)),
                    stage=str(last["stage"]) if last.get("stage") else None,
                    failure_kind=(
                        "review_rejected" if review_rejected else str(last.get("outcome", "failed"))
                    ),
                    message=str(
                        item.get("review_notes")
                        or extraction.get("error")
                        or last.get("error")
                        or "failed"
                    ),
                )
            )
        return tuple(sorted(failures, key=lambda item: item.extraction_id))

    @staticmethod
    def _completion_metadata(completion: CompletionResult | None) -> dict[str, Any]:
        if completion is None:
            return {}
        return {
            "provider_request_id": completion.provider_request_id,
            "model": completion.model,
            "system_fingerprint": completion.system_fingerprint,
            "finish_reason": completion.finish_reason,
            "cache_hit_tokens": completion.cache_hit_tokens,
            "cache_miss_tokens": completion.cache_miss_tokens,
        }

    @staticmethod
    def _write_graph(path: Path, run_id: str, payload: str) -> None:
        destination = path.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{run_id}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)

    @staticmethod
    def _duration_seconds(start: datetime, end: datetime | None) -> float | None:
        if end is None:
            return None
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        return max((end - start).total_seconds(), 0.0)


def workflow_state(run_id: str, options: WorkflowOptions) -> BuildState:
    return {
        "run_id": run_id,
        "max_concurrency": options.max_concurrency,
        "max_attempts": options.max_attempts,
        "limit": options.limit,
        "retry_failed": options.retry_failed,
        "review_required": options.review_required,
        "top_k": options.top_k,
        "retry_backoff_seconds": options.retry_backoff_seconds,
        "lease_seconds": options.lease_seconds,
        "output_path": str(options.output_path) if options.output_path else None,
    }


def workflow_options_payload(options: WorkflowOptions) -> dict[str, Any]:
    payload = asdict(options)
    payload["output_path"] = str(options.output_path) if options.output_path else None
    return payload
