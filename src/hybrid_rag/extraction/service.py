from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from hybrid_rag.deepseek_costs import DeepSeekPricing
from hybrid_rag.extraction.client import ExtractionClient
from hybrid_rag.extraction.reports import GraphBuildReport, GraphStorageStats
from hybrid_rag.extraction.schemas import ExtractionConfig, GraphConfig
from hybrid_rag.extraction.workflow import (
    WORKFLOW_VERSION,
    GraphBuildWorkflow,
    WorkflowOptions,
    workflow_options_payload,
    workflow_state,
)
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.graph_repository import GraphRepository, GraphRepositoryError


class GraphBuildService:
    """Run or resume a durable graph-build workflow."""

    def __init__(
        self,
        database: Database,
        client: ExtractionClient | None,
        extraction_config: ExtractionConfig,
        *,
        checkpoint_path: Path,
        graph_config: GraphConfig | None = None,
        repository: GraphRepository | None = None,
        deepseek_pricing: DeepSeekPricing | None = None,
    ) -> None:
        self.database = database
        self.client = client
        self.extraction_config = extraction_config
        self.graph_config = graph_config or GraphConfig(
            extraction_config_hash=extraction_config.config_hash
        )
        if self.graph_config.extraction_config_hash != extraction_config.config_hash:
            raise ValueError("graph config must reference the active extraction config")
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.repository = repository or GraphRepository()
        self.deepseek_pricing = deepseek_pricing
        self.workflow = GraphBuildWorkflow(
            database,
            client,
            self.repository,
            repair_max_attempts=extraction_config.repair_max_attempts,
            gleaning_max_attempts=extraction_config.gleaning_max_attempts,
        )
        self.last_run_id: str | None = None

    async def build(
        self,
        options: WorkflowOptions,
        *,
        resume_run_id: str | None = None,
    ) -> GraphBuildReport:
        if resume_run_id is None:
            self._validate_attempt_budget(options)
            run_id = self._begin_run(options)
            effective_options = options
        else:
            run_id = resume_run_id
            run = self._get_run(run_id)
            self._validate_resume_config(run)
            effective_options = self._persisted_options(run.report, fallback=options)
            self._validate_attempt_budget(effective_options)
            if run.status in {"completed", "completed_with_failures", "failed"}:
                return self._with_cost(self.workflow.report(run_id, top_k=effective_options.top_k))
            if run.status == "awaiting_review" and run.needs_review_chunks:
                return self._with_cost(self.workflow.report(run_id, top_k=effective_options.top_k))
        self.last_run_id = run_id

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
        config = {
            "configurable": {"thread_id": run_id},
            "recursion_limit": max(100, effective_options.max_attempts * 10 + 20),
        }
        try:
            async with AsyncSqliteSaver.from_conn_string(str(self.checkpoint_path)) as saver:
                graph = self.workflow.compile(checkpointer=saver)
                graph_input: dict[str, Any] | Command | None
                if resume_run_id is None:
                    graph_input = workflow_state(run_id, effective_options)
                else:
                    snapshot = await graph.aget_state(config)
                    if not snapshot.values:
                        graph_input = workflow_state(run_id, effective_options)
                    elif snapshot.next:
                        current = self._get_run(run_id)
                        graph_input = (
                            Command(resume=True) if current.status == "awaiting_review" else None
                        )
                    else:
                        current = self._get_run(run_id)
                        if current.status == "running":
                            graph_input = workflow_state(run_id, effective_options)
                        else:
                            return self._with_cost(
                                self.workflow.report(run_id, top_k=effective_options.top_k)
                            )
                result = await graph.ainvoke(
                    graph_input,
                    config,
                    durability="sync",
                )
        except KeyboardInterrupt:
            raise
        except Exception as error:
            with self.database.session_factory.begin() as session:
                current = self.repository.get_run(session, run_id)
                if current is not None and current.status != "failed":
                    self.repository.finalize_run(
                        session,
                        run_id,
                        status="running",
                        error=f"{type(error).__name__}: {error}",
                    )
            raise

        if isinstance(result, dict) and result.get("report"):
            report = GraphBuildReport.model_validate(result["report"])
        else:
            report = self.workflow.report(run_id, top_k=effective_options.top_k)
        return self._with_cost(report)

    def _with_cost(self, report: GraphBuildReport) -> GraphBuildReport:
        if self.deepseek_pricing is None:
            return report
        return report.model_copy(
            update={
                "deepseek_cost": self.deepseek_pricing.estimate(report.usage.by_operation_and_model)
            }
        )

    def stats(self, *, run_id: str | None = None, top_k: int = 10) -> GraphStorageStats:
        with self.database.session_factory() as session:
            payload = self.repository.stats(session, run_id=run_id, top_k=top_k)
            stats = GraphStorageStats.model_validate(payload)
            if self.deepseek_pricing is not None and stats.run_id is not None:
                stats = stats.model_copy(
                    update={
                        "deepseek_cost": self.deepseek_pricing.estimate(
                            self.repository.deepseek_usage(session, stats.run_id)
                        )
                    }
                )
        return stats

    def inspect(self, object_id: str, *, raw: bool = False) -> dict[str, Any] | None:
        with self.database.session_factory() as session:
            payload = self.repository.inspect(session, object_id)
        if payload is None or raw:
            return payload
        return self._redact_attempt_payload(payload)

    def review(
        self,
        extraction_id: str,
        *,
        decision: str,
        run_id: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        with self.database.session_factory.begin() as session:
            return self.repository.review_extraction(
                session,
                extraction_id,
                decision=decision,
                run_id=run_id,
                notes=note,
            )

    def _begin_run(self, options: WorkflowOptions) -> str:
        with self.database.session_factory.begin() as session:
            run = self.repository.begin_run(
                session,
                extraction_config_hash=self.extraction_config.config_hash,
                graph_config_hash=self.graph_config.config_hash,
                model=self.extraction_config.model,
                prompt_version=self.extraction_config.prompt_version,
                schema_version=self.extraction_config.schema_version,
                workflow_version=WORKFLOW_VERSION,
                review_required=options.review_required,
                limit=options.limit,
            )
            self.repository.finalize_run(
                session,
                run.id,
                status="running",
                report={
                    "execution": workflow_options_payload(options),
                    "extraction_config": self.extraction_config.model_dump(mode="json"),
                    "graph_config": self.graph_config.model_dump(mode="json"),
                },
            )
        return run.id

    def _get_run(self, run_id: str):
        with self.database.session_factory() as session:
            run = self.repository.get_run(session, run_id)
        if run is None:
            raise GraphRepositoryError(f"graph build run not found: {run_id}")
        return run

    def _validate_resume_config(self, run: Any) -> None:
        if run.extraction_config_hash != self.extraction_config.config_hash:
            raise GraphRepositoryError(
                "resume configuration differs from the persisted extraction configuration"
            )
        if run.graph_config_hash != self.graph_config.config_hash:
            raise GraphRepositoryError(
                "resume configuration differs from the persisted graph configuration"
            )
        if run.workflow_version != WORKFLOW_VERSION:
            raise GraphRepositoryError(
                f"run uses workflow {run.workflow_version}; current version is {WORKFLOW_VERSION}"
            )

    def _validate_attempt_budget(self, options: WorkflowOptions) -> None:
        second_pass_budget = max(
            self.extraction_config.repair_max_attempts,
            self.extraction_config.gleaning_max_attempts,
        )
        minimum = second_pass_budget + 1
        if options.max_attempts < minimum:
            raise ValueError(
                "workflow max_attempts must cover the initial extraction and configured "
                "repair-or-gleaning budget "
                f"({options.max_attempts} < {minimum})"
            )

    @staticmethod
    def _persisted_options(report: dict[str, Any], *, fallback: WorkflowOptions) -> WorkflowOptions:
        values = report.get("execution") if isinstance(report, dict) else None
        if not isinstance(values, dict):
            return fallback
        output = values.get("output_path")
        restored = WorkflowOptions(
            max_concurrency=int(values.get("max_concurrency", fallback.max_concurrency)),
            max_attempts=int(values.get("max_attempts", fallback.max_attempts)),
            limit=values.get("limit", fallback.limit),
            retry_failed=bool(values.get("retry_failed", fallback.retry_failed)),
            review_required=bool(values.get("review_required", fallback.review_required)),
            top_k=int(values.get("top_k", fallback.top_k)),
            retry_backoff_seconds=float(
                values.get("retry_backoff_seconds", fallback.retry_backoff_seconds)
            ),
            lease_seconds=float(values.get("lease_seconds", fallback.lease_seconds)),
            output_path=Path(output) if output else fallback.output_path,
        )
        # Concurrency is execution-only and can be safely tuned while resuming.
        return replace(restored, max_concurrency=fallback.max_concurrency)

    @classmethod
    def _redact_attempt_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        value = dict(payload)
        value.pop("raw_response", None)
        value.pop("messages", None)
        attempts = value.get("attempts")
        if isinstance(attempts, list):
            value["attempts"] = [cls._redact_attempt_payload(dict(attempt)) for attempt in attempts]
        return value
