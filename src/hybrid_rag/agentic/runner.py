"""Evidence-first agent loop built on the existing retrieval and query contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.agentic.graph_snapshot import (
    DEFAULT_PROFILE_GRAPH_CACHE,
    ProfileGraphSnapshot,
    ProfileGraphSnapshotCache,
)
from hybrid_rag.agentic.models import (
    AgentAction,
    AgentActionName,
    AgentBudget,
    AgentEvent,
    AgentRunReport,
    ForkSearchArgs,
    SearchWorkerTask,
    ToolOutcome,
)
from hybrid_rag.agentic.planner import AgentPlanner
from hybrid_rag.evaluation.agentic_metrics import score_agentic_events
from hybrid_rag.retrieval.models import ContextItem, RetrievalResult, RetrievalStrategy
from hybrid_rag.retrieval.query import EvidenceItem, GroundedAnswer, QueryClient
from hybrid_rag.retrieval.service import RetrievalOptions, RetrievalService
from hybrid_rag.storage.retrieval_repository import IndexItem, LoadedIndex


class AgentRunRequest(BaseModel):
    """Public request contract for one agentic retrieval run."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4_000)
    profile_id: str | None = Field(default=None, max_length=200)
    budget: AgentBudget = Field(default_factory=AgentBudget)


class _AgentSession:
    def __init__(
        self,
        *,
        service: RetrievalService,
        profile_id: str,
        budget: AgentBudget,
        retrieval_options: RetrievalOptions,
    ) -> None:
        self.service = service
        self.profile_id = profile_id
        self.budget = budget
        self.retrieval_options = retrieval_options
        self.discovered_chunk_ids: dict[str, ContextItem] = {}
        self.read_chunk_ids: dict[str, ContextItem] = {}
        self.discovered_entity_ids: set[str] = set()
        self.discovered_relation_ids: set[str] = set()
        self.traversed_relation_ids: set[str] = set()
        self.fully_expanded_entity_ids: set[str] = set()
        self.graph_frontier_entity_ids: set[str] = set()
        self.entity_depths: dict[str, int] = {}
        self.searches = 0
        self.graph_expansions = 0
        self.reads = 0
        self.action_keys: set[str] = set()

    def claim_action(self, action: AgentAction) -> str | None:
        key = json.dumps(
            {"action": action.action.value, "args": action.args},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if key in self.action_keys:
            return "The same normalized tool call was already executed in this run."
        self.action_keys.add(key)
        return None


class AgentRunner:
    """Run a bounded planner/tool loop and yield events suitable for SSE or NDJSON."""

    def __init__(
        self,
        service: RetrievalService,
        *,
        planner: AgentPlanner,
        answer_client: QueryClient,
        retrieval_options: RetrievalOptions | None = None,
        audit_dir: Path = Path("artifacts/agent-runs"),
        graph_snapshot_cache: ProfileGraphSnapshotCache | None = None,
    ) -> None:
        self.service = service
        self.planner = planner
        self.answer_client = answer_client
        self.retrieval_options = retrieval_options or RetrievalOptions()
        self.audit_dir = audit_dir
        self.graph_snapshot_cache = graph_snapshot_cache or DEFAULT_PROFILE_GRAPH_CACHE

    async def run(self, request: AgentRunRequest) -> AsyncIterator[AgentEvent]:
        started_clock = perf_counter()
        run_id = f"agr_{uuid4().hex[:16]}"
        profile = await asyncio.to_thread(self.service.resolve_profile, request.profile_id)
        session = _AgentSession(
            service=self.service,
            profile_id=profile.id,
            budget=request.budget,
            retrieval_options=self.retrieval_options,
        )
        index_capabilities = await asyncio.to_thread(
            self._index_capabilities,
            session.profile_id,
        )
        events: list[AgentEvent] = []
        planner_state: list[dict[str, Any]] = []
        started = AgentEvent(
            event="run_started",
            run_id=run_id,
            step=0,
            data={
                "profile_id": profile.id,
                "corpus_content_hash": profile.metadata.get("corpus_content_hash"),
                "index_capabilities": index_capabilities,
                "budget": request.budget.model_dump(mode="json"),
                "retrieval": {
                    "top_k": self.retrieval_options.top_k,
                    "reranker_provider": self.retrieval_options.reranker_provider,
                    "reranker_model": self.retrieval_options.reranker_model,
                    "rerank_candidate_multiplier": (
                        self.retrieval_options.rerank_candidate_multiplier
                    ),
                },
            },
        )
        events.append(started)
        yield started

        termination_reason = "step_budget_exhausted"
        final_answer: GroundedAnswer | None = None
        try:
            for step in range(1, request.budget.max_steps + 1):
                action = await self.planner.next_action(
                    question=request.question,
                    state=tuple(planner_state[-8:]),
                    available_chunk_ids=tuple(session.discovered_chunk_ids),
                    read_chunk_ids=tuple(session.read_chunk_ids),
                    remaining_searches=max(request.budget.max_searches - session.searches, 0),
                    graph_frontier={
                        entity_id: session.entity_depths[entity_id]
                        for entity_id in sorted(session.graph_frontier_entity_ids)
                    },
                    fully_expanded_entity_ids=tuple(sorted(session.fully_expanded_entity_ids)),
                    remaining_graph_expansions=max(
                        request.budget.max_graph_expansions - session.graph_expansions,
                        0,
                    ),
                    max_graph_depth=request.budget.max_graph_hops,
                    index_capabilities=index_capabilities,
                )
                action_event = AgentEvent(
                    event="planner_action",
                    run_id=run_id,
                    step=step,
                    data=action.model_dump(mode="json"),
                )
                events.append(action_event)
                yield action_event

                if action.action is AgentActionName.FINISH:
                    final_answer = _insufficient_answer()
                    termination_reason = "planner_finished_without_answer"
                    break

                outcomes = await self._execute_action(session, request.question, action)
                for outcome in outcomes:
                    outcome_event = AgentEvent(
                        event="tool_result",
                        run_id=run_id,
                        step=step,
                        data=outcome.model_dump(mode="json"),
                    )
                    events.append(outcome_event)
                    yield outcome_event
                    planner_state.append(
                        {
                            "tool": outcome.tool.value,
                            "ok": outcome.ok,
                            "summary": outcome.summary,
                            "data": outcome.data,
                        }
                    )

                if (
                    action.action is AgentActionName.ANSWER_FROM_EVIDENCE
                    and outcomes
                    and outcomes[0].ok
                ):
                    final_answer = GroundedAnswer.model_validate(outcomes[0].data["answer"])
                    termination_reason = "answer_generated"
                    break

            if final_answer is None:
                final_answer = _insufficient_answer()
            answer_event = AgentEvent(
                event="answer",
                run_id=run_id,
                step=len(events),
                data={
                    "answer": final_answer.model_dump(mode="json"),
                    "evidence": [
                        item.model_dump(mode="json") for item in session.read_chunk_ids.values()
                    ],
                },
            )
            events.append(answer_event)
            yield answer_event
            duration_seconds = perf_counter() - started_clock
            completed = AgentEvent(
                event="completed",
                run_id=run_id,
                step=len(events),
                data={
                    "termination_reason": termination_reason,
                    "duration_seconds": duration_seconds,
                },
            )
            events.append(completed)
            self._write_report(
                run_id=run_id,
                request=request,
                profile_id=profile.id,
                corpus_content_hash=_optional_string(profile.metadata.get("corpus_content_hash")),
                events=events,
                status="completed",
                termination_reason=termination_reason,
                duration_seconds=duration_seconds,
            )
            yield completed
        except Exception as error:
            duration_seconds = perf_counter() - started_clock
            failed = AgentEvent(
                event="failed",
                run_id=run_id,
                step=len(events),
                data={
                    "error": f"{type(error).__name__}: {error}",
                    "duration_seconds": duration_seconds,
                },
            )
            events.append(failed)
            self._write_report(
                run_id=run_id,
                request=request,
                profile_id=profile.id,
                corpus_content_hash=_optional_string(profile.metadata.get("corpus_content_hash")),
                events=events,
                status="failed",
                termination_reason="exception",
                duration_seconds=duration_seconds,
            )
            yield failed

    async def _execute_action(
        self,
        session: _AgentSession,
        question: str,
        action: AgentAction,
    ) -> tuple[ToolOutcome, ...]:
        duplicate_reason = (
            None if action.action is AgentActionName.EXPAND_GRAPH else session.claim_action(action)
        )
        if duplicate_reason:
            return (
                ToolOutcome(
                    tool=action.action,
                    ok=False,
                    summary=duplicate_reason,
                ),
            )
        if action.action is AgentActionName.ANSWER_FROM_EVIDENCE:
            return (await self._answer(session, question, action),)
        if action.action is AgentActionName.FORK_SEARCH:
            return await self._fork_search(session, action)
        return (
            await asyncio.to_thread(
                self._execute_sync,
                session,
                question,
                action,
            ),
        )

    async def _fork_search(
        self,
        session: _AgentSession,
        action: AgentAction,
    ) -> tuple[ToolOutcome, ...]:
        try:
            fork = ForkSearchArgs.model_validate(action.args)
        except ValueError as error:
            return (
                ToolOutcome(
                    tool=AgentActionName.FORK_SEARCH,
                    ok=False,
                    summary=f"Invalid fork_search tasks: {error}",
                ),
            )
        remaining = session.budget.max_searches - session.searches
        if len(fork.tasks) > remaining:
            return (
                ToolOutcome(
                    tool=AgentActionName.FORK_SEARCH,
                    ok=False,
                    summary=(
                        f"Parallel search requested {len(fork.tasks)} workers but only "
                        f"{max(remaining, 0)} search budget remains."
                    ),
                ),
            )

        workers = [self._worker_session(session) for _ in fork.tasks]
        session.searches += len(fork.tasks)
        results = await asyncio.gather(
            *(
                self._run_search_worker(worker, task)
                for worker, task in zip(workers, fork.tasks, strict=True)
            )
        )
        for worker in workers:
            session.discovered_chunk_ids.update(worker.discovered_chunk_ids)
            session.discovered_entity_ids.update(worker.discovered_entity_ids)
            session.discovered_relation_ids.update(worker.discovered_relation_ids)
            session.graph_frontier_entity_ids.update(worker.graph_frontier_entity_ids)
            for entity_id, depth in worker.entity_depths.items():
                session.entity_depths[entity_id] = min(
                    session.entity_depths.get(entity_id, depth),
                    depth,
                )
        return tuple(results)

    def _worker_session(self, parent: _AgentSession) -> _AgentSession:
        return _AgentSession(
            service=parent.service,
            profile_id=parent.profile_id,
            budget=parent.budget,
            retrieval_options=parent.retrieval_options,
        )

    async def _run_search_worker(
        self,
        worker: _AgentSession,
        task: SearchWorkerTask,
    ) -> ToolOutcome:
        args: dict[str, Any] = {"query": task.query}
        if task.top_k is not None:
            args["top_k"] = task.top_k
        worker_action = AgentAction(
            action=AgentActionName(task.tool),
            args=args,
            rationale=task.objective,
        )
        try:
            outcome = await asyncio.to_thread(
                self._execute_sync,
                worker,
                task.objective,
                worker_action,
            )
        except Exception as error:
            outcome = ToolOutcome(
                tool=worker_action.action,
                ok=False,
                summary=f"Worker failed: {type(error).__name__}: {error}",
            )
        worker_data = {
            "task_id": task.task_id,
            "objective": task.objective,
            "query": task.query,
        }
        return outcome.model_copy(update={"data": {"worker": worker_data, **outcome.data}})

    def _execute_sync(
        self,
        session: _AgentSession,
        question: str,
        action: AgentAction,
    ) -> ToolOutcome:
        if action.action is AgentActionName.SEARCH_CHUNKS:
            return self._search_chunks(session, action)
        if action.action is AgentActionName.SEARCH_ENTITIES:
            return self._search_graph(
                session,
                action,
                RetrievalStrategy.GRAPH_LOCAL,
                "entity",
            )
        if action.action is AgentActionName.SEARCH_RELATIONS:
            return self._search_graph(
                session,
                action,
                RetrievalStrategy.GRAPH_GLOBAL,
                "relation",
            )
        if action.action is AgentActionName.EXPAND_GRAPH:
            return self._expand_graph(session, action)
        if action.action is AgentActionName.READ_EVIDENCE:
            return self._read_evidence(session, action)
        return ToolOutcome(tool=action.action, ok=False, summary="Unsupported action.")

    def _search_chunks(self, session: _AgentSession, action: AgentAction) -> ToolOutcome:
        if session.searches >= session.budget.max_searches:
            return _budget_outcome(action.action, "search")
        query = _required_string(action.args, "query")
        options = _tool_options(session, action.args)
        result = session.service.retrieve(
            query,
            mode=RetrievalStrategy.HYBRID,
            options=options,
            profile_ref=session.profile_id,
            persist=False,
            model_info={"agentic_tool": action.action.value, "strategy": "dense_bm25"},
        )
        session.searches += 1
        self._register_context(session, result.context_items)
        return ToolOutcome(
            tool=action.action,
            ok=True,
            summary=(
                f"Found {len(result.context_items)} chunk candidates with dense + BM25 retrieval."
            ),
            data={
                "trace_id": result.trace_id,
                "strategy": "dense_bm25",
                "rerank": _rerank_summary(result),
                "chunks": [_context_summary(item) for item in result.context_items],
            },
        )

    def _search_graph(
        self,
        session: _AgentSession,
        action: AgentAction,
        mode: RetrievalStrategy,
        expected_kind: Literal["entity", "relation"],
    ) -> ToolOutcome:
        if session.searches >= session.budget.max_searches:
            return _budget_outcome(action.action, "search")
        query = _required_string(action.args, "query")
        result = session.service.retrieve(
            query,
            mode=mode,
            options=_tool_options(session, action.args),
            profile_ref=session.profile_id,
            persist=False,
            model_info={"agentic_tool": action.action.value},
        )
        session.searches += 1
        self._register_context(session, result.context_items)
        route = result.trace.routes.get(mode.value)
        candidates = [
            _graph_candidate(hit.object_id, hit.metadata, hit.source_chunk_ids, hit.score)
            for hit in (route.hits if route else ())
            if hit.kind == expected_kind
        ]
        if expected_kind == "entity":
            for item in candidates:
                entity_id = str(item["id"])
                session.discovered_entity_ids.add(entity_id)
                if entity_id not in session.fully_expanded_entity_ids:
                    session.graph_frontier_entity_ids.add(entity_id)
                session.entity_depths[entity_id] = 0
        else:
            for item in candidates:
                session.discovered_relation_ids.add(str(item["id"]))
                for key in ("source_entity_id", "target_entity_id"):
                    entity_id = item.get(key)
                    if isinstance(entity_id, str) and entity_id:
                        session.discovered_entity_ids.add(entity_id)
                        if entity_id not in session.fully_expanded_entity_ids:
                            session.graph_frontier_entity_ids.add(entity_id)
                        session.entity_depths[entity_id] = 0
        return ToolOutcome(
            tool=action.action,
            ok=True,
            summary=(
                f"Found {len(candidates)} {expected_kind} candidates and "
                f"{len(result.context_items)} source chunks."
            ),
            data={
                "trace_id": result.trace_id,
                "rerank": _rerank_summary(result),
                f"{expected_kind}s": candidates[:8],
            },
        )

    def _expand_graph(self, session: _AgentSession, action: AgentAction) -> ToolOutcome:
        if session.graph_expansions >= session.budget.max_graph_expansions:
            return _budget_outcome(action.action, "graph expansion")
        entity_ids = _string_list(action.args.get("entity_ids"))
        relation_ids = _string_list(action.args.get("relation_ids"))
        if any(value not in session.discovered_entity_ids for value in entity_ids):
            return ToolOutcome(
                tool=action.action,
                ok=False,
                summary="Graph expansion accepts only entity IDs discovered in this session.",
            )
        if any(value not in session.discovered_relation_ids for value in relation_ids):
            return ToolOutcome(
                tool=action.action,
                ok=False,
                summary="Graph expansion accepts only relation IDs discovered in this session.",
            )
        if not entity_ids and not relation_ids:
            return ToolOutcome(
                tool=action.action,
                ok=False,
                summary="Provide discovered entity_ids or relation_ids.",
            )
        requested_hops = action.args.get("max_hops", 1)
        if isinstance(requested_hops, bool) or requested_hops != 1:
            return ToolOutcome(
                tool=action.action,
                ok=False,
                summary="Incremental graph expansion supports exactly one hop per call.",
            )
        limit = _bounded_int(action.args.get("limit"), default=8, minimum=1, maximum=20)
        snapshot = self._load_graph_snapshot(session)
        relations = snapshot.relations_by_id
        seeds = set(entity_ids)
        for relation_id in relation_ids:
            relation = relations.get(relation_id)
            if relation is None:
                continue
            source = str(relation.metadata["source_entity_id"])
            target = str(relation.metadata["target_entity_id"])
            seeds.update((source, target))
            for entity_id in (source, target):
                session.discovered_entity_ids.add(entity_id)
                session.graph_frontier_entity_ids.add(entity_id)
                session.entity_depths.setdefault(entity_id, 0)

        expandable_seeds = tuple(
            seed
            for seed in sorted(seeds)
            if session.entity_depths.get(seed, 0) < session.budget.max_graph_hops
        )
        neighbors, exhausted_seeds = _incremental_graph_neighbors(
            snapshot,
            expandable_seeds,
            traversed_relation_ids=session.traversed_relation_ids,
            limit=limit,
        )
        entities = snapshot.entities_by_id
        for neighbor in neighbors:
            relation = neighbor.relation
            session.traversed_relation_ids.add(relation.object_id)
            session.discovered_relation_ids.add(relation.object_id)
            session.discovered_entity_ids.add(neighbor.neighbor_entity_id)
            seed_depth = session.entity_depths.get(neighbor.seed_entity_id, 0)
            neighbor_depth = seed_depth + 1
            session.entity_depths[neighbor.neighbor_entity_id] = min(
                session.entity_depths.get(neighbor.neighbor_entity_id, neighbor_depth),
                neighbor_depth,
            )
            if neighbor_depth < session.budget.max_graph_hops:
                session.graph_frontier_entity_ids.add(neighbor.neighbor_entity_id)
            self._register_chunk_ids(session, relation.source_chunk_ids, snapshot)

        for seed in exhausted_seeds:
            session.fully_expanded_entity_ids.add(seed)
            session.graph_frontier_entity_ids.discard(seed)
        for seed in expandable_seeds:
            if seed not in exhausted_seeds:
                session.graph_frontier_entity_ids.add(seed)
        _refresh_graph_frontier(session, snapshot)
        session.graph_expansions += 1
        remaining = session.budget.max_graph_expansions - session.graph_expansions
        return ToolOutcome(
            tool=action.action,
            ok=True,
            summary=(
                f"Discovered {len(neighbors)} new one-hop graph edges; "
                f"{remaining} expansion calls remain."
            ),
            data={
                "neighbors": [
                    _graph_neighbor_payload(neighbor, entities) for neighbor in neighbors
                ],
                "frontier_entity_ids": sorted(session.graph_frontier_entity_ids),
                "fully_expanded_entity_ids": sorted(session.fully_expanded_entity_ids),
                "has_more": any(
                    seed in session.graph_frontier_entity_ids for seed in expandable_seeds
                ),
                "remaining_graph_expansions": remaining,
            },
        )

    def _read_evidence(self, session: _AgentSession, action: AgentAction) -> ToolOutcome:
        if session.reads >= session.budget.max_reads:
            return _budget_outcome(action.action, "evidence read")
        requested = _string_list(action.args.get("chunk_ids"))
        if not requested:
            return ToolOutcome(
                tool=action.action,
                ok=False,
                summary="Provide one or more discovered chunk_ids.",
            )
        if any(chunk_id not in session.discovered_chunk_ids for chunk_id in requested):
            return ToolOutcome(
                tool=action.action,
                ok=False,
                summary="Evidence can only be read after it was discovered in this session.",
            )
        values = [session.discovered_chunk_ids[chunk_id] for chunk_id in requested]
        remaining = session.budget.evidence_token_budget - sum(
            item.token_count for item in session.read_chunk_ids.values()
        )
        selected: list[ContextItem] = []
        for item in values:
            if item.chunk_id in session.read_chunk_ids:
                continue
            if (
                item.token_count <= remaining
                and len(session.read_chunk_ids) + len(selected) < session.budget.max_evidence_chunks
            ):
                selected.append(item)
                remaining -= item.token_count
        if not selected:
            return ToolOutcome(
                tool=action.action,
                ok=False,
                summary="No requested chunk fits the remaining evidence budget.",
            )
        session.read_chunk_ids.update({item.chunk_id: item for item in selected})
        session.reads += 1
        return ToolOutcome(
            tool=action.action,
            ok=True,
            summary=f"Read {len(selected)} source chunks; {remaining} evidence tokens remain.",
            data={"evidence": [item.model_dump(mode="json") for item in selected]},
        )

    async def _answer(
        self,
        session: _AgentSession,
        question: str,
        action: AgentAction,
    ) -> ToolOutcome:
        requested = _string_list(action.args.get("chunk_ids")) or list(session.read_chunk_ids)
        if not requested:
            return ToolOutcome(
                tool=action.action,
                ok=False,
                summary="Read evidence before requesting an answer.",
            )
        if any(chunk_id not in session.read_chunk_ids for chunk_id in requested):
            return ToolOutcome(
                tool=action.action,
                ok=False,
                summary="Answers may cite only chunks read in this session.",
            )
        evidence = tuple(
            EvidenceItem(
                citation_id=item.citation_id,
                text=item.text,
                source_chunk_ids=(item.chunk_id,),
            )
            for item in (session.read_chunk_ids[chunk_id] for chunk_id in requested)
        )
        answer = await self.answer_client.answer(question, evidence)
        return ToolOutcome(
            tool=action.action,
            ok=True,
            summary="Generated an answer from session-scoped source evidence.",
            # Keep the strict Python contract intact while the loop consumes the
            # outcome. The SSE event and audit report serialize this tuple to a
            # JSON array only at their external boundary.
            data={"answer": answer.model_dump()},
        )

    def _register_context(self, session: _AgentSession, items: Sequence[ContextItem]) -> None:
        for item in items:
            if len(session.discovered_chunk_ids) >= session.budget.max_evidence_chunks * 3:
                break
            session.discovered_chunk_ids.setdefault(item.chunk_id, item)

    def _register_chunk_ids(
        self,
        session: _AgentSession,
        chunk_ids: Sequence[str],
        snapshot: ProfileGraphSnapshot,
    ) -> None:
        for chunk_id in chunk_ids:
            item = snapshot.chunks_by_id.get(chunk_id)
            if (
                item is None
                or len(session.discovered_chunk_ids) >= session.budget.max_evidence_chunks * 3
            ):
                continue
            session.discovered_chunk_ids.setdefault(
                chunk_id,
                _context_from_index(item, self.service),
            )

    def _load_graph_snapshot(self, session: _AgentSession) -> ProfileGraphSnapshot:
        def load_index() -> LoadedIndex:
            with self.service.database.session_factory() as database_session:
                return self.service.repository.load_index(database_session, session.profile_id)

        return self.graph_snapshot_cache.get_or_load(
            database_identity=self.service.database.url,
            profile_id=session.profile_id,
            loader=load_index,
        )

    def _index_capabilities(self, profile_id: str) -> dict[str, int]:
        with self.service.database.session_factory() as database_session:
            counts = self.service.repository.index_kind_counts(database_session, profile_id)
        return {
            "chunk_count": counts["chunk"],
            "entity_count": counts["entity"],
            "relation_count": counts["relation"],
        }

    def _write_report(
        self,
        *,
        run_id: str,
        request: AgentRunRequest,
        profile_id: str,
        corpus_content_hash: str | None,
        events: Sequence[AgentEvent],
        status: Literal["completed", "failed"],
        termination_reason: str,
        duration_seconds: float,
    ) -> None:
        metrics = score_agentic_events(
            events,
            duration_seconds=duration_seconds,
        )
        report = AgentRunReport(
            run_id=run_id,
            question=request.question,
            profile_id=profile_id,
            corpus_content_hash=corpus_content_hash,
            planner=type(self.planner).__name__,
            budget=request.budget,
            events=tuple(events),
            status=status,
            termination_reason=termination_reason,
            duration_seconds=duration_seconds,
            metrics=metrics.as_dict(),
        )
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        (self.audit_dir / f"{run_id}.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )


def _tool_options(session: _AgentSession, args: Mapping[str, Any]) -> RetrievalOptions:
    return replace(
        session.retrieval_options,
        top_k=_bounded_int(
            args.get("top_k"),
            default=min(session.retrieval_options.top_k, 8),
            minimum=1,
            maximum=8,
        ),
        context_token_budget=session.budget.evidence_token_budget,
        graph_max_hops=session.budget.max_graph_hops,
    )


def _rerank_summary(result: RetrievalResult) -> dict[str, Any] | None:
    rerank = result.trace.rerank
    if rerank is None:
        return None
    return {
        "provider": rerank.provider,
        "model": rerank.model,
        "version": rerank.version,
        "candidate_count": len(rerank.hits),
    }


@dataclass(frozen=True, slots=True)
class _GraphNeighbor:
    seed_entity_id: str
    neighbor_entity_id: str
    relation: IndexItem
    direction: Literal["incoming", "outgoing"]


def _incremental_graph_neighbors(
    snapshot: ProfileGraphSnapshot,
    seed_entity_ids: Sequence[str],
    *,
    traversed_relation_ids: set[str],
    limit: int,
) -> tuple[tuple[_GraphNeighbor, ...], frozenset[str]]:
    candidates = sorted(
        (
            _GraphNeighbor(
                seed_entity_id=seed,
                neighbor_entity_id=neighbor.neighbor_entity_id,
                relation=neighbor.relation,
                direction=neighbor.direction,
            )
            for seed in seed_entity_ids
            for neighbor in snapshot.adjacency.get(seed, ())
            if neighbor.relation.object_id not in traversed_relation_ids
        ),
        key=lambda neighbor: (
            neighbor.seed_entity_id,
            neighbor.relation.object_id,
            neighbor.neighbor_entity_id,
        ),
    )
    selected: list[_GraphNeighbor] = []
    selected_relation_ids: set[str] = set()
    for neighbor in candidates:
        relation_id = neighbor.relation.object_id
        if relation_id in selected_relation_ids:
            continue
        selected.append(neighbor)
        selected_relation_ids.add(relation_id)
        if len(selected) >= limit:
            break

    visible_relation_ids = traversed_relation_ids | selected_relation_ids
    exhausted = frozenset(
        seed
        for seed in seed_entity_ids
        if all(
            neighbor.relation.object_id in visible_relation_ids
            for neighbor in snapshot.adjacency.get(seed, ())
        )
    )
    return tuple(selected), exhausted


def _graph_neighbor_payload(
    neighbor: _GraphNeighbor,
    entities: Mapping[str, IndexItem],
) -> dict[str, Any]:
    entity = entities.get(neighbor.neighbor_entity_id)
    metadata = entity.metadata if entity is not None else {}
    relation_metadata = neighbor.relation.metadata
    return {
        "seed_entity_id": neighbor.seed_entity_id,
        "entity": {
            "id": neighbor.neighbor_entity_id,
            "label": str(metadata.get("canonical_name") or neighbor.neighbor_entity_id),
            "entity_type": str(metadata.get("entity_type") or ""),
        },
        "relation": {
            "id": neighbor.relation.object_id,
            "predicate": str(relation_metadata.get("predicate") or ""),
            "direction": neighbor.direction,
        },
        "source_chunk_ids": list(neighbor.relation.source_chunk_ids),
    }


def _refresh_graph_frontier(
    session: _AgentSession,
    snapshot: ProfileGraphSnapshot,
) -> None:
    for entity_id in session.discovered_entity_ids:
        depth = session.entity_depths.get(entity_id, 0)
        if depth >= session.budget.max_graph_hops:
            session.graph_frontier_entity_ids.discard(entity_id)
            continue
        remaining = (
            snapshot.incident_relation_ids.get(entity_id, frozenset())
            - session.traversed_relation_ids
        )
        if remaining:
            session.graph_frontier_entity_ids.add(entity_id)
            continue
        session.graph_frontier_entity_ids.discard(entity_id)
        session.fully_expanded_entity_ids.add(entity_id)


def _context_from_index(item: IndexItem, service: RetrievalService) -> ContextItem:
    metadata = item.metadata
    text = str(metadata["text"])
    return ContextItem(
        citation_id=item.object_id,
        chunk_id=item.object_id,
        document_id=str(metadata["document_id"]),
        document_title=str(metadata["document_title"]),
        section_path=tuple(str(value) for value in metadata.get("section_path", [])),
        page_start=_optional_int(metadata.get("page_start")),
        page_end=_optional_int(metadata.get("page_end")),
        text=text,
        token_count=service.token_counter.count(text),
        score=0.0,
    )


def _context_summary(item: ContextItem) -> dict[str, Any]:
    return {
        "chunk_id": item.chunk_id,
        "document_title": item.document_title,
        "section_path": list(item.section_path),
        "score": item.score,
        "preview": item.text[:360],
    }


def _graph_candidate(
    object_id: str,
    metadata: Mapping[str, Any],
    source_chunk_ids: Sequence[str],
    score: float,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": object_id,
        "score": score,
        "label": str(metadata.get("canonical_name") or metadata.get("predicate") or object_id),
        "source_chunk_ids": list(source_chunk_ids),
    }
    for key in ("entity_type", "source_entity_id", "target_entity_id"):
        candidate = metadata.get(key)
        if isinstance(candidate, str) and candidate:
            value[key] = candidate
    return value


def _insufficient_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer="Insufficient evidence was collected to answer this question.",
        citations=(),
        insufficient_evidence=True,
    )


def _budget_outcome(action: AgentActionName, name: str) -> ToolOutcome:
    return ToolOutcome(tool=action, ok=False, summary=f"The {name} budget has been exhausted.")


def _required_string(args: Mapping[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _enum_string(args: Mapping[str, Any], key: str, allowed: set[str], default: str) -> str:
    value = args.get(key, default)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{key} must be one of: {', '.join(sorted(allowed))}")
    return value


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("numeric tool arguments must be integers")
    return max(minimum, min(maximum, value))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError("tool ID arguments must be lists of non-empty strings")
    return list(dict.fromkeys(value))


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
