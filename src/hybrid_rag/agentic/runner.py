"""Evidence-first agent loop built on the existing retrieval and query contracts."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.agentic.models import (
    AgentAction,
    AgentActionName,
    AgentBudget,
    AgentEvent,
    AgentRunReport,
    ToolOutcome,
)
from hybrid_rag.agentic.planner import AgentPlanner
from hybrid_rag.retrieval.models import ContextItem, GraphPath, RetrievalMode
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
    ) -> None:
        self.service = service
        self.profile_id = profile_id
        self.budget = budget
        self.discovered_chunk_ids: dict[str, ContextItem] = {}
        self.read_chunk_ids: dict[str, ContextItem] = {}
        self.discovered_entity_ids: set[str] = set()
        self.discovered_relation_ids: set[str] = set()
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
        audit_dir: Path = Path("artifacts/agent-runs"),
    ) -> None:
        self.service = service
        self.planner = planner
        self.answer_client = answer_client
        self.audit_dir = audit_dir

    async def run(self, request: AgentRunRequest) -> AsyncIterator[AgentEvent]:
        run_id = f"agr_{uuid4().hex[:16]}"
        profile = await asyncio.to_thread(self.service.resolve_profile, request.profile_id)
        session = _AgentSession(
            service=self.service,
            profile_id=profile.id,
            budget=request.budget,
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
                "budget": request.budget.model_dump(mode="json"),
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

                duplicate_reason = session.claim_action(action)
                if duplicate_reason:
                    outcome = ToolOutcome(
                        tool=action.action,
                        ok=False,
                        summary=duplicate_reason,
                    )
                elif action.action is AgentActionName.ANSWER_FROM_EVIDENCE:
                    outcome = await self._answer(session, request.question, action)
                else:
                    outcome = await asyncio.to_thread(
                        self._execute_sync,
                        session,
                        request.question,
                        action,
                    )
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

                if action.action is AgentActionName.ANSWER_FROM_EVIDENCE and outcome.ok:
                    final_answer = GroundedAnswer.model_validate(outcome.data["answer"])
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
            completed = AgentEvent(
                event="completed",
                run_id=run_id,
                step=len(events),
                data={"termination_reason": termination_reason},
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
            )
            yield completed
        except Exception as error:
            failed = AgentEvent(
                event="failed",
                run_id=run_id,
                step=len(events),
                data={"error": f"{type(error).__name__}: {error}"},
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
            )
            yield failed

    def _execute_sync(
        self,
        session: _AgentSession,
        question: str,
        action: AgentAction,
    ) -> ToolOutcome:
        if action.action is AgentActionName.SEARCH_CHUNKS:
            return self._search_chunks(session, action)
        if action.action is AgentActionName.SEARCH_ENTITIES:
            return self._search_graph(session, action, RetrievalMode.LOCAL, "entity")
        if action.action is AgentActionName.SEARCH_RELATIONS:
            return self._search_graph(session, action, RetrievalMode.GLOBAL, "relation")
        if action.action is AgentActionName.EXPAND_GRAPH:
            return self._expand_graph(session, action)
        if action.action is AgentActionName.READ_EVIDENCE:
            return self._read_evidence(session, action)
        return ToolOutcome(tool=action.action, ok=False, summary="Unsupported action.")

    def _search_chunks(self, session: _AgentSession, action: AgentAction) -> ToolOutcome:
        if session.searches >= session.budget.max_searches:
            return _budget_outcome(action.action, "search")
        query = _required_string(action.args, "query")
        strategy = _enum_string(action.args, "strategy", {"dense", "bm25", "hybrid"}, "hybrid")
        options = _tool_options(session.budget, action.args)
        if strategy == "dense":
            options = replace(options, naive_dense_weight=1.0, naive_bm25_weight=0.0)
        elif strategy == "bm25":
            options = replace(options, naive_dense_weight=0.0, naive_bm25_weight=1.0)
        result = session.service.retrieve(
            query,
            mode=RetrievalMode.NAIVE,
            options=options,
            profile_ref=session.profile_id,
            persist=True,
            model_info={"agentic_tool": action.action.value, "strategy": strategy},
        )
        session.searches += 1
        self._register_context(session, result.context_items)
        return ToolOutcome(
            tool=action.action,
            ok=True,
            summary=(
                f"Found {len(result.context_items)} chunk candidates with {strategy} retrieval."
            ),
            data={
                "trace_id": result.trace_id,
                "strategy": strategy,
                "chunks": [_context_summary(item) for item in result.context_items],
            },
        )

    def _search_graph(
        self,
        session: _AgentSession,
        action: AgentAction,
        mode: RetrievalMode,
        expected_kind: Literal["entity", "relation"],
    ) -> ToolOutcome:
        if session.searches >= session.budget.max_searches:
            return _budget_outcome(action.action, "search")
        query = _required_string(action.args, "query")
        result = session.service.retrieve(
            query,
            mode=mode,
            options=_tool_options(session.budget, action.args),
            profile_ref=session.profile_id,
            persist=True,
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
            session.discovered_entity_ids.update(item["id"] for item in candidates)
        else:
            session.discovered_relation_ids.update(item["id"] for item in candidates)
        return ToolOutcome(
            tool=action.action,
            ok=True,
            summary=(
                f"Found {len(candidates)} {expected_kind} candidates and "
                f"{len(result.context_items)} source chunks."
            ),
            data={"trace_id": result.trace_id, f"{expected_kind}s": candidates[:8]},
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
        max_hops = min(
            _bounded_int(action.args.get("max_hops"), default=1, minimum=1, maximum=2),
            session.budget.max_graph_hops,
        )
        index = self._load_index(session)
        paths = _bounded_paths(index, entity_ids, relation_ids, max_hops=max_hops)
        for path in paths:
            self._register_chunk_ids(session, path.source_chunk_ids, index)
        session.graph_expansions += 1
        return ToolOutcome(
            tool=action.action,
            ok=True,
            summary=f"Expanded {len(paths)} provenance-bound graph paths up to {max_hops} hops.",
            data={"paths": [path.model_dump(mode="json") for path in paths]},
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
        index: LoadedIndex,
    ) -> None:
        chunks = {item.object_id: item for item in index.chunks}
        for chunk_id in chunk_ids:
            item = chunks.get(chunk_id)
            if (
                item is None
                or len(session.discovered_chunk_ids) >= session.budget.max_evidence_chunks * 3
            ):
                continue
            session.discovered_chunk_ids.setdefault(
                chunk_id,
                _context_from_index(item, self.service),
            )

    def _load_index(self, session: _AgentSession) -> LoadedIndex:
        with self.service.database.session_factory() as database_session:
            return self.service.repository.load_index(database_session, session.profile_id)

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
    ) -> None:
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
        )
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        (self.audit_dir / f"{run_id}.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )


def _tool_options(budget: AgentBudget, args: Mapping[str, Any]) -> RetrievalOptions:
    return RetrievalOptions(
        top_k=_bounded_int(args.get("top_k"), default=6, minimum=1, maximum=8),
        context_token_budget=budget.evidence_token_budget,
        graph_max_hops=budget.max_graph_hops,
        reranker_provider="none",
    )


def _bounded_paths(
    index: LoadedIndex,
    entity_ids: Sequence[str],
    relation_ids: Sequence[str],
    *,
    max_hops: int,
) -> tuple[GraphPath, ...]:
    relations = {item.object_id: item for item in index.relations}
    graph = nx.Graph()
    by_entity: dict[str, list[IndexItem]] = defaultdict(list)
    for relation in relations.values():
        source = str(relation.metadata["source_entity_id"])
        target = str(relation.metadata["target_entity_id"])
        graph.add_edge(source, target, relation_id=relation.object_id)
        by_entity[source].append(relation)
        by_entity[target].append(relation)
    seeds = set(entity_ids)
    for relation_id in relation_ids:
        relation = relations.get(relation_id)
        if relation is not None:
            seeds.add(str(relation.metadata["source_entity_id"]))
            seeds.add(str(relation.metadata["target_entity_id"]))
    paths: list[GraphPath] = []
    for seed in sorted(seeds):
        if seed not in graph:
            continue
        queue: deque[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = deque(
            [(seed, (seed,), (), ())]
        )
        while queue and len(paths) < 12:
            current, nodes, traversed, chunk_ids = queue.popleft()
            if traversed:
                paths.append(
                    GraphPath(
                        node_ids=nodes,
                        relation_ids=traversed,
                        source_chunk_ids=tuple(sorted(set(chunk_ids))),
                        score=1.0 / len(traversed),
                    )
                )
            if len(traversed) == max_hops:
                continue
            for relation in sorted(by_entity.get(current, []), key=lambda item: item.object_id):
                left = str(relation.metadata["source_entity_id"])
                right = str(relation.metadata["target_entity_id"])
                neighbor = right if current == left else left
                if neighbor in nodes:
                    continue
                queue.append(
                    (
                        neighbor,
                        (*nodes, neighbor),
                        (*traversed, relation.object_id),
                        (*chunk_ids, *relation.source_chunk_ids),
                    )
                )
    unique: dict[tuple[str, ...], GraphPath] = {}
    for path in paths:
        unique.setdefault(path.relation_ids, path)
    return tuple(unique.values())[:12]


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
    return {
        "id": object_id,
        "score": score,
        "label": str(metadata.get("canonical_name") or metadata.get("predicate") or object_id),
        "source_chunk_ids": list(source_chunk_ids),
    }


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
