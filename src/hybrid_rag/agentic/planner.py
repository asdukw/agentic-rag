"""Planner adapters: deterministic offline policy and DeepSeek JSON policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from hybrid_rag.agentic.models import AgentAction, AgentActionName
from hybrid_rag.agentic.prompts import build_planner_messages
from hybrid_rag.extraction.client import DeepSeekClient


class AgentPlanner(Protocol):
    async def next_action(
        self,
        *,
        question: str,
        state: Sequence[dict[str, Any]],
        available_chunk_ids: Sequence[str],
        read_chunk_ids: Sequence[str],
        remaining_searches: int,
    ) -> AgentAction: ...


class DeterministicAgentPlanner:
    """Offline policy that still exercises the same tool/session boundaries."""

    async def next_action(
        self,
        *,
        question: str,
        state: Sequence[dict[str, Any]],
        available_chunk_ids: Sequence[str],
        read_chunk_ids: Sequence[str],
        remaining_searches: int,
    ) -> AgentAction:
        if not state:
            return AgentAction(
                action=AgentActionName.SEARCH_CHUNKS,
                args={"query": question, "strategy": "hybrid"},
                rationale="Start with combined lexical and semantic chunk evidence.",
            )
        if available_chunk_ids and not read_chunk_ids:
            return AgentAction(
                action=AgentActionName.READ_EVIDENCE,
                args={"chunk_ids": list(available_chunk_ids[:6])},
                rationale="Read the retrieved source passages before answering.",
            )
        if read_chunk_ids:
            return AgentAction(
                action=AgentActionName.ANSWER_FROM_EVIDENCE,
                args={"chunk_ids": list(read_chunk_ids)},
                rationale="Answer only from passages read in this session.",
            )
        return AgentAction(
            action=AgentActionName.FINISH,
            rationale="No source evidence was found.",
        )


class DeepSeekAgentPlanner:
    """One-action JSON planner backed by the existing OpenAI-compatible adapter."""

    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    async def next_action(
        self,
        *,
        question: str,
        state: Sequence[dict[str, Any]],
        available_chunk_ids: Sequence[str],
        read_chunk_ids: Sequence[str],
        remaining_searches: int,
    ) -> AgentAction:
        completion = await self.client.complete_messages(
            build_planner_messages(
                question=question,
                state=state,
                available_chunk_ids=available_chunk_ids,
                read_chunk_ids=read_chunk_ids,
                remaining_searches=remaining_searches,
            )
        )
        if not completion.content:
            raise ValueError("agent planner returned an empty completion")
        return AgentAction.model_validate_json(completion.content)

    async def close(self) -> None:
        await self.client.close()
