"""Serializable contracts for one bounded agentic retrieval run."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AGENT_ACTION_RATIONALE_MAX_LENGTH = 500


class AgentActionName(StrEnum):
    FORK_SEARCH = "fork_search"
    SEARCH_CHUNKS = "search_chunks"
    SEARCH_ENTITIES = "search_entities"
    SEARCH_RELATIONS = "search_relations"
    EXPAND_GRAPH = "expand_graph"
    READ_EVIDENCE = "read_evidence"
    ANSWER_FROM_EVIDENCE = "answer_from_evidence"
    FINISH = "finish"


class AgentAction(BaseModel):
    """One planner-selected action; arguments are validated by the receiving tool."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: AgentActionName
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=AGENT_ACTION_RATIONALE_MAX_LENGTH)


class SearchWorkerTask(BaseModel):
    """One isolated, read-only retrieval task emitted by the main planner."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    task_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    objective: str = Field(min_length=1, max_length=300)
    tool: Literal["search_chunks", "search_entities", "search_relations"]
    query: str = Field(min_length=1, max_length=1_000)
    strategy: Literal["dense", "bm25", "dense_bm25"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("strategy", mode="before")
    @classmethod
    def normalize_legacy_strategy(cls, value: object) -> object:
        return "dense_bm25" if value == "hybrid" else value

    @model_validator(mode="after")
    def validate_strategy(self) -> SearchWorkerTask:
        if self.tool == "search_chunks":
            return self
        if self.strategy is not None:
            raise ValueError("strategy is only valid for search_chunks workers")
        return self


class ForkSearchArgs(BaseModel):
    """Bounded fan-out selected by the main planner."""

    model_config = ConfigDict(extra="forbid", strict=True)

    tasks: list[SearchWorkerTask] = Field(min_length=2, max_length=3)

    @model_validator(mode="after")
    def validate_task_ids(self) -> ForkSearchArgs:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("fork_search task IDs must be unique")
        return self


class AgentBudget(BaseModel):
    """Hard server-side limits for one run, never chosen by the planner."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_steps: int = Field(default=6, ge=1, le=12)
    max_searches: int = Field(default=3, ge=1, le=8)
    max_graph_expansions: int = Field(default=2, ge=0, le=4)
    max_reads: int = Field(default=2, ge=1, le=4)
    max_evidence_chunks: int = Field(default=8, ge=1, le=16)
    max_graph_hops: int = Field(default=2, ge=1, le=2)
    evidence_token_budget: int = Field(default=2400, ge=128, le=8000)


class AgentEvent(BaseModel):
    """One JSON-serializable event emitted by :class:`AgentRunner`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    event: Literal[
        "run_started",
        "planner_action",
        "tool_result",
        "answer",
        "completed",
        "failed",
    ]
    run_id: str
    step: int = Field(ge=0)
    data: dict[str, Any] = Field(default_factory=dict)


class ToolOutcome(BaseModel):
    """A compact tool result safe to reveal to both planner and browser."""

    model_config = ConfigDict(extra="forbid", strict=True)

    tool: AgentActionName
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class AgentRunReport(BaseModel):
    """Durable, inspectable report written after a completed or failed run."""

    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: str
    question: str
    profile_id: str
    corpus_content_hash: str | None = None
    planner: str
    budget: AgentBudget
    events: tuple[AgentEvent, ...]
    status: Literal["completed", "failed"]
    termination_reason: str
    duration_seconds: float | None = Field(default=None, ge=0.0)
    metrics: dict[str, Any] = Field(default_factory=dict)
