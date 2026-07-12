from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkProgress(_ReportModel):
    total: int = Field(ge=0)
    cached: int = Field(ge=0)
    scheduled: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    needs_review: int = Field(ge=0)
    failed: int = Field(ge=0)
    remaining: int = Field(ge=0)


class AttemptSummary(_ReportModel):
    total: int = Field(ge=0)
    extract: int = Field(ge=0)
    repair: int = Field(ge=0)


class UsageSummary(_ReportModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class TopEntitySummary(_ReportModel):
    id: str
    name: str
    type: str
    degree: int = Field(ge=0)
    source_chunks: int = Field(ge=0)


class GraphSummary(_ReportModel):
    nodes: int = Field(ge=0)
    edges: int = Field(ge=0)
    weakly_connected_components: int = Field(ge=0)
    largest_component_nodes: int = Field(ge=0)
    isolated_nodes: int = Field(ge=0)
    top_entities: tuple[TopEntitySummary, ...] = ()


class BuildFailure(_ReportModel):
    extraction_id: str
    chunk_id: str
    attempt_id: str | None = None
    attempt: int = Field(ge=0)
    stage: str | None = None
    failure_kind: str
    message: str


class GraphBuildReport(_ReportModel):
    run_id: str
    status: str
    model: str
    extraction_config_hash: str
    graph_config_hash: str
    corpus_hash: str
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    chunks: ChunkProgress
    attempts: AttemptSummary
    usage: UsageSummary
    graph: GraphSummary
    failures: tuple[BuildFailure, ...] = ()


class GraphStorageStats(_ReportModel):
    run_id: str | None = None
    status: str | None = None
    chunks: dict[str, int] = Field(default_factory=dict)
    attempts: dict[str, int] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    nodes: int = Field(ge=0)
    edges: int = Field(ge=0)
    weakly_connected_components: int = Field(ge=0)
    largest_component_nodes: int = Field(ge=0)
    isolated_nodes: int = Field(ge=0)
    top_entities: tuple[TopEntitySummary, ...] = ()
