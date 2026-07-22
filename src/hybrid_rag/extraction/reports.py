from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.deepseek_costs import DeepSeekCostSummary, DeepSeekUsage


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
    glean: int = Field(default=0, ge=0)


class UsageSummary(_ReportModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cache_hit_tokens: int | None = Field(default=None, ge=0)
    cache_miss_tokens: int | None = Field(default=None, ge=0)
    cache_breakdown_complete: bool = False
    by_operation_and_model: tuple[DeepSeekUsage, ...] = ()


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


class ExtractionQualitySummary(_ReportModel):
    raw_entities: int = Field(default=0, ge=0)
    accepted_entities: int = Field(default=0, ge=0)
    dropped_entities: int = Field(default=0, ge=0)
    raw_relations: int = Field(default=0, ge=0)
    accepted_relations: int = Field(default=0, ge=0)
    dropped_relations: int = Field(default=0, ge=0)
    sanitized_relation_records: int = Field(default=0, ge=0)
    chunks_with_drops: int = Field(default=0, ge=0)


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
    deepseek_cost: DeepSeekCostSummary | None = None
    extraction_quality: ExtractionQualitySummary = ExtractionQualitySummary()
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
    deepseek_cost: DeepSeekCostSummary | None = None
