"""Strict, serializable domain contracts for phase-three retrieval."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hybrid_rag.deepseek_costs import DeepSeekCostSummary
from hybrid_rag.ids import canonical_json_hash

RETRIEVAL_SCHEMA_VERSION = "3"
RETRIEVAL_MODE_SEMANTICS_VERSION = "lightrag-v2"
INDEX_TEXT_SCHEMA_VERSION = "1"


class RetrievalMode(StrEnum):
    NAIVE = "naive"
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    MIX = "mix"


IndexKind = Literal["chunk", "entity", "relation"]


class _StrictRetrievalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class IndexSemanticConfig(_StrictRetrievalModel):
    """Settings that change an embedding index's semantic identity."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    text_schema_version: str = INDEX_TEXT_SCHEMA_VERSION
    provider_options: dict[str, str | int | bool] = Field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        payload = self.model_dump(mode="json")
        # Preserve profile IDs created before provider-specific options existed.
        if not self.provider_options:
            payload.pop("provider_options")
        return canonical_json_hash(payload)


class ScoreComponent(_StrictRetrievalModel):
    """One named contribution to a route score, retained for inspection."""

    raw_score: float
    normalized_score: float = Field(ge=0.0, le=1.0)
    weighted_score: float = Field(ge=0.0)


class CandidateHit(_StrictRetrievalModel):
    """One scored object retained in an explainable retrieval route."""

    object_id: str = Field(min_length=1)
    kind: IndexKind
    score: float
    raw_score: float | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    rank: int = Field(ge=1)
    route_scores: dict[str, float] = Field(default_factory=dict)
    score_components: dict[str, ScoreComponent] = Field(default_factory=dict)
    source_chunk_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sources(self) -> CandidateHit:
        if len(self.source_chunk_ids) != len(set(self.source_chunk_ids)):
            raise ValueError("source_chunk_ids must not contain duplicates")
        return self


class RouteTrace(_StrictRetrievalModel):
    """Raw candidates for one deterministic recall route."""

    route: RetrievalMode
    candidate_count: int = Field(ge=0)
    hits: tuple[CandidateHit, ...] = ()


class RerankComponentTrace(_StrictRetrievalModel):
    """One explicit contribution retained by a second-stage reranker."""

    raw_score: float
    normalized_score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    weighted_score: float = Field(ge=0.0)


class RerankTraceHit(_StrictRetrievalModel):
    """One candidate before and after the deterministic rerank stage."""

    object_id: str = Field(min_length=1)
    pre_rerank_rank: int = Field(ge=1)
    pre_rerank_score: float
    score: float = Field(ge=0.0)
    final_rank: int = Field(ge=1)
    components: dict[str, RerankComponentTrace] = Field(default_factory=dict)


class RerankTrace(_StrictRetrievalModel):
    """Replayable identity and scores for the post-retrieval rerank stage."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    version: str = Field(min_length=1)
    candidate_limit: int = Field(ge=1)
    hits: tuple[RerankTraceHit, ...] = ()

    @model_validator(mode="after")
    def validate_candidates(self) -> RerankTrace:
        if len(self.hits) > self.candidate_limit:
            raise ValueError("rerank trace has more hits than its candidate limit")
        if len({hit.object_id for hit in self.hits}) != len(self.hits):
            raise ValueError("rerank trace must not contain duplicate candidate IDs")
        return self


class GraphPath(_StrictRetrievalModel):
    """A bounded, retrieval-selected graph path supporting the context."""

    node_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    source_chunk_ids: tuple[str, ...] = ()
    score: float = 0.0

    @model_validator(mode="after")
    def validate_path_shape(self) -> GraphPath:
        if self.relation_ids and len(self.node_ids) != len(self.relation_ids) + 1:
            raise ValueError("a graph path requires one more node than relation IDs")
        if len(self.relation_ids) != len(set(self.relation_ids)):
            raise ValueError("graph paths must not repeat relations")
        if len(self.source_chunk_ids) != len(set(self.source_chunk_ids)):
            raise ValueError("source_chunk_ids must not contain duplicates")
        return self


class ContextItem(_StrictRetrievalModel):
    """One selected source chunk and its stable citation identity."""

    citation_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    section_path: tuple[str, ...] = ()
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)
    token_count: int = Field(ge=0)
    score: float
    route_scores: dict[str, float] = Field(default_factory=dict)
    source_entity_ids: tuple[str, ...] = ()
    source_relation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_citation(self) -> ContextItem:
        if self.citation_id != self.chunk_id:
            raise ValueError("the current citation contract requires citation_id == chunk_id")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must not be before page_start")
        return self


class RetrievalTrace(_StrictRetrievalModel):
    """All decisions needed to inspect or replay a retrieval result."""

    schema_version: str = RETRIEVAL_SCHEMA_VERSION
    profile_id: str = Field(min_length=1)
    index_config_hash: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expanded_query: str = Field(min_length=1)
    mode: RetrievalMode
    keywords: tuple[str, ...] = ()
    routes: dict[str, RouteTrace] = Field(default_factory=dict)
    rerank: RerankTrace | None = None
    fused_hits: tuple[CandidateHit, ...] = ()
    graph_paths: tuple[GraphPath, ...] = ()
    context_items: tuple[ContextItem, ...] = ()
    context_token_budget: int = Field(ge=1)
    context_tokens: int = Field(ge=0)
    settings: dict[str, Any] = Field(default_factory=dict)
    deepseek_cost: DeepSeekCostSummary | None = None

    @model_validator(mode="after")
    def validate_context_budget(self) -> RetrievalTrace:
        if self.context_tokens > self.context_token_budget:
            raise ValueError("context_tokens exceeds context_token_budget")
        return self


class RetrievalResult(_StrictRetrievalModel):
    """The shared result schema returned by all five retriever modes."""

    trace_id: str | None = None
    profile_id: str = Field(min_length=1)
    mode: RetrievalMode
    query: str = Field(min_length=1)
    keywords: tuple[str, ...] = ()
    hits: tuple[CandidateHit, ...] = ()
    graph_paths: tuple[GraphPath, ...] = ()
    context_items: tuple[ContextItem, ...] = ()
    context: str
    context_tokens: int = Field(ge=0)
    trace: RetrievalTrace


class IndexBuildReport(_StrictRetrievalModel):
    """Outcome of one deterministic three-way index build."""

    profile_id: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    corpus_content_hash: str = Field(min_length=1)
    source_corpus_hash: str = Field(min_length=1)
    graph_build_run_id: str | None = None
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    chunks: int = Field(ge=0)
    entities: int = Field(ge=0)
    relations: int = Field(ge=0)
    reused: bool
