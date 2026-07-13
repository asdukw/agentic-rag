"""Project-owned indexing, retrieval, trace, and evidence-answer orchestration."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import combinations, pairwise
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict

from hybrid_rag.deepseek_costs import DeepSeekCostStatus, DeepSeekPricing, DeepSeekUsage
from hybrid_rag.ids import canonical_json_hash
from hybrid_rag.ingest.tokenizer import TokenCounter
from hybrid_rag.retrieval.bm25 import (
    BM25_SCORER_VERSION,
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    LEXICAL_TOKENIZER_VERSION,
    BM25Config,
    BM25Scorer,
)
from hybrid_rag.retrieval.embedding import EmbeddingProvider, cosine_similarity
from hybrid_rag.retrieval.fusion import (
    rank_ids,
    select_token_budget,
    weighted_average_fusion,
    weighted_fusion,
)
from hybrid_rag.retrieval.models import (
    INDEX_TEXT_SCHEMA_VERSION,
    CandidateHit,
    ContextItem,
    GraphPath,
    IndexBuildReport,
    IndexSemanticConfig,
    RerankComponentTrace,
    RerankTrace,
    RerankTraceHit,
    RetrievalMode,
    RetrievalResult,
    RetrievalTrace,
    RouteTrace,
    ScoreComponent,
)
from hybrid_rag.retrieval.query import (
    DeterministicQueryClient,
    EvidenceItem,
    GroundedAnswer,
    KeywordExtractor,
    QueryClient,
)
from hybrid_rag.retrieval.reranker import (
    FLAG_EMBEDDING_RERANKER_MODEL,
    RerankCandidate,
    Reranker,
    RerankHit,
)
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.retrieval_repository import (
    IndexItem,
    IndexProfile,
    LoadedIndex,
    RetrievalRepository,
    SourceChunk,
    SourceEntity,
    SourceRelation,
    SourceSnapshot,
    StoredIndexProfile,
    make_profile_id,
)
from hybrid_rag.storage.retrieval_repository import (
    RetrievalTrace as StoredTraceInput,
)


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    """Execution-only retrieval settings, persisted into each trace for replay."""

    top_k: int = 8
    candidate_multiplier: int = 4
    context_token_budget: int = 2400
    graph_max_hops: int = 2
    naive_weight: float = 1.0
    local_weight: float = 1.0
    global_weight: float = 1.0
    naive_dense_weight: float = 1.0
    naive_bm25_weight: float = 1.0
    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B
    reranker_provider: str = "none"
    reranker_model: str = FLAG_EMBEDDING_RERANKER_MODEL
    reranker_use_fp16: bool = False
    rerank_candidate_multiplier: int = 4

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        if self.context_token_budget < 1:
            raise ValueError("context_token_budget must be positive")
        if not 1 <= self.graph_max_hops <= 4:
            raise ValueError("graph_max_hops must be between 1 and 4")
        weights = (
            self.naive_weight,
            self.local_weight,
            self.global_weight,
            self.naive_dense_weight,
            self.naive_bm25_weight,
        )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("fusion weights must not be negative")
        if self.naive_weight + self.local_weight + self.global_weight <= 0:
            raise ValueError("at least one fusion weight must be positive")
        if self.naive_dense_weight + self.naive_bm25_weight <= 0:
            raise ValueError("at least one naive subroute weight must be positive")
        BM25Config(k1=self.bm25_k1, b=self.bm25_b)
        if not self.reranker_provider.strip():
            raise ValueError("reranker_provider must not be empty")
        if not self.reranker_model.strip():
            raise ValueError("reranker_model must not be empty")
        if not isinstance(self.reranker_use_fp16, bool):
            raise TypeError("reranker_use_fp16 must be a boolean")
        if self.rerank_candidate_multiplier < 1:
            raise ValueError("rerank_candidate_multiplier must be positive")

    @property
    def config_hash(self) -> str:
        return canonical_json_hash(asdict(self))

    @property
    def weights(self) -> dict[str, float]:
        return {
            RetrievalMode.NAIVE.value: self.naive_weight,
            RetrievalMode.LOCAL.value: self.local_weight,
            RetrievalMode.GLOBAL.value: self.global_weight,
        }

    @property
    def naive_subroute_weights(self) -> dict[str, float]:
        return {
            "dense": self.naive_dense_weight,
            "bm25": self.naive_bm25_weight,
        }

    @property
    def bm25_config(self) -> BM25Config:
        return BM25Config(k1=self.bm25_k1, b=self.bm25_b)

    @property
    def rerank_enabled(self) -> bool:
        return self.reranker_provider != "none"

    @property
    def rerank_candidate_limit(self) -> int:
        return self.top_k * self.rerank_candidate_multiplier


class AnswerResult(BaseModel):
    """A retrieval result plus an answer validated against its citation allowlist."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    retrieval: RetrievalResult
    answer: GroundedAnswer


@dataclass(frozen=True, slots=True)
class _RouteResult:
    route: RetrievalMode
    chunk_scores: dict[str, float]
    chunk_raw_scores: dict[str, float]
    entity_scores: dict[str, float]
    relation_scores: dict[str, float]
    chunk_score_components: dict[str, dict[str, ScoreComponent]]
    paths: tuple[GraphPath, ...]


class RetrievalService:
    """Build and query three independent indexes without delegating retrieval decisions.

    The service deliberately keeps vector comparison, graph expansion, fusion, and
    context selection in project code.  Provider clients only create query terms
    and evidence-bound answers; they never select tools or retrieve data.
    """

    def __init__(
        self,
        database: Database,
        embedding_provider: EmbeddingProvider,
        token_counter: TokenCounter,
        *,
        repository: RetrievalRepository | None = None,
        reranker: Reranker | None = None,
        deepseek_pricing: DeepSeekPricing | None = None,
    ) -> None:
        self.database = database
        self.embedding_provider = embedding_provider
        self.token_counter = token_counter
        self.repository = repository or RetrievalRepository()
        self.reranker = reranker
        self.deepseek_pricing = deepseek_pricing

    def build_index(
        self,
        *,
        build_run_id: str | None = None,
        force: bool = False,
    ) -> IndexBuildReport:
        """Embed chunk/entity/relation texts and atomically activate one profile."""

        semantic = self._index_semantic_config()
        with self.database.session_factory() as session:
            snapshot = self.repository.load_source_snapshot(session, build_run_id=build_run_id)
            metadata: dict[str, Any] = {
                "corpus_content_hash": snapshot.corpus_content_hash,
                "graph_corpus_hash": snapshot.graph_corpus_hash,
                "tokenizer": self.token_counter.name,
            }
            if semantic.provider_options:
                metadata["embedding_options"] = semantic.provider_options
            profile = IndexProfile(
                config_hash=semantic.config_hash,
                provider=semantic.provider,
                model=semantic.model,
                dimensions=semantic.dimensions,
                schema_version=semantic.text_schema_version,
                source_corpus_hash=snapshot.source_corpus_hash,
                source_graph_run_id=snapshot.build_run_id,
                metadata=metadata,
                id=make_profile_id(
                    semantic.config_hash,
                    snapshot.source_corpus_hash,
                    snapshot.build_run_id,
                ),
            )
            existing = self.repository.get_profile(session, profile.id)
            if existing is not None and existing.status == "ready" and not force:
                loaded = self.repository.load_index(session, existing.id)
                expected = len(snapshot.chunks) + len(snapshot.entities) + len(snapshot.relations)
                if len(loaded.items) == expected:
                    refreshed_metadata = dict(existing.metadata)
                    refreshed_metadata.update(profile.metadata)
                    if refreshed_metadata != existing.metadata:
                        existing = self.repository.update_profile_metadata(
                            session,
                            existing.id,
                            refreshed_metadata,
                        )
                        session.commit()
                    return self._build_report(existing, snapshot, reused=True)

        items = self._index_items(snapshot)
        embeddings = self.embedding_provider.embed(tuple(item.embedding_text for item in items))
        if len(embeddings) != len(items):
            raise RuntimeError("embedding provider returned a different number of vectors")
        embedded_items = tuple(
            IndexItem(
                object_id=item.object_id,
                kind=item.kind,
                embedding_text=item.embedding_text,
                embedding=embedding,
                source_chunk_ids=item.source_chunk_ids,
                build_run_id=item.build_run_id,
                source_content_hash=item.source_content_hash,
                metadata=item.metadata,
            )
            for item, embedding in zip(items, embeddings, strict=True)
        )
        with self.database.session_factory.begin() as session:
            stored = self.repository.replace_index(session, profile, embedded_items)
        return self._build_report(stored, snapshot, reused=False)

    def resolve_profile(self, profile_ref: str | None = None) -> StoredIndexProfile:
        """Resolve one ready index profile before a multi-query operation.

        Callers that execute several retrievals (such as a benchmark) can pin
        the returned ID rather than repeatedly resolving whichever profile is
        active at each individual query.
        """

        with self.database.session_factory() as session:
            profile = self.repository.get_profile(session, profile_ref)
        if profile is None:
            qualifier = profile_ref or "an active profile"
            raise ValueError(f"no ready embedding index for {qualifier}")
        return profile

    def retrieve(
        self,
        question: str,
        *,
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
        options: RetrievalOptions | None = None,
        profile_ref: str | None = None,
        keywords: Sequence[str] = (),
        persist: bool = True,
        model_info: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        """Run one of naive/local/global/hybrid retrieval and optionally save its trace."""

        normalized_question = _question(question)
        selected_mode = RetrievalMode(mode)
        effective = options or RetrievalOptions()
        normalized_keywords = _keywords(keywords)
        expanded_query = _expanded_query(normalized_question, normalized_keywords)
        with self.database.session_factory() as session:
            index = self.repository.load_index(session, profile_ref)
        self._validate_query_provider(index.profile)
        query_vector = self.embedding_provider.embed((expanded_query,))[0]
        result = self._execute(
            index,
            query=normalized_question,
            expanded_query=expanded_query,
            keywords=normalized_keywords,
            query_vector=query_vector,
            mode=selected_mode,
            options=effective,
        )
        if not persist:
            return result
        return self._persist_result(result, model_info=model_info)

    async def retrieve_with_keywords(
        self,
        question: str,
        *,
        keyword_extractor: KeywordExtractor | None = None,
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
        options: RetrievalOptions | None = None,
        profile_ref: str | None = None,
        persist: bool = True,
        model_info: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        extractor = keyword_extractor or DeterministicQueryClient()
        extracted = await extractor.extract_keywords(question)
        details = dict(model_info or {})
        details.setdefault("keyword_extractor", type(extractor).__name__)
        result = self.retrieve(
            question,
            mode=mode,
            options=options,
            profile_ref=profile_ref,
            keywords=extracted.keywords,
            persist=False,
        )
        result = self._with_query_cost(result, extractor)
        if not persist:
            return result
        return self._persist_result(result, model_info=details)

    async def ask(
        self,
        question: str,
        *,
        query_client: QueryClient | None = None,
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
        options: RetrievalOptions | None = None,
        profile_ref: str | None = None,
    ) -> AnswerResult:
        """Retrieve first, then allow a constrained client to answer from those chunks only."""

        client = query_client or DeterministicQueryClient()
        retrieval = await self.retrieve_with_keywords(
            question,
            keyword_extractor=client,
            mode=mode,
            options=options,
            profile_ref=profile_ref,
            persist=False,
            model_info={"query_client": type(client).__name__},
        )
        evidence = tuple(
            EvidenceItem(
                citation_id=item.citation_id,
                text=item.text,
                source_chunk_ids=(item.chunk_id,),
            )
            for item in retrieval.context_items
        )
        answer = await client.answer(question, evidence)
        retrieval = self._with_query_cost(retrieval, client)
        persisted = self._persist_result(
            retrieval,
            answer=answer,
            model_info={"query_client": type(client).__name__},
        )
        return AnswerResult(retrieval=persisted, answer=answer)

    def _with_query_cost(
        self,
        result: RetrievalResult,
        client: KeywordExtractor | QueryClient,
    ) -> RetrievalResult:
        """Attach only observed DeepSeek response usage to a replayable trace."""

        if self.deepseek_pricing is None:
            return result
        records = getattr(client, "usage", None)
        if not isinstance(records, tuple) or not all(
            isinstance(record, DeepSeekUsage) for record in records
        ):
            return result
        cost = self.deepseek_pricing.estimate(records)
        if cost.status is DeepSeekCostStatus.NOT_APPLICABLE:
            return result
        trace = result.trace.model_copy(update={"deepseek_cost": cost})
        return result.model_copy(update={"trace": trace})

    def replay(self, trace_id: str) -> RetrievalResult:
        """Load the exact stored retrieval result without re-embedding or re-ranking."""

        with self.database.session_factory() as session:
            stored = self.repository.load_trace(session, trace_id)
        if stored is None:
            raise ValueError(f"retrieval trace not found: {trace_id}")
        payload = stored.output_json or {}
        retrieval_payload = payload.get("retrieval")
        if not isinstance(retrieval_payload, Mapping):
            raise ValueError(f"retrieval trace {trace_id} has no replayable result payload")
        result = RetrievalResult.model_validate_json(json.dumps(retrieval_payload))
        return result.model_copy(update={"trace_id": stored.id})

    def replay_answer(self, trace_id: str) -> AnswerResult | None:
        with self.database.session_factory() as session:
            stored = self.repository.load_trace(session, trace_id)
        if stored is None:
            raise ValueError(f"retrieval trace not found: {trace_id}")
        payload = stored.output_json or {}
        answer_payload = payload.get("answer")
        retrieval_payload = payload.get("retrieval")
        if not isinstance(answer_payload, Mapping) or not isinstance(retrieval_payload, Mapping):
            return None
        return AnswerResult(
            retrieval=RetrievalResult.model_validate_json(json.dumps(retrieval_payload)).model_copy(
                update={"trace_id": stored.id}
            ),
            answer=GroundedAnswer.model_validate_json(json.dumps(answer_payload)),
        )

    def _index_items(self, snapshot: SourceSnapshot) -> tuple[IndexItem, ...]:
        documents = {document.id: document for document in snapshot.documents}
        entities = {entity.id: entity for entity in snapshot.entities}
        items: list[IndexItem] = []
        for chunk in snapshot.chunks:
            document = documents[chunk.document_id]
            items.append(
                IndexItem(
                    object_id=chunk.id,
                    kind="chunk",
                    embedding_text=_chunk_embedding_text(chunk, document.title),
                    embedding=(),
                    source_chunk_ids=(chunk.id,),
                    source_content_hash=chunk.content_hash,
                    metadata={
                        "document_id": chunk.document_id,
                        "document_title": document.title,
                        "section_path": list(chunk.section_path),
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "token_count": chunk.token_count,
                        "text": chunk.contextualized_text,
                    },
                )
            )
        for entity in snapshot.entities:
            items.append(
                IndexItem(
                    object_id=entity.id,
                    kind="entity",
                    embedding_text=_entity_embedding_text(entity),
                    embedding=(),
                    source_chunk_ids=entity.source_chunk_ids,
                    build_run_id=entity.build_run_id,
                    source_content_hash=_entity_content_hash(entity),
                    metadata={
                        "canonical_name": entity.canonical_name,
                        "normalized_name": entity.normalized_name,
                        "entity_type": entity.entity_type,
                        "description": entity.description,
                        "aliases": list(entity.aliases),
                    },
                )
            )
        for relation in snapshot.relations:
            source = entities.get(relation.source_entity_id)
            target = entities.get(relation.target_entity_id)
            if source is None or target is None:
                raise RuntimeError(
                    f"relation {relation.id} references an entity outside the snapshot"
                )
            items.append(
                IndexItem(
                    object_id=relation.id,
                    kind="relation",
                    embedding_text=_relation_embedding_text(relation, source, target),
                    embedding=(),
                    source_chunk_ids=relation.source_chunk_ids,
                    build_run_id=relation.build_run_id,
                    source_content_hash=_relation_content_hash(relation),
                    metadata={
                        "source_entity_id": relation.source_entity_id,
                        "target_entity_id": relation.target_entity_id,
                        "source_entity_name": source.canonical_name,
                        "target_entity_name": target.canonical_name,
                        "predicate": relation.predicate,
                        "description": relation.description,
                    },
                )
            )
        return tuple(items)

    def _build_report(
        self,
        profile: StoredIndexProfile,
        snapshot: SourceSnapshot,
        *,
        reused: bool,
    ) -> IndexBuildReport:
        return IndexBuildReport(
            profile_id=profile.id,
            config_hash=profile.config_hash,
            source_corpus_hash=snapshot.source_corpus_hash,
            graph_build_run_id=snapshot.build_run_id,
            provider=profile.provider,
            model=profile.model,
            dimensions=profile.dimensions,
            chunks=len(snapshot.chunks),
            entities=len(snapshot.entities),
            relations=len(snapshot.relations),
            reused=reused,
        )

    def _index_semantic_config(self) -> IndexSemanticConfig:
        return IndexSemanticConfig(
            provider=self.embedding_provider.provider,
            model=self.embedding_provider.model,
            dimensions=self.embedding_provider.dimensions,
            text_schema_version=INDEX_TEXT_SCHEMA_VERSION,
            provider_options=_embedding_semantic_options(self.embedding_provider),
        )

    def _validate_query_provider(self, profile: StoredIndexProfile) -> None:
        if self.embedding_provider.dimensions != profile.dimensions:
            raise ValueError(
                "embedding provider dimensions differ from the selected index "
                f"({self.embedding_provider.dimensions} != {profile.dimensions})"
            )
        if self.embedding_provider.provider != profile.provider:
            raise ValueError(
                "embedding provider differs from the selected index "
                f"({self.embedding_provider.provider!r} != {profile.provider!r})"
            )
        if self.embedding_provider.model != profile.model:
            raise ValueError(
                "embedding model differs from the selected index "
                f"({self.embedding_provider.model!r} != {profile.model!r})"
            )
        if self._index_semantic_config().config_hash != profile.config_hash:
            raise ValueError(
                "embedding encoding options differ from the selected index; rebuild the index "
                "with the current embedding configuration"
            )

    def _execute(
        self,
        index: LoadedIndex,
        *,
        query: str,
        expanded_query: str,
        keywords: tuple[str, ...],
        query_vector: Sequence[float],
        mode: RetrievalMode,
        options: RetrievalOptions,
    ) -> RetrievalResult:
        candidate_limit = options.top_k * options.candidate_multiplier
        if mode is RetrievalMode.HYBRID:
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="hybrid-recall") as executor:
                futures = {
                    RetrievalMode.NAIVE: executor.submit(
                        self._naive_route,
                        index,
                        query_vector,
                        expanded_query,
                        candidate_limit,
                        options,
                    ),
                    RetrievalMode.LOCAL: executor.submit(
                        self._local_route, index, query_vector, candidate_limit
                    ),
                    RetrievalMode.GLOBAL: executor.submit(
                        self._global_route, index, query_vector, candidate_limit
                    ),
                }
                routes = {route: future.result() for route, future in futures.items()}
            fused_scores, components = weighted_fusion(
                {route.value: result.chunk_scores for route, result in routes.items()},
                options.weights,
            )
            raw_scores = _max_scores(result.chunk_raw_scores for result in routes.values())
            paths = self._expand_hybrid_paths(index, routes, options.graph_max_hops)
            route_traces = {
                route.value: self._route_trace(result, index) for route, result in routes.items()
            }
        else:
            if mode is RetrievalMode.NAIVE:
                route = self._naive_route(
                    index,
                    query_vector,
                    expanded_query,
                    candidate_limit,
                    options,
                )
            else:
                route = {
                    RetrievalMode.LOCAL: self._local_route,
                    RetrievalMode.GLOBAL: self._global_route,
                }[mode](index, query_vector, candidate_limit)
            routes = {mode: route}
            fused_scores = dict(route.chunk_scores)
            components = {chunk_id: {mode.value: score} for chunk_id, score in fused_scores.items()}
            raw_scores = dict(route.chunk_raw_scores)
            paths = route.paths
            route_traces = {mode.value: self._route_trace(route, index)}

        chunks = {item.object_id: item for item in index.chunks}
        rerank_trace: RerankTrace | None = None
        if options.rerank_enabled:
            if self.reranker is None:
                raise ValueError(
                    "reranking was requested but no reranker is configured; "
                    "set reranker_provider=none or provide a FlagEmbedding reranker"
                )
            rerank_candidate_ids = rank_ids(
                fused_scores,
                limit=options.rerank_candidate_limit,
            )
            ordered_reranked_ids, final_scores, rerank_trace = self._rerank(
                query=query,
                candidate_ids=rerank_candidate_ids,
                chunks=chunks,
                fused_scores=fused_scores,
                options=options,
            )
            ordered_chunk_ids = ordered_reranked_ids[: options.top_k]
        else:
            final_scores = dict(fused_scores)
            ordered_chunk_ids = rank_ids(final_scores, limit=options.top_k)
        selected_context = self._context_items(
            ordered_chunk_ids,
            chunks,
            final_scores,
            components,
            routes,
            budget=options.context_token_budget,
        )
        context = "\n\n".join(_render_context(item) for item in selected_context)
        context_tokens = sum(item.token_count for item in selected_context)
        hit_values = tuple(
            CandidateHit(
                object_id=chunk_id,
                kind="chunk",
                score=float(final_scores[chunk_id]),
                raw_score=float(raw_scores.get(chunk_id, fused_scores[chunk_id])),
                retrieval_score=float(fused_scores[chunk_id]),
                rerank_score=float(final_scores[chunk_id]) if rerank_trace is not None else None,
                rank=rank,
                route_scores={
                    route: float(score)
                    for route, score in sorted(components.get(chunk_id, {}).items())
                },
                score_components=(
                    routes[RetrievalMode.NAIVE].chunk_score_components.get(chunk_id, {})
                    if mode is RetrievalMode.NAIVE
                    else {}
                ),
                source_chunk_ids=(chunk_id,),
                metadata=dict(chunks[chunk_id].metadata),
            )
            for rank, chunk_id in enumerate(ordered_chunk_ids, start=1)
        )
        trace = RetrievalTrace(
            profile_id=index.profile.id,
            index_config_hash=index.profile.config_hash,
            query=query,
            expanded_query=expanded_query,
            mode=mode,
            keywords=keywords,
            routes=route_traces,
            rerank=rerank_trace,
            fused_hits=hit_values,
            graph_paths=paths,
            context_items=selected_context,
            context_token_budget=options.context_token_budget,
            context_tokens=context_tokens,
            settings={
                **asdict(options),
                "embedding_provider": index.profile.provider,
                "embedding_model": index.profile.model,
                "embedding_dimensions": index.profile.dimensions,
                "tokenizer": self.token_counter.name,
                "naive_lexical_scorer": BM25_SCORER_VERSION,
                "naive_lexical_tokenizer": LEXICAL_TOKENIZER_VERSION,
                "rerank_enabled": options.rerank_enabled,
                "reranker_provider": options.reranker_provider,
                "reranker_model": options.reranker_model,
                "reranker_use_fp16": options.reranker_use_fp16,
                "reranker_version": (
                    self.reranker.version
                    if options.rerank_enabled and self.reranker is not None
                    else "none"
                ),
            },
        )
        return RetrievalResult(
            profile_id=index.profile.id,
            mode=mode,
            query=query,
            keywords=keywords,
            hits=hit_values,
            graph_paths=paths,
            context_items=selected_context,
            context=context,
            context_tokens=context_tokens,
            trace=trace,
        )

    def _rerank(
        self,
        *,
        query: str,
        candidate_ids: Sequence[str],
        chunks: Mapping[str, IndexItem],
        fused_scores: Mapping[str, float],
        options: RetrievalOptions,
    ) -> tuple[tuple[str, ...], dict[str, float], RerankTrace]:
        """Rerank first-stage candidates without changing their recall provenance."""

        reranker = self.reranker
        if reranker is None:
            raise ValueError("reranking was requested but no reranker is configured")
        if options.reranker_provider != reranker.provider:
            raise ValueError(
                "reranker provider differs from the requested retrieval configuration "
                f"({reranker.provider!r} != {options.reranker_provider!r})"
            )
        if options.reranker_model != reranker.model:
            raise ValueError(
                "reranker model differs from the requested retrieval configuration "
                f"({reranker.model!r} != {options.reranker_model!r})"
            )
        candidates = tuple(
            RerankCandidate(
                object_id=chunk_id,
                text=chunks[chunk_id].embedding_text,
                prior_score=float(fused_scores[chunk_id]),
            )
            for chunk_id in candidate_ids
        )
        returned = reranker.rerank(query, candidates)
        if any(not isinstance(hit, RerankHit) for hit in returned):
            raise TypeError("reranker must return RerankHit values")
        expected_ids = {candidate.object_id for candidate in candidates}
        returned_ids = {hit.candidate.object_id for hit in returned}
        if len(returned) != len(candidates) or returned_ids != expected_ids:
            raise RuntimeError("reranker must return one score for every supplied candidate")

        original_candidates = {candidate.object_id: candidate for candidate in candidates}
        pre_rerank_ranks = {chunk_id: rank for rank, chunk_id in enumerate(candidate_ids, start=1)}
        ordered = tuple(
            sorted(
                returned,
                key=lambda hit: (
                    -hit.score,
                    -original_candidates[hit.candidate.object_id].prior_score,
                    hit.candidate.object_id,
                ),
            )
        )
        trace_hits = tuple(
            RerankTraceHit(
                object_id=hit.candidate.object_id,
                pre_rerank_rank=pre_rerank_ranks[hit.candidate.object_id],
                pre_rerank_score=original_candidates[hit.candidate.object_id].prior_score,
                score=hit.score,
                final_rank=rank,
                components={
                    name: RerankComponentTrace(
                        raw_score=component.raw_score,
                        normalized_score=component.normalized_score,
                        weight=component.weight,
                        weighted_score=component.weighted_score,
                    )
                    for name, component in sorted(hit.components.items())
                },
            )
            for rank, hit in enumerate(ordered, start=1)
        )
        return (
            tuple(hit.candidate.object_id for hit in ordered),
            {hit.candidate.object_id: float(hit.score) for hit in ordered},
            RerankTrace(
                provider=reranker.provider,
                model=reranker.model,
                version=reranker.version,
                candidate_limit=options.rerank_candidate_limit,
                hits=trace_hits,
            ),
        )

    def _naive_route(
        self,
        index: LoadedIndex,
        query_vector: Sequence[float],
        lexical_query: str,
        limit: int,
        options: RetrievalOptions,
    ) -> _RouteResult:
        dense_scores = (
            {item.object_id: score for item, score in _score(index.chunks, query_vector, limit)}
            if options.naive_dense_weight > 0.0
            else {}
        )
        bm25_scores = (
            {
                hit.item.object_id: hit.score
                for hit in BM25Scorer(index.chunks, config=options.bm25_config).score(
                    lexical_query,
                    limit=limit,
                )
            }
            if options.naive_bm25_weight > 0.0
            else {}
        )
        scores, score_components = weighted_average_fusion(
            {"dense": dense_scores, "bm25": bm25_scores},
            options.naive_subroute_weights,
        )
        return _RouteResult(
            route=RetrievalMode.NAIVE,
            chunk_scores=scores,
            chunk_raw_scores=dict(scores),
            entity_scores={},
            relation_scores={},
            chunk_score_components=score_components,
            paths=(),
        )

    def _local_route(
        self,
        index: LoadedIndex,
        query_vector: Sequence[float],
        limit: int,
    ) -> _RouteResult:
        entity_scored = _score(index.entities, query_vector, limit)
        relations_by_entity: dict[str, list[IndexItem]] = defaultdict(list)
        for relation in index.relations:
            source = str(relation.metadata["source_entity_id"])
            target = str(relation.metadata["target_entity_id"])
            relations_by_entity[source].append(relation)
            relations_by_entity[target].append(relation)
        chunk_scores: dict[str, float] = {}
        relation_scores: dict[str, float] = {}
        paths: list[GraphPath] = []
        for entity, score in entity_scored:
            _accumulate(chunk_scores, entity.source_chunk_ids, score)
            for relation in relations_by_entity.get(entity.object_id, []):
                expanded_score = score * 0.9
                relation_scores[relation.object_id] = max(
                    relation_scores.get(relation.object_id, float("-inf")), expanded_score
                )
                _accumulate(chunk_scores, relation.source_chunk_ids, expanded_score)
                paths.append(_direct_path(relation, expanded_score))
        entity_scores = {item.object_id: score for item, score in entity_scored}
        return _RouteResult(
            route=RetrievalMode.LOCAL,
            chunk_scores=chunk_scores,
            chunk_raw_scores=dict(chunk_scores),
            entity_scores=entity_scores,
            relation_scores=relation_scores,
            chunk_score_components={},
            paths=_unique_paths(paths),
        )

    def _global_route(
        self,
        index: LoadedIndex,
        query_vector: Sequence[float],
        limit: int,
    ) -> _RouteResult:
        relation_scored = _score(index.relations, query_vector, limit)
        chunk_scores: dict[str, float] = {}
        entity_scores: dict[str, float] = {}
        paths: list[GraphPath] = []
        for relation, score in relation_scored:
            _accumulate(chunk_scores, relation.source_chunk_ids, score)
            source = str(relation.metadata["source_entity_id"])
            target = str(relation.metadata["target_entity_id"])
            entity_scores[source] = max(entity_scores.get(source, float("-inf")), score)
            entity_scores[target] = max(entity_scores.get(target, float("-inf")), score)
            paths.append(_direct_path(relation, score))
        relation_scores = {item.object_id: score for item, score in relation_scored}
        return _RouteResult(
            route=RetrievalMode.GLOBAL,
            chunk_scores=chunk_scores,
            chunk_raw_scores=dict(chunk_scores),
            entity_scores=entity_scores,
            relation_scores=relation_scores,
            chunk_score_components={},
            paths=_unique_paths(paths),
        )

    def _route_trace(self, route: _RouteResult, index: LoadedIndex) -> RouteTrace:
        all_items = {item.object_id: item for item in index.items}
        raw_values: list[tuple[str, str, float]] = []
        raw_values.extend(
            (item_id, "chunk", score) for item_id, score in route.chunk_raw_scores.items()
        )
        raw_values.extend(
            (item_id, "entity", score) for item_id, score in route.entity_scores.items()
        )
        raw_values.extend(
            (item_id, "relation", score) for item_id, score in route.relation_scores.items()
        )
        raw_values.sort(key=lambda value: (-value[2], value[1], value[0]))
        return RouteTrace(
            route=route.route,
            candidate_count=len(raw_values),
            hits=tuple(
                CandidateHit(
                    object_id=object_id,
                    kind=kind,  # type: ignore[arg-type]
                    score=float(score),
                    raw_score=float(score),
                    rank=rank,
                    route_scores={route.route.value: float(score)},
                    score_components=route.chunk_score_components.get(object_id, {}),
                    source_chunk_ids=all_items[object_id].source_chunk_ids,
                    metadata=dict(all_items[object_id].metadata),
                )
                for rank, (object_id, kind, score) in enumerate(raw_values, start=1)
            ),
        )

    def _expand_hybrid_paths(
        self,
        index: LoadedIndex,
        routes: Mapping[RetrievalMode, _RouteResult],
        max_hops: int,
    ) -> tuple[GraphPath, ...]:
        paths = [path for route in routes.values() for path in route.paths]
        graph = nx.Graph()
        relations_by_pair: dict[frozenset[str], list[IndexItem]] = defaultdict(list)
        for relation in index.relations:
            source = str(relation.metadata["source_entity_id"])
            target = str(relation.metadata["target_entity_id"])
            graph.add_edge(source, target)
            relations_by_pair[frozenset((source, target))].append(relation)
        for values in relations_by_pair.values():
            values.sort(key=lambda item: item.object_id)
        entity_scores = _max_scores(route.entity_scores for route in routes.values())
        selected = rank_ids(entity_scores, limit=6)
        for source, target in combinations(selected, 2):
            try:
                node_ids = nx.shortest_path(graph, source, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if len(node_ids) - 1 > max_hops:
                continue
            relation_ids: list[str] = []
            source_chunks: list[str] = []
            path_score = min(entity_scores[source], entity_scores[target])
            for left, right in pairwise(node_ids):
                candidates = relations_by_pair[frozenset((left, right))]
                relation = candidates[0]
                relation_ids.append(relation.object_id)
                source_chunks.extend(relation.source_chunk_ids)
            paths.append(
                GraphPath(
                    node_ids=tuple(node_ids),
                    relation_ids=tuple(relation_ids),
                    source_chunk_ids=tuple(sorted(set(source_chunks))),
                    score=float(path_score),
                )
            )
        return _unique_paths(paths)

    def _context_items(
        self,
        ordered_chunk_ids: Sequence[str],
        chunks: Mapping[str, IndexItem],
        scores: Mapping[str, float],
        components: Mapping[str, Mapping[str, float]],
        routes: Mapping[RetrievalMode, _RouteResult],
        *,
        budget: int,
    ) -> tuple[ContextItem, ...]:
        token_counts = {
            chunk_id: self.token_counter.count(_render_chunk_candidate(chunk_id, chunks[chunk_id]))
            for chunk_id in ordered_chunk_ids
        }
        selected_ids = list(select_token_budget(ordered_chunk_ids, token_counts, budget=budget))
        consumed = sum(token_counts[item] for item in selected_ids)
        if not selected_ids and ordered_chunk_ids:
            first = ordered_chunk_ids[0]
            text = str(chunks[first].metadata["text"])
            header = _render_chunk_candidate(first, _with_chunk_text(chunks[first], ""))
            remaining = budget - self.token_counter.count(header)
            clipped = _clip_text(text, remaining, self.token_counter) if remaining > 0 else ""
            if clipped:
                chunks = dict(chunks)
                chunks[first] = _with_chunk_text(chunks[first], clipped)
                selected_ids.append(first)
                consumed = self.token_counter.count(_render_chunk_candidate(first, chunks[first]))
        support_entities: dict[str, set[str]] = defaultdict(set)
        support_relations: dict[str, set[str]] = defaultdict(set)
        for route in routes.values():
            for entity_id in route.entity_scores:
                # The source IDs are recovered from the route's graph paths below.
                for path in route.paths:
                    if entity_id in path.node_ids:
                        support_entities_by_path(support_entities, path, entity_id)
            for relation_id in route.relation_scores:
                for path in route.paths:
                    if relation_id in path.relation_ids:
                        for chunk_id in path.source_chunk_ids:
                            support_relations[chunk_id].add(relation_id)
        values: list[ContextItem] = []
        for chunk_id in selected_ids:
            item = chunks[chunk_id]
            metadata = item.metadata
            rendered_tokens = self.token_counter.count(_render_chunk_candidate(chunk_id, item))
            values.append(
                ContextItem(
                    citation_id=chunk_id,
                    chunk_id=chunk_id,
                    document_id=str(metadata["document_id"]),
                    document_title=str(metadata["document_title"]),
                    section_path=tuple(str(value) for value in metadata.get("section_path", [])),
                    page_start=_optional_int(metadata.get("page_start")),
                    page_end=_optional_int(metadata.get("page_end")),
                    text=str(metadata["text"]),
                    token_count=rendered_tokens,
                    score=float(scores[chunk_id]),
                    route_scores={
                        route: float(score)
                        for route, score in sorted(components.get(chunk_id, {}).items())
                    },
                    source_entity_ids=tuple(sorted(support_entities.get(chunk_id, set()))),
                    source_relation_ids=tuple(sorted(support_relations.get(chunk_id, set()))),
                )
            )
        total = sum(item.token_count for item in values)
        if total > budget or total != consumed:
            raise RuntimeError("context selection did not respect its token budget")
        return tuple(values)

    def _persist_result(
        self,
        result: RetrievalResult,
        *,
        answer: GroundedAnswer | None = None,
        model_info: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        output: dict[str, Any] = {"retrieval": result.model_dump(mode="json")}
        if answer is not None:
            output["answer"] = answer.model_dump(mode="json")
        with self.database.session_factory.begin() as session:
            stored = self.repository.save_trace(
                session,
                StoredTraceInput(
                    profile_id=result.profile_id,
                    index_config_hash=result.trace.index_config_hash,
                    query_text=result.query,
                    mode=result.mode.value,
                    retrieval_config_hash=_options_hash_from_trace(result.trace),
                    trace_json=result.trace.model_dump(mode="json"),
                    graph_build_run_id=None,
                    output_json=output,
                    model_info=dict(model_info or {}),
                ),
            )
        return result.model_copy(update={"trace_id": stored.id})


def _embedding_semantic_options(provider: EmbeddingProvider) -> dict[str, str | int | bool]:
    """Read optional provider settings that can change a persisted embedding."""

    raw = getattr(provider, "semantic_options", {})
    if not isinstance(raw, Mapping):
        raise TypeError("embedding semantic_options must be a mapping")
    options: dict[str, str | int | bool] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise TypeError("embedding semantic option names must be non-empty strings")
        if not isinstance(value, (str, int, bool)):
            raise TypeError(
                "embedding semantic option values must be strings, integers, or booleans"
            )
        options[key] = value
    return options


def _score(
    items: Sequence[IndexItem],
    query_vector: Sequence[float],
    limit: int,
) -> tuple[tuple[IndexItem, float], ...]:
    if limit < 1:
        return ()
    scored = [(item, float(cosine_similarity(query_vector, item.embedding))) for item in items]
    scored.sort(key=lambda value: (-value[1], value[0].object_id))
    return tuple(scored[:limit])


def _accumulate(scores: dict[str, float], object_ids: Sequence[str], score: float) -> None:
    for object_id in object_ids:
        scores[object_id] = max(scores.get(object_id, float("-inf")), score)


def _max_scores(score_maps: Sequence[Mapping[str, float]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for scores in score_maps:
        for object_id, score in scores.items():
            output[object_id] = max(output.get(object_id, float("-inf")), score)
    return output


def _direct_path(relation: IndexItem, score: float) -> GraphPath:
    return GraphPath(
        node_ids=(
            str(relation.metadata["source_entity_id"]),
            str(relation.metadata["target_entity_id"]),
        ),
        relation_ids=(relation.object_id,),
        source_chunk_ids=tuple(sorted(set(relation.source_chunk_ids))),
        score=float(score),
    )


def _unique_paths(paths: Sequence[GraphPath]) -> tuple[GraphPath, ...]:
    chosen: dict[tuple[str, ...], GraphPath] = {}
    for path in paths:
        key = path.relation_ids or path.node_ids
        current = chosen.get(key)
        if current is None or path.score > current.score:
            chosen[key] = path
    return tuple(
        sorted(chosen.values(), key=lambda item: (-item.score, item.relation_ids, item.node_ids))
    )


def _chunk_embedding_text(chunk: SourceChunk, title: str) -> str:
    section = " > ".join(chunk.section_path) if chunk.section_path else "(untitled section)"
    return f"Document: {title}\nSection: {section}\n\n{chunk.contextualized_text}"


def _entity_embedding_text(entity: SourceEntity) -> str:
    aliases = ", ".join(entity.aliases) if entity.aliases else "(none)"
    return (
        f"Entity: {entity.canonical_name}\nType: {entity.entity_type}\n"
        f"Aliases: {aliases}\nDescription: {entity.description}"
    )


def _relation_embedding_text(
    relation: SourceRelation,
    source: SourceEntity,
    target: SourceEntity,
) -> str:
    return (
        f"Relation: {source.canonical_name} --{relation.predicate}--> {target.canonical_name}\n"
        f"Description: {relation.description}"
    )


def _entity_content_hash(entity: SourceEntity) -> str:
    return canonical_json_hash(
        {
            "id": entity.id,
            "name": entity.canonical_name,
            "type": entity.entity_type,
            "description": entity.description,
            "aliases": list(entity.aliases),
            "sources": list(entity.source_chunk_ids),
        }
    )


def _relation_content_hash(relation: SourceRelation) -> str:
    return canonical_json_hash(
        {
            "id": relation.id,
            "source": relation.source_entity_id,
            "target": relation.target_entity_id,
            "predicate": relation.predicate,
            "description": relation.description,
            "sources": list(relation.source_chunk_ids),
        }
    )


def _question(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("question must not be blank")
    return normalized


def _keywords(values: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        folded = normalized.casefold()
        if not normalized or folded in seen:
            continue
        output.append(normalized)
        seen.add(folded)
    return tuple(output[:12])


def _expanded_query(question: str, keywords: Sequence[str]) -> str:
    return question if not keywords else f"{question}\nKeywords: {', '.join(keywords)}"


def _render_chunk_candidate(chunk_id: str, item: IndexItem) -> str:
    metadata = item.metadata
    section = " > ".join(str(value) for value in metadata.get("section_path", [])) or "(untitled)"
    pages = _page_label(metadata.get("page_start"), metadata.get("page_end"))
    return (
        f"[citation={chunk_id}; document={metadata['document_title']}; section={section}; "
        f"pages={pages}]\n{metadata['text']}"
    )


def _with_chunk_text(item: IndexItem, text: str) -> IndexItem:
    return IndexItem(
        id=item.id,
        object_id=item.object_id,
        kind=item.kind,
        embedding_text=item.embedding_text,
        embedding=item.embedding,
        source_chunk_ids=item.source_chunk_ids,
        build_run_id=item.build_run_id,
        source_content_hash=item.source_content_hash,
        metadata={**item.metadata, "text": text},
    )


def _render_context(item: ContextItem) -> str:
    section = " > ".join(item.section_path) or "(untitled)"
    pages = _page_label(item.page_start, item.page_end)
    return (
        f"[citation={item.citation_id}; document={item.document_title}; section={section}; "
        f"pages={pages}]\n{item.text}"
    )


def _page_label(start: object, end: object) -> str:
    if isinstance(start, int) and isinstance(end, int):
        return str(start) if start == end else f"{start}-{end}"
    if isinstance(start, int):
        return str(start)
    return "n/a"


def _clip_text(text: str, budget: int, counter: TokenCounter) -> str:
    if counter.count(text) <= budget:
        return text
    suffix = "…"
    lower, upper = 0, len(text)
    best = ""
    while lower <= upper:
        middle = (lower + upper) // 2
        candidate = text[:middle].rstrip() + suffix
        if counter.count(candidate) <= budget:
            best = candidate
            lower = middle + 1
        else:
            upper = middle - 1
    return best


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def support_entities_by_path(
    output: dict[str, set[str]],
    path: GraphPath,
    entity_id: str,
) -> None:
    for chunk_id in path.source_chunk_ids:
        output[chunk_id].add(entity_id)


def _options_hash_from_trace(trace: RetrievalTrace) -> str:
    values = {
        key: value
        for key, value in trace.settings.items()
        if key
        in {
            "top_k",
            "candidate_multiplier",
            "context_token_budget",
            "graph_max_hops",
            "naive_weight",
            "local_weight",
            "global_weight",
            "naive_dense_weight",
            "naive_bm25_weight",
            "bm25_k1",
            "bm25_b",
            "naive_lexical_scorer",
            "naive_lexical_tokenizer",
            "rerank_enabled",
            "reranker_provider",
            "reranker_model",
            "reranker_use_fp16",
            "reranker_version",
            "rerank_candidate_multiplier",
        }
    }
    return canonical_json_hash(values)
