"""Persistence boundary for phase-three indexes and retrieval traces.

The database deliberately stores vectors as JSON.  Vector similarity, graph
traversal, score fusion, and context selection remain domain code in the
retrieval package; this adapter only provides transactional, reproducible
snapshots for those algorithms.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from hybrid_rag.ids import canonical_json_hash, sha256_text, stable_id
from hybrid_rag.storage.models import (
    ChunkRecord,
    DocumentRecord,
    EmbeddingProfileRecord,
    EmbeddingVectorRecord,
    EntityEvidenceRecord,
    EntityRecord,
    GraphBuildRunRecord,
    RelationEvidenceRecord,
    RelationRecord,
    RetrievalTraceRecord,
)

INDEX_KINDS = frozenset({"chunk", "entity", "relation"})
RETRIEVAL_MODES = frozenset({"naive", "local", "global", "hybrid"})


class RetrievalRepositoryError(RuntimeError):
    """Raised when persisted retrieval data violates its domain contract."""


@dataclass(frozen=True, slots=True)
class IndexProfile:
    """Input contract for a reproducible embedding index.

    ``id`` is optional.  When omitted, :func:`make_profile_id` creates a stable
    ID from the semantic index config, source corpus hash, and source graph
    snapshot run. A no-graph index retains the legacy two-part identity so
    profiles created before graph extraction remain addressable after upgrade.
    """

    config_hash: str
    provider: str
    model: str
    dimensions: int
    schema_version: str
    source_corpus_hash: str
    source_graph_run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True, slots=True)
class StoredIndexProfile:
    """An index profile read from the database, with a guaranteed ID."""

    id: str
    config_hash: str
    provider: str
    model: str
    dimensions: int
    schema_version: str
    source_corpus_hash: str
    source_graph_run_id: str | None
    metadata: dict[str, Any]
    status: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    error: str | None


@dataclass(frozen=True, slots=True)
class IndexItem:
    """A vector row independent from any vector-store implementation."""

    object_id: str
    kind: str
    embedding_text: str
    embedding: tuple[float, ...]
    source_chunk_ids: tuple[str, ...] = ()
    build_run_id: str | None = None
    source_content_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str | None = None


# Kept as a clear structural name for retriever implementations.  It is an
# alias, not a second storage abstraction: both inputs and loaded vectors have
# the same stable attributes.
VectorRecord = IndexItem


@dataclass(frozen=True, slots=True)
class LoadedIndex:
    profile: StoredIndexProfile
    chunks: tuple[VectorRecord, ...]
    entities: tuple[VectorRecord, ...]
    relations: tuple[VectorRecord, ...]

    @property
    def items(self) -> tuple[VectorRecord, ...]:
        return self.chunks + self.entities + self.relations

    @property
    def by_kind(self) -> dict[str, tuple[VectorRecord, ...]]:
        return {
            "chunk": self.chunks,
            "entity": self.entities,
            "relation": self.relations,
        }


@dataclass(frozen=True, slots=True)
class SourceDocument:
    id: str
    title: str
    source_uri: str
    source_type: str
    content_hash: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SourceChunk:
    id: str
    document_id: str
    ordinal: int
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    text: str
    contextualized_text: str
    token_count: int
    content_hash: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SourceEntity:
    id: str
    build_run_id: str
    canonical_name: str
    normalized_name: str
    entity_type: str
    description: str
    aliases: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceRelation:
    id: str
    build_run_id: str
    source_entity_id: str
    target_entity_id: str
    predicate: str
    description: str
    source_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    id: str
    owner_kind: str
    owner_id: str
    chunk_id: str
    extraction_id: str
    mention_id: str | None
    quote: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Stable database source rows used to derive all three index texts."""

    source_corpus_hash: str
    corpus_content_hash: str
    graph_corpus_hash: str | None
    build_run_id: str | None
    documents: tuple[SourceDocument, ...]
    chunks: tuple[SourceChunk, ...]
    entities: tuple[SourceEntity, ...]
    relations: tuple[SourceRelation, ...]
    entity_evidence: tuple[SourceEvidence, ...]
    relation_evidence: tuple[SourceEvidence, ...]


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Input contract for an explainable retrieval trace and optional answer."""

    profile_id: str
    index_config_hash: str
    query_text: str
    mode: str
    retrieval_config_hash: str
    trace_json: Mapping[str, Any]
    graph_build_run_id: str | None = None
    output_json: Mapping[str, Any] | None = None
    model_info: Mapping[str, Any] = field(default_factory=dict)
    query_hash: str | None = None
    id: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class StoredRetrievalTrace:
    id: str
    profile_id: str
    index_config_hash: str
    query_text: str
    query_hash: str
    mode: str
    retrieval_config_hash: str
    trace_json: dict[str, Any]
    graph_build_run_id: str | None
    output_json: dict[str, Any] | None
    model_info: dict[str, Any]
    created_at: datetime


def make_profile_id(
    config_hash: str,
    source_corpus_hash: str,
    source_graph_run_id: str | None = None,
) -> str:
    """Return the deterministic profile ID for one configuration/snapshot.

    The source graph run is part of the identity whenever a graph was used to
    derive entity and relation vectors. Keeping the no-graph form unchanged
    lets pre-``0004`` chunk-only profiles remain stable across the migration.
    """

    if source_graph_run_id is None:
        return stable_id("idx", config_hash, source_corpus_hash)
    return stable_id("idx", config_hash, source_corpus_hash, source_graph_run_id)


class RetrievalRepository:
    """Transaction-friendly storage operations for phase-three retrieval."""

    def load_source_snapshot(
        self,
        session: Session,
        *,
        build_run_id: str | None = None,
    ) -> SourceSnapshot:
        """Load immutable source rows for chunk/entity/relation embedding texts.

        With no explicit run, the currently persisted graph snapshot is used.
        Chunks are always loaded from the current corpus so naive retrieval can
        work before a graph exists; entity/relation rows are empty in that case.
        """

        run = self._snapshot_run(session, build_run_id)
        document_records = list(session.scalars(select(DocumentRecord).order_by(DocumentRecord.id)))
        chunk_records = list(
            session.scalars(
                select(ChunkRecord).order_by(
                    ChunkRecord.document_id,
                    ChunkRecord.ordinal,
                    ChunkRecord.id,
                )
            )
        )
        if run is None:
            entity_records: list[EntityRecord] = []
            relation_records: list[RelationRecord] = []
        else:
            entity_records = list(
                session.scalars(
                    select(EntityRecord)
                    .where(EntityRecord.build_run_id == run.id)
                    .order_by(EntityRecord.id)
                )
            )
            relation_records = list(
                session.scalars(
                    select(RelationRecord)
                    .where(RelationRecord.build_run_id == run.id)
                    .order_by(RelationRecord.id)
                )
            )

        entity_ids = [record.id for record in entity_records]
        relation_ids = [record.id for record in relation_records]
        entity_evidence = self._entity_evidence(session, entity_ids)
        relation_evidence = self._relation_evidence(session, relation_ids)
        document_parts = [(record.id, record.content_hash) for record in document_records]
        chunk_parts = [(record.id, record.content_hash) for record in chunk_records]
        corpus_content_hash = canonical_json_hash(
            {
                "documents": document_parts,
                "chunks": chunk_parts,
            }
        )
        source_corpus_hash = canonical_json_hash(
            {
                "chunks": chunk_parts,
                "graph_build_run_id": run.id if run is not None else None,
                "graph_corpus_hash": run.corpus_hash if run is not None else None,
            }
        )
        return SourceSnapshot(
            source_corpus_hash=source_corpus_hash,
            corpus_content_hash=corpus_content_hash,
            graph_corpus_hash=run.corpus_hash if run is not None else None,
            build_run_id=run.id if run is not None else None,
            documents=tuple(
                SourceDocument(
                    id=record.id,
                    title=record.title,
                    source_uri=record.source_uri,
                    source_type=record.source_type,
                    content_hash=record.content_hash,
                    metadata=dict(record.metadata_json),
                )
                for record in document_records
            ),
            chunks=tuple(
                SourceChunk(
                    id=record.id,
                    document_id=record.document_id,
                    ordinal=record.ordinal,
                    section_path=tuple(record.section_path_json),
                    page_start=record.page_start,
                    page_end=record.page_end,
                    text=record.text,
                    contextualized_text=record.contextualized_text,
                    token_count=record.token_count,
                    content_hash=record.content_hash,
                    metadata=dict(record.metadata_json),
                )
                for record in chunk_records
            ),
            entities=tuple(
                SourceEntity(
                    id=record.id,
                    build_run_id=record.build_run_id,
                    canonical_name=record.canonical_name,
                    normalized_name=record.normalized_name,
                    entity_type=record.entity_type,
                    description=record.description,
                    aliases=tuple(record.aliases_json),
                    source_chunk_ids=tuple(record.source_chunk_ids_json),
                )
                for record in entity_records
            ),
            relations=tuple(
                SourceRelation(
                    id=record.id,
                    build_run_id=record.build_run_id,
                    source_entity_id=record.source_entity_id,
                    target_entity_id=record.target_entity_id,
                    predicate=record.predicate,
                    description=record.description,
                    source_chunk_ids=tuple(record.source_chunk_ids_json),
                )
                for record in relation_records
            ),
            entity_evidence=entity_evidence,
            relation_evidence=relation_evidence,
        )

    def invalidate_indexes_for_document(
        self,
        session: Session,
        document_id: str,
        *,
        reason: str = "source document changed after retrieval indexing",
    ) -> int:
        """Deactivate profiles that contain source vectors for an about-to-change document.

        Vector rows intentionally have no polymorphic foreign key to chunks, so
        their stable source IDs are inspected before the ingestion transaction
        deletes/replaces the old chunk rows.  Profiles and traces remain stored
        for audit/replay, but ``load_index`` will reject the failed profile until
        a deterministic rebuild activates a new one.
        """

        chunk_ids = set(
            session.scalars(select(ChunkRecord.id).where(ChunkRecord.document_id == document_id))
        )
        if not chunk_ids:
            return 0
        profile_ids: set[str] = set()
        for vector in session.scalars(select(EmbeddingVectorRecord)):
            source_ids = set(vector.source_chunk_ids_json)
            if vector.object_id in chunk_ids or source_ids.intersection(chunk_ids):
                profile_ids.add(vector.profile_id)
        if not profile_ids:
            return 0
        now = datetime.now(UTC)
        for profile in session.scalars(
            select(EmbeddingProfileRecord).where(EmbeddingProfileRecord.id.in_(profile_ids))
        ):
            profile.status = "failed"
            profile.is_active = False
            profile.error = reason
            profile.updated_at = now
        return len(profile_ids)

    def replace_index(
        self,
        session: Session,
        profile: IndexProfile,
        items: Sequence[IndexItem],
    ) -> StoredIndexProfile:
        """Atomically replace all vectors for a config/corpus/graph profile.

        Call this inside ``session_factory.begin()``.  The profile becomes the
        sole active index only after its complete replacement is flushed.
        """

        profile_id = profile.id or make_profile_id(
            profile.config_hash,
            profile.source_corpus_hash,
            profile.source_graph_run_id,
        )
        self._validate_profile(profile, profile_id)
        normalized_items = self._normalize_items(profile, profile_id, items)
        record = self._upsert_profile(session, profile, profile_id)

        session.execute(
            delete(EmbeddingVectorRecord).where(EmbeddingVectorRecord.profile_id == record.id)
        )
        session.flush()
        session.add_all(
            [
                EmbeddingVectorRecord(
                    id=item.id or make_vector_id(profile_id, item.kind, item.object_id),
                    profile_id=profile_id,
                    kind=item.kind,
                    object_id=item.object_id,
                    build_run_id=item.build_run_id or profile.source_graph_run_id,
                    source_content_hash=item.source_content_hash,
                    embedding_text=item.embedding_text,
                    embedding_json=list(item.embedding),
                    source_chunk_ids_json=list(item.source_chunk_ids),
                    metadata_json=self._json_mapping(item.metadata, "index item metadata"),
                )
                for item in normalized_items
            ]
        )
        for active in session.scalars(
            select(EmbeddingProfileRecord).where(EmbeddingProfileRecord.is_active.is_(True))
        ):
            active.is_active = False
        record.is_active = True
        record.status = "ready"
        record.error = None
        record.updated_at = datetime.now(UTC)
        session.flush()
        return self._stored_profile(record)

    def get_profile(
        self,
        session: Session,
        profile_ref: str | None = None,
    ) -> StoredIndexProfile | None:
        """Resolve an explicit profile ID/config hash or the active profile."""

        record = self._profile_record(session, profile_ref)
        return self._stored_profile(record) if record is not None else None

    def update_profile_metadata(
        self,
        session: Session,
        profile_id: str,
        metadata: Mapping[str, Any],
    ) -> StoredIndexProfile:
        """Refresh non-identity provenance without replacing ready vectors."""

        record = session.get(EmbeddingProfileRecord, profile_id)
        if record is None:
            raise RetrievalRepositoryError(f"unknown embedding profile: {profile_id}")
        record.metadata_json = self._json_mapping(metadata, "embedding profile metadata")
        record.updated_at = datetime.now(UTC)
        session.flush()
        return self._stored_profile(record)

    def list_profiles(
        self,
        session: Session,
        *,
        active_only: bool = False,
    ) -> list[StoredIndexProfile]:
        statement = select(EmbeddingProfileRecord).order_by(
            EmbeddingProfileRecord.is_active.desc(),
            EmbeddingProfileRecord.updated_at.desc(),
            EmbeddingProfileRecord.id,
        )
        if active_only:
            statement = statement.where(EmbeddingProfileRecord.is_active.is_(True))
        return [self._stored_profile(record) for record in session.scalars(statement)]

    def load_index(self, session: Session, profile_ref: str | None = None) -> LoadedIndex:
        """Load a ready index, split into stable chunk/entity/relation rows."""

        record = self._profile_record(session, profile_ref)
        if record is None:
            qualifier = profile_ref or "an active profile"
            raise RetrievalRepositoryError(f"no ready embedding index for {qualifier}")
        vectors = list(
            session.scalars(
                select(EmbeddingVectorRecord)
                .where(EmbeddingVectorRecord.profile_id == record.id)
                .order_by(EmbeddingVectorRecord.kind, EmbeddingVectorRecord.object_id)
            )
        )
        rows: dict[str, list[VectorRecord]] = {kind: [] for kind in INDEX_KINDS}
        for vector in vectors:
            if vector.kind not in rows:
                raise RetrievalRepositoryError(
                    f"index {record.id} contains unsupported vector kind {vector.kind!r}"
                )
            rows[vector.kind].append(
                VectorRecord(
                    id=vector.id,
                    object_id=vector.object_id,
                    kind=vector.kind,
                    embedding_text=vector.embedding_text,
                    embedding=self._vector(vector.embedding_json, record.id, vector.id),
                    source_chunk_ids=tuple(vector.source_chunk_ids_json),
                    build_run_id=vector.build_run_id,
                    source_content_hash=vector.source_content_hash,
                    metadata=dict(vector.metadata_json),
                )
            )
        return LoadedIndex(
            profile=self._stored_profile(record),
            chunks=tuple(rows["chunk"]),
            entities=tuple(rows["entity"]),
            relations=tuple(rows["relation"]),
        )

    def save_trace(self, session: Session, trace: RetrievalTrace) -> StoredRetrievalTrace:
        """Persist one replayable retrieval trace and its optional answer output."""

        profile = session.get(EmbeddingProfileRecord, trace.profile_id)
        if profile is None:
            raise RetrievalRepositoryError(f"unknown embedding profile: {trace.profile_id}")
        if profile.config_hash != trace.index_config_hash:
            raise RetrievalRepositoryError(
                "trace index config hash does not match its embedding profile"
            )
        if trace.mode not in RETRIEVAL_MODES:
            raise ValueError(f"unsupported retrieval mode: {trace.mode!r}")
        if not trace.query_text.strip():
            raise ValueError("retrieval trace query_text cannot be blank")
        self._require_hash("retrieval_config_hash", trace.retrieval_config_hash)
        trace_id = trace.id or f"rtr_{uuid4().hex}"
        if not trace_id.startswith("rtr_"):
            raise ValueError("retrieval trace IDs must use the rtr_ prefix")
        if session.get(RetrievalTraceRecord, trace_id) is not None:
            raise RetrievalRepositoryError(f"retrieval trace already exists: {trace_id}")
        query_hash = trace.query_hash or sha256_text(trace.query_text)
        self._require_hash("query_hash", query_hash)
        created_at = self._utc(trace.created_at)
        record = RetrievalTraceRecord(
            id=trace_id,
            profile_id=trace.profile_id,
            index_config_hash=trace.index_config_hash,
            graph_build_run_id=trace.graph_build_run_id or profile.source_graph_run_id,
            query_text=trace.query_text,
            query_hash=query_hash,
            mode=trace.mode,
            retrieval_config_hash=trace.retrieval_config_hash,
            trace_json=self._json_mapping(trace.trace_json, "retrieval trace"),
            output_json=(
                self._json_mapping(trace.output_json, "retrieval output")
                if trace.output_json is not None
                else None
            ),
            model_info_json=self._json_mapping(trace.model_info, "retrieval model info"),
            created_at=created_at,
        )
        session.add(record)
        session.flush()
        return self._stored_trace(record)

    def load_trace(self, session: Session, trace_id: str) -> StoredRetrievalTrace | None:
        """Load exactly the JSON captured at retrieval time for offline replay."""

        record = session.get(RetrievalTraceRecord, trace_id)
        return self._stored_trace(record) if record is not None else None

    def _snapshot_run(
        self,
        session: Session,
        build_run_id: str | None,
    ) -> GraphBuildRunRecord | None:
        if build_run_id is not None:
            run = session.get(GraphBuildRunRecord, build_run_id)
            if run is None:
                raise RetrievalRepositoryError(f"unknown graph build run: {build_run_id}")
            return run
        current_id = session.scalar(select(EntityRecord.build_run_id).limit(1))
        return session.get(GraphBuildRunRecord, current_id) if current_id is not None else None

    @staticmethod
    def _entity_evidence(session: Session, entity_ids: Sequence[str]) -> tuple[SourceEvidence, ...]:
        if not entity_ids:
            return ()
        records = session.scalars(
            select(EntityEvidenceRecord)
            .where(EntityEvidenceRecord.entity_id.in_(entity_ids))
            .order_by(
                EntityEvidenceRecord.entity_id,
                EntityEvidenceRecord.chunk_id,
                EntityEvidenceRecord.char_start,
                EntityEvidenceRecord.id,
            )
        )
        return tuple(
            SourceEvidence(
                id=record.id,
                owner_kind="entity",
                owner_id=record.entity_id,
                chunk_id=record.chunk_id,
                extraction_id=record.extraction_id,
                mention_id=record.mention_id,
                quote=record.quote,
                char_start=record.char_start,
                char_end=record.char_end,
            )
            for record in records
        )

    @staticmethod
    def _relation_evidence(
        session: Session,
        relation_ids: Sequence[str],
    ) -> tuple[SourceEvidence, ...]:
        if not relation_ids:
            return ()
        records = session.scalars(
            select(RelationEvidenceRecord)
            .where(RelationEvidenceRecord.relation_id.in_(relation_ids))
            .order_by(
                RelationEvidenceRecord.relation_id,
                RelationEvidenceRecord.chunk_id,
                RelationEvidenceRecord.char_start,
                RelationEvidenceRecord.id,
            )
        )
        return tuple(
            SourceEvidence(
                id=record.id,
                owner_kind="relation",
                owner_id=record.relation_id,
                chunk_id=record.chunk_id,
                extraction_id=record.extraction_id,
                mention_id=record.mention_id,
                quote=record.quote,
                char_start=record.char_start,
                char_end=record.char_end,
            )
            for record in records
        )

    def _upsert_profile(
        self,
        session: Session,
        profile: IndexProfile,
        profile_id: str,
    ) -> EmbeddingProfileRecord:
        by_id = session.get(EmbeddingProfileRecord, profile_id)
        identity_filters = [
            EmbeddingProfileRecord.config_hash == profile.config_hash,
            EmbeddingProfileRecord.source_corpus_hash == profile.source_corpus_hash,
        ]
        if profile.source_graph_run_id is None:
            identity_filters.append(EmbeddingProfileRecord.source_graph_run_id.is_(None))
        else:
            identity_filters.append(
                EmbeddingProfileRecord.source_graph_run_id == profile.source_graph_run_id
            )
        by_key = session.scalar(select(EmbeddingProfileRecord).where(*identity_filters))
        if by_id is not None and by_key is not None and by_id.id != by_key.id:
            raise RetrievalRepositoryError(
                "embedding profile ID conflicts with an existing config/corpus/graph profile"
            )
        record = by_id or by_key
        if record is None:
            record = EmbeddingProfileRecord(
                id=profile_id,
                config_hash=profile.config_hash,
                provider=profile.provider,
                model=profile.model,
                dimensions=profile.dimensions,
                schema_version=profile.schema_version,
                source_graph_run_id=profile.source_graph_run_id,
                source_corpus_hash=profile.source_corpus_hash,
                metadata_json=self._json_mapping(profile.metadata, "embedding profile metadata"),
                status="ready",
                is_active=False,
            )
            session.add(record)
            session.flush()
            return record
        expected = (
            profile.config_hash,
            profile.provider,
            profile.model,
            profile.dimensions,
            profile.schema_version,
            profile.source_corpus_hash,
            profile.source_graph_run_id,
        )
        actual = (
            record.config_hash,
            record.provider,
            record.model,
            record.dimensions,
            record.schema_version,
            record.source_corpus_hash,
            record.source_graph_run_id,
        )
        if actual != expected:
            raise RetrievalRepositoryError(
                f"embedding profile {record.id} conflicts with the supplied semantic configuration"
            )
        record.metadata_json = self._json_mapping(profile.metadata, "embedding profile metadata")
        record.updated_at = datetime.now(UTC)
        return record

    @staticmethod
    def _profile_record(
        session: Session,
        profile_ref: str | None,
    ) -> EmbeddingProfileRecord | None:
        if profile_ref is not None:
            by_id = session.get(EmbeddingProfileRecord, profile_ref)
            if by_id is not None:
                return by_id if by_id.status == "ready" else None
            return session.scalar(
                select(EmbeddingProfileRecord)
                .where(
                    EmbeddingProfileRecord.config_hash == profile_ref,
                    EmbeddingProfileRecord.status == "ready",
                )
                .order_by(
                    EmbeddingProfileRecord.is_active.desc(),
                    EmbeddingProfileRecord.updated_at.desc(),
                    EmbeddingProfileRecord.id,
                )
            )
        return session.scalar(
            select(EmbeddingProfileRecord)
            .where(
                EmbeddingProfileRecord.is_active.is_(True),
                EmbeddingProfileRecord.status == "ready",
            )
            .order_by(EmbeddingProfileRecord.updated_at.desc(), EmbeddingProfileRecord.id)
        )

    @staticmethod
    def _stored_profile(record: EmbeddingProfileRecord) -> StoredIndexProfile:
        return StoredIndexProfile(
            id=record.id,
            config_hash=record.config_hash,
            provider=record.provider,
            model=record.model,
            dimensions=record.dimensions,
            schema_version=record.schema_version,
            source_corpus_hash=record.source_corpus_hash,
            source_graph_run_id=record.source_graph_run_id,
            metadata=dict(record.metadata_json),
            status=record.status,
            is_active=record.is_active,
            created_at=record.created_at,
            updated_at=record.updated_at,
            error=record.error,
        )

    @staticmethod
    def _stored_trace(record: RetrievalTraceRecord) -> StoredRetrievalTrace:
        return StoredRetrievalTrace(
            id=record.id,
            profile_id=record.profile_id,
            index_config_hash=record.index_config_hash,
            graph_build_run_id=record.graph_build_run_id,
            query_text=record.query_text,
            query_hash=record.query_hash,
            mode=record.mode,
            retrieval_config_hash=record.retrieval_config_hash,
            trace_json=dict(record.trace_json),
            output_json=dict(record.output_json) if record.output_json is not None else None,
            model_info=dict(record.model_info_json),
            created_at=record.created_at,
        )

    @staticmethod
    def _validate_profile(profile: IndexProfile, profile_id: str) -> None:
        if not profile_id.startswith("idx_"):
            raise ValueError("embedding profile IDs must use the idx_ prefix")
        for name, value in (
            ("config_hash", profile.config_hash),
            ("source_corpus_hash", profile.source_corpus_hash),
        ):
            RetrievalRepository._require_hash(name, value)
        if profile.source_graph_run_id is not None and not profile.source_graph_run_id.strip():
            raise ValueError("source_graph_run_id cannot be blank when provided")
        if not profile.provider.strip():
            raise ValueError("embedding provider cannot be blank")
        if not profile.model.strip():
            raise ValueError("embedding model cannot be blank")
        if not profile.schema_version.strip():
            raise ValueError("embedding schema_version cannot be blank")
        if profile.dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")

    @staticmethod
    def _normalize_items(
        profile: IndexProfile,
        profile_id: str,
        items: Sequence[IndexItem],
    ) -> tuple[IndexItem, ...]:
        seen: set[tuple[str, str]] = set()
        normalized: list[IndexItem] = []
        for item in items:
            if item.kind not in INDEX_KINDS:
                raise ValueError(f"unsupported embedding item kind: {item.kind!r}")
            if not item.object_id.strip():
                raise ValueError("embedding item object_id cannot be blank")
            if not item.embedding_text.strip():
                raise ValueError(f"embedding text cannot be blank for {item.kind}:{item.object_id}")
            key = (item.kind, item.object_id)
            if key in seen:
                raise RetrievalRepositoryError(
                    f"duplicate vector in index {profile_id}: {item.kind}:{item.object_id}"
                )
            seen.add(key)
            embedding = tuple(float(value) for value in item.embedding)
            if len(embedding) != profile.dimensions:
                raise ValueError(
                    f"vector {item.kind}:{item.object_id} has {len(embedding)} dimensions; "
                    f"expected {profile.dimensions}"
                )
            if not all(isfinite(value) for value in embedding):
                raise ValueError(f"vector {item.kind}:{item.object_id} contains a non-finite value")
            if item.id is not None and not item.id.startswith("vec_"):
                raise ValueError("embedding vector IDs must use the vec_ prefix")
            normalized.append(
                IndexItem(
                    id=item.id,
                    object_id=item.object_id,
                    kind=item.kind,
                    embedding_text=item.embedding_text,
                    embedding=embedding,
                    source_chunk_ids=tuple(item.source_chunk_ids),
                    build_run_id=item.build_run_id or profile.source_graph_run_id,
                    source_content_hash=item.source_content_hash,
                    metadata=item.metadata,
                )
            )
        return tuple(normalized)

    @staticmethod
    def _vector(value: Sequence[Any], profile_id: str, vector_id: str) -> tuple[float, ...]:
        try:
            embedding = tuple(float(component) for component in value)
        except (TypeError, ValueError) as error:
            raise RetrievalRepositoryError(
                f"vector {vector_id} in profile {profile_id} is not numeric"
            ) from error
        if not embedding or not all(isfinite(component) for component in embedding):
            raise RetrievalRepositoryError(
                f"vector {vector_id} in profile {profile_id} is empty or non-finite"
            )
        return embedding

    @staticmethod
    def _json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
        try:
            serialized = json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be JSON-serializable") from error
        decoded = json.loads(serialized)
        if not isinstance(decoded, dict):  # Defensive; dict(value) above guarantees this today.
            raise ValueError(f"{label} must encode as a JSON object")
        return decoded

    @staticmethod
    def _require_hash(name: str, value: str) -> None:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")

    @staticmethod
    def _utc(value: datetime | None) -> datetime:
        current = value or datetime.now(UTC)
        if current.tzinfo is None:
            return current.replace(tzinfo=UTC)
        return current.astimezone(UTC)


def make_vector_id(profile_id: str, kind: str, object_id: str) -> str:
    """Return the deterministic row ID for one profile/object pair."""

    return stable_id("vec", profile_id, kind, object_id)
