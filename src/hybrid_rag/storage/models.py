from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (JSON, Boolean, CheckConstraint, DateTime, Float,
                        ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint, text)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    local_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    parsed_text: Mapped[str] = mapped_column(Text, nullable=False)
    parser_name: Mapped[str] = mapped_column(String(100), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(50), nullable=False)
    processing_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    chunks: Mapped[list[ChunkRecord]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChunkRecord.ordinal",
    )


class ChunkRecord(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        Index("ix_chunks_document_id", "document_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    contextualized_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chunker_name: Mapped[str] = mapped_column(String(100), nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")
    extractions: Mapped[list[ChunkExtractionRecord]] = relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    entity_evidence: Mapped[list[EntityEvidenceRecord]] = relationship(
        back_populates="chunk",
        passive_deletes=True,
    )
    relation_evidence: Mapped[list[RelationEvidenceRecord]] = relationship(
        back_populates="chunk",
        passive_deletes=True,
    )


class IngestRunRecord(Base):
    __tablename__ = "ingest_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    discovered: Mapped[int] = mapped_column(Integer, nullable=False)
    inserted: Mapped[int] = mapped_column(Integer, nullable=False)
    updated: Mapped[int] = mapped_column(Integer, nullable=False)
    skipped: Mapped[int] = mapped_column(Integer, nullable=False)
    failed: Mapped[int] = mapped_column(Integer, nullable=False)
    chunks_written: Mapped[int] = mapped_column(Integer, nullable=False)
    failures_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)


class GraphBuildRunRecord(Base):
    __tablename__ = "graph_build_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'awaiting_review', 'completed', "
            "'completed_with_failures', 'failed')",
            name="ck_graph_build_runs_status",
        ),
        CheckConstraint(
            "total_chunks >= 0 AND cached_chunks >= 0 AND scheduled_chunks >= 0 "
            "AND succeeded_chunks >= 0 AND needs_review_chunks >= 0 "
            "AND failed_chunks >= 0",
            name="ck_graph_build_runs_chunk_counts",
        ),
        CheckConstraint(
            "attempt_count >= 0 AND extract_attempt_count >= 0 AND repair_attempt_count >= 0",
            name="ck_graph_build_runs_attempt_counts",
        ),
        CheckConstraint(
            "entity_count >= 0 AND relation_count >= 0 AND component_count >= 0 "
            "AND largest_component_nodes >= 0 AND isolated_entity_count >= 0",
            name="ck_graph_build_runs_graph_counts",
        ),
        CheckConstraint(
            "prompt_tokens >= 0 AND completion_tokens >= 0 AND total_tokens >= 0",
            name="ck_graph_build_runs_usage_counts",
        ),
        Index("ix_graph_build_runs_status", "status"),
        Index(
            "ix_graph_build_runs_graph_config_corpus",
            "graph_config_hash",
            "corpus_hash",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    extraction_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    graph_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    corpus_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(50), nullable=False)
    workflow_version: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    review_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cached_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scheduled_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    succeeded_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    needs_review_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    extract_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    repair_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    entity_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    relation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    component_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    largest_component_nodes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    isolated_entity_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text)

    items: Mapped[list[GraphBuildItemRecord]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="GraphBuildItemRecord.ordinal",
    )
    attempts: Mapped[list[ExtractionAttemptRecord]] = relationship(
        back_populates="run",
        passive_deletes=True,
    )
    entities: Mapped[list[EntityRecord]] = relationship(
        back_populates="build_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    relations: Mapped[list[RelationRecord]] = relationship(
        back_populates="build_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChunkExtractionRecord(Base):
    __tablename__ = "chunk_extractions"
    __table_args__ = (
        UniqueConstraint(
            "chunk_id",
            "extraction_config_hash",
            name="uq_chunk_extractions_chunk_config",
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'needs_review', 'failed')",
            name="ck_chunk_extractions_status",
        ),
        CheckConstraint("attempt_count >= 0", name="ck_chunk_extractions_attempt_count"),
        Index(
            "ix_chunk_extractions_config_status",
            "extraction_config_hash",
            "status",
        ),
        Index("ix_chunk_extractions_chunk_id", "chunk_id"),
        Index("ix_chunk_extractions_lease", "status", "lease_expires_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    extraction_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending")
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_token: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_notes: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    chunk: Mapped[ChunkRecord] = relationship(back_populates="extractions")
    attempts: Mapped[list[ExtractionAttemptRecord]] = relationship(
        back_populates="extraction",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ExtractionAttemptRecord.ordinal",
    )
    build_items: Mapped[list[GraphBuildItemRecord]] = relationship(
        back_populates="extraction",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    entity_evidence: Mapped[list[EntityEvidenceRecord]] = relationship(
        back_populates="extraction",
        passive_deletes=True,
    )
    relation_evidence: Mapped[list[RelationEvidenceRecord]] = relationship(
        back_populates="extraction",
        passive_deletes=True,
    )


class ExtractionAttemptRecord(Base):
    __tablename__ = "extraction_attempts"
    __table_args__ = (
        UniqueConstraint(
            "extraction_id",
            "ordinal",
            name="uq_extraction_attempts_extraction_ordinal",
        ),
        CheckConstraint("ordinal > 0", name="ck_extraction_attempts_ordinal"),
        CheckConstraint("stage IN ('extract', 'repair')", name="ck_extraction_attempts_stage"),
        CheckConstraint(
            "prompt_tokens >= 0 AND completion_tokens >= 0 AND total_tokens >= 0",
            name="ck_extraction_attempts_usage",
        ),
        Index("ix_extraction_attempts_extraction_id", "extraction_id"),
        Index("ix_extraction_attempts_run_id", "run_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    extraction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chunk_extractions.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("graph_build_runs.id", ondelete="SET NULL")
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    stage: Mapped[str] = mapped_column(String(16), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    messages_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    raw_response: Mapped[str | None] = mapped_column(Text)
    response_metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latency_seconds: Mapped[float | None] = mapped_column(Float)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    extraction: Mapped[ChunkExtractionRecord] = relationship(back_populates="attempts")
    run: Mapped[GraphBuildRunRecord | None] = relationship(back_populates="attempts")


class GraphBuildItemRecord(Base):
    __tablename__ = "graph_build_items"
    __table_args__ = (
        UniqueConstraint("run_id", "ordinal", name="uq_graph_build_items_run_ordinal"),
        CheckConstraint("ordinal >= 0", name="ck_graph_build_items_ordinal"),
        CheckConstraint(
            "disposition IN ('scheduled', 'cached')",
            name="ck_graph_build_items_disposition",
        ),
        CheckConstraint(
            "review_status IN ('not_required', 'pending', 'approved', 'rejected')",
            name="ck_graph_build_items_review_status",
        ),
        Index("ix_graph_build_items_extraction_id", "extraction_id"),
    )

    run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("graph_build_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    extraction_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("chunk_extractions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    disposition: Mapped[str] = mapped_column(String(16), nullable=False)
    review_status: Mapped[str] = mapped_column(String(16), nullable=False, default="not_required")
    review_notes: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    run: Mapped[GraphBuildRunRecord] = relationship(back_populates="items")
    extraction: Mapped[ChunkExtractionRecord] = relationship(back_populates="build_items")


class EntityRecord(Base):
    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint(
            "build_run_id",
            "normalized_name",
            "entity_type",
            name="uq_entities_run_normalized_name_type",
        ),
        Index("ix_entities_build_run_id", "build_run_id"),
        Index("ix_entities_normalized_name", "normalized_name"),
        Index("ix_entities_type", "entity_type"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    build_run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("graph_build_runs.id", ondelete="CASCADE"), nullable=False
    )
    graph_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_name: Mapped[str] = mapped_column(Text, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    aliases_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    source_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    build_run: Mapped[GraphBuildRunRecord] = relationship(back_populates="entities")
    evidence: Mapped[list[EntityEvidenceRecord]] = relationship(
        back_populates="entity",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    outgoing_relations: Mapped[list[RelationRecord]] = relationship(
        back_populates="source_entity",
        foreign_keys="RelationRecord.source_entity_id",
        passive_deletes=True,
    )
    incoming_relations: Mapped[list[RelationRecord]] = relationship(
        back_populates="target_entity",
        foreign_keys="RelationRecord.target_entity_id",
        passive_deletes=True,
    )


class EntityEvidenceRecord(Base):
    __tablename__ = "entity_evidence"
    __table_args__ = (
        UniqueConstraint(
            "entity_id",
            "extraction_id",
            "quote",
            "char_start",
            "char_end",
            name="uq_entity_evidence_source_span",
        ),
        CheckConstraint(
            "char_start >= 0 AND char_end >= char_start",
            name="ck_entity_evidence_char_span",
        ),
        Index("ix_entity_evidence_entity_id", "entity_id"),
        Index("ix_entity_evidence_chunk_id", "chunk_id"),
        Index("ix_entity_evidence_extraction_id", "extraction_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    extraction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chunk_extractions.id", ondelete="CASCADE"), nullable=False
    )
    mention_id: Mapped[str | None] = mapped_column(String(64))
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    entity: Mapped[EntityRecord] = relationship(back_populates="evidence")
    chunk: Mapped[ChunkRecord] = relationship(back_populates="entity_evidence")
    extraction: Mapped[ChunkExtractionRecord] = relationship(back_populates="entity_evidence")


class RelationRecord(Base):
    __tablename__ = "relations"
    __table_args__ = (
        UniqueConstraint(
            "build_run_id",
            "source_entity_id",
            "target_entity_id",
            "predicate",
            name="uq_relations_run_endpoints_predicate",
        ),
        Index("ix_relations_build_run_id", "build_run_id"),
        Index("ix_relations_source_entity_id", "source_entity_id"),
        Index("ix_relations_target_entity_id", "target_entity_id"),
        Index("ix_relations_predicate", "predicate"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    build_run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("graph_build_runs.id", ondelete="CASCADE"), nullable=False
    )
    graph_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_entity_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    target_entity_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    predicate: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    source_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    build_run: Mapped[GraphBuildRunRecord] = relationship(back_populates="relations")
    source_entity: Mapped[EntityRecord] = relationship(
        back_populates="outgoing_relations", foreign_keys=[source_entity_id]
    )
    target_entity: Mapped[EntityRecord] = relationship(
        back_populates="incoming_relations", foreign_keys=[target_entity_id]
    )
    evidence: Mapped[list[RelationEvidenceRecord]] = relationship(
        back_populates="relation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class RelationEvidenceRecord(Base):
    __tablename__ = "relation_evidence"
    __table_args__ = (
        UniqueConstraint(
            "relation_id",
            "extraction_id",
            "quote",
            "char_start",
            "char_end",
            name="uq_relation_evidence_source_span",
        ),
        CheckConstraint(
            "char_start >= 0 AND char_end >= char_start",
            name="ck_relation_evidence_char_span",
        ),
        Index("ix_relation_evidence_relation_id", "relation_id"),
        Index("ix_relation_evidence_chunk_id", "chunk_id"),
        Index("ix_relation_evidence_extraction_id", "extraction_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    relation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("relations.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    extraction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chunk_extractions.id", ondelete="CASCADE"), nullable=False
    )
    mention_id: Mapped[str | None] = mapped_column(String(64))
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    relation: Mapped[RelationRecord] = relationship(back_populates="evidence")
    chunk: Mapped[ChunkRecord] = relationship(back_populates="relation_evidence")
    extraction: Mapped[ChunkExtractionRecord] = relationship(back_populates="relation_evidence")


class EmbeddingProfileRecord(Base):
    """A reproducible embedding configuration over one corpus snapshot.

    The actual vector search implementation intentionally lives outside the ORM.
    This record only captures the immutable inputs that make an index rebuild
    reproducible and points at the graph snapshot, when one was available.
    """

    __tablename__ = "embedding_profiles"
    __table_args__ = (
        UniqueConstraint(
            "config_hash",
            "source_corpus_hash",
            "source_graph_run_id",
            name="uq_embedding_profiles_config_corpus_graph_run",
        ),
        CheckConstraint("dimensions > 0", name="ck_embedding_profiles_dimensions"),
        CheckConstraint(
            "status IN ('ready', 'failed')",
            name="ck_embedding_profiles_status",
        ),
        Index("ix_embedding_profiles_active", "is_active"),
        Index("ix_embedding_profiles_source_graph_run", "source_graph_run_id"),
        Index(
            "uq_embedding_profiles_config_corpus_no_graph",
            "config_hash",
            "source_corpus_hash",
            unique=True,
            sqlite_where=text("source_graph_run_id IS NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(50), nullable=False)
    source_graph_run_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("graph_build_runs.id", ondelete="SET NULL"),
    )
    source_corpus_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ready")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )
    error: Mapped[str | None] = mapped_column(Text)

    vectors: Mapped[list[EmbeddingVectorRecord]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    retrieval_traces: Mapped[list[RetrievalTraceRecord]] = relationship(
        back_populates="profile",
        passive_deletes=True,
    )


class EmbeddingVectorRecord(Base):
    """One persisted vector for a chunk, canonical entity, or relation."""

    __tablename__ = "embedding_vectors"
    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "kind",
            "object_id",
            name="uq_embedding_vectors_profile_kind_object",
        ),
        CheckConstraint(
            "kind IN ('chunk', 'entity', 'relation')",
            name="ck_embedding_vectors_kind",
        ),
        Index("ix_embedding_vectors_profile_kind", "profile_id", "kind"),
        Index("ix_embedding_vectors_build_run", "build_run_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("embedding_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    object_id: Mapped[str] = mapped_column(String(64), nullable=False)
    build_run_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("graph_build_runs.id", ondelete="SET NULL"),
    )
    source_content_hash: Mapped[str | None] = mapped_column(String(64))
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_json: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    source_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    profile: Mapped[EmbeddingProfileRecord] = relationship(back_populates="vectors")


class RetrievalTraceRecord(Base):
    """A durable, serializable retrieval result that can be replayed offline."""

    __tablename__ = "retrieval_traces"
    __table_args__ = (
        CheckConstraint(
            "mode IN ('naive', 'local', 'global', 'hybrid')",
            name="ck_retrieval_traces_mode",
        ),
        Index("ix_retrieval_traces_profile_created", "profile_id", "created_at"),
        Index("ix_retrieval_traces_query_hash", "query_hash"),
        Index("ix_retrieval_traces_build_run", "graph_build_run_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("embedding_profiles.id", ondelete="RESTRICT"),
        nullable=False,
    )
    index_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    graph_build_run_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("graph_build_runs.id", ondelete="SET NULL"),
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    retrieval_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    trace_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    output_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    model_info_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    profile: Mapped[EmbeddingProfileRecord] = relationship(back_populates="retrieval_traces")
