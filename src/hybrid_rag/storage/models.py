from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
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
