from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TextSegment(BaseModel):
    text: str
    section_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    char_start: int = 0
    char_end: int = 0


class ParsedDocument(BaseModel):
    id: str
    title: str
    source_type: str
    source_uri: str
    local_path: str
    content_hash: str
    text: str = ""
    segments: list[TextSegment] = Field(default_factory=list)
    parser_name: str
    parser_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkData(BaseModel):
    id: str
    document_id: str
    ordinal: int
    section_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    char_start: int
    char_end: int
    text: str
    contextualized_text: str
    token_count: int
    content_hash: str
    chunker_name: str
    chunker_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FileFailure(BaseModel):
    path: str
    error_type: str
    message: str


class IngestReport(BaseModel):
    run_id: str
    source_path: str
    config_hash: str
    started_at: datetime
    finished_at: datetime
    discovered: int
    inserted: int
    updated: int
    skipped: int
    failed: int
    chunks_written: int
    failures: list[FileFailure] = Field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return self.inserted + self.updated + self.skipped

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


class StorageStats(BaseModel):
    documents: int
    chunks: int
    total_tokens: int
    min_chunk_tokens: int | None
    max_chunk_tokens: int | None
    average_chunk_tokens: float | None
