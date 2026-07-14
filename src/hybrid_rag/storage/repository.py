from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from hybrid_rag.schemas import ChunkData, IngestReport, ParsedDocument, StorageStats
from hybrid_rag.storage.graph_repository import GraphRepository
from hybrid_rag.storage.models import ChunkRecord, DocumentRecord, IngestRunRecord
from hybrid_rag.storage.retrieval_repository import RetrievalRepository


@dataclass(frozen=True, slots=True)
class WriteResult:
    status: str
    chunks_written: int


class IngestRepository:
    def is_current(
        self,
        session: Session,
        document_id: str,
        content_hash: str,
        parser_name: str,
        parser_version: str,
        processing_config_hash: str,
    ) -> bool:
        existing = session.get(DocumentRecord, document_id)
        return bool(
            existing is not None
            and existing.content_hash == content_hash
            and existing.parser_name == parser_name
            and existing.parser_version == parser_version
            and existing.processing_config_hash == processing_config_hash
        )

    def upsert_document(
        self,
        session: Session,
        document: ParsedDocument,
        chunks: list[ChunkData],
        processing_config_hash: str,
    ) -> WriteResult:
        existing = session.get(DocumentRecord, document.id)
        if (
            existing is not None
            and existing.content_hash == document.content_hash
            and existing.parser_name == document.parser_name
            and existing.parser_version == document.parser_version
            and existing.processing_config_hash == processing_config_hash
        ):
            return WriteResult("skipped", 0)

        values = {
            "title": document.title,
            "source_type": document.source_type,
            "source_uri": document.source_uri,
            "local_path": document.local_path,
            "content_hash": document.content_hash,
            "parsed_text": document.text,
            "parser_name": document.parser_name,
            "parser_version": document.parser_version,
            "processing_config_hash": processing_config_hash,
            "metadata_json": document.metadata,
        }

        if existing is None:
            existing = DocumentRecord(id=document.id, **values)
            session.add(existing)
            session.flush()
            status = "inserted"
        else:
            for key, value in values.items():
                setattr(existing, key, value)
            GraphRepository().invalidate_runs_for_document(session, document.id)
            RetrievalRepository().invalidate_indexes_for_document(session, document.id)
            session.execute(delete(ChunkRecord).where(ChunkRecord.document_id == document.id))
            session.flush()
            status = "updated"

        session.add_all([self._chunk_record(chunk) for chunk in chunks])
        return WriteResult(status, len(chunks))

    def record_run(self, session: Session, report: IngestReport) -> None:
        session.add(
            IngestRunRecord(
                id=report.run_id,
                source_path=report.source_path,
                config_hash=report.config_hash,
                started_at=report.started_at,
                finished_at=report.finished_at,
                duration_seconds=report.duration_seconds,
                discovered=report.discovered,
                inserted=report.inserted,
                updated=report.updated,
                skipped=report.skipped,
                failed=report.failed,
                chunks_written=report.chunks_written,
                failures_json=[failure.model_dump() for failure in report.failures],
            )
        )

    def stats(self, session: Session) -> StorageStats:
        documents = session.scalar(select(func.count()).select_from(DocumentRecord)) or 0
        row = session.execute(
            select(
                func.count(ChunkRecord.id),
                func.coalesce(func.sum(ChunkRecord.token_count), 0),
                func.min(ChunkRecord.token_count),
                func.max(ChunkRecord.token_count),
                func.avg(ChunkRecord.token_count),
            )
        ).one()
        return StorageStats(
            documents=documents,
            chunks=row[0],
            total_tokens=row[1],
            min_chunk_tokens=row[2],
            max_chunk_tokens=row[3],
            average_chunk_tokens=float(row[4]) if row[4] is not None else None,
        )

    def get_document(self, session: Session, document_id: str) -> DocumentRecord | None:
        statement = (
            select(DocumentRecord)
            .where(DocumentRecord.id == document_id)
            .options(selectinload(DocumentRecord.chunks))
        )
        return session.scalar(statement)

    @staticmethod
    def _chunk_record(chunk: ChunkData) -> ChunkRecord:
        return ChunkRecord(
            id=chunk.id,
            document_id=chunk.document_id,
            ordinal=chunk.ordinal,
            section_path_json=list(chunk.section_path),
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            text=chunk.text,
            contextualized_text=chunk.contextualized_text,
            token_count=chunk.token_count,
            content_hash=chunk.content_hash,
            chunker_name=chunk.chunker_name,
            chunker_version=chunk.chunker_version,
            metadata_json=chunk.metadata,
        )
