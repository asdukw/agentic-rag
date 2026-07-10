from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from sqlalchemy import func, select

from hybrid_rag.config import sqlite_url
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.storage.database import Database, sqlite_foreign_keys_enabled
from hybrid_rag.storage.models import ChunkRecord, DocumentRecord
from hybrid_rag.storage.repository import IngestRepository


class WordCounter:
    name = "test-words"

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


def _service(tmp_path: Path) -> tuple[Database, IngestionService]:
    database = Database(sqlite_url(tmp_path / "app.db"))
    database.create_schema()
    chunker = SectionTokenChunker(WordCounter(), max_tokens=24, overlap_tokens=4)
    return database, IngestionService(database, chunker)


def _copy_corpus(tmp_path: Path) -> Path:
    source = Path(__file__).parents[1] / "fixtures" / "corpus"
    target = tmp_path / "corpus"
    shutil.copytree(source, target)
    return target


def test_ingest_is_idempotent_and_updates_one_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _copy_corpus(tmp_path)
    database, service = _service(tmp_path)
    try:
        first = service.ingest(corpus)
        with database.session_factory() as session:
            first_stats = IngestRepository().stats(session)

        text_loader = service.loaders.for_path(corpus / "rag_notes.txt")

        def fail_if_reparsed(*_: object) -> None:
            raise AssertionError("unchanged documents should skip before parsing")

        with monkeypatch.context() as patch:
            patch.setattr(text_loader, "load", fail_if_reparsed)
            second = service.ingest(corpus)
        with database.session_factory() as session:
            second_stats = IngestRepository().stats(session)

        note = corpus / "rag_notes.txt"
        note.write_text(note.read_text(encoding="utf-8") + "\n\nNew evidence.", encoding="utf-8")
        third = service.ingest(corpus)
        with database.session_factory() as session:
            third_stats = IngestRepository().stats(session)
            orphan_count = session.scalar(
                select(func.count())
                .select_from(ChunkRecord)
                .outerjoin(DocumentRecord)
                .where(DocumentRecord.id.is_(None))
            )

        assert first.inserted == 2 and first.failed == 0
        assert second.skipped == 2 and second.chunks_written == 0
        assert first_stats.documents == second_stats.documents == 2
        assert first_stats.chunks == second_stats.chunks
        assert third.updated == 1 and third.skipped == 1
        assert third_stats.documents == 2
        assert orphan_count == 0
        assert sqlite_foreign_keys_enabled(database.engine)
    finally:
        database.dispose()


def test_corrupt_pdf_does_not_block_valid_document(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "valid.txt").write_text("Valid evidence for the index.", encoding="utf-8")
    (corpus / "broken.pdf").write_bytes(b"this is not a pdf")
    database, service = _service(tmp_path)
    try:
        report = service.ingest(corpus)
    finally:
        database.dispose()

    assert report.inserted == 1
    assert report.failed == 1
    assert report.failures[0].path.endswith("broken.pdf")
