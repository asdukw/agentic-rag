from pathlib import Path

from sqlalchemy import inspect, text

from hybrid_rag.config import sqlite_url
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database


def test_alembic_builds_empty_database(tmp_path: Path) -> None:
    url = sqlite_url(tmp_path / "migrated.db")

    upgrade_database(url)
    upgrade_database(url)
    database = Database(url)
    try:
        assert set(inspect(database.engine).get_table_names()) >= {
            "alembic_version",
            "documents",
            "chunks",
            "ingest_runs",
            "graph_build_runs",
            "chunk_extractions",
            "extraction_attempts",
            "graph_build_items",
            "entities",
            "entity_evidence",
            "relations",
            "relation_evidence",
        }
    finally:
        database.dispose()


def test_phase_two_upgrade_preserves_phase_one_data(tmp_path: Path) -> None:
    url = sqlite_url(tmp_path / "phase-one.db")
    upgrade_database(url, "0001_ingestion")
    database = Database(url)
    try:
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO documents (
                        id, title, source_type, source_uri, local_path, content_hash,
                        parsed_text, parser_name, parser_version, processing_config_hash,
                        metadata_json, created_at, updated_at
                    ) VALUES (
                        :id, :title, :source_type, :source_uri, :local_path, :content_hash,
                        :parsed_text, :parser_name, :parser_version, :processing_config_hash,
                        :metadata_json, :created_at, :updated_at
                    )
                    """
                ),
                {
                    "id": "doc_existing",
                    "title": "Existing",
                    "source_type": "txt",
                    "source_uri": "file:test/existing.txt",
                    "local_path": "existing.txt",
                    "content_hash": "d" * 64,
                    "parsed_text": "Existing chunk.",
                    "parser_name": "test",
                    "parser_version": "1",
                    "processing_config_hash": "p" * 64,
                    "metadata_json": "{}",
                    "created_at": "2026-07-11 00:00:00",
                    "updated_at": "2026-07-11 00:00:00",
                },
            )
            connection.execute(
                text(
                    """
                    INSERT INTO chunks (
                        id, document_id, ordinal, section_path_json, page_start, page_end,
                        char_start, char_end, text, contextualized_text, token_count,
                        content_hash, chunker_name, chunker_version, metadata_json, created_at
                    ) VALUES (
                        :id, :document_id, 0, :section_path_json, NULL, NULL,
                        0, 15, :text, :text, 3, :content_hash, :chunker_name,
                        :chunker_version, :metadata_json, :created_at
                    )
                    """
                ),
                {
                    "id": "chk_existing",
                    "document_id": "doc_existing",
                    "section_path_json": "[]",
                    "text": "Existing chunk.",
                    "content_hash": "c" * 64,
                    "chunker_name": "test",
                    "chunker_version": "1",
                    "metadata_json": "{}",
                    "created_at": "2026-07-11 00:00:00",
                },
            )
    finally:
        database.dispose()

    upgrade_database(url)
    upgraded = Database(url)
    try:
        with upgraded.engine.connect() as connection:
            assert connection.scalar(text("SELECT count(*) FROM documents")) == 1
            assert connection.scalar(text("SELECT count(*) FROM chunks")) == 1
            assert "graph_build_runs" in inspect(connection).get_table_names()
    finally:
        upgraded.dispose()
