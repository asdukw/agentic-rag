from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from hybrid_rag.config import sqlite_url
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database
from hybrid_rag.storage.models import (EmbeddingProfileRecord,
                                       EmbeddingVectorRecord,
                                       GraphBuildRunRecord,
                                       RetrievalTraceRecord)
from hybrid_rag.storage.retrieval_repository import (make_profile_id,
                                                     make_vector_id)


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
            "embedding_profiles",
            "embedding_vectors",
            "retrieval_traces",
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


def test_embedding_profile_graph_identity_migration_rekeys_and_round_trips(tmp_path: Path) -> None:
    url = sqlite_url(tmp_path / "profile-identity.db")
    config_hash = "a" * 64
    source_corpus_hash = "c" * 64
    graph_run_id = "gbr_graph"
    legacy_profile_id = make_profile_id(config_hash, source_corpus_hash)
    snapshot_profile_id = make_profile_id(config_hash, source_corpus_hash, graph_run_id)
    legacy_vector_id = make_vector_id(legacy_profile_id, "chunk", "chk_one")

    upgrade_database(url, "0003_retrieval_indexes")
    legacy = Database(url)
    try:
        with legacy.session_factory.begin() as session:
            session.add(
                GraphBuildRunRecord(
                    id=graph_run_id,
                    extraction_config_hash="e" * 64,
                    graph_config_hash="g" * 64,
                    corpus_hash="h" * 64,
                    model="test",
                    prompt_version="1",
                    schema_version="1",
                    workflow_version="1",
                    status="completed",
                    review_required=False,
                    report_json={},
                )
            )
            session.flush()
            session.add(
                EmbeddingProfileRecord(
                    id=legacy_profile_id,
                    config_hash=config_hash,
                    provider="hash",
                    model="hash-token-v1",
                    dimensions=2,
                    schema_version="index-text-v1",
                    source_graph_run_id=graph_run_id,
                    source_corpus_hash=source_corpus_hash,
                    metadata_json={},
                    status="ready",
                    is_active=True,
                )
            )
            session.add(
                EmbeddingVectorRecord(
                    id=legacy_vector_id,
                    profile_id=legacy_profile_id,
                    kind="chunk",
                    object_id="chk_one",
                    build_run_id=graph_run_id,
                    source_content_hash="d" * 64,
                    embedding_text="One",
                    embedding_json=[1.0, 0.0],
                    source_chunk_ids_json=["chk_one"],
                    metadata_json={},
                )
            )
            session.add(
                RetrievalTraceRecord(
                    id="rtr_legacy",
                    profile_id=legacy_profile_id,
                    index_config_hash=config_hash,
                    graph_build_run_id=graph_run_id,
                    query_text="What is one?",
                    query_hash="q" * 64,
                    mode="naive",
                    retrieval_config_hash="r" * 64,
                    trace_json={"profile_id": legacy_profile_id},
                    output_json={"retrieval": {"profile_id": legacy_profile_id}},
                    model_info_json={},
                )
            )
    finally:
        legacy.dispose()

    upgrade_database(url)
    upgraded = Database(url)
    try:
        with upgraded.session_factory() as session:
            profile = session.get(EmbeddingProfileRecord, snapshot_profile_id)
            assert profile is not None
            assert session.get(EmbeddingProfileRecord, legacy_profile_id) is None
            vector = session.get(
                EmbeddingVectorRecord,
                make_vector_id(snapshot_profile_id, "chunk", "chk_one"),
            )
            assert vector is not None and vector.profile_id == snapshot_profile_id
            trace = session.get(RetrievalTraceRecord, "rtr_legacy")
            assert trace is not None and trace.profile_id == snapshot_profile_id
            assert trace.trace_json["profile_id"] == snapshot_profile_id
            assert trace.output_json == {"retrieval": {"profile_id": snapshot_profile_id}}
        unique_sets = {
            tuple(constraint["column_names"])
            for constraint in inspect(upgraded.engine).get_unique_constraints("embedding_profiles")
        }
        assert ("config_hash", "source_corpus_hash", "source_graph_run_id") in unique_sets
    finally:
        upgraded.dispose()

    _downgrade_database(url, "0003_retrieval_indexes")
    downgraded = Database(url)
    try:
        with downgraded.session_factory() as session:
            assert session.get(EmbeddingProfileRecord, legacy_profile_id) is not None
            assert session.get(EmbeddingVectorRecord, legacy_vector_id) is not None
            trace = session.get(RetrievalTraceRecord, "rtr_legacy")
            assert trace is not None and trace.profile_id == legacy_profile_id
            assert trace.trace_json["profile_id"] == legacy_profile_id
            assert trace.output_json == {"retrieval": {"profile_id": legacy_profile_id}}
        unique_sets = {
            tuple(constraint["column_names"])
            for constraint in inspect(downgraded.engine).get_unique_constraints(
                "embedding_profiles"
            )
        }
        assert ("config_hash", "source_corpus_hash") in unique_sets
    finally:
        downgraded.dispose()

    upgrade_database(url)
    migrated_again = Database(url)
    try:
        with migrated_again.session_factory() as session:
            assert session.get(EmbeddingProfileRecord, snapshot_profile_id) is not None
    finally:
        migrated_again.dispose()


def _downgrade_database(database_url: str, revision: str) -> None:
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.downgrade(config, revision)
