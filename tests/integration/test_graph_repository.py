from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from hybrid_rag.config import sqlite_url
from hybrid_rag.schemas import ChunkData, ParsedDocument
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.graph_repository import (
    GraphRepository,
    GraphRepositoryError,
    StaleExtractionLeaseError,
)
from hybrid_rag.storage.models import (
    ChunkRecord,
    DocumentRecord,
    EntityEvidenceRecord,
    EntityRecord,
    ExtractionAttemptRecord,
    RelationRecord,
)
from hybrid_rag.storage.repository import IngestRepository


def _database(tmp_path: Path) -> Database:
    database = Database(sqlite_url(tmp_path / "graph.db"))
    database.create_schema()
    with database.session_factory.begin() as session:
        session.add(
            DocumentRecord(
                id="doc_test",
                title="Test",
                source_type="txt",
                source_uri="file:test/document.txt",
                local_path="document.txt",
                content_hash="d" * 64,
                parsed_text="Alpha uses Beta. Gamma is unrelated.",
                parser_name="test",
                parser_version="1",
                processing_config_hash="p" * 64,
                metadata_json={},
            )
        )
        session.add_all(
            [
                _chunk("chk_alpha", 0, "Alpha uses Beta."),
                _chunk("chk_gamma", 1, "Gamma is unrelated."),
            ]
        )
    return database


def _chunk(chunk_id: str, ordinal: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        document_id="doc_test",
        ordinal=ordinal,
        section_path_json=[],
        char_start=0,
        char_end=len(text),
        text=text,
        contextualized_text=text,
        token_count=4,
        content_hash=("a" if ordinal == 0 else "b") * 64,
        chunker_name="test",
        chunker_version="1",
        metadata_json={},
    )


def _begin(
    repository: GraphRepository,
    database: Database,
    *,
    review_required: bool = False,
) -> tuple[str, list[str]]:
    with database.session_factory.begin() as session:
        run = repository.begin_run(
            session,
            extraction_config_hash="e" * 64,
            graph_config_hash="g" * 64,
            model="test-model",
            prompt_version="1",
            schema_version="1",
            workflow_version="1",
            review_required=review_required,
        )
        preparation = repository.prepare_jobs(session, run.id)
        jobs = repository.list_pending_jobs(session, run.id)
    assert preparation.total == preparation.scheduled == 2
    return run.id, [job["id"] for job in jobs]


def _succeed(
    repository: GraphRepository,
    database: Database,
    extraction_id: str,
    *,
    result: dict | None = None,
) -> None:
    with database.session_factory.begin() as session:
        claim = repository.claim_extraction(
            session,
            extraction_id,
            stage="extract",
            messages=[{"role": "user", "content": "Return JSON"}],
        )
    assert claim is not None
    with database.session_factory.begin() as session:
        repository.record_attempt(
            session,
            claim,
            outcome="succeeded",
            raw_response='{"entities": [], "relations": []}',
            response_metadata={"request_id": "request-test"},
            prompt_tokens=10,
            completion_tokens=2,
        )
        repository.complete_extraction(
            session,
            claim,
            result or {"entities": [], "relations": []},
        )


def test_attempts_are_audited_and_resume_reclaims_expired_lease(tmp_path: Path) -> None:
    database = _database(tmp_path)
    repository = GraphRepository()
    try:
        run_id, extraction_ids = _begin(repository, database)
        started = datetime(2026, 7, 11, tzinfo=UTC)
        with database.session_factory.begin() as session:
            first = repository.claim_extraction(
                session,
                extraction_ids[0],
                stage="extract",
                messages=[{"role": "user", "content": "JSON"}],
                lease_seconds=1,
                now=started,
            )
        assert first is not None

        with database.session_factory.begin() as session:
            second = repository.claim_extraction(
                session,
                extraction_ids[0],
                stage="repair",
                messages=[{"role": "user", "content": "Repair JSON"}],
                now=started + timedelta(seconds=2),
            )
        assert second is not None and second.attempt_ordinal == 2

        with (
            database.session_factory.begin() as session,
            pytest.raises(StaleExtractionLeaseError),
        ):
            repository.complete_extraction(session, first, {"entities": [], "relations": []})

        with database.session_factory.begin() as session:
            repository.record_attempt(
                session,
                second,
                outcome="invalid_json",
                raw_response="{",
                error="invalid JSON",
            )
            repository.requeue_extraction(
                session,
                second,
                outcome="invalid_json",
                error="invalid JSON",
            )

        with database.session_factory.begin() as session:
            third = repository.claim_extraction(
                session,
                extraction_ids[0],
                stage="repair",
                messages=[{"role": "user", "content": "Repair again"}],
            )
        assert third is not None and third.attempt_ordinal == 3
        with database.session_factory.begin() as session:
            repository.record_attempt(session, third, outcome="succeeded")
            repository.complete_extraction(session, third, {"entities": [], "relations": []})
            state = repository.finalize_run(session, run_id)
            attempts = list(
                session.scalars(
                    select(ExtractionAttemptRecord)
                    .where(ExtractionAttemptRecord.extraction_id == extraction_ids[0])
                    .order_by(ExtractionAttemptRecord.ordinal)
                )
            )

        assert [attempt.outcome for attempt in attempts] == [
            "interrupted",
            "invalid_json",
            "succeeded",
        ]
        assert state.status == "running"
        with database.session_factory() as session:
            pending = repository.list_pending_jobs(session, run_id)
        assert [job["id"] for job in pending] == [extraction_ids[1]]
    finally:
        database.dispose()


def test_overlapping_runs_charge_attempt_to_the_claiming_run_only(tmp_path: Path) -> None:
    database = _database(tmp_path)
    repository = GraphRepository()
    try:
        first_run, first_ids = _begin(repository, database)
        second_run, second_ids = _begin(repository, database)
        assert first_ids == second_ids

        with (
            database.session_factory.begin() as session,
            pytest.raises(GraphRepositoryError, match="unambiguous active run"),
        ):
            repository.claim_extraction(
                session,
                first_ids[0],
                stage="extract",
                messages=[{"role": "user", "content": "JSON"}],
            )

        with database.session_factory.begin() as session:
            claim = repository.claim_extraction(
                session,
                first_ids[0],
                run_id=first_run,
                stage="extract",
                messages=[{"role": "user", "content": "JSON"}],
            )
        assert claim is not None and claim.run_id == first_run
        with database.session_factory.begin() as session:
            repository.record_attempt(
                session,
                claim,
                outcome="succeeded",
                prompt_tokens=7,
                completion_tokens=3,
            )
            repository.complete_extraction(session, claim, {"entities": [], "relations": []})
            first = repository.finalize_run(session, first_run)
            second = repository.finalize_run(session, second_run)

        assert first.attempt_count == 1
        assert first.total_tokens == 10
        assert second.attempt_count == 0
        assert second.total_tokens == 0
    finally:
        database.dispose()


def test_review_policy_is_scoped_to_each_overlapping_run(tmp_path: Path) -> None:
    database = _database(tmp_path)
    repository = GraphRepository()
    try:
        plain_run, extraction_ids = _begin(repository, database)
        with database.session_factory.begin() as session:
            review_run = repository.begin_run(
                session,
                extraction_config_hash="e" * 64,
                graph_config_hash="g" * 64,
                model="test-model",
                prompt_version="1",
                schema_version="1",
                workflow_version="1",
                review_required=True,
            )
            repository.prepare_jobs(session, review_run.id)
            second_review_run = repository.begin_run(
                session,
                extraction_config_hash="e" * 64,
                graph_config_hash="g" * 64,
                model="test-model",
                prompt_version="1",
                schema_version="1",
                workflow_version="1",
                review_required=True,
            )
            repository.prepare_jobs(session, second_review_run.id)

        with database.session_factory.begin() as session:
            claim = repository.claim_extraction(
                session,
                extraction_ids[0],
                run_id=plain_run,
                stage="extract",
                messages=[{"role": "user", "content": "JSON"}],
            )
        assert claim is not None
        with database.session_factory.begin() as session:
            repository.record_attempt(session, claim, outcome="succeeded")
            repository.complete_extraction(session, claim, {"entities": [], "relations": []})
            plain = repository.get_run(session, plain_run)
            awaiting = repository.get_run(session, review_run.id)
            also_awaiting = repository.get_run(session, second_review_run.id)
            plain_results = repository.accepted_results(session, plain_run)
            review_results = repository.accepted_results(session, review_run.id)

        assert plain is not None and plain.succeeded_chunks == 1
        assert plain.needs_review_chunks == 0
        assert awaiting is not None and awaiting.needs_review_chunks == 1
        assert awaiting.status == "awaiting_review"
        assert also_awaiting is not None and also_awaiting.needs_review_chunks == 1
        assert len(plain_results) == 1
        assert review_results == []

        with (
            database.session_factory.begin() as session,
            pytest.raises(GraphRepositoryError, match="multiple runs"),
        ):
            repository.review_extraction(session, extraction_ids[0], decision="approve")

        with database.session_factory.begin() as session:
            reviewed = repository.review_extraction(
                session,
                extraction_ids[0],
                decision="approve",
                run_id=review_run.id,
            )
            approved = repository.get_run(session, review_run.id)
            still_awaiting = repository.get_run(session, second_review_run.id)
        assert reviewed["reviewed_run_ids"] == [review_run.id]
        assert approved is not None and approved.succeeded_chunks == 1
        assert approved.needs_review_chunks == 0
        assert approved.status == "running"
        assert still_awaiting is not None and still_awaiting.needs_review_chunks == 1
        assert still_awaiting.status == "awaiting_review"
    finally:
        database.dispose()


def test_snapshot_replace_provenance_stats_and_inspection(tmp_path: Path) -> None:
    database = _database(tmp_path)
    repository = GraphRepository()
    try:
        run_id, extraction_ids = _begin(repository, database)
        _succeed(repository, database, extraction_ids[0])
        _succeed(repository, database, extraction_ids[1])
        with database.session_factory.begin() as session:
            first_stats = repository.replace_snapshot(
                session,
                run_id,
                entities=[
                    {
                        "id": "ent_alpha",
                        "canonical_name": "Alpha",
                        "normalized_name": "alpha",
                        "entity_type": "METHOD",
                        "description": "Alpha method",
                        "aliases": ["A", "Alpha"],
                        "source_chunk_ids": ["chk_alpha"],
                        "evidence": [
                            {
                                "source_chunk_id": "chk_alpha",
                                "quote": "Alpha",
                                "char_start": 0,
                                "char_end": 5,
                            }
                        ],
                    },
                    {
                        "id": "ent_beta",
                        "canonical_name": "Beta",
                        "entity_type": "METHOD",
                        "description": "Beta method",
                        "aliases": [],
                        "source_chunk_ids": ["chk_alpha"],
                        "evidence": [
                            {
                                "source_chunk_id": "chk_alpha",
                                "quote": "Beta",
                                "char_start": 11,
                                "char_end": 15,
                            }
                        ],
                    },
                ],
                relations=[
                    {
                        "id": "rel_uses",
                        "source_entity_id": "ent_alpha",
                        "target_entity_id": "ent_beta",
                        "predicate": "USES",
                        "description": "Alpha uses Beta",
                        "source_chunk_ids": ["chk_alpha"],
                        "evidence": [
                            {
                                "source_chunk_id": "chk_alpha",
                                "quote": "Alpha uses Beta.",
                                "char_start": 0,
                                "char_end": 16,
                            }
                        ],
                    }
                ],
                component_count=1,
                largest_component_nodes=2,
            )
            final = repository.finalize_run(session, run_id)
            entity_inspect = repository.inspect(session, "ent_alpha")
            relation_inspect = repository.inspect(session, "rel_uses")

        assert first_stats["nodes"] == 2 and first_stats["edges"] == 1
        assert first_stats["top_entities"][0]["degree"] == 1
        assert final.status == "completed"
        assert entity_inspect is not None
        assert entity_inspect["source_chunk_ids"] == ["chk_alpha"]
        assert relation_inspect is not None
        assert relation_inspect["evidence"][0]["extraction_id"] == extraction_ids[0]

        with (
            pytest.raises(GraphRepositoryError),
            database.session_factory.begin() as session,
        ):
            repository.replace_snapshot(
                session,
                run_id,
                entities=[],
                relations=[
                    {
                        "id": "rel_invalid",
                        "source_entity_id": "ent_missing",
                        "target_entity_id": "ent_missing",
                        "predicate": "USES",
                        "description": "Invalid",
                        "source_chunk_ids": ["chk_alpha"],
                        "evidence": [],
                    }
                ],
            )
        with database.session_factory() as session:
            assert session.get(EntityRecord, "ent_alpha") is not None
            assert session.get(RelationRecord, "rel_uses") is not None

        with database.session_factory.begin() as session:
            second_stats = repository.replace_snapshot(
                session,
                run_id,
                entities=[
                    {
                        "id": "ent_gamma",
                        "canonical_name": "Gamma",
                        "entity_type": "CONCEPT",
                        "description": "Gamma",
                        "aliases": [],
                        "source_chunk_ids": ["chk_gamma"],
                        "evidence": [
                            {
                                "source_chunk_id": "chk_gamma",
                                "quote": "Gamma",
                                "char_start": 0,
                                "char_end": 5,
                            }
                        ],
                    }
                ],
                relations=[],
                component_count=1,
                largest_component_nodes=1,
                isolated_entity_count=1,
            )
        assert second_stats["nodes"] == 1 and second_stats["edges"] == 0
        with database.session_factory() as session:
            assert session.get(EntityRecord, "ent_alpha") is None
            assert session.get(RelationRecord, "rel_uses") is None
            assert session.scalar(select(func.count()).select_from(EntityEvidenceRecord)) == 1
    finally:
        database.dispose()


def test_validated_result_can_pause_for_review_and_resume(tmp_path: Path) -> None:
    database = _database(tmp_path)
    repository = GraphRepository()
    try:
        run_id, extraction_ids = _begin(repository, database, review_required=True)
        with database.session_factory.begin() as session:
            claim = repository.claim_extraction(
                session,
                extraction_ids[0],
                stage="extract",
                messages=[{"role": "user", "content": "JSON"}],
            )
        assert claim is not None
        payload = {"entities": [], "relations": []}
        with database.session_factory.begin() as session:
            repository.record_attempt(session, claim, outcome="succeeded")
            paused = repository.complete_extraction(session, claim, payload, needs_review=True)
            run = repository.get_run(session, run_id)
        assert paused["status"] == "succeeded"
        assert paused["result"] == payload
        assert run is not None and run.status == "awaiting_review"

        with database.session_factory.begin() as session:
            approved = repository.review_extraction(
                session, extraction_ids[0], decision="approve", notes="verified"
            )
            resumed = repository.get_run(session, run_id)
        assert approved["status"] == "succeeded"
        assert approved["review_status"] == "approved"
        assert resumed is not None and resumed.status == "running"
    finally:
        database.dispose()


def test_document_change_invalidation_removes_current_snapshot(tmp_path: Path) -> None:
    database = _database(tmp_path)
    repository = GraphRepository()
    try:
        run_id, extraction_ids = _begin(repository, database)
        for extraction_id in extraction_ids:
            _succeed(repository, database, extraction_id)
        with database.session_factory.begin() as session:
            repository.replace_snapshot(
                session,
                run_id,
                entities=[
                    {
                        "id": "ent_alpha",
                        "canonical_name": "Alpha",
                        "entity_type": "METHOD",
                        "description": "Alpha",
                        "aliases": [],
                        "source_chunk_ids": ["chk_alpha"],
                        "evidence": [
                            {
                                "source_chunk_id": "chk_alpha",
                                "quote": "Alpha",
                                "char_start": 0,
                                "char_end": 5,
                            }
                        ],
                    }
                ],
                relations=[],
            )
            result = IngestRepository().upsert_document(
                session,
                ParsedDocument(
                    id="doc_test",
                    title="Changed",
                    source_type="txt",
                    source_uri="file:test/document.txt",
                    local_path="document.txt",
                    content_hash="n" * 64,
                    text="Replacement text.",
                    parser_name="test",
                    parser_version="1",
                ),
                [
                    ChunkData(
                        id="chk_replacement",
                        document_id="doc_test",
                        ordinal=0,
                        char_start=0,
                        char_end=17,
                        text="Replacement text.",
                        contextualized_text="Replacement text.",
                        token_count=2,
                        content_hash="r" * 64,
                        chunker_name="test",
                        chunker_version="1",
                    )
                ],
                "p" * 64,
            )
        assert result.status == "updated"
        with database.session_factory() as session:
            assert repository.get_run(session, run_id).status == "failed"  # type: ignore[union-attr]
            assert session.scalar(select(func.count()).select_from(EntityRecord)) == 0
            assert session.scalar(select(func.count()).select_from(ExtractionAttemptRecord)) == 0
        with (
            pytest.raises(GraphRepositoryError, match="cannot resume"),
            database.session_factory.begin() as session,
        ):
            repository.prepare_jobs(session, run_id)
        with (
            pytest.raises(GraphRepositoryError, match="cannot be reopened"),
            database.session_factory.begin() as session,
        ):
            repository.finalize_run(session, run_id, status="running")
    finally:
        database.dispose()
