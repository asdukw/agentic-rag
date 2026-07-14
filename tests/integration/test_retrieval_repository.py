from __future__ import annotations

from sqlalchemy.orm import Session

from hybrid_rag.ids import sha256_text
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.models import (ChunkExtractionRecord, ChunkRecord,
                                       DocumentRecord, EntityEvidenceRecord,
                                       EntityRecord, GraphBuildRunRecord,
                                       RelationEvidenceRecord, RelationRecord)
from hybrid_rag.storage.retrieval_repository import (IndexItem, IndexProfile,
                                                     RetrievalRepository,
                                                     RetrievalTrace,
                                                     make_profile_id)


def test_rebuild_load_snapshot_and_replay_trace() -> None:
    database = Database("sqlite:///:memory:")
    database.create_schema()
    repository = RetrievalRepository()
    try:
        with database.session_factory.begin() as session:
            _seed_graph_snapshot(session)
            snapshot = repository.load_source_snapshot(session)

            assert snapshot.build_run_id == "gbr_graph"
            assert len(snapshot.corpus_content_hash) == 64
            assert [chunk.id for chunk in snapshot.chunks] == ["chk_one"]
            assert [entity.id for entity in snapshot.entities] == ["ent_alpha", "ent_beta"]
            assert [relation.id for relation in snapshot.relations] == ["rel_alpha_beta"]
            assert snapshot.entity_evidence[0].quote == "Alpha is connected to Beta."
            assert snapshot.relation_evidence[0].owner_kind == "relation"

            profile = IndexProfile(
                config_hash="a" * 64,
                provider="deterministic-hash",
                model="hash-v1",
                dimensions=3,
                schema_version="1",
                source_corpus_hash=snapshot.source_corpus_hash,
                source_graph_run_id=snapshot.build_run_id,
                metadata={"normalizer": "l2"},
            )
            stored = repository.replace_index(
                session,
                profile,
                [
                    IndexItem(
                        object_id="chk_one",
                        kind="chunk",
                        embedding_text="Alpha is connected to Beta.",
                        embedding=(1.0, 0.0, 0.0),
                        source_chunk_ids=("chk_one",),
                        source_content_hash="c" * 64,
                    ),
                    IndexItem(
                        object_id="ent_alpha",
                        kind="entity",
                        embedding_text="Alpha concept",
                        embedding=(0.0, 1.0, 0.0),
                        source_chunk_ids=("chk_one",),
                        source_content_hash=sha256_text("Alpha concept"),
                    ),
                    IndexItem(
                        object_id="rel_alpha_beta",
                        kind="relation",
                        embedding_text="Alpha CONNECTS Beta",
                        embedding=(0.0, 0.0, 1.0),
                        source_chunk_ids=("chk_one",),
                        source_content_hash=sha256_text("Alpha CONNECTS Beta"),
                    ),
                ],
            )
            assert stored.id.startswith("idx_")
            assert stored.is_active is True

            loaded = repository.load_index(session)
            assert loaded.profile.id == stored.id
            assert [item.object_id for item in loaded.chunks] == ["chk_one"]
            assert [item.object_id for item in loaded.entities] == ["ent_alpha"]
            assert [item.object_id for item in loaded.relations] == ["rel_alpha_beta"]
            assert loaded.chunks[0].id is not None and loaded.chunks[0].id.startswith("vec_")

            trace = repository.save_trace(
                session,
                RetrievalTrace(
                    profile_id=stored.id,
                    index_config_hash=stored.config_hash,
                    query_text="How are Alpha and Beta connected?",
                    mode="hybrid",
                    retrieval_config_hash="b" * 64,
                    trace_json={"context_chunk_ids": ["chk_one"], "scores": {"chk_one": 1.0}},
                    output_json={"answer": "Alpha connects to Beta.", "citations": ["chk_one"]},
                    model_info={"answer_model": "offline"},
                ),
            )
            assert trace.id.startswith("rtr_")

        with database.session_factory() as session:
            replay = repository.load_trace(session, trace.id)
            assert replay is not None
            assert replay.trace_json["context_chunk_ids"] == ["chk_one"]
            assert replay.output_json == {
                "answer": "Alpha connects to Beta.",
                "citations": ["chk_one"],
            }
            assert replay.model_info == {"answer_model": "offline"}
    finally:
        database.dispose()


def test_replace_index_replaces_vectors_and_switches_active_profile() -> None:
    database = Database("sqlite:///:memory:")
    database.create_schema()
    repository = RetrievalRepository()
    try:
        with database.session_factory.begin() as session:
            first = repository.replace_index(
                session,
                _profile("a" * 64, "c" * 64),
                [_item("chk_one", (1.0, 0.0))],
            )
            repository.replace_index(
                session,
                _profile("b" * 64, "c" * 64),
                [_item("chk_two", (0.0, 1.0))],
            )
            assert repository.get_profile(session, first.id).is_active is False  # type: ignore[union-attr]

            active = repository.load_index(session)
            assert active.profile.config_hash == "b" * 64
            assert [item.object_id for item in active.chunks] == ["chk_two"]

            repository.replace_index(
                session,
                _profile("b" * 64, "c" * 64),
                [_item("chk_three", (0.5, 0.5))],
            )
            rebuilt = repository.load_index(session, active.profile.id)
            assert [item.object_id for item in rebuilt.chunks] == ["chk_three"]
    finally:
        database.dispose()


def test_embedding_profiles_are_distinct_for_different_graph_snapshot_runs() -> None:
    """A config/corpus pair must not overwrite vectors from another graph run."""

    database = Database("sqlite:///:memory:")
    database.create_schema()
    repository = RetrievalRepository()
    config_hash = "a" * 64
    source_corpus_hash = "c" * 64
    try:
        with database.session_factory.begin() as session:
            _seed_graph_snapshot(session)
            session.add(
                GraphBuildRunRecord(
                    id="gbr_graph_alternate",
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
            first_snapshot = repository.load_source_snapshot(session, build_run_id="gbr_graph")
            second_snapshot = repository.load_source_snapshot(
                session,
                build_run_id="gbr_graph_alternate",
            )
            assert first_snapshot.corpus_content_hash == second_snapshot.corpus_content_hash
            assert first_snapshot.source_corpus_hash != second_snapshot.source_corpus_hash
            first = repository.replace_index(
                session,
                _profile(config_hash, source_corpus_hash, "gbr_graph"),
                [_item("chk_one", (1.0, 0.0))],
            )
            second = repository.replace_index(
                session,
                _profile(config_hash, source_corpus_hash, "gbr_graph_alternate"),
                [_item("chk_one", (0.0, 1.0))],
            )
            rebuilt = repository.replace_index(
                session,
                _profile(config_hash, source_corpus_hash, "gbr_graph"),
                [_item("chk_one", (0.5, 0.5))],
            )

            assert first.id == make_profile_id(config_hash, source_corpus_hash, "gbr_graph")
            assert second.id == make_profile_id(
                config_hash,
                source_corpus_hash,
                "gbr_graph_alternate",
            )
            assert first.id != second.id
            assert rebuilt.id == first.id
            assert make_profile_id(config_hash, source_corpus_hash) != first.id

            profiles = repository.list_profiles(session)
            assert {profile.id for profile in profiles} == {first.id, second.id}
            assert {profile.source_graph_run_id for profile in profiles} == {
                "gbr_graph",
                "gbr_graph_alternate",
            }
            assert repository.load_index(session, first.id).chunks[0].embedding == (0.5, 0.5)
            assert repository.load_index(session, second.id).chunks[0].embedding == (0.0, 1.0)
    finally:
        database.dispose()


def _profile(
    config_hash: str,
    source_corpus_hash: str,
    source_graph_run_id: str | None = None,
) -> IndexProfile:
    return IndexProfile(
        config_hash=config_hash,
        provider="deterministic-hash",
        model="hash-v1",
        dimensions=2,
        schema_version="1",
        source_corpus_hash=source_corpus_hash,
        source_graph_run_id=source_graph_run_id,
    )


def _item(object_id: str, embedding: tuple[float, float]) -> IndexItem:
    return IndexItem(
        object_id=object_id,
        kind="chunk",
        embedding_text=object_id,
        embedding=embedding,
        source_content_hash=sha256_text(object_id),
    )


def _seed_graph_snapshot(session: Session) -> None:
    add = session.add
    add(
        DocumentRecord(
            id="doc_one",
            title="One",
            source_type="txt",
            source_uri="file:test/one.txt",
            local_path="one.txt",
            content_hash="d" * 64,
            parsed_text="Alpha is connected to Beta.",
            parser_name="test",
            parser_version="1",
            processing_config_hash="p" * 64,
            metadata_json={},
        )
    )
    add(
        ChunkRecord(
            id="chk_one",
            document_id="doc_one",
            ordinal=0,
            section_path_json=["Overview"],
            page_start=None,
            page_end=None,
            char_start=0,
            char_end=28,
            text="Alpha is connected to Beta.",
            contextualized_text="[One] Alpha is connected to Beta.",
            token_count=6,
            content_hash="c" * 64,
            chunker_name="test",
            chunker_version="1",
            metadata_json={},
        )
    )
    add(
        GraphBuildRunRecord(
            id="gbr_graph",
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
    add(
        ChunkExtractionRecord(
            id="xtr_one",
            chunk_id="chk_one",
            extraction_config_hash="e" * 64,
            model="test",
            prompt_version="1",
            schema_version="1",
            status="succeeded",
            result_json={},
        )
    )
    add(
        EntityRecord(
            id="ent_alpha",
            build_run_id="gbr_graph",
            graph_config_hash="g" * 64,
            canonical_name="Alpha",
            normalized_name="alpha",
            entity_type="CONCEPT",
            description="Alpha concept",
            aliases_json=[],
            source_chunk_ids_json=["chk_one"],
        )
    )
    add(
        EntityRecord(
            id="ent_beta",
            build_run_id="gbr_graph",
            graph_config_hash="g" * 64,
            canonical_name="Beta",
            normalized_name="beta",
            entity_type="CONCEPT",
            description="Beta concept",
            aliases_json=[],
            source_chunk_ids_json=["chk_one"],
        )
    )
    add(
        RelationRecord(
            id="rel_alpha_beta",
            build_run_id="gbr_graph",
            graph_config_hash="g" * 64,
            source_entity_id="ent_alpha",
            target_entity_id="ent_beta",
            predicate="CONNECTS",
            description="Alpha connects to Beta",
            source_chunk_ids_json=["chk_one"],
        )
    )
    add(
        EntityEvidenceRecord(
            id="ene_alpha",
            entity_id="ent_alpha",
            chunk_id="chk_one",
            extraction_id="xtr_one",
            mention_id="men_alpha",
            quote="Alpha is connected to Beta.",
            char_start=0,
            char_end=28,
        )
    )
    add(
        RelationEvidenceRecord(
            id="rne_alpha_beta",
            relation_id="rel_alpha_beta",
            chunk_id="chk_one",
            extraction_id="xtr_one",
            mention_id="men_relation",
            quote="Alpha is connected to Beta.",
            char_start=0,
            char_end=28,
        )
    )
