from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from hybrid_rag.config import sqlite_url
from hybrid_rag.extraction.client import CompletionResult
from hybrid_rag.extraction.schemas import ExtractionConfig
from hybrid_rag.extraction.service import GraphBuildService
from hybrid_rag.extraction.workflow import WorkflowOptions
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.retrieval.embedding import HashEmbeddingProvider
from hybrid_rag.retrieval.models import RetrievalMode, RetrievalResult
from hybrid_rag.retrieval.service import RetrievalOptions, RetrievalService
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database
from hybrid_rag.storage.retrieval_repository import (
    RetrievalRepository,
    RetrievalRepositoryError,
)


class WordCounter:
    name = "test-words"

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


class ScriptedGraphClient:
    """Offline graph client whose evidence exactly occurs in the ingested corpus."""

    quote = "Atlas connects Beacon through the graph."

    async def extract(self, chunk_text: str, **_: object) -> CompletionResult:
        assert self.quote in chunk_text
        return _completion(_graph_payload(self.quote))

    async def repair(self, chunk_text: str, **_: object) -> CompletionResult:
        assert self.quote in chunk_text
        return _completion(_graph_payload(self.quote))


def test_three_indexes_support_all_retrieval_modes_replay_and_grounded_answer(
    tmp_path: Path,
) -> None:
    database = _ingested_scripted_graph(tmp_path)
    retrieval = RetrievalService(
        database,
        HashEmbeddingProvider(dimensions=256),
        WordCounter(),
    )
    options = RetrievalOptions(
        top_k=3,
        candidate_multiplier=2,
        context_token_budget=40,
        graph_max_hops=2,
    )
    question = "How does Atlas connect Beacon in the graph?"

    try:
        built = retrieval.build_index()
        reused = retrieval.build_index()

        assert (built.chunks, built.entities, built.relations) == (1, 2, 1)
        assert not built.reused
        assert reused.reused
        assert reused.profile_id == built.profile_id
        with database.session_factory.begin() as session:
            RetrievalRepository().update_profile_metadata(
                session,
                built.profile_id,
                {"tokenizer": "test-words"},
            )
        refreshed = retrieval.build_index()
        assert refreshed.reused
        with database.session_factory() as session:
            snapshot = RetrievalRepository().load_source_snapshot(session)
            profile = RetrievalRepository().get_profile(session, built.profile_id)
        assert profile is not None
        assert profile.metadata["corpus_content_hash"] == snapshot.corpus_content_hash
        assert profile.metadata["graph_corpus_hash"] == snapshot.graph_corpus_hash

        results = {
            mode: retrieval.retrieve(
                question,
                mode=mode,
                options=options,
                keywords=("Atlas", "Beacon", "graph"),
            )
            for mode in RetrievalMode
        }

        for mode, result in results.items():
            _assert_shared_result_contract(result, mode, options.context_token_budget)
            expected_routes = (
                {"naive", "local", "global"}
                if mode is RetrievalMode.HYBRID
                else {mode.value}
            )
            assert set(result.trace.routes) == expected_routes

        naive = results[RetrievalMode.NAIVE]
        local = results[RetrievalMode.LOCAL]
        global_result = results[RetrievalMode.GLOBAL]
        hybrid = results[RetrievalMode.HYBRID]

        assert naive.graph_paths == ()
        assert local.graph_paths and global_result.graph_paths and hybrid.graph_paths
        assert all(path.relation_ids for path in hybrid.graph_paths)
        assert all(path.source_chunk_ids for path in hybrid.graph_paths)
        assert set(hybrid.hits[0].route_scores) == {"naive", "local", "global"}
        assert all(item.citation_id == item.chunk_id for item in hybrid.context_items)
        assert all(item.chunk_id in hybrid.context for item in hybrid.context_items)
        assert hybrid.trace.context_tokens <= hybrid.trace.context_token_budget
        assert json.loads(hybrid.trace.model_dump_json())["mode"] == "hybrid"

        assert hybrid.trace_id is not None
        assert retrieval.replay(hybrid.trace_id) == hybrid

        answer_result = asyncio.run(
            retrieval.ask(question, mode=RetrievalMode.HYBRID, options=options)
        )
        assert answer_result.retrieval.trace_id is not None
        assert answer_result.answer.citations == (
            answer_result.retrieval.context_items[0].citation_id,
        )
        assert not answer_result.answer.insufficient_evidence
        assert retrieval.replay_answer(answer_result.retrieval.trace_id) == answer_result
    finally:
        database.dispose()


def test_changed_source_document_invalidates_its_active_retrieval_index(
    tmp_path: Path,
) -> None:
    database = _ingested_scripted_graph(tmp_path)
    retrieval = RetrievalService(
        database,
        HashEmbeddingProvider(dimensions=256),
        WordCounter(),
    )
    options = RetrievalOptions(context_token_budget=40)
    question = "How does Atlas connect Beacon in the graph?"

    try:
        profile = retrieval.build_index()
        original = retrieval.retrieve(question, mode=RetrievalMode.HYBRID, options=options)
        assert original.trace_id is not None

        corpus = tmp_path / "corpus"
        (corpus / "atlas.txt").write_text(
            "Atlas now connects Beacon through a revised graph. "
            "The revised relationship has new evidence.",
            encoding="utf-8",
        )
        changed = IngestionService(
            database,
            SectionTokenChunker(WordCounter(), max_tokens=64, overlap_tokens=5),
        ).ingest(corpus)
        assert changed.failed == 0
        assert changed.updated == 1

        repository = RetrievalRepository()
        with database.session_factory() as session:
            invalidated = next(
                value
                for value in repository.list_profiles(session)
                if value.id == profile.profile_id
            )
            assert invalidated.status == "failed"
            assert not invalidated.is_active
            assert invalidated.error == "source document changed after retrieval indexing"
            with pytest.raises(RetrievalRepositoryError, match="no ready embedding index"):
                repository.load_index(session)
            with pytest.raises(RetrievalRepositoryError, match=profile.profile_id):
                repository.load_index(session, profile.profile_id)

        replayed = retrieval.replay(original.trace_id)
        assert replayed == original
    finally:
        database.dispose()


def _assert_shared_result_contract(
    result: RetrievalResult,
    mode: RetrievalMode,
    context_budget: int,
) -> None:
    assert result.mode is mode
    assert result.trace.mode is mode
    assert result.hits
    assert result.context_items
    assert result.context
    assert result.trace_id is not None
    assert result.context_tokens <= context_budget
    assert result.trace.context_tokens == result.context_tokens
    assert result.trace.context_token_budget == context_budget
    assert tuple(item.citation_id for item in result.context_items)


def _ingested_scripted_graph(tmp_path: Path) -> Database:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "atlas.txt").write_text(
        "Atlas connects Beacon through the graph. "
        "The relationship supports evidence-grounded retrieval with citations.",
        encoding="utf-8",
    )
    database_url = sqlite_url(tmp_path / "retrieval.db")
    upgrade_database(database_url)
    database = Database(database_url)
    ingestion = IngestionService(
        database,
        SectionTokenChunker(WordCounter(), max_tokens=64, overlap_tokens=5),
    )
    report = ingestion.ingest(corpus)
    assert report.failed == 0
    assert report.chunks_written == 1

    graph = GraphBuildService(
        database,
        ScriptedGraphClient(),
        ExtractionConfig(
            base_url="https://example.test",
            model="scripted-graph",
            max_output_tokens=512,
            repair_max_attempts=0,
        ),
        checkpoint_path=tmp_path / "graph-checkpoint.db",
    )
    built = asyncio.run(
        graph.build(
            WorkflowOptions(
                max_concurrency=1,
                max_attempts=1,
                retry_backoff_seconds=0,
            )
        )
    )
    assert built.status == "completed"
    assert (built.graph.nodes, built.graph.edges) == (2, 1)
    return database


def _completion(content: str) -> CompletionResult:
    return CompletionResult(
        provider_request_id="request-retrieval-test",
        model="scripted-graph",
        system_fingerprint=None,
        content=content,
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        cache_hit_tokens=None,
        cache_miss_tokens=None,
        raw_response={},
    )


def _graph_payload(quote: str) -> str:
    return json.dumps(
        {
            "entities": [
                {
                    "ref": "e1",
                    "name": "Atlas",
                    "entity_type": "CONCEPT",
                    "description": "A source graph concept.",
                    "aliases": [],
                    "evidence_quotes": [quote],
                },
                {
                    "ref": "e2",
                    "name": "Beacon",
                    "entity_type": "CONCEPT",
                    "description": "A target graph concept.",
                    "aliases": [],
                    "evidence_quotes": [quote],
                },
            ],
            "relations": [
                {
                    "source_ref": "e1",
                    "target_ref": "e2",
                    "predicate": "CONNECTS",
                    "description": "Atlas connects Beacon through the graph.",
                    "evidence_quotes": [quote],
                }
            ],
        }
    )
