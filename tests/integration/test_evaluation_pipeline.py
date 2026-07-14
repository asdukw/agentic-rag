from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from hybrid_rag.config import sqlite_url
from hybrid_rag.evaluation import (
    EvaluationOptions,
    EvaluationRunner,
    fixture_benchmark_path,
    load_benchmark,
)
from hybrid_rag.extraction.client import CompletionResult
from hybrid_rag.extraction.schemas import ExtractionConfig
from hybrid_rag.extraction.service import GraphBuildService
from hybrid_rag.extraction.workflow import WorkflowOptions
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.ingest.tokenizer import TiktokenCounter
from hybrid_rag.retrieval.embedding import HashEmbeddingProvider
from hybrid_rag.retrieval.models import RetrievalMode
from hybrid_rag.retrieval.service import RetrievalService
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database


class WordCounter:
    name = "test-words"

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


class ScriptedGraphClient:
    quote = (
        "The graph provides an additional route to evidence while the original chunks remain "
        "the source of truth."
    )

    async def extract(self, chunk_text: str, **_: object) -> CompletionResult:
        assert self.quote in chunk_text
        return _completion(_graph_payload(self.quote))

    async def repair(self, chunk_text: str, **_: object) -> CompletionResult:
        assert self.quote in chunk_text
        return _completion(_graph_payload(self.quote))


def test_runner_uses_the_real_sqlite_graph_and_retrieval_pipeline(tmp_path: Path) -> None:
    database = _database_with_scripted_graph(tmp_path)
    retrieval = RetrievalService(database, HashEmbeddingProvider(dimensions=128), WordCounter())
    benchmark = load_benchmark(fixture_benchmark_path())
    options = EvaluationOptions(
        case_ids=("fact-naive-isolated-chunks", "fact-graph-evidence-route"),
        top_k=3,
        candidate_multiplier=2,
        context_token_budget=128,
    )
    try:
        index = retrieval.build_index()
        profile = retrieval.resolve_profile(index.profile_id)
        pinned_benchmark = benchmark.model_copy(
            update={"expected_source_corpus_hash": profile.metadata["corpus_content_hash"]}
        )
        report = EvaluationRunner(retrieval).run(pinned_benchmark, options=options)

        assert index.chunks == 1
        assert (index.entities, index.relations) == (2, 1)
        assert report.run.case_ids == options.case_ids
        assert report.run.index_provenance.profile_id == index.profile_id
        assert (
            report.run.index_provenance.corpus_content_hash
            == profile.metadata["corpus_content_hash"]
        )
        assert report.run.index_provenance.source_corpus_hash == index.source_corpus_hash
        assert len(report.evaluations) == 4
        assert report.cost_disclosure.cost_cny == 0.0
        hybrid = [item for item in report.evaluations if item.mode is RetrievalMode.HYBRID]
        assert all(item.evidence_hit_rate == 1.0 for item in hybrid)
        assert all(item.citation_grounded_faithfulness for item in hybrid)
        assert all(set(item.retrieval.routes) == {"local", "global"} for item in hybrid)
        assert all(item.retrieval.graph_paths for item in hybrid)
        assert all(item.retrieval_trace_id.startswith("rtr_") for item in report.evaluations)
        for measurement in report.evaluations:
            replayed = retrieval.replay(measurement.retrieval_trace_id)
            assert replayed.trace_id == measurement.retrieval_trace_id
            assert replayed.trace == measurement.retrieval
        assert len(report.pairwise_judgments) == 2
    finally:
        database.dispose()


def test_fixed_benchmark_hash_matches_the_default_fixture_corpus_profile(tmp_path: Path) -> None:
    database_url = sqlite_url(tmp_path / "fixture-content-hash.db")
    upgrade_database(database_url)
    database = Database(database_url)
    try:
        counter = TiktokenCounter("cl100k_base")
        ingestion = IngestionService(
            database,
            SectionTokenChunker(counter, max_tokens=512, overlap_tokens=64),
        )
        report = ingestion.ingest(Path(__file__).parents[1] / "fixtures" / "corpus")
        assert report.failed == 0
        retrieval = RetrievalService(database, HashEmbeddingProvider(), counter)
        index = retrieval.build_index()
        profile = retrieval.resolve_profile(index.profile_id)
        benchmark = load_benchmark(fixture_benchmark_path())

        assert profile.metadata["corpus_content_hash"] == benchmark.expected_source_corpus_hash
    finally:
        database.dispose()


def _database_with_scripted_graph(tmp_path: Path) -> Database:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "evaluation.txt"
    source.write_text(
        "Naive retrieval-augmented generation usually retrieves isolated text chunks. "
        "The graph provides an additional route to evidence while the original chunks remain "
        "the source of truth.",
        encoding="utf-8",
    )
    database_url = sqlite_url(tmp_path / "evaluation.db")
    upgrade_database(database_url)
    database = Database(database_url)
    ingestion = IngestionService(
        database,
        SectionTokenChunker(WordCounter(), max_tokens=128, overlap_tokens=0),
    )
    report = ingestion.ingest(corpus)
    assert (report.failed, report.chunks_written) == (0, 1)
    graph = GraphBuildService(
        database,
        ScriptedGraphClient(),
        ExtractionConfig(
            base_url="https://example.test",
            model="scripted-evaluation",
            max_output_tokens=512,
            repair_max_attempts=0,
        ),
        checkpoint_path=tmp_path / "evaluation-checkpoint.db",
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
        provider_request_id="request-evaluation-test",
        model="scripted-evaluation",
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
                    "name": "Naive Retrieval",
                    "entity_type": "METHOD",
                    "description": "A chunk retrieval baseline.",
                    "aliases": [],
                    "evidence_quotes": [quote],
                },
                {
                    "ref": "e2",
                    "name": "Graph Evidence",
                    "entity_type": "CONCEPT",
                    "description": "An additional evidence route.",
                    "aliases": [],
                    "evidence_quotes": [quote],
                },
            ],
            "relations": [
                {
                    "source_ref": "e1",
                    "target_ref": "e2",
                    "predicate": "COMPLEMENTS",
                    "description": "Graph evidence complements naive retrieval.",
                    "evidence_quotes": [quote],
                }
            ],
        }
    )
