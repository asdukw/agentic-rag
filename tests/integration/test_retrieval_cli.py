from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import hybrid_rag.cli as cli_module
from hybrid_rag.cli import app
from hybrid_rag.config import sqlite_url
from hybrid_rag.extraction.client import CompletionResult
from hybrid_rag.extraction.schemas import ExtractionConfig
from hybrid_rag.extraction.service import GraphBuildService
from hybrid_rag.extraction.workflow import WorkflowOptions
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database
from hybrid_rag.storage.retrieval_repository import RetrievalRepository

EVIDENCE = "Atlas connects Beacon through the graph."


class WordCounter:
    name = "test-words"

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


class ScriptedGraphClient:
    async def extract(self, chunk_text: str, **_: object) -> CompletionResult:
        assert EVIDENCE in chunk_text
        return _completion()

    async def repair(self, chunk_text: str, **_: object) -> CompletionResult:
        assert EVIDENCE in chunk_text
        return _completion()


def test_retrieval_cli_builds_queries_answers_and_replays_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = _seed_scripted_graph(tmp_path)
    _patch_offline_retrieval(monkeypatch)
    runner = CliRunner()
    question = "How does Atlas connect Beacon through the graph?"

    index = _invoke_json(
        runner,
        ["build-index", "--db", str(db_path), "--json"],
    )
    assert index["chunks"] == 1
    assert index["entities"] == 2
    assert index["relations"] == 1
    assert index["provider"] == "hash"

    retrieved = _invoke_json(
        runner,
        [
            "retrieve",
            question,
            "--db",
            str(db_path),
            "--mode",
            "hybrid",
            "--json",
        ],
    )
    assert retrieved["mode"] == "hybrid"
    assert retrieved["trace_id"].startswith("rtr_")
    assert retrieved["context_items"]
    assert set(retrieved["trace"]["routes"]) == {"naive", "local", "global"}
    assert retrieved["trace"]["settings"]["naive_dense_weight"] == 0.25
    assert retrieved["trace"]["settings"]["naive_bm25_weight"] == 1.5
    assert retrieved["trace"]["settings"]["bm25_k1"] == 1.7
    assert retrieved["trace"]["settings"]["bm25_b"] == 0.3
    assert retrieved["trace"]["settings"]["rerank_enabled"] is True
    assert retrieved["trace"]["settings"]["reranker_provider"] == "lexical"
    assert retrieved["trace"]["settings"]["rerank_candidate_multiplier"] == 4
    assert retrieved["trace"]["rerank"]["provider"] == "lexical"
    citation_id = retrieved["context_items"][0]["citation_id"]
    assert citation_id == retrieved["context_items"][0]["chunk_id"]

    answered = _invoke_json(
        runner,
        [
            "ask",
            question,
            "--db",
            str(db_path),
            "--mode",
            "hybrid",
            "--json",
        ],
    )
    answer_trace_id = answered["retrieval"]["trace_id"]
    assert answer_trace_id.startswith("rtr_")
    assert answered["retrieval"]["mode"] == "hybrid"
    assert answered["answer"]["citations"] == [
        answered["retrieval"]["context_items"][0]["citation_id"]
    ]
    assert answered["answer"]["insufficient_evidence"] is False

    replay = _invoke_json(
        runner,
        ["retrieval", "replay", answer_trace_id, "--db", str(db_path), "--json"],
    )
    assert replay["retrieval"]["trace_id"] == answer_trace_id
    assert replay["retrieval"]["mode"] == "hybrid"
    assert replay["answer"]["citations"] == answered["answer"]["citations"]


def test_evaluate_cli_writes_offline_artifacts_and_discloses_zero_model_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = _seed_scripted_graph(tmp_path)
    _patch_offline_retrieval(monkeypatch)
    runner = CliRunner()
    index = _invoke_json(runner, ["build-index", "--db", str(db_path), "--json"])

    output_dir = tmp_path / "evaluation-reports"
    fixture_path = Path(__file__).parents[2] / "data" / "evaluation" / "fixture-benchmark-v1.json"
    mismatch = runner.invoke(
        app,
        [
            "evaluate",
            "--benchmark",
            str(fixture_path),
            "--db",
            str(db_path),
            "--limit",
            "2",
            "--json",
        ],
    )
    assert mismatch.exit_code == 1
    assert "expected_source_corpus_hash" in mismatch.output
    corpus_content_hash = _profile_corpus_content_hash(db_path, index["profile_id"])
    benchmark_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    benchmark_payload["expected_source_corpus_hash"] = corpus_content_hash
    benchmark_path = tmp_path / "pinned-benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_payload), encoding="utf-8")
    arguments = [
        "evaluate",
        "--benchmark",
        str(benchmark_path),
        "--db",
        str(db_path),
        "--profile",
        index["profile_id"],
        "--output-dir",
        str(output_dir),
        "--limit",
        "2",
        "--json",
    ]
    report = _invoke_json(
        runner,
        arguments,
    )

    assert report["run"]["benchmark_id"] == "fixture-rag-v1"
    assert report["run"]["options"]["modes"] == ["naive", "hybrid"]
    assert report["run"]["options"]["naive_dense_weight"] == 0.25
    assert report["run"]["options"]["naive_bm25_weight"] == 1.5
    assert report["run"]["options"]["bm25_k1"] == 1.7
    assert report["run"]["options"]["bm25_b"] == 0.3
    assert report["run"]["options"]["reranker_provider"] == "lexical"
    assert report["run"]["options"]["rerank_candidate_multiplier"] == 4
    assert len(report["run"]["case_ids"]) == 2
    assert report["run"]["index_provenance"]["profile_id"] == index["profile_id"]
    assert report["run"]["index_provenance"]["corpus_content_hash"] == corpus_content_hash
    assert report["run"]["index_provenance"]["source_corpus_hash"] == index[
        "source_corpus_hash"
    ]
    assert report["cost_disclosure"] == {
        "status": "not_applicable",
        "retrieval_model_calls": 0,
        "judge_model_calls": 0,
        "cost_usd": 0.0,
        "price_assumption": "offline deterministic retrieval, answer, and judge",
    }

    artifact_stem = f"{report['run']['id']}-{report['run']['execution_id']}"
    json_path = output_dir / f"{artifact_stem}.json"
    markdown_path = output_dir / f"{artifact_stem}.md"
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert markdown_path.is_file()
    assert "# Retrieval evaluation: fixture-rag-v1" in markdown_path.read_text(encoding="utf-8")

    replay = _invoke_json(
        runner,
        [
            "retrieval",
            "replay",
            report["evaluations"][0]["retrieval_trace_id"],
            "--db",
            str(db_path),
            "--json",
        ],
    )
    assert replay["trace_id"] == report["evaluations"][0]["retrieval_trace_id"]

    repeated = _invoke_json(runner, arguments)
    assert repeated["run"]["id"] == report["run"]["id"]
    assert repeated["run"]["execution_id"] != report["run"]["execution_id"]
    repeated_stem = f"{repeated['run']['id']}-{repeated['run']['execution_id']}"
    assert (output_dir / f"{repeated_stem}.json").is_file()
    assert json_path.is_file()


def _patch_offline_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(
        embedding_provider="hash",
        embedding_model="hash-token-v1",
        embedding_dimensions=384,
        embedding_base_url=None,
        embedding_api_key=None,
        top_k=8,
        candidate_multiplier=4,
        context_token_budget=128,
        graph_max_hops=2,
        naive_weight=1.0,
        local_weight=1.0,
        global_weight=1.0,
        naive_dense_weight=0.25,
        naive_bm25_weight=1.5,
        bm25_k1=1.7,
        bm25_b=0.3,
        reranker_provider="lexical",
        reranker_model="lexical-coverage-v1",
        rerank_candidate_multiplier=4,
    )
    monkeypatch.setattr(cli_module, "RetrievalSettings", lambda: settings)
    monkeypatch.setattr(cli_module, "TiktokenCounter", lambda _: WordCounter())


def _seed_scripted_graph(tmp_path: Path) -> Path:
    source = tmp_path / "atlas.txt"
    source.write_text(
        f"{EVIDENCE} This relation enables evidence-grounded retrieval.",
        encoding="utf-8",
    )
    db_path = tmp_path / "retrieval-cli.db"
    url = sqlite_url(db_path)
    upgrade_database(url)
    database = Database(url)
    try:
        ingestion = IngestionService(
            database,
            SectionTokenChunker(WordCounter(), max_tokens=64, overlap_tokens=0),
        )
        report = ingestion.ingest(source)
        assert report.failed == 0
        assert report.chunks_written == 1
        graph = GraphBuildService(
            database,
            ScriptedGraphClient(),
            ExtractionConfig(
                base_url="https://example.test",
                model="scripted-retrieval-cli",
                max_output_tokens=512,
                repair_max_attempts=0,
            ),
            checkpoint_path=tmp_path / "retrieval-cli-checkpoints.db",
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
    finally:
        database.dispose()
    return db_path


def _profile_corpus_content_hash(db_path: Path, profile_id: str) -> str:
    database = Database(sqlite_url(db_path))
    try:
        with database.session_factory() as session:
            profile = RetrievalRepository().get_profile(session, profile_id)
    finally:
        database.dispose()
    assert profile is not None
    value = profile.metadata["corpus_content_hash"]
    assert isinstance(value, str)
    return value


def _invoke_json(runner: CliRunner, arguments: list[str]) -> dict[str, Any]:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def _completion() -> CompletionResult:
    return CompletionResult(
        provider_request_id="request-retrieval-cli",
        model="scripted-retrieval-cli",
        system_fingerprint=None,
        content=json.dumps(
            {
                "entities": [
                    {
                        "ref": "e1",
                        "name": "Atlas",
                        "entity_type": "CONCEPT",
                        "description": "A source graph concept.",
                        "aliases": [],
                        "evidence_quotes": [EVIDENCE],
                    },
                    {
                        "ref": "e2",
                        "name": "Beacon",
                        "entity_type": "CONCEPT",
                        "description": "A target graph concept.",
                        "aliases": [],
                        "evidence_quotes": [EVIDENCE],
                    },
                ],
                "relations": [
                    {
                        "source_ref": "e1",
                        "target_ref": "e2",
                        "predicate": "CONNECTS",
                        "description": EVIDENCE,
                        "evidence_quotes": [EVIDENCE],
                    }
                ],
            }
        ),
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        cache_hit_tokens=None,
        cache_miss_tokens=None,
        raw_response={},
    )
