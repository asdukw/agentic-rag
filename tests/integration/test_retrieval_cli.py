from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

import hybrid_rag.cli as cli_module
from hybrid_rag.cli import app
from hybrid_rag.config import RetrievalSettings, sqlite_url
from hybrid_rag.extraction.client import CompletionResult
from hybrid_rag.extraction.schemas import ExtractionConfig
from hybrid_rag.extraction.service import GraphBuildService
from hybrid_rag.extraction.workflow import WorkflowOptions
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.retrieval.embedding import BGEM3EmbeddingProvider
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database

EVIDENCE = "Atlas connects Beacon through the graph."


class WordCounter:
    name = "test-words"

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


class ScriptedGraphClient:
    async def extract(self, chunk_text: str, **_: object) -> CompletionResult:
        assert EVIDENCE in chunk_text
        return _completion()

    async def repair(
        self,
        chunk_text: str,
        invalid_response: str | None,
        issues: Sequence[str],
        *,
        document_title: str | None = None,
        section_path: Sequence[str] = (),
    ) -> CompletionResult:
        assert EVIDENCE in chunk_text
        assert invalid_response is None or isinstance(invalid_response, str)
        assert all(isinstance(issue, str) for issue in issues)
        assert document_title is None or isinstance(document_title, str)
        assert all(isinstance(section, str) for section in section_path)
        return _completion()


def test_cli_constructs_the_flag_embedding_reranker_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = RetrievalSettings(
        reranker_provider="flagembedding",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_use_fp16=True,
    )
    calls: list[tuple[str, str, bool]] = []
    marker = object()

    def create(provider: str, model: str, *, use_fp16: bool) -> object:
        calls.append((provider, model, use_fp16))
        return marker

    monkeypatch.setattr(cli_module, "create_reranker", create)

    assert cli_module._reranker(settings) is marker
    assert calls == [("flagembedding", "BAAI/bge-reranker-v2-m3", True)]


def test_cli_constructs_bge_m3_embedding_from_settings() -> None:
    settings = RetrievalSettings(
        embedding_provider="flagembedding",
        embedding_model="BAAI/bge-m3",
        embedding_dimensions=1024,
        embedding_batch_size=12,
        embedding_max_length=8192,
        embedding_use_fp16=False,
    )

    provider = cli_module._embedding_provider(settings)

    assert isinstance(provider, BGEM3EmbeddingProvider)
    assert provider.model == "BAAI/bge-m3"
    assert provider.dimensions == 1024
    assert provider.batch_size == 12
    assert provider.max_length == 8192
    assert not provider.use_fp16


def test_cli_rejects_removed_external_embedding_provider() -> None:
    with pytest.raises(typer.BadParameter, match=r"flagembedding.*hash"):
        cli_module._embedding_provider(
            RetrievalSettings(),
            provider="openai-compatible",
            model="embedding-test",
            dimensions=1024,
        )


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
    assert isinstance(index["corpus_content_hash"], str)

    retrieved = _invoke_json(
        runner,
        [
            "retrieve",
            question,
            "--db",
            str(db_path),
            "--json",
        ],
    )
    assert retrieved["mode"] == "mix"
    assert retrieved["trace_id"].startswith("rtr_")
    assert retrieved["context_items"]
    assert set(retrieved["trace"]["routes"]) == {"naive", "local", "global"}
    assert retrieved["trace"]["settings"]["naive_dense_weight"] == 0.25
    assert retrieved["trace"]["settings"]["naive_bm25_weight"] == 1.5
    assert retrieved["trace"]["settings"]["bm25_k1"] == 1.7
    assert retrieved["trace"]["settings"]["bm25_b"] == 0.3
    assert retrieved["trace"]["settings"]["rerank_enabled"] is False
    assert retrieved["trace"]["settings"]["reranker_provider"] == "none"
    assert retrieved["trace"]["settings"]["reranker_use_fp16"] is False
    assert retrieved["trace"]["settings"]["rerank_candidate_multiplier"] == 4
    assert retrieved["trace"]["rerank"] is None
    citation_id = retrieved["context_items"][0]["citation_id"]
    assert citation_id == retrieved["context_items"][0]["chunk_id"]

    hybrid = _invoke_json(
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
    assert hybrid["mode"] == "hybrid"
    assert set(hybrid["trace"]["routes"]) == {"local", "global"}

    answered = _invoke_json(
        runner,
        [
            "ask",
            question,
            "--db",
            str(db_path),
            "--json",
        ],
    )
    answer_trace_id = answered["retrieval"]["trace_id"]
    assert answer_trace_id.startswith("rtr_")
    assert answered["retrieval"]["mode"] == "mix"
    assert answered["answer"]["citations"] == [
        answered["retrieval"]["context_items"][0]["citation_id"]
    ]
    assert answered["answer"]["insufficient_evidence"] is False

    replay = _invoke_json(
        runner,
        ["retrieval", "replay", answer_trace_id, "--db", str(db_path), "--json"],
    )
    assert replay["retrieval"]["trace_id"] == answer_trace_id
    assert replay["retrieval"]["mode"] == "mix"
    assert replay["answer"]["citations"] == answered["answer"]["citations"]


def test_evaluate_cli_runs_provenance_bound_ragas_testset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = _seed_scripted_graph(tmp_path)
    _patch_offline_retrieval(monkeypatch)
    runner = CliRunner()
    index = _invoke_json(runner, ["build-index", "--db", str(db_path), "--json"])
    testset_path = tmp_path / "ragas-testset.json"
    testset_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "corpus_content_hash": index["corpus_content_hash"],
                "cases": [
                    {
                        "user_input": "How does Atlas connect Beacon?",
                        "reference": EVIDENCE,
                        "reference_contexts": [EVIDENCE],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "evaluation-reports" / "ragas.json"
    captured: dict[str, object] = {}

    class FakeReport:
        def __init__(self) -> None:
            self.modes = {"mix": {"means": {"faithfulness": 1.0}}}

        def as_dict(self) -> dict[str, object]:
            return {
                "testset_path": str(testset_path),
                "provenance": {"profile_id": index["profile_id"]},
                "modes": self.modes,
            }

    class FakeRagasRunner:
        def __init__(self, service: object, query_client: object) -> None:
            captured["service"] = service
            captured["query_client"] = query_client

        async def run(self, testset: Path, **kwargs: object) -> FakeReport:
            captured["testset"] = testset
            captured.update(kwargs)
            return FakeReport()

    api_key = SimpleNamespace(get_secret_value=lambda: "test-deepseek-key")
    monkeypatch.setattr(
        cli_module,
        "DeepSeekSettings",
        lambda: SimpleNamespace(
            api_key=api_key,
            query_model="deepseek-v4-flash",
            answer_model="deepseek-v4-flash",
            answer_max_output_tokens=2048,
            judge_model="deepseek-v4-pro",
            judge_max_output_tokens=1024,
            base_url="https://api.example.test",
            timeout_seconds=30.0,
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "EvaluationSettings",
        lambda: SimpleNamespace(output_dir=tmp_path / "default-evaluations"),
    )
    monkeypatch.setattr(
        cli_module,
        "_deepseek_query_client",
        lambda *, required_by: object(),
    )
    monkeypatch.setattr(cli_module, "RagasEvaluationRunner", FakeRagasRunner)
    arguments = [
        "evaluate",
        "--testset",
        str(testset_path),
        "--db",
        str(db_path),
        "--profile",
        index["profile_id"],
        "--output",
        str(output_path),
        "--json",
    ]
    report = _invoke_json(runner, arguments)

    assert report == json.loads(output_path.read_text(encoding="utf-8"))
    assert captured["testset"] == testset_path
    assert captured["profile_ref"] == index["profile_id"]
    assert captured["modes"] == (cli_module.RetrievalMode.MIX,)
    assert captured["judge_model"] == "deepseek-v4-pro"
    assert captured["judge_api_key"] == "test-deepseek-key"
    assert captured["judge_base_url"] == "https://api.example.test"
    assert captured["judge_max_output_tokens"] == 1024
    assert captured["judge_timeout_seconds"] == 30.0
    assert captured["query_client_provenance"] == {
        "keyword_model": "deepseek-v4-flash",
        "answer_model": "deepseek-v4-flash",
        "base_url": "https://api.example.test",
        "max_output_tokens": 2048,
        "timeout_seconds": 30.0,
    }


def _patch_offline_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(
        embedding_provider="hash",
        embedding_model="hash-token-v1",
        embedding_dimensions=384,
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
        reranker_provider="none",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_use_fp16=False,
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
