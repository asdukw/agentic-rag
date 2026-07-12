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
