from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select
from typer.testing import CliRunner

import hybrid_rag.cli as cli_module
from hybrid_rag.cli import app
from hybrid_rag.config import sqlite_url
from hybrid_rag.extraction.client import CompletionResult
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database
from hybrid_rag.storage.models import (ExtractionAttemptRecord,
                                       GraphBuildRunRecord)

EVIDENCE = "LightRAG-style systems extract entities and relations before retrieval."


class WordCounter:
    name = "test-words"

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


class ScriptedDeepSeekClient:
    def __init__(self) -> None:
        self.extract_calls: list[str] = []
        self.repair_calls: list[str] = []
        self.close_calls = 0

    async def extract(self, chunk_text: str, **_: object) -> CompletionResult:
        self.extract_calls.append(chunk_text)
        return _completion(_payload_for(chunk_text))

    async def repair(
        self,
        chunk_text: str,
        _invalid_response: str | None,
        _issues: object,
        **_: object,
    ) -> CompletionResult:
        self.repair_calls.append(chunk_text)
        return _completion(_payload_for(chunk_text))

    async def close(self) -> None:
        self.close_calls += 1


def test_build_graph_json_cached_rerun_and_read_only_commands_need_no_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = _seed_database(tmp_path)
    checkpoint = tmp_path / "build-checkpoints.db"
    client = ScriptedDeepSeekClient()
    _patch_deepseek(monkeypatch, client)
    runner = CliRunner()

    first = _invoke_json(runner, _build_args(db_path, checkpoint))
    calls_after_first = len(client.extract_calls) + len(client.repair_calls)
    _patch_keyless(monkeypatch)
    second = _invoke_json(runner, _build_args(db_path, checkpoint))

    assert first["status"] == "completed"
    assert first["chunks"]["total"] == 1
    assert first["chunks"]["succeeded"] == 1
    assert first["graph"]["nodes"] == 2
    assert first["graph"]["edges"] == 1
    assert second["status"] == "completed"
    assert second["chunks"]["cached"] == second["chunks"]["total"] == 1
    assert second["attempts"]["total"] == 0
    assert second["usage"]["total_tokens"] == 0
    assert len(client.extract_calls) + len(client.repair_calls) == calls_after_first
    assert client.close_calls == 1

    def fail_if_deepseek_settings_are_loaded() -> object:
        raise AssertionError("read-only graph commands must not load DeepSeek credentials")

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        cli_module,
        "DeepSeekSettings",
        fail_if_deepseek_settings_are_loaded,
    )

    stats = _invoke_json(
        runner,
        ["graph", "stats", "--db", str(db_path), "--json"],
    )
    assert stats["nodes"] == 2
    assert stats["edges"] == 1
    assert stats["top_entities"]

    run = _invoke_json(
        runner,
        ["graph", "inspect", second["run_id"], "--db", str(db_path)],
    )
    assert run["kind"] == "graph_build_run"
    extraction_id = run["items"][0]["extraction_id"]

    extraction = _invoke_json(
        runner,
        ["graph", "inspect", extraction_id, "--db", str(db_path)],
    )
    assert extraction["kind"] == "chunk_extraction"
    assert extraction["status"] == "succeeded"
    assert extraction["attempts"]
    assert "raw_response" not in extraction["attempts"][0]

    raw_extraction = _invoke_json(
        runner,
        [
            "graph",
            "inspect",
            extraction_id,
            "--db",
            str(db_path),
            "--raw",
        ],
    )
    assert "raw_response" in raw_extraction["attempts"][0]

    attempt_id = raw_extraction["attempts"][0]["id"]
    attempt = _invoke_json(
        runner,
        ["graph", "inspect", attempt_id, "--db", str(db_path)],
    )
    assert attempt["kind"] == "extraction_attempt"
    assert "raw_response" not in attempt

    entity_id = stats["top_entities"][0]["id"]
    entity = _invoke_json(
        runner,
        ["graph", "inspect", entity_id, "--db", str(db_path)],
    )
    assert entity["kind"] == "entity"
    assert entity["source_chunk_ids"]
    assert entity["evidence"][0]["quote"] == EVIDENCE


def test_graph_review_approves_then_resumes_without_another_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = _seed_database(tmp_path)
    checkpoint = tmp_path / "review-checkpoints.db"
    client = ScriptedDeepSeekClient()
    _patch_deepseek(monkeypatch, client)
    runner = CliRunner()

    paused = _invoke_json(
        runner,
        [*_build_args(db_path, checkpoint), "--review"],
    )
    assert paused["status"] == "awaiting_review"
    assert paused["chunks"]["needs_review"] == 1
    assert paused["graph"]["nodes"] == 0
    calls_before_review = len(client.extract_calls) + len(client.repair_calls)

    run = _invoke_json(
        runner,
        ["graph", "inspect", paused["run_id"], "--db", str(db_path)],
    )
    extraction_id = run["items"][0]["extraction_id"]

    def fail_if_deepseek_settings_are_loaded() -> object:
        raise AssertionError("review must not load DeepSeek credentials")

    monkeypatch.setattr(
        cli_module,
        "DeepSeekSettings",
        fail_if_deepseek_settings_are_loaded,
    )
    reviewed = _invoke_json(
        runner,
        [
            "graph",
            "review",
            extraction_id,
            "--decision",
            "approve",
            "--db",
            str(db_path),
        ],
    )
    assert reviewed["id"] == extraction_id
    assert reviewed["status"] == "succeeded"

    _patch_deepseek(monkeypatch, client)
    resumed = _invoke_json(
        runner,
        _resume_args(db_path, checkpoint, paused["run_id"]),
    )
    assert resumed["run_id"] == paused["run_id"]
    assert resumed["status"] == "completed"
    assert resumed["chunks"]["needs_review"] == 0
    assert resumed["graph"]["nodes"] == 2
    assert len(client.extract_calls) + len(client.repair_calls) == calls_before_review


def test_uncached_keyless_build_stops_before_creating_an_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = _seed_database(tmp_path)
    checkpoint = tmp_path / "keyless-checkpoints.db"
    _patch_keyless(monkeypatch)

    result = CliRunner().invoke(app, _build_args(db_path, checkpoint))

    assert result.exit_code == 1
    assert "DEEPSEEK_API_KEY is required" in result.output
    assert "Resume with: hybrid-rag build-graph --resume" in result.output
    assert "gbr_" in result.output
    database = Database(sqlite_url(db_path))
    try:
        with database.session_factory() as session:
            attempts = session.scalar(select(func.count()).select_from(ExtractionAttemptRecord))
            runs = session.scalar(select(func.count()).select_from(GraphBuildRunRecord))
    finally:
        database.dispose()
    assert attempts == 0
    assert runs == 1


def _seed_database(tmp_path: Path) -> Path:
    source = tmp_path / "graph-source.txt"
    source.write_text(EVIDENCE, encoding="utf-8")
    db_path = tmp_path / "graph-cli.db"
    url = sqlite_url(db_path)
    upgrade_database(url)
    database = Database(url)
    try:
        report = IngestionService(
            database,
            SectionTokenChunker(WordCounter(), max_tokens=32, overlap_tokens=0),
        ).ingest(source)
    finally:
        database.dispose()
    assert report.failed == 0
    assert report.chunks_written == 1
    return db_path


def _patch_deepseek(
    monkeypatch: pytest.MonkeyPatch,
    client: ScriptedDeepSeekClient,
) -> None:
    settings = SimpleNamespace(
        api_key=SecretStr("scripted-test-key"),
        base_url="https://example.test",
        extraction_model="scripted-cli",
        max_output_tokens=1024,
        timeout_seconds=5.0,
    )
    monkeypatch.setattr(cli_module, "DeepSeekSettings", lambda: settings)
    monkeypatch.setattr(cli_module, "DeepSeekClient", lambda **_: client)


def _patch_keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(
        api_key=None,
        base_url="https://example.test",
        extraction_model="scripted-cli",
        max_output_tokens=1024,
        timeout_seconds=5.0,
    )
    monkeypatch.setattr(cli_module, "DeepSeekSettings", lambda: settings)

    def fail_if_client_is_created(**_: object) -> object:
        raise AssertionError("cached graph builds must not create a provider client")

    monkeypatch.setattr(cli_module, "DeepSeekClient", fail_if_client_is_created)


def _build_args(db_path: Path, checkpoint: Path) -> list[str]:
    return [
        "build-graph",
        "--db",
        str(db_path),
        "--checkpoint",
        str(checkpoint),
        "--concurrency",
        "1",
        "--max-attempts",
        "1",
        "--limit",
        "1",
        "--json",
    ]


def _resume_args(db_path: Path, checkpoint: Path, run_id: str) -> list[str]:
    return [
        "build-graph",
        "--db",
        str(db_path),
        "--checkpoint",
        str(checkpoint),
        "--concurrency",
        "1",
        "--resume",
        run_id,
        "--json",
    ]


def _invoke_json(runner: CliRunner, arguments: list[str]) -> dict[str, Any]:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def _completion(content: str) -> CompletionResult:
    return CompletionResult(
        provider_request_id="request-cli-test",
        model="scripted-cli",
        system_fingerprint=None,
        content=content,
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        cache_hit_tokens=None,
        cache_miss_tokens=None,
        raw_response={"id": "request-cli-test"},
    )


def _payload_for(chunk_text: str) -> str:
    assert EVIDENCE in chunk_text
    return json.dumps(
        {
            "entities": [
                {
                    "ref": "e1",
                    "name": "LightRAG",
                    "entity_type": "METHOD",
                    "description": "A graph-oriented retrieval method.",
                    "aliases": ["LightRAG-style systems"],
                    "evidence_quotes": [EVIDENCE],
                },
                {
                    "ref": "e2",
                    "name": "entities and relations",
                    "entity_type": "CONCEPT",
                    "description": "Graph facts extracted before retrieval.",
                    "aliases": [],
                    "evidence_quotes": [EVIDENCE],
                },
            ],
            "relations": [
                {
                    "source_ref": "e1",
                    "target_ref": "e2",
                    "predicate": "EXTRACTS",
                    "description": "LightRAG extracts entities and relations.",
                    "evidence_quotes": [EVIDENCE],
                }
            ],
        }
    )
