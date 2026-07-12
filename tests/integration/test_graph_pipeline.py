from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from hybrid_rag.config import sqlite_url
from hybrid_rag.extraction.client import (
    CompletionResult,
    TerminalProviderError,
)
from hybrid_rag.extraction.schemas import ExtractionConfig
from hybrid_rag.extraction.service import GraphBuildService
from hybrid_rag.extraction.workflow import WorkflowOptions
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.graph_repository import GraphRepository
from hybrid_rag.storage.migrations import upgrade_database


class WordCounter:
    name = "test-words"

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


class ScriptedClient:
    def __init__(self, *, invalid_first: bool = False) -> None:
        self.invalid_first = invalid_first
        self.extract_calls: list[str] = []
        self.repair_calls: list[str] = []

    async def extract(self, chunk_text: str, **_: object) -> CompletionResult:
        self.extract_calls.append(chunk_text)
        if self.invalid_first and len(self.extract_calls) == 1:
            return _completion("not JSON")
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


def _database(tmp_path: Path) -> Database:
    url = sqlite_url(tmp_path / "graph.db")
    upgrade_database(url)
    database = Database(url)
    service = IngestionService(
        database,
        SectionTokenChunker(WordCounter(), max_tokens=60, overlap_tokens=5),
    )
    report = service.ingest(Path(__file__).parents[1] / "fixtures" / "corpus")
    assert report.failed == 0 and report.chunks_written > 1
    return database


def _service(
    tmp_path: Path,
    database: Database,
    client: object,
    *,
    model: str,
    max_attempts: int = 3,
    checkpoint_name: str = "langgraph.db",
) -> GraphBuildService:
    config = ExtractionConfig(
        base_url="https://example.test",
        model=model,
        max_output_tokens=1024,
        repair_max_attempts=max_attempts - 1,
    )
    return GraphBuildService(
        database,
        client,  # type: ignore[arg-type]
        config,
        checkpoint_path=tmp_path / checkpoint_name,
    )


def test_graph_build_repairs_persists_and_reuses_cached_extractions(tmp_path: Path) -> None:
    database = _database(tmp_path)
    client = ScriptedClient(invalid_first=True)
    service = _service(tmp_path, database, client, model="scripted-repair")
    options = WorkflowOptions(
        max_concurrency=3,
        max_attempts=3,
        retry_backoff_seconds=0,
        output_path=tmp_path / "graph.json",
    )
    try:
        first = asyncio.run(service.build(options))
        call_count = len(client.extract_calls) + len(client.repair_calls)
        second = asyncio.run(service.build(options))

        assert first.status == "completed"
        assert first.attempts.repair == 1
        assert first.graph.nodes == 2
        assert first.graph.edges == 1
        assert first.graph.weakly_connected_components == 1
        assert (tmp_path / "graph.json").is_file()
        assert second.chunks.cached == second.chunks.total
        assert second.attempts.total == 0
        assert second.usage.total_tokens == 0
        assert len(client.extract_calls) + len(client.repair_calls) == call_count
        assert service.workflow.report(first.run_id).graph.nodes == 2

        with database.session_factory() as session:
            stats = GraphRepository().stats(session)
            entity_id = stats["top_entities"][0]["id"]
            entity = GraphRepository().inspect(session, entity_id)
            run = GraphRepository().inspect(session, first.run_id)
        assert entity is not None
        assert entity["source_chunk_ids"]
        assert entity["evidence"][0]["quote"]
        assert run is not None
        assert set(run["report"]) >= {
            "execution",
            "extraction_config",
            "graph_config",
            "final_report",
        }
    finally:
        database.dispose()


def test_workflow_attempt_budget_must_match_hashed_extraction_config(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    service = _service(
        tmp_path,
        database,
        ScriptedClient(),
        model="scripted-budget-mismatch",
        max_attempts=1,
    )
    try:
        with pytest.raises(ValueError, match="repair_max_attempts"):
            asyncio.run(
                service.build(
                    WorkflowOptions(
                        max_attempts=2,
                        retry_backoff_seconds=0,
                    )
                )
            )
        assert service.last_run_id is None
    finally:
        database.dispose()


def test_human_review_interrupts_then_resumes_same_thread(tmp_path: Path) -> None:
    database = _database(tmp_path)
    client = ScriptedClient()
    service = _service(tmp_path, database, client, model="scripted-review", max_attempts=1)
    options = WorkflowOptions(
        max_concurrency=2,
        max_attempts=1,
        review_required=True,
        retry_backoff_seconds=0,
        limit=2,
    )
    try:
        paused = asyncio.run(service.build(options))
        assert paused.status == "awaiting_review"
        assert paused.chunks.needs_review == 2
        calls_before_resume = len(client.extract_calls)

        repository = GraphRepository()
        with database.session_factory.begin() as session:
            run = repository.inspect(session, paused.run_id)
            assert run is not None
            for item in run["items"]:
                repository.review_extraction(
                    session,
                    item["extraction_id"],
                    decision="approve",
                )

        completed = asyncio.run(service.build(options, resume_run_id=paused.run_id))
        assert completed.status == "completed"
        assert completed.chunks.succeeded == 2
        assert completed.chunks.needs_review == 0
        assert len(client.extract_calls) == calls_before_resume
    finally:
        database.dispose()


def test_terminal_chunk_failure_does_not_rollback_other_chunks(tmp_path: Path) -> None:
    database = _database(tmp_path)

    class PartlyFailingClient(ScriptedClient):
        async def extract(self, chunk_text: str, **kwargs: object) -> CompletionResult:
            if not self.extract_calls:
                self.extract_calls.append(chunk_text)
                raise TerminalProviderError("scripted terminal failure", status_code=422)
            return await super().extract(chunk_text, **kwargs)

    client = PartlyFailingClient()
    service = _service(tmp_path, database, client, model="scripted-failure", max_attempts=2)
    try:
        report = asyncio.run(
            service.build(
                WorkflowOptions(
                    max_concurrency=1,
                    max_attempts=2,
                    limit=2,
                    retry_backoff_seconds=0,
                )
            )
        )
        assert report.status == "completed_with_failures"
        assert report.chunks.failed == 1
        assert report.chunks.succeeded == 1
        assert report.failures[0].failure_kind == "provider_terminal"
    finally:
        database.dispose()


def test_crashed_worker_is_reclaimed_without_repeating_completed_chunks(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)

    class CrashingClient(ScriptedClient):
        async def extract(self, chunk_text: str, **kwargs: object) -> CompletionResult:
            self.extract_calls.append(chunk_text)
            if len(self.extract_calls) == 2:
                raise RuntimeError("simulated process loss")
            return _completion(_payload_for(chunk_text))

    crashing = CrashingClient()
    service = _service(tmp_path, database, crashing, model="scripted-resume", max_attempts=2)
    options = WorkflowOptions(
        max_concurrency=1,
        max_attempts=2,
        limit=2,
        retry_backoff_seconds=0,
        lease_seconds=0.01,
    )
    with pytest.raises(RuntimeError, match="resume run"):
        asyncio.run(service.build(options))
    run_id = service.last_run_id
    assert run_id is not None
    first_chunk = crashing.extract_calls[0]

    async def resume_after_lease() -> tuple[object, ScriptedClient]:
        await asyncio.sleep(0.2)
        resumed_client = ScriptedClient()
        resumed_service = _service(
            tmp_path,
            database,
            resumed_client,
            model="scripted-resume",
            max_attempts=2,
        )
        report = await resumed_service.build(options, resume_run_id=run_id)
        return report, resumed_client

    try:
        resumed, resumed_client = asyncio.run(resume_after_lease())
        assert resumed.status == "completed"
        assert resumed.chunks.succeeded == 2
        assert resumed.attempts.total == 3
        assert len(resumed_client.extract_calls) == 1
        assert first_chunk not in resumed_client.extract_calls
    finally:
        database.dispose()


def test_crash_at_attempt_limit_is_audited_without_an_extra_call(tmp_path: Path) -> None:
    database = _database(tmp_path)

    class CrashingClient(ScriptedClient):
        async def extract(self, chunk_text: str, **_: object) -> CompletionResult:
            self.extract_calls.append(chunk_text)
            raise RuntimeError("simulated crash at budget limit")

    crashing = CrashingClient()
    service = _service(
        tmp_path,
        database,
        crashing,
        model="scripted-budget-crash",
        max_attempts=1,
    )
    options = WorkflowOptions(
        max_concurrency=1,
        max_attempts=1,
        limit=1,
        retry_backoff_seconds=0,
        lease_seconds=0.01,
    )
    with pytest.raises(RuntimeError, match="resume run"):
        asyncio.run(service.build(options))
    run_id = service.last_run_id
    assert run_id is not None

    async def resume_after_lease():
        await asyncio.sleep(0.2)
        client = ScriptedClient()
        resumed_service = _service(
            tmp_path,
            database,
            client,
            model="scripted-budget-crash",
            max_attempts=1,
            checkpoint_name="replacement-langgraph.db",
        )
        report = await resumed_service.build(options, resume_run_id=run_id)
        return report, client

    try:
        report, resumed_client = asyncio.run(resume_after_lease())
        assert report.status == "failed"
        assert report.attempts.total == 1
        assert report.failures[0].failure_kind == "interrupted"
        assert resumed_client.extract_calls == []
    finally:
        database.dispose()


def test_all_failed_build_preserves_the_previous_valid_snapshot(tmp_path: Path) -> None:
    database = _database(tmp_path)
    seeded_client = ScriptedClient()
    seeded_service = _service(
        tmp_path,
        database,
        seeded_client,
        model="scripted-seed",
        max_attempts=1,
    )
    seed_options = WorkflowOptions(
        max_concurrency=2,
        max_attempts=1,
        retry_backoff_seconds=0,
    )

    class AlwaysFailingClient(ScriptedClient):
        async def extract(self, chunk_text: str, **_: object) -> CompletionResult:
            self.extract_calls.append(chunk_text)
            raise TerminalProviderError("permanent scripted failure", status_code=422)

    try:
        seeded = asyncio.run(seeded_service.build(seed_options))
        assert seeded.graph.nodes == 2
        failing = AlwaysFailingClient()
        failing_service = _service(
            tmp_path,
            database,
            failing,
            model="scripted-all-fail",
            max_attempts=1,
        )
        failed = asyncio.run(
            failing_service.build(
                WorkflowOptions(
                    max_concurrency=2,
                    max_attempts=1,
                    limit=2,
                    retry_backoff_seconds=0,
                )
            )
        )
        assert failed.status == "failed"
        assert failed.graph.nodes == 0
        with database.session_factory() as session:
            current = GraphRepository().stats(session)
        assert current["run_id"] == seeded.run_id
        assert current["nodes"] == 2
    finally:
        database.dispose()


def _completion(content: str) -> CompletionResult:
    return CompletionResult(
        provider_request_id="request-test",
        model="scripted",
        system_fingerprint=None,
        content=content,
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        cache_hit_tokens=None,
        cache_miss_tokens=None,
        raw_response={},
    )


def _payload_for(chunk_text: str) -> str:
    quote = "LightRAG-style systems extract entities and relations before retrieval."
    if quote not in chunk_text:
        return '{"entities":[],"relations":[]}'
    return json.dumps(
        {
            "entities": [
                {
                    "ref": "e1",
                    "name": "LightRAG",
                    "entity_type": "METHOD",
                    "description": "A graph-oriented retrieval method.",
                    "aliases": ["LightRAG-style systems"],
                    "evidence_quotes": [quote],
                },
                {
                    "ref": "e2",
                    "name": "entities and relations",
                    "entity_type": "CONCEPT",
                    "description": "Graph facts extracted before retrieval.",
                    "aliases": [],
                    "evidence_quotes": [quote],
                },
            ],
            "relations": [
                {
                    "source_ref": "e1",
                    "target_ref": "e2",
                    "predicate": "EXTRACTS",
                    "description": "LightRAG extracts entities and relations.",
                    "evidence_quotes": [quote],
                }
            ],
        }
    )
