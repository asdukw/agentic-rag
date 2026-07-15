"""Local HTTP boundary for the agentic RAG workbench.

The browser receives only structured events and evidence selected by the Python
agent. It never receives a provider credential, database handle, or raw vector.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.agentic import AgentRunner, AgentRunRequest
from hybrid_rag.agentic.models import AgentEvent
from hybrid_rag.agentic.planner import AgentPlanner, DeepSeekAgentPlanner, DeterministicAgentPlanner
from hybrid_rag.config import (
    DeepSeekSettings,
    GraphSettings,
    RetrievalSettings,
    Settings,
    sqlite_url,
)
from hybrid_rag.demo import DemoRuntime, create_query_client, create_service
from hybrid_rag.extraction.client import DeepSeekClient
from hybrid_rag.extraction.schemas import ExtractionConfig, GraphConfig
from hybrid_rag.extraction.service import GraphBuildService
from hybrid_rag.extraction.workflow import WorkflowOptions
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.ingest.tokenizer import TiktokenCounter
from hybrid_rag.retrieval.query import QueryClient
from hybrid_rag.retrieval.service import RetrievalOptions
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.graph_repository import GraphRepository
from hybrid_rag.storage.migrations import upgrade_database
from hybrid_rag.storage.repository import IngestRepository
from hybrid_rag.workspace import WorkspaceStore

app = FastAPI(title="Hybrid RAG Lab API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-File-Name"],
)
_ACTIVE_RUNS = asyncio.Semaphore(2)
_WORKSPACES = WorkspaceStore()
# ``uvicorn.error`` is configured by Uvicorn's default logging setup, unlike an
# arbitrary application logger which may otherwise be discarded at INFO level.
logger = logging.getLogger("uvicorn.error")


class AgentWebRequest(AgentRunRequest):
    """Browser-safe execution settings; credentials remain server-side only."""

    database_url: str | None = Field(default=None, max_length=500)
    workspace_id: str | None = Field(default=None, max_length=32)
    embedding_provider: Literal["flagembedding", "hash"] | None = None
    embedding_model: str | None = Field(default=None, max_length=200)
    embedding_dimensions: int | None = Field(default=None, ge=32, le=4096)
    use_deepseek: bool = True
    top_k: int | None = Field(default=None, ge=1, le=8)
    context_token_budget: int | None = Field(default=None, ge=128, le=8000)
    graph_hops: int | None = Field(default=None, ge=1, le=2)
    reranker_enabled: bool = False
    rerank_candidate_multiplier: int | None = Field(default=None, ge=1, le=16)


class BuildIndexRequest(BaseModel):
    """A deliberate admin action, separate from the agent's read-only tool set."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    database_url: str | None = Field(default=None, max_length=500)
    embedding_provider: Literal["flagembedding", "hash"] | None = None
    embedding_model: str | None = Field(default=None, max_length=200)
    embedding_dimensions: int | None = Field(default=None, ge=32, le=4096)
    force: bool = False


class CreateWorkspaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "runtime": "python-agent"}


@app.get("/api/runtime/defaults")
async def runtime_defaults() -> dict[str, Any]:
    settings = Settings()
    retrieval = RetrievalSettings()
    return {
        "database_url": settings.database_url,
        "embedding_provider": retrieval.embedding_provider,
        "embedding_model": retrieval.embedding_model,
        "embedding_dimensions": retrieval.embedding_dimensions,
        "agent_budget": AgentRunRequest(question="defaults").budget.model_dump(mode="json"),
        "retrieval": {
            "top_k": min(retrieval.top_k, 8),
            "context_token_budget": retrieval.context_token_budget,
            "graph_hops": min(retrieval.graph_max_hops, 2),
        },
        "use_deepseek_default": True,
    }


@app.get("/api/workspaces")
async def list_workspaces() -> dict[str, Any]:
    return {"workspaces": [item.model_dump(mode="json") for item in _WORKSPACES.list()]}


@app.post("/api/workspaces")
async def create_workspace(request: CreateWorkspaceRequest) -> dict[str, Any]:
    try:
        workspace = _WORKSPACES.create(request.name)
        upgrade_database(_WORKSPACES.database_url(workspace.id))
        return workspace.model_dump(mode="json")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/workspaces/{workspace_id}/uploads")
async def upload_workspace_file(
    workspace_id: str,
    request: Request,
    filename: str | None = Header(default=None, alias="X-File-Name"),
) -> dict[str, Any]:
    if not filename:
        raise HTTPException(status_code=400, detail="X-File-Name header is required")
    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="uploaded file exceeds the 100 MiB limit")
    try:
        workspace = _WORKSPACES.store_upload(workspace_id, filename, await request.body())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return workspace.model_dump(mode="json")


@app.post("/api/workspaces/{workspace_id}/ingest")
async def ingest_workspace(workspace_id: str) -> dict[str, Any]:
    database = None
    try:
        workspace = _WORKSPACES.get(workspace_id)
        if not workspace.uploads:
            raise ValueError("upload one or more documents before ingesting")
        settings = Settings()
        database_url = _WORKSPACES.database_url(workspace_id)
        upgrade_database(database_url)
        database = Database(database_url)
        chunker = SectionTokenChunker(
            TiktokenCounter(settings.tokenizer_name),
            max_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        report = await asyncio.to_thread(
            IngestionService(database, chunker, repository=IngestRepository()).ingest,
            _WORKSPACES.uploads_path(workspace_id),
        )
        return report.model_dump(mode="json")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        if database is not None:
            database.dispose()


@app.post("/api/workspaces/{workspace_id}/graph/build")
async def build_workspace_graph(workspace_id: str) -> dict[str, Any]:
    database = None
    client: DeepSeekClient | None = None
    try:
        _WORKSPACES.get(workspace_id)
        graph = GraphSettings()
        deepseek = DeepSeekSettings()
        api_key = deepseek.api_key.get_secret_value().strip() if deepseek.api_key else ""
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required to build a knowledge graph")
        database_url = _WORKSPACES.database_url(workspace_id)
        upgrade_database(database_url)
        database = Database(database_url)
        extraction = ExtractionConfig(
            base_url=deepseek.base_url.rstrip("/"),
            model=deepseek.extraction_model,
            max_output_tokens=deepseek.max_output_tokens,
            repair_max_attempts=min(graph.max_attempts - 1, 1),
        )
        client = DeepSeekClient(
            api_key=api_key,
            base_url=extraction.base_url,
            model=extraction.model,
            max_output_tokens=extraction.max_output_tokens,
            timeout_seconds=deepseek.timeout_seconds,
            temperature=extraction.temperature,
        )
        service = GraphBuildService(
            database,
            client,
            extraction,
            checkpoint_path=_WORKSPACES.checkpoint_path(workspace_id),
            graph_config=GraphConfig(extraction_config_hash=extraction.config_hash),
            repository=GraphRepository(),
            deepseek_pricing=deepseek.pricing,
        )
        report = await service.build(
            WorkflowOptions(
                max_concurrency=graph.max_concurrency,
                max_attempts=graph.max_attempts,
                retry_failed=True,
                top_k=graph.top_k,
            )
        )
        return report.model_dump(mode="json")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        await _close(client)
        if database is not None:
            database.dispose()


@app.post("/api/workspaces/{workspace_id}/index/build")
async def build_workspace_index(workspace_id: str) -> dict[str, Any]:
    try:
        database_url = _WORKSPACES.database_url(workspace_id)
        runtime, settings, retrieval, deepseek = _runtime(
            database_url=database_url,
            embedding_provider=None,
            embedding_model=None,
            embedding_dimensions=None,
            top_k=None,
            context_token_budget=None,
            graph_hops=None,
            reranker_enabled=False,
            rerank_candidate_multiplier=None,
            use_deepseek=False,
        )
        database, service = create_service(runtime, settings, retrieval, deepseek)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        report = await asyncio.to_thread(service.build_index)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        database.dispose()
    return report.model_dump(mode="json")


@app.post("/api/index/build")
async def build_index(request: BuildIndexRequest) -> dict[str, Any]:
    """Build an index through an explicit browser action, never an agent action."""

    runtime, settings, retrieval, deepseek = _runtime(
        database_url=request.database_url,
        embedding_provider=request.embedding_provider,
        embedding_model=request.embedding_model,
        embedding_dimensions=request.embedding_dimensions,
        top_k=None,
        context_token_budget=None,
        graph_hops=None,
        reranker_enabled=False,
        rerank_candidate_multiplier=None,
        use_deepseek=False,
    )
    database, service = create_service(runtime, settings, retrieval, deepseek)
    try:
        report = service.build_index(force=request.force)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        database.dispose()
    return report.model_dump(mode="json")


@app.post("/api/agent/runs/stream")
async def stream_agent_run(request: AgentWebRequest) -> StreamingResponse:
    """Send a single agent run as Server-Sent Events."""

    async def events() -> AsyncIterator[str]:
        database = None
        answer_client: QueryClient | None = None
        planner: AgentPlanner | None = None
        try:
            runtime, settings, retrieval, deepseek = _runtime(
                database_url=(
                    _WORKSPACES.database_url(request.workspace_id)
                    if request.workspace_id
                    else request.database_url
                ),
                embedding_provider=request.embedding_provider,
                embedding_model=request.embedding_model,
                embedding_dimensions=request.embedding_dimensions,
                top_k=request.top_k,
                context_token_budget=request.context_token_budget,
                graph_hops=request.graph_hops,
                reranker_enabled=request.reranker_enabled,
                rerank_candidate_multiplier=request.rerank_candidate_multiplier,
                use_deepseek=request.use_deepseek,
            )
            database, service = create_service(runtime, settings, retrieval, deepseek)
            async with _ACTIVE_RUNS:
                answer_client = create_query_client(request.use_deepseek, deepseek)
                planner = _planner(request.use_deepseek, deepseek)
                assert answer_client is not None
                assert planner is not None
                runner = AgentRunner(
                    service,
                    planner=planner,
                    answer_client=answer_client,
                    retrieval_options=runtime.options,
                )
                async for event in runner.run(
                    AgentRunRequest(
                        question=request.question,
                        profile_id=request.profile_id,
                        budget=request.budget,
                    )
                ):
                    _log_agent_event(event)
                    yield _sse(event.event, event.model_dump(mode="json"))
        except (
            Exception
        ) as error:  # UI boundary: turn startup and run errors into a terminal SSE event.
            logger.exception("agent stream failed: %s: %s", type(error).__name__, error)
            yield _sse("failed", {"error": f"{type(error).__name__}: {error}"})
        finally:
            await _close(answer_client)
            await _close(planner)
            if database is not None:
                database.dispose()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _log_agent_event(event: AgentEvent) -> None:
    """Log agent control flow without logging tool payloads or answer text."""

    if event.event == "run_started":
        logger.info(
            "agent run started run_id=%s profile_id=%s",
            event.run_id,
            event.data.get("profile_id"),
        )
        return
    if event.event == "planner_action":
        logger.info(
            "agent planner decision run_id=%s step=%s action=%s rationale=%r args=%s",
            event.run_id,
            event.step,
            event.data.get("action"),
            event.data.get("rationale", ""),
            json.dumps(event.data.get("args", {}), ensure_ascii=False, sort_keys=True),
        )
        return
    if event.event == "tool_result":
        logger.info(
            "agent tool finished run_id=%s step=%s tool=%s ok=%s",
            event.run_id,
            event.step,
            event.data.get("tool"),
            event.data.get("ok"),
        )
        return
    if event.event == "completed":
        logger.info(
            "agent run completed run_id=%s reason=%s",
            event.run_id,
            event.data.get("termination_reason"),
        )
        return
    if event.event == "failed":
        logger.warning("agent run failed run_id=%s", event.run_id)


def _runtime(
    *,
    database_url: str | None,
    embedding_provider: Literal["flagembedding", "hash"] | None,
    embedding_model: str | None,
    embedding_dimensions: int | None,
    top_k: int | None,
    context_token_budget: int | None,
    graph_hops: int | None,
    reranker_enabled: bool,
    rerank_candidate_multiplier: int | None,
    use_deepseek: bool,
) -> tuple[DemoRuntime, Settings, RetrievalSettings, DeepSeekSettings]:
    settings = Settings()
    retrieval = RetrievalSettings()
    deepseek = DeepSeekSettings()
    options = RetrievalOptions(
        top_k=top_k or min(retrieval.top_k, 8),
        candidate_multiplier=retrieval.candidate_multiplier,
        context_token_budget=context_token_budget or retrieval.context_token_budget,
        graph_max_hops=graph_hops or min(retrieval.graph_max_hops, 2),
        naive_weight=retrieval.naive_weight,
        local_weight=retrieval.local_weight,
        global_weight=retrieval.global_weight,
        naive_dense_weight=retrieval.naive_dense_weight,
        naive_bm25_weight=retrieval.naive_bm25_weight,
        bm25_k1=retrieval.bm25_k1,
        bm25_b=retrieval.bm25_b,
        reranker_provider=retrieval.reranker_provider if reranker_enabled else "none",
        reranker_model=retrieval.reranker_model,
        reranker_use_fp16=retrieval.reranker_use_fp16,
        rerank_candidate_multiplier=rerank_candidate_multiplier
        or retrieval.rerank_candidate_multiplier,
    )
    selected_database = (database_url or "").strip()
    if selected_database and "://" not in selected_database:
        selected_database = sqlite_url(Path(selected_database))
    runtime = DemoRuntime(
        database_url=selected_database or settings.database_url,
        embedding_provider=embedding_provider or retrieval.embedding_provider,
        embedding_model=(embedding_model or retrieval.embedding_model).strip(),
        embedding_dimensions=embedding_dimensions or retrieval.embedding_dimensions,
        options=options,
        use_deepseek=use_deepseek,
    )
    return runtime, settings, retrieval, deepseek


def _planner(use_deepseek: bool, settings: DeepSeekSettings) -> AgentPlanner:
    if not use_deepseek:
        return DeterministicAgentPlanner()
    api_key = settings.api_key.get_secret_value().strip() if settings.api_key else ""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required when DeepSeek planner is enabled")
    return DeepSeekAgentPlanner(
        DeepSeekClient(
            api_key=api_key,
            base_url=settings.base_url,
            model=settings.query_model,
            max_output_tokens=512,
            timeout_seconds=settings.timeout_seconds,
            temperature=0,
        )
    )


async def _close(value: object | None) -> None:
    if value is None:
        return
    close = getattr(value, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
