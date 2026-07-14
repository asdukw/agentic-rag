"""Local HTTP boundary for the agentic RAG workbench.

The browser receives only structured events and evidence selected by the Python
agent. It never receives a provider credential, database handle, or raw vector.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.agentic import AgentRunner, AgentRunRequest
from hybrid_rag.agentic.planner import AgentPlanner, DeepSeekAgentPlanner, DeterministicAgentPlanner
from hybrid_rag.config import DeepSeekSettings, RetrievalSettings, Settings, sqlite_url
from hybrid_rag.demo import DemoRuntime, create_query_client, create_service
from hybrid_rag.extraction.client import DeepSeekClient
from hybrid_rag.retrieval.query import QueryClient
from hybrid_rag.retrieval.service import RetrievalOptions

app = FastAPI(title="Hybrid RAG Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
_ACTIVE_RUNS = asyncio.Semaphore(2)


class AgentWebRequest(AgentRunRequest):
    """Browser-safe execution settings; credentials remain server-side only."""

    database_url: str | None = Field(default=None, max_length=500)
    embedding_provider: Literal["flagembedding", "hash"] | None = None
    embedding_model: str | None = Field(default=None, max_length=200)
    embedding_dimensions: int | None = Field(default=None, ge=32, le=4096)
    use_deepseek: bool = False
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
        "use_deepseek_default": False,
    }


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
                database_url=request.database_url,
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
                )
                async for event in runner.run(
                    AgentRunRequest(
                        question=request.question,
                        profile_id=request.profile_id,
                        budget=request.budget,
                    )
                ):
                    yield _sse(event.event, event.model_dump(mode="json"))
        except (
            Exception
        ) as error:  # UI boundary: turn startup and run errors into a terminal SSE event.
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
