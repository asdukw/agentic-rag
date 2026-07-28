from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from hybrid_rag.agentic import AgentRunner, AgentRunRequest
from hybrid_rag.agentic.models import AgentEvent
from hybrid_rag.agentic.planner import DeepSeekAgentPlanner, DeterministicAgentPlanner
from hybrid_rag.config import (
    DeepSeekPricingSettings,
    DeepSeekSettings,
    EvaluationSettings,
    GraphSettings,
    RetrievalSettings,
    Settings,
    sqlite_url,
)
from hybrid_rag.deepseek_costs import DeepSeekCostSummary, DeepSeekPricing
from hybrid_rag.evaluation.ragas_runner import RagasEvaluationRunner
from hybrid_rag.extraction.client import DeepSeekClient
from hybrid_rag.extraction.reports import GraphBuildReport, GraphStorageStats
from hybrid_rag.extraction.schemas import (
    EXTRACTION_CONFIG_VERSION,
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_SCHEMA_VERSION,
    GLEANING_PROMPT_VERSION,
    REPAIR_PROMPT_VERSION,
    ExtractionConfig,
    GraphConfig,
)
from hybrid_rag.extraction.service import GraphBuildService
from hybrid_rag.extraction.workflow import WorkflowOptions
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.ingest.tokenizer import TiktokenCounter
from hybrid_rag.retrieval.embedding import (
    BGEM3EmbeddingProvider,
    EmbeddingConfigurationError,
    HashEmbeddingProvider,
)
from hybrid_rag.retrieval.models import IndexBuildReport, RetrievalResult, RetrievalStrategy
from hybrid_rag.retrieval.query import DeepSeekQueryClient, DeterministicQueryClient, QueryClient
from hybrid_rag.retrieval.reranker import create_reranker
from hybrid_rag.retrieval.service import AnswerResult, RetrievalOptions, RetrievalService
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.graph_repository import GraphRepository
from hybrid_rag.storage.migrations import upgrade_database
from hybrid_rag.storage.repository import IngestRepository
from hybrid_rag.storage.retrieval_repository import RetrievalRepository

app = typer.Typer(no_args_is_help=True, help="Agentic RAG Lab CLI")
db_app = typer.Typer(no_args_is_help=True, help="Database schema commands")
graph_app = typer.Typer(no_args_is_help=True, help="Graph extraction and inspection commands")
retrieval_app = typer.Typer(
    no_args_is_help=True,
    help="Retrieval index inspection and trace replay",
)
app.add_typer(db_app, name="db")
app.add_typer(graph_app, name="graph")
app.add_typer(retrieval_app, name="retrieval")
console = Console()
progress_console = Console(stderr=True)


@app.command("serve")
def serve(
    host: Annotated[
        str, typer.Option("--host", help="Bind host for the local web API")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Bind port")] = 8000,
) -> None:
    """Serve the local Python agent API consumed by the Bun/React workbench."""

    import uvicorn

    uvicorn.run("hybrid_rag.web_api:app", host=host, port=port)


def _database_url(db_path: Path | None, settings: Settings) -> str:
    return sqlite_url(db_path) if db_path is not None else settings.database_url


def _embedding_provider(
    settings: RetrievalSettings,
    *,
    provider: str | None = None,
    model: str | None = None,
    dimensions: int | None = None,
):
    selected_provider = provider or settings.embedding_provider
    selected_model = model or settings.embedding_model
    selected_dimensions = dimensions or settings.embedding_dimensions
    if selected_provider == "flagembedding":
        return BGEM3EmbeddingProvider(
            model=selected_model,
            dimensions=selected_dimensions,
            batch_size=settings.embedding_batch_size,
            max_length=settings.embedding_max_length,
            use_fp16=settings.embedding_use_fp16,
        )
    if selected_provider == "hash":
        return HashEmbeddingProvider(dimensions=selected_dimensions, model=selected_model)
    raise typer.BadParameter(
        "--provider must be 'flagembedding' or 'hash'",
        param_hint="--provider",
    )


def _reranker(settings: RetrievalSettings):
    """Construct the configured reranker without eagerly loading model weights."""

    return create_reranker(
        settings.reranker_provider,
        settings.reranker_model,
        use_fp16=settings.reranker_use_fp16,
    )


def _retrieval_options(
    settings: RetrievalSettings,
    *,
    top_k: int | None = None,
    context_tokens: int | None = None,
    graph_hops: int | None = None,
) -> RetrievalOptions:
    return RetrievalOptions(
        top_k=top_k or settings.top_k,
        candidate_multiplier=settings.candidate_multiplier,
        context_token_budget=context_tokens or settings.context_token_budget,
        graph_max_hops=graph_hops or settings.graph_max_hops,
        hybrid_weight=settings.hybrid_weight,
        graph_local_weight=settings.graph_local_weight,
        graph_global_weight=settings.graph_global_weight,
        hybrid_dense_weight=settings.hybrid_dense_weight,
        hybrid_bm25_weight=settings.hybrid_bm25_weight,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        reranker_provider=settings.reranker_provider,
        reranker_model=settings.reranker_model,
        reranker_use_fp16=settings.reranker_use_fp16,
        rerank_candidate_multiplier=settings.rerank_candidate_multiplier,
    )


def _deepseek_query_client(*, required_by: str = "--deepseek") -> DeepSeekQueryClient:
    settings = DeepSeekSettings()
    api_key = settings.api_key.get_secret_value().strip() if settings.api_key else ""
    if not api_key:
        raise typer.BadParameter(f"{required_by} requires DEEPSEEK_API_KEY")
    return DeepSeekQueryClient(
        api_key=api_key,
        model=settings.query_model,
        answer_model=settings.answer_model,
        base_url=settings.base_url,
        max_output_tokens=max(512, settings.answer_max_output_tokens),
        timeout_seconds=settings.timeout_seconds,
    )


def _configured_deepseek_pricing(settings: object) -> DeepSeekPricing | None:
    """Read pricing only when a fully configured settings object provides it.

    Keeping this tolerant makes cached/read-only commands usable with the
    deliberately minimal settings doubles used by integrations and callers.
    """

    pricing = getattr(settings, "pricing", None)
    return pricing if isinstance(pricing, DeepSeekPricing) else None


async def _close_query_client(client: QueryClient | None) -> None:
    if client is None:
        return
    close = getattr(client, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


def _agent_planner(
    *,
    use_deepseek: bool,
    settings: DeepSeekSettings,
) -> DeepSeekAgentPlanner | DeterministicAgentPlanner:
    if not use_deepseek:
        return DeterministicAgentPlanner()
    api_key = settings.api_key.get_secret_value().strip() if settings.api_key else ""
    if not api_key:
        raise typer.BadParameter("agentic mode requires DEEPSEEK_API_KEY")
    return DeepSeekAgentPlanner(
        DeepSeekClient(
            api_key=api_key,
            base_url=settings.base_url,
            model=settings.query_model,
            max_output_tokens=1_024,
            timeout_seconds=settings.timeout_seconds,
            temperature=0,
        )
    )


def _render_index_build_report(report: IndexBuildReport, *, json_output: bool) -> None:
    if json_output:
        console.print_json(report.model_dump_json())
        return
    table = Table(title=f"Embedding index {report.profile_id}")
    table.add_column("Chunks", justify="right")
    table.add_column("Entities", justify="right")
    table.add_column("Relations", justify="right")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Dimensions", justify="right")
    table.add_column("Reused")
    table.add_row(
        str(report.chunks),
        str(report.entities),
        str(report.relations),
        report.provider,
        report.model,
        str(report.dimensions),
        str(report.reused).lower(),
    )
    console.print(table)
    console.print(f"Corpus content hash: [cyan]{report.corpus_content_hash}[/cyan]")


def _render_retrieval_result(result: RetrievalResult, *, json_output: bool) -> None:
    if json_output:
        console.print_json(result.model_dump_json())
        return
    table = Table(title=f"{result.mode.value} retrieval ({result.profile_id})")
    table.add_column("Rank", justify="right")
    table.add_column("Citation")
    table.add_column("Score", justify="right")
    table.add_column("Routes")
    for rank, item in enumerate(result.context_items, start=1):
        table.add_row(
            str(rank),
            item.citation_id,
            f"{item.score:.4f}",
            ", ".join(f"{route}={score:.3f}" for route, score in item.route_scores.items()),
        )
    console.print(table)
    if rerank := result.trace.rerank:
        console.print(
            "Rerank: "
            f"{rerank.provider}/{rerank.model}/{rerank.version} "
            f"({len(rerank.hits)} candidates, limit={rerank.candidate_limit})"
        )
    if result.trace_id:
        console.print(f"Replay with: hrag retrieval replay {result.trace_id}")
    _render_deepseek_cost(result.trace.deepseek_cost)
    console.print(result.context or "[yellow]No evidence fit the context budget.[/yellow]")


def _render_answer_result(result: AnswerResult, *, json_output: bool) -> None:
    if json_output:
        console.print_json(result.model_dump_json())
        return
    _render_retrieval_result(result.retrieval, json_output=False)
    console.print("\n[bold]Answer[/bold]")
    console.print(result.answer.answer)
    console.print(f"Citations: {', '.join(result.answer.citations) or '(insufficient evidence)'}")


def _render_agent_result(events: list[AgentEvent], *, json_output: bool) -> None:
    if json_output:
        console.print_json(
            json.dumps(
                {
                    "mode": "agentic",
                    "run_id": events[0].run_id if events else None,
                    "events": [event.model_dump(mode="json") for event in events],
                },
                ensure_ascii=False,
            )
        )
        return

    table = Table(title="Agentic retrieval")
    table.add_column("Step", justify="right")
    table.add_column("Event")
    table.add_column("Summary")
    for event in events:
        if event.event == "planner_action":
            summary = f"{event.data.get('action', '')}: {event.data.get('rationale', '')}"
        elif event.event == "tool_result":
            summary = str(event.data.get("summary", ""))
        elif event.event == "completed":
            summary = str(event.data.get("termination_reason", ""))
        else:
            continue
        table.add_row(str(event.step), event.event, summary)
    console.print(table)

    answer_event = next((event for event in events if event.event == "answer"), None)
    if answer_event is None:
        console.print("[yellow]No answer was generated.[/yellow]")
        return
    answer = answer_event.data.get("answer", {})
    citations = answer.get("citations", []) if isinstance(answer, dict) else []
    console.print("\n[bold]Answer[/bold]")
    console.print(answer.get("answer", "") if isinstance(answer, dict) else "")
    console.print(
        f"Citations: {', '.join(str(value) for value in citations) or '(insufficient evidence)'}"
    )
    console.print(f"Run: {answer_event.run_id}")


def _build_extraction_config(
    database: Database,
    repository: GraphRepository,
    *,
    resume_run_id: str | None,
    base_url: str,
    model: str,
    max_output_tokens: int,
    max_attempts: int,
) -> ExtractionConfig:
    if resume_run_id is not None:
        with database.session_factory() as session:
            run = repository.get_run(session, resume_run_id)
        if run is None:
            raise typer.BadParameter(f"graph build run not found: {resume_run_id}")
        persisted = run.report.get("extraction_config")
        if isinstance(persisted, dict):
            expected_versions = {
                "version": EXTRACTION_CONFIG_VERSION,
                "schema_version": EXTRACTION_SCHEMA_VERSION,
                "prompt_version": EXTRACTION_PROMPT_VERSION,
                "repair_prompt_version": REPAIR_PROMPT_VERSION,
                "gleaning_prompt_version": GLEANING_PROMPT_VERSION,
            }
            mismatches = [
                f"{field}={persisted.get(field)!r} (current={expected!r})"
                for field, expected in expected_versions.items()
                if persisted.get(field) != expected
            ]
            if mismatches:
                raise typer.BadParameter(
                    "cannot resume a graph build created with a different extraction contract "
                    f"({', '.join(mismatches)}); start a new build-graph run",
                    param_hint="--resume",
                )
            return ExtractionConfig.model_validate(persisted)
    return ExtractionConfig(
        base_url=base_url.rstrip("/"),
        model=model,
        max_output_tokens=max_output_tokens,
        repair_max_attempts=min(max_attempts - 1, 1),
        gleaning_max_attempts=min(max_attempts - 1, 1),
    )


async def _run_graph_build(
    service: GraphBuildService,
    client: DeepSeekClient | None,
    options: WorkflowOptions,
    resume_run_id: str | None,
    *,
    show_progress: bool,
) -> GraphBuildReport:
    task = asyncio.create_task(service.build(options, resume_run_id=resume_run_id))
    try:
        if show_progress:
            await _monitor_graph_build(service, task, resume_run_id=resume_run_id)
        return await task
    finally:
        if client is not None:
            await client.close()


async def _monitor_graph_build(
    service: GraphBuildService,
    task: asyncio.Task[GraphBuildReport],
    *,
    resume_run_id: str | None,
) -> None:
    if not progress_console.is_terminal:
        await _monitor_graph_build_lines(service, task, resume_run_id=resume_run_id)
        return

    columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}"),
        TimeElapsedColumn(),
        TextColumn(
            "remaining={task.fields[remaining]} cached={task.fields[cached]} "
            "attempts={task.fields[attempts]} "
            "repair={task.fields[repair]} glean={task.fields[glean]} "
            "failed={task.fields[failed]} "
            "tokens={task.fields[tokens]}"
        ),
    )
    with Progress(*columns, console=progress_console, transient=False) as progress:
        progress_task = progress.add_task(
            "Starting graph build",
            total=1,
            remaining="?",
            cached=0,
            attempts=0,
            repair=0,
            glean=0,
            failed=0,
            tokens=0,
        )
        while not task.done():
            run_id = service.last_run_id or resume_run_id
            if run_id is not None:
                try:
                    stats = await asyncio.to_thread(service.stats, run_id=run_id, top_k=0)
                except Exception as error:  # Progress reporting must not fail the durable build.
                    progress.update(progress_task, description=f"Graph {run_id}: {error}")
                else:
                    chunks = stats.chunks
                    attempts = stats.attempts
                    usage = stats.usage
                    total = max(chunks.get("total", 0), 1)
                    completed = min(
                        chunks.get("succeeded", 0)
                        + chunks.get("needs_review", 0)
                        + chunks.get("failed", 0),
                        total,
                    )
                    progress.update(
                        progress_task,
                        description=f"Graph {run_id}",
                        total=total,
                        completed=completed,
                        remaining=max(total - completed, 0),
                        cached=chunks.get("cached", 0),
                        attempts=attempts.get("total", 0),
                        repair=attempts.get("repair", 0),
                        glean=attempts.get("glean", 0),
                        failed=chunks.get("failed", 0),
                        tokens=usage.get("total_tokens", 0),
                    )
            await asyncio.wait({task}, timeout=1.0)

        if not task.cancelled() and task.exception() is None:
            report = task.result()
            progress.update(
                progress_task,
                description=f"Graph {report.run_id} ({report.status})",
                total=max(report.chunks.total, 1),
                completed=report.chunks.total,
                remaining=report.chunks.remaining,
                cached=report.chunks.cached,
                attempts=report.attempts.total,
                repair=report.attempts.repair,
                glean=report.attempts.glean,
                failed=report.chunks.failed,
                tokens=report.usage.total_tokens,
            )


async def _monitor_graph_build_lines(
    service: GraphBuildService,
    task: asyncio.Task[GraphBuildReport],
    *,
    resume_run_id: str | None,
) -> None:
    progress_console.print("Starting graph build...", markup=False)
    last_snapshot: tuple[int, ...] | None = None
    reported_error = False
    while not task.done():
        run_id = service.last_run_id or resume_run_id
        if run_id is not None:
            try:
                stats = await asyncio.to_thread(service.stats, run_id=run_id, top_k=0)
            except Exception as error:  # Progress reporting must not fail the durable build.
                if not reported_error:
                    progress_console.print(
                        f"Graph {run_id}: progress unavailable: {error}",
                        markup=False,
                    )
                    reported_error = True
            else:
                reported_error = False
                chunks = stats.chunks
                attempts = stats.attempts
                usage = stats.usage
                total = chunks.get("total", 0)
                succeeded = chunks.get("succeeded", 0)
                needs_review = chunks.get("needs_review", 0)
                failed = chunks.get("failed", 0)
                completed = min(succeeded + needs_review + failed, total)
                snapshot = (
                    completed,
                    total,
                    chunks.get("cached", 0),
                    attempts.get("total", 0),
                    attempts.get("repair", 0),
                    attempts.get("glean", 0),
                    failed,
                    usage.get("total_tokens", 0),
                )
                if snapshot != last_snapshot:
                    percentage = completed / total * 100 if total else 0.0
                    progress_console.print(
                        f"Graph {run_id}: {completed}/{total} ({percentage:.1f}%) "
                        f"remaining={max(total - completed, 0)} cached={snapshot[2]} "
                        f"attempts={snapshot[3]} repair={snapshot[4]} "
                        f"glean={snapshot[5]} failed={failed} tokens={snapshot[7]}",
                        markup=False,
                    )
                    last_snapshot = snapshot
        await asyncio.wait({task}, timeout=5.0)

    if not task.cancelled() and task.exception() is None:
        report = task.result()
        progress_console.print(
            f"Graph {report.run_id}: {report.status}; "
            f"{report.chunks.total - report.chunks.remaining}/{report.chunks.total} complete, "
            f"failed={report.chunks.failed}, tokens={report.usage.total_tokens}",
            markup=False,
        )


def _render_graph_build_report(report: GraphBuildReport, *, json_output: bool) -> None:
    if json_output:
        console.print_json(report.model_dump_json())
        return
    chunks = Table(title=f"Graph build {report.run_id} ({report.status})")
    chunks.add_column("Total", justify="right")
    chunks.add_column("Cached", justify="right", style="cyan")
    chunks.add_column("Scheduled", justify="right")
    chunks.add_column("Succeeded", justify="right", style="green")
    chunks.add_column("Review", justify="right", style="yellow")
    chunks.add_column("Failed", justify="right", style="red")
    chunks.add_column("Attempts", justify="right")
    chunks.add_column("Extract", justify="right")
    chunks.add_column("Repair", justify="right")
    chunks.add_column("Glean", justify="right", style="cyan")
    chunks.add_row(
        str(report.chunks.total),
        str(report.chunks.cached),
        str(report.chunks.scheduled),
        str(report.chunks.succeeded),
        str(report.chunks.needs_review),
        str(report.chunks.failed),
        str(report.attempts.total),
        str(report.attempts.extract),
        str(report.attempts.repair),
        str(report.attempts.glean),
    )
    console.print(chunks)
    quality = report.extraction_quality
    extraction = Table(title="Extraction quality")
    extraction.add_column("Entities", justify="right")
    extraction.add_column("Entity drops", justify="right", style="yellow")
    extraction.add_column("Relations", justify="right")
    extraction.add_column("Relation drops", justify="right", style="yellow")
    extraction.add_column("Sanitized", justify="right", style="cyan")
    extraction.add_column("Chunks with drops", justify="right")
    extraction.add_row(
        f"{quality.accepted_entities}/{quality.raw_entities}",
        str(quality.dropped_entities),
        f"{quality.accepted_relations}/{quality.raw_relations}",
        str(quality.dropped_relations),
        str(quality.sanitized_relation_records),
        str(quality.chunks_with_drops),
    )
    console.print(extraction)
    _render_deepseek_cost(report.deepseek_cost)
    graph = GraphStorageStats(
        run_id=report.run_id,
        status=report.status,
        chunks={},
        attempts={},
        usage={},
        nodes=report.graph.nodes,
        edges=report.graph.edges,
        weakly_connected_components=report.graph.weakly_connected_components,
        largest_component_nodes=report.graph.largest_component_nodes,
        isolated_nodes=report.graph.isolated_nodes,
        top_entities=report.graph.top_entities,
    )
    _render_graph_stats(graph)
    for failure in report.failures:
        console.print(
            f"[red]{failure.extraction_id}[/red] {failure.failure_kind}: {failure.message}"
        )
    if report.status == "awaiting_review":
        console.print(
            f"Review pending xtr_ items with `graph inspect {report.run_id}`, then resume."
        )


def _render_deepseek_cost(cost: DeepSeekCostSummary | None) -> None:
    """Render the same explicit CNY breakdown used by JSON reports and traces."""

    if cost is None:
        return
    console.print(
        f"DeepSeek cost: status={cost.status.value}, {cost.currency}={cost.cost_cny}; "
        f"{cost.price_assumption}"
    )
    if not cost.usage:
        return
    usage = Table(title="DeepSeek response usage")
    usage.add_column("Operation")
    usage.add_column("Model")
    usage.add_column("Calls", justify="right")
    usage.add_column("Cache hit", justify="right")
    usage.add_column("Cache miss", justify="right")
    usage.add_column("Output", justify="right")
    usage.add_column("Complete")
    for item in cost.usage:
        usage.add_row(
            item.operation,
            item.model,
            str(item.calls),
            str(item.cache_hit_tokens) if item.cache_hit_tokens is not None else "-",
            str(item.cache_miss_tokens) if item.cache_miss_tokens is not None else "-",
            str(item.completion_tokens),
            "yes" if item.cache_breakdown_complete else "no",
        )
    console.print(usage)


def _render_graph_stats(result: GraphStorageStats) -> None:
    metrics = Table(title=f"Knowledge graph ({result.run_id or 'empty'})")
    metrics.add_column("Nodes", justify="right")
    metrics.add_column("Edges", justify="right")
    metrics.add_column("Components", justify="right")
    metrics.add_column("Largest", justify="right")
    metrics.add_column("Isolates", justify="right")
    metrics.add_row(
        str(result.nodes),
        str(result.edges),
        str(result.weakly_connected_components),
        str(result.largest_component_nodes),
        str(result.isolated_nodes),
    )
    console.print(metrics)
    _render_deepseek_cost(result.deepseek_cost)
    if not result.top_entities:
        return
    top = Table(title="Top entities")
    top.add_column("Entity")
    top.add_column("Type")
    top.add_column("Degree", justify="right")
    top.add_column("Sources", justify="right")
    for entity in result.top_entities:
        top.add_row(
            entity.name,
            entity.type,
            str(entity.degree),
            str(entity.source_chunks),
        )
    console.print(top)


def _redact_attempt_payload(payload: dict[str, object]) -> dict[str, object]:
    value = dict(payload)
    value.pop("raw_response", None)
    value.pop("messages", None)
    attempts = value.get("attempts")
    if isinstance(attempts, list):
        value["attempts"] = [
            _redact_attempt_payload(dict(attempt))
            for attempt in attempts
            if isinstance(attempt, dict)
        ]
    return value


def _json_default(value: object) -> str:
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


@db_app.command("upgrade")
def db_upgrade(
    db_path: Annotated[
        Path | None,
        typer.Option("--db", help="SQLite file; defaults to HYBRID_RAG_DATABASE_URL"),
    ] = None,
) -> None:
    settings = Settings()
    url = _database_url(db_path, settings)
    upgrade_database(url)
    console.print(f"Database schema is current: [cyan]{url}[/cyan]")


@app.command()
def ingest(
    source: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="A document or a directory"),
    ],
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite output file")] = None,
    chunk_size: Annotated[int | None, typer.Option("--chunk-size", min=32)] = None,
    overlap: Annotated[int | None, typer.Option("--overlap", min=0)] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print a JSON report")] = False,
) -> None:
    settings = Settings()
    url = _database_url(db_path, settings)
    size = chunk_size if chunk_size is not None else settings.chunk_size_tokens
    overlap_size = overlap if overlap is not None else settings.chunk_overlap_tokens
    if overlap_size >= size:
        raise typer.BadParameter("overlap must be smaller than chunk size", param_hint="--overlap")
    upgrade_database(url)

    database = Database(url)
    chunker = SectionTokenChunker(
        TiktokenCounter(settings.tokenizer_name),
        max_tokens=size,
        overlap_tokens=overlap_size,
    )
    try:
        report = IngestionService(database, chunker).ingest(source)
    finally:
        database.dispose()

    if json_output:
        console.print_json(report.model_dump_json())
    else:
        table = Table(title=f"Ingest run {report.run_id}")
        table.add_column("Discovered", justify="right")
        table.add_column("Inserted", justify="right", style="green")
        table.add_column("Updated", justify="right", style="yellow")
        table.add_column("Skipped", justify="right", style="cyan")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Chunks", justify="right")
        table.add_column("Seconds", justify="right")
        table.add_row(
            str(report.discovered),
            str(report.inserted),
            str(report.updated),
            str(report.skipped),
            str(report.failed),
            str(report.chunks_written),
            f"{report.duration_seconds:.3f}",
        )
        console.print(table)
        for failure in report.failures:
            console.print(f"[red]{failure.path}[/red]: {failure.error_type}: {failure.message}")

    if report.failed:
        raise typer.Exit(code=1)


@app.command("build-graph")
def build_graph(
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite input/output file")] = None,
    model: Annotated[str | None, typer.Option("--model", help="DeepSeek extraction model")] = None,
    concurrency: Annotated[int | None, typer.Option("--concurrency", min=1)] = None,
    max_attempts: Annotated[
        int | None,
        typer.Option("--max-attempts", min=1, help="Maximum calls per selected chunk"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Cost-safe prefix of the current chunks"),
    ] = None,
    retry_failed: Annotated[
        bool,
        typer.Option(
            "--retry-failed/--no-retry-failed",
            help="Retry terminal failed extractions (enabled by default)",
        ),
    ] = True,
    review: Annotated[
        bool,
        typer.Option(
            "--review",
            help="Review newly extracted results; validated cached results remain trusted",
        ),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Resume a persisted gbr_ LangGraph thread"),
    ] = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="LangGraph checkpoint SQLite file"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Write deterministic NetworkX node-link JSON"),
    ] = None,
    top: Annotated[int | None, typer.Option("--top", min=1)] = None,
    progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress",
            help="Show live graph extraction counters in the terminal",
        ),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Print a JSON report")] = False,
) -> None:
    if resume is not None and (
        model is not None or max_attempts is not None or limit is not None or review
    ):
        raise typer.BadParameter(
            "--resume reuses the run's model, attempt budget, corpus, and review policy"
        )
    settings = Settings()
    graph_settings = GraphSettings()
    deepseek = DeepSeekSettings()

    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    repository = GraphRepository()
    try:
        extraction_config = _build_extraction_config(
            database,
            repository,
            resume_run_id=resume,
            base_url=deepseek.base_url,
            model=model or deepseek.extraction_model,
            max_output_tokens=deepseek.max_output_tokens,
            max_attempts=max_attempts or graph_settings.max_attempts,
        )
        graph_config = GraphConfig(extraction_config_hash=extraction_config.config_hash)
        api_key = deepseek.api_key.get_secret_value().strip() if deepseek.api_key else ""
        client = (
            DeepSeekClient(
                api_key=api_key,
                base_url=extraction_config.base_url,
                model=extraction_config.model,
                max_output_tokens=extraction_config.max_output_tokens,
                timeout_seconds=deepseek.timeout_seconds,
                temperature=extraction_config.temperature,
            )
            if api_key
            else None
        )
        options = WorkflowOptions(
            max_concurrency=concurrency or graph_settings.max_concurrency,
            max_attempts=max_attempts or graph_settings.max_attempts,
            limit=limit,
            retry_failed=retry_failed,
            review_required=review,
            top_k=top or graph_settings.top_k,
            output_path=output,
        )
        service = GraphBuildService(
            database,
            client,
            extraction_config,
            checkpoint_path=checkpoint or graph_settings.checkpoint_path,
            graph_config=graph_config,
            repository=repository,
            deepseek_pricing=_configured_deepseek_pricing(deepseek),
        )
        try:
            report = asyncio.run(
                _run_graph_build(
                    service,
                    client,
                    options,
                    resume,
                    show_progress=progress,
                )
            )
        except KeyboardInterrupt as error:
            run_id = service.last_run_id or resume
            console.print("[yellow]Graph build interrupted after durable checkpoints.[/yellow]")
            if run_id:
                console.print(f"Resume with: hrag build-graph --resume {run_id}")
            raise typer.Exit(code=130) from error
        except Exception as error:
            run_id = service.last_run_id or resume
            console.print(f"[red]{type(error).__name__}:[/red] {error}")
            if run_id:
                console.print(f"Resume with: hrag build-graph --resume {run_id}")
            raise typer.Exit(code=1) from error
    finally:
        database.dispose()

    _render_graph_build_report(report, json_output=json_output)
    if report.status in {"completed_with_failures", "failed"}:
        raise typer.Exit(code=1)


@app.command("build-index")
def build_index(
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite input/output file")] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run", help="Use a specific completed graph snapshot; defaults to current"),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="flagembedding (default) or hash (compatibility)",
        ),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", help="Embedding model identity")] = None,
    dimensions: Annotated[int | None, typer.Option("--dimensions", min=1)] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-embed an otherwise current profile"),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print a JSON report")] = False,
) -> None:
    settings = Settings()
    retrieval_settings = RetrievalSettings()
    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    try:
        service = RetrievalService(
            database,
            _embedding_provider(
                retrieval_settings,
                provider=provider,
                model=model,
                dimensions=dimensions,
            ),
            TiktokenCounter(settings.tokenizer_name),
        )
        report = service.build_index(build_run_id=run_id, force=force)
    except (EmbeddingConfigurationError, ValueError, RuntimeError) as error:
        console.print(f"[red]{type(error).__name__}:[/red] {error}")
        raise typer.Exit(code=1) from error
    finally:
        database.dispose()
    _render_index_build_report(report, json_output=json_output)


@app.command("retrieve")
def retrieve(
    question: Annotated[str, typer.Argument(help="Question to retrieve evidence for")],
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite file")] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="dense, bm25, hybrid, graph_local, graph_global, graph_hybrid, or mix",
        ),
    ] = "mix",
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="idx_ profile ID or config hash"),
    ] = None,
    top: Annotated[int | None, typer.Option("--top", min=1)] = None,
    context_tokens: Annotated[int | None, typer.Option("--context-tokens", min=1)] = None,
    graph_hops: Annotated[int | None, typer.Option("--graph-hops", min=1, max=4)] = None,
    deepseek: Annotated[
        bool,
        typer.Option("--deepseek", help="Use DeepSeek only to extract bounded query keywords"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print a JSON result and trace"),
    ] = False,
) -> None:
    try:
        selected_mode = RetrievalStrategy(mode)
    except ValueError as error:
        raise typer.BadParameter(
            "--mode must be dense, bm25, hybrid, graph_local, graph_global, graph_hybrid, or mix"
        ) from error
    settings = Settings()
    retrieval_settings = RetrievalSettings()
    deepseek_settings = DeepSeekSettings()
    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    client: QueryClient | None = None
    try:
        client = _deepseek_query_client() if deepseek else None
        service = RetrievalService(
            database,
            _embedding_provider(retrieval_settings),
            TiktokenCounter(settings.tokenizer_name),
            reranker=_reranker(retrieval_settings),
            deepseek_pricing=_configured_deepseek_pricing(deepseek_settings),
        )
        options = _retrieval_options(
            retrieval_settings,
            top_k=top,
            context_tokens=context_tokens,
            graph_hops=graph_hops,
        )

        async def run() -> RetrievalResult:
            try:
                return await service.retrieve_with_keywords(
                    question,
                    keyword_extractor=client,
                    mode=selected_mode,
                    options=options,
                    profile_ref=profile,
                )
            finally:
                await _close_query_client(client)

        result = asyncio.run(run())
    except (EmbeddingConfigurationError, ValueError, RuntimeError) as error:
        console.print(f"[red]{type(error).__name__}:[/red] {error}")
        raise typer.Exit(code=1) from error
    finally:
        database.dispose()
    _render_retrieval_result(result, json_output=json_output)


@app.command("ask")
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer from retrieved evidence")],
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite file")] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help=("agentic, dense, bm25, hybrid, graph_local, graph_global, graph_hybrid, or mix"),
        ),
    ] = "agentic",
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="idx_ profile ID or config hash"),
    ] = None,
    top: Annotated[int | None, typer.Option("--top", min=1)] = None,
    context_tokens: Annotated[int | None, typer.Option("--context-tokens", min=1)] = None,
    graph_hops: Annotated[int | None, typer.Option("--graph-hops", min=1, max=4)] = None,
    deepseek: Annotated[
        bool,
        typer.Option(
            "--deepseek/--no-deepseek",
            help="Use DeepSeek for planning, keywords, and citation-bound answers",
        ),
    ] = True,
    rerank: Annotated[
        bool,
        typer.Option(
            "--rerank/--no-rerank",
            help="Rerank candidates returned by agentic search tools",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print JSON answer, evidence, and trace"),
    ] = False,
) -> None:
    selected_mode: RetrievalStrategy | None = None
    if mode != "agentic":
        try:
            selected_mode = RetrievalStrategy(mode)
        except ValueError as error:
            raise typer.BadParameter(
                "--mode must be agentic, dense, bm25, hybrid, graph_local, "
                "graph_global, graph_hybrid, or mix"
            ) from error
    settings = Settings()
    retrieval_settings = RetrievalSettings()
    deepseek_settings = DeepSeekSettings()
    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    client: QueryClient | None = None
    planner: DeepSeekAgentPlanner | DeterministicAgentPlanner | None = None
    agent_events: list[AgentEvent] | None = None
    result: AnswerResult | None = None
    try:
        service = RetrievalService(
            database,
            _embedding_provider(retrieval_settings),
            TiktokenCounter(settings.tokenizer_name),
            reranker=_reranker(retrieval_settings),
            deepseek_pricing=_configured_deepseek_pricing(deepseek_settings),
        )
        options = _retrieval_options(
            retrieval_settings,
            top_k=top,
            context_tokens=context_tokens,
            graph_hops=graph_hops,
        )

        if selected_mode is None:
            if not rerank:
                options = replace(options, reranker_provider="none")
            client = (
                _deepseek_query_client(required_by="agentic mode")
                if deepseek
                else (DeterministicQueryClient())
            )
            planner = _agent_planner(use_deepseek=deepseek, settings=deepseek_settings)
            runner = AgentRunner(
                service,
                planner=planner,
                answer_client=client,
                retrieval_options=options,
            )

            async def run_agent() -> list[AgentEvent]:
                try:
                    events = [
                        event
                        async for event in runner.run(
                            AgentRunRequest(question=question, profile_id=profile)
                        )
                    ]
                    failed = next((event for event in events if event.event == "failed"), None)
                    if failed is not None:
                        raise RuntimeError(str(failed.data.get("error", "agent run failed")))
                    return events
                finally:
                    await _close_query_client(client)
                    if planner is not None:
                        close = getattr(planner, "close", None)
                        if callable(close):
                            close_result = close()
                            if inspect.isawaitable(close_result):
                                await close_result

            agent_events = asyncio.run(run_agent())
        else:
            client = _deepseek_query_client() if deepseek else None

            async def run_fixed() -> AnswerResult:
                try:
                    return await service.ask(
                        question,
                        query_client=client,
                        mode=selected_mode,
                        options=options,
                        profile_ref=profile,
                    )
                finally:
                    await _close_query_client(client)

            result = asyncio.run(run_fixed())
    except (EmbeddingConfigurationError, ValueError, RuntimeError) as error:
        console.print(f"[red]{type(error).__name__}:[/red] {error}")
        raise typer.Exit(code=1) from error
    finally:
        database.dispose()
    if selected_mode is None:
        assert agent_events is not None
        _render_agent_result(agent_events, json_output=json_output)
    else:
        assert result is not None
        _render_answer_result(result, json_output=json_output)


@app.command("evaluate")
def evaluate(
    testset_path: Annotated[
        Path,
        typer.Option(
            "--testset",
            exists=True,
            readable=True,
            help="Golden evaluation test-set envelope JSON",
        ),
    ],
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite file")] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Pinned index profile ID or config hash"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="JSON file for evaluation scores and per-case details"),
    ] = None,
    modes: Annotated[
        str,
        typer.Option(
            "--modes",
            help=(
                "Comma-separated modes: agentic, dense, bm25, hybrid, graph_local, "
                "graph_global, graph_hybrid, or mix"
            ),
        ),
    ] = "agentic,mix,hybrid",
    top: Annotated[int | None, typer.Option("--top", min=1)] = None,
    context_tokens: Annotated[int | None, typer.Option("--context-tokens", min=1)] = None,
    graph_hops: Annotated[int | None, typer.Option("--graph-hops", min=1, max=4)] = None,
    agentic_rerank: Annotated[
        bool,
        typer.Option(
            "--agentic-rerank/--no-agentic-rerank",
            help="Rerank candidates returned by Agentic search tools",
        ),
    ] = False,
    smoke: Annotated[
        bool,
        typer.Option(
            "--smoke",
            help=(
                "Evaluate a deterministic six-case subset: 3 single-hop and one each "
                "of summary/reasoning, multi-context, and unanswerable"
            ),
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print the full JSON report")] = False,
) -> None:
    """Score a provenance-bound golden test set against this RAG pipeline."""

    mode_names = tuple(value.strip() for value in modes.split(",") if value.strip())
    if not mode_names or len(mode_names) != len(set(mode_names)):
        raise typer.BadParameter("--modes must contain one or more distinct modes")
    allowed_modes = {"agentic", *(mode.value for mode in RetrievalStrategy)}
    if any(value not in allowed_modes for value in mode_names):
        raise typer.BadParameter(
            "--modes must use agentic, dense, bm25, hybrid, graph_local, "
            "graph_global, graph_hybrid, and/or mix"
        )
    selected_modes = tuple(RetrievalStrategy(value) for value in mode_names if value != "agentic")
    include_agentic = "agentic" in mode_names

    settings = Settings()
    retrieval_settings = RetrievalSettings()
    evaluation_settings = EvaluationSettings()
    deepseek_settings = DeepSeekSettings()
    api_key = (
        deepseek_settings.api_key.get_secret_value().strip() if deepseek_settings.api_key else ""
    )
    if not api_key:
        raise typer.BadParameter("evaluate requires DEEPSEEK_API_KEY")
    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    planner: DeepSeekAgentPlanner | DeterministicAgentPlanner | None = None
    try:
        service = RetrievalService(
            database,
            _embedding_provider(retrieval_settings),
            TiktokenCounter(settings.tokenizer_name),
            reranker=_reranker(retrieval_settings),
            deepseek_pricing=_configured_deepseek_pricing(deepseek_settings),
        )
        options = _retrieval_options(
            retrieval_settings,
            top_k=top,
            context_tokens=context_tokens,
            graph_hops=graph_hops,
        )
        query_client = _deepseek_query_client(required_by="evaluate")
        agentic_runner: AgentRunner | None = None
        if include_agentic:
            planner = _agent_planner(use_deepseek=True, settings=deepseek_settings)
            agentic_options = (
                options if agentic_rerank else replace(options, reranker_provider="none")
            )
            agentic_runner = AgentRunner(
                service,
                planner=planner,
                answer_client=query_client,
                retrieval_options=agentic_options,
            )

        async def run():
            try:
                return await RagasEvaluationRunner(service, query_client).run(
                    testset_path,
                    modes=selected_modes,
                    retrieval_options=options,
                    profile_ref=profile,
                    judge_model=deepseek_settings.judge_model,
                    judge_api_key=api_key,
                    judge_base_url=deepseek_settings.base_url,
                    judge_max_output_tokens=deepseek_settings.judge_max_output_tokens,
                    judge_timeout_seconds=deepseek_settings.timeout_seconds,
                    query_client_provenance={
                        "keyword_model": deepseek_settings.query_model,
                        "answer_model": deepseek_settings.answer_model,
                        "base_url": deepseek_settings.base_url,
                        "max_output_tokens": max(512, deepseek_settings.answer_max_output_tokens),
                        "timeout_seconds": deepseek_settings.timeout_seconds,
                    },
                    agentic_runner=agentic_runner,
                    smoke=smoke,
                )
            finally:
                await _close_query_client(query_client)
                if planner is not None:
                    close = getattr(planner, "close", None)
                    if callable(close):
                        close_result = close()
                        if inspect.isawaitable(close_result):
                            await close_result

        report = asyncio.run(run())
        payload = report.as_dict()
        report_prefix = "ragas-smoke" if smoke else "ragas"
        destination = output or (
            evaluation_settings.output_dir / f"{report_prefix}-{testset_path.stem}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except (EmbeddingConfigurationError, ImportError, OSError, RuntimeError, ValueError) as error:
        console.print(f"[red]{type(error).__name__}:[/red] {error}")
        raise typer.Exit(code=1) from error
    finally:
        database.dispose()
    if json_output:
        console.print_json(json.dumps(payload, ensure_ascii=False))
    else:
        console.print(f"Ragas evaluation written to [cyan]{destination}[/cyan]")
        for mode, result in report.modes.items():
            console.print(f"{mode}: {result['means']}")


@retrieval_app.command("stats")
def retrieval_stats(
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite file")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    settings = Settings()
    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    try:
        with database.session_factory() as session:
            profiles = RetrievalRepository().list_profiles(session)
            payload = [
                {
                    "id": profile.id,
                    "active": profile.is_active,
                    "status": profile.status,
                    "provider": profile.provider,
                    "model": profile.model,
                    "dimensions": profile.dimensions,
                    "source_corpus_hash": profile.source_corpus_hash,
                    "graph_build_run_id": profile.source_graph_run_id,
                    "created_at": profile.created_at,
                    "updated_at": profile.updated_at,
                }
                for profile in profiles
            ]
    finally:
        database.dispose()
    if json_output:
        console.print_json(json.dumps(payload, ensure_ascii=False, default=_json_default))
        return
    table = Table(title="Embedding indexes")
    table.add_column("ID")
    table.add_column("Active")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Dims", justify="right")
    table.add_column("Graph run")
    for profile in payload:
        table.add_row(
            str(profile["id"]),
            "yes" if profile["active"] else "",
            str(profile["provider"]),
            str(profile["model"]),
            str(profile["dimensions"]),
            str(profile["graph_build_run_id"] or "-"),
        )
    console.print(table)


@retrieval_app.command("replay")
def retrieval_replay(
    trace_id: Annotated[str, typer.Argument(help="rtr_ retrieval trace ID")],
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite file")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    if not trace_id.startswith("rtr_"):
        raise typer.BadParameter("replay requires an rtr_ trace ID")
    settings = Settings()
    retrieval_settings = RetrievalSettings()
    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    try:
        service = RetrievalService(
            database,
            _embedding_provider(retrieval_settings),
            TiktokenCounter(settings.tokenizer_name),
        )
        answer = service.replay_answer(trace_id)
        if answer is not None:
            _render_answer_result(answer, json_output=json_output)
        else:
            _render_retrieval_result(service.replay(trace_id), json_output=json_output)
    except (EmbeddingConfigurationError, ValueError, RuntimeError) as error:
        console.print(f"[red]{type(error).__name__}:[/red] {error}")
        raise typer.Exit(code=1) from error
    finally:
        database.dispose()


@graph_app.command("stats")
def graph_stats(
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite file")] = None,
    run_id: Annotated[str | None, typer.Option("--run", help="Specific graph build run")] = None,
    top: Annotated[int, typer.Option("--top", min=1)] = 10,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    settings = Settings()
    deepseek = DeepSeekPricingSettings()
    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    try:
        with database.session_factory() as session:
            repository = GraphRepository()
            result = GraphStorageStats.model_validate(
                repository.stats(session, run_id=run_id, top_k=top)
            )
            pricing = _configured_deepseek_pricing(deepseek)
            if result.run_id is not None and pricing is not None:
                result = result.model_copy(
                    update={
                        "deepseek_cost": pricing.estimate(
                            repository.deepseek_usage(session, result.run_id)
                        )
                    }
                )
    finally:
        database.dispose()
    if json_output:
        console.print_json(result.model_dump_json())
        return
    _render_graph_stats(result)


@graph_app.command("inspect")
def graph_inspect(
    object_id: Annotated[str, typer.Argument(help="gbr_, xtr_, xat_, ent_, or rel_ ID")],
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite file")] = None,
    raw: Annotated[
        bool,
        typer.Option("--raw", help="Include stored prompts/responses containing corpus text"),
    ] = False,
) -> None:
    settings = Settings()
    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    try:
        with database.session_factory() as session:
            payload = GraphRepository().inspect(session, object_id)
    finally:
        database.dispose()
    if payload is None:
        console.print(f"[red]Graph object not found:[/red] {object_id}")
        raise typer.Exit(code=1)
    if not raw:
        payload = _redact_attempt_payload(payload)
    console.print_json(json.dumps(payload, ensure_ascii=False, default=_json_default))


@graph_app.command("review")
def graph_review(
    extraction_id: Annotated[str, typer.Argument(help="xtr_ extraction ID")],
    decision: Annotated[
        str,
        typer.Option("--decision", help="approve or reject"),
    ],
    note: Annotated[str | None, typer.Option("--note")] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run", help="Required when the extraction awaits multiple reviews"),
    ] = None,
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite file")] = None,
) -> None:
    if decision not in {"approve", "reject"}:
        raise typer.BadParameter("decision must be approve or reject", param_hint="--decision")
    if not extraction_id.startswith("xtr_"):
        raise typer.BadParameter("review requires an xtr_ ID")
    settings = Settings()
    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    try:
        with database.session_factory.begin() as session:
            try:
                result = GraphRepository().review_extraction(
                    session,
                    extraction_id,
                    decision=decision,
                    run_id=run_id,
                    notes=note,
                )
            except Exception as error:
                console.print(f"[red]{type(error).__name__}:[/red] {error}")
                raise typer.Exit(code=1) from error
    finally:
        database.dispose()
    console.print_json(json.dumps(result, ensure_ascii=False, default=_json_default))


@app.command()
def stats(
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite file")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    settings = Settings()
    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    try:
        with database.session_factory() as session:
            result = IngestRepository().stats(session)
    finally:
        database.dispose()

    if json_output:
        console.print_json(result.model_dump_json())
        return
    table = Table(title="Index statistics")
    table.add_column("Documents", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Average", justify="right")
    table.add_column("Max", justify="right")
    table.add_row(
        str(result.documents),
        str(result.chunks),
        str(result.total_tokens),
        str(result.min_chunk_tokens or 0),
        f"{result.average_chunk_tokens or 0:.1f}",
        str(result.max_chunk_tokens or 0),
    )
    console.print(table)


@app.command("inspect")
def inspect_document(
    document_id: Annotated[str, typer.Argument(help="Document ID")],
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite file")] = None,
) -> None:
    settings = Settings()
    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    try:
        with database.session_factory() as session:
            document = IngestRepository().get_document(session, document_id)
            if document is None:
                console.print(f"[red]Document not found:[/red] {document_id}")
                raise typer.Exit(code=1)
            payload = {
                "id": document.id,
                "title": document.title,
                "source_uri": document.source_uri,
                "content_hash": document.content_hash,
                "parser": f"{document.parser_name}@{document.parser_version}",
                "chunks": [
                    {
                        "id": chunk.id,
                        "ordinal": chunk.ordinal,
                        "section_path": chunk.section_path_json,
                        "pages": [chunk.page_start, chunk.page_end],
                        "char_span": [chunk.char_start, chunk.char_end],
                        "token_count": chunk.token_count,
                        "text": chunk.text,
                    }
                    for chunk in document.chunks
                ],
            }
    finally:
        database.dispose()
    console.print_json(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    app()
