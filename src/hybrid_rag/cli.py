from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hybrid_rag.config import (
    DeepSeekSettings,
    EvaluationSettings,
    GraphSettings,
    RetrievalSettings,
    Settings,
    sqlite_url,
)
from hybrid_rag.corpus import download_manifest
from hybrid_rag.evaluation import (
    CostDisclosure,
    CostStatus,
    EvaluationOptions,
    EvaluationReport,
    EvaluationRunner,
    load_benchmark,
)
from hybrid_rag.evaluation import (
    write_json as write_evaluation_json,
)
from hybrid_rag.evaluation import (
    write_markdown as write_evaluation_markdown,
)
from hybrid_rag.evaluation.deepseek_judge import DeepSeekBlindJudge
from hybrid_rag.extraction.client import DeepSeekClient
from hybrid_rag.extraction.reports import GraphBuildReport, GraphStorageStats
from hybrid_rag.extraction.schemas import ExtractionConfig, GraphConfig
from hybrid_rag.extraction.service import GraphBuildService
from hybrid_rag.extraction.workflow import WorkflowOptions
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.ingest.tokenizer import TiktokenCounter
from hybrid_rag.retrieval.embedding import (
    EmbeddingConfigurationError,
    HashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from hybrid_rag.retrieval.models import IndexBuildReport, RetrievalMode, RetrievalResult
from hybrid_rag.retrieval.query import DeepSeekQueryClient, QueryClient
from hybrid_rag.retrieval.service import AnswerResult, RetrievalOptions, RetrievalService
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.graph_repository import GraphRepository
from hybrid_rag.storage.migrations import upgrade_database
from hybrid_rag.storage.repository import IngestRepository
from hybrid_rag.storage.retrieval_repository import RetrievalRepository

app = typer.Typer(no_args_is_help=True, help="Hybrid RAG development CLI")
db_app = typer.Typer(no_args_is_help=True, help="Database schema commands")
corpus_app = typer.Typer(no_args_is_help=True, help="Versioned public corpus commands")
graph_app = typer.Typer(no_args_is_help=True, help="Graph extraction and inspection commands")
retrieval_app = typer.Typer(
    no_args_is_help=True,
    help="Retrieval index inspection and trace replay",
)
app.add_typer(db_app, name="db")
app.add_typer(corpus_app, name="corpus")
app.add_typer(graph_app, name="graph")
app.add_typer(retrieval_app, name="retrieval")
console = Console()


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
    if selected_provider == "hash":
        return HashEmbeddingProvider(dimensions=selected_dimensions, model=selected_model)
    if selected_provider == "openai-compatible":
        if not settings.embedding_base_url:
            raise typer.BadParameter(
                "HYBRID_RAG_RETRIEVAL_EMBEDDING_BASE_URL is required for openai-compatible"
            )
        api_key = (
            settings.embedding_api_key.get_secret_value().strip()
            if settings.embedding_api_key
            else ""
        )
        return OpenAICompatibleEmbeddingProvider(
            api_key=api_key or None,
            base_url=settings.embedding_base_url,
            model=selected_model,
            dimensions=selected_dimensions,
        )
    raise typer.BadParameter(
        "--provider must be 'hash' or 'openai-compatible'",
        param_hint="--provider",
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
        naive_weight=settings.naive_weight,
        local_weight=settings.local_weight,
        global_weight=settings.global_weight,
        naive_dense_weight=settings.naive_dense_weight,
        naive_bm25_weight=settings.naive_bm25_weight,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
    )


def _deepseek_query_client() -> DeepSeekQueryClient:
    settings = DeepSeekSettings()
    api_key = settings.api_key.get_secret_value().strip() if settings.api_key else ""
    if not api_key:
        raise typer.BadParameter("--deepseek requires DEEPSEEK_API_KEY")
    return DeepSeekQueryClient(
        api_key=api_key,
        model=settings.query_model,
        answer_model=settings.answer_model,
        base_url=settings.base_url,
        max_output_tokens=max(512, settings.answer_max_output_tokens),
        timeout_seconds=settings.timeout_seconds,
    )


def _deepseek_blind_judge() -> DeepSeekBlindJudge:
    settings = DeepSeekSettings()
    api_key = settings.api_key.get_secret_value().strip() if settings.api_key else ""
    if not api_key:
        raise typer.BadParameter("--deepseek-judge requires DEEPSEEK_API_KEY")
    return DeepSeekBlindJudge(
        api_key=api_key,
        model=settings.judge_model,
        base_url=settings.base_url,
        max_output_tokens=settings.judge_max_output_tokens,
        timeout_seconds=settings.timeout_seconds,
    )


def _evaluation_options(
    evaluation: EvaluationSettings,
    retrieval: RetrievalSettings,
    *,
    modes: str,
    top_k: int | None,
    context_tokens: int | None,
    graph_hops: int | None,
    limit: int | None,
    benchmark_case_ids: tuple[str, ...],
) -> EvaluationOptions:
    try:
        selected_modes = tuple(
            RetrievalMode(value.strip()) for value in modes.split(",") if value.strip()
        )
    except ValueError as error:
        raise typer.BadParameter("--modes must use naive, local, global, and/or hybrid") from error
    if not selected_modes:
        raise typer.BadParameter("--modes must not be empty")
    case_ids = benchmark_case_ids[:limit] if limit is not None else benchmark_case_ids
    return EvaluationOptions(
        modes=selected_modes,
        top_k=top_k or evaluation.top_k,
        candidate_multiplier=retrieval.candidate_multiplier,
        context_token_budget=context_tokens or evaluation.context_token_budget,
        graph_max_hops=graph_hops or evaluation.graph_max_hops,
        naive_weight=retrieval.naive_weight,
        local_weight=retrieval.local_weight,
        global_weight=retrieval.global_weight,
        naive_dense_weight=retrieval.naive_dense_weight,
        naive_bm25_weight=retrieval.naive_bm25_weight,
        bm25_k1=retrieval.bm25_k1,
        bm25_b=retrieval.bm25_b,
        case_ids=case_ids,
    )


def _judge_cost_disclosure(
    judge: DeepSeekBlindJudge,
    settings: EvaluationSettings,
    report: EvaluationReport,
) -> CostDisclosure:
    usage = judge.usage
    if any(judgment.used_fallback for judgment in report.pairwise_judgments):
        return CostDisclosure.unknown_judge_fallback(
            retrieval_model_calls=report.cost_disclosure.retrieval_model_calls,
            judge_model_calls=usage.calls,
        )
    if report.run.index_provenance.embedding_provider.casefold() != "hash":
        return CostDisclosure.unknown_external_embedding(
            provider=report.run.index_provenance.embedding_provider,
            retrieval_model_calls=report.cost_disclosure.retrieval_model_calls or 0,
            judge_model_calls=usage.calls,
        )
    input_price = settings.input_cost_usd_per_million_tokens
    output_price = settings.output_cost_usd_per_million_tokens
    if input_price is None or output_price is None:
        return CostDisclosure(
            status=CostStatus.UNKNOWN,
            retrieval_model_calls=0,
            judge_model_calls=usage.calls,
            cost_usd=None,
            price_assumption=(
                "DeepSeek blind judge was requested, but no verified per-million-token pricing "
                "was configured"
            ),
        )
    cost = (usage.prompt_tokens * input_price + usage.completion_tokens * output_price) / 1_000_000
    return CostDisclosure(
        status=CostStatus.ESTIMATED,
        retrieval_model_calls=0,
        judge_model_calls=usage.calls,
        cost_usd=cost,
        price_assumption=(
            "User-configured DeepSeek judge price assumption: "
            f"input=${input_price}/M tokens, output=${output_price}/M tokens"
        ),
    )


async def _close_query_client(client: QueryClient | None) -> None:
    if client is None:
        return
    close = getattr(client, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


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
    if result.trace_id:
        console.print(f"Replay with: hybrid-rag retrieval replay {result.trace_id}")
    console.print(result.context or "[yellow]No evidence fit the context budget.[/yellow]")


def _render_answer_result(result: AnswerResult, *, json_output: bool) -> None:
    if json_output:
        console.print_json(result.model_dump_json())
        return
    _render_retrieval_result(result.retrieval, json_output=False)
    console.print("\n[bold]Answer[/bold]")
    console.print(result.answer.answer)
    console.print(f"Citations: {', '.join(result.answer.citations) or '(insufficient evidence)'}")


def _render_evaluation_report(
    report: EvaluationReport,
    *,
    json_output: bool,
    json_path: Path,
    markdown_path: Path,
) -> None:
    if json_output:
        console.print_json(report.to_json())
        return
    table = Table(
        title=(
            f"Evaluation {report.run.id} / {report.run.execution_id} "
            f"({report.run.benchmark_id})"
        )
    )
    table.add_column("Mode")
    table.add_column("Evidence hit", justify="right")
    table.add_column("Cited hit", justify="right")
    table.add_column("Citation-grounded", justify="right")
    table.add_column("Mean retrieve ms", justify="right")
    table.add_column("Median retrieve ms", justify="right")
    for summary in report.summaries:
        table.add_row(
            summary.mode.value,
            f"{summary.mean_evidence_hit_rate:.3f}",
            f"{summary.mean_cited_evidence_hit_rate:.3f}",
            f"{summary.citation_grounded_faithfulness_rate:.3f}",
            f"{summary.mean_retrieval_latency_ms:.2f}",
            f"{summary.median_retrieval_latency_ms:.2f}",
        )
    console.print(table)
    comparison = report.comparison_summary
    console.print(
        "Blind comparison: "
        f"naive={comparison.naive_wins}, hybrid={comparison.hybrid_wins}, ties={comparison.ties}"
    )
    cost = report.cost_disclosure
    console.print(
        f"Cost: status={cost.status.value}, usd={cost.cost_usd}, "
        f"retrieval_calls={cost.retrieval_model_calls}, judge_calls={cost.judge_model_calls}"
    )
    console.print(
        "Pinned profile: "
        f"{report.run.index_provenance.profile_id} "
        f"(content={report.run.index_provenance.corpus_content_hash}, "
        f"snapshot={report.run.index_provenance.source_corpus_hash})"
    )
    console.print(f"Wrote: [cyan]{json_path}[/cyan] and [cyan]{markdown_path}[/cyan]")


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
            return ExtractionConfig.model_validate(persisted)
    return ExtractionConfig(
        base_url=base_url.rstrip("/"),
        model=model,
        max_output_tokens=max_output_tokens,
        repair_max_attempts=max_attempts - 1,
    )


async def _run_graph_build(
    service: GraphBuildService,
    client: DeepSeekClient | None,
    options: WorkflowOptions,
    resume_run_id: str | None,
) -> GraphBuildReport:
    try:
        return await service.build(options, resume_run_id=resume_run_id)
    finally:
        if client is not None:
            await client.close()


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
    chunks.add_row(
        str(report.chunks.total),
        str(report.chunks.cached),
        str(report.chunks.scheduled),
        str(report.chunks.succeeded),
        str(report.chunks.needs_review),
        str(report.chunks.failed),
        str(report.attempts.total),
    )
    console.print(chunks)
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


@corpus_app.command("download")
def corpus_download(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, readable=True)] = Path(
        "data/corpus.json"
    ),
    output: Annotated[Path, typer.Option("--output")] = Path("data/raw"),
    delay: Annotated[
        float,
        typer.Option("--delay", min=0, help="Polite delay between arXiv requests"),
    ] = 3.0,
) -> None:
    report = download_manifest(manifest, output, delay_seconds=delay)
    table = Table(title="Corpus download")
    table.add_column("arXiv")
    table.add_column("Status")
    table.add_column("MiB", justify="right")
    table.add_column("Path")
    for result in report.results:
        style = {"downloaded": "green", "skipped": "cyan", "failed": "red"}[result.status]
        table.add_row(
            result.arxiv_id,
            f"[{style}]{result.status}[/{style}]",
            f"{result.size_bytes / 1024 / 1024:.2f}",
            result.path if result.error is None else result.error,
        )
    console.print(table)
    console.print(f"downloaded={report.downloaded} skipped={report.skipped} failed={report.failed}")
    if report.failed:
        raise typer.Exit(code=1)


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
        typer.Option("--retry-failed", help="Explicitly requeue terminal failed extractions"),
    ] = False,
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
    json_output: Annotated[bool, typer.Option("--json", help="Print a JSON report")] = False,
) -> None:
    if resume is not None and (
        model is not None or max_attempts is not None or limit is not None or retry_failed or review
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
        )
        try:
            report = asyncio.run(_run_graph_build(service, client, options, resume))
        except KeyboardInterrupt as error:
            run_id = service.last_run_id or resume
            console.print("[yellow]Graph build interrupted after durable checkpoints.[/yellow]")
            if run_id:
                console.print(f"Resume with: hybrid-rag build-graph --resume {run_id}")
            raise typer.Exit(code=130) from error
        except Exception as error:
            run_id = service.last_run_id or resume
            console.print(f"[red]{type(error).__name__}:[/red] {error}")
            if run_id:
                console.print(f"Resume with: hybrid-rag build-graph --resume {run_id}")
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
        typer.Option("--provider", help="hash (default) or openai-compatible"),
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
    mode: Annotated[str, typer.Option("--mode", help="naive, local, global, or hybrid")] = "hybrid",
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
        selected_mode = RetrievalMode(mode)
    except ValueError as error:
        raise typer.BadParameter("--mode must be naive, local, global, or hybrid") from error
    settings = Settings()
    retrieval_settings = RetrievalSettings()
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
    mode: Annotated[str, typer.Option("--mode", help="naive, local, global, or hybrid")] = "hybrid",
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="idx_ profile ID or config hash"),
    ] = None,
    top: Annotated[int | None, typer.Option("--top", min=1)] = None,
    context_tokens: Annotated[int | None, typer.Option("--context-tokens", min=1)] = None,
    graph_hops: Annotated[int | None, typer.Option("--graph-hops", min=1, max=4)] = None,
    deepseek: Annotated[
        bool,
        typer.Option("--deepseek", help="Use DeepSeek for keywords and a citation-bound answer"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print JSON answer, evidence, and trace"),
    ] = False,
) -> None:
    try:
        selected_mode = RetrievalMode(mode)
    except ValueError as error:
        raise typer.BadParameter("--mode must be naive, local, global, or hybrid") from error
    settings = Settings()
    retrieval_settings = RetrievalSettings()
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
        )
        options = _retrieval_options(
            retrieval_settings,
            top_k=top,
            context_tokens=context_tokens,
            graph_hops=graph_hops,
        )

        async def run() -> AnswerResult:
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

        result = asyncio.run(run())
    except (EmbeddingConfigurationError, ValueError, RuntimeError) as error:
        console.print(f"[red]{type(error).__name__}:[/red] {error}")
        raise typer.Exit(code=1) from error
    finally:
        database.dispose()
    _render_answer_result(result, json_output=json_output)


@app.command("evaluate")
def evaluate(
    benchmark_path: Annotated[
        Path | None,
        typer.Option("--benchmark", exists=True, readable=True, help="Versioned benchmark JSON"),
    ] = None,
    db_path: Annotated[Path | None, typer.Option("--db", help="SQLite file")] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="idx_ profile ID or config hash; pin it for all cases"),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Directory for JSON and Markdown reports"),
    ] = None,
    modes: Annotated[
        str,
        typer.Option("--modes", help="Comma-separated modes; must include naive and hybrid"),
    ] = "naive,hybrid",
    top: Annotated[int | None, typer.Option("--top", min=1)] = None,
    context_tokens: Annotated[int | None, typer.Option("--context-tokens", min=1)] = None,
    graph_hops: Annotated[int | None, typer.Option("--graph-hops", min=1, max=4)] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Prefix of fixed cases"),
    ] = None,
    deepseek_judge: Annotated[
        bool,
        typer.Option(
            "--deepseek-judge",
            help="Use a blind DeepSeek Judge; fallback stays explicit",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print the full JSON report")] = False,
) -> None:
    settings = Settings()
    retrieval_settings = RetrievalSettings()
    evaluation_settings = EvaluationSettings()
    selected_benchmark_path = benchmark_path or evaluation_settings.benchmark_path
    try:
        benchmark = load_benchmark(selected_benchmark_path)
        options = _evaluation_options(
            evaluation_settings,
            retrieval_settings,
            modes=modes,
            top_k=top,
            context_tokens=context_tokens,
            graph_hops=graph_hops,
            limit=limit or evaluation_settings.max_questions,
            benchmark_case_ids=tuple(case.id for case in benchmark.cases),
        )
    except (ValueError, OSError) as error:
        console.print(f"[red]{type(error).__name__}:[/red] {error}")
        raise typer.Exit(code=1) from error

    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    try:
        judge = _deepseek_blind_judge() if deepseek_judge else None
        service = RetrievalService(
            database,
            _embedding_provider(retrieval_settings),
            TiktokenCounter(settings.tokenizer_name),
        )
        report = EvaluationRunner(service, judge=judge).run(
            benchmark,
            options=options,
            profile_ref=profile,
        )
        if judge is not None:
            report = report.model_copy(
                update={
                    "cost_disclosure": _judge_cost_disclosure(
                        judge,
                        evaluation_settings,
                        report,
                    )
                }
            )
        destination = output_dir or evaluation_settings.output_dir
        artifact_stem = f"{report.run.id}-{report.run.execution_id}"
        json_path = write_evaluation_json(report, destination / f"{artifact_stem}.json")
        markdown_path = write_evaluation_markdown(report, destination / f"{artifact_stem}.md")
    except (EmbeddingConfigurationError, OSError, RuntimeError, ValueError) as error:
        console.print(f"[red]{type(error).__name__}:[/red] {error}")
        raise typer.Exit(code=1) from error
    finally:
        database.dispose()
    _render_evaluation_report(
        report,
        json_output=json_output,
        json_path=json_path,
        markdown_path=markdown_path,
    )


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
    url = _database_url(db_path, settings)
    upgrade_database(url)
    database = Database(url)
    try:
        with database.session_factory() as session:
            result = GraphStorageStats.model_validate(
                GraphRepository().stats(session, run_id=run_id, top_k=top)
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
