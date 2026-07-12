from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hybrid_rag.config import DeepSeekSettings, GraphSettings, Settings, sqlite_url
from hybrid_rag.corpus import download_manifest
from hybrid_rag.extraction.client import DeepSeekClient
from hybrid_rag.extraction.reports import GraphBuildReport, GraphStorageStats
from hybrid_rag.extraction.schemas import ExtractionConfig, GraphConfig
from hybrid_rag.extraction.service import GraphBuildService
from hybrid_rag.extraction.workflow import WorkflowOptions
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.ingest.tokenizer import TiktokenCounter
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.graph_repository import GraphRepository
from hybrid_rag.storage.migrations import upgrade_database
from hybrid_rag.storage.repository import IngestRepository

app = typer.Typer(no_args_is_help=True, help="Hybrid RAG development CLI")
db_app = typer.Typer(no_args_is_help=True, help="Database schema commands")
corpus_app = typer.Typer(no_args_is_help=True, help="Versioned public corpus commands")
graph_app = typer.Typer(no_args_is_help=True, help="Graph extraction and inspection commands")
app.add_typer(db_app, name="db")
app.add_typer(corpus_app, name="corpus")
app.add_typer(graph_app, name="graph")
console = Console()


def _database_url(db_path: Path | None, settings: Settings) -> str:
    return sqlite_url(db_path) if db_path is not None else settings.database_url


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
