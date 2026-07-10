from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hybrid_rag.config import Settings, sqlite_url
from hybrid_rag.corpus import download_manifest
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.ingest.tokenizer import TiktokenCounter
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database
from hybrid_rag.storage.repository import IngestRepository

app = typer.Typer(no_args_is_help=True, help="Hybrid RAG development CLI")
db_app = typer.Typer(no_args_is_help=True, help="Database schema commands")
corpus_app = typer.Typer(no_args_is_help=True, help="Versioned public corpus commands")
app.add_typer(db_app, name="db")
app.add_typer(corpus_app, name="corpus")
console = Console()


def _database_url(db_path: Path | None, settings: Settings) -> str:
    return sqlite_url(db_path) if db_path is not None else settings.database_url


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
    console.print(
        f"downloaded={report.downloaded} skipped={report.skipped} failed={report.failed}"
    )
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
            console.print(
                f"[red]{failure.path}[/red]: {failure.error_type}: {failure.message}"
            )

    if report.failed:
        raise typer.Exit(code=1)


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
