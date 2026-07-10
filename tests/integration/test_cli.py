from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import select
from typer.testing import CliRunner

import hybrid_rag.cli as cli_module
from hybrid_rag.cli import app
from hybrid_rag.config import sqlite_url
from hybrid_rag.corpus import DownloadReport, DownloadResult
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.models import DocumentRecord


class WordCounter:
    name = "test-words"

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


def test_cli_ingest_stats_and_inspect(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    corpus = Path(__file__).parents[1] / "fixtures" / "corpus"
    db_path = tmp_path / "cli.db"
    monkeypatch.setattr(cli_module, "TiktokenCounter", lambda _: WordCounter())

    first = runner.invoke(
        app,
        [
            "ingest",
            str(corpus),
            "--db",
            str(db_path),
            "--chunk-size",
            "32",
            "--overlap",
            "4",
            "--json",
        ],
    )
    second = runner.invoke(
        app,
        [
            "ingest",
            str(corpus),
            "--db",
            str(db_path),
            "--chunk-size",
            "32",
            "--overlap",
            "4",
            "--json",
        ],
    )
    stats = runner.invoke(app, ["stats", "--db", str(db_path), "--json"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert '"inserted": 2' in first.output
    assert '"skipped": 2' in second.output
    assert '"documents": 2' in stats.output

    database = Database(sqlite_url(db_path))
    try:
        with database.session_factory() as session:
            document_id = session.scalar(select(DocumentRecord.id).limit(1))
    finally:
        database.dispose()
    assert document_id is not None

    inspected = runner.invoke(app, ["inspect", document_id, "--db", str(db_path)])
    assert inspected.exit_code == 0, inspected.output
    assert document_id in inspected.output
    assert "char_span" in inspected.output


def test_cli_corpus_download_renders_report(tmp_path: Path, monkeypatch) -> None:
    report = DownloadReport(
        manifest=str(tmp_path / "manifest.json"),
        destination=str(tmp_path / "raw"),
        results=[
            DownloadResult(
                arxiv_id="2410.05779v3",
                path=str(tmp_path / "raw" / "2410.05779v3.pdf"),
                status="downloaded",
                size_bytes=1024,
                sha256="a" * 64,
            )
        ],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_module, "download_manifest", lambda *_args, **_kwargs: report)

    result = CliRunner().invoke(
        app,
        ["corpus", "download", "--manifest", str(manifest), "--output", str(tmp_path / "raw")],
    )

    assert result.exit_code == 0, result.output
    assert "downloaded=1 skipped=0 failed=0" in result.output
