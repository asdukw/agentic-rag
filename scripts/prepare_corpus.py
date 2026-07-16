r"""Prepare a corpus for RAG and print its reusable index identity.

The script runs ``ingest -> build graph -> build index``. After ingestion it
checks SQLite for an active graph-backed index. If the unchanged corpus already
has one, graph extraction and embedding are skipped and the stored hashes are
printed directly.

Usage::

    # Prepare data/corpus in the default storage/app.db database.
    uv run scripts/prepare_corpus.py

    # Prepare another corpus and SQLite database.
    uv run scripts/prepare_corpus.py \
      --source-dir storage/workspaces/<workspace-id>/uploads \
      --db storage/workspaces/<workspace-id>/workspace.db

    # Deliberately create a fresh graph run and rebuild the index.
    uv run scripts/prepare_corpus.py --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from hybrid_rag.config import Settings, sqlite_url
from hybrid_rag.evaluation.testset_contract import validate_corpus_content_hash
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.retrieval_repository import RetrievalRepository, StoredIndexProfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = ROOT / "data" / "corpus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Source document directory (default: data/corpus).",
    )
    parser.add_argument("--db", type=Path, default=None, help="SQLite database file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run graph build and force index embedding even when an active index exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database_url = sqlite_url(args.db) if args.db is not None else Settings().database_url
    database_args = ["--db", str(args.db)] if args.db is not None else []

    _run_cli("ingest", str(args.source_dir), *database_args)
    if not args.force:
        existing = _active_graph_index(database_url)
        if existing is not None:
            _print_index(existing, database_url=database_url, status="reused")
            return

    _run_cli("build-graph", *database_args)
    index_args = ["build-index", *database_args]
    if args.force:
        index_args.append("--force")
    _run_cli(*index_args)

    profile = _active_graph_index(database_url)
    if profile is None:
        raise RuntimeError("build-index completed without an active graph-backed index profile")
    _print_index(profile, database_url=database_url, status="built")


def _run_cli(*args: str) -> None:
    command = [sys.executable, "-m", "hybrid_rag.cli", *args]
    subprocess.run(command, check=True, cwd=ROOT)


def _active_graph_index(database_url: str) -> StoredIndexProfile | None:
    database = Database(database_url)
    try:
        with database.session_factory() as session:
            repository = RetrievalRepository()
            profile = repository.get_profile(session)
            snapshot = repository.load_source_snapshot(session) if profile is not None else None
    finally:
        database.dispose()
    if profile is None or profile.source_graph_run_id is None:
        return None
    profile_corpus_hash = validate_corpus_content_hash(
        profile.metadata.get("corpus_content_hash"),
        field="active index corpus_content_hash",
    )
    if snapshot is None:
        return None
    if profile.source_corpus_hash != snapshot.source_corpus_hash:
        return None
    if profile_corpus_hash != snapshot.corpus_content_hash:
        return None
    return profile


def _print_index(
    profile: StoredIndexProfile,
    *,
    database_url: str,
    status: str,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "database_url": database_url,
        "profile_id": profile.id,
        "index_config_hash": profile.config_hash,
        "corpus_content_hash": profile.metadata["corpus_content_hash"],
        "graph_build_run_id": profile.source_graph_run_id,
        "provider": profile.provider,
        "model": profile.model,
        "dimensions": profile.dimensions,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
