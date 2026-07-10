from pathlib import Path

from sqlalchemy import inspect

from hybrid_rag.config import sqlite_url
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database


def test_alembic_builds_empty_database(tmp_path: Path) -> None:
    url = sqlite_url(tmp_path / "migrated.db")

    upgrade_database(url)
    upgrade_database(url)
    database = Database(url)
    try:
        assert set(inspect(database.engine).get_table_names()) >= {
            "alembic_version",
            "documents",
            "chunks",
            "ingest_runs",
        }
    finally:
        database.dispose()
