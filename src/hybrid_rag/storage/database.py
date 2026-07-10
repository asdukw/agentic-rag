from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from hybrid_rag.storage.models import Base


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        _ensure_sqlite_parent(url)
        self.engine = create_engine(url, future=True)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", _enable_sqlite_foreign_keys)
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
        )

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()


def _ensure_sqlite_parent(url: str) -> None:
    parsed = make_url(url)
    if parsed.drivername != "sqlite" or not parsed.database or parsed.database == ":memory:":
        return
    Path(parsed.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def sqlite_foreign_keys_enabled(engine: Engine) -> bool:
    with engine.connect() as connection:
        return bool(connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one())
