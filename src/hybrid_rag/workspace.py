"""Local filesystem-backed workspaces for the web workbench."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.config import sqlite_url

_WORKSPACE_ID = re.compile(r"^ws_[0-9a-f]{16}$")
_SUPPORTED_SUFFIXES = frozenset({".pdf", ".md", ".markdown", ".txt"})


class Workspace(BaseModel):
    """Public, local-only metadata for one isolated RAG corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str = Field(min_length=1, max_length=100)
    created_at: datetime
    uploads: tuple[str, ...] = ()


class WorkspaceStore:
    """Own workspace paths and metadata without introducing another database."""

    def __init__(self, root: Path = Path("storage/workspaces")) -> None:
        self.root = root.expanduser().resolve()

    def list(self) -> tuple[Workspace, ...]:
        if not self.root.exists():
            return ()
        values: list[Workspace] = []
        for metadata_path in self.root.glob("ws_*/workspace.json"):
            try:
                values.append(self._read(metadata_path.parent))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return tuple(sorted(values, key=lambda item: item.created_at, reverse=True))

    def create(self, name: str) -> Workspace:
        normalized = name.strip()
        if not normalized:
            raise ValueError("workspace name must not be blank")
        workspace_id = f"ws_{uuid4().hex[:16]}"
        directory = self._directory(workspace_id)
        (directory / "uploads").mkdir(parents=True, exist_ok=False)
        metadata = {
            "id": workspace_id,
            "name": normalized,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._metadata_path(directory).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self._read(directory)

    def get(self, workspace_id: str) -> Workspace:
        return self._read(self._directory(workspace_id))

    def database_url(self, workspace_id: str) -> str:
        self.get(workspace_id)
        return sqlite_url(self._directory(workspace_id) / "workspace.db")

    def checkpoint_path(self, workspace_id: str) -> Path:
        self.get(workspace_id)
        return self._directory(workspace_id) / "langgraph.db"

    def uploads_path(self, workspace_id: str) -> Path:
        self.get(workspace_id)
        return self._directory(workspace_id) / "uploads"

    def store_upload(self, workspace_id: str, filename: str, content: bytes) -> Workspace:
        if not content:
            raise ValueError("uploaded file is empty")
        clean_name = Path(filename).name
        if not clean_name or Path(clean_name).suffix.lower() not in _SUPPORTED_SUFFIXES:
            supported = ", ".join(sorted(_SUPPORTED_SUFFIXES))
            raise ValueError(f"unsupported upload type; expected one of: {supported}")
        destination = self.uploads_path(workspace_id) / clean_name
        temporary = destination.with_name(f".{clean_name}.{uuid4().hex}.part")
        temporary.write_bytes(content)
        temporary.replace(destination)
        return self.get(workspace_id)

    def _read(self, directory: Path) -> Workspace:
        try:
            raw = json.loads(self._metadata_path(directory).read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"workspace not found: {directory.name}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"workspace metadata is invalid: {directory.name}")
        uploads = tuple(
            path.name for path in sorted((directory / "uploads").glob("*")) if path.is_file()
        )
        return Workspace.model_validate({**raw, "uploads": uploads})

    def _directory(self, workspace_id: str) -> Path:
        if not _WORKSPACE_ID.fullmatch(workspace_id):
            raise ValueError("invalid workspace ID")
        return self.root / workspace_id

    @staticmethod
    def _metadata_path(directory: Path) -> Path:
        return directory / "workspace.json"
