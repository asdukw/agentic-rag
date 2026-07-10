from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: str, length: int = 20) -> str:
    digest = sha256_text("\x1f".join(parts))[:length]
    return f"{prefix}_{digest}"


def canonical_json_hash(value: dict[str, Any]) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(serialized)


def file_source_uri(path: Path, root: Path) -> str:
    """Return a stable, root-relative source URI.

    Case-folding keeps IDs stable on Windows' case-insensitive filesystem while
    the original path remains available in ``local_path``.
    """

    resolved_root = root.resolve()
    relative = path.resolve().relative_to(resolved_root).as_posix()
    root_label = resolved_root.name or "root"
    return f"file:{root_label.casefold()}/{relative.casefold()}"
