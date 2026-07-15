"""Lightweight JSON contract shared by Ragas test-set generation and evaluation."""

from __future__ import annotations

import re
from collections.abc import Mapping

RAGAS_TESTSET_SCHEMA_VERSION = "1"
_SHA256_HEX = re.compile(r"^[a-f0-9]{64}$")


def validate_corpus_content_hash(value: object, *, field: str = "corpus_content_hash") -> str:
    """Validate the lowercase corpus hash shared by generation and evaluation."""

    if not isinstance(value, str):
        raise ValueError(f"{field} must be a 64-character hexadecimal digest in lowercase")
    normalized = value.strip()
    if not _SHA256_HEX.fullmatch(normalized):
        raise ValueError(f"{field} must be a 64-character hexadecimal digest in lowercase")
    return normalized


def validate_testset_sources(value: object) -> list[dict[str, object]]:
    """Validate and normalize the public source citations carried by a test set."""

    if not isinstance(value, list) or not value:
        raise ValueError("Ragas test set sources must be a non-empty JSON array")
    sources: list[dict[str, object]] = []
    source_uris: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Ragas source {index} must be an object")
        source_uri = _required_string(item, "source_uri", index)
        if source_uri in source_uris:
            raise ValueError(f"Ragas source {index} duplicates source_uri {source_uri!r}")
        source_uris.add(source_uri)
        authors = item.get("authors")
        if (
            not isinstance(authors, list)
            or not authors
            or not all(isinstance(author, str) and author.strip() for author in authors)
        ):
            raise ValueError(f"Ragas source {index} authors must be non-empty strings")
        year = item.get("year")
        if isinstance(year, bool) or not isinstance(year, int) or year < 1900:
            raise ValueError(f"Ragas source {index} year must be a valid integer")
        official_url = _required_string(item, "official_url", index)
        if not official_url.startswith("https://"):
            raise ValueError(f"Ragas source {index} official_url must use HTTPS")
        sources.append(
            {
                "source_uri": source_uri,
                "filename": _required_string(item, "filename", index),
                "title": _required_string(item, "title", index),
                "authors": [author.strip() for author in authors],
                "venue": _required_string(item, "venue", index),
                "year": year,
                "pages": _required_string(item, "pages", index),
                "official_url": official_url,
                "rights_notice": _required_string(item, "rights_notice", index),
            }
        )
    return sources


def _required_string(value: Mapping[object, object], field: str, index: int) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Ragas source {index} {field} must be a non-empty string")
    return item.strip()


__all__ = [
    "RAGAS_TESTSET_SCHEMA_VERSION",
    "validate_corpus_content_hash",
    "validate_testset_sources",
]
