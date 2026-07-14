"""Lightweight JSON contract shared by Ragas test-set generation and evaluation."""

from __future__ import annotations

import re

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


__all__ = ["RAGAS_TESTSET_SCHEMA_VERSION", "validate_corpus_content_hash"]
