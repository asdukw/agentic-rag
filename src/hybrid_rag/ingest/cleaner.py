from __future__ import annotations

import re
import unicodedata

from hybrid_rag.schemas import ParsedDocument, TextSegment

CLEANER_NAME = "conservative-text-cleaner"
CLEANER_VERSION = "2"

_SOFT_HYPHEN = "\u00ad"
_LINE_END_HYPHEN = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])")
_SINGLE_LINE_BREAK = re.compile(r"(?<!\n)\n(?!\n)")
_HORIZONTAL_SPACE = re.compile(r"[\t\v\f\r ]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Apply conservative, deterministic cleanup without deleting content."""

    value = unicodedata.normalize("NFC", text).replace(_SOFT_HYPHEN, "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    # Keep the hyphen: without a language-aware dictionary we cannot safely
    # distinguish a split word from a legitimate compound such as "graph-based".
    value = _LINE_END_HYPHEN.sub("-", value)
    value = _SINGLE_LINE_BREAK.sub(" ", value)
    value = _HORIZONTAL_SPACE.sub(" ", value)
    value = "\n".join(line.strip() for line in value.splitlines())
    value = _EXCESS_BLANK_LINES.sub("\n\n", value)
    return value.strip()


def clean_document(document: ParsedDocument) -> ParsedDocument:
    """Clean segments and assign offsets into the reconstructed document text."""

    cleaned: list[TextSegment] = []
    text_parts: list[str] = []
    cursor = 0

    for segment in document.segments:
        value = clean_text(segment.text)
        if not value:
            continue
        if text_parts:
            cursor += 2  # ``\n\n`` between source segments.
        start = cursor
        cursor += len(value)
        cleaned.append(
            segment.model_copy(
                update={"text": value, "char_start": start, "char_end": cursor}
            )
        )
        text_parts.append(value)

    return document.model_copy(update={"text": "\n\n".join(text_parts), "segments": cleaned})
