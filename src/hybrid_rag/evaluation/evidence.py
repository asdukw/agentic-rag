"""Stable evidence locators shared by test-set generation and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hybrid_rag.ids import stable_id


def evidence_ids(
    document_id: str,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
    section_path: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return page/section evidence IDs that survive normal chunk-size changes.

    PDF evidence is identified at page-range granularity. Non-paginated sources
    fall back to their section path. Chunk text and chunk ordinal are deliberately
    excluded, because both change when the ingest chunking configuration changes.
    """

    normalized_document_id = document_id.strip()
    if not normalized_document_id:
        raise ValueError("document_id must be non-empty")
    if page_start is not None and page_start < 1:
        raise ValueError("page_start must be positive when provided")
    if page_end is not None and page_end < 1:
        raise ValueError("page_end must be positive when provided")
    if page_start is not None and page_end is not None and page_end < page_start:
        raise ValueError("page_end must not precede page_start")

    normalized_section = tuple(value.strip() for value in section_path if value.strip())
    first_page = page_start or page_end
    last_page = page_end or first_page
    if first_page is not None and last_page is not None:
        return tuple(
            stable_id("evd", normalized_document_id, f"page:{page}")
            for page in range(first_page, last_page + 1)
        )
    locator = f"section:{' > '.join(normalized_section)}"
    return (stable_id("evd", normalized_document_id, locator),)


def evidence_ids_from_metadata(metadata: Mapping[str, object]) -> tuple[str, ...]:
    """Build evidence IDs from loader or retrieval metadata."""

    document_id = metadata.get("document_id")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("evidence metadata requires a non-empty document_id")
    section_path = _section_path(metadata.get("section_path"))
    return evidence_ids(
        document_id,
        page_start=_optional_positive_int(metadata.get("page_start"), field="page_start"),
        page_end=_optional_positive_int(metadata.get("page_end"), field="page_end"),
        section_path=section_path,
    )


def _section_path(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(">") if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def _optional_positive_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer when provided")
    return value


__all__ = ["evidence_ids", "evidence_ids_from_metadata"]
