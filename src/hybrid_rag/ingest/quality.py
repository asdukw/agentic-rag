from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

ChunkQuality = Literal[
    "normal",
    "references",
    "acknowledgements",
    "copyright",
    "author_affiliation",
    "visualization_label",
]

CHUNK_QUALITY_CLASSIFIER_NAME = "rule-based-chunk-quality"
CHUNK_QUALITY_CLASSIFIER_VERSION = "1"
CHUNK_QUALITY_CLASSES = frozenset(
    {
        "normal",
        "references",
        "acknowledgements",
        "copyright",
        "author_affiliation",
        "visualization_label",
    }
)

_REFERENCE_SECTIONS = {
    "references",
    "bibliography",
    "works cited",
    "参考文献",
}
_ACKNOWLEDGEMENT_SECTIONS = {
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "致谢",
}
_VISUALIZATION_SECTIONS = {
    "attention visualization",
    "attention visualizations",
    "visualization",
    "visualizations",
}
_REFERENCE_ENTRY = re.compile(r"(?:^|\s)\[\d+\]\s+(?=[A-Z])")
_COPYRIGHT_NOTICE = re.compile(
    r"(?:©|copyright|all rights reserved|permission to (?:copy|reproduce)|"
    r"provided proper attribution|版权所有|保留所有权利)",
    re.IGNORECASE,
)
_AUTHOR_AFFILIATION = re.compile(
    r"(?:\b(?:university|institute|department|laboratory|research lab|equal contribution)\b|"
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b)",
    re.IGNORECASE,
)
_VISUALIZATION_LABEL = re.compile(
    r"(?:<\s*(?:eos|bos|pad)\s*>|attention visuali[sz]ations?|input[- ]input layer\d*)",
    re.IGNORECASE,
)


def classify_chunk_quality(
    *,
    section_path: Sequence[str],
    text: str,
    ordinal: int,
    page_start: int | None,
) -> ChunkQuality:
    """Assign one deterministic quality class without discarding source text."""

    sections = {_normalize_section(section) for section in section_path}
    if sections & _REFERENCE_SECTIONS:
        return "references"
    if sections & _ACKNOWLEDGEMENT_SECTIONS:
        return "acknowledgements"
    if sections & _VISUALIZATION_SECTIONS:
        return "visualization_label"

    stripped = text.strip()
    if len(_REFERENCE_ENTRY.findall(stripped)) >= 3:
        return "references"
    if _COPYRIGHT_NOTICE.search(stripped):
        return "copyright"
    if (
        ordinal <= 1
        and page_start in (None, 1)
        and not sections
        and len(_AUTHOR_AFFILIATION.findall(stripped)) >= 2
    ):
        return "author_affiliation"
    if _VISUALIZATION_LABEL.search(stripped):
        return "visualization_label"
    return "normal"


def _normalize_section(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().casefold()
    return re.sub(r"^(?:\d+(?:\.\d+)*|[ivxlcdm]+)[.)]?\s+", "", normalized)


__all__ = [
    "CHUNK_QUALITY_CLASSES",
    "CHUNK_QUALITY_CLASSIFIER_NAME",
    "CHUNK_QUALITY_CLASSIFIER_VERSION",
    "ChunkQuality",
    "classify_chunk_quality",
]
