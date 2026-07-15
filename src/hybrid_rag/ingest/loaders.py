from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from math import ceil
from pathlib import Path

import pymupdf

from hybrid_rag.ids import sha256_file, stable_id
from hybrid_rag.schemas import ParsedDocument, TextSegment

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PDF_DECIMAL_HEADING = re.compile(r"^(\d+(?:\.\d+){0,5})(?:[、.)．]|\s+)")  # noqa: RUF001
_PDF_CHAPTER_HEADING = re.compile(r"^第[一二三四五六七八九十百千万零〇两\d]+([编篇部章节])")
_PDF_CHINESE_HEADING = re.compile(r"^[一二三四五六七八九十百]+[、.．)]")  # noqa: RUF001
_PDF_TRAILING_SENTENCE_MARK = re.compile(r"[。！？；.!?;：:]$")  # noqa: RUF001
_PDF_DIGITS = re.compile(r"\d+")
_PDF_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class _PdfLine:
    page: int
    page_height: float
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    bold: bool

    @property
    def position(self) -> float:
        return self.y0 / self.page_height if self.page_height else 0.5


class UnsupportedFileError(ValueError):
    pass


class DocumentLoader(ABC):
    suffixes: frozenset[str]
    parser_name: str
    parser_version: str

    @abstractmethod
    def load(self, path: Path, source_uri: str) -> ParsedDocument:
        raise NotImplementedError

    def _document(
        self,
        path: Path,
        source_uri: str,
        title: str,
        segments: list[TextSegment],
        metadata: dict[str, object] | None = None,
    ) -> ParsedDocument:
        return ParsedDocument(
            id=stable_id("doc", source_uri),
            title=title.strip() or path.stem,
            source_type=path.suffix.lower().lstrip("."),
            source_uri=source_uri,
            local_path=str(path.resolve()),
            content_hash=sha256_file(path),
            segments=segments,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            metadata=metadata or {},
        )


class TextLoader(DocumentLoader):
    suffixes = frozenset({".txt"})
    parser_name = "builtin-text"
    parser_version = "1"

    def load(self, path: Path, source_uri: str) -> ParsedDocument:
        text = path.read_text(encoding="utf-8-sig")
        segments = [TextSegment(text=value) for value in _paragraphs(text)]
        return self._document(path, source_uri, path.stem, segments)


class MarkdownLoader(DocumentLoader):
    suffixes = frozenset({".md", ".markdown"})
    parser_name = "builtin-markdown"
    parser_version = "1"

    def load(self, path: Path, source_uri: str) -> ParsedDocument:
        text = path.read_text(encoding="utf-8-sig")
        section_stack: list[str] = []
        buffer: list[str] = []
        segments: list[TextSegment] = []
        title = path.stem

        def flush() -> None:
            value = "\n".join(buffer).strip()
            if value:
                segments.append(TextSegment(text=value, section_path=tuple(section_stack)))
            buffer.clear()

        for line in text.splitlines():
            heading = _HEADING.match(line)
            if heading:
                flush()
                level = len(heading.group(1))
                label = heading.group(2).strip()
                if level == 1 and title == path.stem:
                    title = label
                section_stack[level - 1 :] = [label]
                continue
            if not line.strip():
                flush()
            else:
                buffer.append(line)
        flush()

        return self._document(path, source_uri, title, segments)


class PdfLoader(DocumentLoader):
    suffixes = frozenset({".pdf"})
    parser_name = "builtin-pdf-layout"
    parser_version = "2"

    def load(self, path: Path, source_uri: str) -> ParsedDocument:
        with pymupdf.open(path) as pdf:
            metadata = pdf.metadata or {}
            lines = [
                line
                for page_number, page in enumerate(pdf, start=1)
                for line in self._lines(page, page_number)
            ]
            repeated_margins = self._repeated_margin_lines(lines, len(pdf))
            content_lines = [
                line for line in lines if self._margin_key(line) not in repeated_margins
            ]
            body_size = self._body_font_size(content_lines)
            heading_sizes = sorted(
                {
                    round(line.font_size, 1)
                    for line in content_lines
                    if self._is_heading(line, body_size)
                    and self._numbered_heading_level(line.text) is None
                },
                reverse=True,
            )
            segments = self._segments(content_lines, body_size, heading_sizes)
            title = str(metadata.get("title") or path.stem)
            safe_metadata = {
                str(key): str(value) for key, value in metadata.items() if value not in (None, "")
            }
            safe_metadata.update(
                {
                    "page_count": len(pdf),
                    "backend": "pymupdf",
                    "backend_version": version("PyMuPDF"),
                    "repeated_margin_lines_removed": len(repeated_margins),
                }
            )
        return self._document(path, source_uri, title, segments, safe_metadata)

    @staticmethod
    def _lines(page: pymupdf.Page, page_number: int) -> list[_PdfLine]:
        result: list[_PdfLine] = []
        page_height = float(page.rect.height)
        data = page.get_text("dict", sort=True)
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for raw_line in block.get("lines", []):
                spans = [span for span in raw_line.get("spans", []) if span.get("text", "").strip()]
                if not spans:
                    continue
                text = "".join(str(span.get("text", "")) for span in spans).strip()
                if not text:
                    continue
                bbox = raw_line.get("bbox") or block.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                significant = max(spans, key=lambda span: len(str(span.get("text", "")).strip()))
                font_size = max(float(span.get("size", 0.0)) for span in spans)
                bold = any(
                    int(span.get("flags", 0)) & 16
                    or "bold" in str(span.get("font", "")).casefold()
                    for span in spans
                )
                if font_size <= 0:
                    font_size = float(significant.get("size", 0.0))
                result.append(
                    _PdfLine(
                        page=page_number,
                        page_height=page_height,
                        text=text,
                        x0=float(bbox[0]),
                        y0=float(bbox[1]),
                        x1=float(bbox[2]),
                        y1=float(bbox[3]),
                        font_size=font_size,
                        bold=bold,
                    )
                )
        return result

    @classmethod
    def _repeated_margin_lines(cls, lines: list[_PdfLine], page_count: int) -> set[str]:
        if page_count < 2:
            return set()
        pages_by_key: dict[str, set[int]] = {}
        for line in lines:
            key = cls._margin_key(line)
            if key:
                pages_by_key.setdefault(key, set()).add(line.page)
        threshold = max(2, ceil(page_count * 0.6))
        return {key for key, pages in pages_by_key.items() if len(pages) >= threshold}

    @staticmethod
    def _margin_key(line: _PdfLine) -> str:
        if 0.08 < line.position < 0.92:
            return ""
        normalized = _PDF_SPACE.sub(" ", line.text).strip().casefold()
        return _PDF_DIGITS.sub("#", normalized)

    @staticmethod
    def _body_font_size(lines: list[_PdfLine]) -> float:
        weighted_sizes: Counter[float] = Counter()
        for line in lines:
            if line.position <= 0.08 or line.position >= 0.92:
                continue
            weighted_sizes[round(line.font_size, 1)] += max(1, len(line.text))
        if not weighted_sizes:
            for line in lines:
                weighted_sizes[round(line.font_size, 1)] += max(1, len(line.text))
        return weighted_sizes.most_common(1)[0][0] if weighted_sizes else 0.0

    @classmethod
    def _is_heading(cls, line: _PdfLine, body_size: float) -> bool:
        text = line.text.strip()
        if not text or len(text) > 120 or text.count(" ") > 18:
            return False
        numbered = cls._numbered_heading_level(text) is not None
        score = 3 if numbered else 0
        if body_size and line.font_size >= body_size * 1.3:
            score += 3
        elif body_size and line.font_size >= body_size * 1.12:
            score += 2
        if line.bold:
            score += 1
        if len(text) <= 80:
            score += 1
        if _PDF_TRAILING_SENTENCE_MARK.search(text):
            score -= 2
        return score >= 3

    @staticmethod
    def _numbered_heading_level(text: str) -> int | None:
        decimal = _PDF_DECIMAL_HEADING.match(text)
        if decimal:
            return min(6, decimal.group(1).count(".") + 1)
        chapter = _PDF_CHAPTER_HEADING.match(text)
        if chapter:
            return 2 if chapter.group(1) == "节" else 1
        if _PDF_CHINESE_HEADING.match(text):
            return 2
        return None

    @classmethod
    def _heading_level(cls, line: _PdfLine, heading_sizes: list[float]) -> int:
        numbered = cls._numbered_heading_level(line.text)
        if numbered is not None:
            return numbered
        size = round(line.font_size, 1)
        try:
            return min(6, heading_sizes.index(size) + 1)
        except ValueError:
            return 1

    @classmethod
    def _segments(
        cls,
        lines: list[_PdfLine],
        body_size: float,
        heading_sizes: list[float],
    ) -> list[TextSegment]:
        segments: list[TextSegment] = []
        section_stack: list[str] = []
        buffer: list[str] = []
        buffer_page: int | None = None
        previous: _PdfLine | None = None

        def flush() -> None:
            nonlocal buffer_page
            value = "\n".join(buffer).strip()
            if value and buffer_page is not None:
                segments.append(
                    TextSegment(
                        text=value,
                        section_path=tuple(section_stack),
                        page_start=buffer_page,
                        page_end=buffer_page,
                    )
                )
            buffer.clear()
            buffer_page = None

        for line in lines:
            if buffer_page is not None and line.page != buffer_page:
                flush()
                previous = None
            if cls._is_heading(line, body_size):
                flush()
                level = cls._heading_level(line, heading_sizes)
                section_stack[level - 1 :] = [line.text.strip()]
                previous = None
                continue
            if buffer_page is None:
                buffer_page = line.page
            if previous is not None:
                gap = line.y0 - previous.y1
                if gap > max(previous.font_size, line.font_size) * 0.8:
                    buffer.append("")
            buffer.append(line.text)
            previous = line
        flush()
        return segments


class LoaderRegistry:
    def __init__(self, loaders: list[DocumentLoader] | None = None) -> None:
        configured = loaders or [PdfLoader(), MarkdownLoader(), TextLoader()]
        self._by_suffix = {suffix: loader for loader in configured for suffix in loader.suffixes}

    @property
    def supported_suffixes(self) -> frozenset[str]:
        return frozenset(self._by_suffix)

    def for_path(self, path: Path) -> DocumentLoader:
        try:
            return self._by_suffix[path.suffix.lower()]
        except KeyError as error:
            suffix = path.suffix or "<none>"
            raise UnsupportedFileError(f"unsupported file type: {suffix}") from error

    def load(self, path: Path, source_uri: str) -> ParsedDocument:
        return self.for_path(path).load(path, source_uri)


def _paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [value.strip() for value in re.split(r"\n\s*\n", normalized) if value.strip()]
