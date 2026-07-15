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
_PDF_SECTION_NUMBER_ONLY = re.compile(r"^\d+(?:\.\d+){1,5}[.)．]?$")  # noqa: RUF001
_PDF_PURE_NUMBER = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)?|\.\d+)$")
_PDF_SCIENTIFIC_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+$")
_PDF_ARXIV = re.compile(r"(?:arxiv\s*:\s*)?\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
_PDF_DATE = re.compile(
    r"(?:\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|"
    r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|"
    r"\d{4}年\d{1,2}月\d{1,2}日|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b)",
    re.IGNORECASE,
)
_PDF_COPYRIGHT = re.compile(
    r"(?:©|copyright|all rights reserved|版权所有|保留所有权利)", re.IGNORECASE
)
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
    in_table: bool

    @property
    def position(self) -> float:
        return self.y0 / self.page_height if self.page_height else 0.5


@dataclass(frozen=True, slots=True)
class _PdfHeading:
    page: int
    y0: float
    y1: float
    level: int
    text: str
    source: str


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
    parser_version = "3"

    def load(self, path: Path, source_uri: str) -> ParsedDocument:
        with pymupdf.open(path) as pdf:
            metadata = pdf.metadata or {}
            lines: list[_PdfLine] = []
            table_region_count = 0
            for page_index in range(len(pdf)):
                page_number = page_index + 1
                page = pdf[page_index]
                table_rects = self._table_rects(page)
                table_region_count += len(table_rects)
                lines.extend(self._lines(page, page_number, table_rects))
            repeated_margins = self._repeated_margin_lines(lines, len(pdf))
            content_lines = [
                line for line in lines if self._margin_key(line) not in repeated_margins
            ]
            content_lines = self._merge_numbered_title_lines(content_lines)

            headings = self._outline_headings(pdf, content_lines)
            section_source = "outline" if headings else ""
            if not headings:
                headings = self._tagged_headings(pdf)
                section_source = "tagged" if headings else ""
            if not headings:
                headings = self._visual_headings(content_lines)
                section_source = "visual" if headings else "none"

            segments = self._segments(content_lines, headings)
            title = str(metadata.get("title") or path.stem)
            safe_metadata: dict[str, object] = {
                str(key): str(value) for key, value in metadata.items() if value not in (None, "")
            }
            safe_metadata.update(
                {
                    "page_count": len(pdf),
                    "backend": "pymupdf",
                    "backend_version": version("PyMuPDF"),
                    "repeated_margin_lines_removed": len(repeated_margins),
                    "table_regions_detected": table_region_count,
                    "section_source": section_source,
                    "section_heading_count": len(headings),
                }
            )
        return self._document(path, source_uri, title, segments, safe_metadata)

    @staticmethod
    def _table_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
        try:
            pymupdf.no_recommend_layout()
            finder = page.find_tables()
            if finder is None:
                return []
            return [pymupdf.Rect(table.bbox) for table in finder.tables]
        except (RuntimeError, ValueError):
            return []

    @staticmethod
    def _lines(
        page: pymupdf.Page,
        page_number: int,
        table_rects: list[pymupdf.Rect],
    ) -> list[_PdfLine]:
        result: list[_PdfLine] = []
        page_height = float(page.rect.height)
        data = page.get_text("dict", sort=True)
        if not isinstance(data, dict):
            return result
        blocks = data.get("blocks", [])
        if not isinstance(blocks, list):
            return result
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") != 0:
                continue
            raw_lines = block.get("lines", [])
            if not isinstance(raw_lines, list):
                continue
            for raw_line in raw_lines:
                if not isinstance(raw_line, dict):
                    continue
                raw_spans = raw_line.get("spans", [])
                if not isinstance(raw_spans, list):
                    continue
                spans = [
                    span
                    for span in raw_spans
                    if isinstance(span, dict) and str(span.get("text", "")).strip()
                ]
                if not spans:
                    continue
                text = "".join(str(span.get("text", "")) for span in spans).strip()
                text = re.sub(r"^(\d+(?:\.\d+){1,5})(?=[^\d\s.])", r"\1 ", text)
                if not text:
                    continue
                bbox = raw_line.get("bbox") or block.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                significant = max(spans, key=lambda span: len(str(span.get("text", "")).strip()))
                font_size = max(float(span.get("size", 0.0)) for span in spans)
                bold = any(
                    int(span.get("flags", 0)) & 16 or "bold" in str(span.get("font", "")).casefold()
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
                        in_table=PdfLoader._inside_table(bbox, table_rects),
                    )
                )
        return result

    @staticmethod
    def _inside_table(bbox: object, table_rects: list[pymupdf.Rect]) -> bool:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        center = pymupdf.Point(
            (float(bbox[0]) + float(bbox[2])) / 2,
            (float(bbox[1]) + float(bbox[3])) / 2,
        )
        return any(rect.contains(center) for rect in table_rects)

    @classmethod
    def _merge_numbered_title_lines(cls, lines: list[_PdfLine]) -> list[_PdfLine]:
        merged: list[_PdfLine] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if index + 1 >= len(lines) or not _PDF_SECTION_NUMBER_ONLY.fullmatch(line.text):
                merged.append(line)
                index += 1
                continue
            following = lines[index + 1]
            font_size = max(line.font_size, following.font_size, 1.0)
            same_row = (
                following.page == line.page
                and abs(following.y0 - line.y0) <= font_size * 0.6
                and -font_size <= following.x0 - line.x1 <= font_size * 4
            )
            next_row = (
                following.page == line.page
                and -font_size * 0.3 <= following.y0 - line.y1 <= font_size * 1.2
                and abs(following.x0 - line.x0) <= font_size * 4
            )
            usable_title = (
                cls._alphabetic_count(following.text) >= 3
                and not following.in_table
                and not cls._is_visual_noise(following.text)
            )
            if not usable_title or not (same_row or next_row):
                merged.append(line)
                index += 1
                continue
            merged.append(
                _PdfLine(
                    page=line.page,
                    page_height=line.page_height,
                    text=f"{line.text.rstrip('.．)')} {following.text.strip()}",  # noqa: RUF001
                    x0=min(line.x0, following.x0),
                    y0=min(line.y0, following.y0),
                    x1=max(line.x1, following.x1),
                    y1=max(line.y1, following.y1),
                    font_size=max(line.font_size, following.font_size),
                    bold=line.bold or following.bold,
                    in_table=line.in_table or following.in_table,
                )
            )
            index += 2
        return merged

    @classmethod
    def _outline_headings(
        cls,
        pdf: pymupdf.Document,
        lines: list[_PdfLine],
    ) -> list[_PdfHeading]:
        headings: list[_PdfHeading] = []
        lines_by_page: dict[int, list[_PdfLine]] = {}
        for line in lines:
            lines_by_page.setdefault(line.page, []).append(line)
        for item in pdf.get_toc(simple=False):
            if len(item) < 3:
                continue
            level, raw_title, page_number = item[:3]
            title = _PDF_SPACE.sub(" ", str(raw_title)).strip()
            if (
                not title
                or not isinstance(page_number, int)
                or page_number < 1
                or page_number > len(pdf)
            ):
                continue
            details = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
            destination = details.get("to")
            destination_y = float(getattr(destination, "y", 0.0))
            matched = cls._matching_title_line(
                title,
                lines_by_page.get(page_number, []),
                destination_y,
            )
            headings.append(
                _PdfHeading(
                    page=page_number,
                    y0=matched.y0 if matched else max(0.0, destination_y),
                    y1=matched.y1 if matched else max(0.0, destination_y),
                    level=min(6, max(1, int(level))),
                    text=title,
                    source="outline",
                )
            )
        return headings

    @classmethod
    def _matching_title_line(
        cls,
        title: str,
        lines: list[_PdfLine],
        destination_y: float,
    ) -> _PdfLine | None:
        normalized_title = cls._normalized_heading_text(title)
        candidates = []
        for line in lines:
            normalized_line = cls._normalized_heading_text(line.text)
            shorter = min(len(normalized_title), len(normalized_line))
            longer = max(len(normalized_title), len(normalized_line), 1)
            if normalized_title == normalized_line or (
                shorter >= 4
                and shorter / longer >= 0.75
                and (normalized_title in normalized_line or normalized_line in normalized_title)
            ):
                candidates.append(line)
        if not candidates:
            return None
        return min(candidates, key=lambda line: abs(line.y0 - destination_y))

    @classmethod
    def _tagged_headings(cls, pdf: pymupdf.Document) -> list[_PdfHeading]:
        headings: list[_PdfHeading] = []
        flags = pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_COLLECT_STRUCTURE
        for page_index in range(len(pdf)):
            page_number = page_index + 1
            page = pdf[page_index]
            data = page.get_text("dict", flags=flags, sort=True)
            if not isinstance(data, dict):
                continue
            blocks = data.get("blocks", [])
            if isinstance(blocks, list):
                cls._collect_tagged_headings(blocks, page_number, headings)
        return headings

    @classmethod
    def _collect_tagged_headings(
        cls,
        blocks: list[object],
        page_number: int,
        headings: list[_PdfHeading],
    ) -> None:
        for block in blocks:
            if not isinstance(block, dict):
                continue
            nested = block.get("blocks", [])
            nested_blocks = nested if isinstance(nested, list) else []
            tag = str(block.get("std") or block.get("raw") or "").upper()
            if tag in {"H1", "H2", "H3"}:
                text = cls._structured_text(nested_blocks)
                bbox = block.get("bbox")
                if text and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    headings.append(
                        _PdfHeading(
                            page=page_number,
                            y0=float(bbox[1]),
                            y1=float(bbox[3]),
                            level=int(tag[1]),
                            text=text,
                            source="tagged",
                        )
                    )
                continue
            cls._collect_tagged_headings(nested_blocks, page_number, headings)

    @classmethod
    def _structured_text(cls, blocks: list[object]) -> str:
        parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            lines = block.get("lines", [])
            if isinstance(lines, list):
                for line in lines:
                    if not isinstance(line, dict):
                        continue
                    spans = line.get("spans", [])
                    if not isinstance(spans, list):
                        continue
                    value = "".join(
                        str(span.get("text", "")) for span in spans if isinstance(span, dict)
                    ).strip()
                    if value:
                        parts.append(value)
            nested = block.get("blocks", [])
            if isinstance(nested, list):
                nested_text = cls._structured_text(nested)
                if nested_text:
                    parts.append(nested_text)
        return _PDF_SPACE.sub(" ", " ".join(parts)).strip()

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
            if line.position <= 0.08 or line.position >= 0.92 or line.in_table:
                continue
            weighted_sizes[round(line.font_size, 1)] += max(1, len(line.text))
        if not weighted_sizes:
            for line in lines:
                weighted_sizes[round(line.font_size, 1)] += max(1, len(line.text))
        return weighted_sizes.most_common(1)[0][0] if weighted_sizes else 0.0

    @classmethod
    def _visual_heading_score(cls, line: _PdfLine, body_size: float) -> int:
        text = line.text.strip()
        if (
            not text
            or len(text) > 120
            or text.count(" ") > 18
            or line.in_table
            or cls._alphabetic_count(text) < 3
            or cls._is_visual_noise(text)
        ):
            return 0
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
        return score

    @classmethod
    def _visual_headings(cls, lines: list[_PdfLine]) -> list[_PdfHeading]:
        body_size = cls._body_font_size(lines)
        candidates = [line for line in lines if cls._visual_heading_score(line, body_size) >= 5]
        heading_sizes = sorted(
            {
                round(line.font_size, 1)
                for line in candidates
                if cls._numbered_heading_level(line.text) is None
            },
            reverse=True,
        )
        return [
            _PdfHeading(
                page=line.page,
                y0=line.y0,
                y1=line.y1,
                level=cls._heading_level(line, heading_sizes),
                text=line.text.strip(),
                source="visual",
            )
            for line in candidates
        ]

    @staticmethod
    def _alphabetic_count(text: str) -> int:
        return sum(character.isalpha() for character in text)

    @staticmethod
    def _is_visual_noise(text: str) -> bool:
        value = _PDF_SPACE.sub(" ", text).strip()
        return bool(
            _PDF_PURE_NUMBER.fullmatch(value)
            or _PDF_SCIENTIFIC_NUMBER.fullmatch(value)
            or _PDF_ARXIV.search(value)
            or _PDF_DATE.search(value)
            or _PDF_COPYRIGHT.search(value)
        )

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
        headings: list[_PdfHeading],
    ) -> list[TextSegment]:
        segments: list[TextSegment] = []
        section_stack: list[str] = []
        buffer: list[str] = []
        buffer_page: int | None = None
        previous: _PdfLine | None = None
        ordered_headings = sorted(headings, key=lambda heading: (heading.page, heading.y0))
        heading_index = 0

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

        def apply_heading(heading: _PdfHeading) -> None:
            level = min(6, max(1, heading.level))
            section_stack[level - 1 :] = [heading.text.strip()]

        for line in lines:
            if buffer_page is not None and line.page != buffer_page:
                flush()
                previous = None

            line_is_heading = False
            while heading_index < len(ordered_headings):
                heading = ordered_headings[heading_index]
                before_line = heading.page < line.page or (
                    heading.page == line.page
                    and heading.y0 <= line.y0 + max(1.0, line.font_size * 0.25)
                )
                if not before_line:
                    break
                flush()
                apply_heading(heading)
                if cls._line_matches_heading(line, heading):
                    line_is_heading = True
                heading_index += 1
                previous = None
            if line_is_heading:
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

    @classmethod
    def _line_matches_heading(cls, line: _PdfLine, heading: _PdfHeading) -> bool:
        if line.page != heading.page:
            return False
        vertically_close = not (
            line.y1 < heading.y0 - line.font_size * 0.25
            or line.y0 > heading.y1 + line.font_size * 0.25
        )
        if not vertically_close:
            return False
        line_text = cls._normalized_heading_text(line.text)
        heading_text = cls._normalized_heading_text(heading.text)
        if line_text == heading_text:
            return True
        shorter = min(len(line_text), len(heading_text))
        longer = max(len(line_text), len(heading_text), 1)
        return (
            shorter >= 4
            and shorter / longer >= 0.75
            and (line_text in heading_text or heading_text in line_text)
        )

    @staticmethod
    def _normalized_heading_text(text: str) -> str:
        return _PDF_SPACE.sub("", text).casefold()


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
