from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from hybrid_rag.ids import canonical_json_hash, sha256_text, stable_id
from hybrid_rag.ingest.quality import (
    CHUNK_QUALITY_CLASSIFIER_NAME,
    CHUNK_QUALITY_CLASSIFIER_VERSION,
    classify_chunk_quality,
)
from hybrid_rag.ingest.tokenizer import TokenCounter
from hybrid_rag.schemas import ChunkData, ParsedDocument, TextSegment

CHUNKER_NAME = "section-token-chunker"
CHUNKER_VERSION = "3"


@dataclass(frozen=True, slots=True)
class _Piece:
    start: int
    end: int
    section_path: tuple[str, ...]


class SectionTokenChunker:
    """Keep section boundaries when possible and enforce a token ceiling."""

    def __init__(
        self,
        tokenizer: TokenCounter,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @property
    def config(self) -> dict[str, str | int]:
        return {
            "name": CHUNKER_NAME,
            "version": CHUNKER_VERSION,
            "tokenizer": self.tokenizer.name,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "quality_classifier": CHUNK_QUALITY_CLASSIFIER_NAME,
            "quality_classifier_version": CHUNK_QUALITY_CLASSIFIER_VERSION,
        }

    @property
    def config_hash(self) -> str:
        return canonical_json_hash(self.config)

    def split(self, document: ParsedDocument) -> list[ChunkData]:
        if not document.text:
            return []

        pieces = list(self._pieces(document))
        spans: list[tuple[int, int, tuple[str, ...]]] = []
        current_start: int | None = None
        current_end: int | None = None
        current_section: tuple[str, ...] = ()

        def flush() -> None:
            nonlocal current_start, current_end
            if current_start is not None and current_end is not None:
                spans.append((current_start, current_end, current_section))
            current_start = None
            current_end = None

        for piece in pieces:
            if current_start is None or current_end is None:
                current_start, current_end = piece.start, piece.end
                current_section = piece.section_path
                continue

            if piece.section_path != current_section:
                flush()
                current_start, current_end = piece.start, piece.end
                current_section = piece.section_path
                continue

            candidate = document.text[current_start : piece.end]
            if self._count_contextualized(candidate, current_section) <= self.max_tokens:
                current_end = piece.end
                continue

            previous_start = current_start
            previous_end = current_end
            flush()

            overlap_start = self._suffix_start(
                document.text, previous_start, previous_end, self.overlap_tokens
            )
            with_overlap = document.text[overlap_start : piece.end]
            current_start = (
                overlap_start
                if self.overlap_tokens
                and self._count_contextualized(with_overlap, piece.section_path) <= self.max_tokens
                else piece.start
            )
            current_end = piece.end
            current_section = piece.section_path

        flush()

        chunks: list[ChunkData] = []
        for ordinal, (start, end, section_path) in enumerate(spans):
            start, end = self._trim_span(document.text, start, end)
            text = document.text[start:end]
            if not text:
                continue
            pages = self._pages_for_span(document.segments, start, end)
            content_hash = sha256_text(text)
            chunk_id = stable_id(
                "chk",
                document.id,
                str(ordinal),
                content_hash,
                self.config_hash,
            )
            section_label = " > ".join(section_path)
            contextualized = f"{section_label}\n{text}" if section_label else text
            token_count = self.tokenizer.count(contextualized)
            if token_count > self.max_tokens:
                raise AssertionError("chunker produced a chunk over the configured token limit")
            chunks.append(
                ChunkData(
                    id=chunk_id,
                    document_id=document.id,
                    ordinal=ordinal,
                    section_path=section_path,
                    page_start=pages[0],
                    page_end=pages[1],
                    char_start=start,
                    char_end=end,
                    text=text,
                    contextualized_text=contextualized,
                    token_count=token_count,
                    content_hash=content_hash,
                    chunker_name=CHUNKER_NAME,
                    chunker_version=CHUNKER_VERSION,
                    quality_class=classify_chunk_quality(
                        section_path=section_path,
                        text=text,
                        ordinal=ordinal,
                        page_start=pages[0],
                    ),
                    metadata={"tokenizer": self.tokenizer.name},
                )
            )
        return chunks

    def _pieces(self, document: ParsedDocument) -> Iterable[_Piece]:
        for segment in document.segments:
            prefix_tokens = self.tokenizer.count(self._section_prefix(segment.section_path))
            if prefix_tokens >= self.max_tokens:
                raise ValueError("section heading consumes the entire chunk token budget")
            start = segment.char_start
            while start < segment.char_end:
                remainder = document.text[start : segment.char_end]
                if self._count_contextualized(remainder, segment.section_path) <= self.max_tokens:
                    end = segment.char_end
                else:
                    end = self._bounded_end(
                        document.text,
                        start,
                        segment.char_end,
                        segment.section_path,
                    )
                trimmed_start, trimmed_end = self._trim_span(document.text, start, end)
                if trimmed_start < trimmed_end:
                    yield _Piece(trimmed_start, trimmed_end, segment.section_path)
                if end <= start:
                    raise AssertionError("chunker did not advance")
                start = end

    def _bounded_end(
        self,
        text: str,
        start: int,
        upper: int,
        section_path: tuple[str, ...],
    ) -> int:
        low = start + 1
        high = upper
        best = low
        while low <= high:
            middle = (low + high) // 2
            if self._count_contextualized(text[start:middle], section_path) <= self.max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1

        preferred_floor = start + max(1, (best - start) // 2)
        for marker in ("\n\n", ". ", "; ", ", ", " "):
            boundary = text.rfind(marker, preferred_floor, best)
            if boundary != -1:
                candidate = boundary + len(marker)
                if candidate > start:
                    return candidate
        return best

    def _count_contextualized(self, text: str, section_path: tuple[str, ...]) -> int:
        return self.tokenizer.count(f"{self._section_prefix(section_path)}{text}")

    @staticmethod
    def _section_prefix(section_path: tuple[str, ...]) -> str:
        return f"{' > '.join(section_path)}\n" if section_path else ""

    def _suffix_start(self, text: str, start: int, end: int, budget: int) -> int:
        if budget <= 0:
            return end
        low = start
        high = end
        best = end
        while low <= high:
            middle = (low + high) // 2
            if self.tokenizer.count(text[middle:end]) <= budget:
                best = middle
                high = middle - 1
            else:
                low = middle + 1
        boundary = text.find(" ", best, end)
        return boundary + 1 if boundary != -1 else best

    @staticmethod
    def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    @staticmethod
    def _pages_for_span(
        segments: list[TextSegment], start: int, end: int
    ) -> tuple[int | None, int | None]:
        page_starts: list[int] = []
        page_ends: list[int] = []
        for segment in segments:
            if segment.char_end <= start or segment.char_start >= end:
                continue
            if segment.page_start is not None:
                page_starts.append(segment.page_start)
            if segment.page_end is not None:
                page_ends.append(segment.page_end)
        return (
            min(page_starts) if page_starts else None,
            max(page_ends) if page_ends else None,
        )
