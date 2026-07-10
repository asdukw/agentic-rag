from __future__ import annotations

import re
from abc import ABC, abstractmethod
from importlib.metadata import version
from pathlib import Path

from pypdf import PdfReader

from hybrid_rag.ids import sha256_file, stable_id
from hybrid_rag.schemas import ParsedDocument, TextSegment

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


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
    parser_name = "pypdf"
    parser_version = version("pypdf")

    def load(self, path: Path, source_uri: str) -> ParsedDocument:
        reader = PdfReader(path)
        metadata = reader.metadata or {}
        segments: list[TextSegment] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                segments.append(
                    TextSegment(text=text, page_start=page_number, page_end=page_number)
                )
        title = str(metadata.get("/Title") or path.stem)
        safe_metadata = {
            str(key).lstrip("/"): str(value)
            for key, value in metadata.items()
            if value is not None
        }
        safe_metadata["page_count"] = len(reader.pages)
        return self._document(path, source_uri, title, segments, safe_metadata)


class LoaderRegistry:
    def __init__(self, loaders: list[DocumentLoader] | None = None) -> None:
        configured = loaders or [PdfLoader(), MarkdownLoader(), TextLoader()]
        self._by_suffix = {
            suffix: loader for loader in configured for suffix in loader.suffixes
        }

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
