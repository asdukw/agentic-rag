from __future__ import annotations

import re

from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.cleaner import clean_document
from hybrid_rag.schemas import ParsedDocument, TextSegment


class WordCounter:
    name = "test-words"

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


def _document() -> ParsedDocument:
    return clean_document(
        ParsedDocument(
            id="doc_test",
            title="Test",
            source_type="md",
            source_uri="file:test.md",
            local_path="test.md",
            content_hash="hash",
            parser_name="test",
            parser_version="1",
            segments=[
                TextSegment(
                    text="one two three four five six seven eight nine ten eleven twelve",
                    section_path=("First",),
                    page_start=1,
                    page_end=1,
                ),
                TextSegment(
                    text="alpha beta gamma delta epsilon zeta",
                    section_path=("Second",),
                    page_start=2,
                    page_end=2,
                ),
            ],
        )
    )


def test_chunker_is_bounded_deterministic_and_section_aware() -> None:
    chunker = SectionTokenChunker(WordCounter(), max_tokens=5, overlap_tokens=1)
    document = _document()

    first = chunker.split(document)
    second = chunker.split(document)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert all(0 < chunk.token_count <= 5 for chunk in first)
    assert all(document.text[chunk.char_start : chunk.char_end] == chunk.text for chunk in first)
    assert {chunk.section_path for chunk in first} == {("First",), ("Second",)}
    assert not any(chunk.page_start == 1 and chunk.page_end == 2 for chunk in first)


def test_changing_chunk_config_changes_ids() -> None:
    document = _document()
    small = SectionTokenChunker(WordCounter(), max_tokens=5, overlap_tokens=1).split(document)
    large = SectionTokenChunker(WordCounter(), max_tokens=8, overlap_tokens=2).split(document)

    assert [chunk.id for chunk in small] != [chunk.id for chunk in large]
