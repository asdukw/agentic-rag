from hybrid_rag.ingest.cleaner import clean_document, clean_text
from hybrid_rag.schemas import ParsedDocument, TextSegment


def test_clean_text_repairs_line_end_hyphen_and_whitespace() -> None:
    raw = "Retrieval-\naugmented   generation\r\nworks.\n\n\nNext paragraph."

    assert clean_text(raw) == "Retrieval-augmented generation works.\n\nNext paragraph."


def test_clean_document_assigns_traceable_offsets() -> None:
    document = ParsedDocument(
        id="doc_test",
        title="Test",
        source_type="txt",
        source_uri="file:test.txt",
        local_path="test.txt",
        content_hash="hash",
        parser_name="test",
        parser_version="1",
        segments=[
            TextSegment(text=" first ", section_path=("One",)),
            TextSegment(text="second", section_path=("Two",)),
        ],
    )

    cleaned = clean_document(document)

    assert cleaned.text == "first\n\nsecond"
    assert [(segment.char_start, segment.char_end) for segment in cleaned.segments] == [
        (0, 5),
        (7, 13),
    ]
    assert all(
        cleaned.text[segment.char_start : segment.char_end] == segment.text
        for segment in cleaned.segments
    )
