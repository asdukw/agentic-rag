from pathlib import Path

from reportlab.pdfgen.canvas import Canvas

from hybrid_rag.ingest.cleaner import clean_document
from hybrid_rag.ingest.loaders import LoaderRegistry, MarkdownLoader, PdfLoader


def test_markdown_loader_retains_heading_path(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("# Paper title\n\n## Method\n\nGraph evidence.", encoding="utf-8")

    document = clean_document(MarkdownLoader().load(path, "file:paper.md"))

    assert document.title == "Paper title"
    assert document.segments[0].section_path == ("Paper title", "Method")
    assert document.segments[0].text == "Graph evidence."


def test_pdf_loader_retains_page_numbers_and_title(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    canvas = Canvas(str(path))
    canvas.setTitle("Fixture Paper")
    canvas.drawString(72, 760, "Page one evidence about graph retrieval.")
    canvas.showPage()
    canvas.drawString(72, 760, "Page two evidence about vector retrieval.")
    canvas.save()

    document = clean_document(PdfLoader().load(path, "file:paper.pdf"))

    assert document.title == "Fixture Paper"
    assert [segment.page_start for segment in document.segments] == [1, 2]
    assert document.metadata["page_count"] == 2
    assert "graph retrieval" in document.text


def test_loader_registry_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "paper.csv"
    path.write_text("not,supported", encoding="utf-8")

    try:
        LoaderRegistry().for_path(path)
    except ValueError as error:
        assert "unsupported file type" in str(error)
    else:
        raise AssertionError("unknown extension should be rejected")
