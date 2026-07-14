"""Generate a small Ragas test set from PDFs in ``data/raw``.

This is a deliberately small, local demo. It uses OpenAI-compatible chat and
embedding endpoints and writes an inspectable JSON file under ``data/processed``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from pypdf import PdfReader
from ragas.testset import TestsetGenerator

from hybrid_rag.retrieval.embedding import BGEM3EmbeddingProvider

if TYPE_CHECKING:
    from ragas.testset import Testset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "processed" / "ragas-testset-demo.json"
load_dotenv(ROOT / ".env")


class LocalBGEEmbeddings(Embeddings):
    """Expose the project's local BGE-M3 embedding provider to LangChain."""

    def __init__(self) -> None:
        self._provider = BGEM3EmbeddingProvider()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self._provider.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._provider.embed([text])[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--testset-size", type=int, default=5)
    parser.add_argument(
        "--max-documents",
        type=int,
        default=2,
        help="Maximum PDFs to read; keep this small because generation calls an LLM.",
    )
    parser.add_argument("--pages-per-document", type=int, default=6)
    parser.add_argument(
        "--llm-model",
        default=os.getenv("DEEPSEEK_EXTRACTION_MODEL", "deepseek-v4-flash"),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        help="DeepSeek's OpenAI-compatible endpoint URL.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load PDFs without calling an API.")
    return parser.parse_args()


def load_documents(source_dir: Path, max_documents: int, pages_per_document: int) -> list[Document]:
    pdf_paths = sorted(source_dir.glob("*.pdf"))[:max_documents]
    if not pdf_paths:
        raise FileNotFoundError(f"no PDF files found in {source_dir}")

    documents: list[Document] = []
    for path in pdf_paths:
        reader = PdfReader(path)
        for page_number, page in enumerate(reader.pages[:pages_per_document], start=1):
            text = page.extract_text().strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path.relative_to(ROOT)), "page": page_number},
                    )
                )
    if not documents:
        raise ValueError("the selected PDFs did not contain extractable text")
    return documents


def main() -> None:
    args = parse_args()
    if args.testset_size < 1 or args.max_documents < 1 or args.pages_per_document < 1:
        raise ValueError("all size arguments must be positive")

    documents = load_documents(args.source_dir, args.max_documents, args.pages_per_document)
    print(f"Loaded {len(documents)} page documents from {args.source_dir}")
    if args.dry_run:
        return

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("set DEEPSEEK_API_KEY in .env before generating a Ragas test set")

    secret_api_key = SecretStr(api_key)
    llm = ChatOpenAI(
        model=args.llm_model,
        api_key=secret_api_key,
        base_url=args.base_url,
        temperature=0,
    )
    embeddings = LocalBGEEmbeddings()
    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)
    testset = cast(
        "Testset",
        generator.generate_with_langchain_docs(
            documents,
            testset_size=args.testset_size,
            return_executor=False,
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe = testset.to_pandas()
    dataframe.to_json(args.output, orient="records", force_ascii=False, indent=2)
    print(f"Generated {len(dataframe)} cases: {args.output}")


if __name__ == "__main__":
    main()
