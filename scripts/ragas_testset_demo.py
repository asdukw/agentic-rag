"""Generate a small, corpus-bound Ragas test set from PDFs in ``data/raw``.

This deliberately small local demo uses OpenAI-compatible chat and embedding
endpoints. Its output is an inspectable JSON envelope under ``data/processed``;
the caller supplies the exact corpus-content hash of the index to evaluate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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
TESTSET_SCHEMA_VERSION = "1"
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
    parser.add_argument(
        "--corpus-content-hash",
        metavar="SHA256",
        help=(
            "64-character corpus-content hash from the exact index profile to evaluate; "
            "required unless --dry-run is used"
        ),
    )
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


def corpus_content_hash(value: str | None) -> str:
    """Validate the profile-bound corpus hash supplied by the caller.

    The hash is intentionally not recomputed from the selected PDFs: the
    evaluation contract hashes the imported document and chunk identities, which
    depend on the project's ingestion configuration as well as source text.
    """

    normalized = (value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(
            "--corpus-content-hash must be the 64-character lowercase SHA-256 "
            "from the exact index profile"
        )
    return normalized


def cases_from_dataframe(dataframe: Any) -> list[dict[str, object]]:
    """Keep only the Ragas fields required by the project evaluation contract."""

    rows = json.loads(dataframe.to_json(orient="records", force_ascii=False))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Ragas did not generate any test cases")

    cases: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"generated Ragas case {index} is not an object")
        user_input = row.get("user_input")
        reference = row.get("reference")
        reference_contexts = row.get("reference_contexts")
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError(f"generated Ragas case {index} has no user_input")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"generated Ragas case {index} has no reference")
        if not isinstance(reference_contexts, list) or not all(
            isinstance(item, str) for item in reference_contexts
        ):
            raise ValueError(f"generated Ragas case {index} has invalid reference_contexts")
        cases.append(
            {
                "user_input": user_input,
                "reference": reference,
                "reference_contexts": reference_contexts,
            }
        )
    return cases


def main() -> None:
    args = parse_args()
    if args.testset_size < 1 or args.max_documents < 1 or args.pages_per_document < 1:
        raise ValueError("all size arguments must be positive")

    documents = load_documents(args.source_dir, args.max_documents, args.pages_per_document)
    print(f"Loaded {len(documents)} page documents from {args.source_dir}")
    if args.dry_run:
        return

    source_corpus_hash = corpus_content_hash(args.corpus_content_hash)

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
    cases = cases_from_dataframe(testset.to_pandas())
    payload = {
        "schema_version": TESTSET_SCHEMA_VERSION,
        "corpus_content_hash": source_corpus_hash,
        "cases": cases,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(cases)} cases: {args.output}")


if __name__ == "__main__":
    main()
