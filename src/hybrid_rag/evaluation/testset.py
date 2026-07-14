"""Reusable helpers for generating corpus-bound Ragas test sets."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from hybrid_rag.evaluation.testset_contract import (
    RAGAS_TESTSET_SCHEMA_VERSION,
    validate_corpus_content_hash,
)
from hybrid_rag.ids import file_source_uri
from hybrid_rag.ingest.cleaner import clean_document
from hybrid_rag.ingest.loaders import LoaderRegistry
from hybrid_rag.retrieval.embedding import BGEM3EmbeddingProvider

if TYPE_CHECKING:
    from ragas.testset import Testset


class LocalBGEEmbeddings(Embeddings):
    """Expose the project's local BGE-M3 embedding provider to LangChain."""

    def __init__(self) -> None:
        self._provider = BGEM3EmbeddingProvider()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self._provider.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._provider.embed([text])[0])


def load_ragas_documents(
    source: Path,
    *,
    max_documents: int | None = None,
    max_segments_per_document: int | None = None,
    loaders: LoaderRegistry | None = None,
) -> list[Document]:
    """Load supported files into LangChain documents for Ragas generation.

    ``None`` for either limit means no limit. Segments correspond to PDF pages
    and to the loader-produced text or Markdown sections for other formats.
    """

    _validate_limit(max_documents, field="max_documents")
    _validate_limit(max_segments_per_document, field="max_segments_per_document")
    registry = loaders or LoaderRegistry()
    root, paths = _discover_source_files(source, registry)
    selected_paths = paths if max_documents is None else paths[:max_documents]

    documents: list[Document] = []
    for path in selected_paths:
        parsed = clean_document(registry.load(path, file_source_uri(path, root)))
        segments = (
            parsed.segments
            if max_segments_per_document is None
            else parsed.segments[:max_segments_per_document]
        )
        for segment in segments:
            metadata: dict[str, str | int] = {
                "source": parsed.source_uri,
                "document_id": parsed.id,
                "document_title": parsed.title,
                "source_type": parsed.source_type,
            }
            if segment.section_path:
                metadata["section_path"] = " > ".join(segment.section_path)
            if segment.page_start is not None:
                metadata["page_start"] = segment.page_start
            if segment.page_end is not None:
                metadata["page_end"] = segment.page_end
            documents.append(Document(page_content=segment.text, metadata=metadata))

    if not documents:
        raise ValueError(f"no extractable text found in supported files under {source}")
    return documents


def generate_ragas_cases(
    documents: Sequence[Document],
    *,
    api_key: str,
    llm_model: str,
    base_url: str,
    testset_size: int,
) -> list[dict[str, object]]:
    """Generate project-contract cases from already loaded source documents."""

    if not documents:
        raise ValueError("Ragas test-set generation requires at least one document")
    if testset_size < 1:
        raise ValueError("testset_size must be positive")
    if not api_key.strip():
        raise ValueError("Ragas test-set generation requires a non-empty API key")
    if not llm_model.strip():
        raise ValueError("Ragas test-set generation requires a non-empty LLM model")
    if not base_url.strip():
        raise ValueError("Ragas test-set generation requires a non-empty base URL")

    from ragas.testset import TestsetGenerator

    llm = ChatOpenAI(
        model=llm_model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0,
    )
    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embedding_model=LocalBGEEmbeddings(),
    )
    testset = cast(
        "Testset",
        generator.generate_with_langchain_docs(
            list(documents),
            testset_size=testset_size,
            return_executor=False,
        ),
    )
    return cases_from_dataframe(testset.to_pandas())


def build_ragas_testset_envelope(
    corpus_content_hash: object,
    cases: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Build and validate the versioned JSON envelope accepted by ``evaluate``."""

    normalized_hash = validate_corpus_content_hash(corpus_content_hash)
    if not cases:
        raise ValueError("Ragas test set requires at least one case")
    return {
        "schema_version": RAGAS_TESTSET_SCHEMA_VERSION,
        "corpus_content_hash": normalized_hash,
        "cases": [_validated_case(case, index=index) for index, case in enumerate(cases, start=1)],
    }


def write_ragas_testset(path: Path, envelope: dict[str, object]) -> Path:
    """Write a generated test-set envelope as deterministic UTF-8 JSON."""

    normalized_envelope = _validated_envelope(envelope)
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(normalized_envelope, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def cases_from_dataframe(dataframe: Any) -> list[dict[str, object]]:
    """Keep only the Ragas fields required by the project evaluation contract."""

    rows = json.loads(dataframe.to_json(orient="records", force_ascii=False))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Ragas did not generate any test cases")
    return [_validated_case(row, index=index) for index, row in enumerate(rows, start=1)]


def _discover_source_files(source: Path, registry: LoaderRegistry) -> tuple[Path, tuple[Path, ...]]:
    resolved = source.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if resolved.is_file():
        registry.for_path(resolved)
        return resolved.parent, (resolved,)
    paths = tuple(
        sorted(
            (
                path
                for path in resolved.rglob("*")
                if path.is_file() and path.suffix.lower() in registry.supported_suffixes
            ),
            key=lambda path: path.as_posix().casefold(),
        )
    )
    if not paths:
        suffixes = ", ".join(sorted(registry.supported_suffixes))
        raise FileNotFoundError(f"no supported files found in {resolved} ({suffixes})")
    return resolved, paths


def _validated_case(value: object, *, index: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Ragas test set item {index} must be an object")
    user_input = value.get("user_input")
    reference = value.get("reference")
    reference_contexts = value.get("reference_contexts")
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError(f"Ragas test set item {index} has no user_input")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"Ragas test set item {index} has no reference")
    if not isinstance(reference_contexts, list) or not all(
        isinstance(item, str) for item in reference_contexts
    ):
        raise ValueError(f"Ragas test set item {index} has invalid reference_contexts")
    return {
        "user_input": user_input,
        "reference": reference,
        "reference_contexts": list(reference_contexts),
    }


def _validated_envelope(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Ragas test set envelope must be an object")
    if value.get("schema_version") != RAGAS_TESTSET_SCHEMA_VERSION:
        raise ValueError(
            "unsupported Ragas test set schema_version "
            f"{value.get('schema_version')!r}; expected {RAGAS_TESTSET_SCHEMA_VERSION!r}"
        )
    cases = value.get("cases")
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError("Ragas test set envelope cases must be an array of objects")
    return build_ragas_testset_envelope(
        value.get("corpus_content_hash"),
        cast(list[dict[str, object]], cases),
    )


def _validate_limit(value: int | None, *, field: str) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{field} must be positive when provided")


__all__ = [
    "RAGAS_TESTSET_SCHEMA_VERSION",
    "LocalBGEEmbeddings",
    "build_ragas_testset_envelope",
    "cases_from_dataframe",
    "generate_ragas_cases",
    "load_ragas_documents",
    "validate_corpus_content_hash",
    "write_ragas_testset",
]
