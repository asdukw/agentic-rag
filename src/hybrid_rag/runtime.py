"""Shared runtime factories for the CLI-adjacent FastAPI workbench."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_rag.config import DeepSeekSettings, RetrievalSettings, Settings
from hybrid_rag.ingest.tokenizer import TiktokenCounter
from hybrid_rag.retrieval.embedding import BGEM3EmbeddingProvider, HashEmbeddingProvider
from hybrid_rag.retrieval.query import DeepSeekQueryClient, DeterministicQueryClient, QueryClient
from hybrid_rag.retrieval.reranker import create_reranker
from hybrid_rag.retrieval.service import RetrievalOptions, RetrievalService
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database


@dataclass(frozen=True, slots=True)
class ApplicationRuntime:
    """Execution choices supplied by the TypeScript workbench API."""

    database_url: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    options: RetrievalOptions
    use_deepseek: bool


def create_embedding_provider(
    runtime: ApplicationRuntime,
    settings: RetrievalSettings,
) -> BGEM3EmbeddingProvider | HashEmbeddingProvider:
    """Construct the embedding adapter selected by one API request."""

    if runtime.embedding_provider == "flagembedding":
        return BGEM3EmbeddingProvider(
            model=runtime.embedding_model,
            dimensions=runtime.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
            max_length=settings.embedding_max_length,
            use_fp16=settings.embedding_use_fp16,
        )
    if runtime.embedding_provider == "hash":
        return HashEmbeddingProvider(
            dimensions=runtime.embedding_dimensions,
            model=runtime.embedding_model,
        )
    raise ValueError("embedding provider must be 'flagembedding' or 'hash'")


def create_query_client(use_deepseek: bool, settings: DeepSeekSettings) -> QueryClient:
    """Select the offline client or the constrained DeepSeek adapter."""

    if not use_deepseek:
        return DeterministicQueryClient()
    api_key = settings.api_key.get_secret_value().strip() if settings.api_key else ""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required when DeepSeek is enabled")
    return DeepSeekQueryClient(
        api_key=api_key,
        model=settings.query_model,
        answer_model=settings.answer_model,
        base_url=settings.base_url,
        max_output_tokens=max(512, settings.answer_max_output_tokens),
        timeout_seconds=settings.timeout_seconds,
    )


def create_service(
    runtime: ApplicationRuntime,
    settings: Settings,
    retrieval: RetrievalSettings,
    deepseek: DeepSeekSettings | None = None,
) -> tuple[Database, RetrievalService]:
    """Upgrade the selected database and return a short-lived retrieval service."""

    upgrade_database(runtime.database_url)
    database = Database(runtime.database_url)
    service = RetrievalService(
        database,
        create_embedding_provider(runtime, retrieval),
        TiktokenCounter(settings.tokenizer_name),
        reranker=create_reranker(
            runtime.options.reranker_provider,
            runtime.options.reranker_model,
            use_fp16=runtime.options.reranker_use_fp16,
        ),
        deepseek_pricing=(deepseek or DeepSeekSettings()).pricing,
    )
    return database, service


__all__ = [
    "ApplicationRuntime",
    "create_embedding_provider",
    "create_query_client",
    "create_service",
]
