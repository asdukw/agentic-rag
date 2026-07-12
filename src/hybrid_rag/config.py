from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HYBRID_RAG_",
        extra="ignore",
    )

    database_url: str = "sqlite:///storage/app.db"
    chunk_size_tokens: int = Field(default=512, ge=32)
    chunk_overlap_tokens: int = Field(default=64, ge=0)
    tokenizer_name: str = "cl100k_base"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_chunk_window(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class DeepSeekSettings(BaseSettings):
    """DeepSeek credentials and request limits, loaded only by extraction commands."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DEEPSEEK_",
        extra="ignore",
    )

    api_key: SecretStr | None = None
    base_url: str = Field(default="https://api.deepseek.com", min_length=1)
    extraction_model: str = Field(default="deepseek-v4-flash", min_length=1)
    query_model: str = Field(default="deepseek-v4-flash", min_length=1)
    answer_model: str = Field(default="deepseek-v4-flash", min_length=1)
    max_output_tokens: int = Field(default=4096, ge=256)
    answer_max_output_tokens: int = Field(default=2048, ge=128)
    timeout_seconds: float = Field(default=180.0, gt=0)


class GraphSettings(BaseSettings):
    """Local graph-build execution settings; these do not affect extraction identity."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HYBRID_RAG_GRAPH_",
        extra="ignore",
    )

    checkpoint_path: Path = Path("storage/langgraph.db")
    max_concurrency: int = Field(default=8, ge=1)
    max_attempts: int = Field(default=3, ge=1)
    top_k: int = Field(default=10, ge=1)


class RetrievalSettings(BaseSettings):
    """Retrieval execution defaults; semantic index identity is persisted separately."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HYBRID_RAG_RETRIEVAL_",
        extra="ignore",
    )

    embedding_provider: str = Field(default="hash", min_length=1)
    embedding_model: str = Field(default="hash-token-v1", min_length=1)
    embedding_dimensions: int = Field(default=384, ge=32, le=4096)
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_batch_size: int = Field(default=64, ge=1, le=1024)
    top_k: int = Field(default=8, ge=1)
    candidate_multiplier: int = Field(default=4, ge=1, le=32)
    context_token_budget: int = Field(default=2400, ge=128)
    graph_max_hops: int = Field(default=2, ge=1, le=4)
    naive_weight: float = Field(default=1.0, ge=0)
    local_weight: float = Field(default=1.0, ge=0)
    global_weight: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def validate_fusion_weights(self) -> RetrievalSettings:
        if self.naive_weight + self.local_weight + self.global_weight <= 0:
            raise ValueError("at least one retrieval fusion weight must be positive")
        return self


def sqlite_url(path: Path) -> str:
    """Build an absolute SQLAlchemy SQLite URL for a local file."""

    absolute = path.expanduser().resolve()
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{absolute.as_posix()}"
