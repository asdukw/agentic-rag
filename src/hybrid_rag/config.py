from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
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


def sqlite_url(path: Path) -> str:
    """Build an absolute SQLAlchemy SQLite URL for a local file."""

    absolute = path.expanduser().resolve()
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{absolute.as_posix()}"
