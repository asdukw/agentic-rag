"""Loading helpers for the repository's versioned evaluation fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from hybrid_rag.evaluation.contracts import EvaluationBenchmark


class BenchmarkLoadError(ValueError):
    """Raised when a benchmark file is unreadable or violates its contract."""


def fixture_benchmark_path() -> Path:
    """Return the repository-owned fixed v1 benchmark path."""

    return Path(__file__).resolve().parents[3] / "data" / "evaluation" / "fixture-benchmark-v1.json"


def load_benchmark(path: str | Path) -> EvaluationBenchmark:
    """Load and strictly validate one JSON benchmark fixture."""

    benchmark_path = Path(path)
    try:
        raw = benchmark_path.read_text(encoding="utf-8")
    except OSError as error:
        raise BenchmarkLoadError(f"unable to read benchmark {benchmark_path}: {error}") from error
    try:
        json.loads(raw)
    except json.JSONDecodeError as error:
        raise BenchmarkLoadError(
            f"invalid benchmark JSON in {benchmark_path}: {error.msg}"
        ) from error
    try:
        # JSON validation is intentionally separate from Python-object validation:
        # strict Pydantic contracts still accept JSON arrays for tuple fields.
        return EvaluationBenchmark.model_validate_json(raw)
    except ValidationError as error:
        raise BenchmarkLoadError(
            f"invalid benchmark contract in {benchmark_path}: {error}"
        ) from error
