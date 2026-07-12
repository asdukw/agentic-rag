"""Small, replaceable embedding adapters used by retrieval indexing and querying."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    """Adapter boundary for deterministic or external embedding implementations."""

    provider: str
    model: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class HashEmbeddingProvider:
    """A deterministic sparse-feature embedding suitable for offline demos and tests.

    It is deliberately not presented as a substitute for a trained semantic model.
    Its purpose is a reproducible default vector adapter until a benchmark selects
    a hosted or local embedding model.  The vector shape and hashing version are
    included in the persisted index configuration.
    """

    provider = "hash"
    model = "hash-token-v1"

    def __init__(self, *, dimensions: int = 384, model: str | None = None) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        if model is not None:
            if not model.strip():
                raise ValueError("model must not be blank")
            self.model = model

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed_one(text) for text in texts)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        features = Counter(_features(text))
        vector = [0.0] * self.dimensions
        for feature, count in features.items():
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign * float(count)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return tuple(vector)


class EmbeddingConfigurationError(RuntimeError):
    """Raised when an external embedding adapter is selected without credentials."""


class OpenAICompatibleEmbeddingProvider:
    """Thin synchronous adapter for a configured OpenAI-compatible embeddings API.

    It is intentionally optional: the project ships with :class:`HashEmbeddingProvider`
    for offline reproducibility, while deployments can choose a benchmarked provider
    without changing any index or retrieval algorithms.
    """

    provider = "openai-compatible"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str,
        dimensions: int,
        timeout_seconds: float = 180.0,
        sdk_client: object | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be blank")
        if not model.strip():
            raise ValueError("model must not be blank")
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url
        self.model = model
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        self._api_key = api_key
        self._sdk_client = sdk_client

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        response = self._client().embeddings.create(model=self.model, input=list(texts))
        data = sorted(
            _attribute(response, "data", ()),
            key=lambda item: _integer(_attribute(item, "index")),
        )
        if len(data) != len(texts):
            raise RuntimeError("embedding provider returned a different number of vectors")
        vectors = tuple(
            tuple(float(value) for value in _attribute(item, "embedding", ())) for item in data
        )
        if any(len(vector) != self.dimensions for vector in vectors):
            observed = sorted({len(vector) for vector in vectors})
            raise RuntimeError(
                "embedding provider dimensions differ from configured "
                f"{self.dimensions}: {observed}"
            )
        return vectors

    def _client(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        if not self._api_key:
            raise EmbeddingConfigurationError(
                "an API key is required before an OpenAI-compatible embedding request"
            )
        from openai import OpenAI

        self._sdk_client = OpenAI(
            api_key=self._api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=self.timeout_seconds,
        )
        return self._sdk_client


_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def _features(text: str) -> tuple[str, ...]:
    normalized = text.casefold()
    values: list[str] = []
    for token in _WORD_PATTERN.findall(normalized):
        values.append(f"word:{token}")
        if len(token) >= 3:
            values.extend(f"gram:{token[index:index + 3]}" for index in range(len(token) - 2))
    cjk = [char for char in normalized if "\u3400" <= char <= "\u9fff"]
    values.extend(f"cjk:{char}" for char in cjk)
    values.extend(f"cjk2:{''.join(cjk[index:index + 2])}" for index in range(len(cjk) - 1))
    return tuple(values)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return a safe cosine similarity and reject incompatible persisted vectors."""

    if len(left) != len(right):
        raise ValueError(f"embedding dimensions differ: {len(left)} != {len(right)}")
    if not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def min_max_normalize(values: dict[str, float]) -> dict[str, float]:
    """Normalize one route's scores without introducing order-dependent state."""

    if not values:
        return {}
    lower = min(values.values())
    upper = max(values.values())
    if math.isclose(lower, upper):
        return {key: 1.0 for key in values}
    return {key: (value - lower) / (upper - lower) for key, value in values.items()}


def _attribute(value: object | None, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _integer(value: object | None) -> int:
    return value if isinstance(value, int) else 0
