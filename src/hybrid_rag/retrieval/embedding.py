"""Small, replaceable embedding adapters used by retrieval indexing and querying."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Protocol, SupportsFloat, TypeGuard, cast

BGE_M3_PROVIDER = "flagembedding"
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_DIMENSIONS = 1024
BGE_M3_MAX_LENGTH = 8192

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Adapter boundary for deterministic or local embedding implementations."""

    provider: str
    model: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class HashEmbeddingProvider:
    """A deterministic sparse-feature embedding suitable for offline demos and tests.

    It is deliberately not presented as a substitute for a trained semantic model.
    Its vector shape and hashing version are included in the persisted index
    configuration, so it remains useful for compatibility checks and fast tests.
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
    """Raised when a model-backed embedding adapter cannot be initialized."""


class BGEM3EmbeddingProvider:
    """Encode text with FlagEmbedding's local BGE-M3 dense-vector model.

    Model construction is lazy so commands that only inspect existing data do not
    download or initialize model weights. Loaded models are shared by model ID and
    precision mode within a process, which keeps repeated API requests cheap.
    """

    provider = BGE_M3_PROVIDER
    _shared_clients: ClassVar[dict[tuple[str, bool], Any]] = {}
    _shared_client_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        *,
        model: str = BGE_M3_MODEL,
        dimensions: int = BGE_M3_DIMENSIONS,
        batch_size: int = 12,
        max_length: int = BGE_M3_MAX_LENGTH,
        use_fp16: bool = False,
        client: Any | None = None,
    ) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("model must not be blank")
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if not isinstance(use_fp16, bool):
            raise TypeError("use_fp16 must be a boolean")
        self.model = normalized_model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self._model_client = client

    @property
    def semantic_options(self) -> dict[str, bool | int]:
        """Encoding options that can change persisted vector values."""

        return {
            "max_length": self.max_length,
            "normalize_embeddings": True,
            "use_fp16": self.use_fp16,
        }

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        response = self._client().encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        try:
            dense_vectors = response["dense_vecs"]
        except (KeyError, TypeError) as error:
            raise RuntimeError("FlagEmbedding did not return dense_vecs") from error
        return _dense_vectors(
            dense_vectors,
            expected_count=len(texts),
            dimensions=self.dimensions,
        )

    def _client(self) -> Any:
        if self._model_client is not None:
            return self._model_client
        cache_key = (self.model, self.use_fp16)
        with self._shared_client_lock:
            cached = self._shared_clients.get(cache_key)
            if cached is None:
                try:
                    from FlagEmbedding import BGEM3FlagModel
                except ImportError as error:
                    raise EmbeddingConfigurationError(
                        "FlagEmbedding is not installed. Run `uv sync` before building an index."
                    ) from error
                try:
                    devices, cuda_enabled = _preferred_embedding_devices()
                    effective_fp16 = self.use_fp16 and cuda_enabled
                    if cuda_enabled:
                        logger.warning(
                            "BGE-M3 embedding selected CUDA devices=%s fp16=%s",
                            ",".join(devices),
                            effective_fp16,
                        )
                    else:
                        logger.warning(
                            "CUDA is unavailable; BGE-M3 embedding is falling back to CPU "
                            "with fp16 disabled"
                        )
                    model_path = _prefer_cached_huggingface_snapshot(self.model)
                    cached = BGEM3FlagModel(
                        model_path,
                        normalize_embeddings=True,
                        use_fp16=effective_fp16,
                        devices=devices,
                    )
                except Exception as error:
                    raise EmbeddingConfigurationError(
                        f"Unable to load FlagEmbedding embedding model {self.model!r}: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                self._shared_clients[cache_key] = cached
        self._model_client = cached
        return cached


def _prefer_cached_huggingface_snapshot(model: str) -> str:
    """Use an existing local snapshot without making a remote metadata request."""

    if Path(model).expanduser().exists():
        return model
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        return model
    try:
        snapshot = snapshot_download(repo_id=model, local_files_only=True)
    except (LocalEntryNotFoundError, OSError, ValueError):
        logger.warning(
            "Embedding model %s is not cached locally; downloading it from Hugging Face",
            model,
        )
        return model
    logger.warning("Embedding model %s is loading from local cache %s", model, snapshot)
    return snapshot


def _preferred_embedding_devices() -> tuple[list[str], bool]:
    """Prefer every visible CUDA device and explicitly fall back to CPU."""

    try:
        import torch
    except ImportError as error:
        raise EmbeddingConfigurationError(
            "PyTorch is not installed. Run `uv sync` before building an index."
        ) from error
    if torch.cuda.is_available() and (device_count := torch.cuda.device_count()) > 0:
        return [f"cuda:{index}" for index in range(device_count)], True
    return ["cpu"], False


def _dense_vectors(
    dense_vectors: object,
    *,
    expected_count: int,
    dimensions: int,
) -> tuple[tuple[float, ...], ...]:
    """Validate FlagEmbedding's NumPy-like dense-vector result without importing NumPy."""

    if isinstance(dense_vectors, (str, bytes)) or not isinstance(dense_vectors, Iterable):
        raise RuntimeError("FlagEmbedding returned invalid dense vectors")
    try:
        vectors: list[tuple[float, ...]] = []
        for vector in dense_vectors:
            if not _is_vector(vector):
                raise TypeError("dense vector must be iterable")
            vectors.append(tuple(_float_value(value) for value in vector))
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("FlagEmbedding returned invalid dense vectors") from error
    if len(vectors) != expected_count:
        raise RuntimeError("FlagEmbedding returned a different number of vectors")
    if any(len(vector) != dimensions for vector in vectors):
        observed = sorted({len(vector) for vector in vectors})
        raise RuntimeError(
            f"FlagEmbedding dimensions differ from configured {dimensions}: {observed}"
        )
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise RuntimeError("FlagEmbedding returned a non-finite dense vector value")
    return tuple(vectors)


def _is_vector(value: object) -> TypeGuard[Iterable[object]]:
    return not isinstance(value, (str, bytes)) and isinstance(value, Iterable)


def _float_value(value: object) -> float:
    if isinstance(value, (str, bytes, bytearray)):
        return float(value)
    return float(cast(SupportsFloat, value))


_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def _features(text: str) -> tuple[str, ...]:
    normalized = text.casefold()
    values: list[str] = []
    for token in _WORD_PATTERN.findall(normalized):
        values.append(f"word:{token}")
        if len(token) >= 3:
            values.extend(f"gram:{token[index : index + 3]}" for index in range(len(token) - 2))
    cjk = [char for char in normalized if "\u3400" <= char <= "\u9fff"]
    values.extend(f"cjk:{char}" for char in cjk)
    values.extend(f"cjk2:{''.join(cjk[index : index + 2])}" for index in range(len(cjk) - 1))
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
