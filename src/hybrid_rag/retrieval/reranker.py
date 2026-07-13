"""Query-aware reranking contracts and a local FlagEmbedding adapter.

The first-stage retrievers intentionally optimise recall.  This module receives
their already-selected chunk candidates and scores only those candidates again;
they do not load an index or mutate persistence. ``FlagEmbeddingReranker``
lazily loads a local cross-encoder behind the :class:`Reranker` protocol.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, ClassVar, Protocol, runtime_checkable

FLAG_EMBEDDING_RERANKER_VERSION = "flagembedding-reranker-v1"
FLAG_EMBEDDING_RERANKER_PROVIDER = "flagembedding"
FLAG_EMBEDDING_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class RerankerConfigurationError(RuntimeError):
    """Raised when the configured local reranker cannot be constructed."""


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """One preselected chunk that may be reranked for a query.

    ``prior_score`` retains the first-stage score as a stable tie-breaker.
    """

    object_id: str
    text: str
    prior_score: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("object_id must be a non-blank string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-blank string")
        if not math.isfinite(self.prior_score):
            raise ValueError("prior_score must be finite")


@dataclass(frozen=True, slots=True)
class RerankScoreComponent:
    """One normalized contribution to an explainable reranker score."""

    raw_score: float
    normalized_score: float
    weight: float
    weighted_score: float

    def __post_init__(self) -> None:
        values = (
            self.raw_score,
            self.normalized_score,
            self.weight,
            self.weighted_score,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("reranker score components must be finite")
        if not 0.0 <= self.normalized_score <= 1.0:
            raise ValueError("normalized_score must be between zero and one")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be between zero and one")
        if self.weighted_score < 0.0:
            raise ValueError("weighted_score must not be negative")


@dataclass(frozen=True, slots=True)
class RerankHit:
    """A reranked candidate with every score component retained for tracing."""

    candidate: RerankCandidate
    score: float
    components: Mapping[str, RerankScoreComponent]

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or self.score < 0.0:
            raise ValueError("score must be a finite non-negative value")
        if not self.components:
            raise ValueError("components must not be empty")


@runtime_checkable
class Reranker(Protocol):
    """Adapter boundary for query-aware chunk rerankers."""

    provider: str
    model: str
    version: str

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        limit: int | None = None,
    ) -> tuple[RerankHit, ...]: ...


class FlagEmbeddingReranker:
    """Rerank candidates with a local FlagEmbedding cross-encoder.

    The adapter batches all ``[query, passage]`` pairs into one
    :meth:`FlagReranker.compute_score` call.  It keeps the model's raw logit in
    the trace and applies the same sigmoid used by FlagEmbedding's
    ``normalize=True`` option for the final 0--1 relevance score.

    Importing and constructing ``FlagReranker`` is deliberately deferred until
    the first rerank call, so configuration and index-building do not download
    model weights. Loaded models are shared by model ID and precision mode within
    a process so repeated Streamlit actions do not reload the same weights.
    """

    provider = FLAG_EMBEDDING_RERANKER_PROVIDER
    version = FLAG_EMBEDDING_RERANKER_VERSION
    _shared_clients: ClassVar[dict[tuple[str, bool], Any]] = {}
    _shared_client_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        model: str = FLAG_EMBEDDING_RERANKER_MODEL,
        *,
        use_fp16: bool = False,
        client: Any | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("FlagEmbedding reranker model must be a non-blank string")
        if not isinstance(use_fp16, bool):
            raise TypeError("use_fp16 must be a boolean")
        self.model = model
        self.use_fp16 = use_fp16
        self._client = client

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        limit: int | None = None,
    ) -> tuple[RerankHit, ...]:
        """Return all supplied candidates ordered by cross-encoder relevance."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if limit is not None and limit < 0:
            raise ValueError("limit must not be negative")
        if limit == 0:
            return ()

        candidate_rows = tuple(candidates)
        _validate_candidates(candidate_rows)
        if not candidate_rows:
            return ()

        pairs = [[query, candidate.text] for candidate in candidate_rows]
        raw_scores = _coerce_flag_embedding_scores(
            self._get_client().compute_score(pairs, normalize=False),
            expected_count=len(candidate_rows),
        )
        hits = tuple(
            _cross_encoder_hit(candidate, raw_score)
            for candidate, raw_score in zip(candidate_rows, raw_scores, strict=True)
        )
        ordered = tuple(
            sorted(
                hits,
                key=lambda hit: (-hit.score, -hit.candidate.prior_score, hit.candidate.object_id),
            )
        )
        return ordered if limit is None else ordered[:limit]

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        cache_key = (self.model, self.use_fp16)
        with self._shared_client_lock:
            cached = self._shared_clients.get(cache_key)
            if cached is None:
                try:
                    from FlagEmbedding import FlagReranker
                except ImportError as error:
                    raise RerankerConfigurationError(
                        "FlagEmbedding reranker is not installed. Run `uv sync` "
                        "or set HYBRID_RAG_RETRIEVAL_RERANKER_PROVIDER=none."
                    ) from error
                try:
                    cached = FlagReranker(self.model, use_fp16=self.use_fp16)
                except (OSError, RuntimeError, ValueError) as error:
                    raise RerankerConfigurationError(
                        f"Unable to load FlagEmbedding reranker model {self.model!r}: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                self._shared_clients[cache_key] = cached
        self._client = cached
        return cached


def create_reranker(
    provider: str,
    model: str,
    *,
    use_fp16: bool = False,
) -> Reranker | None:
    """Create the configured reranker without loading model weights yet."""

    if provider == "none":
        return None
    if provider == FLAG_EMBEDDING_RERANKER_PROVIDER:
        return FlagEmbeddingReranker(model, use_fp16=use_fp16)
    raise ValueError(f"reranker provider must be 'none' or 'flagembedding' (got {provider!r})")


def _cross_encoder_hit(candidate: RerankCandidate, raw_score: float) -> RerankHit:
    normalized_score = _sigmoid(raw_score)
    component = RerankScoreComponent(
        raw_score=raw_score,
        normalized_score=normalized_score,
        weight=1.0,
        weighted_score=normalized_score,
    )
    return RerankHit(
        candidate=candidate,
        score=normalized_score,
        components={"cross_encoder": component},
    )


def _coerce_flag_embedding_scores(response: Any, *, expected_count: int) -> tuple[float, ...]:
    """Convert FlagEmbedding's dynamic scalar-or-sequence response into finite logits."""

    raw_values: tuple[Any, ...]
    if isinstance(response, (str, bytes)):
        raw_values = (response,)
    else:
        try:
            raw_values = tuple(response)
        except TypeError:
            raw_values = (response,)
    try:
        scores = tuple(float(value) for value in raw_values)
    except (TypeError, ValueError) as error:
        raise RuntimeError("FlagEmbedding returned a non-numeric rerank score") from error
    if len(scores) != expected_count:
        raise RuntimeError(
            "FlagEmbedding returned an unexpected number of rerank scores "
            f"({len(scores)} != {expected_count})"
        )
    if not all(math.isfinite(score) for score in scores):
        raise RuntimeError("FlagEmbedding returned a non-finite rerank score")
    return scores


def _sigmoid(value: float) -> float:
    """Return a stable logistic normalization matching FlagEmbedding's option."""

    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _validate_candidates(candidates: Sequence[RerankCandidate]) -> None:
    object_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, RerankCandidate):
            raise TypeError("candidates must contain RerankCandidate values")
        if candidate.object_id in object_ids:
            raise ValueError(f"reranker received duplicate candidate ID: {candidate.object_id}")
        object_ids.add(candidate.object_id)


__all__ = [
    "FLAG_EMBEDDING_RERANKER_MODEL",
    "FLAG_EMBEDDING_RERANKER_PROVIDER",
    "FLAG_EMBEDDING_RERANKER_VERSION",
    "FlagEmbeddingReranker",
    "RerankCandidate",
    "RerankHit",
    "RerankScoreComponent",
    "Reranker",
    "RerankerConfigurationError",
    "create_reranker",
]
