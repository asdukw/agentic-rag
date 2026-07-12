"""Deterministic, query-aware reranking contracts and an offline baseline.

The first-stage retrievers intentionally optimise recall.  This module receives
their already-selected chunk candidates and scores only those candidates again;
it does not load an index, call a model, or mutate persistence.  The
``LexicalReranker`` is a transparent offline baseline that can later be swapped
for a cross-encoder behind the same :class:`Reranker` protocol.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol, runtime_checkable

from hybrid_rag.retrieval.bm25 import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    LEXICAL_TOKENIZER_VERSION,
    tokenize_lexical,
)

LEXICAL_RERANKER_VERSION = "lexical-reranker-v1"
LEXICAL_RERANKER_PROVIDER = "lexical"
LEXICAL_RERANKER_MODEL = "lexical-coverage-v1"


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """One preselected chunk that may be reranked for a query.

    ``prior_score`` is the first-stage score.  It is deliberately retained as a
    small, explicit component so a lexical no-match does not discard a useful
    dense candidate merely because its wording differs from the question.
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


@dataclass(frozen=True, slots=True)
class LexicalRerankerConfig:
    """Weights and BM25 parameters for :class:`LexicalReranker`.

    The configured weights are converted to shares during scoring, so callers
    may use any non-negative relative weights.  ``prior_weight`` defaults to a
    deliberately small value: dense/graph recall remains a fallback, not an
    opaque replacement for the rerank signal.
    """

    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B
    bm25_weight: float = 0.55
    coverage_weight: float = 0.25
    proximity_weight: float = 0.15
    prior_weight: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.bm25_k1) or self.bm25_k1 <= 0.0:
            raise ValueError("bm25_k1 must be a finite value greater than zero")
        if not math.isfinite(self.bm25_b) or not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError("bm25_b must be a finite value between zero and one")
        weights = self.weights
        if not all(math.isfinite(value) and value >= 0.0 for value in weights.values()):
            raise ValueError("reranker weights must be finite non-negative values")
        if sum(weights.values()) <= 0.0:
            raise ValueError("at least one reranker weight must be positive")

    @property
    def weights(self) -> dict[str, float]:
        """Return component names with their configured, unnormalized weights."""

        return {
            "bm25": self.bm25_weight,
            "coverage": self.coverage_weight,
            "proximity": self.proximity_weight,
            "prior": self.prior_weight,
        }


class LexicalReranker:
    """Rerank preselected chunks by lexical relevance, coverage, and proximity.

    BM25 is calculated only over the supplied candidate set, which makes this a
    second-stage scorer rather than another corpus-wide retrieval route.  The
    score has four components:

    * ``bm25``: term frequency with candidate-set document-frequency and length
      normalization;
    * ``coverage``: IDF-weighted fraction of query terms found in a chunk;
    * ``proximity``: ordered closeness of the matched query terms in the chunk;
    * ``prior``: normalized first-stage score, retained as a low-weight fallback.

    Scores are deterministic and sorted by final score, raw prior score, then
    candidate ID.  All candidates are returned (unless ``limit`` is supplied),
    including candidates with no lexical match.
    """

    provider = LEXICAL_RERANKER_PROVIDER
    model = LEXICAL_RERANKER_MODEL
    version = LEXICAL_RERANKER_VERSION

    def __init__(self, *, config: LexicalRerankerConfig | None = None) -> None:
        self.config = config or LexicalRerankerConfig()

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        limit: int | None = None,
    ) -> tuple[RerankHit, ...]:
        """Return every supplied candidate in deterministic reranked order."""

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

        query_terms = tuple(dict.fromkeys(tokenize_lexical(query)))
        term_frequencies = tuple(Counter(tokenize_lexical(row.text)) for row in candidate_rows)
        document_lengths = tuple(sum(values.values()) for values in term_frequencies)
        average_document_length = sum(document_lengths) / len(document_lengths)
        document_frequencies = Counter(
            term for frequencies in term_frequencies for term in frequencies
        )
        inverse_document_frequencies = {
            term: _inverse_document_frequency(len(candidate_rows), document_frequencies[term])
            for term in query_terms
        }
        idf_total = sum(inverse_document_frequencies.values())

        bm25_raw: dict[str, float] = {}
        coverage_raw: dict[str, float] = {}
        proximity_raw: dict[str, float] = {}
        for candidate, frequencies, document_length in zip(
            candidate_rows, term_frequencies, document_lengths, strict=True
        ):
            bm25_raw[candidate.object_id] = _bm25_score(
                frequencies,
                query_terms,
                inverse_document_frequencies,
                document_length=document_length,
                average_document_length=average_document_length,
                config=self.config,
            )
            coverage_raw[candidate.object_id] = _idf_coverage(
                frequencies,
                inverse_document_frequencies,
                idf_total,
            )
            proximity_raw[candidate.object_id] = _ordered_proximity(query, candidate.text)

        bm25_normalized = _positive_max_normalize(bm25_raw)
        prior_normalized = _min_max_normalize(
            {candidate.object_id: candidate.prior_score for candidate in candidate_rows}
        )
        normalized_scores: dict[str, Mapping[str, float]] = {
            candidate.object_id: {
                "bm25": bm25_normalized[candidate.object_id],
                "coverage": coverage_raw[candidate.object_id],
                "proximity": proximity_raw[candidate.object_id],
                "prior": prior_normalized[candidate.object_id],
            }
            for candidate in candidate_rows
        }
        weight_shares = _weight_shares(self.config.weights)

        hits = tuple(
            _hit(
                candidate,
                raw_scores={
                    "bm25": bm25_raw[candidate.object_id],
                    "coverage": coverage_raw[candidate.object_id],
                    "proximity": proximity_raw[candidate.object_id],
                    "prior": candidate.prior_score,
                },
                normalized_scores=normalized_scores[candidate.object_id],
                weight_shares=weight_shares,
            )
            for candidate in candidate_rows
        )
        ordered = tuple(
            sorted(
                hits,
                key=lambda hit: (-hit.score, -hit.candidate.prior_score, hit.candidate.object_id),
            )
        )
        return ordered if limit is None else ordered[:limit]


def _hit(
    candidate: RerankCandidate,
    *,
    raw_scores: Mapping[str, float],
    normalized_scores: Mapping[str, float],
    weight_shares: Mapping[str, float],
) -> RerankHit:
    components = {
        name: RerankScoreComponent(
            raw_score=float(raw_scores[name]),
            normalized_score=float(normalized_scores[name]),
            weight=float(weight_shares[name]),
            weighted_score=float(normalized_scores[name] * weight_shares[name]),
        )
        for name in ("bm25", "coverage", "proximity", "prior")
    }
    return RerankHit(
        candidate=candidate,
        score=sum(component.weighted_score for component in components.values()),
        components=components,
    )


def _bm25_score(
    frequencies: Counter[str],
    query_terms: Sequence[str],
    inverse_document_frequencies: Mapping[str, float],
    *,
    document_length: int,
    average_document_length: float,
    config: LexicalRerankerConfig,
) -> float:
    length_ratio = document_length / average_document_length if average_document_length else 0.0
    length_normalizer = config.bm25_k1 * (
        1.0 - config.bm25_b + config.bm25_b * length_ratio
    )
    score = 0.0
    for term in query_terms:
        term_frequency = frequencies.get(term, 0)
        if term_frequency == 0:
            continue
        score += inverse_document_frequencies[term] * (
            term_frequency
            * (config.bm25_k1 + 1.0)
            / (term_frequency + length_normalizer)
        )
    return score


def _idf_coverage(
    frequencies: Counter[str],
    inverse_document_frequencies: Mapping[str, float],
    idf_total: float,
) -> float:
    if idf_total <= 0.0:
        return 0.0
    matched = sum(
        inverse_document_frequency
        for term, inverse_document_frequency in inverse_document_frequencies.items()
        if frequencies.get(term, 0) > 0
    )
    return matched / idf_total


def _ordered_proximity(query: str, text: str) -> float:
    query_terms = tuple(dict.fromkeys(_position_terms(query)))
    document_terms = _position_terms(text)
    if len(query_terms) < 2 or not document_terms:
        return 0.0

    positions: dict[str, list[int]] = defaultdict(list)
    for index, term in enumerate(document_terms):
        positions[term].append(index)
    matched_query_terms = tuple(term for term in query_terms if term in positions)
    if len(matched_query_terms) < 2:
        return 0.0

    pair_scores = tuple(
        _ordered_pair_score(positions[left], positions[right])
        for left, right in pairwise(matched_query_terms)
    )
    return sum(pair_scores) / len(pair_scores) if pair_scores else 0.0


def _position_terms(text: str) -> tuple[str, ...]:
    """Return lexical terms in textual order, without synthetic CJK bigrams."""

    return tuple(term for term in tokenize_lexical(text) if not term.startswith("cjk2:"))


def _ordered_pair_score(left_positions: Sequence[int], right_positions: Sequence[int]) -> float:
    right_index = 0
    best_distance: int | None = None
    for left in left_positions:
        while right_index < len(right_positions) and right_positions[right_index] <= left:
            right_index += 1
        if right_index >= len(right_positions):
            break
        distance = right_positions[right_index] - left
        if best_distance is None or distance < best_distance:
            best_distance = distance
    return 1.0 / best_distance if best_distance is not None else 0.0


def _inverse_document_frequency(document_count: int, document_frequency: int) -> float:
    return math.log(
        1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    )


def _positive_max_normalize(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    upper = max(values.values())
    if upper <= 0.0:
        return {key: 0.0 for key in values}
    return {key: value / upper for key, value in values.items()}


def _min_max_normalize(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lower = min(values.values())
    upper = max(values.values())
    if math.isclose(lower, upper):
        normalized = 1.0 if not math.isclose(upper, 0.0) else 0.0
        return {key: normalized for key in values}
    return {key: (value - lower) / (upper - lower) for key, value in values.items()}


def _weight_shares(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def _validate_candidates(candidates: Sequence[RerankCandidate]) -> None:
    object_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, RerankCandidate):
            raise TypeError("candidates must contain RerankCandidate values")
        if candidate.object_id in object_ids:
            raise ValueError(f"reranker received duplicate candidate ID: {candidate.object_id}")
        object_ids.add(candidate.object_id)


__all__ = [
    "LEXICAL_RERANKER_MODEL",
    "LEXICAL_RERANKER_PROVIDER",
    "LEXICAL_RERANKER_VERSION",
    "LEXICAL_TOKENIZER_VERSION",
    "LexicalReranker",
    "LexicalRerankerConfig",
    "RerankCandidate",
    "RerankHit",
    "RerankScoreComponent",
    "Reranker",
]
