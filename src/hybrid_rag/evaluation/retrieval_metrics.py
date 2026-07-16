"""Deterministic ranking metrics for corpus-bound retrieval evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RankedEvidence:
    """One ranked retrieval result with stable and text-based identities."""

    evidence_ids: tuple[str, ...]
    text: str


@dataclass(frozen=True, slots=True)
class RetrievalMetricScores:
    """Binary-relevance retrieval metrics for one test-set case."""

    k: int
    applicable: bool
    matching_method: str
    relevant_count: int
    retrieved_count: int
    matched_count: int
    hit_at_k: float | None
    recall_at_k: float | None
    mrr: float | None
    ndcg_at_k: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def score_retrieval(
    ranked_evidence: Sequence[RankedEvidence],
    *,
    k: int,
    evidence_ids: Sequence[str] | None,
    reference_contexts: Sequence[str],
) -> RetrievalMetricScores:
    """Score one ranked list using IDs, with exact normalized-text fallback.

    ``evidence_ids=None`` means the older test-set schema did not carry stable
    evidence identities. An explicitly empty ID list represents an unanswerable
    case and is therefore excluded from relevance-based aggregate metrics.
    """

    if k < 1:
        raise ValueError("retrieval metric k must be positive")
    selected = tuple(ranked_evidence[:k])
    matched_relevant: set[str] = set()
    if evidence_ids is not None:
        relevant = frozenset(evidence_ids)
        seen: set[str] = set()
        relevance_values: list[bool] = []
        for item in selected:
            identities = set(item.evidence_ids)
            matched = (identities & relevant) - seen
            is_relevant = bool(matched)
            relevance_values.append(is_relevant)
            matched_relevant.update(matched)
            seen.update(identities)
        relevance = tuple(relevance_values)
        matching_method = "evidence_id"
    else:
        relevant = frozenset(
            normalized
            for context in reference_contexts
            if (normalized := normalize_context(context))
        )
        seen = set()
        relevance_values = []
        for item in selected:
            identity = normalize_context(item.text)
            matched = next(
                (
                    context
                    for context in relevant
                    if context not in seen
                    and (
                        identity == context
                        or (min(len(identity), len(context)) >= 80 and identity in context)
                        or (min(len(identity), len(context)) >= 80 and context in identity)
                    )
                ),
                None,
            )
            relevance_values.append(matched is not None)
            if matched is not None:
                seen.add(matched)
            seen.add(identity)
        relevance = tuple(relevance_values)
        matching_method = "normalized_context_overlap"

    relevant_count = len(relevant)
    matched_count = len(matched_relevant) if evidence_ids is not None else sum(relevance)
    if relevant_count == 0:
        return RetrievalMetricScores(
            k=k,
            applicable=False,
            matching_method=matching_method,
            relevant_count=0,
            retrieved_count=len(selected),
            matched_count=0,
            hit_at_k=None,
            recall_at_k=None,
            mrr=None,
            ndcg_at_k=None,
        )

    first_relevant_rank = next(
        (rank for rank, is_relevant in enumerate(relevance, start=1) if is_relevant),
        None,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevance, start=1)
        if is_relevant
    )
    ideal_count = min(relevant_count, k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return RetrievalMetricScores(
        k=k,
        applicable=True,
        matching_method=matching_method,
        relevant_count=relevant_count,
        retrieved_count=len(selected),
        matched_count=matched_count,
        hit_at_k=float(first_relevant_rank is not None),
        recall_at_k=matched_count / relevant_count,
        mrr=(1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0),
        ndcg_at_k=dcg / ideal_dcg,
    )


def aggregate_retrieval_scores(
    scores: Sequence[RetrievalMetricScores],
) -> dict[str, object]:
    """Return macro means, excluding cases without relevant evidence."""

    if not scores:
        raise ValueError("retrieval metric aggregation requires at least one case")
    k_values = {score.k for score in scores}
    if len(k_values) != 1:
        raise ValueError("retrieval metric aggregation requires one shared k")
    applicable = tuple(score for score in scores if score.applicable)
    means = {
        metric: (
            sum(float(getattr(score, metric)) for score in applicable) / len(applicable)
            if applicable
            else None
        )
        for metric in ("hit_at_k", "recall_at_k", "mrr", "ndcg_at_k")
    }
    methods = sorted({score.matching_method for score in scores})
    return {
        "k": next(iter(k_values)),
        "total_cases": len(scores),
        "eligible_cases": len(applicable),
        "excluded_cases": len(scores) - len(applicable),
        "matching_methods": methods,
        "means": means,
    }


def normalize_context(value: str) -> str:
    """Normalize only casing and whitespace for deterministic legacy matching."""

    return " ".join(value.split()).casefold()


__all__ = [
    "RankedEvidence",
    "RetrievalMetricScores",
    "aggregate_retrieval_scores",
    "normalize_context",
    "score_retrieval",
]
