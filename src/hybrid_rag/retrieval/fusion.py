"""Deterministic score fusion and token-budget selection helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hybrid_rag.retrieval.embedding import min_max_normalize
from hybrid_rag.retrieval.models import ScoreComponent


def weighted_fusion(
    route_scores: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Normalize each route, then combine scores by explicit positive weights."""

    total: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for route in sorted(route_scores):
        weight = weights.get(route, 0.0)
        if weight <= 0:
            continue
        normalized = min_max_normalize(dict(route_scores[route]))
        for object_id, score in normalized.items():
            component = score * weight
            total[object_id] = total.get(object_id, 0.0) + component
            components.setdefault(object_id, {})[route] = component
    return total, components


def weighted_average_fusion(
    score_maps: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, dict[str, ScoreComponent]]]:
    """Fuse active score maps as a normalized weighted average.

    This is intentionally separate from :func:`weighted_fusion`: outer hybrid
    route weights remain additive, while sub-scorers inside one route should
    retain a stable 0--1 scale.  Empty or zero-weight scorers are excluded from
    the denominator, so an out-of-vocabulary lexical query cannot dilute dense
    recall.  Each raw, normalized, and weighted contribution is returned for
    trace inspection.
    """

    active = tuple(
        (name, scores, float(weights.get(name, 0.0)))
        for name, scores in sorted(score_maps.items())
        if scores and weights.get(name, 0.0) > 0.0
    )
    total_weight = sum(weight for _, _, weight in active)
    if total_weight <= 0.0:
        return {}, {}

    fused: dict[str, float] = {}
    components: dict[str, dict[str, ScoreComponent]] = {}
    for name, scores, weight in active:
        normalized = min_max_normalize(dict(scores))
        share = weight / total_weight
        for object_id, raw_score in scores.items():
            normalized_score = normalized[object_id]
            weighted_score = normalized_score * share
            fused[object_id] = fused.get(object_id, 0.0) + weighted_score
            components.setdefault(object_id, {})[name] = ScoreComponent(
                raw_score=float(raw_score),
                normalized_score=float(normalized_score),
                weighted_score=float(weighted_score),
            )
    return fused, components


def rank_ids(scores: Mapping[str, float], *, limit: int) -> tuple[str, ...]:
    if limit < 0:
        raise ValueError("limit must not be negative")
    return tuple(sorted(scores, key=lambda value: (-scores[value], value))[:limit])


def select_token_budget(
    ordered_ids: Sequence[str],
    token_counts: Mapping[str, int],
    *,
    budget: int,
) -> tuple[str, ...]:
    """Keep whole passages in score order; never silently exceed the token budget."""

    if budget < 1:
        raise ValueError("budget must be positive")
    selected: list[str] = []
    consumed = 0
    for object_id in ordered_ids:
        tokens = token_counts[object_id]
        if tokens < 0:
            raise ValueError("token counts must be non-negative")
        if consumed + tokens > budget:
            continue
        selected.append(object_id)
        consumed += tokens
    return tuple(selected)
