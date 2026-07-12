"""Deterministic score fusion and token-budget selection helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hybrid_rag.retrieval.embedding import min_max_normalize


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
