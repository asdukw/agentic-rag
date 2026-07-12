from __future__ import annotations

import pytest

from hybrid_rag.retrieval.service import RetrievalOptions


def test_naive_subroute_options_are_hashed_and_must_keep_one_scorer_enabled() -> None:
    baseline = RetrievalOptions()
    lexical_heavy = RetrievalOptions(
        naive_dense_weight=0.25,
        naive_bm25_weight=2.0,
        bm25_k1=1.6,
        bm25_b=0.4,
    )

    assert lexical_heavy.naive_subroute_weights == {"dense": 0.25, "bm25": 2.0}
    assert lexical_heavy.config_hash != baseline.config_hash

    with pytest.raises(ValueError, match="naive subroute weight"):
        RetrievalOptions(naive_dense_weight=0.0, naive_bm25_weight=0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bm25_k1": 0.0}, "k1"),
        ({"bm25_b": -0.01}, "b"),
        ({"bm25_b": 1.01}, "b"),
    ],
)
def test_naive_bm25_tuning_parameters_are_validated(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RetrievalOptions(**kwargs)
