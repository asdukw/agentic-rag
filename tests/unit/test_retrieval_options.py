from __future__ import annotations

import pytest

from hybrid_rag.retrieval.service import RetrievalOptions


def test_hybrid_subroute_options_are_hashed_and_must_keep_one_scorer_enabled() -> None:
    baseline = RetrievalOptions()
    lexical_heavy = RetrievalOptions(
        hybrid_dense_weight=0.25,
        hybrid_bm25_weight=2.0,
        bm25_k1=1.6,
        bm25_b=0.4,
    )

    assert lexical_heavy.hybrid_subroute_weights == {"dense": 0.25, "bm25": 2.0}
    assert lexical_heavy.config_hash != baseline.config_hash

    with pytest.raises(ValueError, match="hybrid-search subroute weight"):
        RetrievalOptions(hybrid_dense_weight=0.0, hybrid_bm25_weight=0.0)


def test_rerank_options_are_hashed_and_can_be_explicitly_enabled() -> None:
    baseline = RetrievalOptions()
    disabled = RetrievalOptions(reranker_provider="none", rerank_candidate_multiplier=2)
    cross_encoder = RetrievalOptions(
        reranker_provider="flagembedding",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_use_fp16=True,
    )

    assert not baseline.rerank_enabled
    assert baseline.rerank_candidate_limit == baseline.top_k * baseline.rerank_candidate_multiplier
    assert not disabled.rerank_enabled
    assert disabled.config_hash != baseline.config_hash
    assert cross_encoder.rerank_enabled
    assert cross_encoder.reranker_use_fp16 is True
    assert cross_encoder.config_hash != baseline.config_hash

    with pytest.raises(ValueError, match="rerank_candidate_multiplier"):
        RetrievalOptions(rerank_candidate_multiplier=0)
    with pytest.raises(TypeError, match="reranker_use_fp16"):
        RetrievalOptions(reranker_use_fp16="true")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bm25_k1": 0.0}, "k1"),
        ({"bm25_b": -0.01}, "b"),
        ({"bm25_b": 1.01}, "b"),
    ],
)
def test_bm25_tuning_parameters_are_validated(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RetrievalOptions(**kwargs)
