from __future__ import annotations

import math

import pytest

from hybrid_rag.retrieval.embedding import (
    BGEM3EmbeddingProvider,
    HashEmbeddingProvider,
    cosine_similarity,
    min_max_normalize,
)
from hybrid_rag.retrieval.fusion import (
    rank_ids,
    select_token_budget,
    weighted_average_fusion,
    weighted_fusion,
)


def test_hash_embeddings_are_deterministic_normalized_and_unicode_aware() -> None:
    provider = HashEmbeddingProvider(dimensions=128)

    first, repeated, chinese, empty = provider.embed(
        (
            "Atlas connects Beacon through graph retrieval.",
            "Atlas connects Beacon through graph retrieval.",
            "图检索连接知识实体。",
            "",
        )
    )

    assert first == repeated
    assert len(first) == 128
    assert math.isclose(sum(value * value for value in first), 1.0)
    assert any(value != 0.0 for value in chinese)
    assert empty == (0.0,) * 128


def test_cosine_similarity_is_safe_and_rejects_mismatched_dimensions() -> None:
    assert cosine_similarity((1.0, 0.0), (3.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0
    assert cosine_similarity((), ()) == 0.0

    with pytest.raises(ValueError, match="embedding dimensions differ"):
        cosine_similarity((1.0,), (1.0, 0.0))


def test_bge_m3_embedding_provider_encodes_dense_vectors_with_explicit_options() -> None:
    client = FakeBGEM3Client({"dense_vecs": [[0.1, 0.2], [0.3, 0.4]]})
    provider = BGEM3EmbeddingProvider(
        model="BAAI/bge-m3",
        dimensions=2,
        batch_size=12,
        max_length=8192,
        use_fp16=False,
        client=client,
    )

    vectors = provider.embed(("first", "second"))

    assert vectors == ((0.1, 0.2), (0.3, 0.4))
    assert client.requests == [
        (
            ["first", "second"],
            {
                "batch_size": 12,
                "max_length": 8192,
                "return_dense": True,
                "return_sparse": False,
                "return_colbert_vecs": False,
            },
        )
    ]
    assert provider.semantic_options == {
        "max_length": 8192,
        "normalize_embeddings": True,
        "use_fp16": False,
    }


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"dense_vecs": [[0.1, 0.2, 0.3]]}, "dimensions differ from configured 2"),
        ({"dense_vecs": [[float("nan"), 0.2]]}, "non-finite dense vector"),
        ({}, "did not return dense_vecs"),
    ],
)
def test_bge_m3_embedding_provider_rejects_invalid_dense_responses(
    response: object,
    message: str,
) -> None:
    provider = BGEM3EmbeddingProvider(
        dimensions=2,
        client=FakeBGEM3Client(response),
    )

    with pytest.raises(RuntimeError, match=message):
        provider.embed(("only",))


def test_score_normalization_fusion_ranking_and_budget_are_deterministic() -> None:
    assert min_max_normalize({"only": 0.4}) == {"only": 1.0}
    fused, components = weighted_fusion(
        {
            "naive": {"chunk_a": 0.1, "chunk_b": 0.9},
            "local": {"chunk_a": 3.0, "chunk_c": 5.0},
            "global": {"chunk_b": 1.0, "chunk_c": 1.0},
        },
        {"naive": 2.0, "local": 1.0, "global": 0.5},
    )

    assert fused == {
        "chunk_a": 0.0,
        "chunk_b": 2.5,
        "chunk_c": 1.5,
    }
    assert components["chunk_b"] == {"global": 0.5, "naive": 2.0}
    assert rank_ids(fused, limit=3) == ("chunk_b", "chunk_c", "chunk_a")
    assert select_token_budget(
        ("chunk_b", "chunk_c", "chunk_a"),
        {"chunk_b": 4, "chunk_c": 5, "chunk_a": 2},
        budget=6,
    ) == ("chunk_b", "chunk_a")


def test_weighted_average_fusion_retains_subscore_breakdown_and_skips_empty_maps() -> None:
    fused, components = weighted_average_fusion(
        {
            "dense": {"chunk_a": 0.1, "chunk_b": 0.9},
            "bm25": {"chunk_b": 2.0, "chunk_c": 4.0},
        },
        {"dense": 0.25, "bm25": 0.75},
    )

    assert fused == {"chunk_a": 0.0, "chunk_b": 0.25, "chunk_c": 0.75}
    assert components["chunk_b"]["dense"].raw_score == 0.9
    assert components["chunk_b"]["dense"].normalized_score == 1.0
    assert components["chunk_b"]["dense"].weighted_score == 0.25
    assert components["chunk_c"]["bm25"].weighted_score == 0.75

    fallback, fallback_components = weighted_average_fusion(
        {"dense": {"chunk_a": 0.3}, "bm25": {}},
        {"dense": 1.0, "bm25": 1.0},
    )
    assert fallback == {"chunk_a": 1.0}
    assert fallback_components["chunk_a"]["dense"].weighted_score == 1.0


class FakeBGEM3Client:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts: list[str], **kwargs: object) -> object:
        self.requests.append((texts, kwargs))
        return self.response
