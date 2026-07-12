from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from hybrid_rag.retrieval.embedding import (
    EmbeddingConfigurationError,
    HashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    cosine_similarity,
    min_max_normalize,
)
from hybrid_rag.retrieval.fusion import rank_ids, select_token_budget, weighted_fusion


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


def test_openai_compatible_embedding_provider_uses_sdk_and_orders_response_vectors() -> None:
    sdk = FakeEmbeddingSdk(
        [
            SimpleNamespace(index=1, embedding=[0.3, 0.4]),
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
        ]
    )
    provider = OpenAICompatibleEmbeddingProvider(
        api_key=None,
        base_url="https://embeddings.example.test/v1",
        model="embedding-test-model",
        dimensions=2,
        sdk_client=sdk,
    )

    vectors = provider.embed(("first", "second"))

    assert vectors == ((0.1, 0.2), (0.3, 0.4))
    assert sdk.embeddings.requests == [
        {"model": "embedding-test-model", "input": ["first", "second"]}
    ]


def test_openai_compatible_embedding_provider_rejects_wrong_response_dimension() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        api_key=None,
        base_url="https://embeddings.example.test/v1",
        model="embedding-test-model",
        dimensions=2,
        sdk_client=FakeEmbeddingSdk([SimpleNamespace(index=0, embedding=[0.1, 0.2, 0.3])]),
    )

    with pytest.raises(RuntimeError, match=r"dimensions differ from configured 2: \[3\]"):
        provider.embed(("only",))


def test_openai_compatible_embedding_provider_requires_credentials_without_fake_sdk() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        api_key=None,
        base_url="https://embeddings.example.test/v1",
        model="embedding-test-model",
        dimensions=2,
    )

    with pytest.raises(EmbeddingConfigurationError, match="API key is required"):
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


class FakeEmbeddingSdk:
    def __init__(self, data: list[object]) -> None:
        self.embeddings = FakeEmbeddings(data)


class FakeEmbeddings:
    def __init__(self, data: list[object]) -> None:
        self.data = data
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return SimpleNamespace(data=self.data)
