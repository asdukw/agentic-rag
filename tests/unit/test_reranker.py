from __future__ import annotations

import pytest

from hybrid_rag.retrieval.reranker import (
    FLAG_EMBEDDING_RERANKER_MODEL,
    FLAG_EMBEDDING_RERANKER_PROVIDER,
    FLAG_EMBEDDING_RERANKER_VERSION,
    FlagEmbeddingReranker,
    RerankCandidate,
    Reranker,
    create_reranker,
)


def _candidate(object_id: str, text: str, *, prior_score: float = 0.0) -> RerankCandidate:
    return RerankCandidate(object_id=object_id, text=text, prior_score=prior_score)


class RecordingFlagReranker:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[list[list[str]], bool]] = []

    def compute_score(self, pairs: list[list[str]], *, normalize: bool = False) -> object:
        self.calls.append((pairs, normalize))
        return self.response


def test_flag_embedding_reranker_batches_logits_and_preserves_normalized_trace() -> None:
    client = RecordingFlagReranker(response=[-4.0, 4.0])
    reranker = FlagEmbeddingReranker(
        FLAG_EMBEDDING_RERANKER_MODEL,
        use_fp16=True,
        client=client,
    )

    hits = reranker.rerank(
        "What is a panda?",
        (
            _candidate("chk-low", "A greeting.", prior_score=0.9),
            _candidate("chk-high", "The giant panda is a bear species.", prior_score=0.1),
        ),
    )

    assert isinstance(reranker, Reranker)
    assert reranker.provider == FLAG_EMBEDDING_RERANKER_PROVIDER
    assert reranker.version == FLAG_EMBEDDING_RERANKER_VERSION
    assert reranker.use_fp16 is True
    assert client.calls == [
        (
            [
                ["What is a panda?", "A greeting."],
                ["What is a panda?", "The giant panda is a bear species."],
            ],
            False,
        )
    ]
    assert [hit.candidate.object_id for hit in hits] == ["chk-high", "chk-low"]
    assert hits[0].components["cross_encoder"].raw_score == 4.0
    assert hits[0].components["cross_encoder"].normalized_score == pytest.approx(0.98201379)
    assert hits[0].score == hits[0].components["cross_encoder"].weighted_score
    assert hits[1].components["cross_encoder"].normalized_score == pytest.approx(0.01798621)


def test_flag_embedding_reranker_validates_scores_and_candidates() -> None:
    scalar = FlagEmbeddingReranker(client=RecordingFlagReranker(response=1.5))

    hits = scalar.rerank("query", (_candidate("chk-one", "passage"),), limit=1)

    assert len(hits) == 1
    assert hits[0].components["cross_encoder"].raw_score == 1.5

    wrong_count = FlagEmbeddingReranker(client=RecordingFlagReranker(response=[0.1]))
    with pytest.raises(RuntimeError, match="unexpected number"):
        wrong_count.rerank(
            "query",
            (_candidate("chk-one", "first"), _candidate("chk-two", "second")),
        )

    non_numeric = FlagEmbeddingReranker(client=RecordingFlagReranker(response=[object()]))
    with pytest.raises(RuntimeError, match="non-numeric"):
        non_numeric.rerank("query", (_candidate("chk-one", "passage"),))

    with pytest.raises(ValueError, match="duplicate"):
        scalar.rerank("query", (_candidate("chk-a", "one"), _candidate("chk-a", "two")))


def test_reranker_factory_supports_only_none_and_flag_embedding() -> None:
    assert create_reranker("none", "unused") is None
    configured = create_reranker("flagembedding", FLAG_EMBEDDING_RERANKER_MODEL, use_fp16=True)
    assert isinstance(configured, FlagEmbeddingReranker)
    assert configured.use_fp16 is True

    with pytest.raises(ValueError, match=r"none.*flagembedding"):
        create_reranker("lexical", "lexical-coverage-v1")
    with pytest.raises(ValueError, match="provider"):
        create_reranker("unsupported", "model")
