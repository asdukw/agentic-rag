from __future__ import annotations

import pytest

from hybrid_rag.retrieval.reranker import (
    FLAG_EMBEDDING_RERANKER_MODEL,
    FLAG_EMBEDDING_RERANKER_PROVIDER,
    FLAG_EMBEDDING_RERANKER_VERSION,
    LEXICAL_RERANKER_VERSION,
    FlagEmbeddingReranker,
    LexicalReranker,
    LexicalRerankerConfig,
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


def test_lexical_reranker_promotes_query_evidence_and_retains_score_breakdown() -> None:
    reranker = LexicalReranker()

    hits = reranker.rerank(
        "How does LightRAG use entities for retrieval?",
        (
            _candidate("chk-prior", "Graph systems optimize data pipelines.", prior_score=0.95),
            _candidate(
                "chk-evidence",
                "LightRAG uses entities and relations to support retrieval.",
                prior_score=0.2,
            ),
            _candidate("chk-other", "The database uses unrelated indexes.", prior_score=0.9),
        ),
    )

    assert isinstance(reranker, Reranker)
    assert hits[0].candidate.object_id == "chk-evidence"
    assert hits[0].components["bm25"].raw_score > 0.0
    assert hits[0].components["coverage"].normalized_score > 0.0
    assert hits[0].components["proximity"].normalized_score > 0.0
    assert hits[0].score == pytest.approx(
        sum(component.weighted_score for component in hits[0].components.values())
    )
    assert LEXICAL_RERANKER_VERSION == "lexical-reranker-v1"


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


def test_flag_embedding_reranker_accepts_one_scalar_score_and_rejects_wrong_count() -> None:
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


def test_reranker_factory_selects_the_configured_adapter() -> None:
    assert create_reranker("none", "unused") is None
    assert isinstance(create_reranker("lexical", "lexical-coverage-v1"), LexicalReranker)
    configured = create_reranker("flagembedding", FLAG_EMBEDDING_RERANKER_MODEL, use_fp16=True)
    assert isinstance(configured, FlagEmbeddingReranker)
    assert configured.use_fp16 is True

    with pytest.raises(ValueError, match="requires model"):
        create_reranker("lexical", FLAG_EMBEDDING_RERANKER_MODEL)
    with pytest.raises(ValueError, match="RERANKER_MODEL"):
        create_reranker("flagembedding", "lexical-coverage-v1")
    with pytest.raises(ValueError, match="provider"):
        create_reranker("unsupported", "model")


def test_lexical_reranker_uses_ordered_proximity_when_other_signals_are_disabled() -> None:
    reranker = LexicalReranker(
        config=LexicalRerankerConfig(
            bm25_weight=0.0,
            coverage_weight=0.0,
            proximity_weight=1.0,
            prior_weight=0.0,
        )
    )

    hits = reranker.rerank(
        "graph retrieval",
        (
            _candidate("chk-reverse", "retrieval graph"),
            _candidate("chk-spread", "graph uses many intermediate concepts before retrieval"),
            _candidate("chk-adjacent", "graph retrieval uses chunks"),
        ),
    )

    assert [hit.candidate.object_id for hit in hits] == [
        "chk-adjacent",
        "chk-spread",
        "chk-reverse",
    ]
    assert hits[0].components["proximity"].normalized_score == 1.0
    assert hits[1].components["proximity"].normalized_score < 1.0
    assert hits[2].components["proximity"].normalized_score == 0.0


def test_lexical_reranker_supports_cjk_and_stable_id_tie_breaking() -> None:
    reranker = LexicalReranker(
        config=LexicalRerankerConfig(
            bm25_weight=0.0,
            coverage_weight=1.0,
            proximity_weight=0.0,
            prior_weight=0.0,
        )
    )

    hits = reranker.rerank(
        "知识图谱检索",
        (
            _candidate("chk-z", "知识图谱检索"),
            _candidate("chk-a", "知识图谱检索"),
            _candidate("chk-other", "天气预报"),
        ),
    )

    assert [hit.candidate.object_id for hit in hits] == ["chk-a", "chk-z", "chk-other"]
    assert hits[0].components["coverage"].normalized_score == pytest.approx(1.0)
    assert hits[-1].score == 0.0


def test_lexical_no_match_falls_back_to_prior_score_and_limit_is_deterministic() -> None:
    reranker = LexicalReranker()
    candidates = (
        _candidate("chk-low", "alpha beta", prior_score=0.1),
        _candidate("chk-high", "gamma delta", prior_score=0.9),
    )

    hits = reranker.rerank("unseen query term", candidates, limit=1)

    assert [hit.candidate.object_id for hit in hits] == ["chk-high"]
    assert hits[0].components["bm25"].normalized_score == 0.0
    assert hits[0].components["coverage"].normalized_score == 0.0
    assert hits[0].components["proximity"].normalized_score == 0.0
    assert hits[0].components["prior"].normalized_score == 1.0
    assert reranker.rerank("unseen", candidates, limit=0) == ()


@pytest.mark.parametrize(
    "config",
    [
        {"bm25_k1": 0.0},
        {"bm25_b": 1.1},
        {
            "bm25_weight": 0.0,
            "coverage_weight": 0.0,
            "proximity_weight": 0.0,
            "prior_weight": 0.0,
        },
    ],
)
def test_lexical_reranker_rejects_invalid_contracts(config: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        LexicalRerankerConfig(**config)

    reranker = LexicalReranker()
    with pytest.raises(ValueError, match="duplicate"):
        reranker.rerank("query", (_candidate("chk-a", "one"), _candidate("chk-a", "two")))
    with pytest.raises(ValueError, match="limit"):
        reranker.rerank("query", (), limit=-1)
