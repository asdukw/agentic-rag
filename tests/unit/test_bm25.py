from __future__ import annotations

import pytest

from hybrid_rag.retrieval.bm25 import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    BM25Config,
    BM25Scorer,
    tokenize_lexical,
)
from hybrid_rag.storage.retrieval_repository import IndexItem


def _chunk(object_id: str, text: str) -> IndexItem:
    return IndexItem(
        object_id=object_id,
        kind="chunk",
        embedding_text=text,
        embedding=(),
        source_chunk_ids=(object_id,),
    )


def test_tokenize_lexical_casefolds_words_and_emits_cjk_unigrams_and_bigrams() -> None:
    assert tokenize_lexical("LightRAG uses 图谱检索 in 2025") == (
        "word:lightrag",
        "word:uses",
        "cjk:图",
        "cjk:谱",
        "cjk:检",
        "cjk:索",
        "cjk2:图谱",
        "cjk2:谱检",
        "cjk2:检索",
        "word:in",
        "word:2025",
    )


def test_bm25_ranks_shorter_exact_chunk_before_longer_match() -> None:
    scorer = BM25Scorer(
        (
            _chunk("chk-long", "LightRAG retrieval " + "background " * 30),
            _chunk("chk-short", "LightRAG retrieval"),
            _chunk("chk-other", "unrelated database indexing"),
        )
    )

    hits = scorer.score("lightrag retrieval")

    assert [hit.item.object_id for hit in hits] == ["chk-short", "chk-long"]
    assert hits[0].score > hits[1].score > 0.0


def test_bm25_uses_cjk_terms_and_deterministic_tie_breaking() -> None:
    scorer = BM25Scorer(
        (
            _chunk("chk-z", "知识图谱用于检索"),
            _chunk("chk-a", "知识图谱用于检索"),
            _chunk("chk-other", "数据库索引"),
        )
    )

    hits = scorer.score("图谱检索")

    assert [hit.item.object_id for hit in hits] == ["chk-a", "chk-z", "chk-other"]
    assert hits[-1].score > 0.0
    assert hits[0].score == pytest.approx(hits[1].score)


def test_bm25_limit_and_empty_query_avoid_zero_score_candidates() -> None:
    scorer = BM25Scorer((_chunk("chk-a", "graph retrieval"), _chunk("chk-b", "other text")))

    assert scorer.score("", limit=2) == ()
    assert scorer.score("missing", limit=2) == ()
    assert [hit.item.object_id for hit in scorer.score("graph", limit=1)] == ["chk-a"]
    assert scorer.score("graph", limit=0) == ()


@pytest.mark.parametrize(
    ("k1", "b"),
    [
        (0.0, DEFAULT_BM25_B),
        (-1.0, DEFAULT_BM25_B),
        (float("inf"), DEFAULT_BM25_B),
        (DEFAULT_BM25_K1, -0.1),
        (DEFAULT_BM25_K1, 1.1),
        (DEFAULT_BM25_K1, float("nan")),
    ],
)
def test_bm25_config_rejects_invalid_parameters(k1: float, b: float) -> None:
    with pytest.raises(ValueError):
        BM25Config(k1=k1, b=b)


def test_bm25_rejects_non_chunk_or_duplicate_items() -> None:
    entity = IndexItem(
        object_id="ent-one",
        kind="entity",
        embedding_text="LightRAG",
        embedding=(),
    )

    with pytest.raises(ValueError, match="only chunk"):
        BM25Scorer((entity,))
    with pytest.raises(ValueError, match="duplicate"):
        BM25Scorer((_chunk("chk-dup", "one"), _chunk("chk-dup", "two")))
