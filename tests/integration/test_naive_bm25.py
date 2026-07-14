from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from hybrid_rag.config import sqlite_url
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.service import IngestionService
from hybrid_rag.retrieval.models import CandidateHit, RetrievalMode
from hybrid_rag.retrieval.reranker import (RerankCandidate, RerankHit,
                                           RerankScoreComponent)
from hybrid_rag.retrieval.service import RetrievalOptions, RetrievalService
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database

_DENSE_MARKER = "DENSE_ONLY_MARKER"
_LEXICAL_MARKER = "LEXICAL_ONLY_MARKER"
_RARE_TERM = "quasarneedle"


class WordCounter:
    name = "test-words"

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


class ScriptedDenseEmbeddingProvider:
    """Make dense and lexical recall disagree so their fusion is observable."""

    provider = "scripted"
    model = "scripted-dense-v1"
    dimensions = 2

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed(text) for text in texts)

    @staticmethod
    def _embed(text: str) -> tuple[float, ...]:
        if _DENSE_MARKER in text:
            return (1.0, 0.0)
        if _LEXICAL_MARKER in text:
            return (0.0, 1.0)
        # The query deliberately favors the non-lexical document in dense space.
        return (1.0, 0.0)


class ScriptedReranker:
    """Make the second-stage rerank decision observable without model calls."""

    provider = "scripted"
    model = "scripted-reranker-v1"
    version = "scripted-v1"

    def __init__(self) -> None:
        self.calls = 0

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        limit: int | None = None,
    ) -> tuple[RerankHit, ...]:
        assert query == _RARE_TERM
        self.calls += 1
        hits = tuple(
            RerankHit(
                candidate=candidate,
                score=1.0 if _LEXICAL_MARKER in candidate.text else 0.0,
                components={
                    "scripted": RerankScoreComponent(
                        raw_score=1.0 if _LEXICAL_MARKER in candidate.text else 0.0,
                        normalized_score=1.0 if _LEXICAL_MARKER in candidate.text else 0.0,
                        weight=1.0,
                        weighted_score=1.0 if _LEXICAL_MARKER in candidate.text else 0.0,
                    )
                },
            )
            for candidate in candidates
        )
        return hits if limit is None else hits[:limit]


def test_naive_route_fuses_dense_and_bm25_and_persists_score_breakdown(
    tmp_path: Path,
) -> None:
    database = _seed_two_chunk_database(tmp_path)
    retrieval = RetrievalService(database, ScriptedDenseEmbeddingProvider(), WordCounter())
    # Keep the lexical query to the rare token so generated embedding-text
    # labels such as "Document" cannot create a BM25 match for both chunks.
    question = _RARE_TERM

    try:
        retrieval.build_index()
        dense_only = retrieval.retrieve(
            question,
            mode=RetrievalMode.NAIVE,
            options=RetrievalOptions(
                top_k=1,
                candidate_multiplier=2,
                context_token_budget=64,
                naive_dense_weight=1.0,
                naive_bm25_weight=0.0,
                reranker_provider="none",
            ),
            persist=False,
        )
        bm25_only = retrieval.retrieve(
            question,
            mode=RetrievalMode.NAIVE,
            options=RetrievalOptions(
                top_k=1,
                candidate_multiplier=2,
                context_token_budget=64,
                naive_dense_weight=0.0,
                naive_bm25_weight=1.0,
                reranker_provider="none",
            ),
            persist=False,
        )
        combined = retrieval.retrieve(
            question,
            mode=RetrievalMode.NAIVE,
            options=RetrievalOptions(
                top_k=2,
                candidate_multiplier=2,
                context_token_budget=64,
                naive_dense_weight=1.0,
                naive_bm25_weight=1.0,
                bm25_k1=1.4,
                bm25_b=0.5,
                reranker_provider="none",
            ),
            persist=False,
        )

        assert _DENSE_MARKER in dense_only.hits[0].metadata["text"]
        assert _LEXICAL_MARKER in bm25_only.hits[0].metadata["text"]
        assert combined.trace.settings["naive_dense_weight"] == 1.0
        assert combined.trace.settings["naive_bm25_weight"] == 1.0
        assert combined.trace.settings["bm25_k1"] == 1.4
        assert combined.trace.settings["bm25_b"] == 0.5
        assert combined.trace.rerank is None

        route_hits = combined.trace.routes[RetrievalMode.NAIVE.value].hits
        dense_hit = _hit_with_marker(route_hits, _DENSE_MARKER)
        lexical_hit = _hit_with_marker(route_hits, _LEXICAL_MARKER)

        assert set(dense_hit.score_components) == {"dense"}
        assert set(lexical_hit.score_components) == {"dense", "bm25"}
        assert dense_hit.score_components["dense"].raw_score == 1.0
        assert lexical_hit.score_components["dense"].raw_score == 0.0
        assert lexical_hit.score_components["bm25"].raw_score > 0.0
        assert lexical_hit.score_components["bm25"].normalized_score == 1.0
        assert lexical_hit.score_components["bm25"].weighted_score == 0.5
    finally:
        database.dispose()


def test_reranker_reorders_fused_candidates_and_trace_replays_without_a_new_call(
    tmp_path: Path,
) -> None:
    database = _seed_two_chunk_database(tmp_path)
    reranker = ScriptedReranker()
    retrieval = RetrievalService(
        database,
        ScriptedDenseEmbeddingProvider(),
        WordCounter(),
        reranker=reranker,
    )

    try:
        retrieval.build_index()
        result = retrieval.retrieve(
            _RARE_TERM,
            mode=RetrievalMode.NAIVE,
            options=RetrievalOptions(
                top_k=2,
                candidate_multiplier=2,
                context_token_budget=64,
                naive_dense_weight=1.0,
                naive_bm25_weight=0.0,
                reranker_provider="scripted",
                reranker_model="scripted-reranker-v1",
            ),
        )

        assert _LEXICAL_MARKER in result.hits[0].metadata["text"]
        assert result.hits[0].rerank_score == 1.0
        assert result.hits[0].retrieval_score == 0.0
        assert result.trace.rerank is not None
        assert result.trace.rerank.provider == "scripted"
        assert result.trace.rerank.hits[0].pre_rerank_rank == 2
        assert result.trace.rerank.hits[0].final_rank == 1
        assert result.trace.rerank.hits[0].components["scripted"].weighted_score == 1.0
        assert result.trace_id is not None
        calls_before_replay = reranker.calls
        assert retrieval.replay(result.trace_id) == result
        assert reranker.calls == calls_before_replay
    finally:
        database.dispose()


def _hit_with_marker(hits: tuple[CandidateHit, ...], marker: str) -> CandidateHit:
    return next(hit for hit in hits if marker in str(hit.metadata["text"]))


def _seed_two_chunk_database(tmp_path: Path) -> Database:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "dense.txt").write_text(
        f"{_DENSE_MARKER} is a general retrieval discussion without the target term.",
        encoding="utf-8",
    )
    (corpus / "lexical.txt").write_text(
        f"{_LEXICAL_MARKER} contains the exact rare keyword {_RARE_TERM} as evidence.",
        encoding="utf-8",
    )
    database_url = sqlite_url(tmp_path / "naive-bm25.db")
    upgrade_database(database_url)
    database = Database(database_url)
    report = IngestionService(
        database,
        SectionTokenChunker(WordCounter(), max_tokens=64, overlap_tokens=0),
    ).ingest(corpus)
    assert report.failed == 0
    assert report.chunks_written == 2
    return database
