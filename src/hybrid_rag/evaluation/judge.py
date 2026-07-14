"""Blind pairwise judging with a deterministic offline fallback."""

from __future__ import annotations

import hashlib
from typing import Protocol

from hybrid_rag.evaluation.contracts import (BlindCandidate, BlindComparison,
                                             BlindJudgment, BlindLabel,
                                             BlindWinner, RetrievalEvaluation)
from hybrid_rag.retrieval.models import RetrievalMode


class BlindJudge(Protocol):
    """A judge only sees labels A/B, never retrieval-mode names."""

    protocol: str

    def judge(self, comparison: BlindComparison) -> BlindJudgment: ...


class DeterministicBlindJudge:
    """Metric-only fallback that is reproducible and intentionally conservative.

    It compares evidence coverage first, then evidence coverage specifically
    present in cited chunks, then citation-grounded faithfulness.  It never uses
    latency to decide quality and returns a tie when those observable quality
    signals are equal.
    """

    protocol = "deterministic-blind-metrics-v1"

    def judge(self, comparison: BlindComparison) -> BlindJudgment:
        left, right = comparison.candidates
        criteria = (
            ("evidence_hit_rate", left.evidence_hit_rate, right.evidence_hit_rate),
            (
                "cited_evidence_hit_rate",
                left.cited_evidence_hit_rate,
                right.cited_evidence_hit_rate,
            ),
            (
                "citation_grounded_faithfulness",
                float(left.citation_grounded_faithfulness),
                float(right.citation_grounded_faithfulness),
            ),
        )
        for criterion, left_value, right_value in criteria:
            if left_value > right_value:
                return BlindJudgment(
                    winner=BlindWinner(left.label.value),
                    rationale=f"{criterion}: {left.label.value} has the stronger observable score",
                )
            if right_value > left_value:
                return BlindJudgment(
                    winner=BlindWinner(right.label.value),
                    rationale=f"{criterion}: {right.label.value} has the stronger observable score",
                )
        return BlindJudgment(
            winner=BlindWinner.TIE,
            rationale="all blind observable quality metrics are equal; latency does not break ties",
        )


def blind_comparison(
    *,
    benchmark_id: str,
    case_id: str,
    question: str,
    naive: RetrievalEvaluation,
    hybrid: RetrievalEvaluation,
) -> tuple[BlindComparison, dict[BlindLabel, RetrievalMode]]:
    """Create a stable A/B assignment that does not privilege either mode."""

    if naive.mode is not RetrievalMode.NAIVE or hybrid.mode is not RetrievalMode.HYBRID:
        raise ValueError("blind comparison requires exactly naive and hybrid evaluations")
    digest = hashlib.sha256(f"{benchmark_id}:{case_id}".encode()).digest()
    ordered = (naive, hybrid) if digest[0] & 1 else (hybrid, naive)
    labels = (BlindLabel.A, BlindLabel.B)
    candidates = tuple(
        _candidate(label, evaluation) for label, evaluation in zip(labels, ordered, strict=True)
    )
    mapping = {label: evaluation.mode for label, evaluation in zip(labels, ordered, strict=True)}
    return BlindComparison(case_id=case_id, question=question, candidates=candidates), mapping


def _candidate(label: BlindLabel, evaluation: RetrievalEvaluation) -> BlindCandidate:
    return BlindCandidate(
        label=label,
        evidence_hit_rate=evaluation.evidence_hit_rate,
        cited_evidence_hit_rate=evaluation.cited_evidence_hit_rate,
        citation_grounded_faithfulness=evaluation.citation_grounded_faithfulness,
        abstained=evaluation.abstained,
        answer=evaluation.answer.answer,
        citation_ids=evaluation.answer.citations,
    )
