from __future__ import annotations

from hybrid_rag.evaluation import (BlindCandidate, BlindComparison, BlindLabel,
                                   BlindWinner, DeterministicBlindJudge)


def test_deterministic_blind_judge_returns_tie_without_using_latency() -> None:
    comparison = BlindComparison(
        case_id="case-tie",
        question="Which candidate is better?",
        candidates=(
            BlindCandidate(
                label=BlindLabel.A,
                evidence_hit_rate=1.0,
                cited_evidence_hit_rate=0.5,
                citation_grounded_faithfulness=True,
                abstained=False,
                answer="same evidence",
                citation_ids=("chk_a",),
            ),
            BlindCandidate(
                label=BlindLabel.B,
                evidence_hit_rate=1.0,
                cited_evidence_hit_rate=0.5,
                citation_grounded_faithfulness=True,
                abstained=False,
                answer="same evidence",
                citation_ids=("chk_b",),
            ),
        ),
    )

    judgment = DeterministicBlindJudge().judge(comparison)

    assert judgment.winner is BlindWinner.TIE
    assert "latency does not break ties" in judgment.rationale


def test_deterministic_blind_judge_prefers_evidence_coverage_before_other_signals() -> None:
    comparison = BlindComparison(
        case_id="case-coverage",
        question="Which candidate is better?",
        candidates=(
            BlindCandidate(
                label=BlindLabel.A,
                evidence_hit_rate=0.5,
                cited_evidence_hit_rate=1.0,
                citation_grounded_faithfulness=True,
                abstained=False,
                answer="first evidence",
                citation_ids=("chk_a",),
            ),
            BlindCandidate(
                label=BlindLabel.B,
                evidence_hit_rate=1.0,
                cited_evidence_hit_rate=0.0,
                citation_grounded_faithfulness=False,
                abstained=False,
                answer="second evidence",
                citation_ids=("chk_b",),
            ),
        ),
    )

    judgment = DeterministicBlindJudge().judge(comparison)

    assert judgment.winner is BlindWinner.B
    assert judgment.rationale.startswith("evidence_hit_rate")
