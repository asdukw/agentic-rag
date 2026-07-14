from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from hybrid_rag.evaluation import (BlindCandidate, BlindComparison,
                                   BlindJudgment, BlindLabel, BlindWinner)
from hybrid_rag.evaluation.deepseek_judge import (DeepSeekBlindJudge,
                                                  DeepSeekJudgeError)
from hybrid_rag.extraction.client import CompletionResult


@dataclass
class FakeJudgeClient:
    content: str
    messages: list[object] = field(default_factory=list)
    closed: bool = False

    async def complete_messages(self, messages: object) -> CompletionResult:
        self.messages.append(messages)
        return CompletionResult(
            provider_request_id="judge-request",
            model="judge-model",
            system_fingerprint=None,
            content=self.content,
            finish_reason="stop",
            prompt_tokens=13,
            completion_tokens=7,
            cache_hit_tokens=5,
            cache_miss_tokens=8,
            raw_response={},
        )

    async def close(self) -> None:
        self.closed = True


def test_deepseek_blind_judge_hides_modes_validates_json_and_tracks_usage() -> None:
    clients: list[FakeJudgeClient] = []

    def factory() -> FakeJudgeClient:
        client = FakeJudgeClient('{"winner":"B","rationale":"B is more direct."}')
        clients.append(client)
        return client

    judge = DeepSeekBlindJudge(
        api_key="",
        model="deepseek-v4-pro",
        base_url="https://judge.example.test/v1",
        max_output_tokens=321,
        timeout_seconds=12.5,
        client_factory=factory,
    )
    result = judge.judge(_comparison())

    assert result == BlindJudgment(winner=BlindWinner.B, rationale="B is more direct.")
    assert judge.usage.calls == 1
    assert judge.usage.prompt_tokens == 13
    assert judge.usage.completion_tokens == 7
    assert judge.usage.records[0].model == "judge-model"
    assert judge.usage.records[0].cache_hit_tokens == 5
    assert judge.usage.records[0].cache_miss_tokens == 8
    assert judge.provenance.model == "deepseek-v4-pro"
    assert judge.provenance.base_url == "https://judge.example.test/v1"
    assert judge.provenance.response_format == "json_object"
    assert judge.provenance.thinking == "disabled"
    assert judge.provenance.temperature == 0.0
    assert judge.provenance.max_output_tokens == 321
    assert judge.provenance.timeout_seconds == 12.5
    assert clients[0].closed
    payload = clients[0].messages[0]
    system, user = payload
    assert "naive" not in user["content"].casefold()
    assert "hybrid" not in user["content"].casefold()
    assert "Candidate labels A and B are randomized" in system["content"]


def test_deepseek_blind_judge_rejects_invalid_provider_json() -> None:
    judge = DeepSeekBlindJudge(
        api_key="",
        client_factory=lambda: FakeJudgeClient('{"winner":"C","rationale":"invalid"}'),
    )

    with pytest.raises(DeepSeekJudgeError, match="violates BlindJudgment"):
        judge.judge(_comparison())


def _comparison() -> BlindComparison:
    return BlindComparison(
        case_id="case-1",
        question="Which answer is more useful?",
        candidates=(
            BlindCandidate(
                label=BlindLabel.A,
                evidence_hit_rate=1.0,
                cited_evidence_hit_rate=1.0,
                citation_grounded_faithfulness=True,
                abstained=False,
                answer="Answer A",
                citation_ids=("chk_a",),
            ),
            BlindCandidate(
                label=BlindLabel.B,
                evidence_hit_rate=1.0,
                cited_evidence_hit_rate=1.0,
                citation_grounded_faithfulness=True,
                abstained=False,
                answer="Answer B",
                citation_ids=("chk_b",),
            ),
        ),
    )
