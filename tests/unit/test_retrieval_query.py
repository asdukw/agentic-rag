from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hybrid_rag.retrieval import (DeepSeekQueryClient,
                                  DeterministicQueryClient, EvidenceItem,
                                  QueryConfigurationError,
                                  QueryValidationError,
                                  QueryValidationFailureKind,
                                  validate_answer_completion,
                                  validate_keyword_completion)


class FakeCompletions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.requests: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(
            id="req_query",
            model=str(kwargs["model"]),
            system_fingerprint="fp_query",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=content),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                prompt_cache_hit_tokens=3,
                prompt_cache_miss_tokens=8,
            ),
        )


class FakeSdk:
    def __init__(self, contents: list[str]) -> None:
        self.completions = FakeCompletions(contents)
        self.chat = SimpleNamespace(completions=self.completions)


def test_deepseek_query_client_uses_json_mode_disabled_thinking_and_allowlisted_evidence() -> None:
    sdk = FakeSdk(
        [
            '{"keywords":["LightRAG","graph"]}',
            '{"answer":"LightRAG uses a graph.","citations":["chk_graph"],'
            '"insufficient_evidence":false}',
        ]
    )
    client = DeepSeekQueryClient(
        api_key=None,
        model="keyword-model",
        answer_model="answer-model",
        max_output_tokens=321,
        sdk_client=sdk,
    )

    keywords = asyncio.run(client.extract_keywords("How does LightRAG use a graph?"))
    answer = asyncio.run(
        client.answer(
            "How does LightRAG use a graph?",
            [EvidenceItem(citation_id="chk_graph", text="LightRAG uses a graph.")],
        )
    )

    assert keywords.keywords == ("LightRAG", "graph")
    assert answer.citations == ("chk_graph",)
    assert len(sdk.completions.requests) == 2
    for request in sdk.completions.requests:
        assert request["response_format"] == {"type": "json_object"}
        assert request["extra_body"] == {"thinking": {"type": "disabled"}}
        assert request["max_tokens"] == 321
        assert request["temperature"] == 0.0
    assert sdk.completions.requests[0]["model"] == "keyword-model"
    assert sdk.completions.requests[1]["model"] == "answer-model"
    keyword_system = sdk.completions.requests[0]["messages"][0]["content"]
    answer_system = sdk.completions.requests[1]["messages"][0]["content"]
    assert "not answer the question" in keyword_system
    assert "ALLOWED_CITATION_IDS_JSON" in sdk.completions.requests[1]["messages"][1]["content"]
    assert "Never invent citations" in answer_system
    assert [(item.operation, item.model) for item in client.usage] == [
        ("answer", "answer-model"),
        ("keyword", "keyword-model"),
    ]
    assert all(item.cache_breakdown_complete for item in client.usage)


def test_answer_rejects_citation_outside_exact_retrieval_allowlist() -> None:
    sdk = FakeSdk(
        ['{"answer":"Unsupported.","citations":["CHK_GRAPH"],"insufficient_evidence":false}']
    )
    client = DeepSeekQueryClient(api_key=None, sdk_client=sdk)

    with pytest.raises(QueryValidationError) as captured:
        asyncio.run(
            client.answer(
                "Question",
                [EvidenceItem(citation_id="chk_graph", text="Supported evidence.")],
            )
        )

    assert captured.value.kind is QueryValidationFailureKind.CITATION_NOT_ALLOWED
    assert captured.value.issues[0].path == "citations.0"


def test_completion_validation_rejects_duplicate_keys_and_unknown_output_fields() -> None:
    with pytest.raises(QueryValidationError) as duplicate:
        validate_keyword_completion(
            content='{"keywords":["graph"],"keywords":["rag"]}',
            finish_reason="stop",
        )
    assert duplicate.value.kind is QueryValidationFailureKind.INVALID_JSON

    with pytest.raises(QueryValidationError) as invalid_schema:
        validate_answer_completion(
            content=(
                '{"answer":"Answer","citations":["chk_1"],'
                '"insufficient_evidence":false,"confidence":1}'
            ),
            finish_reason="stop",
            allowed_citation_ids=("chk_1",),
        )
    assert invalid_schema.value.kind is QueryValidationFailureKind.SCHEMA_INVALID


def test_offline_fallback_is_deterministic_and_never_invents_citations() -> None:
    client = DeterministicQueryClient()
    evidence = [
        EvidenceItem(
            citation_id="chk_1",
            text="Graph retrieval expands relevant entities.",
        )
    ]

    first = asyncio.run(client.extract_keywords("How does graph retrieval expand entities?"))
    second = asyncio.run(client.extract_keywords("How does graph retrieval expand entities?"))
    answer = asyncio.run(client.answer("Question", evidence))
    no_evidence = asyncio.run(client.answer("Question", []))

    assert first == second
    assert "graph" in {keyword.casefold() for keyword in first.keywords}
    assert answer.answer == evidence[0].text
    assert answer.citations == ("chk_1",)
    assert not answer.insufficient_evidence
    assert no_evidence.citations == ()
    assert no_evidence.insufficient_evidence


def test_missing_online_credentials_fail_only_when_a_remote_call_is_needed() -> None:
    client = DeepSeekQueryClient(api_key=None)

    assert asyncio.run(client.answer("Question", [])).insufficient_evidence
    with pytest.raises(QueryConfigurationError):
        asyncio.run(client.extract_keywords("Question"))
