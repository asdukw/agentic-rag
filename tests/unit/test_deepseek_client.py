from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hybrid_rag.extraction.client import (DeepSeekClient,
                                          RetryableProviderError,
                                          TerminalProviderError,
                                          classify_provider_error)


class FakeCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            id="req_test",
            model="deepseek-v4-flash",
            system_fingerprint="fp_test",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"entities":[],"relations":[]}'),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                prompt_cache_hit_tokens=4,
                prompt_cache_miss_tokens=6,
            ),
        )


class FakeSdk:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_initial_and_repair_requests_force_json_and_disable_thinking() -> None:
    sdk = FakeSdk()
    client = DeepSeekClient(api_key="", sdk_client=sdk, max_output_tokens=1234)

    initial = asyncio.run(client.extract("No facts."))
    repaired = asyncio.run(client.repair("No facts.", "not json", ["content: invalid JSON"]))

    assert initial.content == repaired.content == '{"entities":[],"relations":[]}'
    assert initial.prompt_tokens == 10
    assert initial.cache_hit_tokens == 4
    assert len(sdk.completions.requests) == 2
    for request in sdk.completions.requests:
        assert request["model"] == "deepseek-v4-flash"
        assert request["response_format"] == {"type": "json_object"}
        assert request["extra_body"] == {"thinking": {"type": "disabled"}}
        assert request["max_tokens"] == 1234
        assert request["temperature"] == 0.0
        messages = request["messages"]
        assert "JSON" in messages[0]["content"]
    assert "complete replacement JSON" in sdk.completions.requests[1]["messages"][1]["content"]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, TerminalProviderError),
        (401, TerminalProviderError),
        (402, TerminalProviderError),
        (422, TerminalProviderError),
        (429, RetryableProviderError),
        (500, RetryableProviderError),
        (503, RetryableProviderError),
    ],
)
def test_provider_errors_are_classified_by_status(status: int, expected: type[Exception]) -> None:
    error = RuntimeError("provider failure")
    error.status_code = status  # type: ignore[attr-defined]

    assert isinstance(classify_provider_error(error), expected)
