from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from hybrid_rag.extraction.prompts import (
    ChatMessage,
    build_extraction_messages,
    build_repair_messages,
)


@dataclass(frozen=True, slots=True)
class CompletionResult:
    provider_request_id: str | None
    model: str
    system_fingerprint: str | None
    content: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    cache_hit_tokens: int | None
    cache_miss_tokens: int | None
    raw_response: dict[str, Any]


class ExtractionClient(Protocol):
    async def extract(
        self,
        chunk_text: str,
        *,
        document_title: str | None = None,
        section_path: Sequence[str] = (),
    ) -> CompletionResult: ...

    async def repair(
        self,
        chunk_text: str,
        invalid_response: str | None,
        issues: Sequence[str],
        *,
        document_title: str | None = None,
        section_path: Sequence[str] = (),
    ) -> CompletionResult: ...


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.provider_request_id = provider_request_id
        super().__init__(message)


class RetryableProviderError(ProviderError):
    pass


class TerminalProviderError(ProviderError):
    pass


class DeepSeekClient:
    """Thin AsyncOpenAI-compatible DeepSeek adapter with no implicit retries."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        max_output_tokens: int = 4096,
        timeout_seconds: float = 180.0,
        temperature: float = 0.0,
        sdk_client: object | None = None,
    ) -> None:
        if not api_key and sdk_client is None:
            raise ValueError("DeepSeek API key must not be empty")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._owns_client = sdk_client is None
        if sdk_client is None:
            # Keep provider SDK types inside the adapter boundary.
            from openai import AsyncOpenAI

            sdk_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
                timeout=timeout_seconds,
            )
        self._client = sdk_client

    async def extract(
        self,
        chunk_text: str,
        *,
        document_title: str | None = None,
        section_path: Sequence[str] = (),
    ) -> CompletionResult:
        return await self.complete_messages(
            build_extraction_messages(
                chunk_text,
                document_title=document_title,
                section_path=section_path,
            )
        )

    async def repair(
        self,
        chunk_text: str,
        invalid_response: str | None,
        issues: Sequence[str],
        *,
        document_title: str | None = None,
        section_path: Sequence[str] = (),
    ) -> CompletionResult:
        return await self.complete_messages(
            build_repair_messages(
                chunk_text,
                invalid_response,
                issues,
                document_title=document_title,
                section_path=section_path,
            )
        )

    async def complete_messages(self, messages: Sequence[ChatMessage]) -> CompletionResult:
        try:
            response = await self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=[dict(message) for message in messages],
                response_format={"type": "json_object"},
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as error:
            raise classify_provider_error(error) from error

        choices = _attribute(response, "choices", ())
        if not choices:
            raise TerminalProviderError(
                "provider returned no completion choices",
                provider_request_id=_optional_string(_attribute(response, "id")),
            )
        choice = choices[0]
        message = _attribute(choice, "message")
        usage = _attribute(response, "usage")
        return CompletionResult(
            provider_request_id=_optional_string(_attribute(response, "id")),
            model=_optional_string(_attribute(response, "model")) or self.model,
            system_fingerprint=_optional_string(_attribute(response, "system_fingerprint")),
            content=_optional_string(_attribute(message, "content")),
            finish_reason=_optional_string(_attribute(choice, "finish_reason")),
            prompt_tokens=_integer(_attribute(usage, "prompt_tokens")),
            completion_tokens=_integer(_attribute(usage, "completion_tokens")),
            cache_hit_tokens=_optional_integer(_attribute(usage, "prompt_cache_hit_tokens")),
            cache_miss_tokens=_optional_integer(_attribute(usage, "prompt_cache_miss_tokens")),
            raw_response=_response_payload(response),
        )

    async def close(self) -> None:
        if not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    async def __aenter__(self) -> DeepSeekClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


def classify_provider_error(error: Exception) -> ProviderError:
    """Classify SDK errors without leaking SDK-specific types into callers."""

    status_code = _status_code(error)
    request_id = _optional_string(
        getattr(error, "request_id", None)
        or _attribute(getattr(error, "response", None), "request_id")
        or _response_header(getattr(error, "response", None), "x-request-id")
    )
    error_name = type(error).__name__
    message = str(error) or error_name
    retryable_names = {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "TimeoutException",
    }
    retryable = (
        isinstance(error, (ConnectionError, TimeoutError))
        or error_name in retryable_names
        or status_code in {408, 409, 429}
        or (status_code is not None and status_code >= 500)
    )
    error_type = RetryableProviderError if retryable else TerminalProviderError
    return error_type(
        message,
        status_code=status_code,
        provider_request_id=request_id,
    )


def _response_payload(response: object) -> dict[str, Any]:
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    choices = _attribute(response, "choices", ())
    first_choice = choices[0] if choices else None
    message = _attribute(first_choice, "message")
    usage = _attribute(response, "usage")
    return {
        "id": _attribute(response, "id"),
        "model": _attribute(response, "model"),
        "system_fingerprint": _attribute(response, "system_fingerprint"),
        "choices": [
            {
                "finish_reason": _attribute(first_choice, "finish_reason"),
                "message": {"content": _attribute(message, "content")},
            }
        ],
        "usage": {
            "prompt_tokens": _attribute(usage, "prompt_tokens"),
            "completion_tokens": _attribute(usage, "completion_tokens"),
            "prompt_cache_hit_tokens": _attribute(usage, "prompt_cache_hit_tokens"),
            "prompt_cache_miss_tokens": _attribute(usage, "prompt_cache_miss_tokens"),
        },
    }


def _status_code(error: Exception) -> int | None:
    direct = getattr(error, "status_code", None)
    if isinstance(direct, int):
        return direct
    response_status = _attribute(getattr(error, "response", None), "status_code")
    return response_status if isinstance(response_status, int) else None


def _response_header(response: object | None, name: str) -> object | None:
    headers = _attribute(response, "headers")
    if isinstance(headers, Mapping):
        return headers.get(name)
    return None


def _attribute(value: object | None, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _integer(value: object | None) -> int:
    return value if isinstance(value, int) else 0


def _optional_integer(value: object | None) -> int | None:
    return value if isinstance(value, int) else None


def _optional_string(value: object | None) -> str | None:
    return value if isinstance(value, str) else None
