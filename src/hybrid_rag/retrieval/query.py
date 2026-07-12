"""Constrained query-time LLM boundary for retrieval keywording and answers.

The retrievers choose the evidence.  This module deliberately gives an LLM no
tool access and no authority to expand that evidence set: it can only extract
keywords from a question or render an answer whose citations are checked
against a caller-provided allowlist.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from hybrid_rag.extraction.client import CompletionResult, DeepSeekClient
from hybrid_rag.extraction.prompts import ChatMessage

QuestionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=12_000),
]
KeywordText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=160),
]
CitationId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
EvidenceText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000),
]
AnswerText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000),
]

_OFFLINE_INSUFFICIENT_EVIDENCE = "Insufficient evidence in the supplied retrieval context."
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class EvidenceItem(_StrictModel):
    """A preselected evidence passage that an answer may cite exactly once."""

    citation_id: CitationId
    text: EvidenceText
    source_chunk_ids: tuple[CitationId, ...] = ()

    def model_post_init(self, __context: Any) -> None:
        if len(self.source_chunk_ids) != len(set(self.source_chunk_ids)):
            raise ValueError("source_chunk_ids must not contain duplicates")


class KeywordExtraction(_StrictModel):
    """Keyword-only model output used to seed deterministic retrieval routes."""

    keywords: tuple[KeywordText, ...] = Field(max_length=12)

    def model_post_init(self, __context: Any) -> None:
        _reject_casefold_duplicates(self.keywords, field_name="keywords")


class GroundedAnswer(_StrictModel):
    """An answer whose cited IDs are checked after parsing against retrieval context."""

    answer: AnswerText
    citations: tuple[CitationId, ...] = Field(max_length=16)
    insufficient_evidence: bool

    def model_post_init(self, __context: Any) -> None:
        _reject_casefold_duplicates(self.citations, field_name="citations")
        if self.insufficient_evidence and self.citations:
            raise ValueError("insufficient_evidence answers must not include citations")
        if not self.insufficient_evidence and not self.citations:
            raise ValueError("grounded answers must include at least one citation")


class KeywordExtractor(Protocol):
    async def extract_keywords(self, question: str) -> KeywordExtraction: ...


class EvidenceAnswerer(Protocol):
    async def answer(
        self,
        question: str,
        evidence: Sequence[EvidenceItem],
    ) -> GroundedAnswer: ...


class QueryClient(KeywordExtractor, EvidenceAnswerer, Protocol):
    """The only query-time LLM operations allowed by the retrieval service."""


class QueryValidationFailureKind(StrEnum):
    EMPTY_CONTENT = "empty_content"
    INVALID_JSON = "invalid_json"
    SCHEMA_INVALID = "schema_invalid"
    UNEXPECTED_FINISH = "unexpected_finish"
    CITATION_NOT_ALLOWED = "citation_not_allowed"


@dataclass(frozen=True, slots=True)
class QueryValidationIssue:
    path: str
    message: str
    code: str

    def render(self) -> str:
        prefix = f"{self.path}: " if self.path else ""
        return f"{prefix}{self.message} [{self.code}]"


class QueryValidationError(ValueError):
    def __init__(
        self,
        kind: QueryValidationFailureKind,
        issues: tuple[QueryValidationIssue, ...],
    ) -> None:
        self.kind = kind
        self.issues = issues
        super().__init__(f"{kind.value}: {'; '.join(issue.render() for issue in issues)}")


class QueryConfigurationError(RuntimeError):
    """Raised only when an online client is used without credentials."""


class _DuplicateJsonKeyError(ValueError):
    pass


def validate_keyword_completion(
    *,
    content: str | None,
    finish_reason: str | None,
) -> KeywordExtraction:
    """Parse a JSON-only keyword completion under the strict output contract."""

    _validate_json_completion(content=content, finish_reason=finish_reason)
    assert content is not None  # established by _validate_json_completion
    try:
        return KeywordExtraction.model_validate_json(content)
    except ValidationError as error:
        _raise_schema_error(error)


def validate_answer_completion(
    *,
    content: str | None,
    finish_reason: str | None,
    allowed_citation_ids: Sequence[str],
) -> GroundedAnswer:
    """Parse an answer and reject any citation not selected by retrieval.

    This check is intentionally outside the prompt: JSON mode and instructions
    cannot make an untrusted provider response authoritative.
    """

    _validate_json_completion(content=content, finish_reason=finish_reason)
    assert content is not None  # established by _validate_json_completion
    try:
        answer = GroundedAnswer.model_validate_json(content)
    except ValidationError as error:
        _raise_schema_error(error)

    allowed = set(allowed_citation_ids)
    disallowed = tuple(citation for citation in answer.citations if citation not in allowed)
    if disallowed:
        raise QueryValidationError(
            QueryValidationFailureKind.CITATION_NOT_ALLOWED,
            tuple(
                QueryValidationIssue(
                    path=f"citations.{index}",
                    message=f"citation is not in the retrieval allowlist: {citation!r}",
                    code="citation.not_allowed",
                )
                for index, citation in enumerate(answer.citations)
                if citation not in allowed
            ),
        )
    return answer


class DeterministicQueryClient:
    """Offline, reproducible fallback with no network or generative behavior."""

    async def extract_keywords(self, question: str) -> KeywordExtraction:
        return deterministic_keywords(question)

    async def answer(
        self,
        question: str,
        evidence: Sequence[EvidenceItem],
    ) -> GroundedAnswer:
        return deterministic_answer(question, evidence)


class OpenAICompatibleQueryClient:
    """Thin adapter for an OpenAI-compatible JSON chat-completions endpoint.

    A client can be constructed in an offline environment.  Credentials are
    checked only if a method would make a remote request, which lets callers
    choose :class:`DeterministicQueryClient` without special configuration.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str,
        keyword_model: str,
        answer_model: str | None = None,
        max_output_tokens: int = 1_024,
        timeout_seconds: float = 180.0,
        temperature: float = 0.0,
        sdk_client: object | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not keyword_model.strip():
            raise ValueError("keyword_model must not be empty")
        if answer_model is not None and not answer_model.strip():
            raise ValueError("answer_model must not be empty when provided")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.base_url = base_url
        self.keyword_model = keyword_model
        self.answer_model = answer_model or keyword_model
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self._api_key = api_key
        self._sdk_client = sdk_client
        self._owns_sdk_client = False
        self._delegates: dict[str, DeepSeekClient] = {}

    async def extract_keywords(self, question: str) -> KeywordExtraction:
        from hybrid_rag.retrieval.prompts import build_keyword_messages

        completion = await self._complete(
            model=self.keyword_model,
            messages=build_keyword_messages(question),
        )
        return validate_keyword_completion(
            content=completion.content,
            finish_reason=completion.finish_reason,
        )

    async def answer(
        self,
        question: str,
        evidence: Sequence[EvidenceItem],
    ) -> GroundedAnswer:
        from hybrid_rag.retrieval.prompts import build_answer_messages

        normalized_evidence = _validate_evidence(evidence)
        if not normalized_evidence:
            return _insufficient_evidence_answer()
        completion = await self._complete(
            model=self.answer_model,
            messages=build_answer_messages(question, normalized_evidence),
        )
        return validate_answer_completion(
            content=completion.content,
            finish_reason=completion.finish_reason,
            allowed_citation_ids=tuple(item.citation_id for item in normalized_evidence),
        )

    async def close(self) -> None:
        if not self._owns_sdk_client or self._sdk_client is None:
            return
        close = getattr(self._sdk_client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        self._owns_sdk_client = False

    async def __aenter__(self) -> OpenAICompatibleQueryClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _complete(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
    ) -> CompletionResult:
        return await self._delegate_for(model).complete_messages(messages)

    def _delegate_for(self, model: str) -> DeepSeekClient:
        delegate = self._delegates.get(model)
        if delegate is not None:
            return delegate
        delegate = DeepSeekClient(
            api_key="",
            model=model,
            max_output_tokens=self.max_output_tokens,
            timeout_seconds=self.timeout_seconds,
            temperature=self.temperature,
            sdk_client=self._sdk(),
        )
        self._delegates[model] = delegate
        return delegate

    def _sdk(self) -> object:
        if self._sdk_client is not None:
            return self._sdk_client
        if not self._api_key:
            raise QueryConfigurationError(
                "an API key is required before the OpenAI-compatible query client can call a model"
            )
        # Keep the provider SDK inside this adapter boundary.  Importing lazily
        # also keeps deterministic/offline retrieval free from SDK setup.
        from openai import AsyncOpenAI

        self._sdk_client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=self.timeout_seconds,
        )
        self._owns_sdk_client = True
        return self._sdk_client


class DeepSeekQueryClient(OpenAICompatibleQueryClient):
    """DeepSeek defaults over the general OpenAI-compatible query adapter."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "deepseek-v4-flash",
        answer_model: str | None = None,
        base_url: str = "https://api.deepseek.com",
        max_output_tokens: int = 1_024,
        timeout_seconds: float = 180.0,
        temperature: float = 0.0,
        sdk_client: object | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            keyword_model=model,
            answer_model=answer_model,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            sdk_client=sdk_client,
        )


def deterministic_keywords(question: str) -> KeywordExtraction:
    """Return a bounded, stable lexical keyword set without contacting a model."""

    normalized_question = _validate_question(question)
    keywords: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_PATTERN.finditer(normalized_question):
        keyword = match.group(0)
        folded = keyword.casefold()
        if folded in _STOPWORDS or folded in seen:
            continue
        seen.add(folded)
        keywords.append(keyword)
        if len(keywords) == 12:
            break
    return KeywordExtraction(keywords=tuple(keywords))


def deterministic_answer(question: str, evidence: Sequence[EvidenceItem]) -> GroundedAnswer:
    """Return verbatim selected evidence instead of pretending to synthesize it."""

    _validate_question(question)
    normalized_evidence = _validate_evidence(evidence)
    if not normalized_evidence:
        return _insufficient_evidence_answer()
    first = normalized_evidence[0]
    return GroundedAnswer(
        answer=first.text,
        citations=(first.citation_id,),
        insufficient_evidence=False,
    )


def _validate_json_completion(*, content: str | None, finish_reason: str | None) -> None:
    if finish_reason != "stop":
        raise QueryValidationError(
            QueryValidationFailureKind.UNEXPECTED_FINISH,
            (
                QueryValidationIssue(
                    path="finish_reason",
                    message=f"expected provider finish reason 'stop', got {finish_reason!r}",
                    code="finish_reason.unexpected",
                ),
            ),
        )
    if content is None or not content.strip():
        raise QueryValidationError(
            QueryValidationFailureKind.EMPTY_CONTENT,
            (
                QueryValidationIssue(
                    path="content",
                    message="provider returned empty content",
                    code="content.empty",
                ),
            ),
        )
    try:
        json.loads(content, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, _DuplicateJsonKeyError) as error:
        raise QueryValidationError(
            QueryValidationFailureKind.INVALID_JSON,
            (
                QueryValidationIssue(
                    path="content",
                    message=str(error),
                    code="json.invalid",
                ),
            ),
        ) from error


def _raise_schema_error(error: ValidationError) -> None:
    raise QueryValidationError(
        QueryValidationFailureKind.SCHEMA_INVALID,
        tuple(
            QueryValidationIssue(
                path=".".join(str(part) for part in item["loc"]),
                message=str(item["msg"]),
                code=str(item["type"]),
            )
            for item in error.errors(include_url=False)
        ),
    ) from error


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _validate_question(question: str) -> str:
    try:
        # Validate through a local strict model rather than coercing arbitrary
        # caller values into prompt text.
        return _QuestionInput(question=question).question
    except ValidationError as error:
        _raise_schema_error(error)


def _validate_evidence(evidence: Sequence[EvidenceItem]) -> tuple[EvidenceItem, ...]:
    values = tuple(evidence)
    citation_ids = [item.citation_id for item in values]
    if len(citation_ids) != len(set(citation_ids)):
        raise ValueError("evidence citation_id values must be unique")
    return values


def _insufficient_evidence_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer=_OFFLINE_INSUFFICIENT_EVIDENCE,
        citations=(),
        insufficient_evidence=True,
    )


def _reject_casefold_duplicates(values: Sequence[str], *, field_name: str) -> None:
    folded = [value.casefold() for value in values]
    if len(folded) != len(set(folded)):
        raise ValueError(f"{field_name} must not contain duplicates ignoring case")


class _QuestionInput(_StrictModel):
    question: QuestionText
