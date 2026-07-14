from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from hybrid_rag.extraction.schemas import (
    ChunkExtraction,
    EntityMention,
    EvidenceSpan,
    RelationMention,
    ValidatedChunkExtraction,
)
from hybrid_rag.ids import stable_id


class ValidationFailureKind(StrEnum):
    TRUNCATED = "truncated"
    EMPTY_CONTENT = "empty_content"
    INVALID_JSON = "invalid_json"
    SCHEMA_INVALID = "schema_invalid"
    EVIDENCE_INVALID = "evidence_invalid"
    CONTENT_FILTERED = "content_filtered"
    RETRYABLE_PROVIDER_FINISH = "retryable_provider_finish"
    UNEXPECTED_FINISH = "unexpected_finish"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: str
    message: str
    code: str

    def render(self) -> str:
        prefix = f"{self.path}: " if self.path else ""
        return f"{prefix}{self.message} [{self.code}]"


class ExtractionValidationError(ValueError):
    def __init__(
        self,
        kind: ValidationFailureKind,
        issues: tuple[ValidationIssue, ...],
        *,
        repairable: bool,
        retryable_provider: bool = False,
    ) -> None:
        self.kind = kind
        self.issues = issues
        self.repairable = repairable
        self.retryable_provider = retryable_provider
        detail = "; ".join(issue.render() for issue in issues)
        super().__init__(f"{kind.value}: {detail}")

    @property
    def repair_messages(self) -> tuple[str, ...]:
        return tuple(issue.render() for issue in self.issues)


class _DuplicateJsonKeyError(ValueError):
    pass


def validate_completion(
    *,
    extraction_id: str,
    source_chunk_id: str,
    chunk_text: str,
    content: str | None,
    finish_reason: str | None,
) -> ValidatedChunkExtraction:
    """Validate one completion and attach trusted, chunk-local provenance."""

    _validate_finish_reason(finish_reason)
    if content is None or not content.strip():
        _raise(
            ValidationFailureKind.EMPTY_CONTENT,
            "content",
            "provider returned empty content",
            "empty_content",
            repairable=True,
        )

    try:
        json.loads(content, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, _DuplicateJsonKeyError) as error:
        _raise(
            ValidationFailureKind.INVALID_JSON,
            "content",
            str(error),
            "invalid_json",
            repairable=True,
        )

    try:
        extracted = ChunkExtraction.model_validate_json(content)
    except ValidationError as error:
        issues = tuple(
            ValidationIssue(
                path=_format_location(item["loc"]),
                message=str(item["msg"]),
                code=str(item["type"]),
            )
            for item in error.errors(include_url=False)
        )
        raise ExtractionValidationError(
            ValidationFailureKind.SCHEMA_INVALID,
            issues,
            repairable=True,
        ) from error

    evidence_issues = _find_evidence_issues(extracted, chunk_text)
    if evidence_issues:
        raise ExtractionValidationError(
            ValidationFailureKind.EVIDENCE_INVALID,
            evidence_issues,
            repairable=True,
        )

    entity_ids = {
        entity.ref: stable_id("emn", extraction_id, entity.ref) for entity in extracted.entities
    }
    entities = tuple(
        EntityMention(
            id=entity_ids[candidate.ref],
            name=candidate.name,
            entity_type=candidate.entity_type,
            description=candidate.description,
            aliases=tuple(sorted(candidate.aliases, key=_text_sort_key)),
            source_chunk_ids=(source_chunk_id,),
            evidence=_evidence_spans(source_chunk_id, chunk_text, candidate.evidence_quotes),
        )
        for candidate in extracted.entities
    )
    relations = tuple(
        RelationMention(
            id=stable_id(
                "rmn",
                extraction_id,
                str(ordinal),
                entity_ids[candidate.source_ref],
                entity_ids[candidate.target_ref],
                candidate.predicate,
            ),
            source_mention_id=entity_ids[candidate.source_ref],
            target_mention_id=entity_ids[candidate.target_ref],
            predicate=candidate.predicate,
            description=candidate.description,
            source_chunk_ids=(source_chunk_id,),
            evidence=_evidence_spans(source_chunk_id, chunk_text, candidate.evidence_quotes),
        )
        for ordinal, candidate in enumerate(extracted.relations)
    )
    return ValidatedChunkExtraction(
        extraction_id=extraction_id,
        source_chunk_id=source_chunk_id,
        entities=entities,
        relations=relations,
    )


def _validate_finish_reason(finish_reason: str | None) -> None:
    if finish_reason == "stop":
        return
    if finish_reason == "length":
        _raise(
            ValidationFailureKind.TRUNCATED,
            "finish_reason",
            "completion was truncated",
            "finish_reason.length",
            repairable=True,
        )
    if finish_reason == "content_filter":
        _raise(
            ValidationFailureKind.CONTENT_FILTERED,
            "finish_reason",
            "provider content filter omitted the completion",
            "finish_reason.content_filter",
            repairable=False,
        )
    if finish_reason == "insufficient_system_resource":
        _raise(
            ValidationFailureKind.RETRYABLE_PROVIDER_FINISH,
            "finish_reason",
            "provider stopped because inference resources were unavailable",
            "finish_reason.insufficient_system_resource",
            repairable=False,
            retryable_provider=True,
        )
    _raise(
        ValidationFailureKind.UNEXPECTED_FINISH,
        "finish_reason",
        f"unexpected finish reason: {finish_reason!r}",
        "finish_reason.unexpected",
        repairable=True,
    )


def _find_evidence_issues(
    extracted: ChunkExtraction, chunk_text: str
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    for entity_index, entity in enumerate(extracted.entities):
        for quote_index, quote in enumerate(entity.evidence_quotes):
            if quote not in chunk_text:
                issues.append(
                    ValidationIssue(
                        path=f"entities.{entity_index}.evidence_quotes.{quote_index}",
                        message="evidence quote is not a verbatim substring of the source chunk",
                        code="evidence.not_verbatim",
                    )
                )
    for relation_index, relation in enumerate(extracted.relations):
        for quote_index, quote in enumerate(relation.evidence_quotes):
            if quote not in chunk_text:
                issues.append(
                    ValidationIssue(
                        path=f"relations.{relation_index}.evidence_quotes.{quote_index}",
                        message="evidence quote is not a verbatim substring of the source chunk",
                        code="evidence.not_verbatim",
                    )
                )
    return tuple(issues)


def _evidence_spans(
    source_chunk_id: str, chunk_text: str, quotes: list[str]
) -> tuple[EvidenceSpan, ...]:
    spans = []
    for quote in quotes:
        start = chunk_text.find(quote)
        if start < 0:
            raise AssertionError("evidence was not validated before span construction")
        spans.append(
            EvidenceSpan(
                source_chunk_id=source_chunk_id,
                quote=quote,
                char_start=start,
                char_end=start + len(quote),
            )
        )
    return tuple(sorted(spans, key=lambda item: (item.char_start, item.char_end, item.quote)))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _format_location(location: tuple[int | str, ...]) -> str:
    return ".".join(str(part) for part in location)


def _text_sort_key(value: str) -> tuple[str, str]:
    return value.casefold(), value


def _raise(
    kind: ValidationFailureKind,
    path: str,
    message: str,
    code: str,
    *,
    repairable: bool,
    retryable_provider: bool = False,
) -> None:
    raise ExtractionValidationError(
        kind,
        (ValidationIssue(path=path, message=message, code=code),),
        repairable=repairable,
        retryable_provider=retryable_provider,
    )
