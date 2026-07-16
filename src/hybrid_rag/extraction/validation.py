from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NoReturn

from json_repair import loads as repair_json
from pydantic import ValidationError

from hybrid_rag.extraction.schemas import (
    MAX_EXTRACTION_ENTITIES,
    MAX_EXTRACTION_RECORDS,
    EntityCandidate,
    EntityMention,
    EvidenceSpan,
    RelationCandidate,
    RelationMention,
    ValidatedChunkExtraction,
)
from hybrid_rag.ids import stable_id

logger = logging.getLogger(__name__)


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
    """Salvage valid records from one completion and attach trusted provenance."""

    _validate_finish_reason(finish_reason)
    if content is None or not content.strip():
        _raise(
            ValidationFailureKind.EMPTY_CONTENT,
            "content",
            "provider returned empty content",
            "empty_content",
            repairable=True,
        )

    payload = _parse_payload(content)
    warnings: list[str] = []
    raw_entities = _record_list(payload, "entities", warnings)
    raw_relations = _record_list(payload, "relations", warnings)

    entities: list[EntityMention] = []
    entity_ids: dict[str, str] = {}
    if len(raw_entities) > MAX_EXTRACTION_ENTITIES:
        warnings.append(
            f"entities: kept first {MAX_EXTRACTION_ENTITIES} of {len(raw_entities)} records"
        )
    for index, raw in enumerate(raw_entities[:MAX_EXTRACTION_ENTITIES]):
        try:
            candidate = EntityCandidate.model_validate(raw)
        except ValidationError as error:
            warnings.append(_record_error("entities", index, error))
            continue
        if candidate.ref in entity_ids:
            warnings.append(f"entities.{index}: duplicate ref {candidate.ref!r}")
            continue
        if not _quotes_are_verbatim(candidate.evidence_quotes, chunk_text):
            warnings.append(f"entities.{index}: evidence is not verbatim")
            continue
        mention_id = stable_id("emn", extraction_id, candidate.ref)
        entity_ids[candidate.ref] = mention_id
        entities.append(
            EntityMention(
                id=mention_id,
                name=candidate.name,
                entity_type=candidate.entity_type,
                description=candidate.description,
                aliases=tuple(sorted(candidate.aliases, key=_text_sort_key)),
                source_chunk_ids=(source_chunk_id,),
                evidence=_evidence_spans(
                    source_chunk_id,
                    chunk_text,
                    candidate.evidence_quotes,
                ),
            )
        )

    relation_limit = max(0, MAX_EXTRACTION_RECORDS - len(entities))
    if len(raw_relations) > relation_limit:
        warnings.append(f"relations: kept first {relation_limit} of {len(raw_relations)} records")
    relations: list[RelationMention] = []
    relation_keys: set[tuple[str, str, str]] = set()
    sanitized_relation_records = 0
    for index, raw in enumerate(raw_relations[:relation_limit]):
        raw, sanitized = _sanitize_relation_record(raw)
        if sanitized:
            sanitized_relation_records += 1
            warnings.append(f"relations.{index}: ignored unsupported aliases field")
        try:
            candidate = RelationCandidate.model_validate(raw)
        except ValidationError as error:
            warnings.append(_record_error("relations", index, error))
            continue
        if candidate.source_ref == candidate.target_ref:
            warnings.append(f"relations.{index}: self relation was dropped")
            continue
        if candidate.source_ref not in entity_ids or candidate.target_ref not in entity_ids:
            warnings.append(f"relations.{index}: endpoint references a dropped or missing entity")
            continue
        key = candidate.source_ref, candidate.target_ref, candidate.predicate
        if key in relation_keys:
            warnings.append(f"relations.{index}: duplicate relation was dropped")
            continue
        if not _quotes_are_verbatim(candidate.evidence_quotes, chunk_text):
            warnings.append(f"relations.{index}: evidence is not verbatim")
            continue
        relation_keys.add(key)
        relations.append(
            RelationMention(
                id=stable_id(
                    "rmn",
                    extraction_id,
                    str(index),
                    entity_ids[candidate.source_ref],
                    entity_ids[candidate.target_ref],
                    candidate.predicate,
                ),
                source_mention_id=entity_ids[candidate.source_ref],
                target_mention_id=entity_ids[candidate.target_ref],
                predicate=candidate.predicate,
                description=candidate.description,
                source_chunk_ids=(source_chunk_id,),
                evidence=_evidence_spans(
                    source_chunk_id,
                    chunk_text,
                    candidate.evidence_quotes,
                ),
            )
        )
    if warnings:
        logger.warning(
            "%s: salvaged %d entities and %d relations; validation notes: %s",
            extraction_id,
            len(entities),
            len(relations),
            "; ".join(warnings[:12]),
        )
    return ValidatedChunkExtraction(
        extraction_id=extraction_id,
        source_chunk_id=source_chunk_id,
        entities=tuple(entities),
        relations=tuple(relations),
        raw_entity_count=len(raw_entities),
        raw_relation_count=len(raw_relations),
        dropped_entity_count=max(len(raw_entities) - len(entities), 0),
        dropped_relation_count=max(len(raw_relations) - len(relations), 0),
        sanitized_relation_records=sanitized_relation_records,
        validation_warnings=tuple(warnings[:50]),
    )


def _sanitize_relation_record(raw: object) -> tuple[object, bool]:
    """Remove the one known harmless entity-field leak without weakening validation.

    Some providers copy ``aliases`` from the entity schema into otherwise valid relation
    objects.  Relations have no aliases, so the field carries no semantics.  Unknown fields
    other than this one remain schema errors and the record is still dropped.
    """

    if not isinstance(raw, Mapping) or "aliases" not in raw:
        return raw, False
    sanitized = dict(raw)
    sanitized.pop("aliases", None)
    return sanitized, True


def _validate_finish_reason(finish_reason: str | None) -> None:
    if finish_reason not in {"content_filter", "insufficient_system_resource"}:
        return
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


def _parse_payload(content: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(content, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, _DuplicateJsonKeyError):
        try:
            parsed = repair_json(_strip_code_fence(content))
        except Exception as error:
            _raise(
                ValidationFailureKind.INVALID_JSON,
                "content",
                str(error),
                "invalid_json",
                repairable=True,
            )
    if not isinstance(parsed, Mapping):
        _raise(
            ValidationFailureKind.SCHEMA_INVALID,
            "content",
            "extraction result must be a JSON object",
            "object_type",
            repairable=True,
        )
    return parsed


def _strip_code_fence(content: str) -> str:
    value = content.strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if lines:
        lines.pop(0)
    if lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


def _record_list(
    payload: Mapping[str, Any],
    field: str,
    warnings: list[str],
) -> list[Any]:
    value = payload.get(field, [])
    if isinstance(value, list):
        return value
    warnings.append(f"{field}: expected an array")
    return []


def _record_error(field: str, index: int, error: ValidationError) -> str:
    first = error.errors(include_url=False)[0]
    location = ".".join(str(part) for part in first["loc"])
    suffix = f".{location}" if location else ""
    return f"{field}.{index}{suffix}: {first['msg']}"


def _quotes_are_verbatim(quotes: list[str], chunk_text: str) -> bool:
    return all(quote in chunk_text for quote in quotes)


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
) -> NoReturn:
    raise ExtractionValidationError(
        kind,
        (ValidationIssue(path=path, message=message, code=code),),
        repairable=repairable,
        retryable_provider=retryable_provider,
    )
