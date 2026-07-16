from __future__ import annotations

import re
import unicodedata
from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from hybrid_rag.ids import canonical_json_hash

EXTRACTION_SCHEMA_VERSION = "4"
EXTRACTION_CONFIG_VERSION = "1"
GRAPH_SCHEMA_VERSION = "1"
ENTITY_NORMALIZER_VERSION = "2"
RELATION_MERGER_VERSION = "1"
EXTRACTION_PROMPT_VERSION = "5"
REPAIR_PROMPT_VERSION = "5"

MAX_EXTRACTION_ENTITIES = 24
MAX_EXTRACTION_RECORDS = 64
MAX_EXTRACTION_RELATIONS = MAX_EXTRACTION_RECORDS

_NON_ENTITY_TYPE = re.compile(r"[^A-Z0-9]+")
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalize_entity_type(value: object) -> str:
    """Normalize an open model-authored type into stable UPPER_SNAKE_CASE."""

    if not isinstance(value, str):
        raise ValueError("entity_type must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = _CAMEL_CASE_BOUNDARY.sub("_", normalized).upper()
    normalized = _NON_ENTITY_TYPE.sub("_", normalized).strip("_")
    if not normalized:
        raise ValueError("entity_type normalizes to an empty value")
    if len(normalized) > 64:
        raise ValueError("normalized entity_type exceeds 64 characters")
    if not normalized[0].isalpha():
        raise ValueError("normalized entity_type must start with a letter")
    return normalized


LocalEntityRef = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^e[1-9][0-9]*$"),
]
NameText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
DescriptionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
]
EvidenceText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8000),
]
PredicateText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[A-Z][A-Z0-9_]{0,79}$"),
]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
EntityType = Annotated[
    str,
    StringConstraints(max_length=64, pattern=r"^[A-Z][A-Z0-9_]{0,63}$"),
    BeforeValidator(normalize_entity_type),
]


class _StrictOutputModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class _FrozenDomainModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


class EntityCandidate(_StrictOutputModel):
    """Untrusted entity shape emitted by the model for one chunk."""

    ref: LocalEntityRef
    name: NameText
    entity_type: EntityType
    description: DescriptionText
    aliases: list[NameText] = Field(default_factory=list, max_length=12)
    evidence_quotes: list[EvidenceText] = Field(min_length=1, max_length=1)

    @field_validator("aliases", "evidence_quotes")
    @classmethod
    def reject_duplicate_values(cls, values: list[str]) -> list[str]:
        folded = [value.casefold() for value in values]
        if len(folded) != len(set(folded)):
            raise ValueError("values must be unique ignoring case")
        return values


class RelationCandidate(_StrictOutputModel):
    """Untrusted directed relation shape emitted by the model for one chunk."""

    source_ref: LocalEntityRef
    target_ref: LocalEntityRef
    predicate: PredicateText
    description: DescriptionText
    evidence_quotes: list[EvidenceText] = Field(min_length=1, max_length=1)

    @field_validator("evidence_quotes")
    @classmethod
    def reject_duplicate_evidence(cls, values: list[str]) -> list[str]:
        folded = [value.casefold() for value in values]
        if len(folded) != len(set(folded)):
            raise ValueError("evidence quotes must be unique ignoring case")
        return values


class ChunkExtraction(_StrictOutputModel):
    """Complete model output. Empty arrays are a valid no-facts result."""

    entities: list[EntityCandidate] = Field(max_length=MAX_EXTRACTION_ENTITIES)
    relations: list[RelationCandidate] = Field(max_length=MAX_EXTRACTION_RELATIONS)

    @model_validator(mode="after")
    def validate_local_references(self) -> ChunkExtraction:
        if len(self.entities) + len(self.relations) > MAX_EXTRACTION_RECORDS:
            raise ValueError(
                f"entities and relations must total at most {MAX_EXTRACTION_RECORDS} records"
            )
        refs = [entity.ref for entity in self.entities]
        if len(refs) != len(set(refs)):
            raise ValueError("entity refs must be unique")

        known = set(refs)
        dangling = sorted(
            {
                ref
                for relation in self.relations
                for ref in (relation.source_ref, relation.target_ref)
                if ref not in known
            }
        )
        if dangling:
            raise ValueError(f"relation endpoints reference unknown entities: {dangling}")
        return self


class EvidenceSpan(_FrozenDomainModel):
    """A verbatim quote and its offsets relative to the source chunk text."""

    source_chunk_id: Identifier
    quote: EvidenceText
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_span_length(self) -> EvidenceSpan:
        if self.char_end - self.char_start != len(self.quote):
            raise ValueError("evidence span length must equal quote length")
        return self


class EntityMention(_FrozenDomainModel):
    id: Identifier
    name: NameText
    entity_type: EntityType
    description: DescriptionText
    aliases: tuple[NameText, ...] = ()
    source_chunk_ids: tuple[Identifier, ...]
    evidence: tuple[EvidenceSpan, ...]

    @model_validator(mode="after")
    def validate_provenance(self) -> EntityMention:
        _validate_domain_provenance(self.source_chunk_ids, self.evidence)
        return self


class RelationMention(_FrozenDomainModel):
    id: Identifier
    source_mention_id: Identifier
    target_mention_id: Identifier
    predicate: PredicateText
    description: DescriptionText
    source_chunk_ids: tuple[Identifier, ...]
    evidence: tuple[EvidenceSpan, ...]

    @model_validator(mode="after")
    def validate_provenance(self) -> RelationMention:
        _validate_domain_provenance(self.source_chunk_ids, self.evidence)
        return self


class ValidatedChunkExtraction(_FrozenDomainModel):
    extraction_id: Identifier
    source_chunk_id: Identifier
    entities: tuple[EntityMention, ...] = ()
    relations: tuple[RelationMention, ...] = ()
    raw_entity_count: int = Field(default=0, ge=0)
    raw_relation_count: int = Field(default=0, ge=0)
    dropped_entity_count: int = Field(default=0, ge=0)
    dropped_relation_count: int = Field(default=0, ge=0)
    sanitized_relation_records: int = Field(default=0, ge=0)
    validation_warnings: tuple[str, ...] = ()


class CanonicalEntity(_FrozenDomainModel):
    id: Identifier
    canonical_name: NameText
    entity_type: EntityType
    description: DescriptionText
    aliases: tuple[NameText, ...] = ()
    source_chunk_ids: tuple[Identifier, ...]
    evidence: tuple[EvidenceSpan, ...]

    @model_validator(mode="after")
    def validate_provenance(self) -> CanonicalEntity:
        _validate_domain_provenance(self.source_chunk_ids, self.evidence)
        return self


class CanonicalRelation(_FrozenDomainModel):
    id: Identifier
    source_entity_id: Identifier
    target_entity_id: Identifier
    predicate: PredicateText
    description: DescriptionText
    source_chunk_ids: tuple[Identifier, ...]
    evidence: tuple[EvidenceSpan, ...]

    @model_validator(mode="after")
    def validate_provenance(self) -> CanonicalRelation:
        _validate_domain_provenance(self.source_chunk_ids, self.evidence)
        return self


class EntityNormalizationResult(_FrozenDomainModel):
    entities: tuple[CanonicalEntity, ...]
    mention_to_entity: dict[Identifier, Identifier]


class RelationMergeResult(_FrozenDomainModel):
    relations: tuple[CanonicalRelation, ...]


class ExtractionConfig(_FrozenDomainModel):
    """Semantic extraction settings. Secrets and execution tuning are excluded."""

    version: str = EXTRACTION_CONFIG_VERSION
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    response_format: str = "json_object"
    thinking: str = "disabled"
    temperature: float = 0.0
    max_output_tokens: int = Field(default=4096, ge=1)
    schema_version: str = EXTRACTION_SCHEMA_VERSION
    prompt_version: str = EXTRACTION_PROMPT_VERSION
    repair_prompt_version: str = REPAIR_PROMPT_VERSION
    repair_max_attempts: int = Field(default=1, ge=0, le=1)

    @property
    def config_hash(self) -> str:
        return canonical_json_hash(self.model_dump(mode="json"))


class GraphConfig(_FrozenDomainModel):
    """Graph semantics layered over a reusable extraction configuration."""

    version: str = GRAPH_SCHEMA_VERSION
    extraction_config_hash: str
    entity_normalizer_version: str = ENTITY_NORMALIZER_VERSION
    relation_merger_version: str = RELATION_MERGER_VERSION

    @property
    def config_hash(self) -> str:
        return canonical_json_hash(self.model_dump(mode="json"))


def _validate_domain_provenance(
    source_chunk_ids: tuple[str, ...], evidence: tuple[EvidenceSpan, ...]
) -> None:
    if not source_chunk_ids:
        raise ValueError("source_chunk_ids must not be empty")
    if tuple(sorted(set(source_chunk_ids))) != source_chunk_ids:
        raise ValueError("source_chunk_ids must be unique and sorted")
    if not evidence:
        raise ValueError("evidence must not be empty")
    evidence_sources = tuple(sorted({item.source_chunk_id for item in evidence}))
    if evidence_sources != source_chunk_ids:
        raise ValueError("source_chunk_ids must match evidence provenance")
