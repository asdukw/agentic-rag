from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable

from hybrid_rag.extraction.schemas import (
    CanonicalEntity,
    CanonicalRelation,
    EntityMention,
    EntityNormalizationResult,
    EvidenceSpan,
    RelationMention,
    RelationMergeResult,
)
from hybrid_rag.ids import stable_id

_SURROUNDING_QUOTES = " \t\r\n\"'`\N{LEFT DOUBLE QUOTATION MARK}\N{RIGHT DOUBLE QUOTATION MARK}"
_SURROUNDING_QUOTES += "\N{LEFT SINGLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}"
_NON_PREDICATE = re.compile(r"[^A-Z0-9]+")
_MAX_DESCRIPTION_CHARS = 4000


def normalize_entity_alias(value: str) -> str:
    """Conservatively normalize an alias without deleting internal punctuation."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = " ".join(normalized.split()).strip(_SURROUNDING_QUOTES)
    return normalized.casefold()


def normalize_predicate(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().upper()
    normalized = _NON_PREDICATE.sub("_", normalized).strip("_")
    if not normalized:
        raise ValueError("relation predicate normalizes to an empty value")
    if len(normalized) > 80:
        raise ValueError("normalized relation predicate exceeds 80 characters")
    return normalized


def normalize_entities(mentions: Iterable[EntityMention]) -> EntityNormalizationResult:
    """Union mentions only when a normalized alias overlaps within the same type."""

    ordered = sorted(mentions, key=lambda item: item.id)
    _reject_duplicate_ids((item.id for item in ordered), "entity mention")
    if not ordered:
        return EntityNormalizationResult(entities=(), mention_to_entity={})

    union = _UnionFind(len(ordered))
    owner_by_alias: dict[tuple[str, str], int] = {}
    for index, mention in enumerate(ordered):
        aliases = {
            normalized
            for value in (mention.name, *mention.aliases)
            if (normalized := normalize_entity_alias(value))
        }
        if not aliases:
            raise ValueError(f"entity mention {mention.id} has no usable aliases")
        for alias in sorted(aliases):
            key = mention.entity_type.value, alias
            previous = owner_by_alias.setdefault(key, index)
            union.join(index, previous)

    groups: dict[int, list[EntityMention]] = defaultdict(list)
    for index, mention in enumerate(ordered):
        groups[union.find(index)].append(mention)

    entities: list[CanonicalEntity] = []
    mention_to_entity: dict[str, str] = {}
    for members in sorted(groups.values(), key=lambda values: min(item.id for item in values)):
        members.sort(key=lambda item: item.id)
        entity_type = members[0].entity_type
        if any(item.entity_type != entity_type for item in members):
            raise AssertionError("normalizer joined incompatible entity types")

        canonical_name = _choose_canonical_name(members)
        canonical_key = normalize_entity_alias(canonical_name)
        entity_id = stable_id("ent", entity_type.value, canonical_key)
        all_names = _unique_text(value for item in members for value in (item.name, *item.aliases))
        aliases = tuple(value for value in all_names if value != canonical_name)
        evidence = _merge_evidence(item.evidence for item in members)
        source_chunk_ids = tuple(sorted({span.source_chunk_id for span in evidence}))
        entity = CanonicalEntity(
            id=entity_id,
            canonical_name=canonical_name,
            entity_type=entity_type,
            description=_merge_descriptions(item.description for item in members),
            aliases=aliases,
            source_chunk_ids=source_chunk_ids,
            evidence=evidence,
        )
        entities.append(entity)
        for mention in members:
            mention_to_entity[mention.id] = entity_id

    entities.sort(key=lambda item: item.id)
    _reject_duplicate_ids((item.id for item in entities), "canonical entity")
    return EntityNormalizationResult(
        entities=tuple(entities),
        mention_to_entity=dict(sorted(mention_to_entity.items())),
    )


def merge_relations(
    mentions: Iterable[RelationMention],
    mention_to_entity: dict[str, str],
) -> RelationMergeResult:
    """Merge exact directed (source, target, normalized predicate) relations."""

    ordered = sorted(mentions, key=lambda item: item.id)
    _reject_duplicate_ids((item.id for item in ordered), "relation mention")
    grouped: dict[tuple[str, str, str], list[RelationMention]] = defaultdict(list)
    for mention in ordered:
        try:
            source_entity_id = mention_to_entity[mention.source_mention_id]
            target_entity_id = mention_to_entity[mention.target_mention_id]
        except KeyError as error:
            raise ValueError(
                f"relation mention {mention.id} references an unnormalized entity mention: "
                f"{error.args[0]}"
            ) from error
        key = source_entity_id, target_entity_id, normalize_predicate(mention.predicate)
        grouped[key].append(mention)

    relations: list[CanonicalRelation] = []
    for (source_id, target_id, predicate), members in sorted(grouped.items()):
        evidence = _merge_evidence(item.evidence for item in members)
        source_chunk_ids = tuple(sorted({span.source_chunk_id for span in evidence}))
        relations.append(
            CanonicalRelation(
                id=stable_id("rel", source_id, target_id, predicate),
                source_entity_id=source_id,
                target_entity_id=target_id,
                predicate=predicate,
                description=_merge_descriptions(item.description for item in members),
                source_chunk_ids=source_chunk_ids,
                evidence=evidence,
            )
        )
    _reject_duplicate_ids((item.id for item in relations), "canonical relation")
    return RelationMergeResult(relations=tuple(relations))


def _choose_canonical_name(members: list[EntityMention]) -> str:
    primary_counts = Counter(normalize_entity_alias(item.name) for item in members)
    primary_key = min(primary_counts, key=lambda key: (-primary_counts[key], key))
    displays = [item.name for item in members if normalize_entity_alias(item.name) == primary_key]
    display_counts = Counter(displays)
    return min(
        display_counts,
        key=lambda value: (-display_counts[value], value.casefold(), value),
    )


def _unique_text(values: Iterable[str]) -> tuple[str, ...]:
    unique: dict[str, str] = {}
    for value in values:
        key = unicodedata.normalize("NFKC", value)
        current = unique.get(key)
        if current is None or (value.casefold(), value) < (current.casefold(), current):
            unique[key] = value
    return tuple(sorted(unique.values(), key=lambda value: (value.casefold(), value)))


def _merge_descriptions(values: Iterable[str]) -> str:
    unique = sorted(
        set(values),
        key=lambda value: (-len(value), value.casefold(), value),
    )
    selected: list[str] = []
    used = 0
    for value in unique:
        separator = 1 if selected else 0
        if used + separator + len(value) > _MAX_DESCRIPTION_CHARS:
            continue
        selected.append(value)
        used += separator + len(value)
    if not selected:
        # Individual descriptions are schema-bounded to this size, so this is defensive.
        return unique[0][:_MAX_DESCRIPTION_CHARS]
    return "\n".join(selected)


def _merge_evidence(
    groups: Iterable[tuple[EvidenceSpan, ...]],
) -> tuple[EvidenceSpan, ...]:
    by_key: dict[tuple[str, int, int, str], EvidenceSpan] = {}
    for group in groups:
        for item in group:
            key = item.source_chunk_id, item.char_start, item.char_end, item.quote
            by_key[key] = item
    return tuple(by_key[key] for key in sorted(by_key))


def _reject_duplicate_ids(values: Iterable[str], label: str) -> None:
    ordered = list(values)
    if len(ordered) != len(set(ordered)):
        raise ValueError(f"duplicate {label} IDs are not allowed")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parents = list(range(size))

    def find(self, value: int) -> int:
        parent = self._parents[value]
        if parent != value:
            self._parents[value] = self.find(parent)
        return self._parents[value]

    def join(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        lower, higher = sorted((left_root, right_root))
        self._parents[higher] = lower
