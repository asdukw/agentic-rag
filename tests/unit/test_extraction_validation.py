from __future__ import annotations

import json

import pytest

from hybrid_rag.extraction.schemas import EntityType, ExtractionConfig, GraphConfig
from hybrid_rag.extraction.validation import (
    ExtractionValidationError,
    ValidationFailureKind,
    validate_completion,
)


def _payload() -> dict[str, object]:
    return {
        "entities": [
            {
                "ref": "e1",
                "name": "LightRAG",
                "entity_type": "METHOD",
                "description": "A graph retrieval method.",
                "aliases": [],
                "evidence_quotes": ["LightRAG"],
            },
            {
                "ref": "e2",
                "name": "knowledge graph",
                "entity_type": "CONCEPT",
                "description": "A graph representation.",
                "aliases": ["KG"],
                "evidence_quotes": ["knowledge graph"],
            },
        ],
        "relations": [
            {
                "source_ref": "e1",
                "target_ref": "e2",
                "predicate": "USES",
                "description": "LightRAG uses a knowledge graph.",
                "evidence_quotes": ["LightRAG uses a knowledge graph"],
            }
        ],
    }


def test_valid_completion_stamps_ids_and_verbatim_evidence() -> None:
    chunk = "LightRAG uses a knowledge graph for retrieval."

    result = validate_completion(
        extraction_id="xtr_fixture",
        source_chunk_id="chk_fixture",
        chunk_text=chunk,
        content=json.dumps(_payload()),
        finish_reason="stop",
    )

    assert [entity.id[:4] for entity in result.entities] == ["emn_", "emn_"]
    assert result.relations[0].source_mention_id == result.entities[0].id
    assert result.relations[0].target_mention_id == result.entities[1].id
    assert result.entities[0].source_chunk_ids == ("chk_fixture",)
    evidence = result.relations[0].evidence[0]
    assert chunk[evidence.char_start : evidence.char_end] == evidence.quote


def test_empty_extraction_is_valid() -> None:
    result = validate_completion(
        extraction_id="xtr_empty",
        source_chunk_id="chk_empty",
        chunk_text="No graph facts here.",
        content='{"entities":[],"relations":[]}',
        finish_reason="stop",
    )

    assert result.entities == ()
    assert result.relations == ()


def test_system_entity_type_is_accepted() -> None:
    payload = _payload()
    payload["entities"][0]["entity_type"] = "SYSTEM"

    result = validate_completion(
        extraction_id="xtr_system",
        source_chunk_id="chk_system",
        chunk_text="LightRAG uses a knowledge graph for retrieval.",
        content=json.dumps(payload),
        finish_reason="stop",
    )

    assert result.entities[0].entity_type is EntityType.SYSTEM


def test_empty_extraction_still_requires_both_contract_fields() -> None:
    with pytest.raises(ExtractionValidationError) as captured:
        validate_completion(
            extraction_id="xtr_empty",
            source_chunk_id="chk_empty",
            chunk_text="No graph facts here.",
            content="{}",
            finish_reason="stop",
        )

    assert captured.value.kind is ValidationFailureKind.SCHEMA_INVALID


@pytest.mark.parametrize(
    ("mutate", "expected_kind"),
    [
        (
            lambda payload: payload["entities"].append(payload["entities"][0].copy()),
            ValidationFailureKind.SCHEMA_INVALID,
        ),
        (
            lambda payload: payload["relations"][0].update({"target_ref": "e9"}),
            ValidationFailureKind.SCHEMA_INVALID,
        ),
        (
            lambda payload: payload["entities"][0].update(
                {"evidence_quotes": ["fabricated quote"]}
            ),
            ValidationFailureKind.EVIDENCE_INVALID,
        ),
    ],
)
def test_invalid_refs_and_evidence_are_repairable(mutate, expected_kind) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(ExtractionValidationError) as captured:
        validate_completion(
            extraction_id="xtr_fixture",
            source_chunk_id="chk_fixture",
            chunk_text="LightRAG uses a knowledge graph.",
            content=json.dumps(payload),
            finish_reason="stop",
        )

    assert captured.value.kind is expected_kind
    assert captured.value.repairable


def test_duplicate_json_keys_are_rejected() -> None:
    with pytest.raises(ExtractionValidationError) as captured:
        validate_completion(
            extraction_id="xtr_fixture",
            source_chunk_id="chk_fixture",
            chunk_text="text",
            content='{"entities":[],"entities":[],"relations":[]}',
            finish_reason="stop",
        )

    assert captured.value.kind is ValidationFailureKind.INVALID_JSON


@pytest.mark.parametrize(
    ("finish_reason", "kind", "repairable", "retryable"),
    [
        ("length", ValidationFailureKind.TRUNCATED, True, False),
        ("content_filter", ValidationFailureKind.CONTENT_FILTERED, False, False),
        (
            "insufficient_system_resource",
            ValidationFailureKind.RETRYABLE_PROVIDER_FINISH,
            False,
            True,
        ),
        ("tool_calls", ValidationFailureKind.UNEXPECTED_FINISH, True, False),
    ],
)
def test_finish_reasons_are_classified(
    finish_reason: str,
    kind: ValidationFailureKind,
    repairable: bool,
    retryable: bool,
) -> None:
    with pytest.raises(ExtractionValidationError) as captured:
        validate_completion(
            extraction_id="xtr_fixture",
            source_chunk_id="chk_fixture",
            chunk_text="text",
            content='{"entities":[],"relations":[]}',
            finish_reason=finish_reason,
        )

    assert captured.value.kind is kind
    assert captured.value.repairable is repairable
    assert captured.value.retryable_provider is retryable


def test_extraction_and_graph_hashes_have_separate_boundaries() -> None:
    extraction = ExtractionConfig()
    changed_extraction = ExtractionConfig(model="deepseek-v4-pro")
    graph = GraphConfig(extraction_config_hash=extraction.config_hash)
    changed_graph = GraphConfig(
        extraction_config_hash=extraction.config_hash,
        entity_normalizer_version="2",
    )

    assert extraction.config_hash != changed_extraction.config_hash
    assert graph.config_hash != changed_graph.config_hash
    assert graph.extraction_config_hash == changed_graph.extraction_config_hash
