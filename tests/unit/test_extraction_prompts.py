from __future__ import annotations

import json
from typing import Any

from hybrid_rag.extraction.prompts import build_extraction_messages, build_repair_messages
from hybrid_rag.extraction.schemas import (
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_SCHEMA_VERSION,
    REPAIR_PROMPT_VERSION,
    ExtractionConfig,
)


def _schema_from_system_prompt(system_prompt: str) -> dict[str, Any]:
    schema_prefix = "JSON_SCHEMA:\n"
    schema_start = system_prompt.index(schema_prefix) + len(schema_prefix)
    schema_end = system_prompt.index("\n\nEXAMPLE_JSON:", schema_start)
    return json.loads(system_prompt[schema_start:schema_end])


def test_system_type_and_other_fallback_are_explicit_in_both_prompts() -> None:
    extraction_messages = build_extraction_messages("A RAG system serves users.")
    repair_messages = build_repair_messages(
        "A RAG system serves users.",
        '{"entities":[]}',
        ["entities.0.entity_type is invalid"],
    )

    extraction_system = extraction_messages[0]["content"]
    repair_system = repair_messages[0]["content"]
    assert extraction_system == repair_system

    instructions, _separator, _schema_and_example = extraction_system.partition(
        "\n\nJSON_SCHEMA:\n"
    )
    assert "SYSTEM" in instructions
    assert "OTHER" in instructions

    schema = _schema_from_system_prompt(extraction_system)
    entity_type_schema = schema["$defs"]["EntityType"]
    assert "SYSTEM" in entity_type_schema["enum"]

    repair_instructions = repair_messages[1]["content"]
    assert "use SYSTEM" in repair_instructions
    assert "use OTHER" in repair_instructions


def test_system_schema_change_bumps_prompt_and_extraction_config_versions() -> None:
    config = ExtractionConfig()

    assert EXTRACTION_SCHEMA_VERSION == "2"
    assert EXTRACTION_PROMPT_VERSION == "2"
    assert REPAIR_PROMPT_VERSION == "2"
    assert config.schema_version == EXTRACTION_SCHEMA_VERSION
    assert config.prompt_version == EXTRACTION_PROMPT_VERSION
    assert config.repair_prompt_version == REPAIR_PROMPT_VERSION
