from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import typer

from hybrid_rag.cli import _build_extraction_config
from hybrid_rag.extraction.schemas import ExtractionConfig


def _resume_dependencies(persisted: dict[str, object]) -> tuple[object, object]:
    database = SimpleNamespace(session_factory=lambda: nullcontext(object()))
    repository = SimpleNamespace(
        get_run=lambda _session, _run_id: SimpleNamespace(
            report={"extraction_config": persisted}
        )
    )
    return database, repository


@pytest.mark.parametrize(
    "field",
    ("schema_version", "prompt_version", "repair_prompt_version"),
)
def test_resume_rejects_an_extraction_contract_from_an_older_runtime(field: str) -> None:
    persisted = ExtractionConfig(
        base_url="https://example.test",
        model="scripted-resume",
    ).model_dump(mode="json")
    persisted[field] = "1"
    database, repository = _resume_dependencies(persisted)

    with pytest.raises(typer.BadParameter, match=field):
        _build_extraction_config(
            database,  # type: ignore[arg-type]
            repository,  # type: ignore[arg-type]
            resume_run_id="gbr_stale_contract",
            base_url="https://ignored.test",
            model="ignored",
            max_output_tokens=1,
            max_attempts=1,
        )


def test_resume_reuses_an_extraction_contract_from_the_current_runtime() -> None:
    expected = ExtractionConfig(
        base_url="https://example.test",
        model="scripted-resume",
    )
    database, repository = _resume_dependencies(expected.model_dump(mode="json"))

    actual = _build_extraction_config(
        database,  # type: ignore[arg-type]
        repository,  # type: ignore[arg-type]
        resume_run_id="gbr_current_contract",
        base_url="https://ignored.test",
        model="ignored",
        max_output_tokens=1,
        max_attempts=1,
    )

    assert actual == expected
