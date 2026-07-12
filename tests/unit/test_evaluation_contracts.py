from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from hybrid_rag.evaluation import (
    CostDisclosure,
    CostStatus,
    EvaluationBenchmark,
    EvaluationCategory,
    EvaluationOptions,
    fixture_benchmark_path,
    load_benchmark,
)
from hybrid_rag.retrieval.models import RetrievalMode


def test_fixed_fixture_has_a_versioned_balanced_question_set() -> None:
    benchmark = load_benchmark(fixture_benchmark_path())

    assert benchmark.schema_version == "1"
    assert 20 <= len(benchmark.cases) <= 30
    assert {case.category for case in benchmark.cases} == set(EvaluationCategory)
    assert len({case.id for case in benchmark.cases}) == len(benchmark.cases)
    assert all(case.expected_evidence for case in benchmark.cases)
    assert benchmark.expected_source_corpus_hash == (
        "8ef142f6076296e1c93a5de0883d3e800fa34cb0831a1a685c6d0e3ece328761"
    )


def test_fixture_can_round_trip_as_strict_json() -> None:
    benchmark = load_benchmark(fixture_benchmark_path())

    restored = type(benchmark).model_validate_json(benchmark.model_dump_json())

    assert restored == benchmark
    assert json.loads(benchmark.model_dump_json())["id"] == "fixture-rag-v1"


def test_evaluation_options_require_naive_and_hybrid() -> None:
    with pytest.raises(ValidationError, match="must include naive and hybrid"):
        EvaluationOptions(modes=(RetrievalMode.NAIVE, RetrievalMode.LOCAL))

    options = EvaluationOptions(case_ids=("fact-naive-isolated-chunks",))

    assert options.modes == (RetrievalMode.NAIVE, RetrievalMode.HYBRID)
    assert len(options.config_hash) == 64


def test_benchmark_requires_a_valid_expected_source_corpus_hash() -> None:
    benchmark = load_benchmark(fixture_benchmark_path())
    missing = benchmark.model_dump(mode="json")
    missing.pop("expected_source_corpus_hash")

    with pytest.raises(ValidationError, match="expected_source_corpus_hash"):
        EvaluationBenchmark.model_validate(missing)

    with pytest.raises(ValidationError, match="expected_source_corpus_hash"):
        EvaluationBenchmark.model_validate(
            benchmark.model_dump(mode="json") | {"expected_source_corpus_hash": "not-a-hash"}
        )


def test_cost_disclosure_does_not_invent_external_prices() -> None:
    offline = CostDisclosure.offline()
    unknown = CostDisclosure.unknown_external_judge()

    assert offline.status is CostStatus.NOT_APPLICABLE
    assert offline.cost_usd == 0.0
    assert unknown.status is CostStatus.UNKNOWN
    assert unknown.cost_usd is None
    with pytest.raises(ValidationError, match="requires amount and price_assumption"):
        CostDisclosure(
            status=CostStatus.VERIFIED,
            retrieval_model_calls=0,
            judge_model_calls=20,
            cost_usd=None,
            price_assumption=None,
        )
