from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hybrid_rag.evaluation import (
    BenchmarkCase,
    BlindJudgment,
    BlindLabel,
    BlindWinner,
    CostDisclosure,
    CostStatus,
    EvaluationBenchmark,
    EvaluationCategory,
    EvaluationOptions,
    EvaluationRunner,
    ExpectedEvidence,
    JudgeProvenance,
    render_markdown,
    write_json,
    write_markdown,
)
from hybrid_rag.retrieval.models import (
    ContextItem,
    RetrievalMode,
    RetrievalResult,
    RetrievalTrace,
    RouteTrace,
)


class FakeRetrievalService:
    def __init__(self, *, provider: str = "hash") -> None:
        self.persist_values: list[bool] = []
        self.profile_refs: list[str | None] = []
        self.model_infos: list[object] = []
        self.retrieve_calls = 0
        self._profile = SimpleNamespace(
            id="idx-fixture",
            config_hash="fixture-hash",
            provider=provider,
            model="fixture-model",
            dimensions=32,
            schema_version="fixture-schema",
            source_corpus_hash="a" * 64,
            source_graph_run_id="gbr-fixture",
            metadata={
                "corpus_content_hash": "c" * 64,
                "graph_corpus_hash": "b" * 64,
            },
        )

    def resolve_profile(self, profile_ref: str | None = None) -> object:
        self.profile_refs.append(profile_ref)
        return self._profile

    def retrieve(self, question: str, **kwargs: object) -> RetrievalResult:
        mode = kwargs["mode"]
        assert isinstance(mode, RetrievalMode)
        persist = kwargs["persist"]
        assert isinstance(persist, bool)
        self.persist_values.append(persist)
        profile_ref = kwargs["profile_ref"]
        assert profile_ref == "idx-fixture"
        self.profile_refs.append(profile_ref)
        self.model_infos.append(kwargs["model_info"])
        self.retrieve_calls += 1
        case_number = question.rsplit(" ", maxsplit=1)[-1]
        first = f"evidence {case_number} one"
        second = f"evidence {case_number} two"
        texts = (first,) if mode is RetrievalMode.NAIVE else (first, second)
        context_items = tuple(
            ContextItem(
                citation_id=f"chk-{case_number}-{index}",
                chunk_id=f"chk-{case_number}-{index}",
                document_id="doc-fixture",
                document_title="Fixture",
                text=text,
                token_count=3,
                score=1.0,
            )
            for index, text in enumerate(texts, start=1)
        )
        trace = RetrievalTrace(
            profile_id="idx-fixture",
            index_config_hash="fixture-hash",
            query=question,
            expanded_query=question,
            mode=mode,
            routes={mode.value: RouteTrace(route=mode, candidate_count=len(context_items))},
            context_items=context_items,
            context_token_budget=64,
            context_tokens=sum(item.token_count for item in context_items),
        )
        return RetrievalResult(
            trace_id=f"rtr_fixture_{self.retrieve_calls}",
            profile_id="idx-fixture",
            mode=mode,
            query=question,
            context="\n".join(item.text for item in context_items),
            context_tokens=trace.context_tokens,
            context_items=context_items,
            trace=trace,
        )


class BrokenJudge:
    protocol = "broken-test-judge"

    def judge(self, _: object) -> BlindJudgment:
        raise RuntimeError("network unavailable")


class SelectAJudge:
    protocol = "select-a-test-judge"
    provenance = JudgeProvenance(
        provider="test-external",
        protocol=protocol,
        external=True,
        model="test-judge-model",
        base_url="https://judge.example.test",
        max_output_tokens=321,
        timeout_seconds=12.5,
    )

    def judge(self, _: object) -> BlindJudgment:
        return BlindJudgment(winner=BlindWinner.A, rationale="test judge selected blind A")


def test_runner_pins_profiles_persists_replayable_traces_and_serializes_reports(tmp_path) -> None:
    service = FakeRetrievalService()
    runner = EvaluationRunner(service)  # type: ignore[arg-type]
    options = EvaluationOptions(case_ids=("case-01", "case-02"), context_token_budget=64)

    report = runner.run(_benchmark(), options=options)

    assert len(report.evaluations) == 4
    assert report.comparison_summary.hybrid_wins == 2
    assert report.comparison_summary.naive_wins == 0
    assert all(value is True for value in service.persist_values)
    assert service.profile_refs[0] is None
    assert all(value == "idx-fixture" for value in service.profile_refs[1:])
    naive = next(
        item
        for item in report.evaluations
        if item.case_id == "case-01" and item.mode is RetrievalMode.NAIVE
    )
    hybrid = next(
        item
        for item in report.evaluations
        if item.case_id == "case-01" and item.mode is RetrievalMode.HYBRID
    )
    assert naive.evidence_hit_rate == 0.5
    assert hybrid.evidence_hit_rate == 1.0
    assert hybrid.citation_grounded_faithfulness
    assert report.cost_disclosure.cost_cny == 0.0
    assert report.run.index_provenance.profile_id == "idx-fixture"
    assert report.run.index_provenance.corpus_content_hash == "c" * 64
    assert report.run.index_provenance.source_corpus_hash == "a" * 64
    assert report.run.execution_id.startswith("evx_")
    assert all(item.retrieval_trace_id.startswith("rtr_") for item in report.evaluations)
    assert all(
        info["evaluation_execution_id"] == report.run.execution_id for info in service.model_infos
    )
    assert type(report).model_validate_json(report.to_json()) == report

    json_path = write_json(report, tmp_path / "report.json")
    markdown_path = write_markdown(report, tmp_path / "report.md")
    assert json.loads(json_path.read_text(encoding="utf-8"))["run"]["id"] == report.run.id
    assert "# Retrieval evaluation" in markdown_path.read_text(encoding="utf-8")
    assert "Hybrid wins: 2" in render_markdown(report)
    assert report.run.execution_id in render_markdown(report)


def test_runner_falls_back_after_a_custom_blind_judge_failure() -> None:
    report = EvaluationRunner(FakeRetrievalService(), judge=BrokenJudge()).run(  # type: ignore[arg-type]
        _benchmark(),
        options=EvaluationOptions(case_ids=("case-01",)),
    )

    assert report.cost_disclosure.cost_cny is None
    assert report.cost_disclosure.status.value == "unknown"
    assert all(judgment.used_fallback for judgment in report.pairwise_judgments)
    assert "network unavailable" in report.pairwise_judgments[0].fallback_reason


def test_runner_rejects_a_benchmark_for_a_different_pinned_source_snapshot() -> None:
    service = FakeRetrievalService()
    benchmark = _benchmark().model_copy(update={"expected_source_corpus_hash": "d" * 64})

    with pytest.raises(ValueError, match="expected_source_corpus_hash"):
        EvaluationRunner(service).run(benchmark, options=EvaluationOptions(case_ids=("case-01",)))  # type: ignore[arg-type]

    assert service.retrieve_calls == 0


def test_runner_marks_unknown_legacy_embedding_cost_without_full_disclosure() -> None:
    report = EvaluationRunner(FakeRetrievalService(provider="legacy-external")).run(  # type: ignore[arg-type]
        _benchmark(),
        options=EvaluationOptions(case_ids=("case-01",)),
        cost_disclosure=CostDisclosure.offline(),
    )

    assert report.cost_disclosure.status is CostStatus.UNKNOWN
    assert report.cost_disclosure.cost_cny is None
    assert report.cost_disclosure.retrieval_model_calls == 2


def test_runner_treats_local_bge_embedding_as_no_embedding_api_cost() -> None:
    report = EvaluationRunner(FakeRetrievalService(provider="flagembedding")).run(  # type: ignore[arg-type]
        _benchmark(),
        options=EvaluationOptions(case_ids=("case-01",)),
    )

    assert report.cost_disclosure.status is CostStatus.NOT_APPLICABLE
    assert report.cost_disclosure.retrieval_model_calls == 0
    assert report.cost_disclosure.cost_cny == 0.0


def test_runner_accepts_only_a_verified_legacy_embedding_cost_disclosure() -> None:
    supplied = CostDisclosure(
        status=CostStatus.VERIFIED,
        retrieval_model_calls=2,
        judge_model_calls=0,
        cost_cny=0.02,
        price_assumption="provider invoice reconciled for the two query embeddings",
    )
    report = EvaluationRunner(FakeRetrievalService(provider="legacy-external")).run(  # type: ignore[arg-type]
        _benchmark(),
        options=EvaluationOptions(case_ids=("case-01",)),
        cost_disclosure=supplied,
    )

    assert report.cost_disclosure == supplied


def test_runner_invalidates_a_supplied_cost_after_external_judge_fallback() -> None:
    supplied = CostDisclosure(
        status=CostStatus.ESTIMATED,
        retrieval_model_calls=0,
        judge_model_calls=1,
        cost_cny=0.01,
        price_assumption="test-only fully specified price",
    )
    report = EvaluationRunner(FakeRetrievalService(), judge=BrokenJudge()).run(  # type: ignore[arg-type]
        _benchmark(),
        options=EvaluationOptions(case_ids=("case-01",)),
        cost_disclosure=supplied,
    )

    assert report.cost_disclosure.status is CostStatus.UNKNOWN
    assert report.cost_disclosure.cost_cny is None
    assert "fallback" in (report.cost_disclosure.price_assumption or "")


def test_runner_maps_a_blind_custom_judge_result_back_to_its_hidden_mode() -> None:
    report = EvaluationRunner(FakeRetrievalService(), judge=SelectAJudge()).run(  # type: ignore[arg-type]
        _benchmark(),
        options=EvaluationOptions(case_ids=("case-01",)),
    )

    judgment = report.pairwise_judgments[0]
    assert not judgment.used_fallback
    assert judgment.winner_mode is judgment.label_to_mode[BlindLabel.A]
    assert report.judge_provenance == SelectAJudge.provenance


def _benchmark() -> EvaluationBenchmark:
    categories = tuple(EvaluationCategory)
    cases = tuple(
        BenchmarkCase(
            id=f"case-{index:02d}",
            question=f"Question {index:02d}",
            category=categories[(index - 1) % len(categories)],
            expected_evidence=(
                ExpectedEvidence(id=f"evidence-{index:02d}-one", text=f"evidence {index:02d} one"),
                ExpectedEvidence(id=f"evidence-{index:02d}-two", text=f"evidence {index:02d} two"),
            ),
            reference_answer=f"Reference {index:02d}",
        )
        for index in range(1, 21)
    )
    return EvaluationBenchmark(
        id="runner-fixture-v1",
        title="Runner fixture",
        corpus_id="runner-corpus-v1",
        expected_source_corpus_hash="c" * 64,
        description="A deterministic unit-test benchmark.",
        cases=cases,
    )
