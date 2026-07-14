from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from hybrid_rag.deepseek_costs import DeepSeekCostStatus, DeepSeekCostSummary, DeepSeekUsage
from hybrid_rag.evaluation import ragas_runner
from hybrid_rag.retrieval.models import RetrievalMode
from hybrid_rag.retrieval.query import QueryClient
from hybrid_rag.retrieval.service import RetrievalOptions


class FakeRetrievalService:
    def __init__(
        self,
        *,
        corpus_content_hash: str = "c" * 64,
        cost_summaries: tuple[DeepSeekCostSummary | None, ...] = (),
    ) -> None:
        self.resolve_calls: list[str | None] = []
        self.ask_calls: list[dict[str, object]] = []
        self._cost_summaries = cost_summaries
        self._profile = SimpleNamespace(
            id="idx-fixture",
            config_hash="fixture-config-hash",
            provider="flagembedding",
            model="fixture-model",
            dimensions=1024,
            schema_version="fixture-schema",
            source_corpus_hash="a" * 64,
            source_graph_run_id="gbr-fixture",
            metadata={"corpus_content_hash": corpus_content_hash},
        )

    def resolve_profile(self, profile_ref: str | None = None) -> object:
        self.resolve_calls.append(profile_ref)
        return self._profile

    async def ask(self, question: str, **kwargs: object) -> object:
        self.ask_calls.append({"question": question, **kwargs})
        call_number = len(self.ask_calls)
        cost = (
            self._cost_summaries[call_number - 1]
            if call_number <= len(self._cost_summaries)
            else None
        )
        return SimpleNamespace(
            retrieval=SimpleNamespace(
                profile_id="idx-fixture",
                trace_id=f"rtr_fixture_{call_number}",
                context_items=(SimpleNamespace(text=f"context {call_number}"),),
                trace=SimpleNamespace(deepseek_cost=cost),
            ),
            answer=SimpleNamespace(answer=f"answer {call_number}"),
        )


def test_ragas_runner_pins_one_profile_and_serializes_provenance_and_trace_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    testset_path = _write_testset(tmp_path)
    service = FakeRetrievalService()
    monkeypatch.setattr(ragas_runner, "_sample", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(ragas_runner, "_evaluate", _fake_evaluate)

    report = asyncio.run(
        ragas_runner.RagasEvaluationRunner(  # type: ignore[arg-type]
            service, cast(QueryClient, object())
        ).run(
            testset_path,
            modes=(RetrievalMode.NAIVE, RetrievalMode.MIX),
            retrieval_options=RetrievalOptions(),
            profile_ref="requested-profile",
            judge_model="judge-fixture",
            judge_api_key="secret",
            judge_base_url="https://example.test/v1",
            judge_max_output_tokens=768,
            judge_timeout_seconds=30.0,
            query_client_provenance={"answer_model": "answer-fixture"},
        )
    )

    assert service.resolve_calls == ["requested-profile"]
    assert [call["profile_ref"] for call in service.ask_calls] == ["idx-fixture"] * 4
    assert report.provenance == {
        "testset": {
            "schema_version": ragas_runner.RAGAS_TESTSET_SCHEMA_VERSION,
            "corpus_content_hash": "c" * 64,
            "file_sha256": _file_sha256(testset_path),
            "case_count": 2,
        },
        "profile": {
            "id": "idx-fixture",
            "config_hash": "fixture-config-hash",
            "provider": "flagembedding",
            "model": "fixture-model",
            "dimensions": 1024,
            "schema_version": "fixture-schema",
            "source_corpus_hash": "a" * 64,
            "source_graph_run_id": "gbr-fixture",
            "corpus_content_hash": "c" * 64,
        },
        "runtime": {
            "modes": ["naive", "mix"],
            "retrieval_options": {
                "top_k": 8,
                "candidate_multiplier": 4,
                "context_token_budget": 2400,
                "graph_max_hops": 2,
                "naive_weight": 1.0,
                "local_weight": 1.0,
                "global_weight": 1.0,
                "naive_dense_weight": 1.0,
                "naive_bm25_weight": 1.0,
                "bm25_k1": 1.5,
                "bm25_b": 0.75,
                "reranker_provider": "none",
                "reranker_model": "BAAI/bge-reranker-v2-m3",
                "reranker_use_fp16": False,
                "rerank_candidate_multiplier": 4,
            },
            "query_client": {"answer_model": "answer-fixture"},
            "judge": {
                "model": "judge-fixture",
                "base_url": "https://example.test/v1",
                "max_output_tokens": 768,
                "timeout_seconds": 30.0,
                "max_retries": 0,
                "temperature": 0.0,
            },
        },
    }
    assert report.modes["naive"]["means"] == {"faithfulness": 1.0}
    assert report.modes["mix"]["retrieval_trace_ids"] == [
        "rtr_fixture_3",
        "rtr_fixture_4",
    ]
    assert report.modes["mix"]["cases"] == [
        {
            "case_index": 1,
            "retrieval_trace_id": "rtr_fixture_3",
            "scores": {"faithfulness": 1.0},
        },
        {
            "case_index": 2,
            "retrieval_trace_id": "rtr_fixture_4",
            "scores": {"faithfulness": 1.0},
        },
    ]
    assert report.cost["retrieval"]["status"] == "unknown"
    assert report.cost["judge"]["status"] == "unknown"
    assert report.cost["judge"]["cost_cny"] is None
    assert report.cost["total"]["status"] == "unknown"
    assert report.cost["total"]["cost_cny"] is None
    json.dumps(report.as_dict())


def test_ragas_runner_rejects_an_unprovenanced_bare_array(tmp_path: Path) -> None:
    testset_path = tmp_path / "bare.json"
    testset_path.write_text("[]", encoding="utf-8")
    service = FakeRetrievalService()

    with pytest.raises(ValueError, match="bare JSON arrays are unsupported"):
        asyncio.run(
            ragas_runner.RagasEvaluationRunner(  # type: ignore[arg-type]
                service, cast(QueryClient, object())
            ).run(
                testset_path,
                modes=(RetrievalMode.MIX,),
                retrieval_options=RetrievalOptions(),
                profile_ref=None,
                judge_model="judge-fixture",
                judge_api_key="secret",
                judge_base_url="https://example.test/v1",
            )
        )

    assert service.resolve_calls == []
    assert service.ask_calls == []


def test_ragas_testset_requires_a_lowercase_corpus_hash(tmp_path: Path) -> None:
    path = tmp_path / "uppercase-hash.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": ragas_runner.RAGAS_TESTSET_SCHEMA_VERSION,
                "corpus_content_hash": "C" * 64,
                "cases": [
                    {
                        "user_input": "Question",
                        "reference": "Reference",
                        "reference_contexts": ["Context"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="64-character hexadecimal digest"):
        ragas_runner._load_testset(path)


def test_ragas_runner_checks_the_testset_hash_before_any_retrieval(tmp_path: Path) -> None:
    testset_path = _write_testset(tmp_path)
    service = FakeRetrievalService(corpus_content_hash="d" * 64)

    with pytest.raises(ValueError, match="corpus_content_hash does not match"):
        asyncio.run(
            ragas_runner.RagasEvaluationRunner(  # type: ignore[arg-type]
                service, cast(QueryClient, object())
            ).run(
                testset_path,
                modes=(RetrievalMode.MIX,),
                retrieval_options=RetrievalOptions(),
                profile_ref=None,
                judge_model="judge-fixture",
                judge_api_key="secret",
                judge_base_url="https://example.test/v1",
            )
        )

    assert service.resolve_calls == [None]
    assert service.ask_calls == []


def test_ragas_runner_keeps_the_latest_observed_retrieval_cost_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    testset_path = _write_testset(tmp_path)
    service = FakeRetrievalService(
        cost_summaries=(
            _estimated_cost(cost_cny=0.01, calls=2),
            _estimated_cost(cost_cny=0.03, calls=4),
        )
    )
    monkeypatch.setattr(ragas_runner, "_sample", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(ragas_runner, "_evaluate", _fake_evaluate)

    report = asyncio.run(
        ragas_runner.RagasEvaluationRunner(  # type: ignore[arg-type]
            service, cast(QueryClient, object())
        ).run(
            testset_path,
            modes=(RetrievalMode.MIX,),
            retrieval_options=RetrievalOptions(),
            profile_ref=None,
            judge_model="judge-fixture",
            judge_api_key="secret",
            judge_base_url="https://example.test/v1",
        )
    )

    retrieval = report.cost["retrieval"]
    assert retrieval["status"] == "estimated"
    assert retrieval["cost_cny"] == 0.03
    assert retrieval["aggregation"] == "latest_trace_snapshot"
    assert retrieval["missing_trace_ids"] == []
    assert retrieval["observations"] == [
        {
            "retrieval_trace_id": "rtr_fixture_1",
            "cost": _estimated_cost(cost_cny=0.01, calls=2).model_dump(mode="json"),
        },
        {
            "retrieval_trace_id": "rtr_fixture_2",
            "cost": _estimated_cost(cost_cny=0.03, calls=4).model_dump(mode="json"),
        },
    ]
    assert report.cost["judge"]["status"] == "unknown"
    assert report.cost["total"]["status"] == "unknown"
    json.dumps(report.as_dict())


def _write_testset(tmp_path: Path) -> Path:
    path = tmp_path / "ragas-testset.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": ragas_runner.RAGAS_TESTSET_SCHEMA_VERSION,
                "corpus_content_hash": "c" * 64,
                "cases": [
                    {
                        "user_input": "Question one",
                        "reference": "Reference one",
                        "reference_contexts": ["Context one"],
                    },
                    {
                        "user_input": "Question two",
                        "reference": "Reference two",
                        "reference_contexts": ["Context two"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _fake_evaluate(
    samples: list[object],
    *,
    judge_model: str,
    judge_api_key: str,
    judge_base_url: str,
    judge_max_output_tokens: int,
    judge_timeout_seconds: float,
) -> dict[str, object]:
    assert judge_model == "judge-fixture"
    assert judge_api_key == "secret"
    assert judge_base_url == "https://example.test/v1"
    assert judge_max_output_tokens > 0
    assert judge_timeout_seconds > 0
    return {
        "scores": [{"faithfulness": 1.0} for _ in samples],
        "means": {"faithfulness": 1.0},
    }


def _estimated_cost(*, cost_cny: float, calls: int) -> DeepSeekCostSummary:
    return DeepSeekCostSummary(
        status=DeepSeekCostStatus.ESTIMATED,
        cost_cny=cost_cny,
        usage=(
            DeepSeekUsage(
                operation="answer",
                model="deepseek-v4-flash",
                calls=calls,
                prompt_tokens=calls * 10,
                cache_hit_tokens=0,
                cache_miss_tokens=calls * 10,
                completion_tokens=calls * 5,
                cache_breakdown_complete=True,
            ),
        ),
        price_assumption="test pricing",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
