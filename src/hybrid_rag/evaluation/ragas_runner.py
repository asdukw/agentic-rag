"""Ragas-backed evaluation over the project's existing retrieval and answer pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

from hybrid_rag.deepseek_costs import DeepSeekCostStatus, DeepSeekCostSummary
from hybrid_rag.retrieval.models import RetrievalMode
from hybrid_rag.retrieval.query import QueryClient
from hybrid_rag.retrieval.service import RetrievalOptions, RetrievalService
from hybrid_rag.storage.retrieval_repository import StoredIndexProfile

RAGAS_TESTSET_SCHEMA_VERSION = "1"
_SHA256_HEX = re.compile(r"^[a-f0-9]{64}$")


class RagasCase(TypedDict):
    user_input: str
    reference: str
    reference_contexts: list[str]


@dataclass(frozen=True, slots=True)
class RagasTestset:
    """A versioned generated test set bound to one corpus content hash."""

    schema_version: str
    corpus_content_hash: str
    file_sha256: str
    cases: tuple[RagasCase, ...]


@dataclass(frozen=True, slots=True)
class RagasEvaluationReport:
    """Serializable Ragas scores and per-sample details grouped by retrieval mode."""

    testset_path: str
    provenance: dict[str, object]
    cost: dict[str, dict[str, object]]
    modes: dict[str, dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return {
            "testset_path": self.testset_path,
            "provenance": self.provenance,
            "cost": self.cost,
            "modes": self.modes,
        }


class RagasEvaluationRunner:
    """Run generated Ragas cases through ``RetrievalService.ask`` then score them."""

    def __init__(self, retrieval_service: RetrievalService, query_client: QueryClient) -> None:
        self.retrieval_service = retrieval_service
        self.query_client = query_client

    async def run(
        self,
        testset_path: Path,
        *,
        modes: Sequence[RetrievalMode],
        retrieval_options: RetrievalOptions,
        profile_ref: str | None,
        judge_model: str,
        judge_api_key: str,
        judge_base_url: str,
        judge_max_output_tokens: int = 1_024,
        judge_timeout_seconds: float = 180.0,
        query_client_provenance: dict[str, object] | None = None,
    ) -> RagasEvaluationReport:
        if judge_max_output_tokens < 1:
            raise ValueError("Ragas judge_max_output_tokens must be positive")
        if judge_timeout_seconds <= 0:
            raise ValueError("Ragas judge_timeout_seconds must be positive")
        testset = _load_testset(testset_path)
        profile = self.retrieval_service.resolve_profile(profile_ref)
        profile_corpus_content_hash = _profile_corpus_content_hash(profile)
        if testset.corpus_content_hash != profile_corpus_content_hash:
            raise ValueError(
                "Ragas test set corpus_content_hash does not match the pinned index profile "
                f"({testset.corpus_content_hash} != {profile_corpus_content_hash})"
            )

        output: dict[str, dict[str, object]] = {}
        cost_observations: list[RetrievalCostObservation] = []
        for mode in modes:
            samples: list[object] = []
            case_details: list[dict[str, object]] = []
            for case_index, case in enumerate(testset.cases, start=1):
                answer = await self.retrieval_service.ask(
                    case["user_input"],
                    query_client=self.query_client,
                    mode=mode,
                    options=retrieval_options,
                    profile_ref=profile.id,
                )
                if answer.retrieval.profile_id != profile.id:
                    raise RuntimeError(
                        "retrieval returned a profile different from the profile pinned "
                        "for Ragas evaluation"
                    )
                trace_id = answer.retrieval.trace_id
                if not isinstance(trace_id, str) or not trace_id:
                    raise RuntimeError(
                        "Ragas evaluation requires every retrieval result to have "
                        "a persisted trace ID"
                    )
                samples.append(
                    _sample(
                        case,
                        response=answer.answer.answer,
                        retrieved_contexts=[item.text for item in answer.retrieval.context_items],
                    )
                )
                case_details.append(
                    {
                        "case_index": case_index,
                        "retrieval_trace_id": trace_id,
                    }
                )
                cost_observations.append(
                    RetrievalCostObservation(
                        retrieval_trace_id=trace_id,
                        summary=answer.retrieval.trace.deepseek_cost,
                    )
                )
            result = await asyncio.to_thread(
                _evaluate,
                samples,
                judge_model=judge_model,
                judge_api_key=judge_api_key,
                judge_base_url=judge_base_url,
                judge_max_output_tokens=judge_max_output_tokens,
                judge_timeout_seconds=judge_timeout_seconds,
            )
            output[mode.value] = _mode_report(result, case_details)
        return RagasEvaluationReport(
            testset_path=str(testset_path),
            provenance={
                "testset": {
                    "schema_version": testset.schema_version,
                    "corpus_content_hash": testset.corpus_content_hash,
                    "file_sha256": testset.file_sha256,
                    "case_count": len(testset.cases),
                },
                "profile": _profile_provenance(profile, profile_corpus_content_hash),
                "runtime": {
                    "modes": [mode.value for mode in modes],
                    "retrieval_options": asdict(retrieval_options),
                    "query_client": dict(query_client_provenance or {}),
                    "judge": {
                        "model": judge_model,
                        "base_url": judge_base_url,
                        "max_output_tokens": judge_max_output_tokens,
                        "timeout_seconds": judge_timeout_seconds,
                        "max_retries": 0,
                        "temperature": 0.0,
                    },
                },
            },
            cost=_cost_report(cost_observations),
            modes=output,
        )


def _load_testset(path: Path) -> RagasTestset:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read Ragas test set {path}: {error}") from error
    if isinstance(value, list):
        raise ValueError(
            "Ragas test set must use a JSON envelope with schema_version, corpus_content_hash, "
            "and cases; bare JSON arrays are unsupported"
        )
    if not isinstance(value, dict):
        raise ValueError("Ragas test set must be a JSON object envelope")

    schema_version = value.get("schema_version")
    if schema_version != RAGAS_TESTSET_SCHEMA_VERSION:
        raise ValueError(
            "unsupported Ragas test set schema_version "
            f"{schema_version!r}; expected {RAGAS_TESTSET_SCHEMA_VERSION!r}"
        )
    corpus_content_hash = _corpus_content_hash(
        value.get("corpus_content_hash"),
        field="Ragas test set corpus_content_hash",
    )
    values = value.get("cases")
    if not isinstance(values, list) or not values:
        raise ValueError("Ragas test set envelope cases must be a non-empty JSON array")

    cases: list[RagasCase] = []
    for index, item in enumerate(values, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Ragas test set item {index} must be an object")
        question = item.get("user_input")
        reference = item.get("reference")
        contexts = item.get("reference_contexts")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Ragas test set item {index} has no user_input")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"Ragas test set item {index} has no reference")
        cases.append(
            {
                "user_input": question,
                "reference": reference,
                "reference_contexts": _string_list(
                    contexts,
                    field="reference_contexts",
                    index=index,
                ),
            }
        )
    return RagasTestset(
        schema_version=schema_version,
        corpus_content_hash=corpus_content_hash,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        cases=tuple(cases),
    )


def _corpus_content_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a 64-character hexadecimal digest")
    normalized = value.strip()
    if not _SHA256_HEX.fullmatch(normalized):
        raise ValueError(f"{field} must be a 64-character hexadecimal digest")
    return normalized


def _profile_corpus_content_hash(profile: StoredIndexProfile) -> str:
    return _corpus_content_hash(
        profile.metadata.get("corpus_content_hash"),
        field="pinned index profile metadata.corpus_content_hash",
    )


def _profile_provenance(
    profile: StoredIndexProfile,
    corpus_content_hash: str,
) -> dict[str, object]:
    return {
        "id": profile.id,
        "config_hash": profile.config_hash,
        "provider": profile.provider,
        "model": profile.model,
        "dimensions": profile.dimensions,
        "schema_version": profile.schema_version,
        "source_corpus_hash": profile.source_corpus_hash,
        "source_graph_run_id": profile.source_graph_run_id,
        "corpus_content_hash": corpus_content_hash,
    }


def _mode_report(
    result: dict[str, object],
    case_details: list[dict[str, object]],
) -> dict[str, object]:
    """Attach replayable retrieval trace IDs to the Ragas result in case order."""

    scores = result.get("scores")
    if not isinstance(scores, list):
        raise RuntimeError("Ragas evaluation result did not contain per-case scores")
    if len(scores) != len(case_details):
        raise RuntimeError(
            "Ragas evaluation returned a score count different from the test set size"
        )

    cases = [
        {
            **detail,
            "scores": score,
        }
        for detail, score in zip(case_details, scores, strict=True)
    ]
    return {
        **result,
        "retrieval_trace_ids": [detail["retrieval_trace_id"] for detail in case_details],
        "cases": cases,
    }


@dataclass(frozen=True, slots=True)
class RetrievalCostObservation:
    """The observed DeepSeek cost snapshot retained on one retrieval trace."""

    retrieval_trace_id: str
    summary: DeepSeekCostSummary | None


def _cost_report(observations: Sequence[RetrievalCostObservation]) -> dict[str, dict[str, object]]:
    """Report retrieval observations without inventing Ragas judge or total costs.

    The project query client exposes cumulative usage snapshots, so we use the
    final complete trace snapshot instead of summing each trace and risking
    double-counting. A missing snapshot makes retrieval cost unknown.
    """

    serialized_observations = [
        {
            "retrieval_trace_id": observation.retrieval_trace_id,
            "cost": (
                observation.summary.model_dump(mode="json")
                if observation.summary is not None
                else None
            ),
        }
        for observation in observations
    ]
    missing = [
        observation.retrieval_trace_id
        for observation in observations
        if observation.summary is None
    ]
    latest = observations[-1].summary if observations else None
    if latest is None or missing:
        retrieval: dict[str, object] = _unknown_cost(
            "Not every retrieval trace exposed observed DeepSeek response usage; "
            "retrieval cost is not estimated."
        )
    else:
        retrieval = latest.model_dump(mode="json")
        retrieval["aggregation"] = "latest_trace_snapshot"
    retrieval["observations"] = serialized_observations
    retrieval["missing_trace_ids"] = missing

    judge = _unknown_cost(
        "Ragas judge usage is not exposed by the current API; judge cost is not estimated."
    )
    total = _unknown_cost(
        "Ragas judge cost is unknown because usage is not exposed; total cost is not estimated."
    )
    return {
        "retrieval": retrieval,
        "judge": judge,
        "total": total,
    }


def _unknown_cost(reason: str) -> dict[str, object]:
    return {
        "status": DeepSeekCostStatus.UNKNOWN.value,
        "currency": "CNY",
        "cost_cny": None,
        "price_assumption": reason,
    }


def _string_list(value: object, *, field: str, index: int) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Ragas test set item {index} has invalid {field}")
    return list(value)


def _sample(case: RagasCase, *, response: str, retrieved_contexts: list[str]) -> object:
    from ragas import SingleTurnSample

    return SingleTurnSample(
        user_input=case["user_input"],
        reference=case["reference"],
        reference_contexts=case["reference_contexts"],
        response=response,
        retrieved_contexts=retrieved_contexts,
    )


def _evaluate(
    samples: list[object],
    *,
    judge_model: str,
    judge_api_key: str,
    judge_base_url: str,
    judge_max_output_tokens: int,
    judge_timeout_seconds: float,
) -> dict[str, object]:
    from openai import OpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import EvaluationResult
    from ragas.llms import llm_factory
    from ragas.metrics.base import Metric
    from ragas.metrics.collections import (
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )

    judge_llm = llm_factory(
        judge_model,
        provider="openai",
        client=OpenAI(
            api_key=judge_api_key,
            base_url=judge_base_url,
            timeout=judge_timeout_seconds,
            max_retries=0,
        ),
        max_tokens=judge_max_output_tokens,
        temperature=0.0,
    )
    metrics = cast(
        Sequence[Metric],
        [
            Faithfulness(llm=judge_llm),
            FactualCorrectness(llm=judge_llm),
            ContextPrecision(llm=judge_llm),
            ContextRecall(llm=judge_llm),
        ],
    )
    result = cast(
        EvaluationResult,
        evaluate(
            dataset=EvaluationDataset(samples=samples),  # type: ignore[arg-type]
            metrics=metrics,
            show_progress=True,
            raise_exceptions=True,
            return_executor=False,
        ),
    )
    scores = result.scores
    return {"scores": scores, "means": _means(scores)}


def _means(scores: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in scores:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(key, []).append(float(value))
    return {key: sum(items) / len(items) for key, items in values.items() if items}
