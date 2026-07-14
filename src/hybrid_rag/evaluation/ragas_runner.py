"""Ragas-backed evaluation over the project's existing retrieval and answer pipeline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hybrid_rag.retrieval.models import RetrievalMode
from hybrid_rag.retrieval.query import QueryClient
from hybrid_rag.retrieval.service import RetrievalOptions, RetrievalService


@dataclass(frozen=True, slots=True)
class RagasEvaluationReport:
    """Serializable Ragas scores and per-sample details grouped by retrieval mode."""

    testset_path: str
    modes: dict[str, dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return {"testset_path": self.testset_path, "modes": self.modes}


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
    ) -> RagasEvaluationReport:
        cases = _load_cases(testset_path)
        output: dict[str, dict[str, object]] = {}
        for mode in modes:
            samples = []
            for case in cases:
                answer = await self.retrieval_service.ask(
                    case["user_input"],
                    query_client=self.query_client,
                    mode=mode,
                    options=retrieval_options,
                    profile_ref=profile_ref,
                )
                samples.append(
                    _sample(
                        case,
                        response=answer.answer.answer,
                        retrieved_contexts=[item.text for item in answer.retrieval.context_items],
                    )
                )
            result = await asyncio.to_thread(
                _evaluate,
                samples,
                judge_model=judge_model,
                judge_api_key=judge_api_key,
                judge_base_url=judge_base_url,
            )
            output[mode.value] = result
        return RagasEvaluationReport(testset_path=str(testset_path), modes=output)


def _load_cases(path: Path) -> list[dict[str, object]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read Ragas test set {path}: {error}") from error
    if not isinstance(value, list) or not value:
        raise ValueError("Ragas test set must be a non-empty JSON array")
    cases: list[dict[str, object]] = []
    for index, item in enumerate(value, start=1):
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
    return cases


def _string_list(value: object, *, field: str, index: int) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Ragas test set item {index} has invalid {field}")
    return list(value)


def _sample(
    case: dict[str, object], *, response: str, retrieved_contexts: list[str]
) -> object:
    from ragas import SingleTurnSample

    return SingleTurnSample(
        user_input=str(case["user_input"]),
        reference=str(case["reference"]),
        reference_contexts=list(case["reference_contexts"]),
        response=response,
        retrieved_contexts=retrieved_contexts,
    )


def _evaluate(
    samples: list[object], *, judge_model: str, judge_api_key: str, judge_base_url: str
) -> dict[str, object]:
    from openai import OpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )

    judge_llm = llm_factory(
        judge_model,
        provider="openai",
        client=OpenAI(api_key=judge_api_key, base_url=judge_base_url),
    )
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),  # type: ignore[arg-type]
        metrics=[
            Faithfulness(llm=judge_llm),
            FactualCorrectness(llm=judge_llm),
            ContextPrecision(llm=judge_llm),
            ContextRecall(llm=judge_llm),
        ],
        show_progress=True,
        raise_exceptions=True,
    )
    scores = result.scores  # ``return_executor`` defaults to False.
    return {"scores": scores, "means": _means(scores)}


def _means(scores: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in scores:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(key, []).append(float(value))
    return {key: sum(items) / len(items) for key, items in values.items() if items}
