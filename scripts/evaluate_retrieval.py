"""Run a corpus-bound retrieval benchmark without answer or judge model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from hybrid_rag.config import RetrievalSettings, Settings, sqlite_url
from hybrid_rag.evaluation.evidence import evidence_ids, evidence_ids_from_metadata
from hybrid_rag.evaluation.retrieval_metrics import (
    RankedEvidence,
    RetrievalMetricScores,
    SemanticEvidenceScores,
    aggregate_retrieval_scores,
    aggregate_semantic_evidence_scores,
    score_document_retrieval,
    score_retrieval,
    score_semantic_evidence,
)
from hybrid_rag.ingest.tokenizer import TiktokenCounter
from hybrid_rag.retrieval.embedding import (
    BGEM3EmbeddingProvider,
    HashEmbeddingProvider,
    cosine_similarity,
)
from hybrid_rag.retrieval.models import RetrievalStrategy
from hybrid_rag.retrieval.query import deterministic_keywords
from hybrid_rag.retrieval.reranker import create_reranker
from hybrid_rag.retrieval.service import RetrievalOptions, RetrievalService
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database


def main() -> None:
    args = _arguments()
    settings = Settings()
    retrieval = RetrievalSettings()
    modes = _modes(args.modes)
    testset_bytes = args.testset.read_bytes()
    testset = _testset(json.loads(testset_bytes))
    database_url = sqlite_url(args.db)
    upgrade_database(database_url)
    database = Database(database_url)
    try:
        embedding_provider = _embedding_provider(retrieval)
        reranker_provider = "flagembedding" if args.rerank else "none"
        service = RetrievalService(
            database,
            embedding_provider,
            TiktokenCounter(settings.tokenizer_name),
            reranker=create_reranker(
                reranker_provider,
                retrieval.reranker_model,
                use_fp16=retrieval.reranker_use_fp16,
            ),
        )
        profile = service.resolve_profile(args.profile)
        profile_corpus_hash = str(profile.metadata.get("corpus_content_hash", ""))
        testset_corpus_hash = str(testset["corpus_content_hash"])
        if testset_corpus_hash != profile_corpus_hash:
            raise ValueError(
                "test set corpus_content_hash does not match the pinned index profile "
                f"({testset_corpus_hash} != {profile_corpus_hash})"
            )
        options = _retrieval_options(retrieval, top_k=args.top, rerank=args.rerank)
        mode_reports, warmup = _evaluate_modes(
            service,
            modes,
            cases=testset["cases"],
            profile_id=profile.id,
            options=options,
            embedding_provider=embedding_provider,
            semantic_threshold=args.semantic_threshold,
        )
        payload = {
            "benchmark_type": "retrieval_only",
            "testset_path": str(args.testset),
            "provenance": {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "git": _git_provenance(),
                "environment": _environment_provenance(),
                "testset": {
                    "schema_version": testset["schema_version"],
                    "corpus_content_hash": testset_corpus_hash,
                    "file_sha256": hashlib.sha256(testset_bytes).hexdigest(),
                    "case_count": len(testset["cases"]),
                    "review_status_counts": _review_status_counts(testset["cases"]),
                    "question_type_counts": _value_counts(testset["cases"], "question_type"),
                    "answerable_counts": _value_counts(testset["cases"], "answerable"),
                },
                "profile": {
                    "id": profile.id,
                    "config_hash": profile.config_hash,
                    "provider": profile.provider,
                    "model": profile.model,
                    "dimensions": profile.dimensions,
                    "schema_version": profile.schema_version,
                    "source_corpus_hash": profile.source_corpus_hash,
                    "source_graph_run_id": profile.source_graph_run_id,
                    "corpus_content_hash": profile_corpus_hash,
                },
                "runtime": {
                    "modes": [mode.value for mode in modes],
                    "retrieval_options": asdict(options),
                    "keyword_extractor": "deterministic_keywords",
                    "mode_execution_order": "alternated_per_case_after_one_warmup",
                    "metric_contract": {
                        "exact_page": "stable document/page/section evidence IDs",
                        "document": "gold document IDs with duplicate chunks suppressed",
                        "semantic_evidence": {
                            "embedding_provider": embedding_provider.provider,
                            "embedding_model": embedding_provider.model,
                            "cosine_threshold": args.semantic_threshold,
                            "unit": "reference-context coverage",
                        },
                    },
                    "warmup": warmup,
                    "answer_model": None,
                    "judge_model": None,
                    "external_llm_calls": 0,
                },
            },
            "cost": {
                "status": "not_applicable",
                "currency": "CNY",
                "cost_cny": 0.0,
                "reason": "Local embedding, reranking, and deterministic query processing only.",
            },
            "modes": mode_reports,
            "comparison": _paired_comparison(mode_reports, modes),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Retrieval benchmark written to {args.output}")
        for mode, report in mode_reports.items():
            print(f"{mode}: {report['means']}")
    finally:
        database.dispose()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("storage/app.db"))
    parser.add_argument("--profile")
    parser.add_argument("--modes", default="hybrid,mix")
    parser.add_argument("--top", type=int)
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable the configured local FlagEmbedding cross-encoder reranker.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.75,
        help="Cosine threshold for reference-context semantic coverage (default: 0.75).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluations/retrieval-only.json"),
    )
    args = parser.parse_args()
    if args.top is not None and args.top < 1:
        parser.error("--top must be positive")
    if not math.isfinite(args.semantic_threshold) or not -1.0 <= args.semantic_threshold <= 1.0:
        parser.error("--semantic-threshold must be between -1 and 1")
    return args


def _modes(raw: str) -> tuple[RetrievalStrategy, ...]:
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names or len(names) != len(set(names)):
        raise ValueError("--modes must contain one or more distinct retrieval modes")
    try:
        return tuple(RetrievalStrategy(name) for name in names)
    except ValueError as error:
        raise ValueError("--modes contains an unsupported retrieval mode") from error


def _testset(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("test set must be a JSON object envelope")
    if value.get("schema_version") != "2":
        raise ValueError("retrieval benchmark requires test-set schema version 2")
    corpus_hash = value.get("corpus_content_hash")
    cases = value.get("cases")
    if not isinstance(corpus_hash, str) or not corpus_hash:
        raise ValueError("test set must include corpus_content_hash")
    if not isinstance(cases, list) or not cases:
        raise ValueError("test set must include non-empty cases")
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"test-set case {index} must be an object")
        if not isinstance(case.get("user_input"), str) or not case["user_input"].strip():
            raise ValueError(f"test-set case {index} must include user_input")
        if not isinstance(case.get("reference_contexts"), list):
            raise ValueError(f"test-set case {index} must include reference_contexts")
        if not isinstance(case.get("evidence_ids"), list):
            raise ValueError(f"test-set case {index} must include evidence_ids")
    return value


def _embedding_provider(
    settings: RetrievalSettings,
) -> BGEM3EmbeddingProvider | HashEmbeddingProvider:
    if settings.embedding_provider == "flagembedding":
        return BGEM3EmbeddingProvider(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
            max_length=settings.embedding_max_length,
            use_fp16=settings.embedding_use_fp16,
        )
    return HashEmbeddingProvider(
        dimensions=settings.embedding_dimensions,
        model=settings.embedding_model,
    )


def _retrieval_options(
    settings: RetrievalSettings,
    *,
    top_k: int | None,
    rerank: bool,
) -> RetrievalOptions:
    return RetrievalOptions(
        top_k=top_k or settings.top_k,
        candidate_multiplier=settings.candidate_multiplier,
        context_token_budget=settings.context_token_budget,
        graph_max_hops=settings.graph_max_hops,
        naive_weight=settings.hybrid_weight,
        local_weight=settings.graph_local_weight,
        global_weight=settings.graph_global_weight,
        naive_dense_weight=settings.hybrid_dense_weight,
        naive_bm25_weight=settings.hybrid_bm25_weight,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        reranker_provider="flagembedding" if rerank else "none",
        reranker_model=settings.reranker_model,
        reranker_use_fp16=settings.reranker_use_fp16,
        rerank_candidate_multiplier=settings.rerank_candidate_multiplier,
    )


@dataclass(slots=True)
class _ModeState:
    raw_scores: list[RetrievalMetricScores] = field(default_factory=list)
    context_scores: list[RetrievalMetricScores] = field(default_factory=list)
    raw_document_scores: list[RetrievalMetricScores] = field(default_factory=list)
    context_document_scores: list[RetrievalMetricScores] = field(default_factory=list)
    raw_semantic_scores: list[SemanticEvidenceScores] = field(default_factory=list)
    context_semantic_scores: list[SemanticEvidenceScores] = field(default_factory=list)
    semantic_inputs: list[_SemanticInputs] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _SemanticInputs:
    references: tuple[str, ...]
    raw_texts: tuple[str, ...]
    context_texts: tuple[str, ...]


def _evaluate_modes(
    service: RetrievalService,
    modes: tuple[RetrievalStrategy, ...],
    *,
    cases: list[dict[str, Any]],
    profile_id: str,
    options: RetrievalOptions,
    embedding_provider: BGEM3EmbeddingProvider | HashEmbeddingProvider,
    semantic_threshold: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    states = {mode: _ModeState() for mode in modes}
    warmup_keywords = deterministic_keywords(cases[0]["user_input"]).keywords
    warmup_started = time.perf_counter()
    service.retrieve(
        cases[0]["user_input"],
        mode=modes[0],
        options=options,
        profile_ref=profile_id,
        keywords=warmup_keywords,
        persist=False,
    )
    warmup = {
        "mode": modes[0].value,
        "duration_seconds": time.perf_counter() - warmup_started,
        "included_in_metrics": False,
    }
    for index, case in enumerate(cases, start=1):
        keywords = deterministic_keywords(case["user_input"]).keywords
        execution_order = modes if index % 2 else tuple(reversed(modes))
        for mode in execution_order:
            _evaluate_case(
                service,
                mode,
                case=case,
                case_index=index,
                case_count=len(cases),
                keywords=keywords,
                profile_id=profile_id,
                options=options,
                state=states[mode],
            )
    _attach_semantic_scores(
        states,
        embedding_provider=embedding_provider,
        threshold=semantic_threshold,
    )
    return (
        {mode.value: _mode_report(states[mode]) for mode in modes},
        warmup,
    )


def _evaluate_case(
    service: RetrievalService,
    mode: RetrievalStrategy,
    *,
    case: dict[str, Any],
    case_index: int,
    case_count: int,
    keywords: tuple[str, ...],
    profile_id: str,
    options: RetrievalOptions,
    state: _ModeState,
) -> None:
    started = time.perf_counter()
    result = service.retrieve(
        case["user_input"],
        mode=mode,
        options=options,
        profile_ref=profile_id,
        keywords=keywords,
        persist=False,
    )
    latency = time.perf_counter() - started
    raw_texts = tuple(str(hit.metadata.get("text", "")) for hit in result.hits)
    context_texts = tuple(item.text for item in result.context_items)
    raw_score = score_retrieval(
        [
            RankedEvidence(
                evidence_ids=evidence_ids_from_metadata(hit.metadata),
                text=str(hit.metadata.get("text", "")),
            )
            for hit in result.hits
        ],
        k=options.top_k,
        evidence_ids=case["evidence_ids"],
        reference_contexts=case["reference_contexts"],
    )
    context_score = score_retrieval(
        [
            RankedEvidence(
                evidence_ids=evidence_ids(
                    item.document_id,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    section_path=item.section_path,
                ),
                text=item.text,
            )
            for item in result.context_items
        ],
        k=options.top_k,
        evidence_ids=case["evidence_ids"],
        reference_contexts=case["reference_contexts"],
    )
    gold_document_ids = (
        tuple(str(value) for value in case.get("document_ids", ()))
        if case.get("answerable", True)
        else ()
    )
    raw_document_score = score_document_retrieval(
        tuple(str(hit.metadata.get("document_id", "")) for hit in result.hits),
        k=options.top_k,
        document_ids=gold_document_ids,
    )
    context_document_score = score_document_retrieval(
        tuple(item.document_id for item in result.context_items),
        k=options.top_k,
        document_ids=gold_document_ids,
    )
    state.raw_scores.append(raw_score)
    state.context_scores.append(context_score)
    state.raw_document_scores.append(raw_document_score)
    state.context_document_scores.append(context_document_score)
    state.semantic_inputs.append(
        _SemanticInputs(
            references=(
                tuple(str(value) for value in case["reference_contexts"])
                if case.get("answerable", True)
                else ()
            ),
            raw_texts=raw_texts,
            context_texts=context_texts,
        )
    )
    state.latencies.append(latency)
    state.details.append(
        {
            "case_index": case_index,
            "question_type": case.get("question_type", "legacy"),
            "answerable": case.get("answerable", True),
            "gold_evidence_ids": case["evidence_ids"],
            "gold_document_ids": list(gold_document_ids),
            "latency_seconds": latency,
            "keywords": list(keywords),
            "expanded_query": result.trace.expanded_query,
            "context_tokens": result.context_tokens,
            "retrieval_metrics": {
                "raw_top_k": raw_score.as_dict(),
                "delivered_context": context_score.as_dict(),
                "document_raw_top_k": raw_document_score.as_dict(),
                "document_delivered_context": context_document_score.as_dict(),
            },
            "trace": result.trace.model_dump(mode="json"),
        }
    )
    print(f"{mode.value} {case_index}/{case_count}")


def _mode_report(state: _ModeState) -> dict[str, Any]:
    raw_aggregate = aggregate_retrieval_scores(state.raw_scores)
    context_aggregate = aggregate_retrieval_scores(state.context_scores)
    raw_document_aggregate = aggregate_retrieval_scores(state.raw_document_scores)
    context_document_aggregate = aggregate_retrieval_scores(state.context_document_scores)
    raw_semantic_aggregate = aggregate_semantic_evidence_scores(state.raw_semantic_scores)
    context_semantic_aggregate = aggregate_semantic_evidence_scores(state.context_semantic_scores)
    raw_micro_recall = _micro_recall(state.raw_scores)
    context_micro_recall = _micro_recall(state.context_scores)
    means = {
        **{f"raw_{key}": value for key, value in raw_aggregate["means"].items()},
        **{f"context_{key}": value for key, value in context_aggregate["means"].items()},
        **{f"document_raw_{key}": value for key, value in raw_document_aggregate["means"].items()},
        **{
            f"document_context_{key}": value
            for key, value in context_document_aggregate["means"].items()
        },
        **{f"semantic_raw_{key}": value for key, value in raw_semantic_aggregate["means"].items()},
        **{
            f"semantic_context_{key}": value
            for key, value in context_semantic_aggregate["means"].items()
        },
        "raw_micro_recall": raw_micro_recall,
        "context_micro_recall": context_micro_recall,
    }
    means.update(_latency_summary(state.latencies))
    return {
        "raw_top_k": {**raw_aggregate, "micro_recall": raw_micro_recall},
        "delivered_context": {
            **context_aggregate,
            "micro_recall": context_micro_recall,
        },
        "metric_labels": {
            "raw_top_k": "exact_page",
            "delivered_context": "exact_page",
        },
        "document": {
            "raw_top_k": raw_document_aggregate,
            "delivered_context": context_document_aggregate,
        },
        "semantic_evidence": {
            "raw_top_k": raw_semantic_aggregate,
            "delivered_context": context_semantic_aggregate,
        },
        "by_question_type": _question_type_reports(state),
        "latency": _latency_summary(state.latencies),
        "means": means,
        "cases": state.details,
    }


def _attach_semantic_scores(
    states: dict[RetrievalStrategy, _ModeState],
    *,
    embedding_provider: BGEM3EmbeddingProvider | HashEmbeddingProvider,
    threshold: float,
) -> None:
    unique_texts = tuple(
        dict.fromkeys(
            text
            for state in states.values()
            for semantic_input in state.semantic_inputs
            for text in (
                *semantic_input.references,
                *semantic_input.raw_texts,
                *semantic_input.context_texts,
            )
            if text.strip()
        )
    )
    vectors = embedding_provider.embed(unique_texts)
    if len(vectors) != len(unique_texts):
        raise RuntimeError("semantic evaluator returned a different number of vectors")
    vector_by_text = dict(zip(unique_texts, vectors, strict=True))
    for state in states.values():
        for detail, semantic_input in zip(
            state.details,
            state.semantic_inputs,
            strict=True,
        ):
            raw_score = _semantic_score(
                semantic_input.references,
                semantic_input.raw_texts,
                vector_by_text=vector_by_text,
                threshold=threshold,
            )
            context_score = _semantic_score(
                semantic_input.references,
                semantic_input.context_texts,
                vector_by_text=vector_by_text,
                threshold=threshold,
            )
            state.raw_semantic_scores.append(raw_score)
            state.context_semantic_scores.append(context_score)
            detail["retrieval_metrics"]["semantic_raw_top_k"] = raw_score.as_dict()
            detail["retrieval_metrics"]["semantic_delivered_context"] = context_score.as_dict()


def _semantic_score(
    references: tuple[str, ...],
    retrieved: tuple[str, ...],
    *,
    vector_by_text: dict[str, tuple[float, ...]],
    threshold: float,
) -> SemanticEvidenceScores:
    usable_references = tuple(text for text in references if text.strip())
    usable_retrieved = tuple(text for text in retrieved if text.strip())
    similarities = tuple(
        tuple(
            max(
                -1.0,
                min(
                    1.0,
                    cosine_similarity(vector_by_text[reference], vector_by_text[text]),
                ),
            )
            for text in usable_retrieved
        )
        for reference in usable_references
    )
    return score_semantic_evidence(
        similarities,
        threshold=threshold,
        retrieved_count=len(usable_retrieved),
    )


def _question_type_reports(state: _ModeState) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for index, detail in enumerate(state.details):
        grouped.setdefault(str(detail["question_type"]), []).append(index)
    return {
        question_type: _question_type_report(state, indices)
        for question_type, indices in sorted(grouped.items())
    }


def _question_type_report(
    state: _ModeState,
    indices: list[int],
) -> dict[str, Any]:
    raw_scores = [state.raw_scores[index] for index in indices]
    context_scores = [state.context_scores[index] for index in indices]
    raw_document_scores = [state.raw_document_scores[index] for index in indices]
    context_document_scores = [state.context_document_scores[index] for index in indices]
    raw_semantic_scores = [state.raw_semantic_scores[index] for index in indices]
    context_semantic_scores = [state.context_semantic_scores[index] for index in indices]
    return {
        "case_count": len(indices),
        "answerable_case_count": sum(bool(state.details[index]["answerable"]) for index in indices),
        "exact_page": {
            "raw_top_k": aggregate_retrieval_scores(raw_scores),
            "delivered_context": aggregate_retrieval_scores(context_scores),
        },
        "document": {
            "raw_top_k": aggregate_retrieval_scores(raw_document_scores),
            "delivered_context": aggregate_retrieval_scores(context_document_scores),
        },
        "semantic_evidence": {
            "raw_top_k": aggregate_semantic_evidence_scores(raw_semantic_scores),
            "delivered_context": aggregate_semantic_evidence_scores(context_semantic_scores),
        },
        "latency": _latency_summary([state.latencies[index] for index in indices]),
    }


def _micro_recall(scores: list[RetrievalMetricScores]) -> float | None:
    eligible = [score for score in scores if score.applicable]
    relevant_count = sum(score.relevant_count for score in eligible)
    if relevant_count == 0:
        return None
    return sum(score.matched_count for score in eligible) / relevant_count


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "latency_mean_seconds": sum(ordered) / len(ordered),
        "latency_p50_seconds": _percentile(ordered, 0.50),
        "latency_p95_seconds": _percentile(ordered, 0.95),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]


def _review_status_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        status = str(case.get("review_status", "missing"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _value_counts(cases: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        value = str(case.get(field_name, "missing")).lower()
        counts[value] = counts.get(value, 0) + 1
    return counts


def _paired_comparison(
    reports: dict[str, dict[str, Any]], modes: tuple[RetrievalStrategy, ...]
) -> dict[str, Any]:
    if len(modes) != 2:
        return {"status": "not_applicable", "reason": "comparison requires exactly two modes"}
    baseline, candidate = (mode.value for mode in modes)
    baseline_cases = reports[baseline]["cases"]
    candidate_cases = reports[candidate]["cases"]
    metrics: dict[str, dict[str, Any]] = {}
    scopes = (
        (
            "raw_top_k",
            "raw_top_k",
            ("hit_at_k", "recall_at_k", "mrr", "ndcg_at_k"),
        ),
        (
            "delivered_context",
            "delivered_context",
            ("hit_at_k", "recall_at_k", "mrr", "ndcg_at_k"),
        ),
        (
            "document.raw_top_k",
            "document_raw_top_k",
            ("hit_at_k", "recall_at_k", "mrr", "ndcg_at_k"),
        ),
        (
            "document.delivered_context",
            "document_delivered_context",
            ("hit_at_k", "recall_at_k", "mrr", "ndcg_at_k"),
        ),
        (
            "semantic_evidence.raw_top_k",
            "semantic_raw_top_k",
            ("mean_max_similarity", "threshold_recall", "all_references_covered"),
        ),
        (
            "semantic_evidence.delivered_context",
            "semantic_delivered_context",
            ("mean_max_similarity", "threshold_recall", "all_references_covered"),
        ),
    )
    for scope, detail_key, metric_names in scopes:
        for metric in metric_names:
            deltas: list[float] = []
            for left, right in zip(baseline_cases, candidate_cases, strict=True):
                if left["case_index"] != right["case_index"]:
                    raise RuntimeError("paired comparison received misaligned cases")
                left_value = left["retrieval_metrics"][detail_key][metric]
                right_value = right["retrieval_metrics"][detail_key][metric]
                if left_value is None or right_value is None:
                    continue
                deltas.append(float(right_value) - float(left_value))
            metrics[f"{scope}.{metric}"] = {
                "eligible_pairs": len(deltas),
                "candidate_wins": sum(delta > 1e-12 for delta in deltas),
                "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
                "baseline_wins": sum(delta < -1e-12 for delta in deltas),
                "mean_delta": sum(deltas) / len(deltas) if deltas else None,
            }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta_direction": f"{candidate} - {baseline}",
        "metrics": metrics,
    }


def _git_provenance() -> dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain")
    return {
        "commit": commit or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def _git_output(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _environment_provenance() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            name: _package_version(name) for name in ("hybrid-rag-lab", "FlagEmbedding", "torch")
        },
        "accelerator": _accelerator_provenance(),
    }


def _accelerator_provenance() -> dict[str, Any]:
    result: dict[str, Any] = {
        "probe_status": "torch_not_installed",
        "probe_error": None,
        "torch_version": _package_version("torch"),
        "torch_cuda_build": None,
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
    }
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as error:
        if error.name != "torch":
            result["probe_status"] = "error"
            result["probe_error"] = type(error).__name__
        return result
    except Exception as error:
        result["probe_status"] = "error"
        result["probe_error"] = type(error).__name__
        return result

    result["probe_status"] = "ok"
    try:
        torch_version = getattr(torch, "__version__", None)
        if torch_version is not None:
            result["torch_version"] = str(torch_version)
        cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
        result["torch_cuda_build"] = str(cuda_build) if cuda_build is not None else None
        cuda = torch.cuda
        cuda_available = bool(cuda.is_available())
        device_count = int(cuda.device_count())
        result["cuda_available"] = cuda_available
        result["device_count"] = device_count
        result["device_names"] = [str(cuda.get_device_name(index)) for index in range(device_count)]
    except Exception as error:
        result["probe_status"] = "error"
        result["probe_error"] = type(error).__name__
    return result


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
