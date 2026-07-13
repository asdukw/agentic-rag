"""Portable JSON and Markdown rendering for evaluation reports."""

from __future__ import annotations

from pathlib import Path

from hybrid_rag.evaluation.contracts import EvaluationReport


def render_markdown(report: EvaluationReport) -> str:
    """Render a concise human-reviewable report without hiding raw JSON data."""

    judge_temperature = report.judge_provenance.temperature
    lines = [
        f"# Retrieval evaluation: {report.run.benchmark_id}",
        "",
        f"Reproducibility ID: `{report.run.id}`  ",
        f"Execution ID: `{report.run.execution_id}`  ",
        f"Cases: {len(report.run.case_ids)}  ",
        f"Options hash: `{report.run.options.config_hash}`",
        "",
        "## Pinned index snapshot",
        "",
        f"- Profile: `{report.run.index_provenance.profile_id}`",
        f"- Index config hash: `{report.run.index_provenance.index_config_hash}`",
        "- Embedding: "
        f"`{report.run.index_provenance.embedding_provider}` / "
        f"`{report.run.index_provenance.embedding_model}` "
        f"({report.run.index_provenance.embedding_dimensions} dimensions)",
        "- Graph-independent corpus content hash: "
        f"`{report.run.index_provenance.corpus_content_hash}`",
        f"- Graph-bound source snapshot hash: `{report.run.index_provenance.source_corpus_hash}`",
        "- Graph snapshot: "
        f"`{report.run.index_provenance.source_graph_run_id or 'none'}`; corpus hash "
        f"`{report.run.index_provenance.source_graph_corpus_hash or 'none'}`",
        "",
        "## Judge and cost disclosure",
        "",
        "- Judge: "
        f"`{report.judge_provenance.provider}` / `{report.judge_provenance.protocol}`; "
        f"external={str(report.judge_provenance.external).lower()}",
        "- Judge model/base URL/output limit: "
        f"`{report.judge_provenance.model or 'none'}` / "
        f"`{report.judge_provenance.base_url or 'none'}` / "
        f"`{report.judge_provenance.max_output_tokens or 'none'}`",
        "- Judge response format/thinking/temperature: "
        f"`{report.judge_provenance.response_format or 'none'}` / "
        f"`{report.judge_provenance.thinking or 'none'}` / "
        f"`{judge_temperature if judge_temperature is not None else 'none'}`",
        f"Cost status: `{report.cost_disclosure.status.value}`; "
        f"cost (CNY): `{report.cost_disclosure.cost_cny}`",
        f"Price assumption: {report.cost_disclosure.price_assumption or '—'}",
        "",
    ]
    if report.cost_disclosure.deepseek_usage:
        lines.extend(
            [
                "DeepSeek response usage:",
                "",
                "| Operation | Model | Calls | Cache-hit input | Cache-miss input | Output | "
                "Cache split complete |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for usage in report.cost_disclosure.deepseek_usage:
            lines.append(
                "| "
                f"{usage.operation} | {usage.model} | {usage.calls} | "
                f"{usage.cache_hit_tokens if usage.cache_hit_tokens is not None else '—'} | "
                f"{usage.cache_miss_tokens if usage.cache_miss_tokens is not None else '—'} | "
                f"{usage.completion_tokens} | {str(usage.cache_breakdown_complete).lower()} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Mode summary",
            "",
            "| Mode | Evidence hit | Cited evidence hit | Citation-grounded faithfulness | "
            "Mean retrieval latency (ms) | Median retrieval latency (ms) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in report.summaries:
        lines.append(
            "| "
            f"{summary.mode.value} | {summary.mean_evidence_hit_rate:.3f} | "
            f"{summary.mean_cited_evidence_hit_rate:.3f} | "
            f"{summary.citation_grounded_faithfulness_rate:.3f} | "
            f"{summary.mean_retrieval_latency_ms:.3f} | "
            f"{summary.median_retrieval_latency_ms:.3f} |"
        )
    comparison = report.comparison_summary
    lines.extend(
        [
            "",
            "## Blind naive vs hybrid comparison",
            "",
            f"- Naive wins: {comparison.naive_wins} ({comparison.naive_win_rate:.3f})",
            f"- Hybrid wins: {comparison.hybrid_wins} ({comparison.hybrid_win_rate:.3f})",
            f"- Ties: {comparison.ties} ({comparison.tie_rate:.3f})",
            "",
            "## Per-case outcomes",
            "",
            "| Case | Winner | Protocol | Fallback | Replay traces | Rationale |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    traces_by_case: dict[str, list[str]] = {}
    for evaluation in report.evaluations:
        traces_by_case.setdefault(evaluation.case_id, []).append(evaluation.retrieval_trace_id)
    for judgment in report.pairwise_judgments:
        winner = judgment.winner_mode.value if judgment.winner_mode is not None else "tie"
        rationale = _table_text(judgment.judgment.rationale)
        traces = ", ".join(f"`{trace_id}`" for trace_id in traces_by_case[judgment.case_id])
        lines.append(
            f"| {judgment.case_id} | {winner} | {judgment.protocol} | "
            f"{str(judgment.used_fallback).lower()} | {traces} | {rationale} |"
        )
    return "\n".join(lines) + "\n"


def write_json(report: EvaluationReport, path: str | Path) -> Path:
    """Serialize a report to a caller-selected JSON path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")
    return output


def write_markdown(report: EvaluationReport, path: str | Path) -> Path:
    """Serialize a report to a caller-selected Markdown path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")
    return output


def _table_text(value: str) -> str:
    return " ".join(value.replace("|", "\\|").split())
