"""Deterministic metrics derived from one agentic event timeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from hybrid_rag.evaluation.evidence import evidence_ids

if TYPE_CHECKING:
    from hybrid_rag.agentic.models import AgentEvent


@dataclass(frozen=True, slots=True)
class AgenticMetricScores:
    tool_call_count: int
    successful_tool_calls: int
    tool_calls_by_name: dict[str, int]
    read_evidence_count: int
    cited_evidence_count: int
    evidence_utilization: float | None
    citation_validity: float | None
    citation_reference_precision: float | None
    reference_evidence_precision: float | None
    reference_evidence_recall: float | None
    insufficient_evidence: bool | None
    refusal_correct: bool | None
    duration_seconds: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def score_agentic_events(
    events: Sequence[AgentEvent],
    *,
    reference_evidence_ids: Sequence[str] | None = None,
    answerable: bool | None = None,
    duration_seconds: float | None = None,
) -> AgenticMetricScores:
    """Score tool use, evidence use, citations, latency, and refusal behavior."""

    tool_results = [event for event in events if event.event == "tool_result"]
    by_name: Counter[str] = Counter()
    successful = 0
    for event in tool_results:
        tool = event.data.get("tool")
        if isinstance(tool, str):
            by_name[tool] += 1
        if event.data.get("ok") is True:
            successful += 1

    answer_event = next((event for event in reversed(events) if event.event == "answer"), None)
    evidence_values = answer_event.data.get("evidence", []) if answer_event is not None else []
    evidence = [item for item in evidence_values if isinstance(item, dict)]
    read_chunk_ids = {
        chunk_id
        for item in evidence
        if isinstance((chunk_id := item.get("chunk_id")), str) and chunk_id
    }
    evidence_by_chunk = {
        chunk_id: locator
        for item in evidence
        if isinstance((chunk_id := item.get("chunk_id")), str)
        and (locator := _context_evidence_id(item)) is not None
    }
    retrieved_evidence_ids = {
        identity for identities in evidence_by_chunk.values() for identity in identities
    }

    answer_value = answer_event.data.get("answer") if answer_event is not None else None
    answer = answer_value if isinstance(answer_value, dict) else {}
    citations_value = answer.get("citations", [])
    citations = {item for item in citations_value if isinstance(item, str) and item}
    valid_citations = citations & read_chunk_ids
    cited_reference_ids = {
        identity
        for citation in valid_citations
        if citation in evidence_by_chunk
        for identity in evidence_by_chunk[citation]
    }
    utilization = len(valid_citations) / len(read_chunk_ids) if read_chunk_ids else None
    citation_validity = len(valid_citations) / len(citations) if citations else None

    reference = set(reference_evidence_ids or ())
    if reference:
        matches = retrieved_evidence_ids & reference
        citation_reference_precision = (
            len(cited_reference_ids & reference) / len(cited_reference_ids)
            if cited_reference_ids
            else 0.0
        )
        reference_precision = (
            len(matches) / len(retrieved_evidence_ids) if retrieved_evidence_ids else 0.0
        )
        reference_recall = len(matches) / len(reference)
    else:
        citation_reference_precision = None
        reference_precision = None
        reference_recall = None

    insufficient = answer.get("insufficient_evidence")
    insufficient_value = insufficient if isinstance(insufficient, bool) else None
    refusal_correct = (
        insufficient_value == (not answerable)
        if insufficient_value is not None and answerable is not None
        else None
    )
    return AgenticMetricScores(
        tool_call_count=len(tool_results),
        successful_tool_calls=successful,
        tool_calls_by_name=dict(sorted(by_name.items())),
        read_evidence_count=len(read_chunk_ids),
        cited_evidence_count=len(citations),
        evidence_utilization=utilization,
        citation_validity=citation_validity,
        citation_reference_precision=citation_reference_precision,
        reference_evidence_precision=reference_precision,
        reference_evidence_recall=reference_recall,
        insufficient_evidence=insufficient_value,
        refusal_correct=refusal_correct,
        duration_seconds=duration_seconds,
    )


def aggregate_agentic_scores(scores: Sequence[AgenticMetricScores]) -> dict[str, object]:
    """Return macro means for available per-run agentic metrics."""

    if not scores:
        raise ValueError("agentic metric aggregation requires at least one run")
    mean_fields = (
        "tool_call_count",
        "successful_tool_calls",
        "read_evidence_count",
        "cited_evidence_count",
        "evidence_utilization",
        "citation_validity",
        "citation_reference_precision",
        "reference_evidence_precision",
        "reference_evidence_recall",
        "duration_seconds",
    )
    means: dict[str, float | None] = {}
    for field in mean_fields:
        values = [float(value) for score in scores if (value := getattr(score, field)) is not None]
        means[field] = sum(values) / len(values) if values else None
    refusal_values = [
        score.refusal_correct for score in scores if score.refusal_correct is not None
    ]
    means["refusal_accuracy"] = (
        sum(value is True for value in refusal_values) / len(refusal_values)
        if refusal_values
        else None
    )
    return {"runs": len(scores), "means": means}


def _context_evidence_id(value: dict[str, object]) -> tuple[str, ...] | None:
    document_id = value.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        return None
    section = value.get("section_path")
    section_path = (
        tuple(item for item in section if isinstance(item, str))
        if isinstance(section, list | tuple)
        else ()
    )
    page_start = value.get("page_start")
    page_end = value.get("page_end")
    return evidence_ids(
        document_id,
        page_start=page_start
        if isinstance(page_start, int) and not isinstance(page_start, bool)
        else None,
        page_end=(
            page_end if isinstance(page_end, int) and not isinstance(page_end, bool) else None
        ),
        section_path=section_path,
    )


__all__ = ["AgenticMetricScores", "aggregate_agentic_scores", "score_agentic_events"]
