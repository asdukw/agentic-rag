"""Generate and validate corpus-bound golden evaluation test sets."""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from hybrid_rag.evaluation.evidence import evidence_ids_from_metadata
from hybrid_rag.evaluation.testset_contract import (
    EVALUATION_TESTSET_SCHEMA_VERSION,
    validate_corpus_content_hash,
    validate_testset_sources,
)
from hybrid_rag.extraction.client import DeepSeekClient, RetryableProviderError
from hybrid_rag.extraction.prompts import ChatMessage
from hybrid_rag.ids import file_source_uri
from hybrid_rag.ingest.cleaner import clean_document
from hybrid_rag.ingest.loaders import LoaderRegistry
from hybrid_rag.ingest.quality import classify_chunk_quality

GOLDEN_PROMPT_VERSION = "1"
DEFAULT_QUESTION_DISTRIBUTION: Mapping[str, float] = {
    "single_hop": 0.5,
    "summary_reasoning": 1 / 6,
    "multi_context": 1 / 6,
    "unanswerable": 1 / 6,
}
_LEXICAL_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}")
_LEXICAL_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "before",
        "between",
        "from",
        "into",
        "paper",
        "that",
        "their",
        "these",
        "this",
        "using",
        "with",
    }
)


class QuestionType(StrEnum):
    SINGLE_HOP = "single_hop"
    SUMMARY_REASONING = "summary_reasoning"
    MULTI_CONTEXT = "multi_context"
    UNANSWERABLE = "unanswerable"


@dataclass(frozen=True, slots=True)
class EvaluationDocument:
    """One loader segment eligible to become reference evidence."""

    text: str
    source_uri: str
    document_id: str
    document_title: str
    source_type: str
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    ordinal: int
    evidence_ids: tuple[str, ...]

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "source": self.source_uri,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "source_type": self.source_type,
            "section_path": list(self.section_path),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "ordinal": self.ordinal,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True, slots=True)
class GoldenCasePlan:
    """Deterministic source selection for one model-generated case."""

    index: int
    question_type: QuestionType
    contexts: tuple[EvaluationDocument, ...]

    @property
    def answerable(self) -> bool:
        return self.question_type is not QuestionType.UNANSWERABLE


class GoldenDraft(BaseModel):
    """Strict, untrusted model output before provenance is attached."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    user_input: Annotated[str, Field(min_length=4, max_length=1_000)]
    reference: Annotated[str, Field(min_length=2, max_length=4_000)]
    grounding_statements: list[Annotated[str, Field(min_length=2, max_length=1_000)]] = Field(
        max_length=6
    )
    evidence_quotes: list[Annotated[str, Field(min_length=2, max_length=1_500)]] = Field(
        max_length=6
    )
    insufficient_evidence: bool

    @model_validator(mode="after")
    def validate_grounding_shape(self) -> GoldenDraft:
        if self.insufficient_evidence:
            if self.grounding_statements or self.evidence_quotes:
                raise ValueError(
                    "insufficient-evidence cases must not claim grounding statements or quotes"
                )
        elif not self.grounding_statements or not self.evidence_quotes:
            raise ValueError("answerable cases require grounding statements and evidence quotes")
        return self


def load_evaluation_documents(
    source: Path,
    *,
    max_documents: int | None = None,
    max_segments_per_document: int | None = None,
    loaders: LoaderRegistry | None = None,
) -> list[EvaluationDocument]:
    """Load supported source segments and retain only ``normal`` evidence."""

    _validate_limit(max_documents, field="max_documents")
    _validate_limit(max_segments_per_document, field="max_segments_per_document")
    registry = loaders or LoaderRegistry()
    root, paths = _discover_source_files(source, registry)
    selected_paths = paths if max_documents is None else paths[:max_documents]

    documents: list[EvaluationDocument] = []
    for path in selected_paths:
        parsed = clean_document(registry.load(path, file_source_uri(path, root)))
        segments = (
            parsed.segments
            if max_segments_per_document is None
            else parsed.segments[:max_segments_per_document]
        )
        for ordinal, segment in enumerate(segments):
            if (
                classify_chunk_quality(
                    section_path=segment.section_path,
                    text=segment.text,
                    ordinal=ordinal,
                    page_start=segment.page_start,
                )
                != "normal"
            ):
                continue
            metadata: dict[str, object] = {
                "document_id": parsed.id,
                "section_path": list(segment.section_path),
                "page_start": segment.page_start,
                "page_end": segment.page_end,
            }
            documents.append(
                EvaluationDocument(
                    text=segment.text,
                    source_uri=parsed.source_uri,
                    document_id=parsed.id,
                    document_title=parsed.title,
                    source_type=parsed.source_type,
                    section_path=tuple(segment.section_path),
                    page_start=segment.page_start,
                    page_end=segment.page_end,
                    ordinal=ordinal,
                    evidence_ids=evidence_ids_from_metadata(metadata),
                )
            )

    if not documents:
        raise ValueError(f"no normal extractable text found in supported files under {source}")
    return documents


def plan_golden_cases(
    documents: Sequence[EvaluationDocument],
    *,
    testset_size: int,
    min_cases_per_document: int,
) -> tuple[GoldenCasePlan, ...]:
    """Create a deterministic, coverage-checked 50/16.7/16.7/16.7 plan."""

    if testset_size < 1:
        raise ValueError("testset_size must be positive")
    if min_cases_per_document < 1:
        raise ValueError("min_cases_per_document must be positive")
    grouped: dict[str, list[EvaluationDocument]] = defaultdict(list)
    for document in documents:
        grouped[document.document_id].append(document)
    if not grouped:
        raise ValueError("golden generation requires at least one source document")
    if testset_size < len(grouped) * min_cases_per_document:
        raise ValueError(
            f"testset_size={testset_size} cannot provide {min_cases_per_document} cases for "
            f"each of {len(grouped)} documents"
        )

    document_ids = tuple(sorted(grouped))
    for values in grouped.values():
        values.sort(key=_document_sort_key)
    counts = _distribution_counts(testset_size)
    plans: list[GoldenCasePlan] = []
    offsets: Counter[str] = Counter()

    def next_context(document_id: str) -> EvaluationDocument:
        values = grouped[document_id]
        value = values[offsets[document_id] % len(values)]
        offsets[document_id] += 1
        return value

    for question_type in QuestionType:
        count = counts[question_type]
        for offset in range(count):
            primary_id = document_ids[offset % len(document_ids)]
            primary = next_context(primary_id)
            contexts = (primary,)
            if question_type is QuestionType.SUMMARY_REASONING:
                secondary = next_context(primary_id)
                contexts = _unique_contexts((primary, secondary))
            elif question_type is QuestionType.MULTI_CONTEXT:
                secondary_id = (
                    document_ids[(offset + 1) % len(document_ids)]
                    if len(document_ids) > 1
                    else primary_id
                )
                secondary = _related_context(primary, grouped[secondary_id])
                contexts = _unique_contexts((primary, secondary))
            plans.append(
                GoldenCasePlan(
                    index=len(plans) + 1,
                    question_type=question_type,
                    contexts=contexts,
                )
            )

    coverage = Counter(
        document.document_id
        for plan in plans
        for document in {item.document_id: item for item in plan.contexts}.values()
    )
    below = [
        document_id
        for document_id in document_ids
        if coverage[document_id] < min_cases_per_document
    ]
    if below:
        raise ValueError(
            "generated plan did not meet per-document coverage: "
            + ", ".join(f"{value}={coverage[value]}" for value in below)
        )
    return tuple(plans)


async def generate_golden_cases(
    plans: Sequence[GoldenCasePlan],
    *,
    api_key: str,
    llm_model: str,
    base_url: str,
    max_concurrency: int = 4,
    max_retries: int = 2,
    timeout_seconds: float = 180.0,
    progress: Callable[[int, int, QuestionType], None] | None = None,
) -> list[dict[str, object]]:
    """Generate strict cases from preselected evidence without using Ragas generation."""

    if not plans:
        raise ValueError("golden generation requires at least one planned case")
    if not api_key.strip():
        raise ValueError("golden generation requires a non-empty API key")
    if not llm_model.strip() or not base_url.strip():
        raise ValueError("golden generation requires a model and base URL")
    if max_concurrency < 1 or max_retries < 0:
        raise ValueError("max_concurrency must be positive and max_retries non-negative")

    semaphore = asyncio.Semaphore(max_concurrency)
    async with DeepSeekClient(
        api_key=api_key,
        base_url=base_url,
        model=llm_model,
        max_output_tokens=2_048,
        timeout_seconds=timeout_seconds,
        temperature=0.0,
    ) as client:

        async def generate(
            plan: GoldenCasePlan,
        ) -> tuple[int, QuestionType, tuple[GoldenDraft, tuple[int, ...]]]:
            async with semaphore:
                result = await _generate_draft(client, plan, max_retries=max_retries)
                return plan.index, plan.question_type, result

        generated_by_index: dict[int, tuple[GoldenDraft, tuple[int, ...]]] = {}
        tasks = [asyncio.create_task(generate(plan)) for plan in plans]
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            index, question_type, result = await task
            generated_by_index[index] = result
            if progress is not None:
                progress(completed, len(plans), question_type)
        generated = [generated_by_index[plan.index] for plan in plans]
        seen: set[str] = set()
        cases: list[dict[str, object]] = []
        for plan, (draft, used_contexts) in zip(plans, generated, strict=True):
            normalized_question = " ".join(draft.user_input.split()).casefold()
            if normalized_question in seen:
                async with semaphore:
                    draft, used_contexts = await _generate_draft(
                        client,
                        plan,
                        max_retries=max_retries,
                        extra_issue=(
                            "The previous question duplicated another case; create a distinct one."
                        ),
                    )
                normalized_question = " ".join(draft.user_input.split()).casefold()
                if normalized_question in seen:
                    raise ValueError(
                        f"case {plan.index} still duplicates another generated question"
                    )
            seen.add(normalized_question)
            cases.append(_case_from_draft(plan, draft, used_contexts, llm_model=llm_model))
    return cases


def build_evaluation_testset_envelope(
    corpus_content_hash: object,
    cases: Sequence[dict[str, object]],
    *,
    sources: object | None = None,
) -> dict[str, object]:
    """Build the project test-set envelope consumed by the Ragas scoring runner."""

    normalized_hash = validate_corpus_content_hash(corpus_content_hash)
    if not cases:
        raise ValueError("evaluation test set requires at least one case")
    envelope: dict[str, object] = {
        "schema_version": EVALUATION_TESTSET_SCHEMA_VERSION,
        "corpus_content_hash": normalized_hash,
        "cases": [_validated_case(case, index=index) for index, case in enumerate(cases, start=1)],
    }
    if sources is not None:
        envelope["sources"] = validate_testset_sources(sources)
    return envelope


def write_evaluation_testset(path: Path, envelope: dict[str, object]) -> Path:
    """Write a generated test-set envelope as deterministic UTF-8 JSON."""

    normalized_envelope = _validated_envelope(envelope)
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(normalized_envelope, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


async def _generate_draft(
    client: DeepSeekClient,
    plan: GoldenCasePlan,
    *,
    max_retries: int,
    extra_issue: str | None = None,
) -> tuple[GoldenDraft, tuple[int, ...]]:
    previous: str | None = None
    issues = [extra_issue] if extra_issue else []
    for attempt in range(max_retries + 1):
        messages = _golden_messages(plan, invalid_response=previous, issues=issues)
        try:
            completion = await client.complete_messages(messages)
        except RetryableProviderError as error:
            if attempt >= max_retries:
                raise
            issues = [f"Provider error: {type(error).__name__}: {error}"]
            continue
        previous = completion.content
        try:
            if completion.finish_reason != "stop":
                raise ValueError(f"finish_reason must be 'stop', got {completion.finish_reason!r}")
            if not previous or not previous.strip():
                raise ValueError("provider returned empty content")
            draft = GoldenDraft.model_validate_json(previous)
            if draft.insufficient_evidence == plan.answerable:
                raise ValueError(
                    "insufficient_evidence must be "
                    f"{not plan.answerable} for {plan.question_type.value}"
                )
            used_contexts = _used_context_indexes(draft.evidence_quotes, plan.contexts)
            if plan.answerable and not used_contexts:
                raise ValueError("no evidence quote is verbatim in the selected contexts")
            if plan.question_type is QuestionType.MULTI_CONTEXT and len(used_contexts) < 2:
                raise ValueError("multi_context cases must quote at least two selected contexts")
            return draft, used_contexts
        except (ValidationError, ValueError) as error:
            if attempt >= max_retries:
                raise ValueError(
                    f"case {plan.index} failed validation after {max_retries + 1} attempts: {error}"
                ) from error
            issues = [f"{type(error).__name__}: {error}"]
    raise AssertionError("golden generation retry loop did not return")


def _golden_messages(
    plan: GoldenCasePlan,
    *,
    invalid_response: str | None,
    issues: Sequence[str],
) -> tuple[ChatMessage, ...]:
    context_blocks = []
    for index, document in enumerate(plan.contexts, start=1):
        location = f"pages {document.page_start}-{document.page_end}"
        section = " > ".join(document.section_path) or "<no section>"
        context_blocks.append(
            f"[E{index}] document={document.document_title!r}; {location}; section={section!r}\n"
            f"{document.text}"
        )
    task = {
        QuestionType.SINGLE_HOP: (
            "Create one focused factual or conceptual question answerable from one evidence block."
        ),
        QuestionType.SUMMARY_REASONING: (
            "Create one question requiring a method summary, conclusion, or reasoning across the "
            "selected evidence from the same paper."
        ),
        QuestionType.MULTI_CONTEXT: (
            "Create one comparison or synthesis question that genuinely requires at least two "
            "evidence blocks."
        ),
        QuestionType.UNANSWERABLE: (
            "Create one plausible research question whose requested detail is not stated by the "
            "evidence. The reference must explicitly say the evidence is insufficient."
        ),
    }[plan.question_type]
    system = (
        "You create auditable RAG evaluation cases from scientific papers. Return JSON only. "
        "Write user_input and reference in Chinese. First derive concise grounding_statements, "
        "then form the question. evidence_quotes must be short verbatim substrings copied from "
        "the supplied evidence, never paraphrases. Do not use outside knowledge. "
        "For unanswerable cases, grounding_statements and evidence_quotes must be empty and "
        "insufficient_evidence must be true. For all other cases it must be false.\n"
        'Schema: {"user_input":str,"reference":str,"grounding_statements":[str],'
        '"evidence_quotes":[str],"insufficient_evidence":bool}.'
    )
    user = f"Question type: {plan.question_type.value}\nTask: {task}\n\n" + "\n\n".join(
        context_blocks
    )
    if invalid_response is not None or issues:
        rendered_issues = "\n".join(f"- {issue}" for issue in issues)
        user += (
            "\n\nYour previous response was invalid. Correct every issue and return a complete "
            f"replacement JSON object.\nIssues:\n{rendered_issues}\n"
            f"Previous response:\n{invalid_response or '<empty>'}"
        )
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    )


def _used_context_indexes(
    quotes: Sequence[str], contexts: Sequence[EvaluationDocument]
) -> tuple[int, ...]:
    used: set[int] = set()
    for quote in quotes:
        matches = [index for index, context in enumerate(contexts) if quote in context.text]
        if not matches:
            raise ValueError(f"evidence quote is not verbatim: {quote!r}")
        used.add(matches[0])
    return tuple(sorted(used))


def _case_from_draft(
    plan: GoldenCasePlan,
    draft: GoldenDraft,
    used_contexts: Sequence[int],
    *,
    llm_model: str,
) -> dict[str, object]:
    relevant = tuple(plan.contexts[index] for index in used_contexts)
    case_contexts = relevant if plan.answerable else plan.contexts
    case_documents = tuple(dict.fromkeys(item.document_id for item in case_contexts))
    return _validated_case(
        {
            "user_input": draft.user_input,
            "reference": draft.reference,
            "reference_contexts": [item.text for item in case_contexts],
            "evidence_ids": _flatten_evidence_ids(relevant),
            "context_evidence_ids": _flatten_evidence_ids(plan.contexts),
            "document_ids": list(case_documents),
            "question_type": plan.question_type.value,
            "answerable": plan.answerable,
            "evidence_quotes": list(draft.evidence_quotes),
            "generator_model": llm_model,
            "prompt_version": GOLDEN_PROMPT_VERSION,
            "review_status": "unreviewed",
        },
        index=plan.index,
    )


def _validated_case(value: object, *, index: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"evaluation test set item {index} must be an object")
    user_input = _required_string(value, "user_input", index)
    reference = _required_string(value, "reference", index)
    reference_contexts = _string_list(value.get("reference_contexts"), "reference_contexts", index)
    evidence_ids = _string_list(value.get("evidence_ids"), "evidence_ids", index)
    context_evidence_ids = _string_list(
        value.get("context_evidence_ids"), "context_evidence_ids", index
    )
    document_ids = _string_list(value.get("document_ids"), "document_ids", index)
    evidence_quotes = _string_list(value.get("evidence_quotes"), "evidence_quotes", index)
    question_type = _required_string(value, "question_type", index)
    try:
        normalized_type = QuestionType(question_type).value
    except ValueError as error:
        raise ValueError(f"evaluation test set item {index} has invalid question_type") from error
    answerable = value.get("answerable")
    if not isinstance(answerable, bool):
        raise ValueError(f"evaluation test set item {index} answerable must be boolean")
    if answerable and (not evidence_ids or not evidence_quotes):
        raise ValueError(f"answerable test set item {index} requires evidence IDs and quotes")
    if not answerable and evidence_ids:
        raise ValueError(f"unanswerable test set item {index} must not carry relevant evidence IDs")
    if not reference_contexts or not context_evidence_ids or not document_ids:
        raise ValueError(f"evaluation test set item {index} requires source contexts and documents")
    return {
        "user_input": user_input,
        "reference": reference,
        "reference_contexts": reference_contexts,
        "evidence_ids": evidence_ids,
        "context_evidence_ids": context_evidence_ids,
        "document_ids": document_ids,
        "question_type": normalized_type,
        "answerable": answerable,
        "evidence_quotes": evidence_quotes,
        "generator_model": _required_string(value, "generator_model", index),
        "prompt_version": _required_string(value, "prompt_version", index),
        "review_status": _required_string(value, "review_status", index),
    }


def _validated_envelope(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("evaluation test set envelope must be an object")
    if value.get("schema_version") != EVALUATION_TESTSET_SCHEMA_VERSION:
        raise ValueError(
            "unsupported evaluation test set schema_version "
            f"{value.get('schema_version')!r}; expected {EVALUATION_TESTSET_SCHEMA_VERSION!r}"
        )
    cases = value.get("cases")
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError("evaluation test set envelope cases must be an array of objects")
    return build_evaluation_testset_envelope(
        value.get("corpus_content_hash"),
        cast(list[dict[str, object]], cases),
        sources=value.get("sources"),
    )


def _distribution_counts(testset_size: int) -> dict[QuestionType, int]:
    raw = {
        question_type: testset_size * DEFAULT_QUESTION_DISTRIBUTION[question_type.value]
        for question_type in QuestionType
    }
    counts = {question_type: int(value) for question_type, value in raw.items()}
    remaining = testset_size - sum(counts.values())
    order = sorted(
        QuestionType,
        key=lambda item: (-(raw[item] - counts[item]), tuple(QuestionType).index(item)),
    )
    for question_type in order[:remaining]:
        counts[question_type] += 1
    return counts


def _document_sort_key(value: EvaluationDocument) -> tuple[object, ...]:
    return (
        value.page_start if value.page_start is not None else 10**9,
        value.page_end if value.page_end is not None else 10**9,
        value.section_path,
        value.ordinal,
        value.evidence_ids,
    )


def _related_context(
    primary: EvaluationDocument,
    candidates: Sequence[EvaluationDocument],
) -> EvaluationDocument:
    primary_terms = _lexical_terms(primary.text)
    return min(
        candidates,
        key=lambda candidate: (
            -_jaccard(primary_terms, _lexical_terms(candidate.text)),
            _document_sort_key(candidate),
        ),
    )


def _lexical_terms(value: str) -> frozenset[str]:
    return frozenset(
        token
        for match in _LEXICAL_TOKEN.finditer(value)
        if (token := match.group(0).casefold()) not in _LEXICAL_STOPWORDS
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _unique_contexts(
    values: Sequence[EvaluationDocument],
) -> tuple[EvaluationDocument, ...]:
    return tuple({value.evidence_ids: value for value in values}.values())


def _flatten_evidence_ids(values: Sequence[EvaluationDocument]) -> list[str]:
    return list(dict.fromkeys(identity for value in values for identity in value.evidence_ids))


def _discover_source_files(source: Path, registry: LoaderRegistry) -> tuple[Path, tuple[Path, ...]]:
    resolved = source.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if resolved.is_file():
        registry.for_path(resolved)
        return resolved.parent, (resolved,)
    paths = tuple(
        sorted(
            (
                path
                for path in resolved.rglob("*")
                if path.is_file() and path.suffix.lower() in registry.supported_suffixes
            ),
            key=lambda path: path.as_posix().casefold(),
        )
    )
    if not paths:
        suffixes = ", ".join(sorted(registry.supported_suffixes))
        raise FileNotFoundError(f"no supported files found in {resolved} ({suffixes})")
    return resolved, paths


def _required_string(value: Mapping[str, object], field: str, index: int) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"evaluation test set item {index} {field} must be a non-empty string")
    return item.strip()


def _string_list(value: object, field: str, index: int) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"evaluation test set item {index} {field} must contain strings")
    normalized = [item.strip() for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"evaluation test set item {index} {field} contains duplicates")
    return normalized


def _validate_limit(value: int | None, *, field: str) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{field} must be positive when provided")


__all__ = [
    "DEFAULT_QUESTION_DISTRIBUTION",
    "EVALUATION_TESTSET_SCHEMA_VERSION",
    "GOLDEN_PROMPT_VERSION",
    "EvaluationDocument",
    "GoldenCasePlan",
    "QuestionType",
    "build_evaluation_testset_envelope",
    "generate_golden_cases",
    "load_evaluation_documents",
    "plan_golden_cases",
    "validate_corpus_content_hash",
    "validate_testset_sources",
    "write_evaluation_testset",
]
