r"""Report how well a generated golden test set covers its source documents.

The script reloads the source corpus with the project PDF loader and maps every
``reference_context`` back to its source document. It does not call an LLM.

Usage::

    # Inspect the default test set against data/corpus.
    uv run scripts/ragas_coverage.py

    # Inspect another test set and corpus.
    uv run scripts/ragas_coverage.py \
      --testset artifacts/ragas/release-testset.json \
      --source-dir storage/workspaces/<workspace-id>/uploads

    # Fail when coverage is incomplete or a document has fewer than five cases.
    uv run scripts/ragas_coverage.py --strict --min-questions-per-document 5

    # Emit a machine-readable report.
    uv run scripts/ragas_coverage.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hybrid_rag.evaluation.testset import EvaluationDocument, load_evaluation_documents

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTSET_PATH = ROOT / "artifacts" / "ragas" / "ragas-testset.json"
DEFAULT_SOURCE_PATH = ROOT / "data" / "corpus"
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class DocumentCoverage:
    document_id: str
    source_uri: str
    title: str
    questions: int
    contexts: int


@dataclass(frozen=True, slots=True)
class CoverageReport:
    testset: str
    source_dir: str
    cases: int
    documents: int
    covered_documents: int
    coverage_ratio: float
    multi_document_cases: int
    unmatched_contexts: int
    ambiguous_contexts: int
    cases_without_source: int
    min_questions_per_document: int
    documents_below_minimum: tuple[str, ...]
    document_coverage: tuple[DocumentCoverage, ...]

    @property
    def acceptable(self) -> bool:
        return (
            self.covered_documents == self.documents
            and not self.documents_below_minimum
            and self.unmatched_contexts == 0
            and self.ambiguous_contexts == 0
            and self.cases_without_source == 0
        )


@dataclass(frozen=True, slots=True)
class _SourceSegment:
    document_id: str
    source_uri: str
    title: str
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TestsetData:
    cases: list[dict[str, object]]
    source_titles: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--testset",
        type=Path,
        default=DEFAULT_TESTSET_PATH,
        help="Generated golden test-set JSON (default: artifacts/ragas/ragas-testset.json).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Source document directory (default: data/corpus).",
    )
    parser.add_argument(
        "--min-questions-per-document",
        type=int,
        default=5,
        help="Recommended minimum question count per document (default: 5).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when coverage checks do not pass.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_questions_per_document < 1:
        raise ValueError("--min-questions-per-document must be positive")
    testset = _load_testset(args.testset)
    documents = load_evaluation_documents(args.source_dir)
    report = _coverage_report(
        testset.cases,
        documents,
        testset=args.testset,
        source_dir=args.source_dir,
        minimum=args.min_questions_per_document,
        source_titles=testset.source_titles,
    )
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    if args.strict and not report.acceptable:
        sys.exit(1)


def _load_testset(path: Path) -> _TestsetData:
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read golden test set {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("golden test set must be a JSON envelope")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("golden test set cases must be a non-empty array")
    if not all(isinstance(case, dict) for case in cases):
        raise ValueError("every golden test set case must be an object")
    source_titles: dict[str, str] = {}
    sources = value.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            source_uri = source.get("source_uri")
            title = source.get("title")
            if isinstance(source_uri, str) and isinstance(title, str) and title.strip():
                source_titles[source_uri] = title.strip()
    return _TestsetData(cases=cases, source_titles=source_titles)


def _coverage_report(
    cases: list[dict[str, object]],
    documents: list[EvaluationDocument],
    *,
    testset: Path,
    source_dir: Path,
    minimum: int,
    source_titles: dict[str, str],
) -> CoverageReport:
    segments = tuple(_source_segment(document) for document in documents)
    documents_by_id: dict[str, _SourceSegment] = {}
    exact: dict[str, list[_SourceSegment]] = defaultdict(list)
    by_evidence_id: dict[str, _SourceSegment] = {}
    for segment in segments:
        documents_by_id.setdefault(segment.document_id, segment)
        exact[_normalize(segment.text)].append(segment)
        for evidence_id in segment.evidence_ids:
            by_evidence_id.setdefault(evidence_id, segment)

    question_cases: dict[str, set[int]] = defaultdict(set)
    context_counts: dict[str, int] = defaultdict(int)
    unmatched_contexts = 0
    ambiguous_contexts = 0
    cases_without_source = 0
    multi_document_cases = 0
    for case_index, case in enumerate(cases, start=1):
        direct_document_ids = case.get("document_ids")
        context_evidence_ids = case.get("context_evidence_ids")
        if isinstance(direct_document_ids, list) and all(
            isinstance(item, str) and item in documents_by_id for item in direct_document_ids
        ):
            case_document_ids = set(direct_document_ids)
            if isinstance(context_evidence_ids, list) and all(
                isinstance(item, str) for item in context_evidence_ids
            ):
                for evidence_id in context_evidence_ids:
                    source = by_evidence_id.get(evidence_id)
                    if source is None:
                        unmatched_contexts += 1
                    else:
                        context_counts[source.document_id] += 1
            for document_id in case_document_ids:
                question_cases[document_id].add(case_index)
            if len(case_document_ids) > 1:
                multi_document_cases += 1
            continue
        contexts = case.get("reference_contexts")
        if not isinstance(contexts, list) or not all(isinstance(item, str) for item in contexts):
            raise ValueError(f"golden test set case {case_index} has invalid reference_contexts")
        case_document_ids: set[str] = set()
        for context in contexts:
            matches = _match_context(context, exact=exact, segments=segments)
            document_ids = {match.document_id for match in matches}
            if not document_ids:
                unmatched_contexts += 1
                continue
            if len(document_ids) > 1:
                ambiguous_contexts += 1
                continue
            document_id = next(iter(document_ids))
            case_document_ids.add(document_id)
            context_counts[document_id] += 1
        if not case_document_ids:
            cases_without_source += 1
        if len(case_document_ids) > 1:
            multi_document_cases += 1
        for document_id in case_document_ids:
            question_cases[document_id].add(case_index)

    coverage = tuple(
        DocumentCoverage(
            document_id=document_id,
            source_uri=source.source_uri,
            title=source_titles.get(source.source_uri, source.title),
            questions=len(question_cases[document_id]),
            contexts=context_counts[document_id],
        )
        for document_id, source in sorted(
            documents_by_id.items(),
            key=lambda item: (
                source_titles.get(item[1].source_uri, item[1].title).casefold(),
                item[0],
            ),
        )
    )
    covered = sum(item.questions > 0 for item in coverage)
    below_minimum = tuple(item.title for item in coverage if item.questions < minimum)
    return CoverageReport(
        testset=str(testset.expanduser().resolve()),
        source_dir=str(source_dir.expanduser().resolve()),
        cases=len(cases),
        documents=len(coverage),
        covered_documents=covered,
        coverage_ratio=covered / len(coverage) if coverage else 0.0,
        multi_document_cases=multi_document_cases,
        unmatched_contexts=unmatched_contexts,
        ambiguous_contexts=ambiguous_contexts,
        cases_without_source=cases_without_source,
        min_questions_per_document=minimum,
        documents_below_minimum=below_minimum,
        document_coverage=coverage,
    )


def _source_segment(document: EvaluationDocument) -> _SourceSegment:
    return _SourceSegment(
        document.document_id,
        document.source_uri,
        document.document_title,
        document.text,
        document.evidence_ids,
    )


def _match_context(
    context: str,
    *,
    exact: dict[str, list[_SourceSegment]],
    segments: tuple[_SourceSegment, ...],
) -> tuple[_SourceSegment, ...]:
    normalized = _normalize(context)
    if not normalized:
        return ()
    direct = exact.get(normalized)
    if direct:
        return tuple(direct)
    if len(normalized) < 80:
        return ()
    return tuple(
        segment
        for segment in segments
        if normalized in (segment_text := _normalize(segment.text)) or segment_text in normalized
    )


def _normalize(value: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _print_report(report: CoverageReport) -> None:
    console = Console()
    status = "PASS" if report.acceptable else "REVIEW"
    color = "green" if report.acceptable else "yellow"
    console.print(
        f"[{color}]{status}[/{color}] document coverage: "
        f"{report.covered_documents}/{report.documents} "
        f"({report.coverage_ratio:.1%}); cases={report.cases}"
    )
    console.print(
        f"multi-document cases={report.multi_document_cases}; "
        f"unmatched contexts={report.unmatched_contexts}; "
        f"ambiguous contexts={report.ambiguous_contexts}; "
        f"cases without source={report.cases_without_source}"
    )
    table = Table("Document", "Questions", "Contexts", "Case share", box=None)
    for item in report.document_coverage:
        style = "yellow" if item.questions < report.min_questions_per_document else None
        table.add_row(
            item.title,
            str(item.questions),
            str(item.contexts),
            f"{item.questions / report.cases:.1%}",
            style=style,
        )
    console.print(table)
    if report.documents_below_minimum:
        console.print(
            f"[yellow]Below {report.min_questions_per_document} questions:[/yellow] "
            + "; ".join(report.documents_below_minimum)
        )


if __name__ == "__main__":
    main()
