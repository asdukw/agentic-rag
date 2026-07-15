r"""Generate a corpus-bound Ragas test set from local source documents.

By default, the script reads every supported document under ``data/corpus`` and
writes the generated envelope to ``artifacts/ragas/ragas-testset.json``.

Usage::

    # Inspect the source scope and planned output without calling DeepSeek.
    uv run scripts/ragas_testset.py --dry-run

    # Generate the default 30-case test set for an exact index corpus.
    uv run scripts/ragas_testset.py \
      --corpus-content-hash <build-index-corpus-content-hash>

    # Override the source, total sample count, and output location.
    uv run scripts/ragas_testset.py \
      --source-dir storage/workspaces/<workspace-id>/uploads \
      --testset-size 60 \
      --corpus-content-hash <build-index-corpus-content-hash> \
      --output artifacts/ragas/release-testset.json

Existing output files are never overwritten; ``-1``, ``-2``, and subsequent
numeric suffixes are selected automatically.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from hybrid_rag.evaluation.testset import (
    build_ragas_testset_envelope,
    generate_ragas_cases,
    load_ragas_documents,
    validate_corpus_content_hash,
    validate_testset_sources,
    write_ragas_testset,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = ROOT / "data" / "corpus"
DEFAULT_OUTPUT_PATH = ROOT / "artifacts" / "ragas" / "ragas-testset.json"
DEFAULT_TESTSET_SIZE = 30
load_dotenv(ROOT / ".env")


@dataclass(frozen=True, slots=True)
class ScriptDefaults:
    output: Path = DEFAULT_OUTPUT_PATH
    testset_size: int = DEFAULT_TESTSET_SIZE
    max_documents: int = 0
    max_segments_per_document: int = 0
    description: str | None = None


def parse_args(defaults: ScriptDefaults | None = None) -> argparse.Namespace:
    selected = defaults or ScriptDefaults()
    parser = argparse.ArgumentParser(
        description=selected.description or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Source document directory (default: data/corpus).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=selected.output,
        help=f"Generated test-set JSON (default: {_display_path(selected.output)}).",
    )
    parser.add_argument(
        "--corpus-content-hash",
        metavar="SHA256",
        help=(
            "64-character lowercase corpus-content hash from the exact index profile to "
            "evaluate; required unless --dry-run is used"
        ),
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=None,
        help="Source citation manifest (default: SOURCES.json inside --source-dir).",
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=selected.testset_size,
        help=f"Total number of generated cases (default: {selected.testset_size}).",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=selected.max_documents,
        help=(
            "Maximum supported source files to read; "
            f"0 reads all files (default: {selected.max_documents})."
        ),
    )
    parser.add_argument(
        "--max-segments-per-document",
        "--pages-per-document",
        dest="max_segments_per_document",
        type=int,
        default=selected.max_segments_per_document,
        help=(
            "Maximum loader segments per file; "
            f"0 reads all segments (default: {selected.max_segments_per_document})."
        ),
    )
    parser.add_argument(
        "--all-documents",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("DEEPSEEK_EXTRACTION_MODEL", "deepseek-v4-flash"),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        help="DeepSeek's OpenAI-compatible endpoint URL.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load files without calling an API.")
    return parser.parse_args()


def main(defaults: ScriptDefaults | None = None) -> None:
    args = parse_args(defaults)
    if args.testset_size < 1:
        raise ValueError("testset_size must be positive")
    max_documents, max_segments = _source_limits(args)
    documents = load_ragas_documents(
        args.source_dir,
        max_documents=max_documents,
        max_segments_per_document=max_segments,
    )
    document_ids = {
        document_id
        for document in documents
        if isinstance((document_id := document.metadata.get("document_id")), str) and document_id
    }
    source_uris = {
        source_uri
        for document in documents
        if isinstance((source_uri := document.metadata.get("source")), str) and source_uri
    }
    sources_path = args.sources or args.source_dir / "SOURCES.json"
    sources = _load_sources(sources_path, source_uris)
    output_path = _available_output_path(args.output)
    print(
        f"Loaded {len(document_ids)} source documents ({len(documents)} segments) "
        f"from {args.source_dir}"
    )
    print(f"Test-set size: {args.testset_size}; output: {output_path}")
    print(f"Source citations: {len(sources)} from {sources_path}")
    if args.dry_run:
        return

    corpus_hash = validate_corpus_content_hash(
        args.corpus_content_hash,
        field="--corpus-content-hash",
    )
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("set DEEPSEEK_API_KEY in .env before generating a Ragas test set")
    cases = generate_ragas_cases(
        documents,
        api_key=api_key,
        llm_model=args.llm_model,
        base_url=args.base_url,
        testset_size=args.testset_size,
    )
    envelope = build_ragas_testset_envelope(corpus_hash, cases, sources=sources)
    destination = write_ragas_testset(output_path, envelope)
    print(f"Generated {len(cases)} cases: {destination}")


def _source_limits(args: argparse.Namespace) -> tuple[int | None, int | None]:
    if args.all_documents:
        return None, None
    return (
        _limit_or_all(args.max_documents, flag="--max-documents"),
        _limit_or_all(
            args.max_segments_per_document,
            flag="--max-segments-per-document",
        ),
    )


def _limit_or_all(value: int, *, flag: str) -> int | None:
    if value < 0:
        raise ValueError(f"{flag} must be non-negative")
    return None if value == 0 else value


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _available_output_path(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.exists():
        return candidate
    for index in range(1, 10_000):
        renamed = candidate.with_name(f"{candidate.stem}-{index}{candidate.suffix}")
        if not renamed.exists():
            return renamed
    raise RuntimeError(f"unable to find an available output name for {candidate}")


def _load_sources(path: Path, selected_source_uris: set[str]) -> list[dict[str, object]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read source citation manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"source citation manifest {path} must be a JSON object")
    sources = validate_testset_sources(value.get("sources"))
    by_uri = {str(source["source_uri"]): source for source in sources}
    missing = sorted(selected_source_uris - by_uri.keys())
    if missing:
        raise ValueError(f"source citation manifest {path} is missing: {', '.join(missing)}")
    return [source for source in sources if str(source["source_uri"]) in selected_source_uris]


if __name__ == "__main__":
    main()
