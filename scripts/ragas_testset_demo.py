"""Generate a corpus-bound Ragas test set from supported files in ``data/raw``.

The reusable helpers live in :mod:`hybrid_rag.evaluation.testset`. This CLI
keeps a small default sample for cost-safe demos; pass ``--all-documents`` to
load the entire source directory recursively.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from hybrid_rag.evaluation.testset import (
    build_ragas_testset_envelope,
    generate_ragas_cases,
    load_ragas_documents,
    validate_corpus_content_hash,
    write_ragas_testset,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "processed" / "ragas-testset-demo.json"
load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--corpus-content-hash",
        metavar="SHA256",
        help=(
            "64-character lowercase corpus-content hash from the exact index profile to "
            "evaluate; required unless --dry-run is used"
        ),
    )
    parser.add_argument("--testset-size", type=int, default=5)
    parser.add_argument(
        "--max-documents",
        type=int,
        default=2,
        help="Maximum supported source files to read; use 0 for all files.",
    )
    parser.add_argument(
        "--max-segments-per-document",
        "--pages-per-document",
        dest="max_segments_per_document",
        type=int,
        default=6,
        help="Maximum loader segments per file; use 0 for all segments.",
    )
    parser.add_argument(
        "--all-documents",
        action="store_true",
        help="Read every supported file and every segment recursively; this can be costly.",
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


def main() -> None:
    args = parse_args()
    if args.testset_size < 1:
        raise ValueError("testset_size must be positive")
    max_documents, max_segments = _source_limits(args)
    documents = load_ragas_documents(
        args.source_dir,
        max_documents=max_documents,
        max_segments_per_document=max_segments,
    )
    print(f"Loaded {len(documents)} source segments from {args.source_dir}")
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
    envelope = build_ragas_testset_envelope(corpus_hash, cases)
    destination = write_ragas_testset(args.output, envelope)
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


if __name__ == "__main__":
    main()
