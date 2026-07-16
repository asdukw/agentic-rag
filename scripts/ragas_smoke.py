r"""Generate a small, cost-bounded golden test set for pipeline smoke checks.

The smoke profile loads at most two documents and six segments per document,
generates six cases, and writes to
``artifacts/ragas/ragas-smoke-testset.json`` without overwriting existing files.

Usage::

    # Check the selected documents, segments, citations, and output without an API call.
    uv run scripts/ragas_smoke.py --dry-run

    # Generate six smoke cases for the exact corpus used by the index.
    uv run scripts/ragas_smoke.py \
      --corpus-content-hash <build-index-corpus-content-hash>

    # Smoke-check a workspace corpus and its source citation manifest.
    uv run scripts/ragas_smoke.py \
      --source-dir storage/workspaces/<workspace-id>/uploads \
      --sources storage/workspaces/<workspace-id>/uploads/SOURCES.json \
      --corpus-content-hash <build-index-corpus-content-hash>
"""

from ragas_testset import ROOT, ScriptDefaults, main

SMOKE_DEFAULTS = ScriptDefaults(
    output=ROOT / "artifacts" / "ragas" / "ragas-smoke-testset.json",
    testset_size=6,
    max_documents=2,
    max_segments_per_document=6,
    min_cases_per_document=1,
    description=__doc__,
)


if __name__ == "__main__":
    main(SMOKE_DEFAULTS)
