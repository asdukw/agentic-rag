r"""Evaluate all retrieval modes with reranking disabled.

The wrapper runs the deterministic six-case smoke subset by default. Pass
``--full`` explicitly to evaluate the complete test set. All other arguments
are forwarded to ``hrag evaluate``.

Usage::

    uv run scripts/evaluate_no_rerank.py \
      --testset artifacts/ragas/ragas-testset-1.json

    uv run scripts/evaluate_no_rerank.py \
      --testset artifacts/ragas/ragas-testset-1.json \
      --full
"""

from __future__ import annotations

import os
import sys

DEFAULT_MODES = "dense,bm25,hybrid,graph_local,graph_global,graph_hybrid,mix,agentic"
DEFAULT_SMOKE_OUTPUT = "artifacts/evaluations/six-modes-smoke-no-rerank.json"
DEFAULT_FULL_OUTPUT = "artifacts/evaluations/six-modes-no-rerank.json"


def main() -> None:
    forwarded = [argument for argument in sys.argv[1:] if argument != "--full"]
    full = len(forwarded) != len(sys.argv[1:])
    os.environ["HYBRID_RAG_RETRIEVAL_RERANKER_PROVIDER"] = "none"

    arguments = ["evaluate", *forwarded]
    if not _has_option(forwarded, "--modes"):
        arguments.extend(("--modes", DEFAULT_MODES))
    if not full:
        arguments.append("--smoke")
    if not _has_option(forwarded, "--output"):
        arguments.extend(("--output", DEFAULT_FULL_OUTPUT if full else DEFAULT_SMOKE_OUTPUT))
    arguments.append("--no-agentic-rerank")

    from hybrid_rag.cli import app

    sys.argv = [sys.argv[0], *arguments]
    app()


def _has_option(arguments: list[str], name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in arguments)


if __name__ == "__main__":
    main()
