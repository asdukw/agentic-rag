"""Migrate traces to industry-aligned retrieval strategy names.

Revision ID: 0007_retrieval_strategy_names
Revises: 0006_chunk_quality_class
Create Date: 2026-07-21
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

from hybrid_rag.ids import canonical_json_hash

revision: str = "0007_retrieval_strategy_names"
down_revision: str | None = "0006_chunk_quality_class"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONSTRAINT = "ck_retrieval_traces_mode"
_OLD_MODES = "mode IN ('naive', 'local', 'global', 'hybrid', 'mix')"
_NEW_MODES = (
    "mode IN ('dense', 'bm25', 'hybrid', 'graph_local', 'graph_global', "
    "'graph_hybrid', 'mix')"
)
_TRANSITION_MODES = (
    "mode IN ('naive', 'local', 'global', 'dense', 'bm25', 'hybrid', "
    "'graph_local', 'graph_global', 'graph_hybrid', 'mix')"
)
_MODE_RENAMES = {
    "naive": "hybrid",
    "local": "graph_local",
    "global": "graph_global",
    "hybrid": "graph_hybrid",
    "mix": "mix",
}
_MODE_RENAMES_REVERSE = {value: key for key, value in _MODE_RENAMES.items()}
_SETTING_RENAMES = {
    "naive_weight": "hybrid_weight",
    "local_weight": "graph_local_weight",
    "global_weight": "graph_global_weight",
    "naive_dense_weight": "hybrid_dense_weight",
    "naive_bm25_weight": "hybrid_bm25_weight",
    "naive_lexical_scorer": "hybrid_lexical_scorer",
    "naive_lexical_tokenizer": "hybrid_lexical_tokenizer",
}
_SETTING_RENAMES_REVERSE = {value: key for key, value in _SETTING_RENAMES.items()}
_HASH_KEYS = frozenset(
    {
        "top_k",
        "candidate_multiplier",
        "context_token_budget",
        "graph_max_hops",
        "hybrid_weight",
        "graph_local_weight",
        "graph_global_weight",
        "hybrid_dense_weight",
        "hybrid_bm25_weight",
        "bm25_k1",
        "bm25_b",
        "hybrid_lexical_scorer",
        "hybrid_lexical_tokenizer",
        "mode_semantics_version",
        "rerank_enabled",
        "reranker_provider",
        "reranker_model",
        "reranker_use_fp16",
        "reranker_version",
        "rerank_candidate_multiplier",
        "graph_path_weight",
        "graph_path_candidate_multiplier",
        "multi_context_weight",
    }
)


def upgrade() -> None:
    connection = op.get_bind()
    with op.batch_alter_table("retrieval_traces", recreate="always") as batch:
        batch.drop_constraint(_CONSTRAINT, type_="check")
        batch.create_check_constraint(_CONSTRAINT, _TRANSITION_MODES)
    _migrate_traces(
        connection,
        mode_renames=_MODE_RENAMES,
        setting_renames=_SETTING_RENAMES,
        schema_version="4",
        semantics_version="hybrid-search-v3",
    )
    with op.batch_alter_table("retrieval_traces", recreate="always") as batch:
        batch.drop_constraint(_CONSTRAINT, type_="check")
        batch.create_check_constraint(_CONSTRAINT, _NEW_MODES)


def downgrade() -> None:
    connection = op.get_bind()
    unsupported = connection.execute(
        sa.text("SELECT 1 FROM retrieval_traces WHERE mode IN ('dense', 'bm25') LIMIT 1")
    ).first()
    if unsupported is not None:
        raise RuntimeError("cannot downgrade traces produced by dense or bm25 strategies")
    with op.batch_alter_table("retrieval_traces", recreate="always") as batch:
        batch.drop_constraint(_CONSTRAINT, type_="check")
        batch.create_check_constraint(_CONSTRAINT, _TRANSITION_MODES)
    _migrate_traces(
        connection,
        mode_renames=_MODE_RENAMES_REVERSE,
        setting_renames=_SETTING_RENAMES_REVERSE,
        schema_version="3",
        semantics_version="lightrag-v2",
    )
    with op.batch_alter_table("retrieval_traces", recreate="always") as batch:
        batch.drop_constraint(_CONSTRAINT, type_="check")
        batch.create_check_constraint(_CONSTRAINT, _OLD_MODES)


def _migrate_traces(
    connection: sa.Connection,
    *,
    mode_renames: Mapping[str, str],
    setting_renames: Mapping[str, str],
    schema_version: str,
    semantics_version: str,
) -> None:
    traces = sa.table(
        "retrieval_traces",
        sa.column("id", sa.String(length=64)),
        sa.column("mode", sa.String(length=32)),
        sa.column("retrieval_config_hash", sa.String(length=64)),
        sa.column("trace_json", sa.JSON()),
        sa.column("output_json", sa.JSON()),
    )
    rows = connection.execute(
        sa.select(
            traces.c.id,
            traces.c.mode,
            traces.c.trace_json,
            traces.c.output_json,
        )
    ).mappings().all()
    for row in rows:
        trace = _rename_payload(
            row["trace_json"],
            mode_renames=mode_renames,
            setting_renames=setting_renames,
            schema_version=schema_version,
            semantics_version=semantics_version,
        )
        output = _rename_payload(
            row["output_json"],
            mode_renames=mode_renames,
            setting_renames=setting_renames,
            schema_version=schema_version,
            semantics_version=semantics_version,
        )
        settings = trace.get("settings", {}) if isinstance(trace, dict) else {}
        config_hash = canonical_json_hash(
            {key: value for key, value in settings.items() if key in _HASH_KEYS}
        )
        connection.execute(
            sa.update(traces)
            .where(traces.c.id == row["id"])
            .values(
                mode=mode_renames.get(str(row["mode"]), str(row["mode"])),
                retrieval_config_hash=config_hash,
                trace_json=trace,
                output_json=output,
            )
        )


def _rename_payload(
    value: Any,
    *,
    mode_renames: Mapping[str, str],
    setting_renames: Mapping[str, str],
    schema_version: str,
    semantics_version: str,
) -> Any:
    if isinstance(value, list):
        return [
            _rename_payload(
                item,
                mode_renames=mode_renames,
                setting_renames=setting_renames,
                schema_version=schema_version,
                semantics_version=semantics_version,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    renamed: dict[str, Any] = {}
    for key, item in value.items():
        renamed_key = setting_renames.get(key, key)
        if key in {"routes", "route_scores"} and isinstance(item, dict):
            renamed[renamed_key] = {
                mode_renames.get(route, route): _rename_payload(
                    route_value,
                    mode_renames=mode_renames,
                    setting_renames=setting_renames,
                    schema_version=schema_version,
                    semantics_version=semantics_version,
                )
                for route, route_value in item.items()
            }
            continue
        if key in {"mode", "route"} and isinstance(item, str):
            item = mode_renames.get(item, item)
        renamed[renamed_key] = _rename_payload(
            item,
            mode_renames=mode_renames,
            setting_renames=setting_renames,
            schema_version=schema_version,
            semantics_version=semantics_version,
        )

    if {"mode", "routes", "settings"}.issubset(renamed):
        renamed["schema_version"] = schema_version
        settings = dict(renamed["settings"])
        settings["mode_semantics_version"] = semantics_version
        renamed["settings"] = settings
    return renamed
