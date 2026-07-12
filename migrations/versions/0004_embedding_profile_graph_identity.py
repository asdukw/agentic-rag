"""Bind embedding profile identities to their source graph snapshot run.

Revision ID: 0004_embedding_profile_graph_identity
Revises: 0003_retrieval_indexes
Create Date: 2026-07-12
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "0004_embedding_profile_graph_identity"
down_revision: str | None = "0003_retrieval_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LEGACY_UNIQUE = "uq_embedding_profiles_config_corpus"
_SNAPSHOT_UNIQUE = "uq_embedding_profiles_config_corpus_graph_run"
_NO_GRAPH_UNIQUE = "uq_embedding_profiles_config_corpus_no_graph"


def upgrade() -> None:
    """Make graph-backed profiles distinct even when their corpus hash matches."""

    connection = op.get_bind()
    _rewrite_profile_ids(connection, include_graph_run=True)
    with op.batch_alter_table("embedding_profiles", recreate="always") as batch:
        batch.drop_constraint(_LEGACY_UNIQUE, type_="unique")
        batch.create_unique_constraint(
            _SNAPSHOT_UNIQUE,
            ["config_hash", "source_corpus_hash", "source_graph_run_id"],
        )
    op.create_index(
        _NO_GRAPH_UNIQUE,
        "embedding_profiles",
        ["config_hash", "source_corpus_hash"],
        unique=True,
        sqlite_where=sa.text("source_graph_run_id IS NULL"),
    )


def downgrade() -> None:
    """Restore the old identity only when it can be represented losslessly."""

    connection = op.get_bind()
    ambiguous = connection.execute(
        sa.text(
            """
            SELECT config_hash, source_corpus_hash
            FROM embedding_profiles
            GROUP BY config_hash, source_corpus_hash
            HAVING count(*) > 1
            LIMIT 1
            """
        )
    ).first()
    if ambiguous is not None:
        raise RuntimeError(
            "cannot downgrade embedding profile graph identity while multiple graph snapshots "
            "share one config/corpus pair"
        )

    _rewrite_profile_ids(connection, include_graph_run=False)
    op.drop_index(_NO_GRAPH_UNIQUE, table_name="embedding_profiles")
    with op.batch_alter_table("embedding_profiles", recreate="always") as batch:
        batch.drop_constraint(_SNAPSHOT_UNIQUE, type_="unique")
        batch.create_unique_constraint(
            _LEGACY_UNIQUE,
            ["config_hash", "source_corpus_hash"],
        )


def _rewrite_profile_ids(connection: sa.Connection, *, include_graph_run: bool) -> None:
    rows = connection.execute(
        sa.text(
            """
            SELECT id, config_hash, source_corpus_hash, source_graph_run_id
            FROM embedding_profiles
            ORDER BY id
            """
        )
    ).mappings()
    source_rows = list(rows)
    changes = [
        (
            str(row["id"]),
            _profile_id(
                str(row["config_hash"]),
                str(row["source_corpus_hash"]),
                (
                    str(row["source_graph_run_id"])
                    if include_graph_run and row["source_graph_run_id"]
                    else None
                ),
            ),
        )
        for row in source_rows
    ]
    changes = [(old_id, new_id) for old_id, new_id in changes if old_id != new_id]
    _validate_id_changes(changes, {str(row["id"]) for row in source_rows})

    for old_profile_id, new_profile_id in changes:
        vectors = connection.execute(
            sa.text(
                """
                SELECT id, kind, object_id
                FROM embedding_vectors
                WHERE profile_id = :profile_id
                ORDER BY id
                """
            ),
            {"profile_id": old_profile_id},
        ).mappings()
        for vector in vectors:
            connection.execute(
                sa.text(
                    """
                    UPDATE embedding_vectors
                    SET id = :new_vector_id, profile_id = :new_profile_id
                    WHERE id = :old_vector_id
                    """
                ),
                {
                    "old_vector_id": str(vector["id"]),
                    "new_vector_id": _vector_id(
                        new_profile_id,
                        str(vector["kind"]),
                        str(vector["object_id"]),
                    ),
                    "new_profile_id": new_profile_id,
                },
            )
        traces = connection.execute(
            sa.text(
                """
                SELECT id, trace_json, output_json
                FROM retrieval_traces
                WHERE profile_id = :profile_id
                ORDER BY id
                """
            ),
            {"profile_id": old_profile_id},
        ).mappings()
        for trace in traces:
            connection.execute(
                sa.text(
                    """
                    UPDATE retrieval_traces
                    SET profile_id = :new_profile_id,
                        trace_json = :trace_json,
                        output_json = :output_json
                    WHERE id = :trace_id
                    """
                ),
                {
                    "trace_id": str(trace["id"]),
                    "new_profile_id": new_profile_id,
                    "trace_json": _json_text(
                        _replace_profile_id(
                            _json_value(trace["trace_json"]),
                            old_profile_id,
                            new_profile_id,
                        )
                    ),
                    "output_json": (
                        _json_text(
                            _replace_profile_id(
                                _json_value(trace["output_json"]),
                                old_profile_id,
                                new_profile_id,
                            )
                        )
                        if trace["output_json"] is not None
                        else None
                    ),
                },
            )
        connection.execute(
            sa.text(
                "UPDATE embedding_profiles SET id = :new_profile_id WHERE id = :old_profile_id"
            ),
            {"old_profile_id": old_profile_id, "new_profile_id": new_profile_id},
        )


def _validate_id_changes(changes: Sequence[tuple[str, str]], existing_ids: set[str]) -> None:
    new_ids = [new_id for _, new_id in changes]
    if len(new_ids) != len(set(new_ids)):
        raise RuntimeError("embedding profile ID migration produced duplicate IDs")
    conflict = next(
        (
            new_id
            for old_id, new_id in changes
            if new_id != old_id and new_id in existing_ids
        ),
        None,
    )
    if conflict is not None:
        raise RuntimeError(
            "embedding profile ID migration conflicts with an existing profile ID: " + conflict
        )


def _profile_id(
    config_hash: str,
    source_corpus_hash: str,
    source_graph_run_id: str | None,
) -> str:
    parts = [config_hash, source_corpus_hash]
    if source_graph_run_id is not None:
        parts.append(source_graph_run_id)
    return _stable_id("idx", *parts)


def _vector_id(profile_id: str, kind: str, object_id: str) -> str:
    return _stable_id("vec", profile_id, kind, object_id)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _json_value(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _replace_profile_id(value: Any, old_profile_id: str, new_profile_id: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_profile_id(item, old_profile_id, new_profile_id)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_profile_id(item, old_profile_id, new_profile_id) for item in value]
    return new_profile_id if value == old_profile_id else value
