"""Allow industry-aligned retrieval strategy names.

Revision ID: 0007_retrieval_strategy_names
Revises: 0006_chunk_quality_class
Create Date: 2026-07-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_retrieval_strategy_names"
down_revision: str | None = "0006_chunk_quality_class"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONSTRAINT = "ck_retrieval_traces_mode"
_OLD_MODES = "mode IN ('naive', 'local', 'global', 'hybrid', 'mix')"
_TRANSITION_MODES = (
    "mode IN ('naive', 'local', 'global', 'dense', 'bm25', 'hybrid', "
    "'graph_local', 'graph_global', 'graph_hybrid', 'mix')"
)
_NEW_ONLY_MODES = ("dense", "bm25", "graph_local", "graph_global", "graph_hybrid")


def upgrade() -> None:
    with op.batch_alter_table("retrieval_traces", recreate="always") as batch:
        batch.drop_constraint(_CONSTRAINT, type_="check")
        batch.create_check_constraint(_CONSTRAINT, _TRANSITION_MODES)


def downgrade() -> None:
    connection = op.get_bind()
    placeholders = ", ".join(f":mode_{index}" for index in range(len(_NEW_ONLY_MODES)))
    parameters = {
        f"mode_{index}": mode for index, mode in enumerate(_NEW_ONLY_MODES)
    }
    contains_new_mode = connection.execute(
        sa.text(
            f"SELECT 1 FROM retrieval_traces WHERE mode IN ({placeholders}) LIMIT 1"
        ),
        parameters,
    ).first()
    if contains_new_mode is not None:
        raise RuntimeError(
            "cannot downgrade while retrieval_traces contains current strategy names"
        )
    with op.batch_alter_table("retrieval_traces", recreate="always") as batch:
        batch.drop_constraint(_CONSTRAINT, type_="check")
        batch.create_check_constraint(_CONSTRAINT, _OLD_MODES)
