"""Allow LightRAG-aligned mix retrieval traces.

Revision ID: 0005_retrieval_trace_mix_mode
Revises: 0004_embedding_profile_graph_identity
Create Date: 2026-07-14
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_retrieval_trace_mix_mode"
down_revision: str | None = "0004_embedding_profile_graph_identity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONSTRAINT = "ck_retrieval_traces_mode"
_OLD_MODES = "mode IN ('naive', 'local', 'global', 'hybrid')"
_NEW_MODES = "mode IN ('naive', 'local', 'global', 'hybrid', 'mix')"


def upgrade() -> None:
    with op.batch_alter_table("retrieval_traces", recreate="always") as batch:
        batch.drop_constraint(_CONSTRAINT, type_="check")
        batch.create_check_constraint(_CONSTRAINT, _NEW_MODES)


def downgrade() -> None:
    connection = op.get_bind()
    contains_mix = connection.execute(
        sa.text("SELECT 1 FROM retrieval_traces WHERE mode = 'mix' LIMIT 1")
    ).first()
    if contains_mix is not None:
        raise RuntimeError("cannot downgrade while retrieval_traces contains mix mode rows")
    with op.batch_alter_table("retrieval_traces", recreate="always") as batch:
        batch.drop_constraint(_CONSTRAINT, type_="check")
        batch.create_check_constraint(_CONSTRAINT, _OLD_MODES)
