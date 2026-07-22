"""Allow graph extraction gleaning attempts.

Revision ID: 0008_graph_gleaning
Revises: 0007_retrieval_strategy_names
Create Date: 2026-07-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_graph_gleaning"
down_revision: str | None = "0007_retrieval_strategy_names"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONSTRAINT = "ck_extraction_attempts_stage"
_OLD_STAGES = "stage IN ('extract', 'repair')"
_NEW_STAGES = "stage IN ('extract', 'repair', 'glean')"


def upgrade() -> None:
    with op.batch_alter_table("extraction_attempts", recreate="always") as batch:
        batch.drop_constraint(_CONSTRAINT, type_="check")
        batch.create_check_constraint(_CONSTRAINT, _NEW_STAGES)


def downgrade() -> None:
    connection = op.get_bind()
    contains_glean = connection.execute(
        sa.text("SELECT 1 FROM extraction_attempts WHERE stage = 'glean' LIMIT 1")
    ).first()
    if contains_glean is not None:
        raise RuntimeError(
            "cannot downgrade while extraction_attempts contains glean stage rows"
        )
    with op.batch_alter_table("extraction_attempts", recreate="always") as batch:
        batch.drop_constraint(_CONSTRAINT, type_="check")
        batch.create_check_constraint(_CONSTRAINT, _OLD_STAGES)
