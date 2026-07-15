"""Persist deterministic chunk quality classifications.

Revision ID: 0006_chunk_quality_class
Revises: 0005_retrieval_trace_mix_mode
Create Date: 2026-07-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from hybrid_rag.ingest.quality import classify_chunk_quality

revision: str = "0006_chunk_quality_class"
down_revision: str | None = "0005_retrieval_trace_mix_mode"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONSTRAINT = "ck_chunks_quality_class"
_INDEX = "ix_chunks_quality_class"
_VALUES = (
    "quality_class IN ('normal', 'references', 'acknowledgements', 'copyright', "
    "'author_affiliation', 'visualization_label')"
)


def upgrade() -> None:
    with op.batch_alter_table("chunks", recreate="always") as batch:
        batch.add_column(
            sa.Column(
                "quality_class",
                sa.String(length=32),
                nullable=False,
                server_default=sa.text("'normal'"),
            )
        )
        batch.create_check_constraint(_CONSTRAINT, _VALUES)
        batch.create_index(_INDEX, ["quality_class"], unique=False)

    chunks = sa.table(
        "chunks",
        sa.column("id", sa.String(length=64)),
        sa.column("ordinal", sa.Integer()),
        sa.column("section_path_json", sa.JSON()),
        sa.column("page_start", sa.Integer()),
        sa.column("text", sa.Text()),
        sa.column("quality_class", sa.String(length=32)),
    )
    connection = op.get_bind()
    rows = connection.execute(
        sa.select(
            chunks.c.id,
            chunks.c.ordinal,
            chunks.c.section_path_json,
            chunks.c.page_start,
            chunks.c.text,
        )
    ).mappings()
    for row in rows:
        raw_sections = row["section_path_json"]
        sections = raw_sections if isinstance(raw_sections, list) else []
        quality_class = classify_chunk_quality(
            section_path=sections,
            text=str(row["text"]),
            ordinal=int(row["ordinal"]),
            page_start=int(row["page_start"]) if row["page_start"] is not None else None,
        )
        connection.execute(
            sa.update(chunks).where(chunks.c.id == row["id"]).values(quality_class=quality_class)
        )


def downgrade() -> None:
    with op.batch_alter_table("chunks", recreate="always") as batch:
        batch.drop_index(_INDEX)
        batch.drop_constraint(_CONSTRAINT, type_="check")
        batch.drop_column("quality_class")
