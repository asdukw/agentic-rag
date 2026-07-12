"""Create durable embedding indexes and retrieval traces.

Revision ID: 0003_retrieval_indexes
Revises: 0002_graph_extraction
Create Date: 2026-07-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_retrieval_indexes"
down_revision: str | None = "0002_graph_extraction"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "embedding_profiles",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("config_hash", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(length=50), nullable=False),
        sa.Column("source_graph_run_id", sa.String(length=64), nullable=True),
        sa.Column("source_corpus_hash", sa.String(length=64), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint("dimensions > 0", name="ck_embedding_profiles_dimensions"),
        sa.CheckConstraint(
            "status IN ('ready', 'failed')",
            name="ck_embedding_profiles_status",
        ),
        sa.ForeignKeyConstraint(
            ["source_graph_run_id"],
            ["graph_build_runs.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "config_hash",
            "source_corpus_hash",
            name="uq_embedding_profiles_config_corpus",
        ),
    )
    op.create_index(
        "ix_embedding_profiles_active",
        "embedding_profiles",
        ["is_active"],
        unique=False,
    )
    op.create_index(
        "ix_embedding_profiles_source_graph_run",
        "embedding_profiles",
        ["source_graph_run_id"],
        unique=False,
    )

    op.create_table(
        "embedding_vectors",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("profile_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("object_id", sa.String(length=64), nullable=False),
        sa.Column("build_run_id", sa.String(length=64), nullable=True),
        sa.Column("source_content_hash", sa.String(length=64), nullable=True),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column("embedding_json", sa.JSON(), nullable=False),
        sa.Column("source_chunk_ids_json", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "kind IN ('chunk', 'entity', 'relation')",
            name="ck_embedding_vectors_kind",
        ),
        sa.ForeignKeyConstraint(["profile_id"], ["embedding_profiles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["build_run_id"], ["graph_build_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "profile_id",
            "kind",
            "object_id",
            name="uq_embedding_vectors_profile_kind_object",
        ),
    )
    op.create_index(
        "ix_embedding_vectors_profile_kind",
        "embedding_vectors",
        ["profile_id", "kind"],
        unique=False,
    )
    op.create_index(
        "ix_embedding_vectors_build_run",
        "embedding_vectors",
        ["build_run_id"],
        unique=False,
    )

    op.create_table(
        "retrieval_traces",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("profile_id", sa.String(length=64), nullable=False),
        sa.Column("index_config_hash", sa.String(length=64), nullable=False),
        sa.Column("graph_build_run_id", sa.String(length=64), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("query_hash", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("retrieval_config_hash", sa.String(length=64), nullable=False),
        sa.Column("trace_json", sa.JSON(), nullable=False),
        sa.Column("output_json", sa.JSON(), nullable=True),
        sa.Column("model_info_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "mode IN ('naive', 'local', 'global', 'hybrid')",
            name="ck_retrieval_traces_mode",
        ),
        sa.ForeignKeyConstraint(["profile_id"], ["embedding_profiles.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["graph_build_run_id"],
            ["graph_build_runs.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_retrieval_traces_profile_created",
        "retrieval_traces",
        ["profile_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_retrieval_traces_query_hash",
        "retrieval_traces",
        ["query_hash"],
        unique=False,
    )
    op.create_index(
        "ix_retrieval_traces_build_run",
        "retrieval_traces",
        ["graph_build_run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_retrieval_traces_build_run", table_name="retrieval_traces")
    op.drop_index("ix_retrieval_traces_query_hash", table_name="retrieval_traces")
    op.drop_index("ix_retrieval_traces_profile_created", table_name="retrieval_traces")
    op.drop_table("retrieval_traces")
    op.drop_index("ix_embedding_vectors_build_run", table_name="embedding_vectors")
    op.drop_index("ix_embedding_vectors_profile_kind", table_name="embedding_vectors")
    op.drop_table("embedding_vectors")
    op.drop_index("ix_embedding_profiles_source_graph_run", table_name="embedding_profiles")
    op.drop_index("ix_embedding_profiles_active", table_name="embedding_profiles")
    op.drop_table("embedding_profiles")
