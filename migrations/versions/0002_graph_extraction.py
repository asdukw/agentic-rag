"""Create resumable extraction and current graph snapshot tables.

Revision ID: 0002_graph_extraction
Revises: 0001_ingestion
Create Date: 2026-07-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_graph_extraction"
down_revision: str | None = "0001_ingestion"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "graph_build_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("extraction_config_hash", sa.String(length=64), nullable=False),
        sa.Column("graph_config_hash", sa.String(length=64), nullable=False),
        sa.Column("corpus_hash", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("prompt_version", sa.String(length=50), nullable=False),
        sa.Column("schema_version", sa.String(length=50), nullable=False),
        sa.Column("workflow_version", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("review_required", sa.Boolean(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_chunks", sa.Integer(), nullable=False),
        sa.Column("cached_chunks", sa.Integer(), nullable=False),
        sa.Column("scheduled_chunks", sa.Integer(), nullable=False),
        sa.Column("succeeded_chunks", sa.Integer(), nullable=False),
        sa.Column("needs_review_chunks", sa.Integer(), nullable=False),
        sa.Column("failed_chunks", sa.Integer(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("extract_attempt_count", sa.Integer(), nullable=False),
        sa.Column("repair_attempt_count", sa.Integer(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("entity_count", sa.Integer(), nullable=False),
        sa.Column("relation_count", sa.Integer(), nullable=False),
        sa.Column("component_count", sa.Integer(), nullable=False),
        sa.Column("largest_component_nodes", sa.Integer(), nullable=False),
        sa.Column("isolated_entity_count", sa.Integer(), nullable=False),
        sa.Column("report_json", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'awaiting_review', 'completed', "
            "'completed_with_failures', 'failed')",
            name="ck_graph_build_runs_status",
        ),
        sa.CheckConstraint(
            "total_chunks >= 0 AND cached_chunks >= 0 AND scheduled_chunks >= 0 "
            "AND succeeded_chunks >= 0 AND needs_review_chunks >= 0 "
            "AND failed_chunks >= 0",
            name="ck_graph_build_runs_chunk_counts",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0 AND extract_attempt_count >= 0 AND repair_attempt_count >= 0",
            name="ck_graph_build_runs_attempt_counts",
        ),
        sa.CheckConstraint(
            "entity_count >= 0 AND relation_count >= 0 AND component_count >= 0 "
            "AND largest_component_nodes >= 0 AND isolated_entity_count >= 0",
            name="ck_graph_build_runs_graph_counts",
        ),
        sa.CheckConstraint(
            "prompt_tokens >= 0 AND completion_tokens >= 0 AND total_tokens >= 0",
            name="ck_graph_build_runs_usage_counts",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_graph_build_runs_graph_config_corpus",
        "graph_build_runs",
        ["graph_config_hash", "corpus_hash"],
        unique=False,
    )
    op.create_index("ix_graph_build_runs_status", "graph_build_runs", ["status"], unique=False)

    op.create_table(
        "chunk_extractions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("extraction_config_hash", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("prompt_version", sa.String(length=50), nullable=False),
        sa.Column("schema_version", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("lease_token", sa.String(length=64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'needs_review', 'failed')",
            name="ck_chunk_extractions_status",
        ),
        sa.CheckConstraint("attempt_count >= 0", name="ck_chunk_extractions_attempt_count"),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "chunk_id",
            "extraction_config_hash",
            name="uq_chunk_extractions_chunk_config",
        ),
    )
    op.create_index(
        "ix_chunk_extractions_chunk_id", "chunk_extractions", ["chunk_id"], unique=False
    )
    op.create_index(
        "ix_chunk_extractions_config_status",
        "chunk_extractions",
        ["extraction_config_hash", "status"],
        unique=False,
    )
    op.create_index(
        "ix_chunk_extractions_lease",
        "chunk_extractions",
        ["status", "lease_expires_at"],
        unique=False,
    )

    op.create_table(
        "extraction_attempts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("extraction_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("messages_json", sa.JSON(), nullable=False),
        sa.Column("raw_response", sa.Text(), nullable=True),
        sa.Column("response_metadata_json", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("latency_seconds", sa.Float(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.CheckConstraint("ordinal > 0", name="ck_extraction_attempts_ordinal"),
        sa.CheckConstraint("stage IN ('extract', 'repair')", name="ck_extraction_attempts_stage"),
        sa.CheckConstraint(
            "prompt_tokens >= 0 AND completion_tokens >= 0 AND total_tokens >= 0",
            name="ck_extraction_attempts_usage",
        ),
        sa.ForeignKeyConstraint(["extraction_id"], ["chunk_extractions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["graph_build_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "extraction_id",
            "ordinal",
            name="uq_extraction_attempts_extraction_ordinal",
        ),
    )
    op.create_index(
        "ix_extraction_attempts_extraction_id",
        "extraction_attempts",
        ["extraction_id"],
        unique=False,
    )
    op.create_index(
        "ix_extraction_attempts_run_id",
        "extraction_attempts",
        ["run_id"],
        unique=False,
    )

    op.create_table(
        "graph_build_items",
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("extraction_id", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("disposition", sa.String(length=16), nullable=False),
        sa.Column("review_status", sa.String(length=16), nullable=False),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("ordinal >= 0", name="ck_graph_build_items_ordinal"),
        sa.CheckConstraint(
            "disposition IN ('scheduled', 'cached')",
            name="ck_graph_build_items_disposition",
        ),
        sa.CheckConstraint(
            "review_status IN ('not_required', 'pending', 'approved', 'rejected')",
            name="ck_graph_build_items_review_status",
        ),
        sa.ForeignKeyConstraint(["extraction_id"], ["chunk_extractions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["graph_build_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "extraction_id"),
        sa.UniqueConstraint("run_id", "ordinal", name="uq_graph_build_items_run_ordinal"),
    )
    op.create_index(
        "ix_graph_build_items_extraction_id",
        "graph_build_items",
        ["extraction_id"],
        unique=False,
    )

    op.create_table(
        "entities",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("build_run_id", sa.String(length=64), nullable=False),
        sa.Column("graph_config_hash", sa.String(length=64), nullable=False),
        sa.Column("canonical_name", sa.Text(), nullable=False),
        sa.Column("normalized_name", sa.Text(), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("aliases_json", sa.JSON(), nullable=False),
        sa.Column("source_chunk_ids_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["build_run_id"], ["graph_build_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "build_run_id",
            "normalized_name",
            "entity_type",
            name="uq_entities_run_normalized_name_type",
        ),
    )
    op.create_index("ix_entities_build_run_id", "entities", ["build_run_id"], unique=False)
    op.create_index("ix_entities_normalized_name", "entities", ["normalized_name"], unique=False)
    op.create_index("ix_entities_type", "entities", ["entity_type"], unique=False)

    op.create_table(
        "relations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("build_run_id", sa.String(length=64), nullable=False),
        sa.Column("graph_config_hash", sa.String(length=64), nullable=False),
        sa.Column("source_entity_id", sa.String(length=64), nullable=False),
        sa.Column("target_entity_id", sa.String(length=64), nullable=False),
        sa.Column("predicate", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("source_chunk_ids_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["build_run_id"], ["graph_build_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_entity_id"], ["entities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_entity_id"], ["entities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "build_run_id",
            "source_entity_id",
            "target_entity_id",
            "predicate",
            name="uq_relations_run_endpoints_predicate",
        ),
    )
    op.create_index("ix_relations_build_run_id", "relations", ["build_run_id"], unique=False)
    op.create_index("ix_relations_predicate", "relations", ["predicate"], unique=False)
    op.create_index(
        "ix_relations_source_entity_id", "relations", ["source_entity_id"], unique=False
    )
    op.create_index(
        "ix_relations_target_entity_id", "relations", ["target_entity_id"], unique=False
    )

    op.create_table(
        "entity_evidence",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("extraction_id", sa.String(length=64), nullable=False),
        sa.Column("mention_id", sa.String(length=64), nullable=True),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "char_start >= 0 AND char_end >= char_start",
            name="ck_entity_evidence_char_span",
        ),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["entity_id"], ["entities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["extraction_id"], ["chunk_extractions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "entity_id",
            "extraction_id",
            "quote",
            "char_start",
            "char_end",
            name="uq_entity_evidence_source_span",
        ),
    )
    op.create_index("ix_entity_evidence_chunk_id", "entity_evidence", ["chunk_id"], unique=False)
    op.create_index("ix_entity_evidence_entity_id", "entity_evidence", ["entity_id"], unique=False)
    op.create_index(
        "ix_entity_evidence_extraction_id",
        "entity_evidence",
        ["extraction_id"],
        unique=False,
    )

    op.create_table(
        "relation_evidence",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("relation_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("extraction_id", sa.String(length=64), nullable=False),
        sa.Column("mention_id", sa.String(length=64), nullable=True),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "char_start >= 0 AND char_end >= char_start",
            name="ck_relation_evidence_char_span",
        ),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["extraction_id"], ["chunk_extractions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["relation_id"], ["relations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "relation_id",
            "extraction_id",
            "quote",
            "char_start",
            "char_end",
            name="uq_relation_evidence_source_span",
        ),
    )
    op.create_index(
        "ix_relation_evidence_chunk_id", "relation_evidence", ["chunk_id"], unique=False
    )
    op.create_index(
        "ix_relation_evidence_extraction_id",
        "relation_evidence",
        ["extraction_id"],
        unique=False,
    )
    op.create_index(
        "ix_relation_evidence_relation_id",
        "relation_evidence",
        ["relation_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_relation_evidence_relation_id", table_name="relation_evidence")
    op.drop_index("ix_relation_evidence_extraction_id", table_name="relation_evidence")
    op.drop_index("ix_relation_evidence_chunk_id", table_name="relation_evidence")
    op.drop_table("relation_evidence")
    op.drop_index("ix_entity_evidence_extraction_id", table_name="entity_evidence")
    op.drop_index("ix_entity_evidence_entity_id", table_name="entity_evidence")
    op.drop_index("ix_entity_evidence_chunk_id", table_name="entity_evidence")
    op.drop_table("entity_evidence")
    op.drop_index("ix_relations_target_entity_id", table_name="relations")
    op.drop_index("ix_relations_source_entity_id", table_name="relations")
    op.drop_index("ix_relations_predicate", table_name="relations")
    op.drop_index("ix_relations_build_run_id", table_name="relations")
    op.drop_table("relations")
    op.drop_index("ix_entities_type", table_name="entities")
    op.drop_index("ix_entities_normalized_name", table_name="entities")
    op.drop_index("ix_entities_build_run_id", table_name="entities")
    op.drop_table("entities")
    op.drop_index("ix_graph_build_items_extraction_id", table_name="graph_build_items")
    op.drop_table("graph_build_items")
    op.drop_index("ix_extraction_attempts_run_id", table_name="extraction_attempts")
    op.drop_index("ix_extraction_attempts_extraction_id", table_name="extraction_attempts")
    op.drop_table("extraction_attempts")
    op.drop_index("ix_chunk_extractions_lease", table_name="chunk_extractions")
    op.drop_index("ix_chunk_extractions_config_status", table_name="chunk_extractions")
    op.drop_index("ix_chunk_extractions_chunk_id", table_name="chunk_extractions")
    op.drop_table("chunk_extractions")
    op.drop_index("ix_graph_build_runs_status", table_name="graph_build_runs")
    op.drop_index("ix_graph_build_runs_graph_config_corpus", table_name="graph_build_runs")
    op.drop_table("graph_build_runs")
