"""Grounded reports and specification clauses with pgvector embeddings.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "spec_clauses",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("clause_id", sa.String(64), nullable=False, unique=True),
        sa.Column("part", sa.String(64), nullable=False),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("verdict", sa.String(16), nullable=True),
        sa.Column("embedder", sa.String(128), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
    )
    op.create_index("ix_spec_clauses_part", "spec_clauses", ["part"])
    op.create_table(
        "reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "inspection_id",
            sa.Integer(),
            sa.ForeignKey("inspections.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("verdict", sa.String(16), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("retrieved", sa.JSON(), nullable=False),
        sa.Column("generator", sa.String(64), nullable=False),
        sa.Column("rejected_llm_output", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("reports")
    op.drop_index("ix_spec_clauses_part", table_name="spec_clauses")
    op.drop_table("spec_clauses")
