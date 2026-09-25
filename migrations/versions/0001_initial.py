"""Inspections and findings.

Revision ID: 0001
Revises:
Create Date: 2026-09-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "inspections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("part", sa.String(64), nullable=False),
        sa.Column("method", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("image_sha256", sa.String(64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("is_defective", sa.Boolean(), nullable=False),
        sa.Column("score", sa.Double(), nullable=False),
        sa.Column("threshold", sa.Double(), nullable=False),
        sa.Column("latency_ms", sa.Double(), nullable=False),
        sa.Column("raw_output", sa.Text(), nullable=True),
    )
    op.create_index("ix_inspections_created_at", "inspections", ["created_at"])
    op.create_index("ix_inspections_image_sha256", "inspections", ["image_sha256"])
    op.create_index("ix_inspections_is_defective", "inspections", ["is_defective"])
    op.create_index("ix_inspections_part_created", "inspections", ["part", "created_at"])
    op.create_table(
        "findings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "inspection_id",
            sa.Integer(),
            sa.ForeignKey("inspections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("label", sa.String(128), nullable=False),
        sa.Column("score", sa.Double(), nullable=False),
        sa.Column("x1", sa.Double(), nullable=False),
        sa.Column("y1", sa.Double(), nullable=False),
        sa.Column("x2", sa.Double(), nullable=False),
        sa.Column("y2", sa.Double(), nullable=False),
    )
    op.create_index("ix_findings_inspection_id", "findings", ["inspection_id"])


def downgrade() -> None:
    op.drop_table("findings")
    op.drop_table("inspections")
