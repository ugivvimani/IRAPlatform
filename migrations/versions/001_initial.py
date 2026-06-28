"""Initial migration: Create watchlist and assessment tables.

Revision ID: 001_initial
Revises: 
Create Date: 2026-06-16 09:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial schema."""
    op.create_table(
        "watchlist",
        sa.Column("entity_id", sa.String(255), primary_key=True),
        sa.Column("entity_name", sa.String(255), nullable=False),
        sa.Column("risk_level", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("last_assessed", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    
    op.create_table(
        "assessment_audit",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("entity_id", sa.String(255), nullable=False),
        sa.Column("entity_name", sa.String(255), nullable=False),
        sa.Column("risk_level", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("evidence_chain", sa.JSON(), nullable=False),
        sa.Column("assessed_at", sa.DateTime(timezone=True), nullable=False),
    )
    
    op.create_index("ix_assessment_audit_entity_id", "assessment_audit", ["entity_id"])
    op.create_index("ix_assessment_audit_assessed_at", "assessment_audit", ["assessed_at"])


def downgrade() -> None:
    """Drop initial schema."""
    op.drop_index("ix_assessment_audit_assessed_at", table_name="assessment_audit")
    op.drop_index("ix_assessment_audit_entity_id", table_name="assessment_audit")
    op.drop_table("assessment_audit")
    op.drop_table("watchlist")
