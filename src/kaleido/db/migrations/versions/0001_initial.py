"""Initial schema: facets, conversations, turns, facet_scores, review_queue.

Revision ID: 0001
Revises:
Create Date: 2026-06-30
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # pgvector extension — no-op on Postgres without the extension installed.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "facets",
        sa.Column("facet_id", sa.String(), primary_key=True),
        sa.Column("facet_name", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(64), nullable=False),
        sa.Column("subdomain", sa.String(128), nullable=False, server_default=""),
        sa.Column("facet_type", sa.String(32), nullable=False),
        sa.Column("value_polarity", sa.String(16), nullable=False),
        sa.Column("text_observability", sa.String(32), nullable=False),
        sa.Column("applicability_scope", sa.String(16), nullable=False),
        sa.Column("score_scale", sa.String(32), nullable=False, server_default="-2,-1,0,1,2"),
        sa.Column("score_anchors", sa.JSON(), nullable=False),
        sa.Column("definition", sa.Text(), nullable=False),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        # VECTOR(384) in Postgres via pgvector; raw text column name kept for compat.
        sa.Column("embedding", sa.Text(), nullable=True),
        sa.Column("difficulty", sa.String(8), nullable=False, server_default="easy"),
        sa.Column("needs_review", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("version", sa.String(32), nullable=False),
    )
    op.create_index("ix_facets_domain", "facets", ["domain"])
    op.create_index("ix_facets_applicability_scope", "facets", ["applicability_scope"])

    op.create_table(
        "conversations",
        sa.Column("conversation_id", sa.String(), primary_key=True),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "turns",
        sa.Column("turn_id", sa.String(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(),
            sa.ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
    )

    op.create_table(
        "facet_scores",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "turn_id",
            sa.String(),
            sa.ForeignKey("turns.turn_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("facet_id", sa.String(), sa.ForeignKey("facets.facet_id"), nullable=False),
        sa.Column("applies", sa.Boolean(), nullable=False),
        sa.Column("score", sa.SmallInteger(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("abstained", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("evidence_span", sa.Text(), nullable=True),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("registry_version", sa.String(32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("score IS NULL OR score BETWEEN -2 AND 2", name="score_range"),
        sa.CheckConstraint(
            "(applies = FALSE AND score IS NULL) OR applies = TRUE",
            name="score_applies",
        ),
        sa.UniqueConstraint(
            "turn_id", "facet_id", "registry_version", name="uq_turn_facet_version"
        ),
    )

    op.create_table(
        "review_queue",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "score_id",
            sa.BigInteger(),
            sa.ForeignKey("facet_scores.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("reason", sa.String(32), nullable=False),
        sa.Column("resolved", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("human_score", sa.SmallInteger(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("review_queue")
    op.drop_table("facet_scores")
    op.drop_table("turns")
    op.drop_table("conversations")
    op.drop_table("facets")
