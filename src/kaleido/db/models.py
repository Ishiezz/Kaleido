from __future__ import annotations

from datetime import datetime  # noqa: TC003 — SQLAlchemy resolves Mapped[datetime] at runtime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kaleido.db.base import Base

# Use JSONB on Postgres (binary, indexed) but fall back to plain JSON on SQLite.
_JSONB = JSON().with_variant(JSONB(), "postgresql")
# SQLite doesn't autoincrement BIGINT; use Integer there.
_BIGINT = BigInteger().with_variant(Integer(), "sqlite")

# pgvector type — use Vector(384) on Postgres; fall back to Text on SQLite.
# with_variant ensures the pgvector type processor never runs on non-Postgres dialects.
try:
    from pgvector.sqlalchemy import Vector

    VECTOR_TYPE: Any = Text().with_variant(Vector(384), "postgresql")
except ImportError:
    VECTOR_TYPE = Text()


class FacetModel(Base):
    __tablename__ = "facets"

    facet_id: Mapped[str] = mapped_column(String, primary_key=True)
    facet_name: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    subdomain: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    facet_type: Mapped[str] = mapped_column(String(32), nullable=False)
    value_polarity: Mapped[str] = mapped_column(String(16), nullable=False)
    text_observability: Mapped[str] = mapped_column(String(32), nullable=False)
    applicability_scope: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    score_scale: Mapped[str] = mapped_column(String(32), nullable=False, default="-2,-1,0,1,2")
    score_anchors: Mapped[dict[str, str]] = mapped_column(_JSONB, nullable=False)
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)
    # VECTOR(384) in Postgres; Text in SQLite stub.
    embedding: Mapped[Any] = mapped_column(VECTOR_TYPE, nullable=True)
    difficulty: Mapped[str] = mapped_column(String(8), nullable=False, default="easy")
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)

    scores: Mapped[list[FacetScoreModel]] = relationship(
        "FacetScoreModel", back_populates="facet", passive_deletes=True
    )


class ConversationModel(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String, primary_key=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(_JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    turns: Mapped[list[TurnModel]] = relationship(
        "TurnModel", back_populates="conversation", cascade="all, delete-orphan"
    )


class TurnModel(Base):
    __tablename__ = "turns"

    turn_id: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String, ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    conversation: Mapped[ConversationModel] = relationship(
        "ConversationModel", back_populates="turns"
    )
    scores: Mapped[list[FacetScoreModel]] = relationship(
        "FacetScoreModel", back_populates="turn", cascade="all, delete-orphan"
    )


class FacetScoreModel(Base):
    __tablename__ = "facet_scores"
    __table_args__ = (
        # Core invariants: applies=False ⇒ score is NULL; score ∈ {-2..2}.
        CheckConstraint("score IS NULL OR score BETWEEN -2 AND 2", name="score_range"),
        CheckConstraint(
            "(applies = FALSE AND score IS NULL) OR applies = TRUE",
            name="score_applies",
        ),
        UniqueConstraint("turn_id", "facet_id", "registry_version", name="uq_turn_facet_version"),
    )

    id: Mapped[int] = mapped_column(_BIGINT, primary_key=True, autoincrement=True)
    turn_id: Mapped[str] = mapped_column(
        String, ForeignKey("turns.turn_id", ondelete="CASCADE"), nullable=False
    )
    facet_id: Mapped[str] = mapped_column(String, ForeignKey("facets.facet_id"), nullable=False)
    applies: Mapped[bool] = mapped_column(Boolean, nullable=False)
    score: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    confidence: Mapped[float] = mapped_column(nullable=False)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    evidence_span: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    registry_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    turn: Mapped[TurnModel] = relationship("TurnModel", back_populates="scores")
    facet: Mapped[FacetModel] = relationship("FacetModel", back_populates="scores")
    review_items: Mapped[list[ReviewQueueModel]] = relationship(
        "ReviewQueueModel", back_populates="facet_score", cascade="all, delete-orphan"
    )


class ReviewQueueModel(Base):
    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(_BIGINT, primary_key=True, autoincrement=True)
    score_id: Mapped[int] = mapped_column(
        _BIGINT,
        ForeignKey("facet_scores.id", ondelete="CASCADE"),
        nullable=False,
    )
    # 'low_confidence' | 'disagreement' | 'needs_review'
    reason: Mapped[str] = mapped_column(String(32), nullable=False)
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    human_score: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    facet_score: Mapped[FacetScoreModel] = relationship(
        "FacetScoreModel", back_populates="review_items"
    )
