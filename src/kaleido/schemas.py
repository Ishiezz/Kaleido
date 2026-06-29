from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# IO-layer schemas (Pydantic v2).  Internal pipeline uses @dataclass(slots=True).
# ---------------------------------------------------------------------------


class Turn(BaseModel):
    """A single turn in a conversation, with bounded prior-context window."""

    turn_id: str
    conversation_id: str
    index: int
    role: Literal["user", "assistant", "system"]
    text: str
    # Bounded window of prior turn texts injected by the pipeline.
    context: list[str] = Field(default_factory=list)


class Conversation(BaseModel):
    """Full conversation submitted to the evaluation pipeline."""

    conversation_id: str
    meta: dict[str, Any] = Field(default_factory=dict)
    turns: list[Turn]

    @model_validator(mode="after")
    def _turns_reference_conversation(self) -> Conversation:
        for t in self.turns:
            if t.conversation_id != self.conversation_id:
                raise ValueError(
                    f"Turn {t.turn_id!r} has conversation_id {t.conversation_id!r}"
                    f" but conversation is {self.conversation_id!r}"
                )
        return self


class Facet(BaseModel):
    """One registry row — the complete scoring contract for a single facet.

    Adding a new facet requires only an INSERT into the facets table.
    No code path may hardcode a facet_id or facet_name.
    """

    facet_id: str
    facet_name: str
    domain: str
    subdomain: str = ""
    facet_type: Literal["qualitative_trait", "level_score", "frequency_count", "binary"]
    value_polarity: Literal["positive", "negative", "bipolar", "neutral"]
    text_observability: Literal[
        "observable",
        "requires_explicit_mention",
        "requires_external_data",
        "not_text_observable",
    ]
    applicability_scope: Literal["universal", "conditional", "rare"]
    # Fixed five-label ordinal scale: {-2, -1, 0, 1, 2}.
    score_scale: str = "-2,-1,0,1,2"
    # Keys are the string labels "-2" through "2".
    score_anchors: dict[str, str]
    definition: str
    embedding_text: str
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    needs_review: bool = False
    version: str = "2026.06.0"

    @model_validator(mode="after")
    def _anchors_cover_scale(self) -> Facet:
        required = {"-2", "-1", "0", "1", "2"}
        missing = required - self.score_anchors.keys()
        if missing:
            raise ValueError(f"score_anchors missing keys: {missing}")
        return self


class FacetScore(BaseModel):
    """The output of scoring one (turn, facet) pair."""

    facet_id: str
    facet_name: str = ""
    domain: str = ""
    turn_id: str
    # False means the gate determined this facet is not evidenced by the turn.
    applies: bool
    # None iff applies is False — the DB constraint encodes this invariant.
    score: int | None = Field(default=None, ge=-2, le=2)
    # Calibrated probability ∈ [0, 1].
    confidence: float = Field(ge=0.0, le=1.0)
    # True when confidence < abstain_tau; item is routed to the review queue.
    abstained: bool = False
    # Substring of turn.text that most supports the score.
    evidence_span: str | None = None
    model_name: str
    registry_version: str

    @model_validator(mode="after")
    def _score_consistent_with_applies(self) -> FacetScore:
        if not self.applies and self.score is not None:
            raise ValueError("score must be None when applies is False")
        if self.applies and self.score is None and not self.abstained:
            raise ValueError("score must be set when applies is True and not abstained")
        return self


# ---------------------------------------------------------------------------
# Request / response wrappers for the API layer.
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    conversation: Conversation
    # Optional: restrict to specific facet IDs (empty = all activated by gate).
    facet_ids: list[str] = Field(default_factory=list)


class ScoreResponse(BaseModel):
    conversation_id: str
    scores: list[FacetScore]


class FacetCreateRequest(BaseModel):
    facet_name: str
    domain: str
    # True → call synthesize_contract; False → require `facet` to be provided directly.
    auto_synthesize: bool = True
    facet: Facet | None = None


class ReviewResolveRequest(BaseModel):
    human_score: int = Field(ge=-2, le=2)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    registry_version: str
    n_facets: int
    backend: str
