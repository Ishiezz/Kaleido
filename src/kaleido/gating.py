from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from kaleido.embedding import TextEncoder
    from kaleido.registry import FacetRegistry
    from kaleido.schemas import Facet, Turn

log = structlog.get_logger(__name__)

# Observability levels that are never text-scorable.
_NEVER_APPLIES = frozenset({"requires_external_data", "not_text_observable"})


@dataclass(slots=True)
class Candidate:
    """A facet considered for activation on a particular turn."""

    facet: Facet
    retrieval_score: float
    applies: bool
    applicability_score: float  # 1.0 if universally applicable, else retrieval_score


class FacetGate:
    """Two-stage sparse facet activator.

    Stage 1 — Retrieve: bi-encoder kNN ∪ universals.
    Stage 2 — Verify:   rule-based applicability filter per text_observability.

    Only the activated set (applies=True) is forwarded to the scorer.
    Typically <1% of all facets fire on any given turn.
    """

    def __init__(
        self,
        registry: FacetRegistry,
        encoder: TextEncoder,
        *,
        top_k: int = 64,
        threshold: float = 0.35,
    ) -> None:
        self._registry = registry
        self._encoder = encoder
        self._top_k = top_k
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def activate(self, turn: Turn) -> list[Candidate]:
        """Return all candidates (applies=True and applies=False)."""
        turn_vec = self._encoder.encode_one(turn.text)

        # Stage 1: retrieve top_k by cosine ∪ universals (deduped by facet_id).
        retrieved: dict[str, tuple[Facet, float]] = {}
        for facet, score in self._registry.search(turn_vec, top_k=self._top_k):
            retrieved[facet.facet_id] = (facet, score)
        for facet in self._registry.universals():
            if facet.facet_id not in retrieved:
                retrieved[facet.facet_id] = (facet, 1.0)

        # Stage 2: verify applicability.
        candidates: list[Candidate] = []
        for facet, ret_score in retrieved.values():
            applies, app_score = self._verify(turn, facet, ret_score)
            candidates.append(
                Candidate(
                    facet=facet,
                    retrieval_score=ret_score,
                    applies=applies,
                    applicability_score=app_score,
                )
            )

        log.debug(
            "gate.activated",
            turn_id=turn.turn_id,
            total=len(candidates),
            applicable=sum(1 for c in candidates if c.applies),
        )
        return candidates

    def applicable(self, turn: Turn) -> list[Candidate]:
        """Convenience: only the applies=True subset."""
        return [c for c in self.activate(turn) if c.applies]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _verify(self, turn: Turn, facet: Facet, ret_score: float) -> tuple[bool, float]:
        """Return (applies, applicability_score) for a single (turn, facet) pair."""
        obs = facet.text_observability

        # Never applies — no text signal can satisfy these.
        if obs in _NEVER_APPLIES:
            return False, 0.0

        # Always applies regardless of content.
        if facet.applicability_scope == "universal":
            return True, 1.0

        # Requires the facet's concept to be mentioned explicitly.
        if obs == "requires_explicit_mention":
            return self._mention_check(turn, facet), ret_score

        # Default: observable — threshold gate on retrieval score.
        return ret_score >= self._threshold, ret_score

    @staticmethod
    def _mention_check(turn: Turn, facet: Facet) -> bool:
        """True if any token from the facet name appears in the turn text."""
        tokens = re.findall(r"[a-z]+", facet.facet_name.lower())
        text_lower = turn.text.lower()
        # Require at least one content token (len>2) to match.
        content_tokens = [t for t in tokens if len(t) > 2]
        if not content_tokens:
            return True  # can't filter a single-char facet name — pass through
        return any(t in text_lower for t in content_tokens)
