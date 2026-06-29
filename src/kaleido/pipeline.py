from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from kaleido.confidence import Calibrator, fuse_confidence
from kaleido.db.base import get_session
from kaleido.db.models import ConversationModel, FacetScoreModel, ReviewQueueModel, TurnModel
from kaleido.schemas import FacetScore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from kaleido.gating import FacetGate
    from kaleido.registry import FacetRegistry
    from kaleido.schemas import Conversation, Turn
    from kaleido.scoring import OrdinalScorer

log = structlog.get_logger(__name__)


class EvaluationPipeline:
    """Orchestrates the 5-stage evaluation pipeline for each turn.

    Stage 1 — Ingest:      persist conversation/turn records.
    Stage 2 — Gate:        sparse facet activation.
    Stage 3 — Applicability: already resolved inside the gate.
    Stage 4 — Score:       per-facet ordinal scoring (concurrent).
    Stage 5 — Calibrate:  confidence fusion + abstention + persist scores.
    """

    def __init__(
        self,
        registry: FacetRegistry,
        gate: FacetGate,
        scorer: OrdinalScorer,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        abstain_tau: float = 0.30,
        consistency_samples: int = 3,
        registry_version: str = "2026.06.0",
        calibrator: Calibrator | None = None,
    ) -> None:
        self._registry = registry
        self._gate = gate
        self._scorer = scorer
        self._factory = session_factory
        self._abstain_tau = abstain_tau
        self._consistency_samples = consistency_samples
        self._registry_version = registry_version
        self._calibrator = calibrator or Calibrator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate_conversation(self, conv: Conversation) -> list[FacetScore]:
        """Score every turn in a conversation.  Returns flat list of FacetScores."""
        await self._persist_conversation(conv)
        results: list[FacetScore] = []
        for turn in conv.turns:
            scores = await self.evaluate_turn(turn)
            results.extend(scores)
        log.info(
            "pipeline.conversation_done",
            conversation_id=conv.conversation_id,
            n_turns=len(conv.turns),
            n_scores=len(results),
        )
        return results

    async def evaluate_turn(self, turn: Turn) -> list[FacetScore]:
        """Score a single turn against all applicable facets."""
        await self._persist_turn(turn)

        # Stage 2+3: gate → applicable candidates only.
        candidates = self._gate.applicable(turn)

        if not candidates:
            log.debug("pipeline.no_candidates", turn_id=turn.turn_id)
            return []

        # Stage 4: score each applicable facet concurrently (thread-pool since scorer is sync).
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, self._score_candidate, turn, c) for c in candidates]
        raw_scores: list[FacetScore | None] = list(await asyncio.gather(*tasks))

        scores = [s for s in raw_scores if s is not None]

        # Stage 5: persist + route low-confidence to review queue.
        await self._persist_scores(scores)
        return scores

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_candidate(self, turn: Turn, candidate: object) -> FacetScore | None:
        """Score one (turn, candidate) pair.  Runs in a thread-pool executor."""
        from kaleido.gating import Candidate

        if not isinstance(candidate, Candidate):
            return None

        facet = candidate.facet

        try:
            label, logprobs, evidence = self._scorer.score(turn, facet)
        except Exception:
            log.exception("pipeline.score_error", turn_id=turn.turn_id, facet_id=facet.facet_id)
            return None

        # Self-consistency samples for confidence.
        samples: list[int] = []
        if self._consistency_samples > 1:
            try:
                samples = self._scorer.score_samples(
                    turn, facet, n_samples=self._consistency_samples
                )
            except Exception:
                log.warning("pipeline.samples_error", facet_id=facet.facet_id)

        confidence = fuse_confidence(logprobs, samples, self._calibrator)
        abstained = confidence < self._abstain_tau

        return FacetScore(
            facet_id=facet.facet_id,
            facet_name=facet.facet_name,
            domain=facet.domain,
            turn_id=turn.turn_id,
            applies=True,
            score=None if abstained else label,
            confidence=confidence,
            abstained=abstained,
            evidence_span=evidence,
            model_name=self._scorer._backend.model_name,
            registry_version=self._registry_version,
        )

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    async def _persist_conversation(self, conv: Conversation) -> None:
        async with get_session(self._factory) as session:
            existing = await session.get(ConversationModel, conv.conversation_id)
            if existing is None:
                session.add(
                    ConversationModel(
                        conversation_id=conv.conversation_id,
                        meta=dict(conv.meta),
                    )
                )

    async def _persist_turn(self, turn: Turn) -> None:
        async with get_session(self._factory) as session:
            existing = await session.get(TurnModel, turn.turn_id)
            if existing is None:
                session.add(
                    TurnModel(
                        turn_id=turn.turn_id,
                        conversation_id=turn.conversation_id,
                        idx=turn.index,
                        role=turn.role,
                        text=turn.text,
                    )
                )

    async def _persist_scores(self, scores: list[FacetScore]) -> None:
        async with get_session(self._factory) as session:
            for fs in scores:
                score_model = FacetScoreModel(
                    turn_id=fs.turn_id,
                    facet_id=fs.facet_id,
                    applies=fs.applies,
                    score=fs.score,
                    confidence=fs.confidence,
                    abstained=fs.abstained,
                    evidence_span=fs.evidence_span,
                    model_name=fs.model_name,
                    registry_version=fs.registry_version,
                )
                session.add(score_model)
                if fs.abstained:
                    # Flush to get the auto-generated score_model.id before creating FK.
                    await session.flush()
                    session.add(
                        ReviewQueueModel(
                            score_id=score_model.id,
                            reason="low_confidence",
                        )
                    )
