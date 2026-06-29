from __future__ import annotations

import pytest

from kaleido.confidence import Calibrator
from kaleido.gating import FacetGate
from kaleido.pipeline import EvaluationPipeline
from kaleido.schemas import Conversation, Facet, FacetScore, Turn
from kaleido.scoring import OrdinalScorer, StubBackend

# ---------------------------------------------------------------------------
# Minimal stubs for registry, encoder, DB session factory
# ---------------------------------------------------------------------------


def _make_facet(facet_id: str = "f1") -> Facet:
    return Facet(
        facet_id=facet_id,
        facet_name="Spelling Accuracy",
        domain="linguistic_quality",
        subdomain="",
        facet_type="qualitative_trait",
        value_polarity="positive",
        text_observability="observable",
        applicability_scope="universal",
        score_scale="-2,-1,0,1,2",
        score_anchors={
            "-2": "Very poor",
            "-1": "Below average",
            "0": "Neutral",
            "1": "Above average",
            "2": "Excellent",
        },
        definition="Evaluates spelling correctness.",
        embedding_text="spelling accuracy linguistic quality",
    )


class _FakeRegistry:
    def search(self, turn_embedding: list[float], top_k: int = 64) -> list[tuple[Facet, float]]:
        return [(_make_facet(), 0.9)]

    def universals(self) -> list[Facet]:
        return [_make_facet()]


class _FakeEncoder:
    @property
    def dim(self) -> int:
        return 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]

    def encode_one(self, text: str) -> list[float]:
        return [0.0] * 4


class _NullSessionFactory:
    """Session factory that does nothing — avoids DB in pipeline unit tests."""

    async def __call__(self) -> object:
        return None


class _NullPipeline(EvaluationPipeline):
    """Override DB persistence to no-ops for unit testing."""

    async def _persist_conversation(self, conv: Conversation) -> None:
        pass

    async def _persist_turn(self, turn: Turn) -> None:
        pass

    async def _persist_scores(self, scores: list[FacetScore]) -> None:
        pass


@pytest.fixture()
def pipeline() -> _NullPipeline:
    registry = _FakeRegistry()
    encoder = _FakeEncoder()
    gate = FacetGate(registry, encoder, top_k=10, threshold=0.0)  # type: ignore[arg-type]
    scorer = OrdinalScorer(StubBackend())
    return _NullPipeline(
        registry,  # type: ignore[arg-type]
        gate,
        scorer,
        _NullSessionFactory(),  # type: ignore[arg-type]
        abstain_tau=0.0,  # never abstain in tests (stub returns uniform 0.5 confidence)
        consistency_samples=1,
        registry_version="test",
        calibrator=Calibrator(),
    )


@pytest.fixture()
def sample_turn() -> Turn:
    return Turn(
        turn_id="t1",
        conversation_id="c1",
        index=0,
        role="user",
        text="Hello, how are you today?",
    )


@pytest.fixture()
def sample_conversation(sample_turn: Turn) -> Conversation:
    return Conversation(
        conversation_id="c1",
        turns=[sample_turn],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluateTurn:
    @pytest.mark.asyncio
    async def test_returns_list_of_facet_scores(
        self, pipeline: _NullPipeline, sample_turn: Turn
    ) -> None:
        scores = await pipeline.evaluate_turn(sample_turn)
        assert isinstance(scores, list)
        assert len(scores) > 0
        assert all(isinstance(s, FacetScore) for s in scores)

    @pytest.mark.asyncio
    async def test_score_applies_true(self, pipeline: _NullPipeline, sample_turn: Turn) -> None:
        scores = await pipeline.evaluate_turn(sample_turn)
        assert all(s.applies for s in scores)

    @pytest.mark.asyncio
    async def test_score_values_valid(self, pipeline: _NullPipeline, sample_turn: Turn) -> None:
        scores = await pipeline.evaluate_turn(sample_turn)
        for s in scores:
            if s.score is not None:
                assert -2 <= s.score <= 2

    @pytest.mark.asyncio
    async def test_confidence_in_unit_interval(
        self, pipeline: _NullPipeline, sample_turn: Turn
    ) -> None:
        scores = await pipeline.evaluate_turn(sample_turn)
        assert all(0.0 <= s.confidence <= 1.0 for s in scores)

    @pytest.mark.asyncio
    async def test_facet_id_in_score(self, pipeline: _NullPipeline, sample_turn: Turn) -> None:
        scores = await pipeline.evaluate_turn(sample_turn)
        assert all(s.facet_id == "f1" for s in scores)


class TestEvaluateConversation:
    @pytest.mark.asyncio
    async def test_returns_flat_list(
        self, pipeline: _NullPipeline, sample_conversation: Conversation
    ) -> None:
        scores = await pipeline.evaluate_conversation(sample_conversation)
        assert isinstance(scores, list)
        assert len(scores) > 0

    @pytest.mark.asyncio
    async def test_invariant_applies_false_score_none(
        self, pipeline: _NullPipeline, sample_conversation: Conversation
    ) -> None:
        scores = await pipeline.evaluate_conversation(sample_conversation)
        for s in scores:
            if not s.applies:
                assert s.score is None
