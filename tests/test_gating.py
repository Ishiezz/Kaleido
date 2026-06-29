from __future__ import annotations

import pytest

from kaleido.gating import _NEVER_APPLIES, Candidate, FacetGate
from kaleido.schemas import Facet, Turn

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_facet(
    facet_id: str = "f_test",
    facet_name: str = "Test Facet",
    observability: str = "observable",
    scope: str = "conditional",
) -> Facet:
    return Facet(
        facet_id=facet_id,
        facet_name=facet_name,
        domain="test",
        subdomain="",
        facet_type="qualitative_trait",
        value_polarity="bipolar",
        text_observability=observability,  # type: ignore[arg-type]
        applicability_scope=scope,  # type: ignore[arg-type]
        score_scale="-2,-1,0,1,2",
        score_anchors={
            "-2": "bad",
            "-1": "below avg",
            "0": "neutral",
            "1": "above avg",
            "2": "great",
        },
        definition="A test facet",
        embedding_text="test facet definition text",
    )


def _make_turn(text: str = "Hello, how are you?") -> Turn:
    return Turn(
        turn_id="t1",
        conversation_id="c1",
        index=0,
        role="user",
        text=text,
    )


class _FakeRegistry:
    """Minimal stub registry for gate tests."""

    def __init__(
        self,
        search_results: list[tuple[Facet, float]] | None = None,
        universal_facets: list[Facet] | None = None,
    ) -> None:
        self._search = search_results or []
        self._universals = universal_facets or []

    def search(self, turn_embedding: list[float], top_k: int = 64) -> list[tuple[Facet, float]]:
        return self._search[:top_k]

    def universals(self) -> list[Facet]:
        return self._universals


class _FakeEncoder:
    """Stub encoder returning a zero vector."""

    @property
    def dim(self) -> int:
        return 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    def encode_one(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNeverApplies:
    def test_requires_external_data_excluded(self) -> None:
        facet = _make_facet(observability="requires_external_data")
        registry = _FakeRegistry(search_results=[(facet, 0.9)])
        gate = FacetGate(registry, _FakeEncoder(), top_k=10, threshold=0.35)  # type: ignore[arg-type]
        candidates = gate.activate(_make_turn())
        assert any(c.facet.facet_id == facet.facet_id for c in candidates)
        matching = [c for c in candidates if c.facet.facet_id == facet.facet_id]
        assert matching[0].applies is False

    def test_not_text_observable_excluded(self) -> None:
        facet = _make_facet(observability="not_text_observable")
        registry = _FakeRegistry(search_results=[(facet, 0.9)])
        gate = FacetGate(registry, _FakeEncoder(), top_k=10, threshold=0.35)  # type: ignore[arg-type]
        candidates = gate.activate(_make_turn())
        matching = [c for c in candidates if c.facet.facet_id == facet.facet_id]
        assert matching[0].applies is False

    def test_never_applies_set_has_correct_members(self) -> None:
        assert "requires_external_data" in _NEVER_APPLIES
        assert "not_text_observable" in _NEVER_APPLIES
        assert "observable" not in _NEVER_APPLIES


class TestUniversals:
    def test_universal_always_applies(self) -> None:
        u = _make_facet(facet_id="u1", scope="universal")
        registry = _FakeRegistry(search_results=[], universal_facets=[u])
        gate = FacetGate(registry, _FakeEncoder(), threshold=0.99)  # type: ignore[arg-type]
        candidates = gate.activate(_make_turn())
        assert len(candidates) == 1
        assert candidates[0].applies is True
        assert candidates[0].applicability_score == 1.0

    def test_universals_deduped_when_also_in_search(self) -> None:
        u = _make_facet(facet_id="u1", scope="universal")
        registry = _FakeRegistry(search_results=[(u, 0.8)], universal_facets=[u])
        gate = FacetGate(registry, _FakeEncoder())  # type: ignore[arg-type]
        candidates = gate.activate(_make_turn())
        ids = [c.facet.facet_id for c in candidates]
        assert ids.count("u1") == 1


class TestObservableThreshold:
    def test_above_threshold_applies(self) -> None:
        facet = _make_facet(observability="observable")
        registry = _FakeRegistry(search_results=[(facet, 0.8)])
        gate = FacetGate(registry, _FakeEncoder(), threshold=0.35)  # type: ignore[arg-type]
        candidates = gate.activate(_make_turn())
        assert candidates[0].applies is True

    def test_below_threshold_does_not_apply(self) -> None:
        facet = _make_facet(observability="observable")
        registry = _FakeRegistry(search_results=[(facet, 0.2)])
        gate = FacetGate(registry, _FakeEncoder(), threshold=0.35)  # type: ignore[arg-type]
        candidates = gate.activate(_make_turn())
        assert candidates[0].applies is False


class TestExplicitMentionCheck:
    def test_applies_when_name_token_present(self) -> None:
        facet = _make_facet(
            facet_name="Spelling Accuracy", observability="requires_explicit_mention"
        )
        registry = _FakeRegistry(search_results=[(facet, 0.9)])
        gate = FacetGate(registry, _FakeEncoder())  # type: ignore[arg-type]
        turn = _make_turn("Your spelling is terrible")
        candidates = gate.activate(turn)
        assert candidates[0].applies is True

    def test_does_not_apply_when_name_absent(self) -> None:
        facet = _make_facet(facet_name="Empathy", observability="requires_explicit_mention")
        registry = _FakeRegistry(search_results=[(facet, 0.9)])
        gate = FacetGate(registry, _FakeEncoder())  # type: ignore[arg-type]
        turn = _make_turn("Hello, how are you doing today?")
        candidates = gate.activate(turn)
        assert candidates[0].applies is False


class TestApplicableConvenience:
    def test_applicable_filters_to_applies_true(self) -> None:
        obs = _make_facet(facet_id="f_obs", observability="observable")
        ext = _make_facet(facet_id="f_ext", observability="requires_external_data")
        registry = _FakeRegistry(search_results=[(obs, 0.9), (ext, 0.9)])
        gate = FacetGate(registry, _FakeEncoder())  # type: ignore[arg-type]
        result = gate.applicable(_make_turn())
        assert all(c.applies for c in result)
        assert any(c.facet.facet_id == "f_obs" for c in result)
        assert not any(c.facet.facet_id == "f_ext" for c in result)


class TestCandidateDataclass:
    def test_candidate_slots(self) -> None:
        c = Candidate(
            facet=_make_facet(), retrieval_score=0.5, applies=True, applicability_score=0.5
        )
        assert c.retrieval_score == pytest.approx(0.5)
        assert c.applies is True
