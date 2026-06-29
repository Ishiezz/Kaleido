from __future__ import annotations

import pytest

from kaleido.schemas import Facet
from kaleido.synthesis import _rule_based_contract, _to_slug, synthesize_contract


class TestRuleBasedContract:
    def test_returns_facet(self) -> None:
        f = _rule_based_contract("Response Fluency", "linguistic_quality")
        assert isinstance(f, Facet)

    def test_facet_name_preserved(self) -> None:
        f = _rule_based_contract("Response Fluency", "linguistic_quality")
        assert f.facet_name == "Response Fluency"

    def test_domain_preserved(self) -> None:
        f = _rule_based_contract("Empathy", "pragmatics")
        assert f.domain == "pragmatics"

    def test_all_five_anchors_present(self) -> None:
        f = _rule_based_contract("Clarity", "discourse")
        assert set(f.score_anchors.keys()) == {"-2", "-1", "0", "1", "2"}

    def test_anchors_non_empty(self) -> None:
        f = _rule_based_contract("Clarity", "discourse")
        assert all(v.strip() for v in f.score_anchors.values())

    def test_needs_review_true(self) -> None:
        f = _rule_based_contract("Novelty", "creativity")
        assert f.needs_review is True

    def test_facet_id_contains_slug(self) -> None:
        f = _rule_based_contract("Response Fluency", "linguistic_quality")
        assert "response_fluency" in f.facet_id

    def test_score_scale_correct(self) -> None:
        f = _rule_based_contract("Clarity", "discourse")
        assert f.score_scale == "-2,-1,0,1,2"

    def test_idempotent_fields(self) -> None:
        f1 = _rule_based_contract("Tone", "pragmatics")
        f2 = _rule_based_contract("Tone", "pragmatics")
        # facet_ids differ (uuid suffix) but other fields are stable.
        assert f1.facet_name == f2.facet_name
        assert f1.domain == f2.domain


class TestSlug:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Response Fluency", "response_fluency"),
            ("Empathy", "empathy"),
            ("Self-Awareness", "self_awareness"),
            ("Risk-Taking Behavior", "risk_taking_behavior"),
        ],
    )
    def test_to_slug(self, name: str, expected: str) -> None:
        assert _to_slug(name) == expected


class TestSynthesizeContractStub:
    @pytest.mark.asyncio
    async def test_stub_returns_facet(self) -> None:
        f = await synthesize_contract("Assertiveness", "personality_trait", backend="stub")
        assert isinstance(f, Facet)

    @pytest.mark.asyncio
    async def test_stub_anchors_valid(self) -> None:
        f = await synthesize_contract("Clarity", "discourse", backend="stub")
        assert set(f.score_anchors.keys()) == {"-2", "-1", "0", "1", "2"}

    @pytest.mark.asyncio
    async def test_stub_pydantic_validates(self) -> None:
        f = await synthesize_contract("Warmth", "emotional", backend="stub")
        # Re-instantiate to confirm it passes Pydantic validators.
        f2 = Facet(**f.model_dump())
        assert f2.facet_id == f.facet_id
