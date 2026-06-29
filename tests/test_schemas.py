from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaleido.schemas import (
    Conversation,
    Facet,
    FacetScore,
    Turn,
)


class TestTurn:
    def test_valid(self, sample_turn_data: dict[str, object]) -> None:
        t = Turn(**sample_turn_data)  # type: ignore[arg-type]
        assert t.role == "user"
        assert t.index == 0

    def test_invalid_role(self, sample_turn_data: dict[str, object]) -> None:
        sample_turn_data["role"] = "bot"
        with pytest.raises(ValidationError):
            Turn(**sample_turn_data)  # type: ignore[arg-type]


class TestConversation:
    def test_valid(self, sample_turn_data: dict[str, object]) -> None:
        conv = Conversation(
            conversation_id="conv_0001",
            turns=[Turn(**sample_turn_data)],  # type: ignore[arg-type]
        )
        assert len(conv.turns) == 1

    def test_mismatched_conversation_id(self, sample_turn_data: dict[str, object]) -> None:
        sample_turn_data["conversation_id"] = "OTHER"
        with pytest.raises(ValidationError):
            Conversation(
                conversation_id="conv_0001",
                turns=[Turn(**sample_turn_data)],  # type: ignore[arg-type]
            )


class TestFacet:
    def test_valid(self, sample_facet_data: dict[str, object]) -> None:
        f = Facet(**sample_facet_data)  # type: ignore[arg-type]
        assert f.facet_id == "K0001"
        assert len(f.score_anchors) == 5

    def test_missing_anchor(self, sample_facet_data: dict[str, object]) -> None:
        anchors = dict(sample_facet_data["score_anchors"])  # type: ignore[arg-type]
        del anchors["-2"]
        sample_facet_data["score_anchors"] = anchors
        with pytest.raises(ValidationError):
            Facet(**sample_facet_data)  # type: ignore[arg-type]


class TestFacetScore:
    def test_not_applicable(self) -> None:
        fs = FacetScore(
            facet_id="K0001",
            turn_id="t1",
            applies=False,
            score=None,
            confidence=0.9,
            abstained=False,
            model_name="stub",
            registry_version="2026.06.0",
        )
        assert fs.score is None

    def test_applicable_with_score(self) -> None:
        fs = FacetScore(
            facet_id="K0001",
            turn_id="t1",
            applies=True,
            score=1,
            confidence=0.75,
            abstained=False,
            model_name="stub",
            registry_version="2026.06.0",
        )
        assert fs.score == 1

    def test_score_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            FacetScore(
                facet_id="K0001",
                turn_id="t1",
                applies=True,
                score=3,  # invalid
                confidence=0.5,
                model_name="stub",
                registry_version="2026.06.0",
            )

    def test_applies_true_score_none_not_abstained_raises(self) -> None:
        with pytest.raises(ValidationError):
            FacetScore(
                facet_id="K0001",
                turn_id="t1",
                applies=True,
                score=None,
                confidence=0.5,
                abstained=False,
                model_name="stub",
                registry_version="2026.06.0",
            )

    def test_applies_false_score_set_raises(self) -> None:
        with pytest.raises(ValidationError):
            FacetScore(
                facet_id="K0001",
                turn_id="t1",
                applies=False,
                score=1,  # invalid — applies is False
                confidence=0.9,
                model_name="stub",
                registry_version="2026.06.0",
            )

    def test_abstained_score_none_ok(self) -> None:
        fs = FacetScore(
            facet_id="K0001",
            turn_id="t1",
            applies=True,
            score=None,
            confidence=0.1,
            abstained=True,
            model_name="stub",
            registry_version="2026.06.0",
        )
        assert fs.abstained is True
