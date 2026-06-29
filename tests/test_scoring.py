from __future__ import annotations

import pytest

from kaleido.schemas import Facet, Turn
from kaleido.scoring import (
    VALID_LABELS,
    OrdinalScorer,
    StubBackend,
    _build_prompt,
    _extract_evidence,
    _parse_label,
)


@pytest.fixture()
def sample_facet(sample_facet_data: dict[str, object]) -> Facet:
    return Facet(**sample_facet_data)  # type: ignore[arg-type]


@pytest.fixture()
def sample_turn(sample_turn_data: dict[str, object]) -> Turn:
    return Turn(**sample_turn_data)  # type: ignore[arg-type]


class TestPromptTemplate:
    def test_no_hardcoded_facet_name(self, sample_turn: Turn, sample_facet: Facet) -> None:
        prompt = _build_prompt(sample_turn, sample_facet)
        # Facet name injected from data, not from code.
        assert "Spelling Accuracy" in prompt
        assert "linguistic quality" in prompt.lower()

    def test_all_anchors_present(self, sample_turn: Turn, sample_facet: Facet) -> None:
        prompt = _build_prompt(sample_turn, sample_facet)
        for label in ["-2", "-1", "0", "1", "2"]:
            assert label in prompt

    def test_turn_text_in_prompt(self, sample_turn: Turn, sample_facet: Facet) -> None:
        prompt = _build_prompt(sample_turn, sample_facet)
        assert sample_turn.text in prompt


class TestParseLabel:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("-2", -2),
            ("-1", -1),
            ("0", 0),
            ("1", 1),
            ("2", 2),
            ("  2  ", 2),
            ("label: -1", -1),
            ("garbage", 0),
        ],
    )
    def test_parse(self, raw: str, expected: int) -> None:
        assert _parse_label(raw) == expected


class TestExtractEvidence:
    def test_extracts_matching_span(self) -> None:
        turn_text = "Hello, how are you today?"
        raw = "1\nEvidence: Hello, how are you today?"
        span = _extract_evidence(raw, turn_text)
        assert span == "Hello, how are you today?"

    def test_returns_none_when_not_in_turn(self) -> None:
        span = _extract_evidence("1\nEvidence: unrelated text", "Hello!")
        assert span is None

    def test_returns_none_when_no_evidence_line(self) -> None:
        span = _extract_evidence("1", "Hello!")
        assert span is None


class TestOrdinalScorer:
    def test_stub_returns_valid_label(self, sample_turn: Turn, sample_facet: Facet) -> None:
        scorer = OrdinalScorer(StubBackend())
        label, logprobs, evidence = scorer.score(sample_turn, sample_facet)
        assert label in VALID_LABELS

    def test_stub_logprobs_sum_to_one(self, sample_turn: Turn, sample_facet: Facet) -> None:
        scorer = OrdinalScorer(StubBackend())
        _, logprobs, _ = scorer.score(sample_turn, sample_facet)
        assert abs(sum(logprobs.values()) - 1.0) < 1e-5

    def test_stub_logprob_keys_are_ints(self, sample_turn: Turn, sample_facet: Facet) -> None:
        scorer = OrdinalScorer(StubBackend())
        _, logprobs, _ = scorer.score(sample_turn, sample_facet)
        assert all(isinstance(k, int) for k in logprobs)
        assert set(logprobs.keys()) == {-2, -1, 0, 1, 2}

    def test_stub_score_samples_returns_list(self, sample_turn: Turn, sample_facet: Facet) -> None:
        scorer = OrdinalScorer(StubBackend())
        samples = scorer.score_samples(sample_turn, sample_facet, n_samples=3)
        assert len(samples) == 3
        assert all(s in VALID_LABELS for s in samples)
