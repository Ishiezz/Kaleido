from __future__ import annotations

import pytest

from kaleido.confidence import Calibrator, fuse_confidence


class TestFuseConfidence:
    def _uniform_logprobs(self) -> dict[int, float]:
        return {-2: 0.2, -1: 0.2, 0: 0.2, 1: 0.2, 2: 0.2}

    def _peaked_logprobs(self) -> dict[int, float]:
        return {-2: 0.01, -1: 0.01, 0: 0.95, 1: 0.02, 2: 0.01}

    def test_returns_float_in_unit_interval(self) -> None:
        conf = fuse_confidence(self._uniform_logprobs(), [0, 0, 0])
        assert 0.0 <= conf <= 1.0

    def test_peaked_logprobs_higher_confidence(self) -> None:
        peaked = fuse_confidence(self._peaked_logprobs(), [0, 0, 0])
        uniform = fuse_confidence(self._uniform_logprobs(), [0, 0, 1])
        assert peaked > uniform

    def test_full_consistency_boosts_confidence(self) -> None:
        consistent = fuse_confidence(self._peaked_logprobs(), [0, 0, 0])
        inconsistent = fuse_confidence(self._peaked_logprobs(), [-2, 1, 2])
        assert consistent > inconsistent

    def test_empty_samples_does_not_crash(self) -> None:
        conf = fuse_confidence(self._uniform_logprobs(), [])
        assert 0.0 <= conf <= 1.0

    def test_empty_logprobs_does_not_crash(self) -> None:
        conf = fuse_confidence({}, [0, 0, 0])
        assert 0.0 <= conf <= 1.0

    def test_weights_sum_one_gives_same_result(self) -> None:
        a = fuse_confidence(
            self._peaked_logprobs(),
            [0, 0],
            weight_logprob=0.5,
            weight_consistency=0.3,
            weight_ordinal_var=0.2,
        )
        b = fuse_confidence(
            self._peaked_logprobs(),
            [0, 0],
            weight_logprob=0.5,
            weight_consistency=0.3,
            weight_ordinal_var=0.2,
        )
        assert a == pytest.approx(b)

    def test_high_variance_samples_lower_confidence(self) -> None:
        stable = fuse_confidence(self._peaked_logprobs(), [0, 0, 0])
        noisy = fuse_confidence(self._peaked_logprobs(), [-2, 2, -2, 2])
        assert stable > noisy

    def test_with_identity_calibrator(self) -> None:
        cal = Calibrator(temperature=1.0)
        conf = fuse_confidence(self._peaked_logprobs(), [0, 0, 0], calibrator=cal)
        no_cal = fuse_confidence(self._peaked_logprobs(), [0, 0, 0])
        assert conf == pytest.approx(no_cal)


class TestCalibrator:
    def test_identity_by_default(self) -> None:
        cal = Calibrator()
        assert cal.apply(0.7) == pytest.approx(0.7)

    def test_temperature_gt1_shrinks_confidence(self) -> None:
        cal = Calibrator(temperature=2.0)
        high = cal.apply(0.9)
        assert high < 0.9

    def test_temperature_lt1_sharpens_confidence(self) -> None:
        cal = Calibrator(temperature=0.5)
        high = cal.apply(0.9)
        assert high > 0.9

    def test_apply_returns_unit_interval(self) -> None:
        for T in [0.5, 1.0, 2.0]:
            cal = Calibrator(temperature=T)
            for p in [0.1, 0.5, 0.9]:
                assert 0.0 <= cal.apply(p) <= 1.0

    def test_fit_sets_temperature(self) -> None:
        cal = Calibrator()
        # Perfect predictions → expect T close to 1.
        raw_scores = [0.9, 0.8, 0.85]
        correct = [True, True, True]
        cal.fit(raw_scores, correct)
        assert isinstance(cal.temperature, float)
        assert cal.temperature > 0.0

    def test_apply_edge_cases(self) -> None:
        cal = Calibrator(temperature=1.0)
        assert cal.apply(0.0) == pytest.approx(0.0, abs=1e-6)
        assert cal.apply(1.0) == pytest.approx(1.0, abs=1e-3)

    def test_symmetry(self) -> None:
        cal = Calibrator(temperature=2.0)
        # apply(0.5) should remain ~0.5 (logit of 0.5 = 0).
        assert cal.apply(0.5) == pytest.approx(0.5, abs=1e-6)
