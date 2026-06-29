from __future__ import annotations

import math

import structlog

log = structlog.get_logger(__name__)

# Ordinal labels in order — used for variance calculation.
_LABELS: list[int] = [-2, -1, 0, 1, 2]
_LABEL_SET: frozenset[int] = frozenset(_LABELS)
# Max possible variance of a distribution over {-2,-1,0,1,2}: ~4.0 (all mass at extremes).
_MAX_VAR: float = 4.0


class Calibrator:
    """Temperature-scaling calibrator for confidence scores.

    Fit with `fit(raw_confidences, correctness_flags)`.
    Apply with `apply(raw)`.
    Identity (T=1.0) until `fit` is called.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self._T = temperature

    @property
    def temperature(self) -> float:
        return self._T

    def fit(self, raw_scores: list[float], correct: list[bool]) -> None:
        """Fit temperature T on validation data via NLL minimisation.

        raw_scores: predicted confidences in [0,1]
        correct:    whether each prediction matched ground truth
        """
        try:
            from scipy.optimize import minimize_scalar
        except ImportError:
            log.warning("scipy not available; calibrator remains identity")
            return

        def nll(T: float) -> float:
            total = 0.0
            for p, c in zip(raw_scores, correct, strict=False):
                # Clamp logit to avoid log(0).
                logit = math.log(max(p, 1e-9)) - math.log(max(1 - p, 1e-9))
                q = _sigmoid(logit / max(T, 1e-6))
                total -= math.log(q if c else (1 - q) + 1e-9)
            return total

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self._T = float(result.x)
        log.info("calibrator.fit", temperature=self._T)

    def apply(self, raw: float) -> float:
        """Scale raw confidence through the learned temperature."""
        if self._T == 1.0:
            return float(raw)
        logit = math.log(max(raw, 1e-9)) - math.log(max(1 - raw, 1e-9))
        return _sigmoid(logit / self._T)


def fuse_confidence(
    logprobs: dict[int, float],
    samples: list[int],
    calibrator: Calibrator | None = None,
    *,
    weight_logprob: float = 0.5,
    weight_consistency: float = 0.3,
    weight_ordinal_var: float = 0.2,
) -> float:
    """Fuse three signals into a single calibrated confidence in [0,1].

    Signals:
      1. Logprob margin   — winner probability minus runner-up.
      2. Self-consistency — fraction of samples agreeing with the MAP label.
      3. Ordinal variance — 1 - (variance / max_var) over the sample distribution.

    Weights must sum to 1.0 (not enforced, but recommended).
    """
    # --- Signal 1: logprob margin ---
    if logprobs:
        sorted_probs = sorted(logprobs.values(), reverse=True)
        winner_prob = sorted_probs[0]
        runner_up = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin = (winner_prob - runner_up) / 2.0  # normalise to [0, 0.5]
        margin = min(max(margin, 0.0), 1.0)
    else:
        margin = 0.5

    # --- Signal 2: self-consistency ---
    if samples:
        map_label = max(logprobs, key=logprobs.__getitem__) if logprobs else 0
        consistency = sum(1 for s in samples if s == map_label) / len(samples)
    else:
        consistency = 1.0  # no samples → neutral

    # --- Signal 3: ordinal variance ---
    if samples and len(samples) > 1:
        mean = sum(samples) / len(samples)
        var = sum((s - mean) ** 2 for s in samples) / len(samples)
        ordinal_stability = 1.0 - min(var / _MAX_VAR, 1.0)
    else:
        ordinal_stability = 1.0

    raw = (
        weight_logprob * margin
        + weight_consistency * consistency
        + weight_ordinal_var * ordinal_stability
    )
    raw = min(max(raw, 0.0), 1.0)

    if calibrator is not None:
        raw = calibrator.apply(raw)

    return raw


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
