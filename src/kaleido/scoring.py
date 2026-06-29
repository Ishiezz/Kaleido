from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import structlog

from kaleido.errors import ScoringError

if TYPE_CHECKING:
    from kaleido.schemas import Facet, Turn

log = structlog.get_logger(__name__)

# The five valid ordinal labels (centred, signed).
VALID_LABELS: frozenset[int] = frozenset({-2, -1, 0, 1, 2})
_LABEL_STRS: list[str] = ["-2", "-1", "0", "1", "2"]

# One fixed template — never per-facet branching.
_SCORE_TEMPLATE = """\
You are an objective conversation evaluator.

## Facet to evaluate
Name: {facet_name}
Definition: {definition}

## Ordinal scale (choose exactly one)
{anchors}

## Conversation turn
Role: {role}
Text: {turn_text}

## Prior context (most recent first)
{context}

## Task
Read the turn carefully. Choose the single integer label ({score_scale}) that \
best describes how this facet applies to this turn.
If you quote any evidence from the turn, do so after the label on a new line \
starting with "Evidence:".

Respond with ONLY the label integer (e.g. -2) on the first line.\
"""


@runtime_checkable
class ScorerBackend(Protocol):
    """Interface for LLM inference backends."""

    model_name: str

    def score_one(
        self,
        prompt: str,
        *,
        seed: int,
        temperature: float,
    ) -> tuple[str, dict[str, float]]:
        """Return (raw_label_token, logprobs_over_5_labels)."""
        ...


class StubBackend:
    """CPU-only backend: always returns 0 with uniform logprobs.  No model needed."""

    model_name: str = "stub"

    def score_one(
        self,
        prompt: str,
        *,
        seed: int,
        temperature: float,
    ) -> tuple[str, dict[str, float]]:
        uniform = {k: 0.2 for k in _LABEL_STRS}
        return "0", uniform


class VLLMBackend:
    """vLLM OpenAI-compat inference backend with guided decoding."""

    def __init__(self, base_url: str, model: str) -> None:
        self.model_name = model
        self._base_url = base_url.rstrip("/")

    def score_one(
        self,
        prompt: str,
        *,
        seed: int,
        temperature: float,
    ) -> tuple[str, dict[str, float]]:
        try:
            import httpx
        except ImportError as exc:
            raise ScoringError("httpx is required for VLLMBackend") from exc

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": temperature,
            "seed": seed,
            "logprobs": 5,
            "guided_choice": _LABEL_STRS,
        }
        try:
            resp = httpx.post(
                f"{self._base_url}/completions",
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise ScoringError(f"vLLM request failed: {exc}") from exc

        choice = data["choices"][0]
        raw_label = choice["text"].strip()
        top_logprobs: dict[str, float] = {}
        if choice.get("logprobs") and choice["logprobs"].get("top_logprobs"):
            top_logprobs = choice["logprobs"]["top_logprobs"][0]

        # Ensure all five labels are present in the logprob dict.
        import math

        logprob_map: dict[str, float] = {}
        for lbl in _LABEL_STRS:
            logprob_map[lbl] = math.exp(top_logprobs.get(lbl, -20.0))

        total = sum(logprob_map.values())
        if total > 0:
            logprob_map = {k: v / total for k, v in logprob_map.items()}

        return raw_label, logprob_map


def _build_prompt(turn: Turn, facet: Facet) -> str:
    """Render the single fixed template with facet data injected as strings."""
    anchors_str = "\n".join(
        f"  {label}: {desc}" for label, desc in sorted(facet.score_anchors.items())
    )
    context_str = "\n".join(f"- {t}" for t in turn.context[::-1]) if turn.context else "(none)"
    return _SCORE_TEMPLATE.format(
        facet_name=facet.facet_name,
        definition=facet.definition,
        anchors=anchors_str,
        role=turn.role,
        turn_text=turn.text,
        context=context_str,
        score_scale=facet.score_scale,
    )


def _parse_label(raw: str) -> int:
    """Extract the integer label from a raw model output string."""
    m = re.search(r"-?[012]", raw.strip())
    if not m:
        return 0
    val = int(m.group())
    return val if val in VALID_LABELS else 0


def _extract_evidence(raw_output: str, turn_text: str) -> str | None:
    """Pull the 'Evidence: ...' line from model output, if present."""
    for line in raw_output.splitlines():
        if line.lower().startswith("evidence:"):
            span = line.split(":", 1)[1].strip()
            if span and span in turn_text:
                return span
    return None


class OrdinalScorer:
    """Score a single (turn, facet) pair via the configured backend.

    Each call is its own constrained generation — never a one-shot mega-prompt.
    """

    def __init__(self, backend: ScorerBackend, temperature: float = 0.0) -> None:
        self._backend = backend
        self._temperature = temperature

    def score(
        self,
        turn: Turn,
        facet: Facet,
        *,
        seed: int = 0,
    ) -> tuple[int, dict[int, float], str | None]:
        """Score one (turn, facet) pair.

        Returns:
            label: int in {-2,-1,0,1,2}
            logprobs: dict mapping each label to its softmax probability
            evidence_span: optional substring of turn.text
        """
        prompt = _build_prompt(turn, facet)
        raw_label, str_logprobs = self._backend.score_one(
            prompt, seed=seed, temperature=self._temperature
        )
        label = _parse_label(raw_label)
        # Convert str keys to int.
        logprobs: dict[int, float] = {int(k): v for k, v in str_logprobs.items()}
        evidence = _extract_evidence(raw_label, turn.text)
        log.debug(
            "scored",
            facet_id=facet.facet_id,
            turn_id=turn.turn_id,
            label=label,
            model=self._backend.model_name,
        )
        return label, logprobs, evidence

    def score_samples(
        self,
        turn: Turn,
        facet: Facet,
        *,
        n_samples: int = 3,
        temperature: float = 0.7,
        base_seed: int = 0,
    ) -> list[int]:
        """Generate multiple stochastic samples for self-consistency confidence."""
        orig_temp = self._temperature
        self._temperature = temperature
        samples: list[int] = []
        try:
            for i in range(n_samples):
                label, _, _ = self.score(turn, facet, seed=base_seed + i)
                samples.append(label)
        finally:
            self._temperature = orig_temp
        return samples


def make_scorer(backend: Literal["stub", "vllm"], **kwargs: object) -> OrdinalScorer:
    """Factory that selects the backend from the config string."""
    if backend == "stub":
        return OrdinalScorer(StubBackend())
    base_url = str(kwargs.get("vllm_base_url", "http://localhost:8000/v1"))
    model = str(kwargs.get("scorer_model", "Qwen/Qwen2.5-7B-Instruct"))
    return OrdinalScorer(VLLMBackend(base_url, model))
