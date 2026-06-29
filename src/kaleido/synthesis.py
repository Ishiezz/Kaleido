from __future__ import annotations

import json
import uuid
from typing import Any

import structlog

from kaleido.errors import SynthesisError
from kaleido.schemas import Facet

log = structlog.get_logger(__name__)

# Fixed prompt: ask the LLM to output a JSON Facet contract.
_SYNTHESIS_TEMPLATE = """\
You are a conversation quality expert creating scoring rubrics.

Generate a JSON scoring contract for the following facet:
  Facet name: {facet_name}
  Domain: {domain}

The JSON must have these exact keys:
  facet_id, facet_name, domain, subdomain, facet_type, value_polarity,
  text_observability, applicability_scope, score_scale, score_anchors,
  definition, embedding_text, difficulty

score_anchors must be a JSON object with keys "-2", "-1", "0", "1", "2".
score_scale must be "-2,-1,0,1,2".
facet_id must be a unique slug like "{slug}".

Respond with ONLY the JSON object, no markdown fences.\
"""


async def synthesize_contract(
    facet_name: str,
    domain: str,
    *,
    backend: str = "stub",
    vllm_base_url: str = "http://localhost:8000/v1",
    scorer_model: str = "Qwen/Qwen2.5-7B-Instruct",
) -> Facet:
    """Generate a Facet schema contract using the configured LLM backend.

    stub mode → returns a rule-based default contract without calling any model.
    vllm mode → calls the LLM with JSON-mode constrained generation.
    """
    if backend == "stub":
        return _rule_based_contract(facet_name, domain)

    raw = await _vllm_synthesize(facet_name, domain, vllm_base_url, scorer_model)
    return _parse_and_validate(raw, facet_name, domain)


# ---------------------------------------------------------------------------
# Rule-based fallback (also used in stub mode)
# ---------------------------------------------------------------------------


def _rule_based_contract(facet_name: str, domain: str) -> Facet:
    slug = _to_slug(facet_name)
    return Facet(
        facet_id=f"synth_{slug}_{uuid.uuid4().hex[:6]}",
        facet_name=facet_name,
        domain=domain,
        subdomain="",
        facet_type="qualitative_trait",
        value_polarity="bipolar",
        text_observability="observable",
        applicability_scope="conditional",
        score_scale="-2,-1,0,1,2",
        score_anchors={
            "-2": f"Very poor {facet_name.lower()}",
            "-1": f"Below-average {facet_name.lower()}",
            "0": f"Average / no clear signal for {facet_name.lower()}",
            "1": f"Above-average {facet_name.lower()}",
            "2": f"Excellent {facet_name.lower()}",
        },
        definition=f"Assesses the {facet_name.lower()} quality of the conversational turn.",
        embedding_text=f"{facet_name} {domain} quality assessment rubric",
        difficulty="medium",
        needs_review=True,
    )


# ---------------------------------------------------------------------------
# vLLM path
# ---------------------------------------------------------------------------


async def _vllm_synthesize(
    facet_name: str,
    domain: str,
    base_url: str,
    model: str,
) -> str:
    try:
        import httpx
    except ImportError as exc:
        raise SynthesisError("httpx required for vLLM synthesis") from exc

    slug = _to_slug(facet_name)
    prompt = _SYNTHESIS_TEMPLATE.format(facet_name=facet_name, domain=domain, slug=slug)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2,
        "guided_json": _facet_json_schema(),
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{base_url.rstrip('/')}/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise SynthesisError(f"vLLM synthesis request failed: {exc}") from exc

    return str(data["choices"][0]["text"])


def _parse_and_validate(raw: str, facet_name: str, domain: str) -> Facet:
    try:
        data: dict[str, Any] = json.loads(raw.strip())
        return Facet(**data)
    except Exception as exc:
        log.warning("synthesis.parse_failed", error=str(exc), fallback=True)
        return _rule_based_contract(facet_name, domain)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_slug(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _facet_json_schema() -> dict[str, Any]:
    """Minimal JSON schema for guided vLLM decoding."""
    return {
        "type": "object",
        "properties": {
            "facet_id": {"type": "string"},
            "facet_name": {"type": "string"},
            "domain": {"type": "string"},
            "subdomain": {"type": "string"},
            "facet_type": {
                "type": "string",
                "enum": ["qualitative_trait", "level_score", "frequency_count", "binary"],
            },
            "value_polarity": {
                "type": "string",
                "enum": ["positive", "negative", "bipolar", "neutral"],
            },
            "text_observability": {
                "type": "string",
                "enum": [
                    "observable",
                    "requires_explicit_mention",
                    "requires_external_data",
                    "not_text_observable",
                ],
            },
            "applicability_scope": {
                "type": "string",
                "enum": ["universal", "conditional", "rare"],
            },
            "score_scale": {"type": "string"},
            "score_anchors": {
                "type": "object",
                "properties": {
                    "-2": {"type": "string"},
                    "-1": {"type": "string"},
                    "0": {"type": "string"},
                    "1": {"type": "string"},
                    "2": {"type": "string"},
                },
                "required": ["-2", "-1", "0", "1", "2"],
            },
            "definition": {"type": "string"},
            "embedding_text": {"type": "string"},
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        },
        "required": [
            "facet_id",
            "facet_name",
            "domain",
            "facet_type",
            "value_polarity",
            "text_observability",
            "applicability_scope",
            "score_scale",
            "score_anchors",
            "definition",
            "embedding_text",
        ],
    }
