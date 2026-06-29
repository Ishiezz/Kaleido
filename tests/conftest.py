from __future__ import annotations

import os

import pytest

# Force stub backend for all tests — no GPU, no model downloads.
os.environ.setdefault("KALEIDO_BACKEND", "stub")
os.environ.setdefault("KALEIDO_DATABASE_URL", "sqlite+aiosqlite:///./test_kaleido.db")


@pytest.fixture()
def sample_facet_data() -> dict[str, object]:
    return {
        "facet_id": "K0001",
        "facet_name": "Spelling Accuracy",
        "domain": "linguistic_quality",
        "subdomain": "linguistic_quality",
        "facet_type": "level_score",
        "value_polarity": "positive",
        "text_observability": "observable",
        "applicability_scope": "universal",
        "score_scale": "-2,-1,0,1,2",
        "score_anchors": {
            "-2": "Very poor spelling.",
            "-1": "Below-average spelling.",
            "0": "Average or not evidenced.",
            "1": "Above-average spelling.",
            "2": "Excellent spelling.",
        },
        "definition": "Degree to which the turn exhibits 'Spelling Accuracy' (linguistic quality).",
        "embedding_text": "Spelling Accuracy | linguistic quality | ...",
        "difficulty": "easy",
        "needs_review": False,
        "version": "2026.06.0",
    }


@pytest.fixture()
def sample_turn_data() -> dict[str, object]:
    return {
        "turn_id": "conv_0001_t0",
        "conversation_id": "conv_0001",
        "index": 0,
        "role": "user",
        "text": "Hello, how are you today?",
        "context": [],
    }
