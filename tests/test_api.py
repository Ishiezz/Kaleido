from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import FastAPI

from kaleido.schemas import (
    Facet,
    FacetScore,
    Turn,
)

# ---------------------------------------------------------------------------
# Minimal stubs so the API can start without a real DB or model
# ---------------------------------------------------------------------------


def _make_facet() -> Facet:
    return Facet(
        facet_id="f_test",
        facet_name="Spelling Accuracy",
        domain="linguistic_quality",
        subdomain="",
        facet_type="qualitative_trait",
        value_polarity="positive",
        text_observability="observable",
        applicability_scope="universal",
        score_scale="-2,-1,0,1,2",
        score_anchors={"-2": "bad", "-1": "below", "0": "ok", "1": "good", "2": "great"},
        definition="Spelling test facet.",
        embedding_text="spelling accuracy",
    )


def _make_turn(conv_id: str = "c1", idx: int = 0) -> Turn:
    return Turn(
        turn_id=f"{conv_id}_t{idx}",
        conversation_id=conv_id,
        index=idx,
        role="user",
        text="Hello there!",
    )


def _make_facet_score(conv_id: str = "c1") -> FacetScore:
    return FacetScore(
        facet_id="f_test",
        turn_id=f"{conv_id}_t0",
        applies=True,
        score=0,
        confidence=0.6,
        abstained=False,
        model_name="stub",
        registry_version="test",
    )


# ---------------------------------------------------------------------------
# App fixture that bypasses lifespan / real DB
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_registry() -> MagicMock:
    reg = MagicMock()
    reg.n_facets.return_value = 42
    reg.get.return_value = _make_facet()
    reg.list_facets = AsyncMock(return_value=[_make_facet()])
    reg.insert_facet = AsyncMock()
    return reg


@pytest.fixture()
def mock_pipeline(mock_registry: MagicMock) -> MagicMock:
    pipe = MagicMock()
    pipe.evaluate_conversation = AsyncMock(return_value=[_make_facet_score()])
    pipe.evaluate_turn = AsyncMock(return_value=[_make_facet_score()])
    return pipe


@pytest.fixture()
def app(mock_registry: MagicMock, mock_pipeline: MagicMock) -> FastAPI:
    """Return a FastAPI app with all singletons pre-injected (no lifespan)."""
    import kaleido.api as api_module

    api_module._registry = mock_registry  # type: ignore[attr-defined]
    api_module._pipeline = mock_pipeline  # type: ignore[attr-defined]
    api_module._settings = MagicMock(  # type: ignore[attr-defined]
        backend="stub",
        registry_version="test",
        vllm_base_url="http://localhost:8000/v1",
        scorer_model="stub",
    )

    # Return the app without triggering lifespan.
    from kaleido.api import app as _app

    return _app


@pytest.fixture()
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthz:
    async def test_ok(self, client: AsyncClient) -> None:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["n_facets"] == 42


class TestScoreEndpoint:
    def _payload(self, conv_id: str = "c1") -> dict[str, Any]:
        return {
            "conversation": {
                "conversation_id": conv_id,
                "turns": [
                    {
                        "turn_id": f"{conv_id}_t0",
                        "conversation_id": conv_id,
                        "index": 0,
                        "role": "user",
                        "text": "Hello there!",
                    }
                ],
            }
        }

    async def test_returns_200(self, client: AsyncClient) -> None:
        resp = await client.post("/score", json=self._payload())
        assert resp.status_code == 200

    async def test_response_shape(self, client: AsyncClient) -> None:
        resp = await client.post("/score", json=self._payload())
        data = resp.json()
        assert "conversation_id" in data
        assert "scores" in data
        assert isinstance(data["scores"], list)

    async def test_score_fields_present(self, client: AsyncClient) -> None:
        resp = await client.post("/score", json=self._payload())
        scores = resp.json()["scores"]
        assert len(scores) == 1
        s = scores[0]
        assert "facet_id" in s
        assert "applies" in s
        assert "confidence" in s
        assert 0.0 <= s["confidence"] <= 1.0

    async def test_score_turn_endpoint(self, client: AsyncClient, mock_pipeline: MagicMock) -> None:
        resp = await client.post(
            "/score/turn",
            json={
                "turn_id": "c1_t0",
                "conversation_id": "c1",
                "index": 0,
                "role": "user",
                "text": "Hello!",
            },
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestFacetsEndpoint:
    async def test_list_facets(self, client: AsyncClient) -> None:
        resp = await client.get("/facets")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["facet_id"] == "f_test"

    async def test_get_facet_by_id(self, client: AsyncClient) -> None:
        resp = await client.get("/facets/f_test")
        assert resp.status_code == 200
        assert resp.json()["facet_id"] == "f_test"

    async def test_get_facet_404(self, client: AsyncClient, mock_registry: MagicMock) -> None:
        from kaleido.errors import FacetNotFoundError

        mock_registry.get.side_effect = FacetNotFoundError("no_such_facet")
        resp = await client.get("/facets/no_such_facet")
        assert resp.status_code == 404


class TestInvariants:
    """Property tests: API output must satisfy core score invariants."""

    async def test_applies_false_score_none(
        self, client: AsyncClient, mock_pipeline: MagicMock
    ) -> None:
        na_score = FacetScore(
            facet_id="f_test",
            turn_id="c1_t0",
            applies=False,
            score=None,
            confidence=0.0,
            model_name="stub",
            registry_version="test",
        )
        mock_pipeline.evaluate_conversation = AsyncMock(return_value=[na_score])
        resp = await client.post(
            "/score",
            json={
                "conversation": {
                    "conversation_id": "c1",
                    "turns": [
                        {
                            "turn_id": "c1_t0",
                            "conversation_id": "c1",
                            "index": 0,
                            "role": "user",
                            "text": "hi",
                        }
                    ],
                }
            },
        )
        assert resp.status_code == 200
        s = resp.json()["scores"][0]
        assert s["applies"] is False
        assert s["score"] is None

    async def test_confidence_always_in_unit_interval(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/score",
            json={
                "conversation": {
                    "conversation_id": "c2",
                    "turns": [
                        {
                            "turn_id": "c2_t0",
                            "conversation_id": "c2",
                            "index": 0,
                            "role": "assistant",
                            "text": "Sure!",
                        }
                    ],
                }
            },
        )
        for s in resp.json()["scores"]:
            assert 0.0 <= s["confidence"] <= 1.0
