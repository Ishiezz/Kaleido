from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from kaleido.db.base import Base
from kaleido.db.models import (
    ConversationModel,
    FacetModel,
    FacetScoreModel,
    TurnModel,
)


@pytest.fixture()
async def engine():  # type: ignore[return]
    eng = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture()
async def session(engine):  # type: ignore[return]
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s, s.begin():
        yield s


async def _seed_facet(session, facet_id: str = "K0001") -> FacetModel:
    f = FacetModel(
        facet_id=facet_id,
        facet_name="Spelling Accuracy",
        domain="linguistic_quality",
        subdomain="linguistic_quality",
        facet_type="level_score",
        value_polarity="positive",
        text_observability="observable",
        applicability_scope="universal",
        score_anchors={"-2": "poor", "-1": "below", "0": "avg", "1": "above", "2": "excel"},
        definition="Degree of spelling accuracy.",
        embedding_text="Spelling Accuracy | linguistic quality",
        version="2026.06.0",
    )
    session.add(f)
    return f


class TestModels:
    async def test_facet_insert(self, session) -> None:  # type: ignore[return]
        await _seed_facet(session)
        from sqlalchemy import select

        result = await session.execute(select(FacetModel))
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].facet_id == "K0001"

    async def test_conversation_turn_insert(self, session) -> None:  # type: ignore[return]
        conv = ConversationModel(conversation_id="conv_0001", meta={})
        session.add(conv)
        turn = TurnModel(
            turn_id="conv_0001_t0",
            conversation_id="conv_0001",
            idx=0,
            role="user",
            text="Hello!",
        )
        session.add(turn)

        from sqlalchemy import select

        result = await session.execute(select(TurnModel))
        turns = result.scalars().all()
        assert len(turns) == 1

    async def test_facet_score_insert(self, session) -> None:  # type: ignore[return]
        await _seed_facet(session)
        conv = ConversationModel(conversation_id="conv_0001", meta={})
        session.add(conv)
        turn = TurnModel(
            turn_id="conv_0001_t0",
            conversation_id="conv_0001",
            idx=0,
            role="user",
            text="Hello!",
        )
        session.add(turn)
        score = FacetScoreModel(
            turn_id="conv_0001_t0",
            facet_id="K0001",
            applies=True,
            score=1,
            confidence=0.8,
            abstained=False,
            model_name="stub",
            registry_version="2026.06.0",
        )
        session.add(score)

        from sqlalchemy import select

        result = await session.execute(select(FacetScoreModel))
        scores = result.scalars().all()
        assert len(scores) == 1
        assert scores[0].score == 1
