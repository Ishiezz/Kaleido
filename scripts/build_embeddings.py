"""Backfill the embedding column for all facets in the database.

Useful after switching embedding models or when rows were inserted without embeddings.

Usage:
    python scripts/build_embeddings.py [--db URL] [--model MODEL_NAME]
"""
from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, "src")

from sqlalchemy import select

from kaleido.config import Settings
from kaleido.db.base import get_session, make_engine, make_session_factory
from kaleido.db.models import FacetModel
from kaleido.embedding import make_encoder


async def main() -> None:
    cfg = Settings()
    engine = make_engine(cfg.database_url)
    factory = make_session_factory(engine)
    encoder = make_encoder(cfg.backend, cfg.embedding_model)

    async with get_session(factory) as session:
        result = await session.execute(select(FacetModel))
        facets = result.scalars().all()
        updated = 0
        for f in facets:
            if f.embedding is None:
                vec = encoder.encode_one(f.embedding_text)
                f.embedding = json.dumps(vec)
                updated += 1
        print(f"[build_embeddings] Updated {updated}/{len(facets)} facets.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
