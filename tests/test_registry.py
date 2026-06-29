from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from kaleido.db.base import Base
from kaleido.embedding import HashStubEmbedder
from kaleido.registry import FacetRegistry

ENRICHED_CSV = Path("data/processed/facets_enriched.csv")


@pytest.fixture()
async def tmp_db():  # type: ignore[return]
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture()
async def registry(tmp_db):  # type: ignore[return]
    encoder = HashStubEmbedder()
    reg = FacetRegistry.__new__(FacetRegistry)
    reg._engine = tmp_db
    reg._factory = async_sessionmaker(tmp_db, expire_on_commit=False)
    reg._encoder = encoder
    reg._version = "2026.06.0"
    reg._cache = {}
    reg._loaded = False
    return reg


class TestEmbedding:
    def test_stub_dim(self) -> None:
        enc = HashStubEmbedder()
        assert enc.dim == 384

    def test_stub_deterministic(self) -> None:
        enc = HashStubEmbedder()
        v1 = enc.encode_one("hello world")
        v2 = enc.encode_one("hello world")
        assert v1 == v2

    def test_stub_unit_norm(self) -> None:
        import numpy as np

        enc = HashStubEmbedder()
        v = enc.encode_one("test")
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-5

    def test_stub_batch(self) -> None:
        enc = HashStubEmbedder()
        vecs = enc.encode(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)


@pytest.mark.skipif(not ENRICHED_CSV.exists(), reason="enriched CSV not present")
class TestRegistry:
    async def test_load_from_csv(self, registry: FacetRegistry) -> None:
        n = await registry.load_from_csv(str(ENRICHED_CSV))
        assert n >= 300  # brief says ≥300 scorable facets

    async def test_get_facet(self, registry: FacetRegistry) -> None:
        await registry.load_from_csv(str(ENRICHED_CSV))
        facet = registry.get("K0001")
        assert facet.facet_id == "K0001"
        assert len(facet.score_anchors) == 5

    async def test_universals(self, registry: FacetRegistry) -> None:
        await registry.load_from_csv(str(ENRICHED_CSV))
        universals = registry.universals()
        assert len(universals) >= 1
        assert all(f.applicability_scope == "universal" for f in universals)

    async def test_search_returns_top_k(self, registry: FacetRegistry) -> None:
        await registry.load_from_csv(str(ENRICHED_CSV))
        query = registry._encoder.encode_one("How do you spell correctly?")
        results = registry.search(query, top_k=5)
        assert len(results) == 5
        # Scores must be in descending order.
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    async def test_external_data_facets_present(self, registry: FacetRegistry) -> None:
        await registry.load_from_csv(str(ENRICHED_CSV))
        ext = [
            f
            for f in registry._cache.values()
            if f.facet.text_observability == "requires_external_data"
        ]
        assert len(ext) >= 1  # e.g. FSH level facet

    async def test_idempotent_load(self, registry: FacetRegistry) -> None:
        n1 = await registry.load_from_csv(str(ENRICHED_CSV))
        n2 = await registry.load_from_csv(str(ENRICHED_CSV))
        assert n1 == n2
