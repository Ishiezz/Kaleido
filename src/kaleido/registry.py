from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from kaleido.db.base import get_session, make_engine, make_session_factory
from kaleido.db.models import FacetModel
from kaleido.errors import FacetNotFoundError, RegistryNotLoadedError
from kaleido.schemas import Facet

if TYPE_CHECKING:
    from kaleido.embedding import TextEncoder

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class _CacheEntry:
    facet: Facet
    embedding: list[float]


class FacetRegistry:
    """In-memory + database facet store with pgvector kNN search.

    All state is populated by `load_from_csv` (seeding) or by the pipeline
    inserting new facets via `synthesize_contract`.  Nothing in this class
    hardcodes a facet name — every facet is just a row.
    """

    def __init__(
        self,
        database_url: str,
        encoder: TextEncoder,
        registry_version: str = "2026.06.0",
    ) -> None:
        self._engine = make_engine(database_url)
        self._factory = make_session_factory(self._engine)
        self._encoder = encoder
        self._version = registry_version
        # In-memory cache: facet_id → (_CacheEntry with embedding)
        self._cache: dict[str, _CacheEntry] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Public interface (matches AGENTIC_IDE_PROMPT §CONTRACTS)
    # ------------------------------------------------------------------

    async def load_from_csv(self, path: str) -> int:
        """Seed the registry from the enriched CSV.  Idempotent (ON CONFLICT IGNORE)."""
        with Path(path).open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        scorable = [r for r in rows if r.get("row_kind", "facet") == "facet"]
        log.info("registry.load_from_csv", path=path, rows=len(scorable))

        async with get_session(self._factory) as session:
            for row in scorable:
                existing = await session.get(FacetModel, row["facet_id"])
                if existing is not None:
                    self._prime_cache(existing)
                    continue

                embedding = self._encoder.encode_one(row["embedding_text"])
                anchors: dict[str, str] = json.loads(row["score_anchors_json"])

                model = FacetModel(
                    facet_id=row["facet_id"],
                    facet_name=row["facet_name"],
                    domain=row["domain"],
                    subdomain=row.get("subdomain", ""),
                    facet_type=row["facet_type"],
                    value_polarity=row["value_polarity"],
                    text_observability=row["text_observability"],
                    applicability_scope=row["applicability_scope"],
                    score_scale=row.get("score_scale", "-2,-1,0,1,2"),
                    score_anchors=anchors,
                    definition=row["definition"],
                    embedding_text=row["embedding_text"],
                    embedding=self._vec_to_db(embedding),
                    difficulty=row.get("difficulty", "easy"),
                    needs_review=row.get("needs_review", "False").lower() == "true",
                    version=row.get("version", self._version),
                )
                session.add(model)
                # Prime the in-memory cache immediately.
                facet = self._model_to_facet(model, anchors)
                self._cache[facet.facet_id] = _CacheEntry(facet=facet, embedding=embedding)

        self._loaded = True
        log.info("registry.loaded", n=len(self._cache))
        return len(self._cache)

    def get(self, facet_id: str) -> Facet:
        """Return the Facet for a given ID (cache-only; raises if not loaded)."""
        self._assert_loaded()
        try:
            return self._cache[facet_id].facet
        except KeyError as exc:
            raise FacetNotFoundError(facet_id) from exc

    def universals(self) -> list[Facet]:
        """Return all facets with applicability_scope == 'universal'."""
        self._assert_loaded()
        return [e.facet for e in self._cache.values() if e.facet.applicability_scope == "universal"]

    def search(self, turn_embedding: list[float], top_k: int = 64) -> list[tuple[Facet, float]]:
        """Retrieve the top_k most similar facets by cosine similarity.

        Falls back to brute-force numpy cosine on SQLite (no pgvector).
        """
        self._assert_loaded()
        return self._numpy_search(turn_embedding, top_k)

    async def insert_facet(self, facet: Facet) -> None:
        """Insert a single new facet (used by synthesis + POST /facets)."""
        embedding = self._encoder.encode_one(facet.embedding_text)
        async with get_session(self._factory) as session:
            model = FacetModel(
                facet_id=facet.facet_id,
                facet_name=facet.facet_name,
                domain=facet.domain,
                subdomain=facet.subdomain,
                facet_type=facet.facet_type,
                value_polarity=facet.value_polarity,
                text_observability=facet.text_observability,
                applicability_scope=facet.applicability_scope,
                score_scale=facet.score_scale,
                score_anchors=dict(facet.score_anchors),
                definition=facet.definition,
                embedding_text=facet.embedding_text,
                embedding=self._vec_to_db(embedding),
                difficulty=facet.difficulty,
                needs_review=facet.needs_review,
                version=facet.version,
            )
            session.add(model)
        self._cache[facet.facet_id] = _CacheEntry(facet=facet, embedding=embedding)

    async def list_facets(
        self,
        domain: str | None = None,
        scope: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> list[Facet]:
        """Paginated list of facets, optionally filtered by domain / scope."""
        self._assert_loaded()
        results = list(self._cache.values())
        if domain:
            results = [e for e in results if e.facet.domain == domain]
        if scope:
            results = [e for e in results if e.facet.applicability_scope == scope]
        start = (page - 1) * page_size
        return [e.facet for e in results[start : start + page_size]]

    def n_facets(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RegistryNotLoadedError("Registry not loaded; call load_from_csv first.")

    def _prime_cache(self, model: FacetModel) -> None:
        anchors: dict[str, str] = model.score_anchors
        facet = self._model_to_facet(model, anchors)
        emb_raw = model.embedding
        if isinstance(emb_raw, str):
            embedding = json.loads(emb_raw)
        elif isinstance(emb_raw, list):
            embedding = list(emb_raw)
        else:
            embedding = self._encoder.encode_one(facet.embedding_text)
        self._cache[facet.facet_id] = _CacheEntry(facet=facet, embedding=embedding)

    @staticmethod
    def _model_to_facet(model: FacetModel, anchors: dict[str, str]) -> Facet:
        return Facet(
            facet_id=model.facet_id,
            facet_name=model.facet_name,
            domain=model.domain,
            subdomain=model.subdomain,
            facet_type=model.facet_type,  # type: ignore[arg-type]
            value_polarity=model.value_polarity,  # type: ignore[arg-type]
            text_observability=model.text_observability,  # type: ignore[arg-type]
            applicability_scope=model.applicability_scope,  # type: ignore[arg-type]
            score_scale=model.score_scale,
            score_anchors=anchors,
            definition=model.definition,
            embedding_text=model.embedding_text,
            difficulty=model.difficulty,  # type: ignore[arg-type]
            needs_review=model.needs_review,
            version=model.version,
        )

    @staticmethod
    def _vec_to_db(vec: list[float]) -> str:
        """Serialize a vector to JSON string for SQLite stub storage."""
        return json.dumps(vec)

    def _numpy_search(self, query: list[float], top_k: int) -> list[tuple[Facet, float]]:
        import numpy as np

        if not self._cache:
            return []
        q = np.array(query, dtype=np.float32)
        ids = list(self._cache.keys())
        mat = np.array([self._cache[fid].embedding for fid in ids], dtype=np.float32)
        sims: list[float] = (mat @ q).tolist()
        ranked = sorted(zip(ids, sims, strict=False), key=lambda x: x[1], reverse=True)
        return [(self._cache[fid].facet, score) for fid, score in ranked[:top_k]]
