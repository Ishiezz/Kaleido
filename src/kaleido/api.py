from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from kaleido.confidence import Calibrator
from kaleido.config import Settings
from kaleido.db.base import make_engine, make_session_factory
from kaleido.db.models import ReviewQueueModel
from kaleido.embedding import make_encoder
from kaleido.errors import FacetNotFoundError
from kaleido.gating import FacetGate
from kaleido.pipeline import EvaluationPipeline
from kaleido.registry import FacetRegistry
from kaleido.schemas import (
    Facet,
    FacetCreateRequest,
    FacetScore,
    HealthResponse,
    ReviewResolveRequest,
    ScoreRequest,
    ScoreResponse,
    Turn,
)
from kaleido.scoring import make_scorer
from kaleido.synthesis import synthesize_contract

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Application-level singletons (initialised in lifespan)
# ---------------------------------------------------------------------------

_settings: Settings
_registry: FacetRegistry
_pipeline: EvaluationPipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _settings, _registry, _pipeline

    _settings = Settings()
    encoder = make_encoder(_settings.backend, _settings.embedding_model)
    engine = make_engine(_settings.database_url)
    factory = make_session_factory(engine)

    # Auto-create tables (idempotent). Required for SQLite stub mode;
    # on Postgres the Alembic migration handles this.
    from kaleido.db.base import Base as _Base

    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)

    _registry = FacetRegistry(
        _settings.database_url, encoder, registry_version=_settings.registry_version
    )
    await _registry.load_from_csv(_settings.facets_csv_path)

    gate = FacetGate(
        _registry,
        encoder,
        top_k=_settings.gate_top_k,
        threshold=_settings.gate_threshold,
    )
    scorer = make_scorer(
        _settings.backend,
        vllm_base_url=_settings.vllm_base_url,
        scorer_model=_settings.scorer_model,
    )
    _pipeline = EvaluationPipeline(
        _registry,
        gate,
        scorer,
        factory,
        abstain_tau=_settings.abstain_tau,
        consistency_samples=_settings.self_consistency_samples,
        registry_version=_settings.registry_version,
        calibrator=Calibrator(),
    )

    log.info("kaleido.ready", n_facets=_registry.n_facets(), backend=_settings.backend)
    yield
    await engine.dispose()


app = FastAPI(title="Kaleido", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(
        status="ok",
        n_facets=_registry.n_facets(),
        backend=_settings.backend,
        registry_version=_settings.registry_version,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@app.post("/score", response_model=ScoreResponse)
async def score_conversation(req: ScoreRequest) -> ScoreResponse:
    """Score every turn in a conversation."""
    scores = await _pipeline.evaluate_conversation(req.conversation)
    return ScoreResponse(
        conversation_id=req.conversation.conversation_id,
        scores=scores,
    )


@app.post("/score/turn", response_model=list[FacetScore])
async def score_turn(turn: Turn) -> list[FacetScore]:
    """Score a single turn."""
    return await _pipeline.evaluate_turn(turn)


# ---------------------------------------------------------------------------
# Facet registry
# ---------------------------------------------------------------------------


@app.get("/facets", response_model=list[Facet])
async def list_facets(
    domain: str | None = Query(default=None),
    scope: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> list[Facet]:
    return await _registry.list_facets(domain=domain, scope=scope, page=page, page_size=page_size)


@app.get("/facets/{facet_id}", response_model=Facet)
async def get_facet(facet_id: str) -> Facet:
    try:
        return _registry.get(facet_id)
    except FacetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/facets", response_model=Facet, status_code=201)
async def create_facet(req: FacetCreateRequest) -> Facet:
    """Synthesise or directly insert a new facet."""
    if req.auto_synthesize:
        facet = await synthesize_contract(
            req.facet_name,
            req.domain,
            backend=_settings.backend,
            vllm_base_url=_settings.vllm_base_url,
            scorer_model=_settings.scorer_model,
        )
    else:
        if req.facet is None:
            raise HTTPException(status_code=422, detail="facet required when auto_synthesize=False")
        facet = req.facet

    await _registry.insert_facet(facet)
    return facet


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------


@app.get("/review", response_model=list[dict[str, Any]])
async def list_review_items(resolved: bool = False) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from kaleido.db.base import get_session, make_session_factory
    from kaleido.db.base import make_engine as _make_engine

    engine = _make_engine(_settings.database_url)
    factory = make_session_factory(engine)
    async with get_session(factory) as session:
        stmt = select(ReviewQueueModel).where(ReviewQueueModel.resolved == resolved)
        result = await session.execute(stmt)
        items = result.scalars().all()
        return [
            {
                "id": item.id,
                "score_id": item.score_id,
                "reason": item.reason,
                "resolved": item.resolved,
                "human_score": item.human_score,
            }
            for item in items
        ]


@app.post("/review/{item_id}", response_model=dict[str, Any])
async def resolve_review_item(item_id: int, req: ReviewResolveRequest) -> dict[str, Any]:
    from kaleido.db.base import get_session, make_session_factory
    from kaleido.db.base import make_engine as _make_engine

    engine = _make_engine(_settings.database_url)
    factory = make_session_factory(engine)
    async with get_session(factory) as session:
        item = await session.get(ReviewQueueModel, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Review item {item_id} not found")
        item.resolved = True
        item.human_score = req.human_score
        return {"id": item.id, "resolved": True, "human_score": item.human_score}


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _global_error_handler(request: object, exc: Exception) -> JSONResponse:
    log.exception("api.unhandled_error", exc=str(exc))
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
