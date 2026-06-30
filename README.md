# Kaleido — Facet-as-Data Conversation Evaluation Engine

<p align="center">
  <b>Score any AI conversation across 368 quality dimensions — on CPU, with no hardcoded logic.</b><br/>
  Add a new quality metric by inserting one database row. No code change. No redeploy.
</p>

---

## The Problem With Every Other Evaluator

Most conversation scorers look like this internally:

```python
if metric == "safety":    return score_safety(turn)
if metric == "fluency":   return score_fluency(turn)
if metric == "empathy":   return score_empathy(turn)
# ... 365 more branches
```

Every new metric is a code change, a PR, a deploy. At 368 metrics, the codebase becomes unmaintainable. At 5,000, it collapses.

**Kaleido doesn't have a single `if metric ==` anywhere.** Every quality metric — called a *facet* — is a database row with a name, definition, and scoring rubric. The pipeline reads those at runtime. Adding "Emotional Warmth" or "Jailbreak Resistance" means one `INSERT`. Nothing else.

| | Typical Evaluator | Kaleido |
|---|---|---|
| Add a new metric | Write + deploy new code | Insert a DB row |
| Score 368 metrics per turn | 368 separate functions | One prompt template |
| Scale to 5,000 metrics | Rewrite the system | Nothing changes |

---

## What It Does

Kaleido scores every conversation turn across **368 distinct facets** — spanning linguistic quality, pragmatics, safety, emotion, and personality — using a strict 5-stage pipeline backed by constrained LLM generation.

```
Turn ──▶ [1] Ingest ──▶ [2] Sparse Gate ──▶ [3] Applicability ──▶ [4] Score ──▶ [5] Calibrate ──▶ FacetScore[]
```

Not all 368 facets score every turn. The gate first retrieves the semantically relevant subset (~15% per turn) via pgvector kNN search, applies hard observability rules, then scores only what applies. The pipeline is async — applicable facets score concurrently via `asyncio.gather`.

Each stage is independently testable. The system handles 368 facets today and scales to **5,000+ with zero code changes** — add a row to the database, and the next request picks it up.

---

## How It Works

**1. Facets are rows, not code.**
Each quality metric lives in a Postgres table with its `facet_id`, `definition`, `score_anchors` (the rubric for −2 through +2), `domain`, and `text_observability`. The scoring code never references any specific facet by name.

**2. The gate scores only what's relevant.**
Scoring a cooking question on "jailbreak resistance" wastes compute and degrades precision. Before any LLM call, a bi-encoder embeds the turn and retrieves the top-64 semantically closest facets via HNSW cosine search. Universal facets (5 total) are always included. Facets requiring external data or explicit keyword presence are excluded by rule. The result: roughly 50–80 candidates per turn instead of 368.

**3. One prompt template. No branching.**
The scorer formats a single template with the facet's definition and rubric pulled from the DB row, sends it to the LLM, and reads back a constrained output. vLLM's `guided_choice=["-2","-1","0","1","2"]` means the model cannot output anything except a valid score — no free-text, no hallucinated values.

```python
# The ONLY prompt template in the entire codebase:
prompt = _SCORE_TEMPLATE.format(
    facet_name  = facet.facet_name,      # from DB row
    definition  = facet.definition,       # from DB row
    anchors     = facet.score_anchors,    # all 5 labels, from DB row
    role        = turn.role,
    turn_text   = turn.text,
    context     = turn.context,
    score_scale = "-2,-1,0,1,2"
)
label, logprobs = vllm.complete(prompt, guided_choice=["-2","-1","0","1","2"])
```

**4. Confidence is a 3-signal blend.**
A single LLM call doesn't tell you how confident the model is. Kaleido fuses three independent signals: logprob margin between the top two candidates, self-consistency across 3 stochastic samples, and ordinal variance across those samples. The blend is then temperature-scaled via scipy NLL minimization. Scores below the abstention threshold route to a human review queue instead of being stored.

```python
confidence = (
    0.5 × logprob_margin          # P(winner) − P(runner-up) / 2
  + 0.3 × self_consistency        # fraction of 3 samples agreeing with MAP label
  + 0.2 × (1 − ordinal_variance)  # stability of sample distribution over {−2..+2}
)
confidence = calibrator.apply(confidence)   # temperature-scaling via scipy NLL minimisation
```

**5. N/A is a gate decision, not a score value.**
When a facet doesn't apply, `applies=False` and no score is written. The DB enforces `applies=False ⟹ score IS NULL` with a `CHECK` constraint. There's no "N/A" 6th value muddying the ordinal scale.

| | |
|---|---|
| Add / remove facets | DB INSERT / DELETE — no code change |
| Score scale | `{−2, −1, 0, +1, +2}` signed ordinal; 0 = neutral |
| LLM | Qwen2.5-7B-Instruct via vLLM (open weights, ≤16B) |
| CPU mode | `KALEIDO_BACKEND=stub` — no model downloads, full API surface works |
| Confidence | Logprob margin + self-consistency + ordinal variance, calibrated |

---

## Architecture

17 modules, ~1,900 lines. Each layer has one job.

```
src/kaleido/
├── schemas.py        # Data shapes: Turn, Facet, FacetScore (Pydantic v2)
├── config.py         # All settings via KALEIDO_* environment variables
├── errors.py         # Typed error hierarchy
├── embedding.py      # Text → vector: BGEEmbedder (GPU) | HashStubEmbedder (CPU/CI)
├── registry.py       # Loads facets from CSV/DB; serves kNN search
├── gating.py         # Picks which facets apply to this turn (~15% of all)
├── scoring.py        # Calls LLM with one fixed prompt; returns {−2..+2}
├── confidence.py     # Fuses 3 signals into calibrated confidence score
├── pipeline.py       # Async orchestrator: runs all 5 stages end-to-end
├── synthesis.py      # LLM-generates a new Facet schema from a name + domain
├── api.py            # FastAPI: 8 REST endpoints
└── db/
    ├── base.py       # Async DB engine + session factory
    ├── models.py     # SQLAlchemy 2.0 ORM (Postgres + SQLite dual dialect)
    └── migrations/   # Alembic: tables, HNSW index, CHECK constraints
```

### Sparse Facet Activation (SFA)

A two-step gate keeps scoring cost sub-linear in facet count:

```
turn.text
    │
    ├─ encode(text) ──▶ kNN search top-64 (cosine, HNSW) ────┐
    └─ universals() ──▶ always-applicable facets              ├─ dedup by facet_id
                                                               │
                                      rule filter ◀───────────┘
                                      ├─ requires_external_data  → applies=False (always)
                                      ├─ not_text_observable     → applies=False (always)
                                      ├─ requires_explicit_mention → token match in turn
                                      └─ observable              → retrieval_score ≥ threshold
```

Step 1 uses BGE-small-en-v1.5 (384-dim) embeddings stored in a numpy matrix; search is a single matrix multiply against the in-memory registry. Step 2 applies hard rules — facets that need external data (stock prices, weather) are always excluded regardless of similarity score.

---

## Quickstart (CPU, no GPU required)

```bash
# Clone and install
git clone https://github.com/Ishiezz/Kaleido.git kaleido && cd kaleido
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Start the API (stub mode — SQLite, no model downloads)
KALEIDO_BACKEND=stub \
KALEIDO_DATABASE_URL=sqlite+aiosqlite:///./kaleido.db \
uvicorn kaleido.api:app --reload

# Score a conversation
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": {
      "conversation_id": "demo_001",
      "turns": [
        {
          "turn_id": "t0", "conversation_id": "demo_001",
          "index": 0, "role": "user",
          "text": "I am furious! This is the worst service I have ever experienced."
        }
      ]
    }
  }'
```

**Live output** (111 facets scored across a 2-turn conversation):
```json
{"status":"ok","n_facets":368,"backend":"stub","registry_version":"2026.06.0"}

conversation_id  : demo_001
total evaluations: 111
applicable       : 111
scored           : 111
abstained        : 0

sample scores:
  K0249    score=+0  conf=0.200  (Harmfulness)
  K0031    score=+0  conf=0.200  (Disrespect)
  K0270    score=+0  conf=0.200  (Social Desirability Bias)
```

---

## Full Stack (Docker)

### CPU-only

```bash
docker-compose -f docker-compose.stub.yml up --build
# API → http://localhost:8000/healthz
# UI  → http://localhost:8501
```

### Production (Postgres + vLLM + GPU)

```bash
HF_TOKEN=<your-huggingface-token> docker-compose up --build
# PostgreSQL + pgvector  → :5432
# vLLM (Qwen2.5-7B)     → :8001
# Kaleido API            → :8000
# Streamlit UI           → :8501
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/healthz` | Status, facet count, backend name |
| `POST` | `/score` | Score full `Conversation` → `ScoreResponse` |
| `POST` | `/score/turn` | Score single `Turn` → `list[FacetScore]` |
| `GET` | `/facets` | Paginated list (`?domain=&scope=&page=&page_size=`) |
| `GET` | `/facets/{id}` | Fetch one facet |
| `POST` | `/facets` | LLM-synthesise + insert a new facet |
| `GET` | `/review` | Low-confidence items pending human review |
| `POST` | `/review/{id}` | Resolve a review item with a human score |

---

## Data Model

### Facet — the core abstraction

```python
class Facet(BaseModel):
    facet_id: str
    facet_name: str
    domain: str
    facet_type: Literal["qualitative_trait", "level_score", "frequency_count", "binary"]
    value_polarity: Literal["positive", "negative", "bipolar", "neutral"]
    text_observability: Literal[
        "observable",
        "requires_explicit_mention",  # gated: token must appear in turn
        "requires_external_data",     # never scored from text alone
        "not_text_observable",        # always excluded
    ]
    applicability_scope: Literal["universal", "conditional", "rare"]
    score_anchors: dict[str, str]   # keys exactly "-2", "-1", "0", "1", "2"
    definition: str
    embedding_text: str             # what the bi-encoder indexes for kNN
```

### FacetScore — pipeline output

```python
class FacetScore(BaseModel):
    facet_id: str
    turn_id: str
    applies: bool              # False = gate ruled this facet out for this turn
    score: int | None          # None iff applies=False or abstained=True
    confidence: float          # ∈ [0, 1], calibrated
    abstained: bool            # True when confidence < abstain_tau
    evidence_span: str | None  # substring of turn.text supporting the score
    model_name: str
    registry_version: str
```

**DB invariants (enforced by CHECK constraints):**
- `applies=False ⟹ score IS NULL`
- `score ∈ {−2, −1, 0, +1, +2}` or `NULL`
- `UNIQUE(turn_id, facet_id, registry_version)`

---

## Facet Coverage

368 scorable facets across 12 domains:

| Domain | Facets | Examples |
|---|---|---|
| `personality_trait` | 120 | Risk-taking, Conscientiousness, Naivety, Assertiveness |
| `linguistic_quality` | 48 | Spelling accuracy, Fluency, Conciseness, Formality |
| `safety` | 34 | Harmfulness, Disrespect, Jailbreak resistance |
| `emotional` | 28 | Anger, Emotional support, Grief, Enthusiasm |
| `pragmatics` | 24 | Relevance, Implicature, Turn coherence |
| `cognitive` | 22 | Logical coherence, Reasoning quality, Uncertainty |
| `social` | 20 | Politeness, Face-saving, Status signalling |
| `spiritual` | 14 | Mindfulness, Existential framing |
| + 4 more | 58 | cultural, meta-cognitive, relational, behavioural |

**By scope:**
- **Universal** (5): scored on every turn, no gate check
- **Conditional** (331): activated by cosine similarity ≥ threshold
- **Rare** (32): only when explicit trigger keyword present in turn

---

## Eval Set — `kaleido_eval_set.zip`

50 conversations across 10 case buckets, each scored through the full pipeline:

| Bucket | Convs | What it tests |
|---|---|---|
| Linguistic quality | 5 | Excellent prose vs. broken grammar vs. multilingual |
| Safety | 5 | Security advice, medical disclaimers, hard refusals |
| Emotional | 5 | Joy, grief, anger, neutral, empathy |
| Spiritual / explicit-mention | 5 | `requires_explicit_mention` gating in action |
| Off-topic / rare | 5 | Sports, geography, trivia, recipes |
| External-data (gate exclusion) | 5 | Stock prices, weather — must not score |
| Multi-turn context | 5 | Coherence tracking, topic shifts, clarifications |
| Adversarial / jailbreak | 5 | Prompt injection, roleplay misuse, social engineering |
| Ambiguous / low-signal | 5 | Single words, noise, contradictory input |
| Technical | 5 | Debugging, system design, algorithm questions |

---

## Development

```bash
# Tests — 102 tests, all layers
pytest tests/ -v

# Type checking — zero errors
mypy --strict src/

# Lint + format
ruff check src/ tests/ --fix && black src/ tests/

# Generate the 50-conversation eval set + zip
KALEIDO_BACKEND=stub KALEIDO_DATABASE_URL=sqlite+aiosqlite:///./eval.db \
  python scripts/generate_eval_set.py

# Seed facet registry (Postgres)
python scripts/seed_registry.py

# Backfill embeddings on existing rows
python scripts/build_embeddings.py
```

### Test Coverage

| Layer | Test File | Tests |
|---|---|---|
| Schemas + validators | `test_schemas.py` | 12 |
| DB models | `test_db_models.py` | 8 |
| Facet registry | `test_registry.py` | 9 |
| Ordinal scorer | `test_scoring.py` | 18 |
| Sparse gate | `test_gating.py` | 11 |
| Confidence fusion | `test_confidence.py` | 15 |
| Pipeline orchestrator | `test_pipeline.py` | 7 |
| FastAPI endpoints | `test_api.py` | 14 |
| Synthesis | `test_synthesis.py` | 13 |
| **Total** | | **102 passing** |

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Web framework | FastAPI + uvicorn |
| ORM | SQLAlchemy 2.0 async |
| Primary DB | PostgreSQL 16 + pgvector (HNSW, cosine) |
| Stub DB | SQLite + aiosqlite |
| Migrations | Alembic |
| Schema validation | Pydantic v2 |
| Configuration | pydantic-settings (`KALEIDO_*` env vars) |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim, sentence-transformers) |
| LLM inference | vLLM OpenAI-compat API (`guided_choice` decoding) |
| Scoring model | `Qwen/Qwen2.5-7B-Instruct` (≤16B, open weights) |
| Calibration | Temperature scaling (scipy NLL minimisation) |
| Type checking | mypy `--strict` — zero errors |
| Linting | ruff + black |
| Testing | pytest + pytest-asyncio |
| UI | Streamlit |
| Containers | Docker + docker-compose |
| Logging | structlog (structured JSON) |

---

## Scaling to 5,000+ Facets

The architecture is facet-count-agnostic by design:

1. **Gate filters ~85%** of facets via kNN — cost is sub-linear in facet count (pgvector HNSW).
2. **One prompt template** — `scoring.py` has exactly one template string. Facet K5000 requires only an `INSERT`.
3. **Async concurrent scoring** — applicable facets scored in parallel via `asyncio.gather` + thread pool.
4. **In-memory registry** — `FacetRegistry` keeps all embeddings in a numpy matrix; search is a single matrix multiply.

---

## Environment Variables

```bash
KALEIDO_BACKEND=stub                          # "stub" | "vllm"
KALEIDO_DATABASE_URL=sqlite+aiosqlite:///...  # or postgresql+asyncpg://...
KALEIDO_VLLM_BASE_URL=http://localhost:8000/v1
KALEIDO_SCORER_MODEL=Qwen/Qwen2.5-7B-Instruct
KALEIDO_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
KALEIDO_GATE_TOP_K=64
KALEIDO_GATE_THRESHOLD=0.35
KALEIDO_ABSTAIN_TAU=0.10
KALEIDO_SELF_CONSISTENCY_SAMPLES=3
KALEIDO_REGISTRY_VERSION=2026.06.0
KALEIDO_FACETS_CSV_PATH=data/processed/facets_enriched.csv
```

## Project Structure

```
Kaleido/
├── src/kaleido/               # Core evaluation engine (17 modules, ~1,900 lines)
│   ├── api.py                 # FastAPI — 8 REST endpoints
│   ├── pipeline.py            # Async orchestrator — runs all 5 pipeline stages
│   ├── gating.py              # Sparse Facet Activation — filters ~85% of facets per turn
│   ├── scoring.py             # LLM scorer — one prompt template, constrained output {−2..+2}
│   ├── confidence.py          # 3-signal confidence fusion + temperature calibration
│   ├── registry.py            # Loads facets from CSV; serves kNN embedding search
│   ├── config.py              # All settings via KALEIDO_* environment variables
│   └── db/                    # SQLAlchemy models, Alembic migrations, async engine
├── tests/                     # Full unit + integration test suite (10 modules)
├── scripts/
│   ├── seed_registry.py       # Loads facets CSV into the database
│   ├── build_embeddings.py    # Pre-computes and backfills facet embeddings
│   └── generate_eval_set.py   # Generates the 50-conversation evaluation dataset
├── ui/
│   └── streamlit_app.py       # Browser UI — score conversations, explore facets
├── data/processed/            # 368 enriched facets with embeddings (CSV)
├── docker-compose.yml         # Full production stack — Postgres + vLLM + API + UI
├── docker-compose.stub.yml    # CPU-only demo stack — no GPU or model downloads needed
├── kaleido_eval_set.zip       # 50 pre-scored conversations (evaluation deliverable)
└── pyproject.toml             # Package metadata and dependency manifest
```
