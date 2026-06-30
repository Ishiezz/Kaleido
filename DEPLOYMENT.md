# Kaleido — Deployment Guide

Three options below. **Option A** (Render + Streamlit Cloud) is the fastest — free, no credit card, ~10 minutes end-to-end.

---

## Option A: Render (API) + Streamlit Cloud (UI) — Recommended

### Step 1: Deploy the API to Render

**1.1 — Create account**
→ Go to [render.com](https://render.com) → **Get Started for Free** → Sign up with GitHub

**1.2 — New Web Service**
→ Dashboard → **New +** → **Web Service**

**1.3 — Connect repository**
→ Select `Ishiezz/Kaleido` from the list
(If not listed, click "Configure account" and grant access to that repo)

**1.4 — Configure the service**

| Field | Value |
|---|---|
| Name | `kaleido-api` |
| Region | Oregon (US West) |
| Branch | `main` |
| Runtime | `Python 3` |
| Build Command | `pip install -e .` |
| Start Command | `uvicorn kaleido.api:app --host 0.0.0.0 --port $PORT` |
| Instance Type | **Free** |

**1.5 — Set environment variables**

Click **"Advanced"** → **"Add Environment Variable"** → add all of these:

| Key | Value |
|---|---|
| `KALEIDO_BACKEND` | `stub` |
| `KALEIDO_DATABASE_URL` | `sqlite+aiosqlite:///./kaleido.db` |
| `KALEIDO_REGISTRY_VERSION` | `2026.06.0` |
| `KALEIDO_ABSTAIN_TAU` | `0.10` |
| `KALEIDO_GATE_TOP_K` | `64` |

**1.6 — Deploy**
→ Click **"Create Web Service"**
→ Wait 3–5 minutes for the build log to show `Application startup complete`

**1.7 — Copy your API URL**
It looks like: `https://kaleido-api-xxxx.onrender.com`
→ Click the URL at the top of the service page to verify `/healthz` returns `{"status":"ok",...}`

> **Free tier note:** Render spins down free services after 15 minutes of inactivity. The first request after sleep takes ~30 seconds to wake. This is normal.

---

### Step 2: Deploy the UI to Streamlit Community Cloud

**2.1 — Go to Streamlit Cloud**
→ [share.streamlit.io](https://share.streamlit.io) → **Sign in with GitHub**

**2.2 — Create a new app**
→ Click **"Create app"** (top right)

**2.3 — Configure**

| Field | Value |
|---|---|
| Repository | `Ishiezz/Kaleido` |
| Branch | `main` |
| Main file path | `ui/streamlit_app.py` |
| App URL | choose a custom slug (e.g. `kaleido-eval`) |

**2.4 — Add the API secret**
→ Click **"Advanced settings"**
→ In the **Secrets** text box, paste:

```toml
KALEIDO_API_URL = "https://kaleido-api-xxxx.onrender.com"
```

Replace `kaleido-api-xxxx` with your actual Render URL from Step 1.7.

**2.5 — Deploy**
→ Click **"Deploy!"**
→ Wait 2–3 minutes

**2.6 — Your live URL**
```
https://kaleido-eval.streamlit.app
```
(or whatever slug you chose)

---

## Option B: Local Docker (CPU-only, no GPU)

```bash
git clone https://github.com/Ishiezz/Kaleido.git kaleido
cd kaleido
docker-compose -f docker-compose.stub.yml up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000/healthz |
| UI | http://localhost:8501 |

Stop with `Ctrl+C`. Data persists in a named Docker volume.

---

## Option C: Local Python (Development)

```bash
git clone https://github.com/Ishiezz/Kaleido.git kaleido
cd kaleido
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

**Terminal 1 — API:**
```bash
KALEIDO_BACKEND=stub \
KALEIDO_DATABASE_URL=sqlite+aiosqlite:///./kaleido.db \
uvicorn kaleido.api:app --reload
```

**Terminal 2 — UI:**
```bash
KALEIDO_API_URL=http://localhost:8000 \
streamlit run ui/streamlit_app.py
```

→ UI opens at http://localhost:8501

---

## Option D: Full Production Stack (GPU required)

For real Qwen2.5-7B scoring (not stub mode):

```bash
HF_TOKEN=<your-huggingface-token> docker-compose up --build
```

Requires:
- NVIDIA GPU with ≥24 GB VRAM
- NVIDIA Container Toolkit installed
- Hugging Face token with access to `Qwen/Qwen2.5-7B-Instruct`

Services:
- PostgreSQL 16 + pgvector → `:5432`
- vLLM (Qwen2.5-7B) → `:8001`
- Kaleido API → `:8000`
- Streamlit UI → `:8501`

---

## Verifying the deployment

Once the API is live, test it:

```bash
# Health check
curl https://your-api-url.onrender.com/healthz

# Expected:
# {"status":"ok","n_facets":368,"backend":"stub","registry_version":"2026.06.0"}

# Score a conversation
curl -X POST https://your-api-url.onrender.com/score \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": {
      "conversation_id": "test_001",
      "turns": [{
        "turn_id": "t0", "conversation_id": "test_001",
        "index": 0, "role": "user",
        "text": "Hello, how are you today?"
      }]
    }
  }'
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `KALEIDO_BACKEND` | `stub` | `stub` = CPU-only demo; `vllm` = real model |
| `KALEIDO_DATABASE_URL` | postgres URL | SQLite or PostgreSQL async URL |
| `KALEIDO_API_URL` | `http://localhost:8000` | UI → API base URL |
| `KALEIDO_REGISTRY_VERSION` | `2026.06.0` | Facet registry version tag |
| `KALEIDO_ABSTAIN_TAU` | `0.10` | Confidence threshold for abstention |
| `KALEIDO_GATE_TOP_K` | `64` | kNN facets retrieved per turn |
| `KALEIDO_GATE_THRESHOLD` | `0.35` | Cosine similarity floor for activation |
| `KALEIDO_SELF_CONSISTENCY_SAMPLES` | `3` | Stochastic samples for confidence |
| `KALEIDO_VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM OpenAI-compat endpoint |
| `KALEIDO_SCORER_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Model served by vLLM |
