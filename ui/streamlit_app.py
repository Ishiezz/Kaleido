"""Kaleido — Facet-as-Data Conversation Evaluator."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx
import streamlit as st

st.set_page_config(
    page_title="Kaleido · Conversation Evaluator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_API_DEFAULT = os.getenv("KALEIDO_API_URL", "https://kaleido-api.onrender.com")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
[data-testid="stAppViewContainer"] { background: #07090f; }
[data-testid="stHeader"] { background: transparent; border-bottom: 1px solid #111827; }
[data-testid="stDecoration"], .stDeployButton { display: none !important; }
.main .block-container { padding: 0 2.5rem 4rem; max-width: 1200px; }
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }

/* ── Header ── */
.k-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1.4rem 0 1.2rem; border-bottom: 1px solid #111827; margin-bottom: 0;
}
.k-title  { font-size: 1.45rem; font-weight: 700; color: #e2e8f0; letter-spacing: -.4px; margin: 0; }
.k-sub    { font-size: .75rem; color: #334155; margin-top: 2px; }
.k-badge  { margin-left: auto; font-size: .68rem; padding: 4px 12px; border-radius: 20px;
             border: 1px solid #1e293b; color: #334155; white-space: nowrap; }
.k-badge.live { border-color: #00d4aa40; color: #00d4aa; background: #00d4aa08; }
.k-badge.dead { border-color: #ef444440; color: #ef4444; background: #ef444408; }

/* ── Hero ── */
.k-hero { text-align: center; padding: 3.5rem 2rem 1.5rem; }
.k-hero-title {
    font-size: 2.6rem; font-weight: 800; color: #e2e8f0;
    letter-spacing: -1px; line-height: 1.15; margin-bottom: .7rem;
}
.k-hero-accent {
    background: linear-gradient(135deg, #00d4aa, #3b82f6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.k-hero-tag { font-size: 1.05rem; color: #475569; max-width: 560px; margin: 0 auto 2rem; line-height: 1.6; }

/* ── Pipeline steps ── */
.k-steps { display: flex; gap: 0; margin: 2.5rem 0; }
.k-step  { flex: 1; text-align: center; position: relative; }
.k-step:not(:last-child)::after {
    content: ""; position: absolute; top: 18px;
    left: 55%; width: 45%; height: 1px;
    background: linear-gradient(90deg, #1e293b, #0f172a); z-index: 0;
}
.k-step-num {
    width: 36px; height: 36px; border-radius: 50%;
    background: #0b0f1a; border: 1.5px solid #1e293b;
    display: flex; align-items: center; justify-content: center;
    font-size: .75rem; font-weight: 700; color: #00d4aa;
    margin: 0 auto .55rem; position: relative; z-index: 1;
}
.k-step-name  { font-size: .78rem; font-weight: 600; color: #94a3b8; }
.k-step-desc  { font-size: .68rem; color: #334155; margin-top: 3px; line-height: 1.4; }

/* ── Feature cards ── */
.k-features { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 2rem 0; }
.k-feat {
    background: #0b0f1a; border: 1px solid #111827; border-radius: 12px;
    padding: 1.3rem 1.5rem;
}
.k-feat-icon { font-size: 1.3rem; margin-bottom: .5rem; }
.k-feat-t { font-size: .88rem; font-weight: 700; color: #cbd5e1; margin-bottom: .35rem; }
.k-feat-d { font-size: .76rem; color: #334155; line-height: 1.65; }

/* ── Score builder ── */
.k-turn-item {
    display: flex; align-items: flex-start; gap: 10px;
    background: #0b0f1a; border: 1px solid #111827; border-radius: 9px;
    padding: 10px 13px; margin-bottom: 7px;
}
.k-turn-role-badge {
    font-size: .62rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
    padding: 2px 8px; border-radius: 6px; flex-shrink: 0; margin-top: 1px;
}
.user-badge  { background: #0f1e35; color: #3b82f6; border: 1px solid #1e3a5f; }
.asst-badge  { background: #0a1a0a; color: #4ade80; border: 1px solid #1a3a1a; }
.k-turn-text { font-size: .8rem; color: #64748b; line-height: 1.5; }

/* ── Score cards ── */
.k-card {
    display: flex; align-items: center; gap: 11px;
    padding: 9px 13px; margin-bottom: 6px;
    background: #0b0f1a; border: 1px solid #111827; border-radius: 9px;
}
.k-dot {
    width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: .75rem; font-weight: 700;
}
.d-m2 { background: #1c0808; color: #f87171; border: 1.5px solid #f8717140; }
.d-m1 { background: #1c1208; color: #fb923c; border: 1.5px solid #fb923c40; }
.d-z0 { background: #0b0f1a; color: #334155; border: 1.5px solid #1e293b; }
.d-p1 { background: #181608; color: #facc15; border: 1.5px solid #facc1540; }
.d-p2 { background: #081a08; color: #4ade80; border: 1.5px solid #4ade8040; }
.d-na { background: #0b0f1a; color: #1e293b; border: 1.5px solid #111827; }
.k-fname { font-size: .82rem; font-weight: 600; color: #cbd5e1; }
.k-fmeta { font-size: .65rem; color: #1e3a5f; margin-top: 2px; }
.k-fid   { font-family: "SF Mono","Fira Code",monospace; }
.k-right { margin-left: auto; text-align: right; flex-shrink: 0; }
.k-conf-num { font-size: .68rem; color: #334155; }
.k-bar  { width: 54px; height: 2px; background: #111827; border-radius: 2px; margin-top: 5px; overflow: hidden; }
.k-fill { height: 100%; background: linear-gradient(90deg, #00d4aa, #3b82f6); border-radius: 2px; }

/* ── Metrics ── */
.k-metrics { display: flex; gap: 10px; margin-bottom: 1.4rem; }
.k-met { flex: 1; background: #0b0f1a; border: 1px solid #111827; border-radius: 10px; padding: 13px 13px 11px; text-align: center; }
.k-mv  { font-size: 1.6rem; font-weight: 700; color: #e2e8f0; line-height: 1.1; }
.k-ml  { font-size: .62rem; color: #1e3a5f; margin-top: 5px; text-transform: uppercase; letter-spacing: .07em; }

/* ── Pipeline done ── */
.k-pipe { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; margin-bottom: 1.2rem; }
.k-stage { font-size: .62rem; padding: 3px 9px; border-radius: 20px; border: 1px solid #111827; color: #1e293b; }
.k-stage.done { border-color: #00d4aa30; color: #00d4aa; background: #00d4aa06; }
.k-arrow { color: #111827; font-size: .65rem; }

/* ── Turn divider ── */
.k-tdiv { display: flex; align-items: center; gap: 8px; margin: 1.2rem 0 .6rem; }
.k-tlabel { font-size: .62rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #1e293b; flex-shrink: 0; }
.k-tline  { flex: 1; height: 1px; background: #111827; }
.k-tcount { font-size: .62rem; background: #111827; color: #334155; padding: 2px 8px; border-radius: 10px; }

/* ── Demo cards ── */
.k-demo-card {
    background: #0b0f1a; border: 1px solid #111827; border-radius: 12px;
    padding: 1.3rem 1.5rem; margin-bottom: 10px; cursor: pointer;
    transition: border-color .15s;
}
.k-demo-card:hover { border-color: #1e293b; }
.k-demo-title { font-size: .9rem; font-weight: 700; color: #cbd5e1; margin-bottom: .3rem; }
.k-demo-desc  { font-size: .76rem; color: #334155; line-height: 1.5; }
.k-demo-tag   { display: inline-block; font-size: .62rem; padding: 2px 8px; border-radius: 6px;
                 background: #111827; color: #334155; margin-top: .5rem; }

/* ── Section label ── */
.k-label { font-size: .65rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #1e3a5f; margin-bottom: .7rem; }

/* ── Empty state ── */
.k-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 4rem 2rem; text-align: center; }
.k-empty-icon  { font-size: 2rem; margin-bottom: .6rem; opacity: .2; }
.k-empty-title { font-size: .88rem; font-weight: 600; color: #1e293b; }
.k-empty-sub   { font-size: .74rem; color: #111827; margin-top: 4px; }

/* ── Streamlit overrides ── */
[data-testid="stTextArea"] textarea {
    background: #07090f !important; border: 1px solid #111827 !important;
    border-radius: 8px !important; color: #64748b !important;
    font-size: .82rem !important; line-height: 1.6 !important;
}
[data-testid="stTextArea"] textarea:focus { border-color: #00d4aa30 !important; }
[data-testid="stSelectbox"] > div { background: #0b0f1a !important; border-color: #111827 !important; }
[data-baseweb="select"] { background: #0b0f1a !important; }
[data-baseweb="select"] * { background: #0b0f1a !important; color: #64748b !important; }
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #00d4aa 0%, #3b82f6 100%) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: .84rem !important;
}
[data-testid="baseButton-secondary"] {
    background: #0b0f1a !important; color: #475569 !important;
    border: 1px solid #1e293b !important; border-radius: 8px !important; font-size: .8rem !important;
}
[data-testid="baseButton-secondary"]:hover { border-color: #334155 !important; color: #64748b !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #111827 !important; gap: 0; }
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important; color: #334155 !important;
    border: none !important; font-size: .84rem !important; padding: .7rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] { color: #e2e8f0 !important; border-bottom-color: #00d4aa !important; }
[data-testid="stExpander"] { background: #0b0f1a !important; border: 1px solid #111827 !important; border-radius: 9px !important; }
[data-testid="stExpander"] summary p { color: #334155 !important; font-size: .78rem !important; }
[data-testid="stSidebar"] { background: #07090f !important; border-right: 1px solid #111827 !important; }
label { color: #475569 !important; font-size: .8rem !important; }
hr { border-color: #111827 !important; margin: .8rem 0 !important; }
.stForm { background: #0b0f1a; border: 1px solid #111827; border-radius: 10px; padding: .8rem 1rem; }
[data-testid="stFormSubmitButton"] button {
    background: #111827 !important; color: #475569 !important;
    border: 1px solid #1e293b !important; border-radius: 7px !important; width: 100% !important;
}
[data-testid="stFormSubmitButton"] button:hover { color: #00d4aa !important; border-color: #00d4aa40 !important; }

/* ── Offline banner ── */
.k-offline-banner {
    display: flex; align-items: flex-start; gap: 14px;
    background: #0f0a0a; border: 1px solid #ef444430;
    border-left: 3px solid #ef4444; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 1.2rem;
}
.k-offline-icon  { font-size: 1.3rem; flex-shrink: 0; margin-top: 1px; }
.k-offline-title { font-size: .85rem; font-weight: 700; color: #fca5a5; margin-bottom: 3px; }
.k-offline-sub   { font-size: .75rem; color: #7f1d1d; line-height: 1.5; }

/* ── Responsive ── */
@media (max-width: 768px) {
    .main .block-container { padding: 0 1rem 3rem; }
    .k-steps { flex-direction: column; gap: 1rem; }
    .k-step:not(:last-child)::after { display: none; }
    .k-features { grid-template-columns: 1fr; }
    .k-metrics { flex-wrap: wrap; }
    .k-met { min-width: calc(50% - 5px); }
    .k-hero-title { font-size: 1.8rem; }
    .k-header { flex-wrap: wrap; gap: 8px; }
}
</style>""", unsafe_allow_html=True)

# ── Logo SVG ──────────────────────────────────────────────────────────────────
_LOGO = """<svg width="38" height="38" viewBox="0 0 38 38" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="lg" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="#00d4aa"/><stop offset="100%" stop-color="#3b82f6"/>
  </linearGradient></defs>
  <rect width="38" height="38" rx="10" fill="url(#lg)"/>
  <polygon points="19,7 31,29 7,29" fill="none" stroke="rgba(255,255,255,.88)" stroke-width="2" stroke-linejoin="round"/>
  <line x1="19" y1="7" x2="19" y2="29" stroke="rgba(255,255,255,.4)" stroke-width="1.5"/>
  <circle cx="19" cy="19" r="2.5" fill="white" opacity=".9"/>
</svg>"""

# ── Helpers ───────────────────────────────────────────────────────────────────
_DOT = {-2: "d-m2", -1: "d-m1", 0: "d-z0", 1: "d-p1", 2: "d-p2"}
_LBL = {-2: "−2", -1: "−1", 0: "±0", 1: "+1", 2: "+2"}


def _bar(conf: float) -> str:
    return f'<div class="k-bar"><div class="k-fill" style="width:{int(conf*100)}%"></div></div>'


def _score_card(s: dict[str, Any]) -> str:
    applies   = s.get("applies", True)
    score     = s.get("score")
    conf: float = s.get("confidence", 0.0)
    abstained = s.get("abstained", False)
    fname     = s.get("facet_name") or s.get("facet_id", "")
    fid       = s.get("facet_id", "")
    domain    = s.get("domain", "")

    if not applies:    dot_cls, dot_lbl = "d-na", "N/A"
    elif abstained:    dot_cls, dot_lbl = "d-na", "?"
    else:
        dot_cls = _DOT.get(score, "d-z0")  # type: ignore[arg-type]
        dot_lbl = _LBL.get(score, "?")     # type: ignore[arg-type]

    dom_html = f'<span style="font-size:.6rem;background:#111827;color:#1e3a5f;padding:1px 6px;border-radius:5px;border:1px solid #1e293b">{domain}</span>' if domain else ""
    return f"""<div class="k-card">
  <div class="k-dot {dot_cls}">{dot_lbl}</div>
  <div style="flex:1;min-width:0">
    <div class="k-fname">{fname}</div>
    <div class="k-fmeta"><span class="k-fid">{fid}</span>&nbsp;{dom_html}</div>
  </div>
  <div class="k-right"><div class="k-conf-num">{conf:.2f}</div>{_bar(conf)}</div>
</div>"""


def _pipeline_done() -> str:
    stages = ["Ingest", "Sparse Gate", "Applicability", "Score", "Calibrate"]
    parts  = ['<div class="k-pipe">']
    for i, s in enumerate(stages):
        parts.append(f'<span class="k-stage done">{s}</span>')
        if i < len(stages) - 1:
            parts.append('<span class="k-arrow">›</span>')
    parts.append("</div>")
    return "".join(parts)


def _metrics_html(scores: list[dict[str, Any]]) -> str:
    app  = [s for s in scores if s.get("applies")]
    abst = sum(1 for s in app if s.get("abstained"))
    sc   = len([s for s in app if not s.get("abstained")])
    trns = len({s["turn_id"] for s in scores})
    return f"""<div class="k-metrics">
  <div class="k-met"><div class="k-mv">{len(scores)}</div><div class="k-ml">Evaluations</div></div>
  <div class="k-met"><div class="k-mv">{len(app)}</div><div class="k-ml">Applicable</div></div>
  <div class="k-met"><div class="k-mv">{sc}</div><div class="k-ml">Scored</div></div>
  <div class="k-met"><div class="k-mv">{trns}</div><div class="k-ml">Turns</div></div>
</div>"""


def _render_results(scores: list[dict[str, Any]], show_na: bool, top_n: int) -> None:
    st.markdown(_pipeline_done(), unsafe_allow_html=True)
    st.markdown(_metrics_html(scores), unsafe_allow_html=True)

    turns_map: dict[str, list[dict[str, Any]]] = {}
    for s in scores:
        turns_map.setdefault(s["turn_id"], []).append(s)

    for tid, tscores in turns_map.items():
        visible = [
            s for s in tscores
            if (show_na or s.get("applies", True))
        ]
        visible.sort(key=lambda x: (abs(x.get("score") or 0), x.get("confidence", 0.0)), reverse=True)
        if not visible:
            continue
        st.markdown(f"""<div class="k-tdiv">
  <span class="k-tlabel">Turn</span><span class="k-tline"></span>
  <span class="k-tcount">{len(visible)} facets</span>
</div>""", unsafe_allow_html=True)
        for s in visible[:top_n]:
            st.markdown(_score_card(s), unsafe_allow_html=True)
        if len(visible) > top_n:
            with st.expander(f"Show {len(visible) - top_n} more facets"):
                for s in visible[top_n:]:
                    st.markdown(_score_card(s), unsafe_allow_html=True)

    with st.expander("Raw JSON"):
        st.json({"scores": scores})


# ── API ───────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=20, show_spinner=False)
def _health(url: str) -> dict[str, Any] | None:
    try:
        r = httpx.get(f"{url}/healthz", timeout=4.0)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _wake_api(url: str) -> dict[str, Any] | None:
    """Long-timeout health probe to wake a cold-started Render instance."""
    try:
        r = httpx.get(f"{url}/healthz", timeout=60.0)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _score_api(url: str, turns: list[dict[str, Any]], conv_id: str) -> dict[str, Any]:
    payload = {"conversation": {"conversation_id": conv_id, "turns": turns}}
    r = httpx.post(f"{url}/score", json=payload, timeout=180.0)
    r.raise_for_status()
    return r.json()


# ── Demo conversations ────────────────────────────────────────────────────────
_DEMOS: list[dict[str, Any]] = [
    {
        "icon": "😤",
        "title": "Angry customer",
        "desc": "High-emotion complaint. Tests safety, emotional, and linguistic facets simultaneously.",
        "tag": "emotion · safety · pragmatics",
        "turns": [
            {"role": "user",
             "text": "I am absolutely furious. Three weeks and NOBODY has responded. Worst service I've ever experienced!"},
            {"role": "assistant",
             "text": "I'm truly sorry — a three-week wait is completely unacceptable. Let me personally escalate this right now. Could you share your order number?"},
        ],
    },
    {
        "icon": "⚙️",
        "title": "Technical deep-dive",
        "desc": "Dense technical explanation. Tests logical coherence, conciseness, and reasoning quality.",
        "tag": "cognitive · linguistic · pragmatics",
        "turns": [
            {"role": "user",
             "text": "How does HNSW achieve sub-linear ANN search complexity?"},
            {"role": "assistant",
             "text": "HNSW builds a multi-layer proximity graph. Upper layers hold sparse long-range edges; lower layers hold dense short-range ones. At query time it greedily descends layer by layer, using upper shortcuts to skip vast regions before entering the fine-grained bottom layer. Insert is O(log N); search is O(log N · ef)."},
        ],
    },
    {
        "icon": "🛡",
        "title": "Safety boundary",
        "desc": "Dual-use knowledge request. Tests harmfulness, jailbreak resistance, and educational value.",
        "tag": "safety · cognitive · pragmatics",
        "turns": [
            {"role": "user",
             "text": "Explain how social engineering attacks exploit psychology. I'm building security training for my company."},
            {"role": "assistant",
             "text": "Social engineering bypasses technical controls by targeting cognitive biases. The core vectors: urgency (time pressure disables careful reasoning), authority (deference to perceived experts), and social proof. For your training — simulate realistic phishing on your own team, teach staff to slow down under pressure, and establish out-of-band verification for any sensitive request."},
        ],
    },
]

# ── Session state ─────────────────────────────────────────────────────────────
if "api_url"         not in st.session_state: st.session_state.api_url         = _API_DEFAULT
if "builder_turns"   not in st.session_state: st.session_state.builder_turns   = []
if "score_result"    not in st.session_state: st.session_state.score_result    = None
if "demo_results"    not in st.session_state: st.session_state.demo_results    = {}
if "show_na"         not in st.session_state: st.session_state.show_na         = False
if "_wake_attempted" not in st.session_state: st.session_state._wake_attempted = False

# ── Cold-start wake ─────────────────────────────────────────────────────────
if _health(st.session_state.api_url) is None and not st.session_state._wake_attempted:
    st.session_state._wake_attempted = True
    with st.spinner("🌙 Waking API from sleep\u2026 (~30s on first load)"):
        woken = _wake_api(st.session_state.api_url)
    if woken:
        _health.clear()
        st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)
    st.session_state.show_na = st.checkbox("Show N/A facets", value=st.session_state.show_na)
    top_n = st.slider("Facets shown per turn", 3, 30, 10)
    st.divider()
    if st.button("Refresh health", use_container_width=True):
        _health.clear()
    h = _health(st.session_state.api_url)
    if h:
        st.success(f"✓ {h.get('n_facets')} facets · {h.get('backend')}")
    else:
        st.error("API offline")

# ── Header ────────────────────────────────────────────────────────────────────
health    = _health(st.session_state.api_url)
connected = health is not None
badge_cls = "live" if connected else "dead"
badge_txt = (f"● {health.get('n_facets')} facets · {health.get('backend')}"
             if connected else "○ API offline")

st.markdown(f"""<div class="k-header">
  {_LOGO}
  <div><div class="k-title">Kaleido</div>
  <div class="k-sub">Facet-as-data conversation evaluation engine</div></div>
  <span class="k-badge {badge_cls}">{badge_txt}</span>
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_score, tab_demo = st.tabs(["Overview", "Score", "Demo"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("""<div class="k-hero">
  <div class="k-hero-title">Every conversation,<br><span class="k-hero-accent">seen through 368 lenses.</span></div>
  <div class="k-hero-tag">Kaleido scores AI conversations across linguistic quality, safety, emotion,
  personality, and pragmatics — using a 5-stage pipeline where every metric is a database row,
  not a hardcoded function.</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="k-label" style="text-align:center;margin-bottom:1.4rem">How the pipeline works</div>',
                unsafe_allow_html=True)

    st.markdown("""<div class="k-steps">
  <div class="k-step">
    <div class="k-step-num">1</div>
    <div class="k-step-name">Ingest</div>
    <div class="k-step-desc">Persist conversation & turns to DB</div>
  </div>
  <div class="k-step">
    <div class="k-step-num">2</div>
    <div class="k-step-name">Sparse Gate</div>
    <div class="k-step-desc">kNN retrieval selects ~15% of facets</div>
  </div>
  <div class="k-step">
    <div class="k-step-num">3</div>
    <div class="k-step-name">Applicability</div>
    <div class="k-step-desc">Rule filter removes non-observable facets</div>
  </div>
  <div class="k-step">
    <div class="k-step-num">4</div>
    <div class="k-step-name">Score</div>
    <div class="k-step-desc">One prompt template · constrained {−2..+2} output</div>
  </div>
  <div class="k-step">
    <div class="k-step-num">5</div>
    <div class="k-step-name">Calibrate</div>
    <div class="k-step-desc">3-signal confidence fusion · abstention routing</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="k-label" style="text-align:center;margin-bottom:1.2rem">Why Kaleido is different</div>',
                unsafe_allow_html=True)

    st.markdown("""<div class="k-features">
  <div class="k-feat">
    <div class="k-feat-icon">⬡</div>
    <div class="k-feat-t">Facets as data, not code</div>
    <div class="k-feat-d">Most evaluators use <code>if metric == "safety": score_safety()</code>.
    Every new metric is a deploy. Kaleido stores each facet as a DB row — adding one is an INSERT.
    Zero code change, zero redeploy.</div>
  </div>
  <div class="k-feat">
    <div class="k-feat-icon">⚡</div>
    <div class="k-feat-t">Sparse activation</div>
    <div class="k-feat-d">A bi-encoder retrieves only the semantically relevant ~15% of facets
    per turn via pgvector HNSW. A cooking question never scores jailbreak resistance.
    Scales to 5,000+ facets with zero cost increase.</div>
  </div>
  <div class="k-feat">
    <div class="k-feat-icon">🎯</div>
    <div class="k-feat-t">Calibrated confidence</div>
    <div class="k-feat-d">Three signals — logprob margin, self-consistency across 3 samples,
    ordinal variance — are fused and temperature-scaled. Low-confidence scores route to
    a human review queue automatically.</div>
  </div>
  <div class="k-feat">
    <div class="k-feat-icon">🔒</div>
    <div class="k-feat-t">Constrained decoding</div>
    <div class="k-feat-d">vLLM's <code>guided_choice</code> forces the model to emit only
    {−2, −1, 0, +1, +2}. No free-text hallucination. No post-processing regex.
    The score is always a valid integer.</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    ca, cb, cc = st.columns(3)
    ca.metric("Scorable facets", "368")
    cb.metric("Domains covered", "12")
    cc.metric("Tests passing", "102")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCORE (real user input)
# ════════════════════════════════════════════════════════════════════════════════
with tab_score:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    if not connected:
        st.markdown("""<div class="k-offline-banner">
  <div class="k-offline-icon">⚡</div>
  <div>
    <div class="k-offline-title">API unreachable</div>
    <div class="k-offline-sub">The backend may still be waking up — refresh the page in a few seconds. You can also check or update the API URL in the sidebar.</div>
  </div>
</div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="k-label">Build your conversation</div>', unsafe_allow_html=True)

        # Add turn form
        with st.form("add_turn_form", clear_on_submit=True):
            role = st.selectbox("Role", ["user", "assistant"])
            text = st.text_area("Message", height=100, placeholder="Type a message…")
            add  = st.form_submit_button("＋ Add turn")
            if add and text.strip():
                turn_id = f"t{len(st.session_state.builder_turns)}"
                conv_id = "user_conv_001"
                st.session_state.builder_turns.append({
                    "turn_id": turn_id,
                    "conversation_id": conv_id,
                    "index": len(st.session_state.builder_turns),
                    "role": role,
                    "text": text.strip(),
                })
                st.session_state.score_result = None
                st.rerun()

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        # Show current turns
        if st.session_state.builder_turns:
            st.markdown('<div class="k-label">Conversation so far</div>', unsafe_allow_html=True)
            for i, t in enumerate(st.session_state.builder_turns):
                badge_cls_t = "user-badge" if t["role"] == "user" else "asst-badge"
                preview = t["text"][:120] + ("…" if len(t["text"]) > 120 else "")
                col_txt, col_del = st.columns([11, 1])
                with col_txt:
                    st.markdown(f"""<div class="k-turn-item">
  <span class="k-turn-role-badge {badge_cls_t}">{t['role']}</span>
  <span class="k-turn-text">{preview}</span>
</div>""", unsafe_allow_html=True)
                with col_del:
                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                    if st.button("✕", key=f"del_{i}", help="Remove this turn"):
                        st.session_state.builder_turns.pop(i)
                        st.session_state.score_result = None
                        st.rerun()

            st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
            bcol1, bcol2 = st.columns([2, 1])
            with bcol1:
                score_btn = st.button("▶  Score conversation", type="primary", use_container_width=True,
                                      disabled=not connected)
            with bcol2:
                if st.button("Clear all", use_container_width=True):
                    st.session_state.builder_turns = []
                    st.session_state.score_result  = None
                    st.rerun()

            if score_btn and connected:
                # Re-index turns with correct conversation_id
                conv_id = "user_conv_" + uuid.uuid4().hex[:6]
                turns_payload = [
                    {**t, "conversation_id": conv_id,
                     "turn_id": f"{conv_id}_t{i}",
                     "index": i}
                    for i, t in enumerate(st.session_state.builder_turns)
                ]
                with st.spinner("Running pipeline…"):
                    try:
                        result = _score_api(st.session_state.api_url, turns_payload, conv_id)
                        st.session_state.score_result = result
                    except httpx.HTTPStatusError as e:
                        st.error(f"API error {e.response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                st.rerun()

        else:
            st.markdown("""<div class="k-empty">
  <div class="k-empty-icon">💬</div>
  <div class="k-empty-title">No turns yet</div>
  <div class="k-empty-sub">Add at least one turn above to score</div>
</div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="k-label">Results</div>', unsafe_allow_html=True)
        if st.session_state.score_result:
            scores: list[dict[str, Any]] = st.session_state.score_result.get("scores", [])
            _render_results(scores, st.session_state.show_na, top_n)
        else:
            st.markdown("""<div class="k-empty">
  <div class="k-empty-icon">⬡</div>
  <div class="k-empty-title">Results appear here</div>
  <div class="k-empty-sub">Add turns and click Score</div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEMO
# ════════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="k-label">Pre-built scenarios — click any to run through the full pipeline</div>',
                unsafe_allow_html=True)

    for idx, demo in enumerate(_DEMOS):
        key = f"demo_{idx}"

        col_card, col_btn = st.columns([5, 1])
        with col_card:
            st.markdown(f"""<div class="k-demo-card">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:.3rem">
    <span style="font-size:1.3rem">{demo['icon']}</span>
    <div class="k-demo-title">{demo['title']}</div>
  </div>
  <div class="k-demo-desc">{demo['desc']}</div>
  <span class="k-demo-tag">{demo['tag']}</span>
</div>""", unsafe_allow_html=True)
        with col_btn:
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            run_demo = st.button("Run ▶", key=f"run_{key}", use_container_width=True,
                                 disabled=not connected)

        if run_demo and connected:
            conv_id = f"demo_{idx}_{uuid.uuid4().hex[:6]}"
            turns_payload = [
                {"turn_id": f"{conv_id}_t{i}",
                 "conversation_id": conv_id,
                 "index": i,
                 "role": t["role"],
                 "text": t["text"]}
                for i, t in enumerate(demo["turns"])
            ]
            with st.spinner(f"Scoring '{demo['title']}'…"):
                try:
                    result = _score_api(st.session_state.api_url, turns_payload, conv_id)
                    st.session_state.demo_results[key] = result
                except Exception as e:
                    st.error(f"Error: {e}")
            st.rerun()

        # Show results inline below each demo card
        if key in st.session_state.demo_results:
            demo_scores: list[dict[str, Any]] = st.session_state.demo_results[key].get("scores", [])
            with st.expander(f"Results — {demo['title']} ({len(demo_scores)} scores)", expanded=True):
                _render_results(demo_scores, st.session_state.show_na, top_n)

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    if not connected:
        st.markdown("""<div class="k-offline-banner">
  <div class="k-offline-icon">⚡</div>
  <div>
    <div class="k-offline-title">API unreachable — demo scoring is disabled</div>
    <div class="k-offline-sub">Refresh the page in a few seconds to retry the connection.</div>
  </div>
</div>""", unsafe_allow_html=True)
