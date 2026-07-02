"""Streamlit demo: an AI agent that does data science over the dbt-modeled healthcare warehouse.

Run:  streamlit run app.py     (needs OPENAI_API_KEY in agent/.env, or Streamlit Cloud secrets)

Design: "clinical data terminal" — deep-ink dark, clinical-teal accent, Fraunces display +
IBM Plex Sans/Mono, custom guardrail badges. Refined, recruiter-facing.
"""
from __future__ import annotations

import datetime as _dt
import html
import json
import os
from pathlib import Path

import streamlit as st

_FEEDBACK_LOG = Path(__file__).resolve().parent / "logs" / "feedback.jsonl"

# Bridge Streamlit Cloud secrets -> env vars BEFORE importing the agent (llm.py reads env at import).
try:
    for _k in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets.toml locally — falls back to agent/.env

from agent.agent import run_analysis
from agent.charts import build_chart, kpi_cards
from agent.llm import MODEL
from agent.retrieval import load_catalog

st.set_page_config(page_title="Clinical Insight Agent", page_icon="🩺",
                   layout="wide", initial_sidebar_state="collapsed")

EXAMPLES = [
    "Which chronic conditions drive the highest total encounter cost?",
    "What is the 30-day readmission rate, and does it vary by age group?",
    "What is the prevalence of hypertension by age group?",
    "How does average encounter cost differ by encounter class?",
    "What predicts 30-day readmission, adjusting for age and sex?",
]

# ───────────────────────────── design system ─────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:#0a0e13; --surface:#131c27; --surface-2:#0f1620;
  --border:#20303f; --border-soft:#1a2531;
  --text:#e8eef4; --muted:#8ea0b0; --faint:#5b6b7a;
  --accent:#4fd1c5; --accent-2:#8ab4f8;
  --caution:#f87171; --warn:#f5c451; --info:#4fd1c5;
  --font-display:'Fraunces',Georgia,serif;
  --font-body:'IBM Plex Sans',system-ui,sans-serif;
  --font-mono:'IBM Plex Mono',ui-monospace,monospace;
}

.stApp {
  background:
    radial-gradient(1100px 620px at 10% -10%, rgba(79,209,197,.10), transparent 60%),
    radial-gradient(900px 520px at 100% -5%, rgba(138,180,248,.06), transparent 55%),
    var(--bg);
  color: var(--text);
  font-family: var(--font-body);
}
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer { visibility:hidden; }
.block-container{ max-width:1020px; padding-top:1.4rem; padding-bottom:4rem; }

/* hero */
.hero{ padding:2rem 0 1.3rem; border-bottom:1px solid var(--border-soft); }
.hero-eyebrow{ font-family:var(--font-mono); font-size:.72rem; letter-spacing:.28em;
  color:var(--muted); text-transform:uppercase; }
.hero-title{ font-family:var(--font-display); font-weight:600; font-size:3.5rem; line-height:1.02;
  letter-spacing:-.02em; margin:.55rem 0 0; color:var(--text); }
.hero-title .accent{ color:var(--accent); font-style:italic; }
.hero-sub{ color:var(--muted); font-size:1.05rem; max-width:64ch; line-height:1.62; margin:.95rem 0 0; }
.pill-row{ display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.2rem; }
.pill{ font-family:var(--font-mono); font-size:.72rem; color:var(--accent);
  background:rgba(79,209,197,.08); border:1px solid rgba(79,209,197,.28);
  border-radius:999px; padding:.26rem .72rem; }
.meta{ font-family:var(--font-mono); font-size:.73rem; color:var(--faint); margin-top:1rem;
  display:flex; gap:.5rem; align-items:center; }
.meta .dot{ color:var(--border); }

/* eyebrow section labels */
.eyebrow{ font-family:var(--font-mono); font-size:.72rem; letter-spacing:.2em;
  text-transform:uppercase; color:var(--accent); margin:1.7rem 0 .6rem; }
.eyebrow .n{ color:var(--faint); }

/* cards */
.card{ background:var(--surface); border:1px solid var(--border-soft); border-radius:14px;
  padding:1rem 1.15rem; color:var(--text); }
.card.hypo{ font-size:1.06rem; line-height:1.55; border-left:3px solid var(--accent); }
.card.hypo .plan{ color:var(--muted); font-size:.9rem; margin-top:.55rem; line-height:1.5; }

/* guardrail badges */
.gr-list{ display:flex; flex-direction:column; gap:.5rem; }
.gr-badge{ display:flex; align-items:flex-start; gap:.7rem; background:var(--surface);
  border:1px solid var(--border-soft); border-left:3px solid var(--c); border-radius:11px;
  padding:.62rem .85rem; }
.gr-tag{ font-family:var(--font-mono); font-size:.63rem; letter-spacing:.1em; color:var(--c);
  border:1px solid var(--c); border-radius:6px; padding:.12rem .42rem; flex:none; margin-top:.12rem; }
.gr-kind{ font-family:var(--font-mono); font-size:.8rem; color:var(--muted); flex:none; margin-top:.16rem; }
.gr-msg{ color:var(--text); font-size:.9rem; line-height:1.5; }
.sev-caution{ --c:var(--caution);} .sev-warn{ --c:var(--warn);} .sev-info{ --c:var(--info);}
.heal-note{ font-family:var(--font-mono); font-size:.78rem; color:var(--warn); margin:.2rem 0 .5rem; }
.cite{ font-family:var(--font-mono); font-size:.72rem; color:var(--faint); margin:.5rem 0 0;
  display:flex; gap:.4rem; align-items:center; flex-wrap:wrap; }
.cite .pill{ font-size:.68rem; }
.trace{ font-family:var(--font-mono); font-size:.72rem; color:var(--faint); margin-top:1.4rem;
  padding-top:.8rem; border-top:1px solid var(--border-soft); }
.kpi-row{ display:flex; gap:.7rem; flex-wrap:wrap; margin:.3rem 0 1.1rem; }
.kpi-card{ flex:1 1 140px; background:var(--surface); border:1px solid var(--border-soft);
  border-radius:14px; padding:.85rem 1rem; border-top:2px solid var(--accent); }
.kpi-label{ font-family:var(--font-mono); font-size:.66rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); }
.kpi-value{ font-family:var(--font-display); font-size:1.85rem; font-weight:600; color:var(--text);
  line-height:1.08; margin-top:.2rem; }
.kpi-sub{ font-family:var(--font-mono); font-size:.74rem; color:var(--accent); margin-top:.15rem; }

/* buttons — chips + primary CTA */
.stButton > button{ font-family:var(--font-mono); font-size:.78rem; font-weight:400;
  background:var(--surface); color:var(--muted); border:1px solid var(--border-soft);
  border-radius:10px; padding:.5rem .7rem; transition:.15s; text-align:left; line-height:1.3; }
.stButton > button:hover{ border-color:var(--accent); color:var(--text);
  background:rgba(79,209,197,.05); }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--accent); color:#04231f; border:none; font-weight:600; font-family:var(--font-body);
  border-radius:10px; padding:.6rem 1.6rem; letter-spacing:.01em; text-align:center; }
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{
  filter:brightness(1.08); box-shadow:0 8px 26px rgba(79,209,197,.28); }

/* input */
[data-testid="stTextInput"] input{ background:var(--surface); border:1px solid var(--border);
  color:var(--text); border-radius:11px; font-family:var(--font-body); padding:.7rem .9rem; }
[data-testid="stTextInput"] input:focus{ border-color:var(--accent);
  box-shadow:0 0 0 3px rgba(79,209,197,.14); }
[data-testid="stWidgetLabel"]{ display:none; }

/* code / SQL */
[data-testid="stCode"], pre{ background:var(--surface-2) !important;
  border:1px solid var(--border-soft); border-radius:12px; }
code, pre, kbd{ font-family:var(--font-mono) !important; }

/* dataframe */
[data-testid="stDataFrame"]{ border:1px solid var(--border-soft); border-radius:12px; overflow:hidden; }

/* interpretation markdown */
[data-testid="stMarkdownContainer"] p{ line-height:1.7; }
[data-testid="stMarkdownContainer"] strong{ color:var(--accent); font-weight:600; }

/* footer */
.footer{ margin-top:3rem; padding-top:1.2rem; border-top:1px solid var(--border-soft);
  color:var(--faint); font-family:var(--font-mono); font-size:.75rem;
  display:flex; gap:.6rem; flex-wrap:wrap; align-items:center; }
.footer a{ color:var(--muted); text-decoration:none; }
.footer a:hover{ color:var(--accent); }
.footer .dot{ color:var(--border); }
.review-gate{ background:rgba(248,113,113,.10); border:1px solid rgba(248,113,113,.4);
  border-left:3px solid var(--caution); border-radius:11px; padding:.7rem 1rem; margin:.6rem 0 1rem;
  color:var(--text); font-size:.92rem; }
.hitl-note{ font-family:var(--font-mono); font-size:.72rem; color:var(--faint); margin-top:.4rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _data_dictionary() -> str:
    """Human-readable 'what data is available' from the semantic catalog — so users know the scope."""
    tables = load_catalog()["tables"]
    groups = {"Dimensions — who / what": "dim_", "Facts — events": "fct_", "Analytics — question-shaped": "mart_"}
    lines = []
    for label, prefix in groups.items():
        items = [t for t in tables if t["name"].startswith(prefix)]
        if not items:
            continue
        lines.append(f"**{label}**")
        for t in items:
            lines.append(f"- `{t['name']}` — {t['description'].split('.')[0][:110]}")
        lines.append("")
    lines.append("**You can ask for:** counts, rates, costs, averages, prevalence, comparisons by "
                 "demographic (age / sex / race), or filter by a condition / medication name. "
                 "Vague asks (e.g. *'show me the trends'*) get a clarifying question back.")
    return "\n".join(lines)


# ───────────────────────────── hero ─────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">Healthcare analytics · agentic AI</div>
  <h1 class="hero-title">Clinical Insight <span class="accent">Agent</span></h1>
  <p class="hero-sub">Ask in plain English. The agent retrieves schema &amp; metric context, writes
  and self-corrects SQL over a dbt-modeled warehouse, interprets the result — then flags the
  statistical caveats a text-to-SQL bot misses.</p>
  <div class="pill-row">
    <span class="pill">dbt star schema</span>
    <span class="pill">RAG semantic catalog</span>
    <span class="pill">self-healing SQL</span>
    <span class="pill">Wilson CIs · small-N</span>
    <span class="pill">read-only</span>
  </div>
  <div class="meta">model {html.escape(MODEL)} <span class="dot">·</span> DuckDB (Synthea, synthetic)
    <span class="dot">·</span> read-only, row-capped</div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────── ask ─────────────────────────────
if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0]

with st.expander("📋 What data can I ask about?"):
    st.markdown(_data_dictionary())

st.markdown("<div class='eyebrow'>Try one</div>", unsafe_allow_html=True)
for start in range(0, len(EXAMPLES), 3):
    cols = st.columns(3)
    for c, ex in zip(cols, EXAMPLES[start:start + 3]):
        if c.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.question = ex

st.markdown("<div class='eyebrow'>Your question</div>", unsafe_allow_html=True)
question = st.text_input("q", value=st.session_state.question, label_visibility="collapsed")
go = st.button("Run analysis  →", type="primary")


def badge(f) -> str:
    tag = {"caution": "CAUTION", "warn": "WARN", "info": "NOTE"}.get(f.severity, "NOTE")
    return (f"<div class='gr-badge sev-{f.severity}'>"
            f"<span class='gr-tag'>{tag}</span>"
            f"<span class='gr-kind'>{html.escape(f.kind)}</span>"
            f"<span class='gr-msg'>{html.escape(f.message)}</span></div>")


def eyebrow(text: str):
    st.markdown(f"<div class='eyebrow'>{text}</div>", unsafe_allow_html=True)


def _log_feedback(result, question: str, thumbs: str, correction: str) -> None:
    """Append the human's verdict/correction to logs/feedback.jsonl (the human-in-the-loop signal)."""
    try:
        _FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": _dt.datetime.now().isoformat(timespec="seconds"), "question": question,
               "thumbs": thumbs, "correction": correction or "", "sql": result.sql,
               "confidence": (result.verification or {}).get("confidence")}
        with _FEEDBACK_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _render_model(m: dict) -> str:
    """Render an inferential model result (from AgentResult.model) as a markdown table."""
    if m.get("error"):
        return f"⚠ The model could not be fit: {m['error']}"
    head = f"**{m['model_type'].upper()}** · outcome `{m['outcome']}` · n={m['n']:,}"
    if m.get("fit_stat"):
        head += f" · {m['fit_stat']}"
    lines = [head, "", f"| term | {m['effect_label']} | 95% CI | p-value |", "|---|---|---|---|"]
    for t in m["terms"]:
        lo = t["ci_low"]
        ci = "—" if lo != lo else f"[{lo:.3f}, {t['ci_high']:.3f}]"     # lo != lo  →  NaN
        star = " ✳️" if t["p"] < 0.05 else ""
        lines.append(f"| `{t['name']}` | {t['estimate']:.3f} | {ci} | {t['p']:.4f}{star} |")
    if m.get("note"):
        lines.append(f"\n_{m['note']}_")
    return "\n".join(lines)


# ───────────────────────────── run ─────────────────────────────
if go and question:
    with st.spinner("Retrieving context → planning → writing SQL → executing → interpreting…"):
        st.session_state.result = run_analysis(question)
        st.session_state.result_q = question

result = st.session_state.get("result")
result_q = st.session_state.get("result_q", "")

if result is not None:
    if result.clarification:
        eyebrow("Clarification needed")
        st.markdown(f"<div class='card' style='border-left:3px solid var(--accent-2)'>🤔 "
                    f"{html.escape(result.clarification)}</div>", unsafe_allow_html=True)
        st.stop()

    if result.error and result.dataframe is None and not result.attempts:
        st.error(result.error)
        st.stop()

    # ───────── inferential model view (adjusted effects) ─────────
    if result.model is not None:
        if result.hypothesis:
            eyebrow("Hypothesis")
            st.markdown(f"<div class='card hypo'>{html.escape(result.hypothesis)}</div>",
                        unsafe_allow_html=True)
        eyebrow("Analytic query")
        st.code(result.sql, language="sql")
        if result.citations:
            chips = "".join(f"<span class='pill'>{html.escape(t)}</span>" for t in result.citations)
            st.markdown(f"<div class='cite'>tables used: {chips}</div>", unsafe_allow_html=True)
        eyebrow("Statistical model")
        st.markdown(_render_model(result.model))
        eyebrow("Interpretation & recommendation")
        st.markdown(result.interpretation)
        if result.dataframe is not None:
            with st.expander(f"analytic data · {result.n_rows} rows"):
                st.dataframe(result.dataframe, use_container_width=True)
        if result.trace:
            t = result.trace
            toks = t.get("prompt_tokens", 0) + t.get("completion_tokens", 0)
            st.markdown(f"<div class='trace'>⏱ {t.get('latency_ms', 0)} ms · {t.get('calls', 0)} LLM calls · "
                        f"{toks:,} tokens · est. ${t.get('est_cost_usd', 0):.4f}</div>", unsafe_allow_html=True)
        eyebrow("Was this useful?")
        _c = st.columns([1, 1, 8])
        _up, _down = _c[0].button("👍", key="fb_up"), _c[1].button("👎", key="fb_down")
        if _up or _down:
            _log_feedback(result, result_q, "up" if _up else "down", "")
            st.success("Thanks — logged as the human-in-the-loop signal.")
        st.stop()

    # HITL review gate — flag low-confidence / caution answers for human review before acting
    needs_review = bool(
        (result.verification and (result.verification.get("confidence") == "low"
         or not result.verification.get("answers_question", True)))
        or any(f.severity == "caution" for f in result.findings)
    )
    if needs_review:
        st.markdown("<div class='review-gate'>⚠ <b>Human review recommended</b> before acting on this "
                    "— the critic flagged low confidence or the guardrail raised a statistical caution "
                    "(see below).</div>", unsafe_allow_html=True)

    if result.hypothesis:
        eyebrow("Hypothesis")
        plan = f"<div class='plan'>{html.escape(result.plan)}</div>" if result.plan else ""
        st.markdown(f"<div class='card hypo'>{html.escape(result.hypothesis)}{plan}</div>",
                    unsafe_allow_html=True)

    eyebrow("Generated SQL")
    if len(result.attempts) > 1:
        st.markdown(f"<div class='heal-note'>⟳ self-healed after "
                    f"{len(result.attempts) - 1} failed attempt(s)</div>", unsafe_allow_html=True)
        for i, a in enumerate(result.attempts[:-1], 1):
            with st.expander(f"Attempt {i} — {a['error'][:80]}"):
                st.code(a["sql"], language="sql")
    st.code(result.sql, language="sql")
    if result.citations:
        chips = "".join(f"<span class='pill'>{html.escape(t)}</span>" for t in result.citations)
        st.markdown(f"<div class='cite'>tables used: {chips}</div>", unsafe_allow_html=True)

    if result.error:
        st.error(result.error)
        st.stop()

    eyebrow(f"Result <span class='n'>· {result.n_rows} row(s)</span>")

    cards = kpi_cards(result.dataframe, result_q)
    if cards:
        chtml = "".join(
            f"<div class='kpi-card'><div class='kpi-label'>{html.escape(c['label'])}</div>"
            f"<div class='kpi-value'>{html.escape(str(c['value']))}</div>"
            f"<div class='kpi-sub'>{html.escape(str(c.get('sub', '')))}</div></div>"
            for c in cards)
        st.markdown(f"<div class='kpi-row'>{chtml}</div>", unsafe_allow_html=True)

    _chart = build_chart(result.dataframe, result_q)
    if _chart is not None:
        st.altair_chart(_chart, use_container_width=True)

    with st.expander(f"data table · {result.n_rows} rows"):
        st.dataframe(result.dataframe, use_container_width=True)

    if result.verification:
        v = result.verification
        sev = {"high": "info", "medium": "warn", "low": "caution"}.get(v.get("confidence", "medium"), "warn")
        ans = "answers the question" if v.get("answers_question", True) else "may not fully answer the question"
        msg = f"critic confidence <b>{html.escape(str(v.get('confidence', 'medium')))}</b> — {ans}."
        if v.get("issues"):
            msg += " issues: " + "; ".join(html.escape(str(i)) for i in v["issues"])
        eyebrow("Self-verification")
        st.markdown(f"<div class='gr-badge sev-{sev}'><span class='gr-tag'>VERIFY</span>"
                    f"<span class='gr-msg'>{msg}</span></div>", unsafe_allow_html=True)

    eyebrow("Statistical guardrail")
    if result.findings:
        st.markdown("<div class='gr-list'>" + "".join(badge(f) for f in result.findings) + "</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='gr-badge sev-info'><span class='gr-tag'>NOTE</span>"
                    "<span class='gr-msg'>No statistical red flags detected.</span></div>",
                    unsafe_allow_html=True)

    eyebrow("Interpretation & recommendation")
    st.markdown(result.interpretation)

    if result.trace:
        t = result.trace
        toks = t.get("prompt_tokens", 0) + t.get("completion_tokens", 0)
        st.markdown(f"<div class='trace'>⏱ {t.get('latency_ms', 0)} ms · {t.get('calls', 0)} LLM calls · "
                    f"{toks:,} tokens · est. ${t.get('est_cost_usd', 0):.4f}</div>",
                    unsafe_allow_html=True)

    # HITL feedback loop — capture the human verdict/correction for improvement
    eyebrow("Was this useful?")
    fc = st.columns([1, 1, 8])
    up = fc[0].button("👍", key="fb_up")
    down = fc[1].button("👎", key="fb_down")
    correction = st.text_input("correction", key="fb_text", label_visibility="collapsed",
                               placeholder="Optional: what should it have done differently?")
    if up or down:
        _log_feedback(result, result_q, "up" if up else "down", correction)
        st.success("Thanks — logged as the human-in-the-loop signal.")
    st.markdown("<div class='hitl-note'>Read-only agent · shows its SQL + guardrail · a human stays on "
                "the loop. Feedback → logs/feedback.jsonl.</div>", unsafe_allow_html=True)

# ───────────────────────────── footer ─────────────────────────────
st.markdown("""
<div class="footer">
  <span>Synthetic data (Synthea) · no PHI</span><span class="dot">·</span>
  <a href="https://github.com/Ediebah/healthcare-warehouse-agent" target="_blank">GitHub ↗</a>
  <span class="dot">·</span><span>dbt · DuckDB · OpenAI</span>
</div>
""", unsafe_allow_html=True)
