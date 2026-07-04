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
import shutil
from pathlib import Path

import pandas as pd
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
from agent.charts import (
    build_chart,
    experiment_chart,
    forecast_chart,
    forest_plot,
    importance_chart,
    kpi_cards,
    ni_plot,
    power_curve_chart,
    radar_chart,
    survival_plot,
)
from agent.retrieval import load_catalog

st.set_page_config(page_title="Clinical Insight Agent", page_icon="🩺",
                   layout="wide", initial_sidebar_state="collapsed")

EXAMPLE_GROUPS = {
    "Clinical inference & modeling": [
        "How does patient survival differ by sex?",
        "What are the strongest risk factors for patient mortality?",
        "What predicts 30-day readmission, adjusting for age and sex?",
        "What is the effect of insurance coverage on mortality, adjusting for age and income?",
    ],
    "Clinical trials — non-inferiority": [
        "Is the new antibiotic non-inferior to standard of care on cure rate (10-point margin)?",
        "Is the new device non-inferior to standard of care on cure rate (10-point margin)?",
    ],
    "Trial design — power & sample size": [
        "How many patients per arm to detect a 10-point rise in cure rate from 70%, at 80% power?",
        "Sample size for a non-inferiority trial: 85% control cure rate, 10-point margin, 90% power?",
    ],
    "Operational & product analytics": [
        "Which chronic conditions drive the highest total encounter cost?",
        "What is the 30-day readmission rate, and does it vary by age group?",
        "Forecast monthly encounter volume for the next 12 months.",
        "Analyze the checkout redesign A/B test — should we ship it?",
    ],
}
EXAMPLES = [q for qs in EXAMPLE_GROUPS.values() for q in qs]

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
/* Visually hide widget labels but keep them in the a11y tree for screen readers (not display:none). */
[data-testid="stWidgetLabel"]{ position:absolute; width:1px; height:1px; overflow:hidden;
  clip-path:inset(50%); white-space:nowrap; }

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


# ───────────────────────────── monitoring (ops surface) ─────────────────────────────
def _mon_kpis(cards: list[dict]) -> None:
    html_cards = "".join(
        f"<div class='kpi-card'><div class='kpi-label'>{c['label']}</div>"
        f"<div class='kpi-value'>{c['value']}</div><div class='kpi-sub'>{c.get('sub', '')}</div></div>"
        for c in cards)
    st.markdown(f"<div class='kpi-row'>{html_cards}</div>", unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def _data_health() -> dict:
    """Live, automated data-quality checks against the warehouse — the detect/diagnose half of a data-
    quality agent: row volumes, primary-key uniqueness, referential integrity, completeness, metric sanity."""
    from agent import warehouse as W

    def scalar(sql: str):
        return W.run_query(sql).iloc[0, 0]

    try:
        tables = ["dim_patient", "fct_encounters", "fct_conditions", "mart_readmissions",
                  "mart_cost_by_condition"]
        counts = {t: int(scalar(f"select count(*) as n from {t}")) for t in tables}
        checks = []
        pk = int(scalar("select (count(*) = count(distinct patient_id))::int from dim_patient"))
        checks.append(("Primary-key uniqueness", "dim_patient.patient_id has no duplicates", pk == 1))
        orphans = int(scalar("select count(*) from fct_encounters e "
                             "left join dim_patient p on e.patient_id = p.patient_id "
                             "where p.patient_id is null"))
        checks.append(("Referential integrity", f"fct_encounters → dim_patient: {orphans} orphan row(s)",
                       orphans == 0))
        nullg = float(scalar("select avg((gender is null)::int) from dim_patient"))
        checks.append(("Completeness", f"dim_patient.gender null rate {nullg:.1%}", nullg < 0.05))
        readm = float(scalar("select avg(cast(is_30d_readmission as int)) from mart_readmissions"))
        checks.append(("Metric sanity", f"30-day readmission rate {readm:.1%} (plausible band 3–25%)",
                       0.03 <= readm <= 0.25))
        return {"ok": True, "counts": counts, "checks": checks}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": str(e), "counts": {}, "checks": []}


def _render_monitoring() -> None:
    """The ops surface a data team watches in production: agent usage/latency/cost, human feedback, and
    live warehouse data-quality checks. Reads logs/*.jsonl (per-deployment) + queries the warehouse."""
    import altair as alt

    from agent import observe
    traces = observe.load_traces()
    s = observe.summary(traces)

    st.markdown("<div class='eyebrow'>System monitoring · agent + warehouse</div>", unsafe_allow_html=True)
    if s["runs"]:
        _mon_kpis([
            {"label": "runs (this deploy)", "value": f"{s['runs']:,}"},
            {"label": "success rate", "value": f"{s['success_rate'] * 100:.0f}%",
             "sub": f"{s['errors']} errored"},
            {"label": "latency p50", "value": f"{s['p50_ms'] / 1000:.1f}s", "sub": f"p95 {s['p95_ms'] / 1000:.1f}s"},
            {"label": "avg tokens / run", "value": f"{s['avg_tokens']:,}"},
            {"label": "est. spend", "value": f"${s['spend_usd']:.2f}", "sub": f"${s['avg_cost_usd']:.4f}/run"},
        ])
        dft = pd.DataFrame(traces)
        dft["day"] = dft["ts"].astype(str).str[:10]
        by_day = dft.groupby("day").size().reset_index(name="runs")
        st.markdown("<div class='eyebrow'>Activity</div>", unsafe_allow_html=True)
        ch = (alt.Chart(by_day).mark_bar(color="#4fd1c5", cornerRadiusEnd=2)
              .encode(x=alt.X("day:O", title=None), y=alt.Y("runs:Q", title="runs"))
              .properties(height=160, background="transparent")
              .configure_axis(labelColor="#8ea0b0", titleColor="#8ea0b0", gridColor="#1a2531",
                              domainColor="#20303f")
              .configure_view(strokeWidth=0))
        st.altair_chart(ch, use_container_width=True)
        st.markdown("<div class='eyebrow'>Most-asked questions</div>", unsafe_allow_html=True)
        top = dft["question"].value_counts().head(8).reset_index()
        top.columns = ["question", "runs"]
        st.dataframe(top, use_container_width=True, hide_index=True)
    else:
        st.info("No agent runs recorded in this deployment yet — run an analysis on the **Analyze** tab and "
                "usage metrics populate here. (Traces are per-deployment and reset on redeploy.)")

    fb = []
    try:
        if _FEEDBACK_LOG.exists():
            for line in _FEEDBACK_LOG.read_text().splitlines():
                try:
                    fb.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass
    if fb:
        up = sum(1 for f in fb if f.get("thumbs") == "up")
        down = sum(1 for f in fb if f.get("thumbs") == "down")
        st.markdown("<div class='eyebrow'>Human-in-the-loop feedback</div>", unsafe_allow_html=True)
        _mon_kpis([{"label": "total ratings", "value": f"{len(fb):,}"},
                   {"label": "👍 helpful", "value": f"{up:,}"},
                   {"label": "👎 needs work", "value": f"{down:,}"}])
        corr = [f for f in fb if f.get("correction")]
        for f in corr[-3:]:
            st.markdown(f"<div class='card' style='margin-bottom:.4rem'><b>{html.escape(str(f.get('question', ''))[:90])}"
                        f"</b><br><span style='color:#8ea0b0'>{html.escape(str(f.get('correction', ''))[:200])}</span></div>",
                        unsafe_allow_html=True)

    st.markdown("<div class='eyebrow'>Warehouse data health · automated checks</div>", unsafe_allow_html=True)
    health = _data_health()
    if not health["ok"]:
        st.warning(f"Health checks unavailable: {health.get('err', '')}")
    else:
        _mon_kpis([{"label": t.replace("_", " "), "value": f"{n:,}", "sub": "rows"}
                   for t, n in health["counts"].items()])
        for name, detail, ok in health["checks"]:
            sev, tag = ("info", "PASS") if ok else ("caution", "FAIL")
            st.markdown(f"<div class='gr-badge sev-{sev}'><span class='gr-tag'>{tag}</span>"
                        f"<span class='gr-kind'>{name}</span><span class='gr-msg'>{detail}</span></div>",
                        unsafe_allow_html=True)
    st.markdown("<div class='trace'>Telemetry: logs/traces.jsonl · audit.jsonl · feedback.jsonl "
                "(per-deployment) + live warehouse checks — the ops surface a data team watches in "
                "production.</div>", unsafe_allow_html=True)


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
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Healthcare analytics · agentic AI</div>
  <h1 class="hero-title">Clinical Insight <span class="accent">Agent</span></h1>
  <p class="hero-sub">An AI agent that modernizes clinical data analysis. Ask in plain English — it
  engineers the data (missingness, collinearity), fits the right model (survival, regression,
  non-inferiority, causal, or ML), checks the assumptions (proportional hazards, separation, VIF), and
  reports the result with the caveats a text-to-SQL bot skips.</p>
  <div class="pill-row">
    <span class="pill">dbt star schema</span>
    <span class="pill">RAG semantic catalog</span>
    <span class="pill">self-healing SQL</span>
    <span class="pill">survival · regression · causal</span>
    <span class="pill">non-inferiority · sample size</span>
    <span class="pill">VIF · PH · assumption checks</span>
    <span class="pill">Wilson CIs · FDR</span>
  </div>
  <div class="meta">OpenAI <span class="dot">·</span> DuckDB (Synthea, synthetic)
    <span class="dot">·</span> read-only, row-capped</div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────── nav ─────────────────────────────
_view = st.segmented_control("nav", ["🔬 Analyze", "📊 Monitoring"], default="🔬 Analyze",
                             label_visibility="collapsed")
if _view == "📊 Monitoring":                             # render the ops surface and stop (skip the analyze flow)
    _render_monitoring()
    st.stop()

# ───────────────────────────── ask ─────────────────────────────
if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0]
if "byod" not in st.session_state:
    st.session_state.byod = None          # holds {db_path, dir, catalog, table} once a file is loaded


def _clear_byod() -> None:
    """Drop the current BYOD session AND delete its mkdtemp DuckDB dir (else one leaks per upload/toggle)."""
    _b = st.session_state.get("byod")
    if _b and _b.get("dir"):
        shutil.rmtree(_b["dir"], ignore_errors=True)
    st.session_state.byod = None


_src = st.radio("Data source", ["Demo warehouse", "Bring your own data"],
                horizontal=True, label_visibility="collapsed")
_byod_mode = _src == "Bring your own data"
if st.session_state.get("_prev_src") != _src:        # source changed → drop the prior result
    st.session_state.pop("result", None)
    st.session_state._prev_src = _src

if _byod_mode:
    st.markdown("<div class='eyebrow'>Bring your own data</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#f5c451;font-size:.82rem;margin:-.2rem 0 .5rem'>⚠ Uploads reach the "
                "server and the model sees your column names + query results — use synthetic / "
                "non-sensitive data only (no PHI or confidential data).</div>", unsafe_allow_html=True)
    _up = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"],
                           label_visibility="collapsed")
    _b = st.session_state.byod
    _fid = getattr(_up, "file_id", None) if _up is not None else None
    if _up is None:                                  # no file (never uploaded or removed) → clear session
        _clear_byod()                                # also delete any leaked temp dir from a prior upload
        st.info("Upload a table, then ask in plain English — the agent queries it and picks the right "
                "method (regression, survival, non-inferiority, forecast…).")
    elif _b and _fid is not None and _b.get("file_id") == _fid:   # same file → don't re-materialize
        st.success(f"Loaded **{_b['table']}** — {_b['rows']:,} rows × {_b['ncols']} columns.")
        st.markdown(f"<div style='color:#8ea0b0;font-size:.85rem'>Ask about: {_b['cols']}</div>",
                    unsafe_allow_html=True)
    elif (getattr(_up, "size", None) or 0) > 50 * 1024 * 1024:   # cap BEFORE read → don't OOM on huge files
        _clear_byod()
        st.error(f"That file is {_up.size / 1024 / 1024:.0f} MB, over the 50 MB limit — please upload a "
                 "smaller CSV/Excel file (or pre-aggregate it first).")
    else:
        try:
            from agent import userdata
            if _up.name.lower().endswith(".csv"):
                _df = pd.read_csv(_up)
            else:                                    # .xlsx → flag extra sheets before reading the first one
                _xl = pd.ExcelFile(_up)
                if len(_xl.sheet_names) > 1:
                    st.warning(f"This workbook has {len(_xl.sheet_names)} sheets — only the first "
                               f"(**{_xl.sheet_names[0]}**) was analyzed.")
                _df = pd.read_excel(_xl, sheet_name=_xl.sheet_names[0])
            _orig_rows, _orig_cols = _df.shape       # remember pre-cap size so we can disclose truncation
            _clear_byod()                            # delete the previous upload's temp dir before the new one
            _dbp, _cat, _tbl, _clean = userdata.prepare_upload(_df, _up.name)
            _cols = ", ".join(f"`{c['name']}`" for c in _cat["tables"][0]["columns"])
            st.session_state.byod = {
                "db_path": str(_dbp), "dir": str(_dbp.parent), "catalog": _cat, "table": _tbl,
                "file_id": _fid, "rows": len(_clean), "ncols": len(_clean.columns), "cols": _cols}
            st.session_state.pop("result", None)     # new data → drop the stale prior result
            st.success(f"Loaded **{_tbl}** — {len(_clean):,} rows × {len(_clean.columns)} columns.")
            if _orig_rows > len(_clean) or _orig_cols > len(_clean.columns):   # honest about what was kept
                st.info(f"Analyzing the first {len(_clean):,} of {_orig_rows:,} rows and "
                        f"{len(_clean.columns)} of {_orig_cols} columns "
                        f"(capped at {userdata.MAX_ROWS:,} rows × {userdata.MAX_COLS} columns).")
            st.dataframe(_clean.head(8), use_container_width=True)
            st.markdown(f"<div style='color:#8ea0b0;font-size:.85rem'>Ask about: {_cols}</div>",
                        unsafe_allow_html=True)
        except Exception as _e:  # noqa: BLE001
            _clear_byod()
            st.error(f"Could not read that file: {_e}")
else:
    _clear_byod()                                    # leaving BYOD → drop its temp dir (else it leaks)
    with st.expander("📋 What data can I ask about?"):
        st.markdown(_data_dictionary())
    st.markdown("<div class='eyebrow'>Try one — the agent auto-selects the method</div>",
                unsafe_allow_html=True)
    for _group, _qs in EXAMPLE_GROUPS.items():
        st.markdown(f"<div style='color:var(--accent);font-size:.98rem;font-weight:700;"
                    f"letter-spacing:.01em;margin:1rem 0 .4rem'>{_group}</div>", unsafe_allow_html=True)
        for start in range(0, len(_qs), 3):
            cols = st.columns(3)
            for c, ex in zip(cols, _qs[start:start + 3]):
                if c.button(ex, key=f"ex_{ex}", use_container_width=True):
                    st.session_state.question = ex

st.markdown("<div class='eyebrow'>Your question</div>", unsafe_allow_html=True)
question = st.text_input("q", key="question", label_visibility="collapsed")   # persists typed text
go = st.button("Run analysis  →", type="primary")


def badge(f) -> str:
    tag = {"caution": "CAUTION", "warn": "WARN", "info": "NOTE"}.get(f.severity, "NOTE")
    return (f"<div class='gr-badge sev-{f.severity}'>"
            f"<span class='gr-tag'>{tag}</span>"
            f"<span class='gr-kind'>{html.escape(f.kind)}</span>"
            f"<span class='gr-msg'>{html.escape(f.message)}</span></div>")


def eyebrow(text: str):
    st.markdown(f"<div class='eyebrow'>{text}</div>", unsafe_allow_html=True)


def _result_key(result) -> str:
    """Content hash of a result — the report cache is keyed on THIS, never on the question text, so a
    cross-session cache hit can only mean identical content (no leak of one user's uploaded data)."""
    import hashlib
    parts = [result.question or "", getattr(result, "hypothesis", "") or "", result.sql or "",
             getattr(result, "interpretation", "") or "",
             json.dumps(result.model or {}, default=str, sort_keys=True)]
    df = getattr(result, "dataframe", None)
    if df is not None:
        try:
            parts.append(str(int(pd.util.hash_pandas_object(df, index=True).sum())))
        except Exception:  # noqa: BLE001
            parts.append(str(df.shape))
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_report(_result, content_key: str):   # keyed on content_key (a content hash), not the question
    from agent.report import build_docx
    return build_docx(_result)


# Re-enabled: the .docx now renders figures with a dedicated light/print theme (white background, dark
# ink, subtle gridlines) at 200 ppi, so titles, axis labels and CI whiskers are legible on the page.
_REPORT_EXPORT_ENABLED = True


def _report_button(result, view: str):
    """Offer the analysis as a regulated-style .docx report (cached by content so charts render once)."""
    if not _REPORT_EXPORT_ENABLED:
        return
    try:
        data = _cached_report(result, _result_key(result))
    except Exception as _e:  # noqa: BLE001
        st.caption(f"Report export unavailable: {_e}")
        return
    st.download_button(
        "⬇  Export report (.docx)", data=data, key=f"dl_{view}",
        file_name="statistical_analysis_report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


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
    if m.get("model_type") == "sample_size":
        lines = [f"**Required sample size** · {m.get('fit_stat', '')}", ""]
        for a in m.get("arms", []):
            tag = " (control)" if a.get("is_baseline") else ""
            lines.append(f"- **{a['arm']}**{tag}: {a['n']:,} subjects")
        lines.append(f"- **total**: {m['n']:,} subjects")
        if m.get("note"):
            lines.append(f"\n_{m['note']}_")
        return "\n".join(lines)
    if m.get("model_type") in ("experiment", "noninferiority") and m.get("arms"):   # arms + issues
        binm = all(0 <= a["value"] <= 1 for a in m["arms"])
        metric = "conversion" if binm else "mean"
        lines = [head, "", f"| variant | {metric} | 95% CI | n |", "|---|---|---|---|"]
        for a in m.get("arms", []):
            val = f"{a['value'] * 100:.1f}%" if binm else f"{a['value']:.2f}"
            ci = (f"[{a['ci_low'] * 100:.1f}%, {a['ci_high'] * 100:.1f}%]" if binm
                  else f"[{a['ci_low']:.2f}, {a['ci_high']:.2f}]")
            tag = " · control" if a["is_baseline"] else (" · ✅ winner" if a["is_winner"] else "")
            lines.append(f"| `{a['arm']}`{tag} | {val} | {ci} | {a['n']:,} |")
        if m.get("issues"):
            lines.append("")
            lines += [f"- ⚠️ {iss}" for iss in m["issues"]]
        if m.get("note"):
            lines.append(f"\n_{m['note']}_")
        return "\n".join(lines)
    if m.get("model_type") == "timeseries":                          # forecast periods, not effect terms
        fc = [p for p in m.get("series", []) if p["kind"] == "forecast"]
        lines = [head, "", "| period | forecast | 95% band |", "|---|---|---|"]
        for p in fc:
            lines.append(f"| {p['time'][:10]} | {p['value']:.1f} | [{p['lower']:.1f}, {p['upper']:.1f}] |")
        if m.get("note"):
            lines.append(f"\n_{m['note']}_")
        return "\n".join(lines)
    terms = m["terms"]
    has_ci = any(t["ci_low"] == t["ci_low"] for t in terms)          # drop columns that don't apply
    has_p = any(t["p"] == t["p"] for t in terms)                     # (e.g. forest importance: neither)
    has_n = any(t.get("n") is not None for t in terms)               # per-category subjects (categoricals)
    has_ev = any(t.get("events") is not None for t in terms)         # per-category events (event models)
    cols = (["term"] + (["n"] if has_n else []) + (["events"] if has_ev else [])
            + [m["effect_label"]] + (["95% CI"] if has_ci else []) + (["p-value"] if has_p else []))
    lines = [head, "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for t in terms:
        row = [f"`{t['name']}`"]
        if has_n:
            row.append("—" if t.get("n") is None else f"{t['n']:,}")
        if has_ev:
            row.append("—" if t.get("events") is None else f"{t['events']:,}")
        row.append(f"{t['estimate']:.3f}")
        if has_ci:
            lo = t["ci_low"]
            row.append("—" if lo != lo else f"[{lo:.3f}, {t['ci_high']:.3f}]")    # lo != lo → NaN
        if has_p:
            pv = t["p"]
            row.append("—" if pv != pv else f"{pv:.4f}" + (" ✳️" if pv < 0.05 else ""))
        lines.append("| " + " | ".join(row) + " |")
    if m.get("issues"):
        lines.append("")
        lines += [f"- ⚠️ {iss}" for iss in m["issues"]]
    if m.get("note"):
        lines.append(f"\n_{m['note']}_")
    return "\n".join(lines)


# ───────────────────────────── run ─────────────────────────────
if go and question and _byod_mode and not st.session_state.byod:
    st.warning("Upload a CSV or Excel file first, or switch to the demo warehouse.")
elif go and question:
    _b = st.session_state.byod
    with st.spinner("Retrieving context → planning → writing SQL → executing → interpreting…"):
        st.session_state.result = run_analysis(
            question, catalog=(_b or {}).get("catalog"), db_path=(_b or {}).get("db_path"))
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
        if result.sql:                               # sample-size is a design calc — no analytic query
            eyebrow("Analytic query")
            st.code(result.sql, language="sql")
            if result.citations:
                chips = "".join(f"<span class='pill'>{html.escape(t)}</span>" for t in result.citations)
                st.markdown(f"<div class='cite'>tables used: {chips}</div>", unsafe_allow_html=True)
        eyebrow("Statistical model")
        _mt = result.model.get("model_type")
        if _mt in ("experiment", "noninferiority", "sample_size"):   # decision/design → verdict badge
            _v = result.model.get("verdict", {})
            _call = _v.get("call", "")
            _color = ("#4fd1c5" if _mt == "sample_size" else
                      {"SHIP": "#4fd1c5", "NON-INFERIOR": "#4fd1c5", "DO NOT SHIP": "#f87171",
                       "NOT NON-INFERIOR": "#f87171", "INCONCLUSIVE": "#f5c451"}.get(_call, "#8ea0b0"))
            st.markdown(
                f"<div style='display:inline-block;border:2px solid {_color};color:{_color};"
                f"padding:.45rem 1.1rem;border-radius:9px;font-weight:700;font-size:1.15rem;"
                f"letter-spacing:.06em;margin:.2rem 0 .5rem'>{html.escape(_call)}</div>"
                f"<div style='color:#cfe0ec;margin-bottom:.5rem;max-width:60ch'>"
                f"{html.escape(_v.get('reason', ''))}</div>", unsafe_allow_html=True)
            _dc = (experiment_chart(result.model) if _mt == "experiment"
                   else ni_plot(result.model) if _mt == "noninferiority"
                   else power_curve_chart(result.model))
            if _dc is not None:
                st.altair_chart(_dc, use_container_width=True)
        if result.model.get("km"):                       # survival → Kaplan-Meier curves
            st.altair_chart(survival_plot(result.model["km"]), use_container_width=True)
        if _mt == "timeseries" and result.model.get("series"):   # time-series → history + forecast
            st.altair_chart(forecast_chart(result.model["series"]), use_container_width=True)
        if _mt == "forest":                              # ML → feature-importance bars
            _imp = importance_chart(result.model)
            if _imp is not None:
                st.altair_chart(_imp, use_container_width=True)
        # regression/survival/causal/experiment → effect forest (NI already shows it vs the margin)
        _fp = forest_plot(result.model) if _mt != "noninferiority" else None
        if _fp is not None:
            st.altair_chart(_fp, use_container_width=True)
        st.markdown(_render_model(result.model))
        eyebrow("Interpretation & recommendation")
        st.markdown(result.interpretation)
        _report_button(result, "model")
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

    _radar = radar_chart(result.dataframe, result_q)
    if _radar is not None:
        eyebrow("Radar — entities across metrics (normalized)")
        st.plotly_chart(_radar, use_container_width=True)

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
    _report_button(result, "agg")

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
