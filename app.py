"""Streamlit demo: an AI agent that does data science over the dbt-modeled healthcare warehouse.

Run:  streamlit run app.py     (needs OPENAI_API_KEY in agent/.env, or Streamlit Cloud secrets)
"""
from __future__ import annotations
import os
import streamlit as st

# Bridge Streamlit Cloud secrets -> env vars BEFORE importing the agent (llm.py reads env at import).
try:
    for _k in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets.toml locally — falls back to agent/.env

from agent import guardrails
from agent.agent import run_analysis
from agent.llm import MODEL

st.set_page_config(page_title="AI Data Scientist · Healthcare Warehouse", layout="wide")

EXAMPLES = [
    "Which chronic conditions drive the highest total encounter cost?",
    "What is the 30-day readmission rate, and does it vary by age group?",
    "What is the prevalence of hypertension by age group?",
    "How does average encounter cost differ by encounter class?",
    "Which conditions are most prevalent in patients 75 and older?",
]

SEVERITY_STYLE = {"caution": st.error, "warn": st.warning, "info": st.info}

with st.sidebar:
    st.header("About")
    st.markdown(
        "An agent that answers a natural-language question by retrieving schema/metric context, "
        "writing SQL against a **dbt-modeled DuckDB warehouse**, self-correcting on errors, and "
        "interpreting the result — then applying a **biostatistics guardrail** (small samples, "
        "confidence intervals, multiple comparisons) that a generic text-to-SQL bot misses."
    )
    st.markdown(
        "**Pipeline:** retrieve → hypothesize → SQL → execute → self-heal → interpret → "
        "recommend → **stat guardrail**"
    )
    st.caption(f"Model: `{MODEL}` · Warehouse: DuckDB (Synthea, synthetic) · Read-only, row-capped SQL")

st.title("🩺 AI Data Scientist over a Healthcare Warehouse")
st.caption("Ask a question in plain English. The agent shows its work — and flags the statistics.")

if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0]

st.write("**Try one:**")
cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    if c.button(ex, use_container_width=True):
        st.session_state.question = ex

question = st.text_input("Your question", value=st.session_state.question)
go = st.button("Run analysis", type="primary")

if go and question:
    with st.spinner("Retrieving context → planning → writing SQL → executing → interpreting…"):
        result = run_analysis(question)

    if result.error and result.dataframe is None and not result.attempts:
        st.error(result.error)
        st.stop()

    if result.hypothesis:
        st.subheader("Hypothesis")
        st.write(result.hypothesis)
        if result.plan:
            st.caption(result.plan)

    st.subheader("SQL")
    if len(result.attempts) > 1:
        st.caption(f"🔧 Self-healed after {len(result.attempts) - 1} failed attempt(s).")
        for i, a in enumerate(result.attempts[:-1], 1):
            with st.expander(f"Attempt {i} — failed: {a['error'][:80]}"):
                st.code(a["sql"], language="sql")
    st.code(result.sql, language="sql")

    if result.error:
        st.error(result.error)
        st.stop()

    st.subheader(f"Result · {result.n_rows} row(s)")
    st.dataframe(result.dataframe, use_container_width=True, height=280)

    st.subheader("Statistical guardrail")
    if result.findings:
        for f in result.findings:
            SEVERITY_STYLE.get(f.severity, st.info)(f"**{f.kind}** — {f.message}")
    else:
        st.success("No statistical red flags detected.")

    st.subheader("Interpretation & recommendation")
    st.markdown(result.interpretation)
