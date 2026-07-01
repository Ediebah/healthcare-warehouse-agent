# AI Data Scientist over a Healthcare Warehouse

An AI agent that does **end-to-end data science over a dbt-modeled healthcare warehouse**: ask a
natural-language question, and the agent retrieves the right schema/metric context, forms a
hypothesis, writes and runs SQL against a modeled warehouse, **self-corrects on errors**,
interprets the result, and drafts a recommendation — **with statistical caveats a generic
text-to-SQL bot misses** (small samples, confidence intervals, multiple comparisons, base-rate
traps).

Built on synthetic EHR data (zero PHI), so the whole thing is public and reproducible.

<!-- 🔗 Live demo: add your Streamlit Community Cloud URL here after deploying -->

---

## Architecture

```
┌──────────┐  generate   ┌───────────┐  load   ┌──────────────────┐   dbt: staging → star schema → analytics
│ Synthea  │────────────▶│ raw CSVs  │────────▶│ DuckDB (raw)     │───────────────────────────────────────┐
│ (Java)   │ seed 12345  │ 10 tables │         │ faithful VARCHAR │                                         │
└──────────┘             └───────────┘         └──────────────────┘                                         ▼
                                                                          ┌───────────────────────────────────────────┐
   dbt docs generate → manifest.json + catalog.json ──────────┐          │ 10 stg_ views · 6 dim_ · 5 fct_ · 3 mart_   │
                                                               │          │ 90 tests · docs on every model              │
                                                               ▼          └───────────────────────────────────────────┘
                                              ┌──────────────────────────┐
                                              │ semantic_catalog.json     │  tables · grain · keys · types ·
                                              │ (LLM-readable)            │  example values · metric SQL + caveats
                                              └────────────┬──────────────┘
                                                           │ RAG (token-overlap retrieval)
                                                           ▼
     ┌──────────────────────────────────  agent loop  ──────────────────────────────────┐
     │ clarify? → retrieve → hypothesize → SQL → execute(read-only) → SELF-HEAL           │
     │   → cite → GUARDRAIL (CIs · FDR · confounding) → VERIFY → interpret → recommend    │
     └───────────────────────────────────────┬───────────────────────────────────────────┘
                                              ▼   Streamlit UI (shows its work + cost trace)
```

---

## What's built

**Both weekends — plus a rigor & production-hardening pass (A–E) — are complete and pushed.**

### Weekend 1 — the warehouse (data-engineering substrate)
- **Data:** [Synthea](https://github.com/synthetichealth/synthea) → 1,139 synthetic patients, reproducible (seed 12345).
- **Warehouse:** [DuckDB](https://duckdb.org); raw CSVs loaded as a faithful all-`VARCHAR` copy.
- **Modeling:** [dbt-core](https://docs.getdbt.com) 1.11 + `dbt-duckdb` + `dbt_utils` — **24 models, 90 passing tests.**

| layer | models | role |
|---|---|---|
| staging (views) | 10 × `stg_*` | clean, rename, cast; one per source |
| marts/core (tables) | 6 × `dim_*`, 5 × `fct_*` | star schema — dims + facts (surrogate keys, measures) |
| marts/analytics (tables) | `mart_readmissions`, `mart_cost_by_condition`, `mart_condition_prevalence` | question-shaped models |

### Weekend 2 — the agent
- **Semantic catalog** auto-generated from dbt artifacts: tables, grain, keys, types, **example values**, and metric SQL **with statistical caveats**.
- **RAG** over the catalog (deterministic token-overlap retrieval — no embedding calls).
- **Self-healing agent loop** (`retrieve → hypothesize → SQL → execute → self-heal → interpret → recommend`) — self-heals on SQL errors *and* on empty/degenerate results.
- **Streamlit UI** that shows the agent's work.

### Weekend 3 — rigor & production hardening (A–E)
- **A · Guardrail → real inference:** Wilson CIs, **Newcombe CIs on group differences + Benjamini-Hochberg FDR**, **confounding** + **Simpson's-paradox** detection, and **skew-aware** summaries (median/IQR + bootstrap mean CI). Measured: **precision/recall 100/100** on labeled cases (`agent/guardrail_eval.py`).
- **B · Trustworthiness:** a **clarify-gate** (asks instead of guessing on vague questions), a **verifier/critic** pass (does the SQL answer *this* question? confidence + issues), and **citations** (which tables were used).
- **C · Eval at grade:** 19 categorized known-answer questions + clarify cases + a **caveat-faithfulness** metric + regression logging → **19/19 accuracy, faithfulness 100%**.
- **D · Real warehouse:** a Snowflake `prod` target (identical models via `dbt build --target prod`) + **GitHub Actions CI** that rebuilds the warehouse, runs `dbt build` (90 tests), and runs the guardrail eval on every push.
- **E · Ops & trust:** read-only + validated + row-capped SQL, a **query audit log**, **prompt-injection** blocking, and **cost/latency tracing** — see [GOVERNANCE.md](GOVERNANCE.md).

Sample: asked "prevalence of hypertension by age group," the agent returns the correct age gradient, computes that **5/6 pairwise contrasts survive FDR** (largest 65–74 vs 18–39, risk difference +47.8pp, 95% CI [38.5, 56.7]), and warns the comparison is **unadjusted for confounders** — inference a plain text-to-SQL bot can't do.

---

## Quickstart

Prereqs: `git`, [`uv`](https://docs.astral.sh/uv/), a **JDK 17+** (only for regenerating data), and an OpenAI API key.

```bash
# 1. Environment
uv venv --python 3.12 && uv pip install -r requirements.txt

# 2. (Weekend 1) generate data → load → build+test the warehouse
cd synthea && java -jar synthea-with-dependencies.jar \
  --exporter.csv.export true --exporter.fhir.export false -p 1000 -s 12345 -cs 12345 Massachusetts && cd ..
.venv/bin/python scripts/load_raw.py
cd warehouse && ../.venv/bin/dbt build --profiles-dir . && ../.venv/bin/dbt docs generate --profiles-dir . && cd ..

# 3. (Weekend 2) build the semantic catalog + add your key
.venv/bin/python agent/build_catalog.py
cp agent/.env.example agent/.env      # then put your OPENAI_API_KEY in agent/.env

# 4. Run the agent (CLI) + the evals
.venv/bin/python -m agent.agent "Which conditions are most prevalent in patients 75 and older?"
.venv/bin/python -m agent.eval             # accuracy eval           -> 19/19
.venv/bin/python -m agent.guardrail_eval   # guardrail precision/recall (no key) -> 100/100

# 5. Run the demo UI
.venv/bin/streamlit run app.py
```

### Deploy (Streamlit Community Cloud)
The repo ships a 28 MB slim `data/healthcare_demo.duckdb` (marts only) so the app runs without
rebuilding the warehouse. Push to GitHub, then at [share.streamlit.io](https://share.streamlit.io):
new app → this repo → `app.py` → add `OPENAI_API_KEY` under **Secrets**.

---

## Repo layout

```
├── README.md · CONCEPTS.md · RUNBOOK.md · GOVERNANCE.md   docs: pitch / WHY / HOW / trust
├── requirements.txt
├── app.py                                    Streamlit demo UI
├── .github/workflows/ci.yml                  dbt build + 90 tests + guardrail eval, every push
├── scripts/load_raw.py                       Synthea CSV → DuckDB raw
├── agent/
│   ├── build_catalog.py                      dbt artifacts → semantic_catalog.{json,md}
│   ├── retrieval.py                          RAG over the catalog (token-overlap)
│   ├── warehouse.py                          read-only, validated, audited SQL execution
│   ├── guardrails.py                         statistical guardrail (Wilson/Newcombe CIs, FDR, …)
│   ├── llm.py · agent.py                      OpenAI wrapper (traced) · the self-healing loop
│   ├── eval.py                               19-question accuracy eval
│   └── guardrail_eval.py                     guardrail precision/recall eval
├── warehouse/                                the dbt project (staging + marts + tests + docs)
└── data/healthcare_demo.duckdb               slim marts DB for the deployed demo (committed)
```

---

## What this demonstrates (mapped to the job description)

| Feature | JD language it hits |
|---|---|
| Self-healing agent loop (hypothesis → SQL → retry → interpret → recommend) | "AI agents that conduct full analyses autonomously" |
| Semantic catalog + RAG from dbt artifacts | "make the warehouse AI-readable… query accurately and reliably" |
| dbt tests + agent SQL-error retry | "self-healing pipelines that detect and fix issues" |
| staging → star schema → analytics marts, deep SQL | "data models, warehouse, deep SQL" |
| Streamlit app, read-only guardrails, deployable | "ship a working tool in Python, production" |
| **Statistical guardrail → inference (Newcombe CIs + FDR, confounding, Simpson's, skew)** | "interpret results, flag statistical issues" — the biostatistics moat |
| Verifier/critic pass + clarify-gate + citations | trustworthy agents, human-in-the-loop |
| CI (dbt build + 90 tests + guardrail eval) + Snowflake target | "self-healing pipelines," warehouse breadth |
| Audit log, prompt-injection guard, cost tracing, governance doc | production ops + security posture |

---

## Honest limits
- **Synthetic data.** Synthea is structurally realistic but generated from care-process models; magnitudes are illustrative, not empirical.
- **Deployed demo DB samples `fct_observations`** (60k of 886k rows) to fit repo limits — no demo/eval question depends on it; the full local warehouse has everything.
- **Cost attribution is intentionally narrow** (`mart_cost_by_condition` = diagnosing-encounter cost, not lifetime cost of care).
- The agent is grounded to the catalog and read-only; it can still write a well-formed query that answers a subtly different question than intended — the shown SQL + guardrail are there so a human stays in the loop.
