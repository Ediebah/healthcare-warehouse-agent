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
     │ retrieve → hypothesize → SQL → execute(read-only) → SELF-HEAL → interpret          │
     │                                                → recommend → STATISTICAL GUARDRAIL │
     └───────────────────────────────────────┬───────────────────────────────────────────┘
                                              ▼   Streamlit UI  (shows its work)
```

---

## What's built

**Both weekends are complete.**

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
- **Semantic catalog** auto-generated from dbt artifacts (`manifest.json`/`catalog.json`): tables, grain, keys, types, **example values**, and metric SQL **with statistical caveats**.
- **RAG** over the catalog (deterministic token-overlap retrieval — no embedding calls).
- **Self-healing agent loop** (`retrieve → hypothesize → SQL → execute → self-heal → interpret → recommend`).
- **Statistical guardrail** (deterministic): small-N flags, **Wilson confidence intervals**, multiple-comparison warnings, missing-denominator/base-rate checks, synthetic-data caveat.
- **Safety:** read-only DuckDB connection + SQL validation (blocks `drop`/`delete`/`update`/multi-statement) + row cap.
- **Streamlit UI** that shows the agent's work (hypothesis, SQL + any retries, result, guardrail, recommendation).
- **Accuracy eval:** 8 questions with known answers → **8/8 (100%) SQL accuracy.**

Sample: given "prevalence of hypertension by age group," the agent returns the correct age gradient **and** flags that a rate was shown without its denominator — the base-rate discipline a plain text-to-SQL bot lacks.

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

# 4. Run the agent (CLI) or the eval
.venv/bin/python -m agent.agent "Which conditions are most prevalent in patients 75 and older?"
.venv/bin/python -m agent.eval        # -> Accuracy: 8/8

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
├── README.md · CONCEPTS.md · RUNBOOK.md      docs: pitch / the WHY / the HOW (line-by-line)
├── requirements.txt
├── app.py                                    Streamlit demo UI
├── scripts/load_raw.py                       Synthea CSV → DuckDB raw
├── agent/
│   ├── build_catalog.py                      dbt artifacts → semantic_catalog.{json,md}
│   ├── retrieval.py                          RAG over the catalog
│   ├── warehouse.py                          read-only, validated SQL execution
│   ├── guardrails.py                         statistical guardrail (Wilson CI, small-N, …)
│   ├── llm.py · agent.py                      OpenAI wrapper · the self-healing loop
│   └── eval.py                               8-question accuracy eval
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
| **Statistical guardrail (Wilson CI, small-N, multiple comparisons)** | "interpret results, flag statistical issues" — the biostatistics moat |

---

## Honest limits
- **Synthetic data.** Synthea is structurally realistic but generated from care-process models; magnitudes are illustrative, not empirical.
- **Deployed demo DB samples `fct_observations`** (60k of 886k rows) to fit repo limits — no demo/eval question depends on it; the full local warehouse has everything.
- **Cost attribution is intentionally narrow** (`mart_cost_by_condition` = diagnosing-encounter cost, not lifetime cost of care).
- The agent is grounded to the catalog and read-only; it can still write a well-formed query that answers a subtly different question than intended — the shown SQL + guardrail are there so a human stays in the loop.
