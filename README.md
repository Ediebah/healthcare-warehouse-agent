# AI Data Scientist over a Healthcare Warehouse

An AI agent that does **end-to-end data science over a dbt-modeled healthcare warehouse**: ask a
natural-language question, and the agent retrieves the right schema/metric context, forms a
hypothesis, writes and runs SQL against a modeled warehouse, **self-corrects on errors**,
interprets the result, and drafts a recommendation — **with statistical caveats a generic
text-to-SQL bot misses** (small samples, confidence intervals, multiple comparisons, base-rate
traps).

Built on synthetic EHR data (zero PHI), so the whole thing is public and reproducible.

**🔗 Live demo:** [healthcare-warehouse-agent.streamlit.app](https://healthcare-warehouse-agent.streamlit.app) · **Repo CI:** dbt build + 97 tests + 47 unit tests + guardrail eval on every push.

![Clinical Insight Agent — a natural-language answer rendered as KPI cards, a bar chart with Wilson 95% confidence-interval whiskers, self-verification, and the statistical guardrail (contrasts + FDR, confounding).](assets/dashboard.png)

---

## Architecture

```
┌──────────┐  generate   ┌───────────┐  load   ┌──────────────────┐   dbt: staging → star schema → analytics
│ Synthea  │────────────▶│ raw CSVs  │────────▶│ DuckDB (raw)     │───────────────────────────────────────┐
│ (Java)   │ seed 12345  │ 10 tables │         │ faithful VARCHAR │                                         │
└──────────┘             └───────────┘         └──────────────────┘                                         ▼
                                                                          ┌───────────────────────────────────────────┐
   dbt docs generate → manifest.json + catalog.json ──────────┐          │ 10 stg_ views · 6 dim_ · 5 fct_ · 3 mart_   │
                                                               │          │ 97 tests · docs on every model              │
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
- **Modeling:** [dbt-core](https://docs.getdbt.com) 1.11 + `dbt-duckdb` + `dbt_utils` — **25 models, 97 passing tests.**

| layer | models | role |
|---|---|---|
| staging (views) | 10 × `stg_*` | clean, rename, cast; one per source |
| marts/core (tables) | 6 × `dim_*`, 5 × `fct_*` | star schema — dims + facts (surrogate keys, measures) |
| marts/analytics (tables) | `mart_readmissions`, `mart_cost_by_condition`, `mart_condition_prevalence`, `mart_experiments` | question-shaped models |

### Weekend 2 — the agent
- **Semantic catalog** auto-generated from dbt artifacts: tables, grain, keys, types, **example values**, and metric SQL **with statistical caveats**.
- **RAG** over the catalog (deterministic token-overlap retrieval — no embedding calls).
- **Self-healing agent loop** (`retrieve → hypothesize → SQL → execute → self-heal → interpret → recommend`) — self-heals on SQL errors *and* on empty/degenerate results.
- **Streamlit UI** that shows the agent's work.

### Weekend 3 — rigor & production hardening (A–E)
- **A · Guardrail → real inference:** Wilson CIs, **Newcombe CIs on group differences + Benjamini-Hochberg FDR**, **confounding** + **Simpson's-paradox** detection, and **skew-aware** summaries (median/IQR + bootstrap mean CI). Measured: **precision/recall 100/100** on labeled cases (`agent/guardrail_eval.py`).
- **A+ · Inferential-modeling layer** (`agent/modeling.py`): the agent auto-routes each question to the most appropriate model — **logistic / OLS regression, Cox + Kaplan-Meier survival, random-forest feature importance, Holt-Winters time-series forecasting, causal T-learner uplift, an association test, or an A/B experiment ship-call** — builds the patient-level (or per-period / per-assignment) analytic dataset via SQL, fits it with `statsmodels` / `scikit-learn`, and returns **adjusted odds/hazard ratios, coefficients, feature importances, forecasts, treatment uplift, or a ship / no-ship verdict with 95% CIs** — each rendered as the right visual (**forest plot, Kaplan-Meier curve, importance bars, forecast band, or a verdict badge + variant chart**). This is the covariate-adjusted model the guardrail keeps recommending — now the agent *does* it, not just flags the need.
- **A+ · Experiment analysis** (`mart_experiments` + `fit_experiment`): "should we ship variant B?" → per-arm conversion/revenue with Wilson CIs, lift with a **Newcombe difference CI + two-proportion z-test**, **BH-FDR across multiple variants**, flagged issues (imbalance, underpowered, multiple comparisons), and a decisive **SHIP / DO NOT SHIP / INCONCLUSIVE** call — the exact "interpret A/B results, flag statistical issues, draft a ship/no-ship recommendation" workflow product teams need. The same engine runs **non-inferiority tests** (`fit_noninferiority`) — is the treatment within a pre-specified margin of control? — with the decision made by the 95% CI vs the margin (one-sided α=0.025), the clinical-trial sibling of the ship-call. Every proportion statistic (Wilson, Newcombe difference CI, two-proportion z, BH-FDR) and the NI decision are **cross-validated against `statsmodels`** (the NI call matches the Farrington–Manning score test).
- **A+ · Automated dbt model generation** (`agent/model_builder.py`): a plain-English data need → the agent drafts a dbt model (`{{ ref() }}` over the star schema) **plus schema tests**, writes the `.sql` + `schema.yml`, runs `dbt build`, and **self-heals** (reads the compile/test failure → rewrites → rebuilds) until it's green — model creation *and* validation, autonomously.
- **B · Trustworthiness:** a **clarify-gate** (asks instead of guessing on vague questions), a **verifier/critic** pass (does the SQL answer *this* question? confidence + issues), and **citations** (which tables were used).
- **C · Evaluation suite:** one labeled `GOLD` dataset (33 cases) drives every metric — **accuracy 33/33**, **guardrail precision/recall 100/100**, **retrieval recall 97% / MRR 0.88**, and an **LLM-as-a-judge** for **factual consistency (0% hallucination) + relevance 5.0/5**, plus caveat-faithfulness and a regression log.
- **D · Real warehouse:** a Snowflake `prod` target (identical models via `dbt build --target prod`) + **GitHub Actions CI** that rebuilds the warehouse, runs `dbt build` (97 tests), and runs the guardrail eval on every push.
- **E · Ops & trust:** read-only + validated + row-capped SQL, a **query audit log**, **prompt-injection** blocking, and **cost/latency tracing**.
- **Plus:** an **industry-grade dashboard** per answer (KPI cards + annotated chart with value labels and **Wilson 95% CI whiskers** — uncertainty shown, not hidden) and a **self-healing pipeline demo** (`agent/pipeline_healer.py`) — a dbt test fails → the agent diagnoses the root cause and proposes a fix → rebuild → green again.

![Survival analysis — "How does patient survival differ by sex?" returns Kaplan-Meier survival curves by group with 95% CI bands and a Cox hazard-ratio forest plot.](assets/survival.png)

![Experiment analysis — "Analyze the checkout redesign A/B test — should we ship it?" returns a SHIP / NO-SHIP verdict, per-variant conversion with 95% CIs, and a lift forest plot (Newcombe CI + two-proportion z-test, BH-FDR across variants).](assets/ab-experiment.png)

![Non-inferiority — the treatment−control effect with its 95% CI shown against the non-inferiority margin (gold) and no-difference (grey); non-inferior when the CI stays inside the margin. Decision cross-validated against the Farrington–Manning score test.](assets/ni-noninferiority.png)

### Production hardening
- **Tests + lint in CI:** 47 keyless `pytest` unit tests (guardrail stats, SQL validation, retrieval, charts, agent helpers) + `ruff`, run on every push.
- **Resilience:** OpenAI client with retries + backoff + timeout; API errors degrade gracefully.
- **Security:** prompt-injection guard (patterns + length cap) on top of the read-only/validated SQL guarantee.
- **Observability:** every run's tokens/latency/cost persisted; `agent/observe.py` reports run count, error rate, **latency p50/p95**, spend.
- **Human-in-the-loop:** shown SQL + guardrail + citations (oversight), a **"review recommended" gate** on low-confidence answers, and a **👍/👎 + correction feedback loop** logged for improvement.

Sample: asked "prevalence of hypertension by age group," the agent returns the correct age gradient, computes that **5/6 pairwise contrasts survive FDR** (largest 65–74 vs 18–39, risk difference +47.8pp, 95% CI [38.5, 56.7]), and warns the comparison is **unadjusted for confounders** — inference a plain text-to-SQL bot can't do.

---

## Quickstart

Prereqs: `git`, [`uv`](https://docs.astral.sh/uv/), a **JDK 17+** (only for regenerating data), and an OpenAI API key.

```bash
# 1. Environment (dev = app + dbt; the deployed app itself installs only requirements.txt)
uv venv --python 3.12 && uv pip install -r requirements-dev.txt

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
.venv/bin/pytest                           # 47 keyless unit tests   (ruff check . to lint)
.venv/bin/python -m agent.eval             # accuracy eval               -> 33/33
.venv/bin/python -m agent.guardrail_eval   # guardrail precision/recall (no key) -> 100/100
.venv/bin/python -m agent.eval_retrieval   # retrieval precision/recall/MRR (no key) -> recall 97%
.venv/bin/python -m agent.eval_judge       # LLM-as-judge: factual consistency + relevance
.venv/bin/python -m agent.pipeline_healer  # self-healing demo: dbt test fails → agent fixes → green
.venv/bin/python -m agent.model_builder    # autogen a dbt model + tests from plain English → build → self-heal
.venv/bin/python -m agent.observe          # observability: run count, error rate, latency p50/p95, spend

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
├── README.md                                 project overview, architecture, quickstart
├── requirements.txt
├── app.py                                    Streamlit demo UI
├── .github/workflows/ci.yml                  dbt build + 97 tests + guardrail eval, every push
├── scripts/load_raw.py                       Synthea CSV → DuckDB raw
├── agent/
│   ├── build_catalog.py                      dbt artifacts → semantic_catalog.{json,md}
│   ├── retrieval.py                          RAG over the catalog (token-overlap)
│   ├── warehouse.py                          read-only, validated, audited SQL execution
│   ├── guardrails.py                         statistical guardrail (Wilson/Newcombe CIs, FDR, …)
│   ├── llm.py · agent.py                      OpenAI wrapper (traced, retry) · the self-healing loop
│   ├── charts.py                             dashboard: KPI cards + annotated chart (Wilson CI whiskers)
│   ├── pipeline_healer.py · observe.py        self-healing pipeline demo · observability summary
│   ├── model_builder.py                       autogen + validate a dbt model from plain English
│   ├── eval_dataset.py                       GOLD: one labeled ground-truth dataset for every eval
│   ├── eval.py · guardrail_eval.py           accuracy (33/33) · guardrail precision/recall (100/100)
│   └── eval_retrieval.py · eval_judge.py     retrieval precision/recall/MRR · LLM-as-judge (factual consistency)
├── tests/                                    47 keyless pytest unit tests (guardrail stats, SQL, retrieval, charts)
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
| **Self-healing pipeline** (dbt test fails → agent diagnoses → repairs → verifies) + CI + Snowflake target | "self-healing pipelines that detect and fix data issues" |
| **A/B experiment analyzer** (per-arm CIs, Newcombe lift + FDR, SHIP / NO-SHIP / INCONCLUSIVE verdict) | "ship AI-powered experiment analysis — interpret A/B results, flag statistical issues, draft ship/no-ship recommendations" |
| **Automated dbt model generation + validation** (draft model + schema tests → `dbt build` → self-heal until green) | "automate the data lifecycle — automated dbt model generation and validation" |
| **Self-serve interface** (plain English → the right method + visual, replacing ad-hoc SQL requests) | "turn the data team into a product team — self-serve AI interfaces stakeholders use" |
| Audit log, prompt-injection guard, cost tracing, governance doc | production ops + security posture |
| Eval suite: retrieval precision, **hallucination rate, factual consistency, LLM-as-a-judge**, ground-truth dataset | "evaluation metrics + ground-truth datasets; LLM-as-judge setups" |
| Dashboard: KPI cards + annotated chart with confidence-interval whiskers | data storytelling / visualization |
| Unit tests + ruff in CI, LLM retry/backoff, observability (latency p50/p95, cost) | production engineering practices |
| **Human-in-the-loop**: review gate on low confidence + 👍/👎 feedback loop + shown SQL/guardrail | "human-in-the-loop," safe autonomous analysis |

---

## Honest limits
- **Synthetic data.** Synthea is structurally realistic but generated from care-process models; magnitudes are illustrative, not empirical.
- **Deployed demo DB samples `fct_observations`** (60k of 886k rows) to fit repo limits — no demo/eval question depends on it; the full local warehouse has everything.
- **Cost attribution is intentionally narrow** (`mart_cost_by_condition` = diagnosing-encounter cost, not lifetime cost of care).
- The agent is grounded to the catalog and read-only; it can still write a well-formed query that answers a subtly different question than intended — the shown SQL + guardrail are there so a human stays in the loop.
