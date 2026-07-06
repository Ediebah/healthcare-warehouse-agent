# Clinical Insight Agent: an AI data scientist over a dbt warehouse

**An AI agent that runs end-to-end data science over a dbt-modeled healthcare warehouse, not text-to-SQL.**
Ask a question in plain English and the agent retrieves the right schema context, **picks and fits the
appropriate statistical model** (survival, adjusted regression, causal, non-inferiority, forecast, or ML),
**engineers the data and checks the assumptions**, runs a **deterministic statistical guardrail**, verifies
its own work, and reports the result, **with the caveats a generic SQL bot skips**, exportable as a
regulated-style **Word report**.

The differentiator isn't the LLM, it's the **statistical judgment encoded around it**: the pre-modeling
data engineering, the assumption diagnostics, and the guardrail (confidence intervals, multiplicity
correction, confounding) that a text-to-SQL tool never does. Built by a clinical data scientist
(biostatistics + ML).

Built entirely on **synthetic EHR data (zero PHI)**, so the whole thing is public and reproducible.

**🔗 Live demo:** [healthcare-warehouse-agent.streamlit.app](https://healthcare-warehouse-agent.streamlit.app)
· **CI on every push:** `dbt build` + 104 data tests + 105 unit tests + guardrail eval.

![Clinical Insight Agent: a natural-language answer rendered as KPI cards, a bar chart with Wilson 95% confidence-interval whiskers, self-verification, and the statistical guardrail (contrasts + FDR, confounding).](assets/dashboard.png)

---

## What it does (in 30 seconds)

- **Full analyses, autonomously.** One question → retrieve → plan → SQL → **self-heal** (fixes its own SQL on
  errors, empty results, and degenerate all-zero aggregates) → cite → guardrail → **verify** → interpret +
  recommend. It shows all of its work and its cost/latency trace.
- **Auto-routes to the right model**, then fits it with `statsmodels`/`scikit-learn` (deterministic, the
  numbers aren't hallucinated): regression, survival, causal effects, A/B ship-calls, non-inferiority,
  forecasting, feature importance, and design-stage power/sample-size.
- **Encodes biostatistics rigor** a SQL bot skips: covariate adjustment, confidence intervals everywhere,
  FDR multiplicity correction, confounding/Simpson's-paradox flags, and assumption diagnostics.
- **Condition-specific, in plain English.** Name a disease the way people say it — *heart attack*, *COPD*,
  *diabetes*, *MI* — and it resolves the term to the warehouse's real SNOMED codes before querying (so
  "heart attack" finds *Myocardial infarction*), grounds the cohort in what actually exists, and says so
  honestly when a condition isn't in the data instead of analyzing an empty set.
- **Production-shaped:** read-only hardened at the engine, a live monitoring tab, an eval suite, 105 unit
  tests + CI, a Dockerfile, bring-your-own-data upload, and a `.docx` report export.

---

## Architecture

```
┌──────────┐  generate   ┌───────────┐  load   ┌──────────────────┐   dbt: staging → star schema → analytics
│ Synthea  │────────────▶│ raw CSVs  │────────▶│ DuckDB (raw)     │───────────────────────────────────────┐
│ (Java)   │ seed 12345  │ 10 tables │         │ faithful VARCHAR │                                         │
└──────────┘             └───────────┘         └──────────────────┘                                         ▼
                                                                          ┌───────────────────────────────────────────┐
   dbt docs generate → manifest.json + catalog.json ──────────┐          │ 10 stg_ views · 6 dim_ · 5 fct_ · 5 mart_   │
                                                               │          │ 26 models · 104 data tests · docs each      │
                                                               ▼          └───────────────────────────────────────────┘
                                              ┌──────────────────────────┐
                                              │ semantic_catalog.json     │  tables · grain · keys · types ·
                                              │ (AI-readable)             │  example values · metric SQL + caveats
                                              └────────────┬──────────────┘
                                                           │ RAG (token-overlap retrieval, no embeddings)
                                                           ▼
   ┌──────────────────────────────────────  agent loop  ──────────────────────────────────────┐
   │ triage/clarify? → retrieve → route → plan → SQL → execute(read-only) → SELF-HEAL           │
   │   → cite → GUARDRAIL (CIs · FDR · confounding) → VERIFY (LLM critic) → interpret → model    │
   └───────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                ▼  Streamlit UI, shows its work, cost trace, monitoring
```

---

## Capabilities

### The agent loop  (`agent/agent.py`)
Prompt-injection guard → RAG retrieval over the semantic catalog → **model routing** (does this need an
inferential model, or a plain aggregation?) → clarify-gate on vague questions → hypothesis + plan → generate
DuckDB SQL → execute read-only and **self-heal up to 4× on SQL error / empty / degenerate result** → cite the
catalog tables used → statistical guardrail → **LLM verify-critic** ("does this SQL answer *this* question?")
→ interpret with a recommendation. A per-run wall-clock budget and graceful degradation mean it never hangs
or crashes the app; every run's tokens/latency/cost are persisted.

### Condition-specific analysis  (`agent/vocabulary.py`)
Ask about a specific disease in plain English and a deterministic resolver maps the term to the warehouse's
real SNOMED `condition_description` values **before** any SQL runs: *heart attack* → *Myocardial infarction*,
*COPD* → *chronic obstructive bronchitis* **and** *pulmonary emphysema*, plus common abbreviations (MI, HTN,
CKD, IHD) and demonyms/plurals (*diabetics*, *asthmatics*). Both the aggregate and the model paths then filter
on a cohort that exists, so the agent never silently fits a model on an empty set; a condition absent from this
synthetic build (e.g. *flu*) returns an honest clarification naming the closest available conditions, and a
tiny cohort (e.g. *stroke*, n=6) is reported as too small to model rather than forced. It's keyless, and no
user text ever reaches SQL — matching happens over an in-memory copy of the vocabulary, so the injection
surface is zero. Hardened against a **~2,250-scenario adversarial stress test** (0 crashes / false-blocks /
invalid filters throughout); whole-word matching took precision on benign analytical questions to 100%.

### Model families it fits  (`agent/modeling.py`)
| Question shape | Model | Output |
|---|---|---|
| Adjusted risk / effect (binary) | Logistic regression | Adjusted **odds ratios** + 95% CIs |
| Adjusted effect (continuous) | OLS regression | Adjusted **coefficients** + 95% CIs |
| Time-to-event / survival | **Cox PH + Kaplan-Meier** | **Hazard ratios** + KM curves with CI bands |
| Strongest predictors | Random forest importance | Permutation importance, **leakage-safe CV** |
| Forecast / trend | Holt-Winters | Forecast with a horizon-widening band |
| Effect of an intervention | **Cross-fitted AIPW** (doubly-robust) | **ATE** + influence-function CI, positivity trimming |
| "Should we ship variant B?" | A/B experiment | **SHIP / NO-SHIP / INCONCLUSIVE** + lift CI, **BH-FDR** |
| "Is treatment non-inferior?" | **Non-inferiority** | Farrington-Manning test + **Miettinen-Nurminen** CI |
| "How many patients per arm?" | Power / sample-size | n per arm + a sample-size-vs-power curve |
| Two-variable association | Pearson / Welch-t / ANOVA / χ² | Test statistic + p |

**Every model first runs an audited data-engineering pass** (`_prepare`): drop rows with a missing outcome,
drop predictors >10% missing, drop quasi-constant and datetime/high-cardinality/ID-like columns, **pool
sparse categorical levels** into `other` (avoids sparse-category separation), single-impute the rest
(median/mode), and **remove multicollinearity by VIF** *before* fitting. After fitting it surfaces (not
silently fixes) **assumption violations**: events-per-variable, complete/quasi-complete separation,
**proportional hazards** (Schoenfeld residuals), non-linearity (quadratic-term test), and heteroskedasticity
(Breusch-Pagan). The random forest trains in a **scikit-learn `Pipeline`** with imputation fit per fold
(no leakage), 5-fold cross-validation, and class balancing.

### The statistical guardrail  (`agent/guardrails.py`), deterministic, no LLM
Wilson score CIs per group, **pairwise Newcombe difference CIs + two-proportion z-tests corrected with
Benjamini-Hochberg FDR**, skew-aware summaries (median/IQR + bootstrap mean CI), **confounding** and
**Simpson's-paradox** detection, missing-denominator and multiple-comparison flags, and an always-on
synthetic-data note. The LLM may only *phrase* these caveats, never invent or omit them.

### Security, read-only, hardened at the engine  (`agent/warehouse.py`)
The connection is opened `read_only=True` **and** with `enable_external_access=false` + no extension
autoload, so DuckDB rejects, at the engine, every write **and** all filesystem/URL access
(`COPY … TO`, `read_csv`/`read_text`/`read_blob`/`glob`, `ATTACH`, `httpfs`). A **statement denylist**
(write/DDL keywords + file-reading table functions), single-statement enforcement, an outer row cap, and an
append-only **audit log** are defense-in-depth on top. `read_only` alone would not stop file exfiltration , 
this does, and it's covered by tests.

### Monitoring tab  (`app.py`)
A production ops surface: agent usage, success rate, **latency p50/p95**, tokens, and estimated spend; an
activity chart and most-asked questions; **👍/👎 human feedback** with free-text corrections; and **live,
automated data-quality checks** against the warehouse, row volumes, **primary-key uniqueness**,
**referential integrity**, **completeness**, and **metric sanity** (e.g. readmission rate in a plausible band).

### Regulated `.docx` report export  (`agent/report.py`)
Any analysis exports to an industry-format Statistical Analysis Report: a title/approval page (explicit
**DRAFT** status + signature block), synopsis, data sources & analysis population, methods with the **exact
runtime software versions**, numbered results tables (with **mutually-exclusive N/events per category and the
reference level**) and **publication-grade forest / KM / forecast / power figures**, assumption diagnostics,
the guardrail findings, interpretation, and an **ICH-E9 limitations & validation statement**, with a
confidential, page-numbered footer. Rendered and verified in a real Word engine (`python-docx` + `vl-convert`).

### Bring your own data  (`agent/userdata.py`)
Upload a CSV/Excel and the **same agent** runs on it: the table is registered in a session DuckDB, a semantic
catalog is generated from its columns, and the full pipeline (SQL → guardrail → any model, incl. A/B and
non-inferiority) answers questions about *your* data. A non-PHI notice warns to upload non-sensitive data
only, since column names + a few example values reach the LLM.

### Automation
- **Automated dbt model generation** (`agent/model_builder.py`): plain English → the agent drafts a dbt model
  (`{{ ref() }}` over the star schema) **plus schema tests**, writes the `.sql` + `schema.yml`, runs
  `dbt build`, and **self-heals** (reads the failure → rewrites → rebuilds) until it's green.
- **Data-quality auto-fix demo** (`agent/pipeline_healer.py`): a dbt test fails → the agent diagnoses the
  root cause → proposes a fix → rebuild → green.

### The warehouse  (`warehouse/`)
`dbt-core` + `dbt-duckdb` + `dbt_utils`, **26 models across staging → core (star schema) → analytics marts**,
with **104 data tests** and docs on every model. [Synthea](https://github.com/synthetichealth/synthea)
generates **1,139 synthetic patients**, reproducibly (seed 12345). A **semantic catalog** (14 analytics
tables + 6 named metrics with statistical caveats) is auto-generated from the dbt artifacts to make the
warehouse AI-readable, and a deterministic **token-overlap RAG** retrieves over it (no embedding calls).

### Engineering
- **105 keyless `pytest` unit tests** (guardrail stats, SQL validation & security, retrieval, charts, agent
  helpers, modeling) + `ruff` + a coverage gate, run in CI.
- **GitHub Actions CI:** Synthea → DuckDB → `dbt build` (104 tests) → regenerate the catalog → guardrail eval,
  on every push.
- **Eval suite** over one 33-case labeled `GOLD` set: answer accuracy, retrieval precision/recall/MRR
  (keyless), guardrail precision/recall (keyless & deterministic), and an **LLM-as-a-judge** for factual
  consistency (hallucination rate) + relevance.
- **Deployable:** a `Dockerfile` (portable to Cloud Run / Render / Railway / Fly) + `DEPLOY.md`, or Streamlit
  Community Cloud. The OpenAI key is a runtime secret, never baked in.

![Survival analysis, Kaplan-Meier survival curves by group with 95% CI bands and a Cox hazard-ratio forest plot.](assets/survival.png)

![Experiment analysis, a SHIP / NO-SHIP verdict, per-variant conversion with 95% CIs, and a lift forest plot (Newcombe CI + two-proportion z-test, BH-FDR across variants).](assets/ab-experiment.png)

![Non-inferiority, the treatment−control effect with its 95% CI shown against the non-inferiority margin; decided by the Farrington–Manning score test with a Miettinen–Nurminen CI.](assets/ni-noninferiority.png)

![Trial design, a power/sample-size question returns the required n per arm and a sample-size-vs-power curve.](assets/sample-size.png)

---

## Try it / run locally

Prereqs: `git`, [`uv`](https://docs.astral.sh/uv/), a **JDK 17+** (only to regenerate data), and an OpenAI key.

```bash
# 1. Environment (dev = app + dbt; the deployed app installs only requirements.txt)
uv venv --python 3.12 && uv pip install -r requirements-dev.txt

# 2. Generate data → load → build + test the warehouse
cd synthea && java -jar synthea-with-dependencies.jar \
  --exporter.csv.export true --exporter.fhir.export false -p 1000 -s 12345 -cs 12345 Massachusetts && cd ..
.venv/bin/python scripts/load_raw.py
cd warehouse && ../.venv/bin/dbt build --profiles-dir . && ../.venv/bin/dbt docs generate --profiles-dir . && cd ..

# 3. Build the semantic catalog + add your key
.venv/bin/python agent/build_catalog.py
cp agent/.env.example agent/.env      # then put your OPENAI_API_KEY in agent/.env

# 4. Run the agent (CLI) + the checks
.venv/bin/python -m agent.agent "Which conditions are most prevalent in patients 75 and older?"
.venv/bin/python -m agent.agent "How does survival differ for heart attack patients?"   # → Myocardial infarction cohort
.venv/bin/pytest                           # 105 keyless unit tests   (ruff check . to lint)
.venv/bin/python -m agent.guardrail_eval   # guardrail precision/recall (no key)
.venv/bin/python -m agent.eval_retrieval   # retrieval precision/recall/MRR (no key)
.venv/bin/python -m agent.eval             # answer accuracy (needs a key)
.venv/bin/python -m agent.eval_judge       # LLM-as-judge: factual consistency + relevance
.venv/bin/python -m agent.observe          # observability: runs, error rate, latency p50/p95, spend
.venv/bin/python -m agent.model_builder    # autogen a dbt model + tests → build → self-heal
.venv/bin/python -m agent.pipeline_healer  # self-healing demo: dbt test fails → agent fixes → green

# 5. Run the app (Analyze + Monitoring tabs)
.venv/bin/streamlit run app.py
```

**Deploy:** the repo ships a slim `data/healthcare_demo.duckdb` (~32 MB, marts only) so the app runs without
rebuilding the warehouse. Build the container (`docker build -t clinical-agent .`) and run it anywhere, or on
[share.streamlit.io](https://share.streamlit.io) point a new app at `app.py` and add `OPENAI_API_KEY` under
Secrets. See `DEPLOY.md`.

---

## Stack

Python 3.12 · **DuckDB** · **dbt-core + dbt-duckdb + dbt_utils** · **statsmodels** · **scikit-learn** · scipy ·
pandas / numpy · **Altair** (+ vl-convert for print figures) · **python-docx** · **Streamlit** ·
**OpenAI** (`gpt-4o` by default, overridable via `OPENAI_MODEL`) · GitHub Actions · Docker.

---

## Repo layout

```
├── app.py                       Streamlit UI, Analyze + Monitoring tabs
├── agent/
│   ├── agent.py                 the self-healing agent loop
│   ├── modeling.py              inferential models (regression, survival, AIPW causal, A/B, NI, power)
│   ├── guardrails.py            deterministic statistical guardrail (Wilson/Newcombe, FDR, confounding, …)
│   ├── warehouse.py             read-only, engine-hardened, audited SQL execution
│   ├── retrieval.py             token-overlap RAG over the semantic catalog
│   ├── report.py                regulated-style .docx Statistical Analysis Report
│   ├── userdata.py              bring-your-own-data (CSV/Excel → same agent)
│   ├── model_builder.py         autogen + validate a dbt model from plain English
│   ├── pipeline_healer.py       self-healing pipeline demo (dbt test fails → agent fixes)
│   ├── charts.py · llm.py · observe.py · build_catalog.py
│   └── eval*.py · guardrail_eval.py · eval_dataset.py   the eval suite + GOLD set
├── warehouse/                   the dbt project (staging → core → analytics marts + tests + docs)
├── tests/                       105 keyless pytest unit tests
├── scripts/load_raw.py          Synthea CSV → DuckDB raw
├── .github/workflows/ci.yml     Synthea → DuckDB → dbt build → catalog → guardrail eval
├── Dockerfile · DEPLOY.md · GOVERNANCE.md
└── data/healthcare_demo.duckdb  slim marts DB for the deployed demo (committed)
```

---

## Honest limits

- **Synthetic data.** Synthea is structurally realistic but generated from care-process models; magnitudes
  are **illustrative, not empirical**, this demonstrates method, not clinical fact.
- **Exploratory, not confirmatory.** Variable selection is data-driven and CIs/p-values aren't adjusted for
  it; a real regulatory analysis additionally needs a pre-specified SAP and independent double-programming,
  which a qualified biostatistician owns. The `.docx` export says so explicitly.
- **The deployed demo DB samples `fct_observations`** to fit repo limits, no demo/eval question depends on
  it; the full local warehouse has everything.
- The agent is grounded to the catalog and read-only, but it can still write a well-formed query that answers
  a subtly different question than intended, which is exactly why the shown SQL, guardrail, and verify-critic
  keep a human in the loop.
