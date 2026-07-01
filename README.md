# AI Data Scientist over a Healthcare Warehouse

An AI agent that does **end-to-end data science over a dbt-modeled healthcare warehouse**: ask a
natural-language question, and the agent retrieves the right schema/metric context, forms a
hypothesis, writes and runs SQL against a modeled warehouse, self-corrects on errors, interprets
the result, and drafts a recommendation **with statistical caveats a generic text-to-SQL bot
misses** (small samples, confounding, multiple comparisons).

Built on synthetic EHR data (zero PHI), so the whole thing is public and reproducible.

> **Status:** Weekend 1 (the data + warehouse substrate) is **complete and tested**. Weekend 2
> (semantic layer + agent + deploy) is the roadmap below.

---

## Architecture

```
┌─────────────┐   generate    ┌───────────────┐   load (all VARCHAR)   ┌──────────────────────┐
│  Synthea    │──────────────▶│  raw CSVs     │───────────────────────▶│  DuckDB  schema: raw │
│ (synthetic  │  1,139 pts    │ (10 tables)   │   scripts/load_raw.py  │  faithful copy       │
│  EHR, Java) │  seed 12345   └───────────────┘                        └──────────┬───────────┘
└─────────────┘                                                                    │ dbt source()
                                                                                   ▼
   ┌───────────────────────────────  dbt project (warehouse/)  ──────────────────────────────┐
   │  staging (10 views)       →   marts/core (6 dim_ + 5 fct_, star schema)                  │
   │  clean · rename · cast        joins · surrogate keys · measures                          │
   │                           →   marts/analytics (readmissions · cost · prevalence)         │
   │  90 tests (not_null · unique · relationships · accepted_values) · docs on every model    │
   └───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │  dbt docs generate → manifest.json + catalog.json
                                                ▼
                    ┌──────────────────────  WEEKEND 2 (planned)  ──────────────────────┐
                    │  semantic catalog (from manifest) → RAG → agent loop:             │
                    │  retrieve → hypothesize → SQL → execute → self-heal → interpret   │
                    │  → recommend → STATISTICAL GUARDRAIL → Streamlit/Next.js deploy    │
                    └───────────────────────────────────────────────────────────────────┘
```

---

## What's built (Weekend 1)

A **tested, documented, dimensionally-modeled healthcare warehouse**.

- **Data:** [Synthea](https://github.com/synthetichealth/synthea) → 1,139 synthetic patients
  (1,000 living + 139 deceased), Massachusetts, reproducible with a fixed seed.
- **Warehouse:** [DuckDB](https://duckdb.org) (local, zero-setup). Raw CSVs loaded as a faithful
  all-`VARCHAR` copy; all typing/logic happens in dbt.
- **Modeling:** [dbt-core](https://docs.getdbt.com) 1.11 + `dbt-duckdb`, `dbt_utils`.
  **24 models, 90 passing tests.**

| layer | models | grain / role |
|---|---|---|
| **staging** (views) | 10 × `stg_*` | 1:1 with each source; clean, rename, cast (no logic) |
| **marts/core** (tables) | `dim_patient` `dim_condition` `dim_medication` `dim_organization` `dim_provider` `dim_payer` | dimensions — one row per entity |
| | `fct_encounters` `fct_conditions` `fct_medications` `fct_procedures` `fct_observations` | facts — one row per event, FKs + measures |
| **marts/analytics** (tables) | `mart_readmissions` | 30-day inpatient readmission flags (13.5% rate) |
| | `mart_cost_by_condition` | cost of diagnosing encounter, by condition |
| | `mart_condition_prevalence` | prevalence by age band (with denominators exposed) |

Sample validated results: 30-day readmission rate **13.5%**; hypertension prevalence rising
monotonically with age (3.9% → 37% → 52% → 49%) — clinically correct signal, not noise.

---

## Quickstart (reproduce from zero)

Prereqs: `git`, [`uv`](https://docs.astral.sh/uv/), and a **JDK 17+** (Synthea's build target).

```bash
# 1. Python env + dbt
uv venv --python 3.12 && uv pip install -r requirements.txt

# 2. Generate synthetic data (see RUNBOOK Step 4 for the full flagged command)
cd synthea && java -jar synthea-with-dependencies.jar \
  --exporter.csv.export true --exporter.fhir.export false -p 1000 -s 12345 -cs 12345 Massachusetts
cd ..

# 3. Load CSVs → DuckDB (raw schema)
.venv/bin/python scripts/load_raw.py

# 4. Build + test the whole warehouse
cd warehouse
../.venv/bin/dbt build --profiles-dir .          # 24 models + 90 tests
../.venv/bin/dbt docs generate --profiles-dir .  # lineage site + semantic-layer artifacts
../.venv/bin/dbt docs serve --profiles-dir .     # browse the DAG at localhost:8080
```

---

## Repo layout

```
dbt_project/
├── README.md            ← you are here
├── CONCEPTS.md          ← the WHY: dbt/warehouse concepts, in build order (re-readable)
├── RUNBOOK.md           ← the HOW: every command + every line of code explained
├── requirements.txt
├── scripts/load_raw.py  ← Synthea CSV → DuckDB raw schema (all VARCHAR)
├── synthea/             ← generator jar + CSV output   (git-ignored, regenerable)
├── data/                ← healthcare.duckdb            (git-ignored, rebuilt by dbt)
└── warehouse/           ← the dbt project
    ├── dbt_project.yml · profiles.yml · packages.yml
    └── models/
        ├── staging/     ← 10 stg_* + _sources.yml + _staging.yml
        └── marts/
            ├── core/     ← 6 dim_* + 5 fct_* + _core.yml
            └── analytics/← 3 mart_* + _analytics.yml
```

Two companion docs make this a learning artifact, not just code:
**[CONCEPTS.md](CONCEPTS.md)** explains every dbt idea as it was introduced;
**[RUNBOOK.md](RUNBOOK.md)** reproduces the build command-by-command and explains every line of code.

---

## What this demonstrates (mapped to the job description)

| Feature | JD language it hits |
|---|---|
| dbt staging → star schema → analytics marts | "deep SQL, data models, warehouse" |
| 90 tests: not_null / unique / relationships / accepted_values | "data quality," "self-healing pipelines" (agent-ready failures) |
| Descriptions on every model/column → `manifest.json` | "make the warehouse AI-readable" (Weekend 2 semantic layer) |
| Honest metric naming + exposed denominators | "interpret results, flag statistical issues" — the biostatistics moat |
| Reproducible, deployable, documented end to end | "builder mentality, own the full lifecycle" |

---

## Honest limits

- **Synthetic data.** Synthea is realistic in structure but generated from care-process models,
  not real patients. Prevalences and costs are plausible, not empirical — good for demonstrating
  *method*, not for clinical conclusions.
- **Cost attribution is intentionally narrow.** `mart_cost_by_condition` sums the *diagnosing
  encounter* cost, not lifetime cost of care (which would require causal attribution). Columns are
  named to say exactly that.
- **Small denominators.** Some demographic bands are small (75+, n=91); prevalence estimates there
  are noisy — the denominator is exposed so consumers (and the Weekend-2 guardrail) can flag it.

---

## Roadmap (Weekend 2)

Semantic catalog generated from `manifest.json` → RAG retrieval → agent loop (retrieve → hypothesize
→ SQL → execute → **self-heal on error** → interpret → recommend) → **statistical guardrail** step
(small-N, confounding, multiple comparisons, base-rate traps) → Streamlit/Next.js deploy + an
accuracy eval set with an LLM-as-judge.
