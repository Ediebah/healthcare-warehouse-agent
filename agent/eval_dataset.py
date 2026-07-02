"""GOLD — the single labeled ground-truth dataset shared by every eval.

Each case carries whatever the different evals need:
  * reference_sql / is_rate      → accuracy eval (deterministic ground-truth value)
  * expected_tables              → retrieval precision/recall/MRR
  * expect_clarification         → clarify-gate eval
One dataset, many metrics — so labels never drift between evals.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Gold:
    id: str
    question: str
    category: str
    reference_sql: str = ""
    is_rate: bool = False
    expected_tables: tuple[str, ...] = ()
    expect_clarification: bool = False


GOLD: list[Gold] = [
    # ---- counts ----
    Gold("n_patients", "How many patients are in the warehouse?", "count",
         "select count(*) from dim_patient", expected_tables=("dim_patient",)),
    Gold("n_deceased", "How many patients are deceased?", "count",
         "select count(*) from dim_patient where is_deceased", expected_tables=("dim_patient",)),
    Gold("n_inpatient", "How many inpatient encounters are there?", "count",
         "select count(*) from fct_encounters where encounter_class = 'inpatient'",
         expected_tables=("fct_encounters",)),
    Gold("n_conditions", "How many distinct conditions are recorded?", "count",
         "select count(*) from dim_condition", expected_tables=("dim_condition",)),
    Gold("n_med_orders", "What is the total number of medication orders?", "count",
         "select count(*) from fct_medications", expected_tables=("fct_medications",)),
    Gold("n_encounters", "How many encounters are there in total?", "count",
         "select count(*) from fct_encounters", expected_tables=("fct_encounters",)),
    Gold("n_providers", "How many providers are in the warehouse?", "count",
         "select count(*) from dim_provider", expected_tables=("dim_provider",)),
    Gold("n_female", "How many female patients are there?", "count",
         "select count(*) from dim_patient where gender = 'F'", expected_tables=("dim_patient",)),
    # ---- cost ----
    Gold("avg_cost", "What is the average total claim cost per encounter?", "cost",
         "select round(avg(total_claim_cost), 2) from fct_encounters", expected_tables=("fct_encounters",)),
    Gold("max_cost", "What is the most expensive encounter's total claim cost?", "cost",
         "select round(max(total_claim_cost), 2) from fct_encounters", expected_tables=("fct_encounters",)),
    Gold("avg_inpatient_cost", "What is the average claim cost of an inpatient encounter?", "cost",
         "select round(avg(total_claim_cost), 2) from fct_encounters where encounter_class = 'inpatient'",
         expected_tables=("fct_encounters",)),
    Gold("total_med_cost", "What is the total cost of all medication orders?", "cost",
         "select round(sum(total_cost), 0) from fct_medications", expected_tables=("fct_medications",)),
    # ---- rates ----
    Gold("readmit_rate", "What is the overall 30-day readmission rate as a percent?", "rate",
         "select round(100 * avg(is_30d_readmission::int), 1) from mart_readmissions",
         is_rate=True, expected_tables=("mart_readmissions",)),
    Gold("htn_65_74", "What is the prevalence of hypertension in the 65-74 age group, as a percent?", "rate",
         "select round(prevalence_pct, 1) from mart_condition_prevalence "
         "where condition_description ilike '%hypertension%' and age_group = '65-74'",
         is_rate=True, expected_tables=("mart_condition_prevalence",)),
    Gold("pct_deceased", "What percent of patients are deceased?", "rate",
         "select round(100.0 * avg(is_deceased::int), 1) from dim_patient",
         is_rate=True, expected_tables=("dim_patient",)),
    # ---- filter by clinical name ----
    Gold("n_diabetes_pts", "How many patients have a diabetes diagnosis?", "filter_name",
         "select count(distinct c.patient_id) from fct_conditions c join dim_condition d "
         "using (condition_code) where d.condition_description ilike '%diabetes%'",
         expected_tables=("fct_conditions", "dim_condition")),
    # ---- descriptive stat ----
    Gold("avg_age", "What is the average patient age?", "stat",
         "select round(avg(age), 1) from dim_patient", expected_tables=("dim_patient",)),
    # ---- clarify gate ----
    Gold("ambiguous_trends", "show me the trends", "clarify", expect_clarification=True),
    Gold("out_of_scope", "which treatment is clinically best?", "clarify", expect_clarification=True),
]

ANSWERABLE = [g for g in GOLD if not g.expect_clarification]
CLARIFY = [g for g in GOLD if g.expect_clarification]
