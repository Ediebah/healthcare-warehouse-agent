"""GOLD — the single labeled ground-truth dataset shared by every eval.

Each case carries whatever the different evals need:
  * reference_sql / is_rate      → accuracy eval (deterministic ground-truth value)
  * expected_tables              → retrieval precision/recall/MRR
  * expect_clarification         → clarify-gate eval
One dataset, many metrics — so labels never drift between evals.
"""
from __future__ import annotations

from dataclasses import dataclass


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
    # ---- more counts ----
    Gold("n_organizations", "How many healthcare organizations are there?", "count",
         "select count(*) from dim_organization", expected_tables=("dim_organization",)),
    Gold("n_payers", "How many insurance payers are there?", "count",
         "select count(*) from dim_payer", expected_tables=("dim_payer",)),
    Gold("n_male", "How many male patients are there?", "count",
         "select count(*) from dim_patient where gender = 'M'", expected_tables=("dim_patient",)),
    Gold("n_meds_distinct", "How many distinct medications are recorded?", "count",
         "select count(*) from dim_medication", expected_tables=("dim_medication",)),
    Gold("n_emergency", "How many emergency encounters are there?", "count",
         "select count(*) from fct_encounters where encounter_class = 'emergency'",
         expected_tables=("fct_encounters",)),
    Gold("n_procedures", "How many procedures were performed?", "count",
         "select count(*) from fct_procedures", expected_tables=("fct_procedures",)),
    # ---- more cost ----
    Gold("avg_med_cost", "What is the average total cost of a medication order?", "cost",
         "select round(avg(total_cost), 2) from fct_medications", expected_tables=("fct_medications",)),
    Gold("avg_procedure_cost", "What is the average base cost of a procedure?", "cost",
         "select round(avg(base_cost), 2) from fct_procedures", expected_tables=("fct_procedures",)),
    Gold("total_encounter_cost", "What is the total claim cost across all encounters?", "cost",
         "select round(sum(total_claim_cost), 0) from fct_encounters", expected_tables=("fct_encounters",)),
    # ---- more rates ----
    Gold("pct_female", "What percent of patients are female?", "rate",
         "select round(100.0 * avg((gender = 'F')::int), 1) from dim_patient",
         is_rate=True, expected_tables=("dim_patient",)),
    Gold("pct_inpatient", "What percent of encounters are inpatient?", "rate",
         "select round(100.0 * avg((encounter_class = 'inpatient')::int), 1) from fct_encounters",
         is_rate=True, expected_tables=("fct_encounters",)),
    # ---- more filter-by-name ----
    Gold("n_htn_pts", "How many patients have hypertension?", "filter_name",
         "select count(distinct c.patient_id) from fct_conditions c join dim_condition d "
         "using (condition_code) where d.condition_description ilike '%hypertension%'",
         expected_tables=("fct_conditions", "dim_condition")),
    # ---- more descriptive stat ----
    Gold("oldest_age", "What is the age of the oldest patient?", "stat",
         "select max(age) from dim_patient", expected_tables=("dim_patient",)),
    Gold("avg_enc_duration", "What is the average encounter duration in minutes?", "stat",
         "select round(avg(duration_minutes), 1) from fct_encounters", expected_tables=("fct_encounters",)),

    # ---- clarify gate ----
    Gold("ambiguous_trends", "show me the trends", "clarify", expect_clarification=True),
    Gold("out_of_scope", "which treatment is clinically best?", "clarify", expect_clarification=True),
]

ANSWERABLE = [g for g in GOLD if not g.expect_clarification]
CLARIFY = [g for g in GOLD if g.expect_clarification]
