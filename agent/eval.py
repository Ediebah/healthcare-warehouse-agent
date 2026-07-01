"""Accuracy eval: N questions with known answers.

For each case we compute a ground-truth scalar with a hand-written reference SQL (deterministic,
no LLM), then run the agent and check whether the agent's result contains that value. This scores
SQL/answer accuracy objectively. Rate answers are matched across common scalings (fraction vs %).

Run:  .venv/bin/python -m agent.eval        (needs OPENAI_API_KEY in agent/.env)
"""
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd

from .agent import run_analysis
from .warehouse import run_query


@dataclass
class Case:
    id: str
    question: str
    reference_sql: str      # returns a single scalar in row 0, col 0
    is_rate: bool = False   # allow fraction/percent scaling when matching


CASES: list[Case] = [
    Case("n_patients", "How many patients are in the warehouse?",
         "select count(*) from dim_patient"),
    Case("n_deceased", "How many patients are deceased?",
         "select count(*) from dim_patient where is_deceased"),
    Case("n_inpatient", "How many inpatient encounters are there?",
         "select count(*) from fct_encounters where encounter_class = 'inpatient'"),
    Case("n_conditions", "How many distinct conditions are recorded?",
         "select count(*) from dim_condition"),
    Case("n_med_orders", "What is the total number of medication orders?",
         "select count(*) from fct_medications"),
    Case("avg_cost", "What is the average total claim cost per encounter?",
         "select round(avg(total_claim_cost), 2) from fct_encounters"),
    Case("readmit_rate", "What is the overall 30-day readmission rate as a percent?",
         "select round(100 * avg(is_30d_readmission::int), 1) from mart_readmissions", is_rate=True),
    Case("htn_65_74", "What is the prevalence of hypertension in the 65-74 age group, as a percent?",
         "select round(prevalence_pct, 1) from mart_condition_prevalence "
         "where condition_description ilike '%hypertension%' and age_group = '65-74'", is_rate=True),
]


def _reference(case: Case) -> float:
    return float(run_query(case.reference_sql).iloc[0, 0])


def _numeric_cells(df: pd.DataFrame | None) -> list[float]:
    if df is None or len(df) == 0:
        return []
    vals: list[float] = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            vals += [float(v) for v in df[c].dropna().tolist()]
    return vals


def _matches(cells: list[float], ref: float, is_rate: bool, rtol: float = 0.02) -> bool:
    targets = [ref]
    if is_rate:
        targets += [ref / 100.0, ref * 100.0]     # fraction vs percent
    for t in targets:
        for v in cells:
            if abs(v - t) <= max(abs(t) * rtol, 0.5):
                return True
    return False


def main() -> int:
    rows, passed = [], 0
    for case in CASES:
        ref = _reference(case)
        res = run_analysis(case.question)
        ok = res.error is None and _matches(_numeric_cells(res.dataframe), ref, case.is_rate)
        passed += ok
        tries = len(res.attempts) or 0
        rows.append((case.id, "PASS" if ok else "FAIL", ref, res.n_rows, tries, res.error or ""))
        print(f"  {'✅' if ok else '❌'} {case.id:14} ref={ref:<10.2f} rows={res.n_rows:<4} "
              f"tries={tries} {('· ' + res.error) if res.error else ''}")

    acc = passed / len(CASES)
    print(f"\nAccuracy: {passed}/{len(CASES)} = {acc:.0%}")
    return 0 if acc >= 0.75 else 1


if __name__ == "__main__":
    raise SystemExit(main())
