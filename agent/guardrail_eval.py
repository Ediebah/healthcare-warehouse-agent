"""Guardrail precision/recall eval.

Each case is a synthetic result DataFrame with KNOWN statistical properties, labelled with the
finding kinds that SHOULD fire (expected) and MUST NOT fire (forbidden). We run the guardrail and
score it — measuring both recall (does it catch real issues?) and precision (does it avoid false
alarms?). This is deterministic and needs no API key.

Run:  .venv/bin/python -m agent.guardrail_eval
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .guardrails import analyze

ALWAYS_ON = {"synthetic_data"}


@dataclass
class Case:
    name: str
    df: pd.DataFrame
    question: str
    expected: set = field(default_factory=set)     # kinds that should fire
    forbidden: set = field(default_factory=set)     # kinds that must not fire
    severity: dict = field(default_factory=dict)    # optional {kind: expected_severity}


def _skewed_costs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"encounter_cost": rng.lognormal(7.0, 1.0, size=n)})


CASES = [
    Case("tiny_group",
         pd.DataFrame({"age_group": ["18-39", "40-64", "65-74", "75+"],
                       "patients_with_condition": [12, 147, 62, 4],
                       "total_patients_in_age_group": [309, 394, 120, 8],
                       "prevalence_pct": [3.88, 37.31, 51.67, 50.0]}),
         "Which age group has the highest prevalence of hypertension?",
         expected={"small_sample"}, forbidden={"simpsons"}),

    Case("clear_difference",
         pd.DataFrame({"cohort": ["A", "B"], "cases": [200, 50], "total": [1000, 1000]}),
         "Compare the event rate between cohort A and B",
         expected={"contrasts"}, forbidden={"small_sample"}, severity={"contrasts": "warn"}),

    Case("no_difference",
         pd.DataFrame({"cohort": ["A", "B"], "cases": [100, 105], "total": [1000, 1000]}),
         "Compare the event rate between cohort A and B",
         expected={"contrasts"}, forbidden={"small_sample"}, severity={"contrasts": "info"}),

    Case("skewed_cost",
         _skewed_costs(),
         "What do encounter costs look like?",
         expected={"skew"}, forbidden={"contrasts", "small_sample", "missing_denominator", "confounding"}),

    Case("rate_no_denominator",
         pd.DataFrame({"age_group": ["18-39", "40-64", "65-74", "75+"],
                       "prevalence_pct": [3.9, 37.3, 51.7, 49.5]}),
         "Prevalence of hypertension by age group",
         expected={"missing_denominator"}, forbidden={"contrasts"}),

    Case("confounded_comparison",
         pd.DataFrame({"age_group": ["18-39", "40-64", "65-74", "75+"],
                       "readmit_cases": [8, 40, 55, 30],
                       "n_patients": [400, 500, 250, 180],
                       "readmission_rate": [2.0, 8.0, 22.0, 16.7]}),
         "Does the readmission rate differ by age group?",
         expected={"confounding"}, forbidden={"missing_denominator"}),

    Case("simpsons_paradox",
         pd.DataFrame({"treatment": ["A", "B", "A", "B"],
                       "severity": ["mild", "mild", "severe", "severe"],
                       "cases": [81, 234, 192, 55],
                       "total": [87, 270, 263, 80]}),
         "Which treatment has the higher success rate?",
         expected={"simpsons"}, forbidden={"confounding"}),

    Case("clean_balanced_no_overflag",
         pd.DataFrame({"payer": ["Aetna", "Cigna"], "members": [500, 510], "total_eligible": [1000, 1000]}),
         "List members and eligibility for each payer",
         expected=set(),
         forbidden={"small_sample", "confounding", "skew", "simpsons", "missing_denominator", "multiple_comparisons"}),
]


def main() -> int:
    tp = fn = fp = 0
    all_pass = True
    for c in CASES:
        produced = {f.kind: f.severity for f in analyze(c.df, c.question) if f.kind not in ALWAYS_ON}
        pset = set(produced)
        hits = c.expected & pset
        misses = c.expected - pset
        false_alarms = c.forbidden & pset
        sev_ok = all(produced.get(k) == v for k, v in c.severity.items() if k in produced)
        ok = not misses and not false_alarms and sev_ok
        all_pass &= ok
        tp += len(hits); fn += len(misses); fp += len(false_alarms)
        flag = "✅" if ok else "❌"
        detail = []
        if misses: detail.append(f"missed={sorted(misses)}")
        if false_alarms: detail.append(f"false_alarm={sorted(false_alarms)}")
        if not sev_ok: detail.append(f"severity≠{c.severity}")
        print(f"  {flag} {c.name:26} produced={sorted(pset)}  {' '.join(detail)}")

    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    print(f"\n  Precision={prec:.0%}  Recall={rec:.0%}  (TP={tp} FP={fp} FN={fn})")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
