"""Evaluate a model beyond a single score: decision curve analysis and failure analysis.

A high AUC does not mean a model is clinically useful or trustworthy. This example runs the agent's two
evaluation lenses on the UCI heart-disease data, using out-of-fold predictions (no optimism):

  * decision_curve   — net benefit across decision thresholds vs treating everyone / no one
                       (Vickers & Elkin 2006): does acting on the model actually help?
  * failure_analysis — calibration by risk decile, the false-positive / false-negative split, and the
                       patient subgroup the model misclassifies most: where does it fail?

Run:  .venv/bin/python examples/model_evaluation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run as a standalone script
from agent import modeling  # noqa: E402

CSV = Path(__file__).resolve().parent / "heart_disease_cleveland.csv"
PREDICTORS = ["age", "sex", "cp", "trestbps", "chol", "thalach", "exang", "oldpeak", "ca", "thal"]


def main() -> None:
    df = pd.read_csv(CSV)

    print("Decision curve analysis — is the model worth acting on?")
    print("-" * 66)
    dca = modeling.decision_curve(df, "heart_disease", PREDICTORS, model="auto")
    print(f"   {dca.fit_stat}")
    for s in dca.series:
        if round(s["threshold"], 2) in (0.10, 0.20, 0.30, 0.50):
            beats = "yes" if s["nb_model"] > max(s["nb_all"], s["nb_none"]) else "no"
            print(f"     @ threshold {s['threshold']:.0%}: net benefit {s['nb_model']:.3f}  "
                  f"(treat-all {s['nb_all']:.3f}) — beats both: {beats}")

    print("\nFailure analysis — where does it fail?")
    print("-" * 66)
    fa = modeling.failure_analysis(df, "heart_disease", PREDICTORS, model="auto")
    v = fa.verdict
    print(f"   {v['false_positives']} false positives, {v['false_negatives']} false negatives (0.5 cut); "
          f"max calibration gap {v['max_calibration_gap']:.0%}")
    w = v["worst_segment"]
    print(f"   worst subgroup: {w['feature']}={w['level']} — misclassified {w['error_rate']:.0%} (n={w['n']})")
    print("   calibration (predicted vs observed risk, by decile):")
    for s in fa.series:
        print(f"     predicted {s['predicted']:.2f}  ->  observed {s['observed']:.2f}   (n={s['n']})")

    print("\nChecks")
    print("-" * 66)
    at20 = next(s for s in dca.series if s["threshold"] == 0.20)
    checks = [
        ("The model adds net benefit over treat-all/none at a 20% threshold",
         at20["nb_model"] > at20["nb_all"] and at20["nb_model"] > 0),
        ("The model is reasonably calibrated (max gap < 25%)", v["max_calibration_gap"] < 0.25),
        ("A worst-performing subgroup is identified", w is not None),
    ]
    for label, ok in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {label}")


if __name__ == "__main__":
    main()
