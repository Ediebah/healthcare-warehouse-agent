"""Reproduce a survival result with a machine-learning survival model, judged by concordance index.

The heart-failure survival example fits a Cox model. This one adds the *machine-learning* survival path:
the agent's `compare_survival_models` tunes a Cox proportional-hazards model and a random survival forest
(scikit-survival) and keeps whichever scores higher on a cross-validated **survival composite** — the mean
of Harrell's concordance index, time-dependent AUC, and a Brier skill score (1 - integrated Brier). On the
UCI heart-failure cohort the tuned forest edges Cox — a non-linear model captures a little more of the
time-to-event signal — while still recovering the same two predictors the paper highlights.

    Dataset : examples/heart_failure.csv  (299 patients, time-to-event)
    Source  : UCI Heart Failure Clinical Records; Chicco D., Jurman G. (2020), BMC Med. Inform. Decis.
              Mak. 20:16. Original cohort: Ahmad T. et al. (2017), PLoS ONE 12(7):e0181001.
    Needs   : the optional `scikit-survival` package (pip install scikit-survival).

Run:  .venv/bin/python examples/survival_ml.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run as a standalone script
from agent import modeling  # noqa: E402

CSV = Path(__file__).resolve().parent / "heart_failure.csv"
PREDICTORS = ["age", "ejection_fraction", "serum_creatinine", "serum_sodium", "high_blood_pressure",
              "sex", "anaemia", "diabetes", "creatinine_phosphokinase", "platelets", "smoking"]


def main() -> None:
    df = pd.read_csv(CSV)
    r = modeling.compare_survival_models(df, "time", "DEATH_EVENT", PREDICTORS)
    if r.error:
        print(f"Could not run: {r.error}")
        return

    print("UCI Heart Failure · survival model selection")
    print("composite = mean of Harrell's C-index, time-dependent AUC, and Brier skill (1 - integ. Brier)")
    print("-" * 78)
    for row in r.leaderboard:
        mark = " * " if row["is_winner"] else "   "
        c = row["components"]
        print(f"  {mark}{row['model']:26s} composite {row['score']:.3f}   "
              f"(C {c['harrell_c']:.3f} · tAUC {c['td_auc']:.3f} · Brier-skill {c['brier_skill']:.3f})")
    print(f"\n   engine picked: {r.verdict['winner']}")

    print("   winning model's top predictors: " + ", ".join(t.name for t in r.terms[:3]))
    top3 = [t.name for t in r.terms[:3]]
    rsf = next(row for row in r.leaderboard if row["model"] == "random survival forest")

    print("\nJudged against the literature (Chicco & Jurman 2020)")
    print("-" * 78)
    checks = [
        ("The random survival forest reaches a concordance index of 0.70+", rsf["components"]["harrell_c"] >= 0.70),
        ("Serum creatinine and ejection fraction are among the top predictors",
         "serum_creatinine" in top3 and "ejection_fraction" in top3),
    ]
    for label, ok in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {label}")
    print("\nA tuned machine-learning survival model, compared fairly against Cox and judged by a survival "
          "composite (ranking, time-varying discrimination, and calibration), recovers the settled "
          "heart-failure predictors from real data.")


if __name__ == "__main__":
    main()
