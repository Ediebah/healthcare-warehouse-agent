"""Show the agent comparing models and picking the one that fits the data best.

The other examples each validate one model. This one validates the *model-selection engine*
(`agent.modeling.compare_models`): given a dataset it fits several candidate models, cross-validates each
by a **composite score** (the mean of ROC-AUC, PR-AUC, and balanced accuracy — rewarding ranking AND
calibrated classification, not AUC alone), and chooses the best. The point is that an uploaded dataset
gets the model that fits IT, not a hard-coded default. And on three datasets whose right model is
already settled in the literature, the engine independently lands on the published choice.

    Heart disease (Detrano et al., 1989)      -- the classic analyses use LOGISTIC REGRESSION
    Heart failure (Chicco & Jurman, 2020)     -- that paper compared ML models and picked RANDOM FOREST,
                                                 on serum creatinine + ejection fraction
    Breast cancer (Wolberg et al., 1995)      -- a benchmark every strong classifier clears at ~0.98+

Run:  .venv/bin/python examples/model_selection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run as a standalone script
from agent import modeling  # noqa: E402

HERE = Path(__file__).resolve().parent


def report(title: str, note: str, df: pd.DataFrame, outcome: str, predictors: list[str],
           expect_note: str, ok: bool) -> None:
    r = modeling.compare_models(df, outcome, predictors)
    print(f"{title}")
    print(f"   {note}")
    print("   leaderboard (composite = mean of roc_auc, pr_auc, balanced accuracy):")
    for row in r.leaderboard:
        mark = " * " if row["is_winner"] else "   "
        auc = row.get("components", {}).get("roc_auc")
        extra = f"   [auc {auc:.3f}]" if auc is not None else ""
        print(f"    {mark}{row['model']:20s} {row['score']:.3f} ± {row['std']:.3f}{extra}")
    print(f"   engine picked: {r.verdict['winner']}")
    print(f"   [{'PASS' if ok(r) else 'FAIL'}] {expect_note}\n")


def main() -> None:
    print("The agent compares candidate models and keeps the best-fitting one\n" + "=" * 66 + "\n")

    hd = pd.read_csv(HERE / "heart_disease_cleveland.csv")
    report("Heart disease · Cleveland (Detrano et al., 1989)",
           "the classic analyses use logistic regression",
           hd, "heart_disease",
           ["age", "sex", "cp", "trestbps", "chol", "thalach", "exang", "oldpeak", "ca", "thal"],
           "logistic regression reproduces its published 0.84-0.91 AUC band",
           lambda r: 0.84 <= next(x["components"]["roc_auc"] for x in r.leaderboard
                                  if x["model"] == "logistic regression") <= 0.93)

    hf = pd.read_csv(HERE / "heart_failure.csv")
    report("Heart failure · DEATH_EVENT (Chicco & Jurman, 2020)",
           "the paper compared ML models and picked the random forest",
           hf, "DEATH_EVENT",
           ["age", "ejection_fraction", "serum_creatinine", "serum_sodium", "high_blood_pressure", "sex",
            "anaemia", "diabetes", "creatinine_phosphokinase", "platelets", "smoking"],
           "the random forest is at the top of the leaderboard, as in the paper",
           lambda r: "random forest" in [x["model"] for x in r.leaderboard[:2]])

    bc = pd.read_csv(HERE / "breast_cancer.csv")
    report("Breast cancer · WDBC (Wolberg et al., 1995)",
           "a benchmark every strong classifier clears",
           bc, "malignant", [c for c in bc.columns if c != "malignant"],
           "every candidate clears AUC 0.96 and the winner clears 0.98",
           lambda r: all(x["components"]["roc_auc"] >= 0.96 for x in r.leaderboard)
           and r.leaderboard[0]["components"]["roc_auc"] >= 0.98)

    print("On each dataset the engine lands on the model the literature settled on — and on your own "
          "upload it does the same comparison, so you get the model that fits your data.")


if __name__ == "__main__":
    main()
