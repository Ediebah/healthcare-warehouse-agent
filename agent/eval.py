"""Accuracy eval at grade: categorized known-answer questions + clarify-gate + caveat faithfulness.

For each answerable case we compute a ground-truth scalar with a hand-written reference SQL
(deterministic, no LLM), run the agent, and check the agent's result contains that value. Clarify
cases assert the agent asks for clarification instead of guessing. We also measure "caveat
faithfulness" — whether the interpretation actually reflects the guardrail's findings — and log a
summary row to agent/eval_history.jsonl for regression tracking.

Run:  .venv/bin/python -m agent.eval        (needs OPENAI_API_KEY in agent/.env)
Companion: agent.guardrail_eval (guardrail precision/recall, no key needed).
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pandas as pd

from .agent import run_analysis
from .eval_dataset import GOLD as CASES
from .warehouse import run_query

HISTORY = Path(__file__).resolve().parent / "eval_history.jsonl"

# keywords that indicate the interpretation honored a given guardrail finding kind
_FAITHFUL_KEYS = {
    "small_sample": ("small", "sample", "n=", "unstable", "few"),
    "confounding": ("confound", "adjust", "unadjusted", "standardiz", "covariate"),
    "contrasts": ("confidence", "ci", "interval", "difference", "significant"),
    "skew": ("skew", "median", "iqr", "distribution"),
    "missing_denominator": ("denominator", "base rate", "sample size", "group size"),
    "wide_ci": ("confidence", "interval", "ci", "wide"),
    "multiple_comparisons": ("multiple", "correction", "bonferroni", "fdr", "false positive"),
}


def _reference(case) -> float:
    return float(run_query(case.reference_sql).iloc[0, 0])


def _answer_cells(df, allow_multi: bool = False) -> list[float]:
    """The scalar answer as a list of candidate cells, or [] if it isn't a single-row answer.

    Every GOLD question has scalar ground truth, so a correct answer is a single row. Non-rate answers
    must be a single numeric column (scanning a multi-column/multi-row table would let a query that
    merely *contains* the value pass). Rate answers are the exception: the statistical guardrail
    REQUIRES a rate query to also select its numerator and denominator, so a correct rate result has
    three numeric columns — `allow_multi=True` returns all of them and lets `_matches` find the rate.
    """
    if df is None or len(df) != 1:
        return []
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric or (len(numeric) != 1 and not allow_multi):
        return []
    return [float(df[c].iloc[0]) for c in numeric if pd.notna(df[c].iloc[0])]


def _matches(cells, ref, is_rate, rtol=0.02) -> bool:
    if not cells:
        return False
    # A percentage answer may be reported as a percent (ref) or the equivalent fraction (ref/100);
    # never as ref*100. Non-rate answers must match the reference directly.
    targets = [ref, ref / 100.0] if is_rate else [ref]
    return any(abs(v - t) <= max(abs(t) * rtol, 0.05) for t in targets for v in cells)


def _faithful(res) -> bool | None:
    """Did the interpretation reflect the guardrail's (non-synthetic) findings? None if nothing to check."""
    kinds = [f.kind for f in res.findings if f.kind != "synthetic_data"]
    if not kinds or not res.interpretation:
        return None
    text = res.interpretation.lower()
    hit = sum(any(k in text for k in _FAITHFUL_KEYS.get(kind, ())) for kind in kinds)
    return hit >= (len(kinds) + 1) // 2   # true majority (ceil) of flagged concerns reflected


def main() -> int:
    by_cat: dict[str, list[bool]] = {}
    faithful_hits, faithful_total = 0, 0
    rows = []
    for c in CASES:
        if c.expect_clarification:
            res = run_analysis(c.question)
            ok = bool(res.clarification) and res.dataframe is None
            mark = "✅" if ok else "❌"
            print(f"  {mark} {c.id:18} [{c.category}] "
                  f"{'asked for clarification' if ok else 'did NOT clarify (guessed)'}")
        else:
            ref = _reference(c)
            res = run_analysis(c.question)
            ok = res.error is None and _matches(
                _answer_cells(res.dataframe, allow_multi=c.is_rate), ref, c.is_rate)
            f = _faithful(res)
            if f is not None:
                faithful_total += 1
                faithful_hits += int(f)
            print(f"  {'✅' if ok else '❌'} {c.id:18} [{c.category:11}] ref={ref:<11.2f} "
                  f"rows={res.n_rows:<3} tries={len(res.attempts)}")
        by_cat.setdefault(c.category, []).append(ok)
        rows.append({"id": c.id, "category": c.category, "ok": ok})

    total = [ok for v in by_cat.values() for ok in v]
    acc = sum(total) / len(total)
    print("\n  by category:")
    for cat, oks in sorted(by_cat.items()):
        print(f"    {cat:12} {sum(oks)}/{len(oks)}")
    faith = (faithful_hits / faithful_total) if faithful_total else 1.0
    print(f"\n  Accuracy: {sum(total)}/{len(total)} = {acc:.0%}   "
          f"Caveat-faithfulness: {faithful_hits}/{faithful_total} = {faith:.0%}")

    # regression log
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    with HISTORY.open("a") as fh:
        fh.write(json.dumps({"ts": stamp, "accuracy": round(acc, 3),
                             "faithfulness": round(faith, 3), "n": len(total)}) + "\n")
    print(f"  logged → {HISTORY.name}")
    return 0 if acc >= 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(main())
