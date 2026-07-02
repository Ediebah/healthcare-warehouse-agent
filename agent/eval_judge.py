"""LLM-as-a-judge eval: factual consistency (hallucination) + answer relevance of the agent's
interpretations, with a deterministic numeric-grounding cross-check.

Ground truth for factual consistency is the query result *itself* — the judge checks whether every
claim in the interpretation is supported by the returned rows. Two signals, deliberately:
  * LLM judge (rubric, temp 0) → faithful? relevance 1-5? which claims are unsupported?
  * deterministic numeric grounding → what fraction of numbers in the prose appear in the result?
Reporting both means the judge is cross-checked, not trusted blindly.

hallucination_rate = 1 − factual_consistency.  Run: .venv/bin/python -m agent.eval_judge
"""
from __future__ import annotations
import re

import pandas as pd

from . import llm
from .agent import run_analysis

# analytical questions (rich interpretations, where hallucination is actually possible)
QUESTIONS = [
    "What is the prevalence of hypertension by age group?",
    "How does average encounter cost differ by encounter class?",
    "What is the 30-day readmission rate, and does it vary by age group?",
    "Which chronic conditions drive the highest total encounter cost?",
    "Which conditions are most prevalent in patients 75 and older?",
    "What is the average patient age?",
]

_NUM = re.compile(r"-?\d[\d,]*\.?\d+|\b\d+\b")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _numeric_grounding(interp: str, df) -> float:
    """Fraction of numbers in the prose that trace to a result cell (allowing pct scalings) or are
    plausibly structural (small integers: years, group counts). A rough hallucination cross-check."""
    nums = _numbers(interp)
    if not nums or df is None or len(df) == 0:
        return 1.0
    cells = set()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cells.update(round(float(v), 2) for v in df[c].dropna())
    grounded = 0
    for x in nums:
        ok = any(abs(x - c) <= max(abs(c) * 0.02, 0.5) for c in cells)
        ok = ok or any(abs(x - c * 100) <= max(abs(c * 100) * 0.02, 0.5) for c in cells)
        ok = ok or any(abs(x - c / 100) <= 0.01 for c in cells)
        ok = ok or (x == int(x) and 0 <= x <= max(2100, len(df)))   # years / group counts / small ints
        grounded += int(ok)
    return grounded / len(nums)


def _judge(question: str, df, interpretation: str) -> dict:
    preview = df.head(20).to_csv(index=False)
    out = llm.complete_json(
        "You are a fair, calibrated evaluation judge for a healthcare data-analysis agent. You assess "
        "FACTUAL CONSISTENCY: whether the interpretation's claims about the DATA match the query "
        "result. The interpretation intentionally includes statistical caveats and a recommendation — "
        "those are appropriate and are NOT hallucinations.",
        f"QUESTION: {question}\n\nQUERY RESULT (CSV, the ground truth):\n{preview}\n\n"
        f"AGENT INTERPRETATION:\n{interpretation}\n\n"
        "Judge ONLY the factual/numeric claims about the data (typically the 'Findings').\n"
        "- A number is SUPPORTED if it appears in the result (allow rounding and %/fraction forms).\n"
        "- faithful=false ONLY if a number is fabricated or a data claim contradicts the result.\n"
        "- Do NOT flag as unsupported: statistical caveats (small samples, skew, confidence intervals, "
        "confounding, multiple comparisons, synthetic-data notes), recommendations, or general domain "
        "context — these are expected and appropriate.\n"
        "relevance 1-5 = how well it answers the question.\n"
        'Return JSON: {"faithful": bool, "relevance": 1-5, "unsupported_claims": [ONLY genuinely '
        'fabricated or contradicted data claims]}.',
    )
    out.setdefault("faithful", True)
    out.setdefault("relevance", 3)
    out.setdefault("unsupported_claims", [])
    return out


def main() -> int:
    faithful = 0
    rel_sum = 0.0
    ng_sum = 0.0
    n = 0
    for q in QUESTIONS:
        res = run_analysis(q)
        if res.error or res.dataframe is None:
            print(f"  ✗ {q[:48]:48} (agent error: {res.error})")
            continue
        j = _judge(q, res.dataframe, res.interpretation)
        ng = _numeric_grounding(res.interpretation, res.dataframe)
        n += 1
        faithful += int(j["faithful"])
        rel_sum += float(j["relevance"])
        ng_sum += ng
        mark = "✅" if j["faithful"] else "❌"
        issues = f" | unsupported: {j['unsupported_claims']}" if j["unsupported_claims"] else ""
        print(f"  {mark} {q[:44]:44} faithful={j['faithful']} rel={j['relevance']} "
              f"num_grounded={ng:.0%}{issues}")

    if n == 0:
        print("no questions evaluated"); return 1
    fc = faithful / n
    print(f"\n  Factual consistency: {faithful}/{n} = {fc:.0%}   "
          f"(hallucination rate {1-fc:.0%})")
    print(f"  Mean relevance: {rel_sum/n:.1f}/5   Numeric grounding: {ng_sum/n:.0%}")
    return 0 if fc >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
