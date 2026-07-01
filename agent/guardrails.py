"""Statistical guardrail — the biostatistics moat.

Given a result DataFrame (and the question/SQL), return a list of statistical findings a generic
text-to-SQL bot misses: small samples, wide confidence intervals, multiple-comparison risk,
rates reported without denominators, and the synthetic-data caveat.

Everything here is DETERMINISTIC and computed from the data — no LLM. That's the point: the
caveats are reliable and unit-testable, and the LLM only *phrases* them, never invents them.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, asdict

import pandas as pd

MIN_N = 30            # rule-of-thumb for a stable proportion / CLT comfort
TINY_N = 10           # below this, estimates are essentially anecdotal
WIDE_CI_HALFWIDTH = 0.10   # 10 percentage points
MANY_GROUPS = 10      # comparing this many groups inflates false positives

# column-name heuristics
_DENOM_RE = re.compile(r"(^n$|_n$|count|num_|_num|patients|total|denom|sample|cohort|size)", re.I)
_RATE_RE = re.compile(r"(rate|pct|percent|prevalence|proportion|share|ratio)", re.I)
_NUMER_RE = re.compile(r"(patients_with|_with_|numerator|cases|events|successes|readmiss)", re.I)
_COMPARE_RE = re.compile(r"\b(highest|lowest|which|compare|difference|top|rank|vs|versus|between|by|most|least|driver|drives)\b", re.I)


@dataclass
class Finding:
    kind: str          # small_sample | wide_ci | multiple_comparisons | missing_denominator | synthetic_data
    severity: str      # info | warn | caution
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion k/n. Robust for small n (unlike the Wald interval)."""
    if n <= 0:
        return (0.0, 1.0)
    p = min(max(k / n, 0.0), 1.0)   # clamp: guard against mismatched numerator/denominator cols
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _denominator_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in _numeric_cols(df) if _DENOM_RE.search(str(c))]


def _rate_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in _numeric_cols(df) if _RATE_RE.search(str(c))]


def _label_for_row(df: pd.DataFrame, idx) -> str:
    """A human label for a row: join its non-numeric (category) columns."""
    cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not cats:
        return f"row {idx}"
    return " / ".join(str(df.loc[idx, c]) for c in cats[:2])


def analyze(df: pd.DataFrame, question: str = "", sql: str = "") -> list[Finding]:
    findings: list[Finding] = []
    if df is None or len(df) == 0:
        findings.append(Finding("small_sample", "caution", "The query returned no rows — nothing to interpret."))
        return findings

    n_rows = len(df)
    denom_cols = _denominator_cols(df)
    rate_cols = _rate_cols(df)

    # 1) SMALL SAMPLE — per-group denominators below threshold
    if denom_cols:
        dcol = max(denom_cols, key=lambda c: df[c].sum())  # the most "total-like" denominator
        small = df[df[dcol] < MIN_N]
        if len(small) > 0:
            worst = small.nsmallest(min(3, len(small)), dcol)
            examples = "; ".join(f"{_label_for_row(df, i)} (n={int(worst.loc[i, dcol])})" for i in worst.index)
            sev = "caution" if (small[dcol] < TINY_N).any() else "warn"
            findings.append(Finding(
                "small_sample", sev,
                f"{len(small)} group(s) have n < {MIN_N} in `{dcol}` — estimates there are unstable. e.g. {examples}.",
            ))
    elif n_rows < MIN_N and re.search(r"\b(patient|encounter|group|cohort)\b", question, re.I):
        findings.append(Finding(
            "small_sample", "warn",
            f"Only {n_rows} rows returned; if these are the analytic units, the sample is small for firm conclusions.",
        ))

    # 2) WIDE CONFIDENCE INTERVAL on a rate (needs a rate col + a denominator)
    if rate_cols and denom_cols:
        rcol, dcol = rate_cols[0], max(denom_cols, key=lambda c: df[c].sum())
        numer_cols = [c for c in _numeric_cols(df) if _NUMER_RE.search(str(c))]
        widest = None
        for i in df.index:
            n = int(df.loc[i, dcol]) if pd.notna(df.loc[i, dcol]) else 0
            if n <= 0:
                continue
            if numer_cols:
                k = int(df.loc[i, numer_cols[0]])
            else:
                rate = float(df.loc[i, rcol])
                p = rate / 100.0 if rate > 1 else rate      # accept 0-1 or 0-100
                k = round(p * n)
            lo, hi = wilson_ci(k, n)
            half = (hi - lo) / 2
            if widest is None or half > widest[0]:
                widest = (half, i, lo, hi, n)
        if widest and widest[0] > WIDE_CI_HALFWIDTH:
            _, i, lo, hi, n = widest
            findings.append(Finding(
                "wide_ci", "warn",
                f"Widest 95% CI (Wilson) is for {_label_for_row(df, i)}: [{lo*100:.1f}%, {hi*100:.1f}%] "
                f"at n={n} — report the interval, not just the point estimate.",
            ))

    # 3) MULTIPLE COMPARISONS — many groups + a comparative question
    if n_rows >= MANY_GROUPS and _COMPARE_RE.search(question):
        alpha = 0.05
        findings.append(Finding(
            "multiple_comparisons", "warn",
            f"You are comparing/ranking {n_rows} groups. Testing that many inflates false positives; "
            f"use a correction (Bonferroni α/{n_rows} = {alpha/n_rows:.4f}, or Benjamini-Hochberg FDR) "
            f"before calling any single difference 'significant'.",
        ))

    # 4) RATE WITHOUT A DENOMINATOR
    if rate_cols and not denom_cols:
        findings.append(Finding(
            "missing_denominator", "warn",
            f"Rate column(s) {rate_cols} are shown without a denominator/count — a rate is uninterpretable "
            f"(and can be a base-rate trap) without the N it is computed over. Add the group size.",
        ))

    # 5) SYNTHETIC DATA — always true here, always worth stating
    findings.append(Finding(
        "synthetic_data", "info",
        "Data is synthetic (Synthea): structurally realistic but generated from care-process models, "
        "so magnitudes are illustrative, not empirical. Use to demonstrate method, not for clinical claims.",
    ))
    return findings


def render(findings: list[Finding]) -> str:
    icon = {"info": "ℹ️", "warn": "⚠️", "caution": "🛑"}
    return "\n".join(f"{icon.get(f.severity, '•')} [{f.kind}] {f.message}" for f in findings)


if __name__ == "__main__":
    # tiny self-check with a synthetic prevalence-by-age frame
    demo = pd.DataFrame({
        "age_group": ["18-39", "40-64", "65-74", "75+"],
        "patients_with_condition": [12, 147, 62, 4],
        "total_patients_in_age_group": [309, 394, 120, 8],
        "prevalence_pct": [3.88, 37.31, 51.67, 50.0],
    })
    for f in analyze(demo, "Which age group has the highest prevalence of hypertension?"):
        print(render([f]))
