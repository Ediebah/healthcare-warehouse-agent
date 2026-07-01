"""Statistical guardrail — the biostatistics moat.

Given a result DataFrame (+ question/SQL), return statistical findings a generic text-to-SQL
bot misses. Everything here is DETERMINISTIC and computed from the data — no LLM. The LLM may
only *phrase* these caveats, never invent or omit them.

What it does (v2 — inference, not just fragility flags):
  * SMALL SAMPLE      — per-group denominators below a stability threshold.
  * WILSON CI         — proper interval for each group's proportion (honest at small n).
  * CONTRASTS + FDR   — pairwise risk differences with Newcombe CIs and two-proportion z-tests,
                        corrected across the family with Benjamini-Hochberg. Answers the real
                        question ("is A different from B?"), not just per-group point estimates.
  * SKEW-AWARE        — for a mean/measure, flag right-skew; report median/IQR + bootstrap mean CI.
  * CONFOUNDING       — warn when an outcome is compared across groups unadjusted; name plausible
                        confounders and suggest standardization / stratification.
  * SIMPSON'S PARADOX — when the result is stratified, check whether the marginal ordering reverses.
  * MISSING DENOM     — a rate shown without its denominator (base-rate trap).
  * SYNTHETIC DATA    — always.

Pure Python + numpy (no scipy): normal CDF via math.erf, Wilson/Newcombe/BH implemented inline.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# thresholds
MIN_N = 30
TINY_N = 10
WIDE_CI_HALFWIDTH = 0.10
MANY_GROUPS = 10
ALPHA = 0.05
SKEW_MODERATE = 1.0
SKEW_HIGH = 2.0

# column-name heuristics
_DENOM_RE = re.compile(r"(total|denom|cohort|sample|population|_size|(^|_)n($|_)|num_patients|total_patients)", re.I)
_NUMER_RE = re.compile(r"(patients_with|_with_|numerator|cases|events|readmit|readmiss|affected|positive)", re.I)
_RATE_RE = re.compile(r"(rate|pct|percent|prevalence|proportion|share|ratio)", re.I)
_MEAN_RE = re.compile(r"(^avg_|_avg$|average|^mean_|_mean$|median)", re.I)
_MEASURE_RE = re.compile(r"(cost|amount|charge|price|duration|minutes|value|income|expense|\blos\b|days_)", re.I)
_DEMO_RE = re.compile(r"(age_group|age_band|\bage\b|gender|\bsex\b|race|ethnicity|marital)", re.I)
_COMPARE_RE = re.compile(r"\b(highest|lowest|which|compare|difference|differ|top|rank|vs|versus|between|most|least|driver|drives|by )\b", re.I)


@dataclass
class Finding:
    kind: str
    severity: str      # info | warn | caution
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


# ───────────────────────── statistics primitives ─────────────────────────
def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion k/n (clamped; robust at small n)."""
    if n <= 0:
        return (0.0, 1.0)
    p = min(max(k / n, 0.0), 1.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> tuple[float, float, float]:
    """Newcombe method-10 CI for the difference p1-p2 (uses the two Wilson intervals)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci(k1, n1, z)
    l2, u2 = wilson_ci(k2, n2, z)
    d = p1 - p2
    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, max(-1.0, lower), min(1.0, upper)


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided pooled two-proportion z-test p-value."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return 2 * (1 - _norm_cdf(abs(z)))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH step-up adjusted q-values, returned in the original order."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        q[i] = min(prev, 1.0)
    return q


def skewness(vals) -> float:
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 3:
        return 0.0
    s = a.std(ddof=0)
    return 0.0 if s == 0 else float(((a - a.mean()) ** 3).mean() / s ** 3)


def bootstrap_mean_ci(vals, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# ───────────────────────── column detection ─────────────────────────
def _num_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _cat_cols(df):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def _match(cols, rx):
    return [c for c in cols if rx.search(str(c))]


def _label(df, idx) -> str:
    cats = _cat_cols(df)
    return " / ".join(str(df.loc[idx, c]) for c in cats[:2]) if cats else f"row {idx}"


# ───────────────────────── individual checks ─────────────────────────
def _check_small_sample(df, findings):
    denom = [c for c in _num_cols(df) if _DENOM_RE.search(str(c))]
    if not denom:
        return None
    dcol = max(denom, key=lambda c: df[c].sum())
    small = df[df[dcol] < MIN_N]
    if len(small) == 0:
        return dcol
    worst = small.nsmallest(min(3, len(small)), dcol)
    ex = "; ".join(f"{_label(df, i)} (n={int(worst.loc[i, dcol])})" for i in worst.index)
    sev = "caution" if (small[dcol] < TINY_N).any() else "warn"
    findings.append(Finding("small_sample", sev,
        f"{len(small)} group(s) have n < {MIN_N} in `{dcol}` — estimates unstable. e.g. {ex}."))
    return dcol


def _check_contrasts(df, findings):
    """Pairwise proportion contrasts + Newcombe CIs + BH-FDR, when numerator+denominator present."""
    numer = _match(_num_cols(df), _NUMER_RE)
    denom = [c for c in _num_cols(df) if _DENOM_RE.search(str(c))]
    if not numer or not denom or len(df) < 2 or len(df) > 60:
        return
    kcol, ncol = numer[0], max(denom, key=lambda c: df[c].sum())
    grp = [(i, int(df.loc[i, kcol]), int(df.loc[i, ncol])) for i in df.index if df.loc[i, ncol] > 0]
    if len(grp) < 2:
        return
    pairs, pvals = [], []
    for a in range(len(grp)):
        for b in range(a + 1, len(grp)):
            i, k1, n1 = grp[a]
            j, k2, n2 = grp[b]
            d, lo, hi = newcombe_diff_ci(k1, n1, k2, n2)
            p = two_proportion_p(k1, n1, k2, n2)
            pairs.append((i, j, d, lo, hi, p))
            pvals.append(p)
    q = benjamini_hochberg(pvals)
    sig = [(pairs[t], q[t]) for t in range(len(pairs)) if q[t] < ALPHA]
    if not sig:
        findings.append(Finding("contrasts", "info",
            f"No pairwise difference among {len(grp)} groups is significant after "
            f"Benjamini-Hochberg FDR (q≥{ALPHA}); observed gaps are within noise."))
        return
    sig.sort(key=lambda x: abs(x[0][2]), reverse=True)
    (i, j, d, lo, hi, p), qv = sig[0]
    findings.append(Finding("contrasts", "warn",
        f"{len(sig)}/{len(pairs)} pairwise contrasts survive BH-FDR (q<{ALPHA}). Largest: "
        f"{_label(df, i)} vs {_label(df, j)}, risk difference {d*100:+.1f}pp "
        f"(95% CI [{lo*100:.1f}, {hi*100:.1f}], q={qv:.3f}). Report differences with CIs, not raw ranks."))


def _check_skew(df, findings):
    """Skew-aware: raw measure column → skewness + median/IQR + bootstrap mean CI; aggregated mean → note."""
    measures = [c for c in _num_cols(df)
                if _MEASURE_RE.search(str(c)) and not _DENOM_RE.search(str(c)) and not _RATE_RE.search(str(c))]
    # case 1: enough raw rows to assess distribution
    for c in measures:
        vals = df[c].dropna().to_numpy()
        if len(vals) >= 20:
            sk = skewness(vals)
            if abs(sk) >= SKEW_MODERATE:
                med = float(np.median(vals))
                q1, q3 = np.percentile(vals, [25, 75])
                lo, hi = bootstrap_mean_ci(vals)
                sev = "warn" if abs(sk) >= SKEW_HIGH else "info"
                findings.append(Finding("skew", sev,
                    f"`{c}` is right-skewed (skewness {sk:.1f}); the mean overstates the typical value. "
                    f"Median={med:,.0f} (IQR {q1:,.0f}–{q3:,.0f}); bootstrap 95% CI for the mean "
                    f"[{lo:,.0f}, {hi:,.0f}]. Prefer median/IQR for skewed cost/LOS data."))
            return
    # case 2: an aggregated mean with no dispersion shown
    mean_cols = _match(_num_cols(df), _MEAN_RE)
    if mean_cols and not any(_match(_num_cols(df), re.compile(r"(std|sd|median|iqr|p25|p75|min|max|var)", re.I))):
        findings.append(Finding("skew", "info",
            f"{mean_cols[0]} is a mean shown without dispersion. Healthcare cost/LOS are typically "
            f"right-skewed, so the mean can mislead — request median + IQR (or a distribution) to confirm."))


def _check_confounding(df, findings, question):
    """Warn when an outcome is compared across groups with no adjustment; name plausible confounders."""
    cats = _cat_cols(df)
    has_outcome = bool(_match(_num_cols(df), _RATE_RE) or _match(_num_cols(df), _MEAN_RE)
                       or _match(_num_cols(df), _MEASURE_RE))
    if not cats or not has_outcome or not _COMPARE_RE.search(question or ""):
        return
    grouping = cats[0]
    present = " ".join(df.columns).lower() + " " + (question or "").lower()
    candidates = [d for d in ("age", "sex/gender", "comorbidity burden", "condition severity", "payer/access")
                  if d.split("/")[0] not in present or d in ("comorbidity burden", "condition severity")]
    # if the grouping itself is a demographic, the confounders are the *other* demographics + severity
    conf = ", ".join(candidates[:3]) or "age, sex, comorbidity burden"
    findings.append(Finding("confounding", "warn",
        f"This compares an outcome across `{grouping}` **unadjusted**. Groups likely differ in {conf}, "
        f"which can drive the gap. Before any causal reading, age/sex-standardize the rates or fit a "
        f"covariate-adjusted model (e.g. logistic/GLM); a crude comparison is descriptive only."))


def _check_simpsons(df, findings):
    """If the result is stratified (2 category cols + numerator/denominator), check for rank reversal
    between the pooled (marginal) and stratified comparisons."""
    cats = _cat_cols(df)
    numer = _match(_num_cols(df), _NUMER_RE)
    denom = [c for c in _num_cols(df) if _DENOM_RE.search(str(c))]
    if len(cats) < 2 or not numer or not denom:
        return
    g, s = cats[0], cats[1]            # primary group, stratifier
    kcol = numer[0]
    ncol = max(denom, key=lambda c: df[c].sum())

    def rate_by(frame, col):
        out = {}
        for lvl, x in frame.groupby(col):
            tot = x[ncol].sum()
            if tot > 0:
                out[lvl] = x[kcol].sum() / tot
        return out

    marg = rate_by(df, g)              # marginal rate per primary group (pooled over stratifier)
    if len(marg) < 2:
        return
    ordered = sorted(marg, key=marg.get, reverse=True)
    top, bottom = ordered[0], ordered[-1]
    reversed_in = []
    for lvl, sub in df.groupby(s):     # within each stratum, does bottom beat top?
        r = rate_by(sub, g)
        if top in r and bottom in r and r[bottom] > r[top]:
            reversed_in.append(str(lvl))
    if reversed_in:
        findings.append(Finding("simpsons", "caution",
            f"Simpson's paradox risk: marginally `{top}` > `{bottom}`, but within stratum "
            f"{', '.join(reversed_in[:3])} the ordering reverses. The pooled comparison is "
            f"confounded by `{s}` — report stratified, not collapsed."))


def _check_missing_denominator(df, findings):
    rate = _match(_num_cols(df), _RATE_RE)
    denom = [c for c in _num_cols(df) if _DENOM_RE.search(str(c))]
    if rate and not denom:
        findings.append(Finding("missing_denominator", "warn",
            f"Rate column(s) {rate} are shown without a denominator/count — a rate is a base-rate trap "
            f"without the N it is computed over. Add the group size so it can be judged."))


def _check_multiple_comparisons(df, findings, question, did_contrasts):
    if did_contrasts:
        return  # already handled rigorously with FDR
    if len(df) >= MANY_GROUPS and _COMPARE_RE.search(question or ""):
        findings.append(Finding("multiple_comparisons", "warn",
            f"Comparing/ranking {len(df)} groups inflates false positives. Correct before calling any "
            f"single difference real: Bonferroni α/{len(df)} = {ALPHA/len(df):.4f}, or Benjamini-Hochberg FDR."))


# ───────────────────────── entry point ─────────────────────────
def analyze(df: pd.DataFrame, question: str = "", sql: str = "") -> list[Finding]:
    findings: list[Finding] = []
    if df is None or len(df) == 0:
        return [Finding("small_sample", "caution", "The query returned no rows — nothing to interpret.")]

    numer = _match(_num_cols(df), _NUMER_RE)
    denom = [c for c in _num_cols(df) if _DENOM_RE.search(str(c))]
    did_contrasts = bool(numer and denom and 2 <= len(df) <= 60)

    for check in (
        lambda: _check_small_sample(df, findings),
        lambda: _check_contrasts(df, findings),
        lambda: _check_simpsons(df, findings),
        lambda: _check_skew(df, findings),
        lambda: _check_confounding(df, findings, question),
        lambda: _check_missing_denominator(df, findings),
        lambda: _check_multiple_comparisons(df, findings, question, did_contrasts),
    ):
        try:
            check()
        except Exception:
            pass  # a failing check must never break the analysis

    findings.append(Finding("synthetic_data", "info",
        "Data is synthetic (Synthea): structurally realistic but generated from care-process models, "
        "so magnitudes are illustrative, not empirical. Demonstrates method, not clinical fact."))
    return findings


def render(findings: list[Finding]) -> str:
    icon = {"info": "ℹ️", "warn": "⚠️", "caution": "🛑"}
    return "\n".join(f"{icon.get(f.severity, '•')} [{f.kind}] {f.message}" for f in findings)


if __name__ == "__main__":
    demo = pd.DataFrame({
        "age_group": ["18-39", "40-64", "65-74", "75+"],
        "patients_with_condition": [12, 147, 62, 4],
        "total_patients_in_age_group": [309, 394, 120, 8],
        "prevalence_pct": [3.88, 37.31, 51.67, 50.0],
    })
    print(render(analyze(demo, "Which age group has the highest prevalence of hypertension?")))
