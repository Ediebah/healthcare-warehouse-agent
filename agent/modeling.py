"""Inferential-modeling layer — fit real statistical models, not just SQL aggregation.

Given a patient-level analytic DataFrame (one row per unit, outcome + covariates as columns), fit a
model and return a structured result: effect estimates (odds/hazard ratios or coefficients) with
95% CIs and p-values, n, and a fit statistic. Deterministic — the LLM chooses the model and builds
the dataset, but the fitting and inference are done by statsmodels/scipy, so the numbers are
trustworthy. This is the "fit a covariate-adjusted model" the guardrail keeps recommending.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import bayes as _bayes
from . import prespec as _prespec


@dataclass
class Term:
    name: str
    estimate: float          # odds ratio / hazard ratio / coefficient
    ci_low: float
    ci_high: float
    p: float
    n: int | None = None     # subjects in this level (categorical only; mutually exclusive across levels)
    events: int | None = None  # events within this level (event models: logistic / Cox)


@dataclass
class ModelResult:
    model_type: str          # logistic | ols | cox | association
    outcome: str
    n: int
    effect_label: str        # "odds ratio" | "hazard ratio" | "coefficient" | test name
    terms: list = field(default_factory=list)
    km: list = field(default_factory=list)   # Kaplan-Meier curve points [{group,time,survival,ci_low,ci_high}]
    series: list = field(default_factory=list)  # time-series points [{time,value,lower,upper,kind}]
    arms: list = field(default_factory=list)    # A/B arms [{arm,n,value,ci_low,ci_high,is_baseline,is_winner}]
    verdict: dict = field(default_factory=dict)  # experiment call {call, reason}
    robustness: dict = field(default_factory=dict)  # specification-curve multiverse summary (adjusted models)
    prespec: dict = field(default_factory=dict)     # pre-specification lock status {status, lock, drift}
    issues: list = field(default_factory=list)   # flagged statistical issues (strings)
    fit_stat: str = ""
    note: str = ""
    error: str | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


def _formula(outcome: str, predictors: list[str], df: pd.DataFrame) -> str:
    rhs = []
    for p in predictors:
        if p in df.columns and not pd.api.types.is_numeric_dtype(df[p]):
            rhs.append(f"C({p})")            # categorical → dummy-encoded, first level = reference
        else:
            rhs.append(p)
    return f"{outcome} ~ " + (" + ".join(rhs) if rhs else "1")


_CAT_TERM = re.compile(r"^C\((?P<col>[^)]+)\)\[T\.(?P<lvl>.+)\]$")


def _term_counts(name: str, d: pd.DataFrame, event_col: str | None) -> tuple:
    """(n, events) for a categorical dummy term C(col)[T.level]: the subjects in that level and — for an
    event model — the events among them. Mutually exclusive across the levels of one categorical (each
    subject is in exactly one level). (None, None) for a continuous term — a slope has no group."""
    m = _CAT_TERM.match(str(name))
    if not m or m.group("col") not in d.columns:
        return None, None
    mask = d[m.group("col")].astype(str) == m.group("lvl")
    n = int(mask.sum())
    ev = (int(pd.to_numeric(d.loc[mask, event_col], errors="coerce").fillna(0).sum())
          if event_col and event_col in d.columns else None)
    return n, ev


def _reference_terms(d: pd.DataFrame, preds: list[str], event_col: str | None, null_value: float) -> list:
    """One row per categorical predictor's REFERENCE (omitted) level, carrying its n and events so the
    displayed levels form a complete, mutually-exclusive partition of the sample. estimate = the null
    (OR/HR = 1, coef = 0), no CI/p — shown in the results table, not plotted on the forest."""
    refs = []
    for p in preds:
        if p not in d.columns or pd.api.types.is_numeric_dtype(d[p]):
            continue
        levels = sorted(str(x) for x in d[p].dropna().unique())
        if len(levels) < 2:
            continue
        ref = levels[0]                              # statsmodels Treatment coding → first level is reference
        mask = d[p].astype(str) == ref
        ev = (int(pd.to_numeric(d.loc[mask, event_col], errors="coerce").fillna(0).sum())
              if event_col and event_col in d.columns else None)
        refs.append(Term(f"C({p})[{ref}] (ref)", null_value, float("nan"), float("nan"), float("nan"),
                         int(mask.sum()), ev))
    return refs


# Label names whose meaning identifies the event/positive class (or its complement) in a 2-level
# text column. Deliberately conservative: genuinely ambiguous words ("failed" is the event in
# reliability data but the non-event in conversion data) are left out, so they fall through to the
# deterministic alphabetical rule + a loud note rather than a wrong guess.
_POSITIVE_LABELS = frozenset({
    "1", "true", "t", "yes", "y", "event", "dead", "died", "deceased", "death", "success",
    "succeeded", "converted", "cured", "positive", "pos", "case", "responder", "readmitted",
})
_NEGATIVE_LABELS = frozenset({
    "0", "false", "f", "no", "n", "alive", "survived", "censored", "negative", "neg", "none",
    "healthy", "nonresponder", "non-responder",
})


def _event_label(s: pd.Series) -> str | None:
    """Which of a 2-level non-numeric column's labels is the event (coded 1)? A recognized
    positive/negative name wins; otherwise the lexicographically LAST label (case-insensitive) —
    deterministic, unlike order-of-first-appearance, which flipped with row order."""
    cats = sorted({str(c) for c in pd.unique(s.dropna())}, key=str.lower)
    if len(cats) != 2:
        return None
    a, b = cats
    pa, pb = a.lower().strip() in _POSITIVE_LABELS, b.lower().strip() in _POSITIVE_LABELS
    if pa != pb:
        return a if pa else b
    na, nb = a.lower().strip() in _NEGATIVE_LABELS, b.lower().strip() in _NEGATIVE_LABELS
    if na != nb:
        return b if na else a
    return b


def _to_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(set(pd.unique(s.dropna())))
        if set(vals) <= {0, 1}:
            return s.fillna(0).astype(int)
        if len(vals) == 2:                          # 2-level numeric (e.g. {1,2}) → higher level = 1
            return (s == vals[1]).astype(int)
        # continuous / multi-level numeric must NOT be silently thresholded at >0
        raise ValueError(f"outcome '{s.name}' is not binary ({len(vals)} numeric levels); "
                         f"use a regression model or supply a 0/1 outcome")
    cats = list(pd.unique(s.dropna()))
    if len(cats) == 2:
        return (s.astype(str) == _event_label(s)).astype(int)
    raise ValueError(f"outcome '{s.name}' is not binary ({len(cats)} levels)")


def _clean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.dropna(subset=[c for c in cols if c in df.columns]).copy()


def _reduce_collinearity(d: pd.DataFrame, predictors: list[str], vif_max: float = 10.0):
    """Remove multicollinear NUMERIC predictors BEFORE fitting: iteratively drop the highest-VIF
    feature until every VIF ≤ vif_max (the standard multicollinearity screen). Two perfectly/near-
    perfectly correlated predictors can't both stay in a model — one is redundant. Returns
    (kept_predictors, dropped) where dropped = [(name, vif)]. Categoricals are left to the model."""
    num = [p for p in predictors if p in d.columns and pd.api.types.is_numeric_dtype(d[p])]
    other = [p for p in predictors if p not in num]
    if len(num) < 2:
        return list(predictors), []
    dnum = d[num].astype(float)
    dnum = dnum.fillna(dnum.median())              # median-fill for the VIF/corr math (works pre-imputation)
    keep, dropped = list(num), []
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        while len(keep) > 1:
            x = add_constant(dnum[keep], has_constant="add").to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):   # perfect collinearity → 1/0; expected
                vifs = [(keep[i], variance_inflation_factor(x, i + 1)) for i in range(len(keep))]
            name, v = max(vifs, key=lambda kv: kv[1] if np.isfinite(kv[1]) else float("inf"))
            if np.isfinite(v) and v <= vif_max:
                break
            keep.remove(name)
            dropped.append((name, v))
    except Exception:  # noqa: BLE001 — fall back to greedy pairwise-correlation pruning
        cm = dnum.corr().abs()
        changed = True
        while changed and len(keep) > 1:
            changed = False
            sub = cm.loc[keep, keep]
            pairs = sub.where(np.triu(np.ones(sub.shape, dtype=bool), k=1)).stack()
            if len(pairs) and pairs.max() > 0.95:
                keep.remove(pairs.idxmax()[1])
                dropped.append((pairs.idxmax()[1], float("nan")))
                changed = True
    return keep + other, dropped


def _reduce_rare_levels(d: pd.DataFrame, predictors: list[str], min_count: int):
    """Pool sparse levels of each categorical predictor into a single 'other' bucket. A level with fewer
    than min_count subjects cannot support a stable effect estimate — it drives the odds/hazard ratio to
    0 or ∞ with a degenerate [0, ∞] CI (sparse-category separation). Merging such levels is the standard
    consolidation a careful analyst does BEFORE fitting, not something to explain away after. Returns
    (d, pooled, became_constant) with pooled = [(col, [rare_levels], n_pooled)]."""
    pooled, gone = [], []
    for p in predictors:
        if p not in d.columns or pd.api.types.is_numeric_dtype(d[p]):
            continue
        s = d[p].astype("object")
        vc = s.value_counts(dropna=True)
        rare = [lvl for lvl, c in vc.items() if int(c) < min_count and str(lvl) != "other"]
        if not rare:
            continue
        d[p] = s.where(~s.isin(rare), other="other")
        pooled.append((p, rare, int(vc[rare].sum())))
        if d[p].nunique(dropna=True) <= 1:              # collapsed to a single value → caller drops it
            gone.append(p)
    return d, pooled, gone


def _prepare(df: pd.DataFrame, outcome_cols: list[str], predictors: list[str], vif_max: float = 10.0,
             max_missing: float = 0.10, dominance: float = 0.99, impute: bool = True):
    """Comprehensive pre-modeling data engineering, with a transparent record of every step:
      1. drop rows with a missing OUTCOME (a supervised target can't be imputed);
      2. drop predictors with > max_missing missing (default 10%);
      3. drop quasi-constant / zero-variance predictors (no information);
      4. impute the remaining missingness (numeric → median, categorical → mode);
      5. remove multicollinearity (VIF > vif_max).
    Returns (engineered_df, kept_predictors, steps)."""
    d = df.copy()
    steps: list[str] = []

    n0 = len(d)                                        # 1. missing outcome
    d = d.dropna(subset=[c for c in outcome_cols if c in d.columns])
    if n0 - len(d) > 0:
        steps.append(f"Dropped {n0 - len(d):,} of {n0:,} rows with a missing outcome.")
    preds = [p for p in predictors if p in d.columns]

    hi_missing = [p for p in preds if d[p].isna().mean() > max_missing]   # 2. high missingness
    if hi_missing:
        steps.append(f"Dropped {len(hi_missing)} predictor(s) with >{max_missing:.0%} missing: "
                     + ", ".join(f"{p} ({d[p].isna().mean():.0%})" for p in hi_missing[:6])
                     + ("…" if len(hi_missing) > 6 else "") + ".")
        preds = [p for p in preds if p not in hi_missing]

    quasi = []                                         # 3. quasi-constant / zero variance
    for p in preds:
        s = d[p].dropna()
        if len(s) == 0 or s.nunique() <= 1 or s.value_counts(normalize=True).iloc[0] > dominance:
            quasi.append(p)
    if quasi:
        steps.append(f"Dropped {len(quasi)} quasi-constant predictor(s) (one value >"
                     f"{dominance:.0%} or zero variance): " + ", ".join(quasi[:6])
                     + ("…" if len(quasi) > 6 else "") + ".")
        preds = [p for p in preds if p not in quasi]

    # 3b. drop datetime and high-cardinality / ID-like categorical predictors: with one level per row
    #     a categorical term saturates the design (R²→1, singular fit) and a datetime crashes patsy.
    card_cap = max(20, int(0.5 * len(d)))
    unusable = []
    for p in preds:
        s = d[p]
        if pd.api.types.is_datetime64_any_dtype(s):
            unusable.append(p)
        elif not pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > card_cap:
            unusable.append(p)
    if unusable:
        steps.append(f"Dropped {len(unusable)} unusable predictor(s) — datetime or high-cardinality / "
                     f"ID-like (>{card_cap} categories): " + ", ".join(unusable[:6])
                     + ("…" if len(unusable) > 6 else "") + ".")
        preds = [p for p in preds if p not in unusable]

    # 3c. pool sparse levels of a categorical predictor into 'other' so a level with a handful of subjects
    #     can't produce a degenerate effect (OR/HR → 0 or ∞ with a [0, ∞] CI) — sparse-category separation.
    min_level = max(15, int(np.ceil(0.01 * len(d))))
    d, pooled, gone = _reduce_rare_levels(d, preds, min_level)
    for col, lvls, npool in pooled:
        shown = ", ".join(str(x) for x in lvls[:6]) + (f", +{len(lvls) - 6} more" if len(lvls) > 6 else "")
        steps.append(f"Pooled {len(lvls)} sparse level(s) of '{col}' (<{min_level} subjects each: {shown}) "
                     f"into 'other' ({npool} subjects) to avoid sparse-category separation.")
    if gone:
        steps.append(f"Dropped {len(gone)} predictor(s) left single-valued after pooling: "
                     + ", ".join(gone) + ".")
        preds = [p for p in preds if p not in gone]

    if impute:                                         # 4. impute remaining missingness
        imputed = []                                   # (skipped when a downstream pipeline imputes per-fold)
        for p in preds:
            miss = int(d[p].isna().sum())
            if miss > 0:
                if pd.api.types.is_numeric_dtype(d[p]):
                    d[p] = d[p].fillna(d[p].median()); how = "median"
                else:
                    mode = d[p].mode(dropna=True)
                    d[p] = d[p].fillna(mode.iloc[0] if len(mode) else "missing"); how = "mode"
                imputed.append(f"{p} ({miss} by {how})")
        if imputed:
            steps.append("Imputed missing values (single imputation): " + ", ".join(imputed[:6])
                         + (f", +{len(imputed) - 6} more" if len(imputed) > 6 else "")
                         + " — for confirmatory analysis prefer multiple imputation with pooling.")

    kept, dropped = _reduce_collinearity(d, preds, vif_max)   # 5. multicollinearity
    if dropped:
        parts = ", ".join(f"{n} (VIF {'∞' if not np.isfinite(v) else f'{v:.0f}'})" for n, v in dropped)
        steps.append(f"Removed {len(dropped)} collinear predictor(s) (VIF>{vif_max:.0f}): {parts}. "
                     "Each is redundant with a retained predictor; the retained set is identifiable.")

    if impute:
        d = d.dropna(subset=[c for c in [*outcome_cols, *kept] if c in d.columns])
    else:
        d = d.dropna(subset=[c for c in outcome_cols if c in d.columns])   # keep predictor NaN for the pipeline
    return d, kept, steps


def _separation_flag(params, bses) -> str | None:
    """Complete / quasi-complete separation → the logistic MLE diverges (huge |coef| or SE)."""
    bad = [n for n in params.index if n != "Intercept" and (abs(params[n]) > 10 or bses[n] > 10)]
    if bad:
        return ("Possible complete/quasi-complete separation (" + ", ".join(bad[:3]) + "): a predictor "
                "near-perfectly predicts the outcome, so the odds ratios and CIs diverge and are "
                "unreliable — use penalized (Firth) logistic regression or drop the term.")
    return None


def _nonlinearity_flags(refit_pval, d: pd.DataFrame, preds: list[str]) -> list[tuple]:
    """For each continuous predictor, add a quadratic term and flag if it's significant (linearity of
    the log-odds / log-hazard is assumed for continuous covariates)."""
    flags = []
    for p in preds:
        if p in d.columns and pd.api.types.is_numeric_dtype(d[p]) and d[p].nunique() >= 6:
            try:
                pv = refit_pval(p)
                if pv is not None and pv == pv and pv < 0.05:
                    flags.append((p, pv))
            except Exception:  # noqa: BLE001
                continue
    return flags


def _ph_flags(result, durations) -> list[tuple]:
    """Proportional-hazards check: correlate each covariate's Schoenfeld residuals with event time
    (Grambsch–Therneau idea). A significant correlation → the hazard ratio changes over time."""
    from scipy import stats
    sr = np.asarray(result.schoenfeld_residuals)
    mask = ~np.isnan(sr).any(axis=1)
    if mask.sum() < 8:
        return []
    t = np.asarray(durations, dtype=float)[mask]
    flags = []
    for j, name in enumerate(result.model.exog_names):
        rj = sr[mask, j]
        if np.std(rj) == 0:
            continue
        rho, p = stats.spearmanr(t, rj)
        if p == p and p < 0.05:
            flags.append((name, p))
    return flags


def _het_flag(m) -> str | None:
    """Breusch–Pagan heteroskedasticity test for OLS (non-constant residual variance)."""
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        p = het_breuschpagan(m.resid, m.model.exog)[1]
        if p == p and p < 0.05:
            return (f"Heteroskedasticity (Breusch–Pagan p={p:.3g}): residual variance isn't constant — "
                    "use robust (HC) standard errors; coefficients stay unbiased but CIs/p-values may be off.")
    except Exception:  # noqa: BLE001
        pass
    return None


def fit_logistic(df: pd.DataFrame, outcome: str, predictors: list[str]) -> ModelResult:
    """Logistic regression → adjusted odds ratios with 95% CIs (the classic confounding fix)."""
    try:
        import statsmodels.formula.api as smf
        d, preds, issues = _prepare(df, [outcome], predictors)
        if not preds:
            return ModelResult("logistic", outcome, len(d), "odds ratio",
                               error="No usable predictors remained after data screening (all were "
                                     "constant, collinear, too missing, or high-cardinality/ID-like).")
        if (bn := _binary_note(d[outcome])):
            issues.append(bn)
        d[outcome] = _to_binary(d[outcome])
        if d[outcome].nunique() < 2:
            return ModelResult("logistic", outcome, len(d), "odds ratio",
                               error="Outcome has no variation (all one class).")
        m = smf.logit(_formula(outcome, preds, d), data=d).fit(disp=0)
        ci = m.conf_int()
        terms = [Term(name, float(np.exp(m.params[name])), float(np.exp(ci.loc[name, 0])),
                      float(np.exp(ci.loc[name, 1])), float(m.pvalues[name]))
                 for name in m.params.index if name != "Intercept"]
        for t in terms:                              # per-category subjects + events (mutually exclusive)
            t.n, t.events = _term_counts(t.name, d, outcome)
        events = int(min((d[outcome] == 0).sum(), (d[outcome] == 1).sum()))
        if terms and events / len(terms) < 10:       # events-per-variable rule of thumb
            issues.append(f"Low events-per-variable (EPV≈{events / len(terms):.1f}): {events} events for "
                          f"{len(terms)} term(s). Under ~10 EPV, odds ratios can be overfit/unstable — "
                          "reduce predictors or use penalized (e.g. Firth) logistic regression.")
        sep = _separation_flag(m.params, m.bse)      # complete/quasi-complete separation
        if sep:
            issues.append(sep)

        def _refit(p):                               # linearity of the log-odds
            f = _formula(outcome, preds, d) + f" + I({p} ** 2)"
            mm = smf.logit(f, data=d).fit(disp=0)
            key = next((k for k in mm.pvalues.index if k.startswith(f"I({p}")), None)
            return float(mm.pvalues[key]) if key else None
        for p, pv in _nonlinearity_flags(_refit, d, preds):
            issues.append(f"Non-linearity ({p}, quadratic term p={pv:.3g}): {p} isn't linear in the "
                          "log-odds — model it with splines or a polynomial, or categorize it.")
        terms = _reference_terms(d, preds, outcome, 1.0) + terms   # complete the categorical partition
        mr = ModelResult("logistic", outcome, int(m.nobs), "odds ratio", terms,
                         fit_stat=f"pseudo-R²={m.prsquared:.3f}",
                         note="Odds ratio > 1 = higher odds of the outcome, holding the others fixed.")
        mr.issues = issues
        return mr
    except Exception as e:  # noqa: BLE001
        return ModelResult("logistic", outcome, 0, "odds ratio", error=str(e))


def fit_ols(df: pd.DataFrame, outcome: str, predictors: list[str]) -> ModelResult:
    """Linear regression → adjusted coefficients with 95% CIs for a continuous outcome."""
    try:
        import statsmodels.formula.api as smf
        d, preds, issues = _prepare(df, [outcome], predictors)
        if not preds:
            return ModelResult("ols", outcome, len(d), "coefficient",
                               error="No usable predictors remained after data screening (all were "
                                     "constant, collinear, too missing, or high-cardinality/ID-like).")
        if len(d) < len(preds) + 2:
            return ModelResult("ols", outcome, len(d), "coefficient",
                               error=f"Too few rows ({len(d)}) to fit {len(preds)} predictor(s).")
        m = smf.ols(_formula(outcome, preds, d), data=d).fit()
        ci = m.conf_int()
        terms = [Term(name, float(m.params[name]), float(ci.loc[name, 0]),
                      float(ci.loc[name, 1]), float(m.pvalues[name]))
                 for name in m.params.index if name != "Intercept"]
        het = _het_flag(m)                           # non-constant residual variance
        if het:
            issues.append(het)

        def _refit(p):                               # linearity of the mean
            mm = smf.ols(_formula(outcome, preds, d) + f" + I({p} ** 2)", data=d).fit()
            key = next((k for k in mm.pvalues.index if k.startswith(f"I({p}")), None)
            return float(mm.pvalues[key]) if key else None
        for p, pv in _nonlinearity_flags(_refit, d, preds):
            issues.append(f"Non-linearity ({p}, quadratic term p={pv:.3g}): the mean response isn't "
                          "linear in {p} — add a polynomial/spline term.".replace("{p}", p))
        mr = ModelResult("ols", outcome, int(m.nobs), "coefficient", terms,
                         fit_stat=f"R²={m.rsquared:.3f}",
                         note="Coefficient = change in the outcome per 1-unit change, others fixed.")
        mr.issues = issues
        return mr
    except Exception as e:  # noqa: BLE001
        return ModelResult("ols", outcome, 0, "coefficient", error=str(e))


def fit_cox(df: pd.DataFrame, duration: str, event: str, predictors: list[str]) -> ModelResult:
    """Cox proportional-hazards → adjusted hazard ratios (time-to-event / survival)."""
    try:
        from statsmodels.duration.hazard_regression import PHReg
        d, preds, issues = _prepare(df, [duration, event], predictors)
        if not preds:
            return ModelResult("cox", duration, len(d), "hazard ratio",
                               error="No usable predictors remained after data screening (all were "
                                     "constant, collinear, too missing, or high-cardinality/ID-like).")
        if (bn := _binary_note(d[event], role="Event indicator")):
            issues.append(bn)
        d[event] = _to_binary(d[event])
        d = d[pd.to_numeric(d[duration], errors="coerce") > 0]   # survival durations must be positive
        if len(d) < 2:
            return ModelResult("cox", duration, len(d), "hazard ratio",
                               error="No positive follow-up durations to fit a Cox model.")
        if int(d[event].sum()) == 0:
            return ModelResult("cox", duration, len(d), "hazard ratio",
                               error="No events observed (all rows are censored) — a Cox model needs events.")
        mod = PHReg.from_formula(_formula(duration, preds, d), data=d, status=d[event])
        r = mod.fit()
        ci = r.conf_int()
        terms = [Term(name, float(np.exp(r.params[i])), float(np.exp(ci[i, 0])),
                      float(np.exp(ci[i, 1])), float(r.pvalues[i]))
                 for i, name in enumerate(r.model.exog_names)]
        for t in terms:                              # per-category subjects + events (mutually exclusive)
            t.n, t.events = _term_counts(t.name, d, event)
        events = int(d[event].sum())
        if terms and events / len(terms) < 10:       # events-per-variable rule of thumb
            issues.append(f"Low events-per-variable (EPV≈{events / len(terms):.1f}): {events} events for "
                          f"{len(terms)} term(s). Under ~10 EPV, hazard ratios can be overfit/unstable — "
                          "reduce predictors.")
        bse = getattr(r, "bse", None)                # separation guard (logistic has one; Cox needs it too)
        sep = [name for i, name in enumerate(r.model.exog_names)
               if abs(float(r.params[i])) > 10 or (bse is not None and float(bse[i]) > 10)]
        if sep:
            issues.append("Possible separation (" + ", ".join(sep[:3]) + "): a covariate near-perfectly "
                          "predicts the event, so the hazard ratios and CIs diverge and are unreliable — "
                          "use penalized (Firth) Cox or drop the term.")
        ph = _ph_flags(r, d[duration].to_numpy())                # proportional-hazards check (consolidated)
        if ph:
            names = ", ".join(n for n, _ in ph[:6]) + (f", +{len(ph) - 6} more" if len(ph) > 6 else "")
            issues.append(f"Proportional-hazards violation for {len(ph)} covariate(s) ({names}): the hazard "
                          "ratio changes over follow-up — stratify on these or add a time interaction.")

        def _refit(p):                               # linearity of the log-hazard
            mm = PHReg.from_formula(_formula(duration, preds, d) + f" + I({p} ** 2)",
                                    data=d, status=d[event]).fit()
            names = list(mm.model.exog_names)
            key = next((k for k in names if k.startswith(f"I({p}")), None)
            return float(mm.pvalues[names.index(key)]) if key else None
        for p, pv in _nonlinearity_flags(_refit, d, preds):
            issues.append(f"Non-linearity ({p}, quadratic term p={pv:.3g}): {p} isn't linear in the "
                          "log-hazard — model it with splines or a polynomial.")
        terms = _reference_terms(d, preds, event, 1.0) + terms   # complete the categorical partition
        mr = ModelResult("cox", duration, int(r.model.surv.n_obs), "hazard ratio", terms,
                         fit_stat=f"events={int(d[event].sum())}",
                         note="Hazard ratio > 1 = faster time-to-event (higher risk), others fixed.")
        mr.issues = issues
        return mr
    except Exception as e:  # noqa: BLE001
        return ModelResult("cox", duration, 0, "hazard ratio", error=str(e))


def fit_km(df: pd.DataFrame, duration: str, event: str, group: str | None = None) -> list[dict]:
    """Kaplan-Meier survival curve points, optionally stratified by a categorical group."""
    try:
        from statsmodels.duration.survfunc import SurvfuncRight
        d = _clean(df, [duration, event, *([group] if group else [])])
        d[event] = _to_binary(d[event])
        d = d[pd.to_numeric(d[duration], errors="coerce") > 0]   # survival durations must be positive
        groups = list(d.groupby(group)) if (group and group in d.columns) else [("all", d)]
        curves = []
        for gname, gd in groups:
            if len(gd) < 2:
                continue
            sf = SurvfuncRight(gd[duration].to_numpy(float), gd[event].to_numpy(int))
            se = getattr(sf, "surv_prob_se", None)
            for i, (t, s) in enumerate(zip(sf.surv_times, sf.surv_prob)):
                lo = hi = float("nan")
                if se is not None:
                    lo, hi = max(0.0, float(s) - 1.96 * float(se[i])), min(1.0, float(s) + 1.96 * float(se[i]))
                curves.append({"group": str(gname), "time": float(t), "survival": float(s),
                               "ci_low": lo, "ci_high": hi})
        return curves
    except Exception:  # noqa: BLE001 — a malformed event/duration must not crash survival analysis
        return []


def fit_survival(df: pd.DataFrame, duration: str, event: str,
                 predictors: list[str] | None = None, group: str | None = None) -> ModelResult:
    """Survival analysis — Cox hazard ratios (if predictors given) + Kaplan-Meier curves (grouped)."""
    predictors = list(predictors or [])
    if not predictors and group and group in df.columns:
        predictors = [group]          # also quantify the group difference with a Cox hazard ratio
    mr = (fit_cox(df, duration, event, predictors) if predictors
          else ModelResult("cox", duration, len(df), "hazard ratio", note="Kaplan-Meier only."))
    try:
        mr.km = fit_km(df, duration, event, group)
    except Exception:
        mr.km = []
    mr.model_type = "survival"
    return mr


def test_association(df: pd.DataFrame, a: str, b: str) -> ModelResult:
    """Two-variable association: Pearson r (num~num), t-test (num~binary), or chi-square (cat~cat)."""
    try:
        from scipy import stats
        d = _clean(df, [a, b])
        an, bn = pd.api.types.is_numeric_dtype(d[a]), pd.api.types.is_numeric_dtype(d[b])
        if an and bn:
            r, p = stats.pearsonr(d[a], d[b])
            return ModelResult("association", f"{a} vs {b}", len(d), "Pearson r",
                               [Term("r", float(r), float("nan"), float("nan"), float(p))])
        if an != bn:                               # one numeric, one categorical → t-test / ANOVA
            num, cat = (a, b) if an else (b, a)
            groups = [g[num].to_numpy() for _, g in d.groupby(cat)]
            if len(groups) == 2:
                t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                return ModelResult("association", f"{num} by {cat}", len(d), "Welch t-test",
                                   [Term("t", float(t), float("nan"), float("nan"), float(p))])
            f, p = stats.f_oneway(*groups)
            return ModelResult("association", f"{num} by {cat}", len(d), "one-way ANOVA",
                               [Term("F", float(f), float("nan"), float("nan"), float(p))])
        ct = pd.crosstab(d[a], d[b])               # both categorical → chi-square
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        return ModelResult("association", f"{a} vs {b}", len(d), "chi-square",
                           [Term(f"chi2(df={dof})", float(chi2), float("nan"), float("nan"), float(p))])
    except Exception as e:  # noqa: BLE001
        return ModelResult("association", f"{a} vs {b}", 0, "association", error=str(e))


def _is_binary_outcome(y: pd.Series) -> bool:
    if y.dtype == bool:
        return True
    # exactly two distinct non-null values is binary — numeric {0,1} or {1,2}, or a 2-level category
    return y.nunique(dropna=True) == 2


def _binary_note(s: pd.Series, role: str = "Outcome") -> str | None:
    """State how an ambiguous 2-level column was coded, so a flipped direction is never silent.
    Numeric non-{0,1} (e.g. {1,2}): higher value = event. Text: _event_label's choice — flagged as
    a guess when neither label name is recognized."""
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(set(pd.unique(pd.Series(s).dropna())))
        if len(vals) == 2 and set(vals) != {0, 1}:
            return (f"{role} is numeric coded {vals[0]:g}/{vals[1]:g}; the higher value ({vals[1]:g}) is "
                    "treated as the event. If your coding is reversed the effect direction flips — recode to 0/1.")
        return None
    cats = sorted({str(c) for c in pd.unique(pd.Series(s).dropna())}, key=str.lower)
    if len(cats) != 2:
        return None
    ev = _event_label(s)
    other = cats[0] if ev == cats[1] else cats[1]
    low = {c.lower().strip() for c in cats}
    if (low & _POSITIVE_LABELS) or (low & _NEGATIVE_LABELS):
        return f"{role} is text coded '{other}'/'{ev}'; '{ev}' is treated as the event/positive class."
    return (f"{role} is text coded '{other}'/'{ev}'; '{ev}' (alphabetically last) is treated as the "
            "event/positive class — nothing in the names says which is which. If that's backwards the "
            "effect direction flips: recode to 0/1 or rename the levels.")


def fit_forest(df: pd.DataFrame, outcome: str, predictors: list[str]) -> ModelResult:
    """Random forest → permutation feature importances: which factors most predict the outcome.
    Classifier for a binary outcome (scored by AUC), regressor for a continuous one (scored by R²)."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
        from sklearn.pipeline import Pipeline
        preds0 = [p for p in predictors if p in df.columns and p != outcome]
        d, preds, issues = _prepare(df, [outcome], preds0, impute=False)   # impute inside the CV pipeline
        if len(d) < 40 or len(preds) < 2:
            return ModelResult("forest", outcome, len(d), "importance",
                               error="Need ≥40 rows and ≥2 non-collinear predictors for a random forest.")
        X = pd.get_dummies(d[preds], drop_first=True)
        is_class = _is_binary_outcome(d[outcome])
        common = dict(n_estimators=300, random_state=0, n_jobs=-1, min_samples_leaf=5)
        n_splits = 5
        if is_class:
            if (bn := _binary_note(d[outcome])):
                issues.append(bn)
            y = _to_binary(d[outcome])
            if y.nunique() < 2:
                return ModelResult("forest", outcome, len(d), "importance",
                                   error="Outcome has no variation (all one class).")
            minority = int(y.value_counts().min())
            if minority < 5:                          # too few events to cross-validate AUC or split out
                return ModelResult("forest", outcome, len(d), "importance",
                                   error=f"Outcome too rare for a random forest — only {minority} in the "
                                         "smaller class; need ≥5 (≥10 recommended) to score AUC honestly.")
            n_splits = min(5, minority)
            if minority < 20:
                issues.append(f"Rare outcome ({minority} in the smaller class): the AUC and importances are "
                              f"unstable and cross-validation uses {n_splits} folds — interpret with caution.")
            est = RandomForestClassifier(class_weight="balanced", **common)
            scoring, metric = "roc_auc", "AUC"
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)
        else:
            y = d[outcome].astype(float)
            est = RandomForestRegressor(**common)
            scoring, metric = "r2", "R²"
            cv = KFold(5, shuffle=True, random_state=0)
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", est)])
        cvs = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)      # cross-validated skill
        fit_stat = f"{metric}={cvs.mean():.3f}±{cvs.std():.3f} ({n_splits}-fold CV)"
        # permutation importance on a held-out split — imputer fit on train only (no leakage)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                              stratify=y if is_class else None)
        pipe.fit(Xtr, ytr)
        imp = permutation_importance(pipe, Xte, yte, n_repeats=12, random_state=0, scoring=scoring)
        means = pd.Series(imp.importances_mean, index=X.columns)
        terms = []                                  # roll one-hot columns back up to the source predictor
        for p in preds:
            cols = [c for c in X.columns if c == p or c.startswith(f"{p}_")]
            terms.append(Term(p, float(means[cols].sum()) if cols else 0.0,
                              float("nan"), float("nan"), float("nan")))
        terms.sort(key=lambda t: t.estimate, reverse=True)
        mr = ModelResult("forest", outcome, len(d), "importance", terms, fit_stat=fit_stat,
                         note="5-fold cross-validated skill; permutation importance on a held-out split "
                              "with imputation fit inside a scikit-learn pipeline (no leakage). Predictive, "
                              "not causal.")
        mr.issues = issues
        return mr
    except Exception as e:  # noqa: BLE001
        return ModelResult("forest", outcome, 0, "importance", error=str(e))


def fit_timeseries(df: pd.DataFrame, time_col: str, value_col: str,
                   periods: int = 12, seasonal_periods: int = 12) -> ModelResult:
    """Holt-Winters exponential smoothing → forecast `periods` ahead with an approximate 95% band."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        d = _clean(df, [time_col, value_col]).copy()
        d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
        d = d.dropna(subset=[time_col]).groupby(time_col, as_index=False)[value_col].sum()
        d = d.sort_values(time_col)
        s = d.set_index(time_col)[value_col].astype(float)
        if len(s) < 6:
            return ModelResult("timeseries", value_col, len(s), "forecast",
                               error="Need ≥6 time periods to fit a forecast.")
        s = s.tail(120)                              # focus on the recent regime (≤10y monthly), drop sparse tail
        freq = pd.infer_freq(s.index)
        if freq:                                     # a regular index → seasonality + clean forecast dates
            s = s.asfreq(freq).interpolate()
        seasonal = "add" if len(s) >= 2 * seasonal_periods else None
        sp = seasonal_periods if seasonal else None
        fit = ExponentialSmoothing(s, trend="add", seasonal=seasonal, seasonal_periods=sp,
                                   initialization_method="estimated").fit()
        fc_vals = np.asarray(fit.forecast(periods), dtype=float)
        last = s.index[-1]                           # build forecast dates ourselves (index may lack freq)
        if freq:
            fc_times = pd.date_range(start=last, periods=periods + 1, freq=freq)[1:]
        else:
            step = (s.index[-1] - s.index[-2]) if len(s) >= 2 else pd.Timedelta(days=30)
            fc_times = [last + step * (i + 1) for i in range(periods)]
        sigma = float(np.std((s - fit.fittedvalues).dropna())) or 0.0
        series = [{"time": pd.Timestamp(t).isoformat(), "value": float(v), "lower": float("nan"),
                   "upper": float("nan"), "kind": "history"} for t, v in s.items()]
        for i, (t, v) in enumerate(zip(fc_times, fc_vals)):
            band = 1.96 * sigma * float(np.sqrt(i + 1))       # widen the band with the horizon
            series.append({"time": pd.Timestamp(t).isoformat(), "value": float(v),
                           "lower": float(v - band), "upper": float(v + band), "kind": "forecast"})
        mr = ModelResult("timeseries", value_col, int(len(s)), "forecast",
                         fit_stat=f"{periods}-period forecast" + (" (seasonal)" if seasonal else ""),
                         note="Holt-Winters exponential smoothing; the band widens with the horizon and is "
                              "approximate (residual-based). Synthetic data.")
        mr.series = series
        return mr
    except Exception as e:  # noqa: BLE001
        return ModelResult("timeseries", value_col, 0, "forecast", error=str(e))


_UPLIFT_MAX_ROWS = 20000          # cross-fitting stays interactive below this; sampled stratified by arm


def fit_uplift(df: pd.DataFrame, outcome: str, treatment: str,
               predictors: list[str] | None = None) -> ModelResult:
    """Causal T-learner: two random forests (treated vs control) estimate the average uplift (ATE) of a
    binary treatment on the outcome, adjusting for covariates. Observational → illustrative, not RCT-grade."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import StratifiedKFold
        preds = [p for p in (predictors or []) if p in df.columns and p not in (outcome, treatment)]
        d = _clean(df, [outcome, treatment, *preds])
        t, t_note = _treatment_indicator(d[treatment])
        if len(np.unique(t)) < 2 or int((t == 1).sum()) < 20 or int((t == 0).sum()) < 20:
            return ModelResult("causal", outcome, len(d), "uplift",
                               error="Treatment needs ≥20 treated and ≥20 control rows.")
        issues = []
        if t_note:
            issues.append(t_note)
        if len(d) > _UPLIFT_MAX_ROWS:                    # keep cross-fitting interactive on large inputs
            # Stratify by arm so a rare treatment keeps its share, then RE-CHECK the ≥20-per-arm gate:
            # the pre-sample check alone let a 30-treated/100k cohort shrink to ~6 treated rows.
            frac = _UPLIFT_MAX_ROWS / len(d)
            d = d.groupby(d[treatment].astype(str), group_keys=False).sample(frac=frac, random_state=0)
            t = _treatment_indicator(d[treatment])[0]
            n1, n0 = int((t == 1).sum()), int((t == 0).sum())
            if n1 < 20 or n0 < 20:
                return ModelResult(
                    "causal", outcome, len(d), "uplift",
                    error=(f"After subsampling to ~{_UPLIFT_MAX_ROWS:,} rows the treatment arms are too "
                           f"small to cross-fit (treated={n1}, control={n0}; each needs ≥20). Narrow the "
                           "cohort in SQL so the full data fits, or use a more common treatment."))
            issues.append(f"Estimated on a random {len(d):,}-row subsample (stratified by treatment arm) "
                          "for tractability.")
        is_class = _is_binary_outcome(d[outcome])
        if is_class and (y_note := _binary_note(d[outcome])):
            issues.append(y_note)
        y = (_to_binary(d[outcome]) if is_class else d[outcome].astype(float)).to_numpy(dtype=float)
        X = pd.get_dummies(d[preds], drop_first=True) if preds else pd.DataFrame(index=d.index)
        if X.empty:
            X = pd.DataFrame({"_const": np.ones(len(d))}, index=d.index)
        X = X.to_numpy(dtype=float)
        if is_class and (len(np.unique(y[t == 1])) < 2 or len(np.unique(y[t == 0])) < 2):
            return ModelResult("causal", outcome, len(d), "uplift",
                               error="Outcome has no variation within a treatment arm.")

        # Cross-fitted AIPW (doubly-robust) ATE with an influence-function CI: two potential-outcome
        # forests + a propensity forest, all fit OUT-OF-FOLD — so predictions aren't in-sample and the
        # interval reflects model + sampling variability (the old fixed-vector bootstrap understated it ~3×).
        Out = RandomForestClassifier if is_class else RandomForestRegressor
        okw = dict(n_estimators=200, random_state=0, n_jobs=-1, min_samples_leaf=5)
        if is_class:
            okw["class_weight"] = "balanced"

        def _p1(m, xx):                                  # E[Y|X] robust to a single-class training fold
            if not is_class:
                return m.predict(xx)
            if len(m.classes_) == 1:
                return np.full(len(xx), float(m.classes_[0]))
            return m.predict_proba(xx)[:, list(m.classes_).index(1)]

        n = len(y)
        mu1 = np.zeros(n); mu0 = np.zeros(n); ehat = np.zeros(n)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, t):
            ttr = t[tr]
            mu1[te] = _p1(Out(**okw).fit(X[tr][ttr == 1], y[tr][ttr == 1]), X[te])
            mu0[te] = _p1(Out(**okw).fit(X[tr][ttr == 0], y[tr][ttr == 0]), X[te])
            ps = RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1,
                                        min_samples_leaf=5, class_weight="balanced").fit(X[tr], ttr)
            ehat[te] = ps.predict_proba(X[te])[:, list(ps.classes_).index(1)]
        ehat = np.clip(ehat, 0.025, 0.975)               # positivity/overlap trimming for a stable score
        # AIPW score per unit: (mu1-mu0) + T(Y-mu1)/e - (1-T)(Y-mu0)/(1-e); ATE = mean, SE from its spread
        psi = (mu1 - mu0) + t * (y - mu1) / ehat - (1 - t) * (y - mu0) / (1 - ehat)
        ate = float(np.mean(psi))
        se = float(np.std(psi, ddof=1) / np.sqrt(n))     # influence-function SE (asymptotically valid)
        lo, hi = ate - 1.96 * se, ate + 1.96 * se
        if float(ehat.min()) <= 0.03 or float(ehat.max()) >= 0.97:
            issues.append("Weak overlap: some patients have a propensity near 0 or 1, so the effect is "
                          "extrapolated for them (propensity trimmed to [0.025, 0.975]).")
        label = "uplift (Δ risk)" if is_class else "uplift (Δ outcome)"
        mr = ModelResult("causal", outcome, n, label,
                         [Term(f"effect of {treatment}", ate, lo, hi, float("nan"))],
                         fit_stat=f"AIPW doubly-robust; treated={int((t == 1).sum()):,} / control={int((t == 0).sum()):,}",
                         note="Cross-fitted AIPW average treatment effect (two potential-outcome forests + a "
                              "propensity forest); 95% CI from the influence function. Observational — assumes "
                              "no unmeasured confounding and overlap; NOT a randomized causal effect. Synthetic data.")
        mr.issues = issues
        return mr
    except Exception as e:  # noqa: BLE001
        return ModelResult("causal", outcome, 0, "uplift", error=str(e))


_CONTROL_WORDS = ("control", "baseline", "ctrl", "placebo", "standard", "soc", "usual", "sham",
                  "reference", "default", "original", "holdout", "comparator")


def _is_control(arm: str) -> bool:
    """Recognize the reference arm — its name should START with a control word ('control',
    'standard_of_care', 'placebo'), not merely CONTAIN one, so 'new_standard' / 'new_default'
    (the treatment) is not mistaken for the control."""
    a = str(arm).lower().strip()
    if a in ("a", "0", "off"):
        return True
    return any(a == w or a.startswith(w + "_") or a.startswith(w + " ") or a.startswith(w + "-")
               for w in _CONTROL_WORDS)


def _treatment_indicator(s: pd.Series) -> tuple[np.ndarray, str | None]:
    """Code a 2-level treatment column as 1 = treated / 0 = control. A recognized control-arm name
    (_is_control: 'control', 'placebo', 'baseline', ...) becomes the reference; otherwise falls back
    to _to_binary's deterministic coding. Always returns the mapping note alongside the indicator so
    the caller can surface it — a silently flipped arm flips the effect sign."""
    if not pd.api.types.is_numeric_dtype(s) and s.dtype != bool:
        cats = sorted({str(c) for c in pd.unique(s.dropna())}, key=str.lower)
        if len(cats) == 2:
            ctrl = [c for c in cats if _is_control(c)]
            if len(ctrl) == 1:
                treated = cats[1] if ctrl[0] == cats[0] else cats[0]
                return ((s.astype(str) == treated).astype(int).to_numpy(),
                        f"Treatment coded: '{treated}' = treated vs '{ctrl[0]}' = control "
                        "(recognized control label).")
            ev = _event_label(s)
            other = cats[0] if ev == cats[1] else cats[1]
            return ((s.astype(str) == ev).astype(int).to_numpy(),
                    f"Treatment coded: '{ev}' = treated vs '{other}' = control — neither arm name is a "
                    "recognized control (control/placebo/baseline/...), so the alphabetically last arm "
                    "was taken as treated. If that's backwards the effect sign flips: recode to 0/1 or "
                    "rename the arms.")
    return _to_binary(s).to_numpy(), _binary_note(s, role="Treatment")


def fit_experiment(df: pd.DataFrame, group: str, outcome: str, baseline: str | None = None) -> ModelResult:
    """A/B experiment analysis → per-arm rates/means with CIs, lift vs the control arm (Newcombe CI +
    two-proportion z or Welch t), BH-FDR across variants, flagged issues, and a ship / no-ship verdict."""
    try:
        from . import guardrails as gr
        d = _clean(df, [group, outcome])
        d[group] = d[group].astype(str)
        arms = list(d[group].unique())
        if len(arms) < 2:
            return ModelResult("experiment", outcome, len(d), "lift",
                               error="Need at least two variants (an A and a B) to analyze.")
        binary = _is_binary_outcome(d[outcome])
        if not binary and not pd.api.types.is_numeric_dtype(d[outcome]):
            return ModelResult("experiment", outcome, len(d), "lift",
                               error=f"Outcome must be binary or numeric to compare arms — found "
                                     f"{d[outcome].nunique()} non-numeric categories.")
        if not binary and int(d[group].value_counts().min()) < 2:
            return ModelResult("experiment", outcome, len(d), "lift",
                               error="Each arm needs ≥2 observations to compare a continuous outcome.")
        y = _to_binary(d[outcome]) if binary else d[outcome].astype(float)
        d = d.assign(_y=y.values)

        # pick the control/baseline arm: an explicit choice, else an obvious name, else — deterministically
        # — the LARGEST arm (typical control allocation). NOT the lowest-rate arm: choosing the reference by
        # its outcome would force every lift ≥0 and make the "DO NOT SHIP" branch unreachable (selection bias).
        _counts = d[group].value_counts()
        base = baseline if baseline in arms else next((a for a in arms if _is_control(a)), None)
        auto_base = base is None
        if auto_base:
            base = max(sorted(arms), key=lambda a: int(_counts.get(a, 0)))    # largest n, alphabetical tiebreak
        others = [a for a in arms if a != base]

        stat = {a: d.loc[d[group] == a, "_y"] for a in arms}
        arm_rows, terms, raw_ps, comps = [], [], [], []

        def arm_summary(a):
            v = stat[a]
            if binary:
                k, n = int(v.sum()), int(len(v))
                lo, hi = gr.wilson_ci(k, n)
                return {"arm": a, "n": n, "value": (k / n if n else 0.0), "ci_low": lo, "ci_high": hi}
            from scipy import stats
            n = int(len(v))
            m = float(v.mean())
            sd = float(v.std(ddof=1)) if n > 1 else 0.0
            se = sd / (n ** 0.5) if n else 0.0
            tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 1.96        # t, not z, for a mean CI
            return {"arm": a, "n": n, "value": m, "ci_low": m - tcrit * se, "ci_high": m + tcrit * se}

        base_s = arm_summary(base)
        for a in others:
            a_s = arm_summary(a)
            if binary:
                ka, na = int(stat[a].sum()), int(len(stat[a]))
                kb, nb = int(stat[base].sum()), int(len(stat[base]))
                diff, lo, hi = gr.newcombe_diff_ci(ka, na, kb, nb)
                p = gr.two_proportion_p(ka, na, kb, nb)
            else:
                from scipy import stats
                va, vb = stat[a].to_numpy(), stat[base].to_numpy()
                diff = float(va.mean() - vb.mean())
                s2a, s2b, na_, nb_ = va.var(ddof=1), vb.var(ddof=1), len(va), len(vb)
                se = (s2a / na_ + s2b / nb_) ** 0.5
                denom = (s2a / na_) ** 2 / max(na_ - 1, 1) + (s2b / nb_) ** 2 / max(nb_ - 1, 1)
                dfw = ((s2a / na_ + s2b / nb_) ** 2 / denom) if denom > 0 else float(na_ + nb_ - 2)
                tcrit = float(stats.t.ppf(0.975, dfw)) if se > 0 else 1.96   # t-CI consistent with the Welch p
                lo, hi = diff - tcrit * se, diff + tcrit * se
                p = float(stats.ttest_ind(va, vb, equal_var=False).pvalue)
                if p != p:                      # zero-variance arms → NaN p: a null contrast, not significant
                    p = 1.0
            raw_ps.append(p)
            comps.append({"arm": a, "diff": diff, "lo": lo, "hi": hi})
            arm_rows.append(a_s)

        # multiple variants → BH-FDR adjust the comparison p-values
        multi = len(comps) > 1
        sig_ps = gr.benjamini_hochberg(raw_ps) if multi else raw_ps
        for c, sp in zip(comps, sig_ps):
            terms.append(Term(f"{c['arm']} vs {base}", c["diff"], c["lo"], c["hi"], sp))

        pos = [(c, sp) for c, sp in zip(comps, sig_ps) if c["lo"] > 0 and sp < 0.05]
        neg = [(c, sp) for c, sp in zip(comps, sig_ps) if c["hi"] < 0 and sp < 0.05]

        def rate(x):
            return f"{x * 100:.1f}%" if binary else f"{x:.2f}"

        if pos:
            cw, spw = max(pos, key=lambda x: x[0]["diff"])
            winner = cw["arm"]
            call = "SHIP"
            reason = (f"{winner} beats {base} by {rate(cw['diff'])} "
                      f"(95% CI {rate(cw['lo'])}–{rate(cw['hi'])}, "
                      f"{'q' if multi else 'p'}={spw:.3g}). Ship {winner}.")
        elif neg:
            cw = min(neg, key=lambda x: x[0]["diff"])[0]
            winner = None
            call = "DO NOT SHIP"
            reason = (f"{cw['arm']} is worse than {base} by {rate(abs(cw['diff']))} "
                      f"(95% CI {rate(cw['lo'])}–{rate(cw['hi'])}). Keep {base}.")
        else:
            winner = None
            call = "INCONCLUSIVE"
            promising = [(c, sp) for c, sp in zip(comps, sig_ps) if c["lo"] > 0]
            if promising and multi:                        # raw CI clears 0 but fails FDR
                cp, spp = max(promising, key=lambda x: x[0]["diff"])
                reason = (f"{cp['arm']} looks promising (+{rate(cp['diff'])}, 95% CI "
                          f"{rate(cp['lo'])}–{rate(cp['hi'])}) but does not survive multiple-comparison "
                          f"correction (q={spp:.3g}). Confirm in a powered follow-up before shipping.")
            else:
                reason = (f"No variant beats {base} at 95% confidence — the interval spans zero. "
                          f"Likely underpowered for the observed effect; keep {base} or extend the test.")

        # flag statistical issues
        issues = []
        if auto_base:
            issues.append(f"No control arm recognized by name — the largest arm '{base}' "
                          f"(n={int(_counts.get(base, 0))}) was used as the baseline. Pass an explicit "
                          "baseline if that's not the intended reference.")
        _bnote = _binary_note(d[outcome]) if binary else None
        if _bnote:
            issues.append(_bnote)
        sizes = [r["n"] for r in [base_s, *arm_rows]]
        if min(sizes) < 200:
            issues.append(f"Small arm (n={min(sizes)}) — the estimate is imprecise.")
        if max(sizes) / max(1, min(sizes)) > 1.5:
            issues.append(f"Imbalanced arms ({min(sizes)}–{max(sizes)}) — check the assignment/SRM.")
        if multi:
            issues.append(f"{len(comps)} variants compared — p-values are BH-FDR adjusted for "
                          "multiple comparisons.")
        if not winner and not neg:
            issues.append("No detectable effect — report the minimum detectable effect before calling it flat.")

        base_s["is_baseline"], base_s["is_winner"] = True, False
        for r in arm_rows:
            r["is_baseline"], r["is_winner"] = False, (r["arm"] == winner)
        all_arms = sorted([base_s, *arm_rows], key=lambda r: r["value"], reverse=True)

        mr = ModelResult("experiment", outcome, len(d),
                         "lift (Δ conversion)" if binary else "lift (Δ mean)", terms,
                         fit_stat=("two-proportion z-test" if binary else "Welch t-test")
                         + f"; baseline = {base}" + ("; BH-FDR" if multi else ""),
                         note="Ship/no-ship uses whether the lift's 95% CI clears zero. Observational "
                              "guardrails apply; data is synthetic.")
        mr.arms, mr.verdict, mr.issues = all_arms, {"call": call, "reason": reason}, issues
        return mr
    except Exception as e:  # noqa: BLE001
        return ModelResult("experiment", outcome, 0, "lift", error=str(e))


def fit_noninferiority(df: pd.DataFrame, group: str, outcome: str, margin: float | None,
                       higher_is_better: bool = True, control: str | None = None) -> ModelResult:
    """Non-inferiority test — is the treatment NOT worse than control by more than `margin`?
    Decision compares the two-sided 95% CI on (treatment − control) to the NI margin (one-sided α=0.025):
    higher-is-better → NI if the lower CI bound > −margin; lower-is-better → NI if upper bound < +margin.
    Also flags superiority when the CI additionally excludes zero in the favorable direction."""
    try:
        from . import guardrails as gr
        if margin is None:
            return ModelResult("noninferiority", outcome, 0, "difference",
                               error="A non-inferiority margin is required (on the outcome's scale).")
        margin = abs(float(margin))
        d = _clean(df, [group, outcome])
        d[group] = d[group].astype(str)
        arms = list(d[group].unique())
        if len(arms) != 2:
            return ModelResult("noninferiority", outcome, len(d), "difference",
                               error="Non-inferiority needs exactly two arms (treatment vs control).")
        binary = _is_binary_outcome(d[outcome])
        if not binary and not pd.api.types.is_numeric_dtype(d[outcome]):
            return ModelResult("noninferiority", outcome, len(d), "difference",
                               error=f"Outcome must be binary or numeric — found "
                                     f"{d[outcome].nunique()} non-numeric categories.")
        if not binary and int(d[group].value_counts().min()) < 2:
            return ModelResult("noninferiority", outcome, len(d), "difference",
                               error="Each arm needs ≥2 observations to compare a continuous outcome.")
        d = d.assign(_y=(_to_binary(d[outcome]) if binary else d[outcome].astype(float)).values)
        _counts = d[group].value_counts()
        base = control if control in arms else next((a for a in arms if _is_control(a)), None)
        auto_base = base is None
        if auto_base:                                # deterministic reference: the larger arm (typical control)
            base = max(sorted(arms), key=lambda a: int(_counts.get(a, 0)))
        trt = next(a for a in arms if a != base)
        st = {a: d.loc[d[group] == a, "_y"] for a in arms}

        def summ(a):
            v = st[a]
            if binary:
                k, n = int(v.sum()), int(len(v))
                lo, hi = gr.wilson_ci(k, n)
                return {"arm": a, "n": n, "value": (k / n if n else 0.0), "ci_low": lo, "ci_high": hi}
            from scipy import stats
            n = int(len(v))
            m, se = float(v.mean()), (float(v.std(ddof=1)) / n ** 0.5 if n > 1 else 0.0)
            tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 1.96
            return {"arm": a, "n": n, "value": m, "ci_low": m - tcrit * se, "ci_high": m + tcrit * se}

        if binary:
            kt, nt, kc, nc = int(st[trt].sum()), len(st[trt]), int(st[base].sum()), len(st[base])
            from statsmodels.stats.proportion import confint_proportions_2indep
            diff = (kt / nt if nt else 0.0) - (kc / nc if nc else 0.0)
            # score CI (Miettinen–Nurminen) — the SAME procedure family as the Farrington–Manning test
            # below, so the reported interval, the figure, and the NI verdict can never disagree.
            lo, hi = confint_proportions_2indep(kt, nt, kc, nc, compare="diff", method="score")
            lo, hi = float(lo), float(hi)
            test = "Miettinen–Nurminen score CI on the risk difference"
        else:
            from scipy import stats
            va, vb = st[trt].to_numpy(), st[base].to_numpy()
            diff = float(va.mean() - vb.mean())
            s2a, s2b, na_, nb_ = va.var(ddof=1), vb.var(ddof=1), len(va), len(vb)
            se = (s2a / na_ + s2b / nb_) ** 0.5
            denom = (s2a / na_) ** 2 / max(na_ - 1, 1) + (s2b / nb_) ** 2 / max(nb_ - 1, 1)
            dfw = ((s2a / na_ + s2b / nb_) ** 2 / denom) if denom > 0 else float(na_ + nb_ - 2)
            tcrit = float(stats.t.ppf(0.975, dfw)) if se > 0 else 1.96
            lo, hi = diff - tcrit * se, diff + tcrit * se
            test = "Welch t CI on the mean difference"

        fm_p = None
        if binary:                                   # Farrington–Manning score test IS the NI decision
            from statsmodels.stats.proportion import test_proportions_2indep
            val = -margin if higher_is_better else margin
            alt = "larger" if higher_is_better else "smaller"
            fm_p = float(test_proportions_2indep(kt, nt, kc, nc, value=val, compare="diff",
                                                 method="score", alternative=alt).pvalue)
            ni = fm_p < 0.025                         # one-sided α = 0.025
            test = "Farrington–Manning score test (NI) + Miettinen–Nurminen 95% CI"
        else:                                         # continuous → CI-vs-margin
            ni = (lo > -margin) if higher_is_better else (hi < margin)
        superior = (lo > 0) if higher_is_better else (hi < 0)
        bound = lo if higher_is_better else hi

        def fmt(x):
            return f"{x * 100:.1f}%" if binary else f"{x:.2f}"

        edge = "lower" if higher_is_better else "upper"
        # narrate the SAME test that made the decision — Farrington–Manning for binary, CI-vs-margin for
        # continuous — so the wording can never contradict the call (the Newcombe CI is reported alongside).
        if fm_p is not None:
            basis = f"the Farrington–Manning score test (p={fm_p:.3g}, one-sided α=0.025)"
        else:
            basis = f"the {edge} 95%-CI bound {fmt(bound)} relative to the {fmt(margin)} margin"
        if ni and superior:
            call = "NON-INFERIOR"
            reason = (f"{trt} is non-inferior to {base}, and superior: effect {fmt(diff)} "
                      f"(95% CI {fmt(lo)} to {fmt(hi)}); non-inferiority is met by {basis}, and the CI "
                      "excludes 0 in the favorable direction.")
        elif ni:
            call = "NON-INFERIOR"
            reason = (f"{trt} is non-inferior to {base}: effect {fmt(diff)} (95% CI {fmt(lo)} to {fmt(hi)}); "
                      f"non-inferiority is met by {basis}.")
        else:
            call = "NOT NON-INFERIOR"
            reason = (f"Non-inferiority not shown: effect {fmt(diff)} (95% CI {fmt(lo)} to {fmt(hi)}); "
                      f"the {fmt(margin)} margin is not excluded by {basis}.")

        rows = [summ(base), summ(trt)]
        rows[0]["is_baseline"], rows[0]["is_winner"] = True, False
        rows[1]["is_baseline"], rows[1]["is_winner"] = False, False   # NI ≠ "winner"; verdict says it all
        issues = []
        if auto_base:
            issues.append(f"No control arm recognized by name — the larger arm '{base}' "
                          f"(n={int(_counts.get(base, 0))}) was used as the reference. Pass an explicit "
                          "control if that's not the intended comparator.")
        _bnote = _binary_note(d[outcome]) if binary else None
        if _bnote:
            issues.append(_bnote)
        if binary and min(nt, nc) < 100:
            issues.append(f"Small arm (n={min(nt, nc)}) — the CI is wide; the NI call is fragile.")
        issues.append("NI is sensitive to the margin and analysis population — pre-specify the margin and "
                      "prefer the per-protocol set.")

        mr = ModelResult("noninferiority", outcome, len(d), "difference (treatment − control)",
                         [Term(f"{trt} − {base}", diff, lo, hi, float("nan"))],
                         fit_stat=f"{test}; margin {fmt(margin)} ({'higher' if higher_is_better else 'lower'} is better)",
                         note="NI decision compares the 95% CI bound to the margin (one-sided α=0.025). Synthetic data.")
        mr.arms = sorted(rows, key=lambda r: r["value"], reverse=True)
        mr.verdict = {"call": call, "reason": reason, "margin": margin,
                      "higher_is_better": higher_is_better, "fm_p": fm_p}
        mr.issues = issues
        return mr
    except Exception as e:  # noqa: BLE001
        return ModelResult("noninferiority", outcome, 0, "difference", error=str(e))


def calc_sample_size(kind: str = "superiority", outcome_type: str = "proportion",
                     p_control=None, p_treatment=None, effect=None, margin=None,
                     mean_control=None, mean_treatment=None, sd=None,
                     alpha: float = 0.05, power: float = 0.80, ratio: float = 1.0,
                     higher_is_better: bool = True) -> ModelResult:
    """Design-stage sample-size / power calculation (no data). Two-group superiority or
    non-inferiority, for a proportion or a mean endpoint. Proportions use the closed-form normal
    (Blackwelder) approximation; means use statsmodels' t-test power."""
    try:
        import math

        from scipy import stats
        # note: `x if x is not None else default` (NOT `x or default`) so an explicit 0 isn't swallowed
        alpha = float(alpha) if alpha is not None else 0.05
        power = float(power) if power is not None else 0.80
        ratio = float(ratio) if ratio is not None else 1.0
        ni = kind == "noninferiority"

        def _err(msg):
            return ModelResult("sample_size", "sample size", 0, "n per arm", error=msg)

        if not (0 < alpha < 1):
            return _err("alpha must be between 0 and 1 (e.g. 0.05).")
        if not (0 < power < 1):
            return _err("power must be between 0 and 1 (e.g. 0.80 or 0.90).")
        if ratio <= 0:
            return _err("allocation ratio must be positive (1 for equal arms, 2 for 2:1, …).")
        if ni and margin is None:
            return _err("a non-inferiority margin is required for an NI sample-size calculation.")
        za = stats.norm.ppf(1 - alpha / 2)          # two-sided superiority OR one-sided NI at α/2

        if outcome_type == "mean":
            if mean_control is None or sd is None:
                return _err("for a mean endpoint, provide mean_control and sd (plus mean_treatment or effect).")
            if not ni and mean_treatment is None and effect is None:
                return _err("for a superiority mean test, provide mean_treatment (or an effect).")
            m_c = float(mean_control)
            m_t = (float(mean_treatment) if mean_treatment is not None
                   else m_c + float(effect) if effect is not None else m_c)   # NI defaults to equal means
            s = float(sd)
            if s <= 0:
                return _err("the standard deviation (sd) must be positive.")
            dist = (abs(m_t - m_c) + abs(float(margin))) if ni else abs(m_t - m_c)
            if dist <= 0:
                return _err("the effect is zero — no finite sample size can detect a null difference.")
            d = dist / s
            from statsmodels.stats.power import TTestIndPower
            _alt = "larger" if ni else "two-sided"
            _a = alpha / 2 if ni else alpha              # NI is one-sided at α/2 (matches the za convention)
            _tip = TTestIndPower()

            def _n_ctrl(pw):                             # exact noncentral-t power → per-arm control n
                return float(_tip.solve_power(effect_size=d, alpha=_a, power=pw, ratio=ratio, alternative=_alt))
            detail = (f"means: control {m_c:g}, treatment {m_t:g}, SD {s:g}"
                      + (f", NI margin {abs(float(margin)):g}" if ni else "") + f" (effect size d={d:.2f})")
        else:                                        # proportion — Blackwelder normal approximation
            if p_control is None:
                return _err("for a proportion endpoint, provide p_control (plus p_treatment or effect).")
            if not ni and p_treatment is None and effect is None:
                return _err("for a superiority proportion test, provide p_treatment (or an effect).")
            p_c = float(p_control)
            p_t = (float(p_treatment) if p_treatment is not None
                   else p_c + float(effect) if effect is not None else p_c)   # NI defaults to equal rates
            if not (0 <= p_c <= 1) or not (0 <= p_t <= 1):
                return _err("proportions must be between 0 and 1.")
            null = (-abs(float(margin)) if higher_is_better else abs(float(margin))) if ni else 0.0
            dist = abs((p_t - p_c) - null)
            if dist <= 0:
                return _err("the target difference equals the margin — no finite sample size can detect it.")
            var = p_c * (1 - p_c) + p_t * (1 - p_t) / ratio
            if var <= 0:
                return _err("both proportions are 0 or 1 — the outcome has no variance, so the "
                            "normal-approximation sample size is undefined. Use rates strictly "
                            "between 0 and 1, or an exact/simulation-based method.")
            unit = var / dist ** 2

            def _n_ctrl(pw):                             # Blackwelder normal approximation → per-arm control n
                return (za + stats.norm.ppf(pw)) ** 2 * unit
            detail = (f"proportions: control {p_c:.0%}, treatment {p_t:.0%}"
                      + (f", NI margin {abs(float(margin)):.0%}" if ni else "") + f" (Δ {p_t - p_c:+.0%})")

        n_c = int(math.ceil(_n_ctrl(power)))
        n_t = int(math.ceil(n_c * ratio))
        total = n_c + n_t
        curve = [{"power": pw, "n": int(math.ceil(max(_n_ctrl(pw), _n_ctrl(pw) * ratio)))}
                 for pw in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)]
        side, side_val = ("one-sided α", alpha / 2) if ni else ("two-sided α", alpha)
        mr = ModelResult("sample_size", "sample size", total, "n per arm",
                         fit_stat=f"{kind}; {side}={side_val:g}; power={power:.0%}"
                         + (f"; {ratio:g}:1 allocation" if ratio != 1 else ""),
                         note="Noncentral-t power for means; normal (Blackwelder) approximation for "
                              "proportions. Assumes the stated effect/rates hold; for rare events or small n, "
                              "confirm with simulation. Inflate for expected dropout.")
        arms = [{"arm": "treatment", "n": n_t}, {"arm": "control", "n": n_c}]
        mr.arms = [dict(a, value=float(a["n"]), ci_low=float("nan"), ci_high=float("nan"),
                        is_baseline=(a["arm"] == "control"), is_winner=False) for a in arms]
        mr.verdict = {"call": f"{max(n_c, n_t):,} per arm  ·  total {total:,}",
                      "reason": f"To detect {detail} at {power:.0%} power ({side}={side_val:g}).",
                      "power": power}
        mr.series = curve
        return mr
    except Exception as e:  # noqa: BLE001
        return ModelResult("sample_size", "sample size", 0, "n per arm", error=str(e))


def _pretty_term(name: str) -> str:
    """'C(sex)[T.M]' → 'sex = M'; a continuous predictor name is returned as-is."""
    m = _CAT_TERM.match(str(name))
    return f"{m.group('col')} = {m.group('lvl')}" if m else str(name)


def _term_col(name: str) -> str:
    """The source column behind a term name ('C(sex)[T.M]' → 'sex'; 'age' → 'age')."""
    m = _CAT_TERM.match(str(name))
    return m.group("col") if m else str(name)


def specification_curve(model_type: str, df: pd.DataFrame, predictors: list[str], full: ModelResult, *,
                        outcome: str | None = None, duration: str | None = None,
                        event: str | None = None, max_specs: int = 12) -> dict:
    """Specification-curve / multiverse robustness for an adjusted effect (logistic / OLS / Cox).

    The garden of forking paths: the same data, analyzed with different-but-defensible covariate sets,
    can give different conclusions, and a study usually reports just one path. This refits the SAME model
    across a bounded, defensible multiverse of covariate choices — unadjusted, fully adjusted, and each
    leave-one-covariate-out — and reports whether the HEADLINE effect (its sign and significance) holds
    across all of them. Deterministic; no LLM. Returns {} when it can't run (no covariate to vary, no
    identifiable headline term, or fewer than three specifications fit)."""
    mt = "cox" if model_type in ("cox", "survival") else model_type
    if mt not in ("logistic", "ols", "cox"):
        return {}
    null = 0.0 if mt == "ols" else 1.0
    # headline = the term a reader would emphasize: the smallest-p, non-reference term of the full fit
    cand = [t for t in full.terms if t.p == t.p and not str(t.name).endswith("(ref)")]
    if not cand:
        return {}
    tracked = min(cand, key=lambda t: t.p).name
    primary_col = _term_col(tracked)
    covars = [p for p in predictors if p != primary_col]
    if not covars:                                   # only the exposure itself → nothing to adjust away
        return {}

    subsampled = len(df) > 20000                     # keep ≤12 refits interactive on large cohorts
    work = df.sample(20000, random_state=0) if subsampled else df

    def _fit(cols):
        if mt == "logistic":
            return fit_logistic(work, outcome, cols)
        if mt == "ols":
            return fit_ols(work, outcome, cols)
        return fit_cox(work, duration, event, cols)

    def _primary(mr):                                # the tracked term in a refit (levels/ref are stable)
        return next((t for t in mr.terms if t.name == tracked and t.estimate == t.estimate), None)

    loo = covars[: max(0, max_specs - 2)]
    truncated = len(loo) < len(covars)
    specs = [("fully adjusted", list(predictors)), ("unadjusted", [primary_col])]
    specs += [(f"drop {c}", [p for p in predictors if p != c]) for c in loo]
    seen, uniq = set(), []                           # collapse identical covariate sets (with one covariate,
    for label, cols in specs:                        # "unadjusted" and "drop <it>" are the same model)
        key = tuple(sorted(cols))
        if key not in seen:
            seen.add(key); uniq.append((label, cols))
    specs = uniq

    records = []
    for label, cols in specs:
        t = _primary(_fit(cols))
        if t is None:                                # the exposure was screened out in this spec — skip it
            continue
        records.append({"label": label, "estimate": float(t.estimate), "ci_low": float(t.ci_low),
                        "ci_high": float(t.ci_high), "p": float(t.p),
                        "significant": bool(t.p == t.p and t.p < 0.05)})
    if len(records) < 2:                             # need the anchor plus at least one variation
        return {}

    fa = next((r for r in records if r["label"] == "fully adjusted"), records[0])
    hdir = 1 if fa["estimate"] > null else -1
    for r in records:
        r["same_dir"] = (1 if r["estimate"] > null else -1) == hdir
    ests = [r["estimate"] for r in records]
    n_specs = len(records)
    n_same = sum(r["same_dir"] for r in records)
    n_sig = sum(1 for r in records if r["significant"] and r["same_dir"])
    sign_stable = n_same == n_specs
    agreement = n_sig / n_specs
    verdict = ("robust" if sign_stable and n_sig == n_specs
               else "mostly robust" if sign_stable and agreement >= 0.8
               else "fragile")

    pretty = _pretty_term(tracked)
    lbl = f"the {full.effect_label} for {pretty}"
    rng = f"{min(ests):.3f} to {max(ests):.3f}"
    sub = " (on a 20,000-row subsample)" if subsampled else ""
    summary = (f"{lbl} is {verdict} across {n_specs} defensible specifications{sub} (unadjusted, fully "
               f"adjusted, leave-one-covariate-out): same direction in {n_same}/{n_specs}, significant and "
               f"same-direction in {n_sig}/{n_specs}; estimate ranges {rng}.")
    caveat = ""
    if verdict == "fragile":
        caveat = (f"Specification-fragile: {lbl} does not hold across the covariate multiverse — significant "
                  f"and same-direction in only {n_sig} of {n_specs} defensible specifications, estimate "
                  f"ranges {rng}. The headline may hinge on one covariate choice (the garden of forking "
                  "paths); treat it as exploratory, not a stable finding.")
    return {"label": pretty, "effect_label": full.effect_label, "null": null, "n_specs": n_specs,
            "n_significant": n_sig, "n_same_direction": n_same, "agreement": round(agreement, 2),
            "sign_stable": sign_stable, "estimate_min": float(min(ests)),
            "estimate_median": float(np.median(ests)), "estimate_max": float(max(ests)),
            "headline_estimate": float(fa["estimate"]), "verdict": verdict, "summary": summary,
            "caveat": caveat, "truncated": truncated, "subsampled": subsampled, "specs": records}


def render(r: ModelResult) -> str:
    if r.error:
        return f"model ({r.model_type}) could not be fit: {r.error}"
    lines = [f"{r.model_type.upper()} · outcome: {r.outcome} · n={r.n:,}"
             + (f" · {r.fit_stat}" if r.fit_stat else "")]
    if r.model_type == "timeseries" and r.series:
        hist = [p for p in r.series if p["kind"] == "history"]
        fc = [p for p in r.series if p["kind"] == "forecast"]
        if hist:
            lines.append(f"  last observed {hist[-1]['time'][:10]}: {hist[-1]['value']:.1f}")
        for p in fc[:3] + ([fc[-1]] if len(fc) > 3 else []):
            lines.append(f"  forecast {p['time'][:10]}: {p['value']:.1f}  [{p['lower']:.1f}, {p['upper']:.1f}]")
    if r.model_type in ("experiment", "noninferiority") and r.arms:
        binm = all(0 <= a["value"] <= 1 for a in r.arms)
        if r.verdict:
            lines.append(f"  VERDICT: {r.verdict.get('call')} — {r.verdict.get('reason')}")
        for a in r.arms:
            tag = " (control)" if a.get("is_baseline") else (" ←" if a.get("is_winner") else "")
            val = f"{a['value'] * 100:.1f}%" if binm else f"{a['value']:.2f}"
            lines.append(f"  {a['arm']:16} n={a['n']:,}  {val}{tag}")
    for t in r.terms:
        ci = "" if np.isnan(t.ci_low) else f"  95% CI [{t.ci_low:.3f}, {t.ci_high:.3f}]"
        p = "" if np.isnan(t.p) else f"  p={t.p:.4f}" + (" *" if t.p < 0.05 else "")
        cnt = (f"  n={t.n:,}" + (f", events={t.events:,}" if t.events is not None else "")) if t.n is not None else ""
        lines.append(f"  {t.name:22} {r.effect_label}={t.estimate:.3f}{cnt}{ci}{p}")
    if r.robustness:
        lines.append(f"  ROBUSTNESS: {r.robustness['summary']}")
    for iss in r.issues:
        lines.append(f"  ! {iss}")
    if r.note:
        lines.append(f"  ({r.note})")
    return "\n".join(lines)


DISPATCH = {"logistic": fit_logistic, "ols": fit_ols, "cox": fit_cox, "association": test_association}


# ── Bayesian go/no-go ─────────────────────────────────────────────────────────────────────────────
def _build_prior(endpoint_type, tv, lrv, prior_successes, prior_n, prior_a, prior_b,
                 prior_mu, prior_sd) -> _bayes.Prior:
    """The informed prior, from a previous study if the question supplied one, else weakly informative."""
    if endpoint_type == "mean":
        if prior_mu is None:
            return _bayes.Prior("Vague", "normal", (float(lrv), 10.0 * (abs(tv - lrv) or 1.0)),
                                "Weakly informative (no prior study supplied); centred at the LRV.")
        return _bayes.Prior("Informed", "normal", (float(prior_mu), float(prior_sd or 1.0)),
                            f"Supplied prior: mean {prior_mu:g}, SD {prior_sd:g}.")
    if prior_a is not None and prior_b is not None:
        return _bayes.Prior("Informed", "beta", (float(prior_a), float(prior_b)),
                            f"Supplied prior: Beta({prior_a:g}, {prior_b:g}).")
    if prior_successes is not None and prior_n is not None:
        a, b = _bayes.beta_posterior(1.0, 1.0, int(prior_successes), int(prior_n))
        return _bayes.Prior("Phase-I informed", "beta", (a, b),
                            f"Beta({a:g}, {b:g}), from a uniform prior updated with the previous study: "
                            f"{int(prior_successes)} responses in {int(prior_n)} patients.")
    return _bayes.Prior("Vague", "beta", (1.0, 1.0),
                        "Uniform Beta(1,1) (no prior study supplied): every response rate equally likely.")


def _sensitivity(prior, n_planned, rule, sd) -> tuple[list[dict], bool]:
    """Re-decide under each defensible prior. A verdict that flips is FRAGILE, not an answer."""
    rows, calls = [], []
    for p in _bayes.prior_panel(prior, rule):
        if p.kind == "beta":
            a, b = p.params
            p_tv = _bayes.prob_exceeds("beta", a, b, rule.tv, rule.higher_is_better)
            p_lrv = _bayes.prob_exceeds("beta", a, b, rule.lrv, rule.higher_is_better)
        else:
            mu, s = p.params
            p_tv = _bayes.prob_exceeds("normal", mu, s, rule.tv, rule.higher_is_better)
            p_lrv = _bayes.prob_exceeds("normal", mu, s, rule.lrv, rule.higher_is_better)
        call, _ = _bayes.decide(float(p_tv), float(p_lrv), rule)
        rows.append({"prior": p.name, "params": [round(float(v), 3) for v in p.params],
                     "assurance": round(_bayes.assurance(p, n_planned, rule, sd), 4),
                     "call": call, "provenance": p.provenance})
        calls.append(call)
    return rows, len(set(calls)) > 1


def calc_assurance(endpoint_type: str = "proportion", framing: str = "single_arm",
                   n_planned=None, tv=None, lrv=None,
                   gate_tv: float = 0.80, gate_lrv: float = 0.90, stop_lrv: float = 0.10,
                   higher_is_better: bool = True,
                   prior_successes=None, prior_n=None, prior_a=None, prior_b=None,
                   prior_mu=None, prior_sd=None, sd=None, anchor=None) -> ModelResult:
    """Design-stage Bayesian go/no-go: the probability this trial ends in GO, before it runs.

    Classical power asks "what is the chance of success IF the true effect is exactly X". Assurance
    asks the question a decision-maker actually has: "given everything we believe about X, what is the
    chance this trial succeeds?" It is usually the lower, more honest number.
    """
    def _err(msg):
        return ModelResult("assurance", "go/no-go", 0, "probability of success", error=msg)
    try:
        if n_planned is None or tv is None or lrv is None:
            return _err("an assurance calculation needs a planned sample size, a target value (TV), "
                        "and a lower reference value (LRV).")
        n_planned = int(n_planned)
        tv, lrv = float(tv), float(lrv)
        if n_planned <= 0:
            return _err("the planned sample size must be positive.")
        if endpoint_type == "proportion" and not (0 <= tv <= 1 and 0 <= lrv <= 1):
            return _err("for a proportion endpoint the TV and LRV must be between 0 and 1 "
                        "(express 30% as 0.30).")
        if higher_is_better and lrv > tv:
            return _err("the LRV must not exceed the TV: the minimum worth pursuing cannot be more "
                        "ambitious than the value you hope for.")
        if not higher_is_better and tv > lrv:
            return _err("with a lower-is-better endpoint the TV must not exceed the LRV.")

        rule = _bayes.DecisionRule(tv=tv, lrv=lrv, gate_tv=float(gate_tv), gate_lrv=float(gate_lrv),
                                   stop_lrv=float(stop_lrv), higher_is_better=bool(higher_is_better))
        prior = _build_prior(endpoint_type, tv, lrv, prior_successes, prior_n,
                             prior_a, prior_b, prior_mu, prior_sd)

        # the verdict, from the prior alone -- this is a DESIGN question, there is no data yet
        a1, a2 = prior.params
        p_tv = float(_bayes.prob_exceeds(prior.kind, a1, a2, tv, higher_is_better))
        p_lrv = float(_bayes.prob_exceeds(prior.kind, a1, a2, lrv, higher_is_better))
        call, reason = _bayes.decide(p_tv, p_lrv, rule)

        assur = _bayes.assurance(prior, n_planned, rule, sd)
        oc = _bayes.operating_characteristics(prior, n_planned, rule, sd)
        t1, power = _bayes.type_i_and_power(prior, n_planned, rule, sd)
        panel, fragile = _sensitivity(prior, n_planned, rule, sd)

        params = {"endpoint_type": endpoint_type, "framing": framing, "n_planned": n_planned,
                  "tv": tv, "lrv": lrv, "gate_tv": gate_tv, "gate_lrv": gate_lrv,
                  "stop_lrv": stop_lrv, "higher_is_better": higher_is_better,
                  "prior_a": a1 if prior.kind == "beta" else None,
                  "prior_b": a2 if prior.kind == "beta" else None,
                  "prior_mu": a1 if prior.kind == "normal" else None,
                  "prior_sd": a2 if prior.kind == "normal" else None}
        lock = _prespec.create_lock(params, oc, anchor=anchor)

        mr = ModelResult("assurance", "go/no-go", n_planned, "probability of success",
                         fit_stat=f"assurance={assur:.1%} · n={n_planned:,} · TV={tv:g} / LRV={lrv:g}",
                         note="Design-stage Bayesian go/no-go. Assurance averages the chance of "
                              "success over the prior uncertainty about the true effect; it is not a "
                              "prediction about any one trial. Synthetic data.")
        mr.verdict = {"call": call, "reason": reason, "assurance": round(assur, 4),
                      "power": round(power, 4)}
        mr.series = [{"n": int(nn), "assurance": round(_bayes.assurance(prior, int(nn), rule, sd), 4)}
                     for nn in np.unique(np.linspace(10, max(20, n_planned * 2), 20).astype(int))]
        mr.robustness = {"panel": panel, "fragile": fragile, "oc": oc, "framing": framing,
                         "type_i_error": round(t1, 4), "power": round(power, 4)}
        mr.prespec = {"status": "PRE-SPECIFIED", "lock": lock, "drift": []}

        issues = [_prespec.caveat({"status": "PRE-SPECIFIED", "drift": []}),
                  f"Prior: {prior.provenance}"]
        if fragile:
            flips = ", ".join(f"{r['prior']} -> {r['call']}" for r in panel)
            issues.append(f"FRAGILE: the verdict is not stable across defensible priors ({flips}). "
                          "A skeptical reader would not yet be convinced; this is a prior-driven call, "
                          "not a data-driven one.")
        else:
            issues.append(f"Prior sensitivity: the {call} verdict HOLDS across all four defensible "
                          "priors (informed, vague, skeptical, enthusiastic).")
        issues.append(f"Operating characteristics: type I error {t1:.1%} (the chance of a GO when the "
                      f"true effect is only at the LRV) and power {power:.1%} (the chance of a GO when "
                      f"it is at the TV).")
        if prior.kind == "beta":
            ess = _bayes.prior_ess(prior)
            if ess > n_planned:
                issues.append(f"The prior carries an effective sample size of {ess:.0f}, MORE than the "
                              f"{n_planned:,} patients this trial will enrol: the prior is doing more "
                              "work than the evidence will. Justify it or weaken it.")
        if abs(power - assur) > 0.05:
            if assur < power:
                issues.append(f"Assurance ({assur:.1%}) is below classical power ({power:.1%}) because power "
                              "assumes the effect is exactly the TV, while assurance averages over the "
                              "uncertainty about it. Assurance is the number to budget against.")
            else:
                issues.append(f"Assurance ({assur:.1%}) EXCEEDS classical power at the TV ({power:.1%}) "
                              "because the informed prior is centred above the Target Value, so it expects "
                              "a larger effect than the TV. This makes the assurance depend heavily on that "
                              "optimistic prior; see the prior-sensitivity panel, and treat the classical "
                              "power at the TV as the more conservative planning number.")
        mr.issues = issues
        return mr
    except Exception as e:  # noqa: BLE001 — never raise into the app
        return _err(str(e))


def fit_interim(df: pd.DataFrame, outcome: str, n_planned=None, tv=None, lrv=None,
                gate_tv: float = 0.80, gate_lrv: float = 0.90, stop_lrv: float = 0.10,
                higher_is_better: bool = True,
                prior_successes=None, prior_n=None, prior_a=None, prior_b=None,
                lock=None, endpoint_type: str = "proportion",
                framing: str = "single_arm") -> ModelResult:
    """Interim Bayesian go/no-go: given the patients seen so far, will this trial end in GO?

    The predictive probability of success is the futility signal that stops a trial early and saves
    the money. Verified against the design lock, if one was supplied.
    """
    def _err(msg):
        return ModelResult("interim", outcome or "go/no-go", 0, "posterior (95% credible interval)",
                           error=msg)
    try:
        if n_planned is None or tv is None or lrv is None:
            return _err("an interim analysis needs the planned sample size, a target value (TV), and a "
                        "lower reference value (LRV).")
        n_planned = int(n_planned)
        tv, lrv = float(tv), float(lrv)
        if endpoint_type != "proportion":
            return _err("the interim analysis currently supports a binary endpoint only.")
        if higher_is_better and lrv > tv:
            return _err("the LRV must not exceed the TV.")
        if not (0 <= tv <= 1 and 0 <= lrv <= 1):
            return _err("the TV and LRV must be between 0 and 1 (express 30% as 0.30).")

        d = _clean(df, [outcome])
        if outcome not in d.columns or len(d) == 0:
            return _err("no observed subjects to analyse.")
        y = _to_binary(d[outcome])
        n_obs, x_obs = int(len(y)), int(y.sum())
        if n_obs > n_planned:
            return _err(f"{n_obs:,} subjects observed exceeds the planned enrolment of {n_planned:,}. "
                        "This is a final analysis, not an interim.")

        rule = _bayes.DecisionRule(tv=tv, lrv=lrv, gate_tv=float(gate_tv), gate_lrv=float(gate_lrv),
                                   stop_lrv=float(stop_lrv), higher_is_better=bool(higher_is_better))
        prior = _build_prior("proportion", tv, lrv, prior_successes, prior_n, prior_a, prior_b,
                             None, None)
        pa, pb = prior.params
        post_a, post_b = _bayes.beta_posterior(pa, pb, x_obs, n_obs)

        p_tv = float(_bayes.prob_exceeds("beta", post_a, post_b, tv, higher_is_better))
        p_lrv = float(_bayes.prob_exceeds("beta", post_a, post_b, lrv, higher_is_better))
        call, reason = _bayes.decide(p_tv, p_lrv, rule)
        ppos = _bayes.predictive_prob_success(prior, x_obs, n_obs, n_planned, rule)

        params = {"endpoint_type": "proportion", "framing": framing, "n_planned": n_planned,
                  "tv": tv, "lrv": lrv, "gate_tv": gate_tv, "gate_lrv": gate_lrv,
                  "stop_lrv": stop_lrv, "higher_is_better": higher_is_better,
                  "prior_a": pa, "prior_b": pb, "prior_mu": None, "prior_sd": None}
        ps = _prespec.verify(lock, params)

        mean = post_a / (post_a + post_b)
        lo, hi = stats.beta.ppf([0.025, 0.975], post_a, post_b)
        mr = ModelResult("interim", outcome, n_obs, "posterior rate (95% credible interval)",
                         [Term("response rate", float(mean), float(lo), float(hi), float("nan"))],
                         fit_stat=f"{x_obs}/{n_obs} observed · {n_planned - n_obs} still to enrol · "
                                  f"PPoS={ppos:.1%}",
                         note="Interim Bayesian go/no-go. The predictive probability of success is the "
                              "chance the trial ends in GO if it runs to full enrolment. Synthetic data.")
        # A futile trial is a STOP regardless of where the posterior sits today.
        if ppos < rule.stop_lrv:
            call = "STOP"
            reason = (f"Predictive probability of success is only {ppos:.1%}: even running to full "
                      f"enrolment ({n_planned:,}), this trial is very unlikely to clear its "
                      "pre-specified gates. Stop for futility.")
        mr.verdict = {"call": call, "reason": reason, "predictive_prob": round(ppos, 4),
                      "posterior_mean": round(float(mean), 4)}
        mr.series = [{"n": int(k),
                      "predictive_prob": round(_bayes.predictive_prob_success(
                          prior, int(round(mean * k)), int(k), n_planned, rule), 4)}
                     for k in np.unique(np.linspace(max(1, n_obs // 4), n_planned, 12).astype(int))]
        mr.prespec = {"status": ps["status"], "lock": lock, "drift": ps["drift"]}
        mr.robustness = {"framing": framing}

        issues = [_prespec.caveat(ps), f"Prior: {prior.provenance}"]
        if n_obs == n_planned:
            issues.append("Enrolment is complete, so this is the FINAL decision, not a prediction: the "
                          "predictive probability degenerates to the final GO/no-GO.")
        if (pa + pb) < 1.0 and x_obs in (0, n_obs):
            issues.append("UNRELIABLE / degenerate prior: a near-noninformative Beta prior becomes "
                          "unexpectedly INFORMATIVE when every subject so far is a success (or every one "
                          "a failure), which is exactly the case here. FDA's 2026 draft guidance warns "
                          "about this. Use a proper weakly-informative prior, e.g. Beta(1,1), and re-run.")
        if _bayes.prior_ess(prior) > n_obs:
            issues.append(f"The prior carries an effective sample size of {_bayes.prior_ess(prior):.0f}, "
                          f"more than the {n_obs:,} subjects observed so far: the prior is currently "
                          "doing more work than the data.")
        mr.issues = issues
        return mr
    except Exception as e:  # noqa: BLE001 — never raise into the app
        return _err(str(e))
