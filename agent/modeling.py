"""Inferential-modeling layer — fit real statistical models, not just SQL aggregation.

Given a patient-level analytic DataFrame (one row per unit, outcome + covariates as columns), fit a
model and return a structured result: effect estimates (odds/hazard ratios or coefficients) with
95% CIs and p-values, n, and a fit statistic. Deterministic — the LLM chooses the model and builds
the dataset, but the fitting and inference are done by statsmodels/scipy, so the numbers are
trustworthy. This is the "fit a covariate-adjusted model" the guardrail keeps recommending.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Term:
    name: str
    estimate: float          # odds ratio / hazard ratio / coefficient
    ci_low: float
    ci_high: float
    p: float


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


def _to_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        vals = set(pd.unique(s.dropna()))
        return s.astype(int) if vals <= {0, 1} else (s > 0).astype(int)
    cats = list(pd.unique(s.dropna()))
    if len(cats) == 2:
        return (s == cats[1]).astype(int)
    raise ValueError(f"outcome '{s.name}' is not binary ({len(cats)} levels)")


def _clean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.dropna(subset=[c for c in cols if c in df.columns]).copy()


def fit_logistic(df: pd.DataFrame, outcome: str, predictors: list[str]) -> ModelResult:
    """Logistic regression → adjusted odds ratios with 95% CIs (the classic confounding fix)."""
    try:
        import statsmodels.formula.api as smf
        d = _clean(df, [outcome, *predictors])
        d[outcome] = _to_binary(d[outcome])
        if d[outcome].nunique() < 2:
            return ModelResult("logistic", outcome, len(d), "odds ratio",
                               error="Outcome has no variation (all one class).")
        m = smf.logit(_formula(outcome, predictors, d), data=d).fit(disp=0)
        ci = m.conf_int()
        terms = [Term(name, float(np.exp(m.params[name])), float(np.exp(ci.loc[name, 0])),
                      float(np.exp(ci.loc[name, 1])), float(m.pvalues[name]))
                 for name in m.params.index if name != "Intercept"]
        return ModelResult("logistic", outcome, int(m.nobs), "odds ratio", terms,
                           fit_stat=f"pseudo-R²={m.prsquared:.3f}",
                           note="Odds ratio > 1 = higher odds of the outcome, holding the others fixed.")
    except Exception as e:  # noqa: BLE001
        return ModelResult("logistic", outcome, 0, "odds ratio", error=str(e))


def fit_ols(df: pd.DataFrame, outcome: str, predictors: list[str]) -> ModelResult:
    """Linear regression → adjusted coefficients with 95% CIs for a continuous outcome."""
    try:
        import statsmodels.formula.api as smf
        d = _clean(df, [outcome, *predictors])
        m = smf.ols(_formula(outcome, predictors, d), data=d).fit()
        ci = m.conf_int()
        terms = [Term(name, float(m.params[name]), float(ci.loc[name, 0]),
                      float(ci.loc[name, 1]), float(m.pvalues[name]))
                 for name in m.params.index if name != "Intercept"]
        return ModelResult("ols", outcome, int(m.nobs), "coefficient", terms,
                           fit_stat=f"R²={m.rsquared:.3f}",
                           note="Coefficient = change in the outcome per 1-unit change, others fixed.")
    except Exception as e:  # noqa: BLE001
        return ModelResult("ols", outcome, 0, "coefficient", error=str(e))


def fit_cox(df: pd.DataFrame, duration: str, event: str, predictors: list[str]) -> ModelResult:
    """Cox proportional-hazards → adjusted hazard ratios (time-to-event / survival)."""
    try:
        from statsmodels.duration.hazard_regression import PHReg
        d = _clean(df, [duration, event, *predictors])
        d[event] = _to_binary(d[event])
        d = d[pd.to_numeric(d[duration], errors="coerce") > 0]   # survival durations must be positive
        if len(d) < 2:
            return ModelResult("cox", duration, len(d), "hazard ratio",
                               error="No positive follow-up durations to fit a Cox model.")
        mod = PHReg.from_formula(_formula(duration, predictors, d), data=d, status=d[event])
        r = mod.fit()
        ci = r.conf_int()
        terms = [Term(name, float(np.exp(r.params[i])), float(np.exp(ci[i, 0])),
                      float(np.exp(ci[i, 1])), float(r.pvalues[i]))
                 for i, name in enumerate(r.model.exog_names)]
        return ModelResult("cox", duration, int(r.model.surv.n_obs), "hazard ratio", terms,
                           fit_stat=f"events={int(d[event].sum())}",
                           note="Hazard ratio > 1 = faster time-to-event (higher risk), others fixed.")
    except Exception as e:  # noqa: BLE001
        return ModelResult("cox", duration, 0, "hazard ratio", error=str(e))


def fit_km(df: pd.DataFrame, duration: str, event: str, group: str | None = None) -> list[dict]:
    """Kaplan-Meier survival curve points, optionally stratified by a categorical group."""
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
    if pd.api.types.is_numeric_dtype(y):
        return set(pd.unique(y.dropna())) <= {0, 1}
    return y.nunique(dropna=True) == 2


def fit_forest(df: pd.DataFrame, outcome: str, predictors: list[str]) -> ModelResult:
    """Random forest → permutation feature importances: which factors most predict the outcome.
    Classifier for a binary outcome (scored by AUC), regressor for a continuous one (scored by R²)."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import r2_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        preds = [p for p in predictors if p in df.columns and p != outcome]
        d = _clean(df, [outcome, *preds])
        if len(d) < 40 or len(preds) < 2:
            return ModelResult("forest", outcome, len(d), "importance",
                               error="Need ≥40 rows and ≥2 candidate predictors for a random forest.")
        X = pd.get_dummies(d[preds], drop_first=True)
        is_class = _is_binary_outcome(d[outcome])
        common = dict(n_estimators=300, random_state=0, n_jobs=-1, min_samples_leaf=5)
        if is_class:
            y = _to_binary(d[outcome])
            if y.nunique() < 2:
                return ModelResult("forest", outcome, len(d), "importance",
                                   error="Outcome has no variation (all one class).")
            model, scoring = RandomForestClassifier(**common), "roc_auc"
        else:
            y = d[outcome].astype(float)
            model, scoring = RandomForestRegressor(**common), "r2"
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                              stratify=y if is_class else None)
        model.fit(Xtr, ytr)
        if is_class:
            fit_stat = f"AUC={roc_auc_score(yte, model.predict_proba(Xte)[:, 1]):.3f}"
        else:
            fit_stat = f"R²={r2_score(yte, model.predict(Xte)):.3f}"
        imp = permutation_importance(model, Xte, yte, n_repeats=12, random_state=0, scoring=scoring)
        means = pd.Series(imp.importances_mean, index=X.columns)
        terms = []                                  # roll one-hot columns back up to the source predictor
        for p in preds:
            cols = [c for c in X.columns if c == p or c.startswith(f"{p}_")]
            terms.append(Term(p, float(means[cols].sum()) if cols else 0.0,
                              float("nan"), float("nan"), float("nan")))
        terms.sort(key=lambda t: t.estimate, reverse=True)
        return ModelResult("forest", outcome, len(d), "importance", terms, fit_stat=fit_stat,
                           note="Permutation importance = how much model skill drops when a feature is "
                                "shuffled (bigger = more predictive). Predictive, not causal.")
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


def fit_uplift(df: pd.DataFrame, outcome: str, treatment: str,
               predictors: list[str] | None = None) -> ModelResult:
    """Causal T-learner: two random forests (treated vs control) estimate the average uplift (ATE) of a
    binary treatment on the outcome, adjusting for covariates. Observational → illustrative, not RCT-grade."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        preds = [p for p in (predictors or []) if p in df.columns and p not in (outcome, treatment)]
        d = _clean(df, [outcome, treatment, *preds])
        t = _to_binary(d[treatment])
        if t.nunique() < 2 or (t == 1).sum() < 20 or (t == 0).sum() < 20:
            return ModelResult("causal", outcome, len(d), "uplift",
                               error="Treatment needs ≥20 treated and ≥20 control rows.")
        is_class = _is_binary_outcome(d[outcome])
        y = _to_binary(d[outcome]) if is_class else d[outcome].astype(float)
        X = pd.get_dummies(d[preds], drop_first=True) if preds else pd.DataFrame(index=d.index)
        if X.empty:
            X = pd.DataFrame({"_const": np.ones(len(d))}, index=d.index)
        make = RandomForestClassifier if is_class else RandomForestRegressor
        kw = dict(n_estimators=300, random_state=0, n_jobs=-1, min_samples_leaf=5)
        if is_class and (y[t == 1].nunique() < 2 or y[t == 0].nunique() < 2):
            return ModelResult("causal", outcome, len(d), "uplift",
                               error="Outcome has no variation within a treatment arm.")
        m1, m0 = make(**kw).fit(X[t == 1], y[t == 1]), make(**kw).fit(X[t == 0], y[t == 0])
        pred = (lambda m: m.predict_proba(X)[:, 1]) if is_class else (lambda m: m.predict(X))
        uplift = pred(m1) - pred(m0)
        ate = float(np.mean(uplift))
        rng = np.random.default_rng(0)                    # bootstrap CI on the average uplift
        boots = [float(np.mean(uplift[rng.integers(0, len(uplift), len(uplift))])) for _ in range(300)]
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        label = "uplift (Δ risk)" if is_class else "uplift (Δ outcome)"
        return ModelResult("causal", outcome, len(d), label,
                           [Term(f"effect of {treatment}", ate, lo, hi, float("nan"))],
                           fit_stat=f"treated={(t == 1).sum():,} / control={(t == 0).sum():,}",
                           note="T-learner uplift = predicted outcome if treated minus if untreated, "
                                "averaged over patients. Observational — residual confounding may remain; "
                                "NOT a randomized causal effect. Synthetic data.")
    except Exception as e:  # noqa: BLE001
        return ModelResult("causal", outcome, 0, "uplift", error=str(e))


_BASELINE_NAMES = {"control", "baseline", "ctrl", "a", "off", "holdout", "0", "original", "default"}


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
        y = _to_binary(d[outcome]) if binary else d[outcome].astype(float)
        d = d.assign(_y=y.values)

        # pick the control/baseline arm: an obvious name, else the largest arm
        base = baseline if baseline in arms else next(
            (a for a in arms if a.lower() in _BASELINE_NAMES),
            d[group].value_counts().idxmax())
        others = [a for a in arms if a != base]

        stat = {a: d.loc[d[group] == a, "_y"] for a in arms}
        arm_rows, terms, raw_ps, comps = [], [], [], []

        def arm_summary(a):
            v = stat[a]
            if binary:
                k, n = int(v.sum()), int(len(v))
                lo, hi = gr.wilson_ci(k, n)
                return {"arm": a, "n": n, "value": (k / n if n else 0.0), "ci_low": lo, "ci_high": hi}
            n = int(len(v))
            m, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
            se = sd / (n ** 0.5) if n else 0.0
            return {"arm": a, "n": n, "value": m, "ci_low": m - 1.96 * se, "ci_high": m + 1.96 * se}

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
                se = (va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb)) ** 0.5
                lo, hi = diff - 1.96 * se, diff + 1.96 * se
                p = float(stats.ttest_ind(va, vb, equal_var=False).pvalue)
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
        d = d.assign(_y=(_to_binary(d[outcome]) if binary else d[outcome].astype(float)).values)
        base = control if control in arms else next(
            (a for a in arms if a.lower() in _BASELINE_NAMES), d[group].value_counts().idxmax())
        trt = next(a for a in arms if a != base)
        st = {a: d.loc[d[group] == a, "_y"] for a in arms}

        def summ(a):
            v = st[a]
            if binary:
                k, n = int(v.sum()), int(len(v))
                lo, hi = gr.wilson_ci(k, n)
                return {"arm": a, "n": n, "value": (k / n if n else 0.0), "ci_low": lo, "ci_high": hi}
            n = int(len(v))
            m, se = float(v.mean()), (float(v.std(ddof=1)) / n ** 0.5 if n > 1 else 0.0)
            return {"arm": a, "n": n, "value": m, "ci_low": m - 1.96 * se, "ci_high": m + 1.96 * se}

        if binary:
            kt, nt, kc, nc = int(st[trt].sum()), len(st[trt]), int(st[base].sum()), len(st[base])
            diff, lo, hi = gr.newcombe_diff_ci(kt, nt, kc, nc)
            test = "Newcombe CI on the risk difference"
        else:
            va, vb = st[trt].to_numpy(), st[base].to_numpy()
            diff = float(va.mean() - vb.mean())
            se = (va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb)) ** 0.5
            lo, hi = diff - 1.96 * se, diff + 1.96 * se
            test = "Welch CI on the mean difference"

        if higher_is_better:
            ni, superior, bound = lo > -margin, lo > 0, lo
        else:
            ni, superior, bound = hi < margin, hi < 0, hi

        def fmt(x):
            return f"{x * 100:.1f}%" if binary else f"{x:.2f}"

        edge = "lower" if higher_is_better else "upper"
        if ni and superior:
            call = "NON-INFERIOR"
            reason = (f"{trt} is non-inferior to {base} — and superior: effect {fmt(diff)} "
                      f"(95% CI {fmt(lo)} to {fmt(hi)}) stays inside the {fmt(margin)} margin and excludes 0.")
        elif ni:
            call = "NON-INFERIOR"
            reason = (f"{trt} is non-inferior to {base}: effect {fmt(diff)} (95% CI {fmt(lo)} to {fmt(hi)}); "
                      f"the {edge} bound {fmt(bound)} stays inside the {fmt(margin)} margin.")
        else:
            call = "NOT NON-INFERIOR"
            reason = (f"Non-inferiority not shown: effect {fmt(diff)} (95% CI {fmt(lo)} to {fmt(hi)}) "
                      f"crosses the {fmt(margin)} margin.")

        rows = [summ(base), summ(trt)]
        rows[0]["is_baseline"], rows[0]["is_winner"] = True, False
        rows[1]["is_baseline"], rows[1]["is_winner"] = False, False   # NI ≠ "winner"; verdict says it all
        issues = []
        if binary and min(nt, nc) < 100:
            issues.append(f"Small arm (n={min(nt, nc)}) — the CI is wide; the NI call is fragile.")
        issues.append("NI is sensitive to the margin and analysis population — pre-specify the margin and "
                      "prefer the per-protocol set.")

        mr = ModelResult("noninferiority", outcome, len(d), "difference (treatment − control)",
                         [Term(f"{trt} − {base}", diff, lo, hi, float("nan"))],
                         fit_stat=f"{test}; margin {fmt(margin)} ({'higher' if higher_is_better else 'lower'} is better)",
                         note="NI decision compares the 95% CI bound to the margin (one-sided α=0.025). Synthetic data.")
        mr.arms = sorted(rows, key=lambda r: r["value"], reverse=True)
        mr.verdict = {"call": call, "reason": reason, "margin": margin, "higher_is_better": higher_is_better}
        mr.issues = issues
        return mr
    except Exception as e:  # noqa: BLE001
        return ModelResult("noninferiority", outcome, 0, "difference", error=str(e))


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
        for iss in r.issues:
            lines.append(f"  ! {iss}")
    for t in r.terms:
        ci = "" if np.isnan(t.ci_low) else f"  95% CI [{t.ci_low:.3f}, {t.ci_high:.3f}]"
        p = "" if np.isnan(t.p) else f"  p={t.p:.4f}" + (" *" if t.p < 0.05 else "")
        lines.append(f"  {t.name:22} {r.effect_label}={t.estimate:.3f}{ci}{p}")
    if r.note:
        lines.append(f"  ({r.note})")
    return "\n".join(lines)


DISPATCH = {"logistic": fit_logistic, "ols": fit_ols, "cox": fit_cox, "association": test_association}
