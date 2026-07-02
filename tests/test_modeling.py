"""Unit tests for the inferential-modeling layer (synthetic data, known effects; no API key)."""
import numpy as np
import pandas as pd

from agent import modeling


def _logit_data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    g = rng.choice(["A", "B"], n)
    lp = -0.5 + 1.2 * x + 0.8 * (g == "B")
    y = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(int)
    return pd.DataFrame({"y": y, "x": x, "g": g})


def test_logistic_recovers_positive_effects():
    r = modeling.fit_logistic(_logit_data(), "y", ["x", "g"])
    assert r.error is None and r.model_type == "logistic" and r.n == 800
    terms = {t.name: t for t in r.terms}
    assert terms["x"].estimate > 1 and terms["x"].p < 0.05          # OR > 1 for a positive predictor
    gkey = next(k for k in terms if k.startswith("C(g)"))
    assert terms[gkey].estimate > 1                                  # level B has higher odds


def test_ols_recovers_slope():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"y": 3 + 2 * x + rng.normal(0, 0.5, n), "x": x})
    r = modeling.fit_ols(df, "y", ["x"])
    assert r.error is None
    xt = next(t for t in r.terms if t.name == "x")
    assert 1.8 < xt.estimate < 2.2 and xt.p < 0.001 and xt.ci_low < 2 < xt.ci_high


def test_cox_fits():
    rng = np.random.default_rng(2)
    n = 600
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"time": rng.exponential(np.exp(-0.5 * x)), "event": 1, "x": x})
    r = modeling.fit_cox(df, "time", "event", ["x"])
    assert r.error is None and r.model_type == "cox" and len(r.terms) >= 1


def test_association_chi2_and_ttest():
    rng = np.random.default_rng(3)
    n = 400
    a = rng.choice(["x", "y"], n)
    b = np.where(a == "x", rng.choice(["p", "q"], n, p=[0.8, 0.2]),
                 rng.choice(["p", "q"], n, p=[0.2, 0.8]))
    r = modeling.test_association(pd.DataFrame({"a": a, "b": b}), "a", "b")
    assert r.error is None and "chi" in r.effect_label and r.terms[0].p < 0.05

    g = rng.choice(["A", "B"], n)
    v = np.where(g == "A", rng.normal(0, 1, n), rng.normal(1, 1, n))
    r2 = modeling.test_association(pd.DataFrame({"v": v, "g": g}), "v", "g")
    assert r2.error is None and r2.terms[0].p < 0.05


def test_survival_km_and_cox():
    rng = np.random.default_rng(4)
    n = 500
    g = rng.choice(["A", "B"], n)
    df = pd.DataFrame({"time": rng.exponential(np.where(g == "B", 5.0, 10.0)), "event": 1, "g": g})
    mr = modeling.fit_survival(df, "time", "event", predictors=["g"], group="g")
    assert mr.error is None and mr.model_type == "survival"
    assert mr.terms and mr.km                                   # Cox HR + KM curve points
    assert {c["group"] for c in mr.km} == {"A", "B"}


def test_forest_ranks_informative_feature_top():
    rng = np.random.default_rng(5)
    n = 600
    signal = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    junk = rng.choice(["a", "b", "c"], n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.0 * signal)))).astype(int)
    df = pd.DataFrame({"y": y, "signal": signal, "noise": noise, "junk": junk})
    r = modeling.fit_forest(df, "y", ["signal", "noise", "junk"])
    assert r.error is None and r.model_type == "forest" and r.terms
    assert r.terms[0].name == "signal"                    # sorted desc by importance
    assert "AUC=" in r.fit_stat


def test_timeseries_forecasts_forward():
    rng = np.random.default_rng(6)
    n = 48
    periods = pd.date_range("2018-01-01", periods=n, freq="MS")
    trend = np.arange(n) * 2.0
    season = 10 * np.sin(np.arange(n) * 2 * np.pi / 12)
    df = pd.DataFrame({"period": periods, "n": 100 + trend + season + rng.normal(0, 2, n)})
    r = modeling.fit_timeseries(df, "period", "n", periods=6, seasonal_periods=12)
    assert r.error is None and r.model_type == "timeseries"
    fc = [p for p in r.series if p["kind"] == "forecast"]
    hist = [p for p in r.series if p["kind"] == "history"]
    assert len(fc) == 6 and len(hist) == n
    assert all(p["lower"] <= p["value"] <= p["upper"] for p in fc)


def test_uplift_recovers_positive_effect():
    rng = np.random.default_rng(7)
    n = 800
    x = rng.normal(0, 1, n)
    treat = rng.integers(0, 2, n)
    base = 1 / (1 + np.exp(-(0.5 * x)))
    y = (rng.random(n) < np.clip(base + 0.2 * treat, 0, 1)).astype(int)   # treatment raises risk ~0.2
    df = pd.DataFrame({"y": y, "treat": treat, "x": x})
    r = modeling.fit_uplift(df, "y", "treat", ["x"])
    assert r.error is None and r.model_type == "causal" and r.terms
    ate = r.terms[0]
    assert ate.estimate > 0 and ate.ci_low <= ate.estimate <= ate.ci_high


def test_experiment_ship_and_noship():
    rng = np.random.default_rng(8)
    # treatment clearly better → SHIP
    ctrl = pd.DataFrame({"variant": "control", "converted": (rng.random(3000) < 0.10).astype(int)})
    trt = pd.DataFrame({"variant": "treatment", "converted": (rng.random(3000) < 0.14).astype(int)})
    r = modeling.fit_experiment(pd.concat([ctrl, trt]), "variant", "converted")
    assert r.error is None and r.model_type == "experiment"
    assert r.verdict["call"] == "SHIP" and len(r.arms) == 2
    assert r.terms[0].ci_low > 0                                   # lift CI clears zero

    # treatment worse → DO NOT SHIP
    trt2 = pd.DataFrame({"variant": "treatment", "converted": (rng.random(3000) < 0.06).astype(int)})
    r2 = modeling.fit_experiment(pd.concat([ctrl, trt2]), "variant", "converted")
    assert r2.verdict["call"] == "DO NOT SHIP"


def test_experiment_inconclusive():
    rng = np.random.default_rng(9)
    a = pd.DataFrame({"variant": "control", "converted": (rng.random(2000) < 0.10).astype(int)})
    b = pd.DataFrame({"variant": "treatment", "converted": (rng.random(2000) < 0.105).astype(int)})
    r = modeling.fit_experiment(pd.concat([a, b]), "variant", "converted")
    assert r.error is None and r.verdict["call"] == "INCONCLUSIVE"


def test_noninferiority_concludes_and_fails():
    rng = np.random.default_rng(11)
    # treatment ~ control (within a generous margin) → NON-INFERIOR
    ctrl = pd.DataFrame({"arm": "control", "resp": (rng.random(2500) < 0.60).astype(int)})
    trt = pd.DataFrame({"arm": "treatment", "resp": (rng.random(2500) < 0.59).astype(int)})
    r = modeling.fit_noninferiority(pd.concat([ctrl, trt]), "arm", "resp", margin=0.10,
                                    higher_is_better=True)
    assert r.error is None and r.model_type == "noninferiority"
    assert r.verdict["call"] == "NON-INFERIOR"

    # treatment clearly worse than control, beyond the margin → NOT non-inferior
    trt2 = pd.DataFrame({"arm": "treatment", "resp": (rng.random(2500) < 0.45).astype(int)})
    r2 = modeling.fit_noninferiority(pd.concat([ctrl, trt2]), "arm", "resp", margin=0.05,
                                     higher_is_better=True)
    assert r2.verdict["call"] == "NOT NON-INFERIOR"


def test_noninferiority_lower_is_better():
    rng = np.random.default_rng(12)
    # adverse-event rate: treatment slightly lower than control, lower-is-better → NON-INFERIOR
    ctrl = pd.DataFrame({"arm": "control", "ae": (rng.random(3000) < 0.20).astype(int)})
    trt = pd.DataFrame({"arm": "treatment", "ae": (rng.random(3000) < 0.19).astype(int)})
    r = modeling.fit_noninferiority(pd.concat([ctrl, trt]), "arm", "ae", margin=0.05,
                                    higher_is_better=False)
    assert r.error is None and r.verdict["call"] == "NON-INFERIOR"


def test_to_binary():
    assert list(modeling._to_binary(pd.Series([True, False, True]))) == [1, 0, 1]
    assert list(modeling._to_binary(pd.Series([0, 1, 0]))) == [0, 1, 0]
