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


def test_to_binary():
    assert list(modeling._to_binary(pd.Series([True, False, True]))) == [1, 0, 1]
    assert list(modeling._to_binary(pd.Series([0, 1, 0]))) == [0, 1, 0]
