"""Unit tests for specification-curve / multiverse robustness (agent/modeling.specification_curve).

Deterministic, keyless: refit an adjusted effect across the covariate multiverse (unadjusted, fully
adjusted, leave-one-out) and check whether the headline holds. Data is generated with fixed seeds.
"""
import numpy as np
import pandas as pd

from agent import modeling


def test_robust_effect_holds_across_specifications():
    """A strong effect that doesn't depend on the covariates is reported as robust."""
    rng = np.random.default_rng(0)
    n = 2500
    x = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(1.4 * x + 0.2 * z1 - 0.1 * z2))))
    df = pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})
    full = modeling.fit_logistic(df, "y", ["x", "z1", "z2"])
    rb = modeling.specification_curve("logistic", df, ["x", "z1", "z2"], full, outcome="y")
    assert rb and rb["verdict"] == "robust"
    assert rb["label"] == "x"                       # the strong predictor is the tracked headline
    assert rb["sign_stable"] is True
    assert rb["n_significant"] == rb["n_specs"] and rb["n_specs"] >= 3
    assert not rb["caveat"]                          # a robust result carries no fragility caveat


def test_suppressed_effect_is_flagged_fragile():
    """An effect that is significant only when its counterpart is adjusted for is flagged fragile."""
    rng = np.random.default_rng(7)
    n = 110
    u = rng.normal(size=n)
    x = u + rng.normal(scale=0.3, size=n)
    z = u + rng.normal(scale=0.3, size=n)
    y = x - z + rng.normal(size=n)                  # each effect is masked when the other is dropped
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    full = modeling.fit_ols(df, "y", ["x", "z"])
    rb = modeling.specification_curve("ols", df, ["x", "z"], full, outcome="y")
    assert rb and rb["verdict"] == "fragile"
    assert rb["agreement"] < 1.0
    assert rb["caveat"]                              # fragile → a caveat the agent surfaces


def test_single_predictor_has_no_multiverse():
    """With nothing to adjust for, there is no covariate multiverse: returns an empty dict."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-x)))
    df = pd.DataFrame({"y": y, "x": x})
    full = modeling.fit_logistic(df, "y", ["x"])
    assert modeling.specification_curve("logistic", df, ["x"], full, outcome="y") == {}


def test_robustness_serializes_and_renders():
    """The robustness summary round-trips through as_dict() and shows up in render()."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(1.3 * x + 0.2 * z))))
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    full = modeling.fit_logistic(df, "y", ["x", "z"])
    full.robustness = modeling.specification_curve("logistic", df, ["x", "z"], full, outcome="y")
    assert full.robustness and full.robustness["verdict"] in ("robust", "mostly robust", "fragile")
    assert "robustness" in full.as_dict()
    assert "ROBUSTNESS" in modeling.render(full)
