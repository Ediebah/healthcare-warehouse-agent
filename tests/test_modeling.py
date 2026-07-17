"""Unit tests for the inferential-modeling layer (synthetic data, known effects; no API key)."""
import numpy as np
import pandas as pd
import pytest

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
    gkey = next(k for k in terms if k.startswith("C(g)[T."))        # the dummy level, not the reference row
    assert terms[gkey].estimate > 1                                  # level B has higher odds
    assert terms["C(g)[A] (ref)"].estimate == 1.0                   # reference level shown as OR=1


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


def test_sample_size_matches_textbook():
    # superiority, two proportions 0.70 vs 0.80 → ~291 per arm (textbook ~293)
    r = modeling.calc_sample_size(kind="superiority", outcome_type="proportion",
                                  p_control=0.70, p_treatment=0.80)
    assert r.error is None and r.model_type == "sample_size"
    per_arm = r.arms[0]["n"]
    assert 285 <= per_arm <= 300
    # non-inferiority, equal cure 0.85, margin 0.10 → ~201 per arm
    r2 = modeling.calc_sample_size(kind="noninferiority", outcome_type="proportion",
                                   p_control=0.85, margin=0.10)
    assert 190 <= r2.arms[0]["n"] <= 215
    # means, effect size d=0.5 → ~63-64 per arm
    r3 = modeling.calc_sample_size(kind="superiority", outcome_type="mean",
                                   mean_control=0.0, mean_treatment=0.5, sd=1.0)
    assert 60 <= r3.arms[0]["n"] <= 66
    # power curve is monotonically non-decreasing in power
    ns = [p["n"] for p in r.series]
    assert ns == sorted(ns)


def test_prepare_removes_collinearity_before_fitting():
    rng = np.random.default_rng(20)
    n = 500
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"y": (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int),
                       "x": x, "x_twin": 2 * x + 3, "z": rng.normal(0, 1, n)})  # x_twin ≡ x
    r = modeling.fit_logistic(df, "y", ["x", "x_twin", "z"])
    kept = {t.name for t in r.terms}
    assert not ({"x", "x_twin"} <= kept)              # both perfect-collinear twins cannot remain
    assert any("collinear" in i.lower() for i in r.issues)


def test_logistic_flags_separation_and_nonlinearity():
    rng = np.random.default_rng(21)
    n = 1200
    x = rng.normal(0, 1, n)
    r = modeling.fit_logistic(pd.DataFrame({"y": (x > 0).astype(int), "x": x}), "y", ["x"])
    assert any("separation" in i.lower() for i in r.issues)
    x = rng.normal(0, 1, n)                            # log-odds quadratic in x → non-linear
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1 + 1.5 * x ** 2)))).astype(int)
    r2 = modeling.fit_logistic(pd.DataFrame({"y": y, "x": x}), "y", ["x"])
    assert any("linear" in i.lower() for i in r2.issues)


def test_ols_flags_heteroskedasticity():
    rng = np.random.default_rng(22)
    n = 1000
    x = rng.uniform(1, 10, n)
    r = modeling.fit_ols(pd.DataFrame({"y": 2 * x + rng.normal(0, x, n), "x": x}), "y", ["x"])
    assert any("hetero" in i.lower() for i in r.issues)


def test_cox_flags_ph_violation():
    rng = np.random.default_rng(23)
    n = 1000
    g = rng.integers(0, 2, n)
    t = np.where(g == 0, rng.uniform(0.1, 5, n), rng.uniform(5, 10, n))   # HR changes over time
    r = modeling.fit_cox(pd.DataFrame({"t": t, "e": 1, "g": g.astype(float)}), "t", "e", ["g"])
    assert any("proportional-hazards" in i.lower() for i in r.issues)


def test_sample_size_rejects_degenerate_proportions():
    # p=1.0 (or 0.0) in both arms → zero variance → the closed form returns "0 per arm";
    # that must be an error, not a confident nonsense answer
    r = modeling.calc_sample_size(kind="noninferiority", outcome_type="proportion",
                                  p_control=1.0, margin=0.05)
    assert r.error is not None and "0 or 1" in r.error


def test_to_binary():
    assert list(modeling._to_binary(pd.Series([True, False, True]))) == [1, 0, 1]
    assert list(modeling._to_binary(pd.Series([0, 1, 0]))) == [0, 1, 0]


def test_to_binary_strings_order_invariant():
    # order of first appearance must NOT decide the positive class — it silently flipped
    # effect directions when the first row happened to hold the other level
    fwd = pd.Series(["cured", "failed", "cured", "failed"])
    rev = pd.Series(["failed", "cured", "cured", "failed"])
    assert list(modeling._to_binary(fwd)) == [1, 0, 1, 0]
    assert list(modeling._to_binary(rev)) == [0, 1, 1, 0]


def test_to_binary_recognizes_semantic_labels():
    assert list(modeling._to_binary(pd.Series(["no", "yes"]))) == [0, 1]
    assert list(modeling._to_binary(pd.Series(["yes", "no"]))) == [1, 0]
    assert list(modeling._to_binary(pd.Series(["alive", "dead", "alive"]))) == [0, 1, 0]


def test_to_binary_unrecognized_strings_deterministic():
    # nothing in the names says which is the event → lexicographically LAST level, any row order
    assert list(modeling._to_binary(pd.Series(["zeta", "alpha"]))) == [1, 0]
    assert list(modeling._to_binary(pd.Series(["alpha", "zeta"]))) == [0, 1]


def test_binary_note_on_string_coding():
    # a 2-level text outcome must state its mapping loudly (numeric {1,2} already did)
    note = modeling._binary_note(pd.Series(["cured", "failed"]))
    assert note is not None and "cured" in note and "failed" in note
    # unrecognized labels must additionally warn that the direction is a guess
    note2 = modeling._binary_note(pd.Series(["alpha", "zeta"]))
    assert note2 is not None and "zeta" in note2


def test_experiment_string_outcome_direction():
    rng = np.random.default_rng(21)
    # 'cured' is the event; a 'cured' row appearing FIRST must not flip the rates (the old bug
    # coded the SECOND-appearing level as the event, turning a winning arm into DO NOT SHIP)
    first = pd.DataFrame({"variant": ["control"], "outcome": ["cured"]})
    ctrl = pd.DataFrame({"variant": "control",
                         "outcome": np.where(rng.random(2500) < 0.30, "cured", "failed")})
    trt = pd.DataFrame({"variant": "treatment",
                        "outcome": np.where(rng.random(2500) < 0.45, "cured", "failed")})
    r = modeling.fit_experiment(pd.concat([first, ctrl, trt], ignore_index=True), "variant", "outcome")
    assert r.error is None
    rates = {a["arm"]: a["value"] for a in r.arms}
    assert rates["treatment"] > rates["control"] > 0.2      # cured-rate, not failed-rate
    assert r.verdict["call"] == "SHIP"
    assert any("cured" in i for i in r.issues)              # the coding is stated to the user


def test_uplift_subsample_preserves_and_rechecks_arms(monkeypatch):
    # the ≥20-per-arm gate used to run only BEFORE the tractability subsample: a rare treatment
    # could pass the gate and then fit on a handful of treated rows. The subsample must be
    # stratified by arm and the gate re-checked afterwards.
    monkeypatch.setattr(modeling, "_UPLIFT_MAX_ROWS", 200)
    rng = np.random.default_rng(23)
    n = 1000
    df = pd.DataFrame({"y": rng.integers(0, 2, n).astype(float),
                       "t": np.r_[np.ones(25), np.zeros(n - 25)].astype(int),
                       "x": rng.normal(0, 1, n)})
    r = modeling.fit_uplift(df, "y", "t", ["x"])                 # 25 treated → ~5 after subsample
    assert r.error is not None and "subsampl" in r.error.lower()


def test_uplift_string_treatment_control_recognition():
    rng = np.random.default_rng(22)
    n = 900
    x = rng.normal(0, 1, n)
    treated = rng.integers(0, 2, n)
    base = 1 / (1 + np.exp(-(0.5 * x)))
    y = (rng.random(n) < np.clip(base + 0.2 * treated, 0, 1)).astype(int)
    df = pd.DataFrame({"y": y, "arm": np.where(treated == 1, "drug", "placebo"), "x": x})
    df = df.sort_values("arm").reset_index(drop=True)       # 'drug' rows first → old code flipped
    r = modeling.fit_uplift(df, "y", "arm", ["x"])
    assert r.error is None
    assert r.terms[0].estimate > 0                          # drug raises y; sign must not flip
    assert any("drug" in i and "placebo" in i for i in r.issues)   # mapping stated to the user


# ── Bayesian go/no-go: design stage ───────────────────────────────────────────────────────────────
def test_assurance_verdict_flips_from_go_to_stop_as_the_bar_rises():
    # a strongly positive Phase I (16/20) against a modest bar -> GO
    go = modeling.calc_assurance(n_planned=100, tv=0.30, lrv=0.15,
                                 prior_successes=16, prior_n=20)
    assert go.error is None and go.verdict["call"] == "GO"
    # the same evidence against a bar nobody could clear -> STOP
    stop = modeling.calc_assurance(n_planned=100, tv=0.99, lrv=0.98,
                                   prior_successes=16, prior_n=20)
    assert stop.verdict["call"] == "STOP"


def test_assurance_reports_the_prior_and_its_provenance():
    r = modeling.calc_assurance(n_planned=100, tv=0.30, lrv=0.15, prior_successes=8, prior_n=20)
    assert r.error is None
    joined = " ".join(r.issues)
    assert "Beta(9" in joined and "8" in joined and "20" in joined     # Beta(1,1) + 8/20 -> Beta(9,13)


def test_assurance_flags_an_underpowered_design():
    # a GO whose power at the TV is far below 80% must be flagged UNDER-POWERED, not silently passed.
    # Phase I 8/20 vs TV 0.30 / LRV 0.15 at n=100 -> GO but power ~29%.
    r = modeling.calc_assurance(n_planned=100, tv=0.30, lrv=0.15, prior_successes=8, prior_n=20)
    assert r.error is None and r.verdict["call"] == "GO"
    assert r.robustness["under_powered"] is True
    joined = " ".join(r.issues).lower()
    assert "under-powered" in joined
    assert any(row["prior"] == "Skeptical" for row in r.robustness["panel"])   # panel retained


def test_assurance_does_not_flag_a_well_powered_design():
    # a GO with power >= 80% at the TV must NOT be flagged under-powered, and carries no FRAGILE text.
    # Phase I 16/20 -> GO with power ~89%.
    r = modeling.calc_assurance(n_planned=100, tv=0.30, lrv=0.15, prior_successes=16, prior_n=20)
    assert r.error is None and r.verdict["call"] == "GO"
    assert r.robustness["under_powered"] is False
    joined = " ".join(r.issues).lower()
    assert "under-powered" not in joined and "fragile" not in joined
    assert len(r.robustness["panel"]) == 4                                     # panel retained


def test_assurance_emits_operating_characteristics():
    r = modeling.calc_assurance(n_planned=100, tv=0.30, lrv=0.15, prior_successes=8, prior_n=20)
    assert 0.0 <= r.robustness["type_i_error"] <= 1.0
    assert 0.0 <= r.robustness["power"] <= 1.0
    joined = " ".join(r.issues).lower()
    assert "type i error" in joined


def test_assurance_flags_a_prior_stronger_than_the_planned_data():
    # a 200-observation prior against a 20-patient trial: the prior is doing the work
    r = modeling.calc_assurance(n_planned=20, tv=0.30, lrv=0.15, prior_successes=70, prior_n=200)
    assert any("prior" in i.lower() and "more" in i.lower() for i in r.issues)


def test_assurance_attaches_a_valid_lock():
    from agent import prespec
    r = modeling.calc_assurance(n_planned=100, tv=0.30, lrv=0.15, prior_successes=8, prior_n=20)
    lock = r.prespec["lock"]
    assert prespec.verify(lock, lock["params"])["status"] == "PRE-SPECIFIED"


def test_assurance_device_performance_goal_collapses_to_go_no_go():
    # tv == lrv == the performance goal (0.85): Beta(89,13) puts P(above 0.85) at 76.5%, short of
    # both the 80% TV gate and the 90% LRV gate -> CONSIDER, not a clean GO/STOP
    r = modeling.calc_assurance(n_planned=150, tv=0.85, lrv=0.85, prior_successes=88, prior_n=100)
    assert r.error is None and r.verdict["call"] == "CONSIDER"
    assert r.robustness["framing"] == "single_arm"


def test_assurance_caveat_says_exceeds_when_assurance_is_above_power():
    # a strong informed prior centred ABOVE the TV: assurance should EXCEED classical power at the TV,
    # and the caveat must say so -- not claim the higher number is "below" the lower one
    r = modeling.calc_assurance(n_planned=100, tv=0.30, lrv=0.15, prior_successes=16, prior_n=20)
    assur, power = r.verdict["assurance"], r.robustness["power"]
    assert assur > power and (assur - power) > 0.05
    caveats = [i for i in r.issues if "classical power" in i]
    assert caveats, "expected an assurance-vs-power caveat for this large a gap"
    caveat = caveats[0]
    assert "below" not in caveat.lower()
    assert "exceeds" in caveat.lower()


def test_assurance_caveat_says_below_when_assurance_is_below_power():
    # a weak, pessimistic informed prior (mass well below the TV): assurance should fall BELOW
    # classical power at the TV, and the caveat must correctly say "below"
    r = modeling.calc_assurance(n_planned=200, tv=0.30, lrv=0.15, prior_successes=1, prior_n=15)
    assur, power = r.verdict["assurance"], r.robustness["power"]
    assert assur < power and (power - assur) > 0.05
    caveats = [i for i in r.issues if "classical power" in i]
    assert caveats, "expected an assurance-vs-power caveat for this large a gap"
    caveat = caveats[0]
    assert "is below classical power" in caveat
    assert "exceeds" not in caveat.lower()


def test_assurance_rejects_an_lrv_above_the_tv():
    r = modeling.calc_assurance(n_planned=100, tv=0.15, lrv=0.30, prior_successes=8, prior_n=20)
    assert r.error is not None and "lrv" in r.error.lower()


def test_assurance_rejects_out_of_range_proportions():
    r = modeling.calc_assurance(n_planned=100, tv=1.4, lrv=0.15, prior_successes=8, prior_n=20)
    assert r.error is not None


# ── Bayesian go/no-go: interim ────────────────────────────────────────────────────────────────────
def _interim_df(successes: int, n: int) -> pd.DataFrame:
    return pd.DataFrame({"responded": [1] * successes + [0] * (n - successes)})


def test_interim_stops_for_futility_when_the_data_are_far_below_the_lrv():
    r = modeling.fit_interim(_interim_df(1, 60), "responded", n_planned=70, tv=0.30, lrv=0.15)
    assert r.error is None and r.verdict["call"] == "STOP"
    assert r.verdict["predictive_prob"] < 0.05


def test_interim_goes_when_the_data_are_strong():
    r = modeling.fit_interim(_interim_df(30, 50), "responded", n_planned=60, tv=0.30, lrv=0.15)
    assert r.error is None and r.verdict["call"] == "GO"


def test_interim_reports_the_posterior_with_a_credible_interval():
    r = modeling.fit_interim(_interim_df(12, 40), "responded", n_planned=100, tv=0.30, lrv=0.15)
    t = r.terms[0]
    assert 0.0 < t.ci_low < t.estimate < t.ci_high < 1.0
    assert "credible" in r.effect_label.lower() or "posterior" in r.effect_label.lower()


def test_interim_without_a_lock_is_stamped_exploratory():
    r = modeling.fit_interim(_interim_df(12, 40), "responded", n_planned=100, tv=0.30, lrv=0.15)
    assert r.prespec["status"] == "EXPLORATORY"
    assert any("not pre-specified" in i.lower() for i in r.issues)


def test_interim_with_a_matching_lock_is_pre_specified():
    design = modeling.calc_assurance(n_planned=100, tv=0.30, lrv=0.15, prior_successes=8, prior_n=20)
    lock = design.prespec["lock"]
    r = modeling.fit_interim(_interim_df(12, 40), "responded", n_planned=100, tv=0.30, lrv=0.15,
                             prior_successes=8, prior_n=20, lock=lock)
    assert r.prespec["status"] == "PRE-SPECIFIED"


def test_interim_catches_drift_from_the_locked_design():
    """Lock the design at LRV=0.15, then run the interim at LRV=0.10. That is moving the goalposts."""
    design = modeling.calc_assurance(n_planned=100, tv=0.30, lrv=0.15, prior_successes=8, prior_n=20)
    lock = design.prespec["lock"]
    r = modeling.fit_interim(_interim_df(12, 40), "responded", n_planned=100, tv=0.30, lrv=0.10,
                             prior_successes=8, prior_n=20, lock=lock)
    assert r.prespec["status"] == "DRIFTED"
    assert any(d["field"] == "lrv" for d in r.prespec["drift"])
    assert any("drifted" in i.lower() and "lrv" in i.lower() for i in r.issues)


def test_interim_guards_the_beta_epsilon_degeneracy():
    """FDA's Jan-2026 draft guidance warns that a near-noninformative Beta(eps,eps) prior becomes
    unexpectedly INFORMATIVE at 0% or 100% response. A real trap at an early interim look."""
    r = modeling.fit_interim(_interim_df(20, 20), "responded", n_planned=100, tv=0.30, lrv=0.15,
                             prior_a=0.001, prior_b=0.001)
    assert any("unreliable" in i.lower() or "degenerate" in i.lower() for i in r.issues)


def test_interim_rejects_more_observed_than_planned():
    r = modeling.fit_interim(_interim_df(30, 60), "responded", n_planned=50, tv=0.30, lrv=0.15)
    assert r.error is not None and "planned" in r.error.lower()


def test_interim_at_full_enrollment_reports_the_final_decision():
    r = modeling.fit_interim(_interim_df(30, 50), "responded", n_planned=50, tv=0.30, lrv=0.15)
    assert r.error is None
    assert any("complete" in i.lower() or "final" in i.lower() for i in r.issues)


def test_render_carries_the_go_no_go_verdict_and_survives_bayes_robustness():
    """render() feeds _interpret_model: it must carry the verdict for the LLM to lead with, and it
    must not KeyError on the go/no-go robustness dict, which has no 'summary' (the spec-curve shape)."""
    r = modeling.calc_assurance(n_planned=100, tv=0.30, lrv=0.15, prior_successes=8, prior_n=20)
    assert r.error is None
    text = modeling.render(r)
    assert "GO" in text
    assert "PRIOR SENSITIVITY" in text


def test_render_carries_the_interim_verdict():
    df = pd.DataFrame({"responded": [1] * 12 + [0] * 28})
    r = modeling.fit_interim(df, "responded", n_planned=100, tv=0.30, lrv=0.15)
    assert r.error is None
    assert "VERDICT" in modeling.render(r)


# ── Bayesian go/no-go: two-arm interim ────────────────────────────────────────────────────────────
def _two_arm_df(x_t: int, n_t: int, x_c: int, n_c: int) -> pd.DataFrame:
    rows = ([("treatment", 1)] * x_t + [("treatment", 0)] * (n_t - x_t)
            + [("control", 1)] * x_c + [("control", 0)] * (n_c - x_c))
    return pd.DataFrame(rows, columns=["arm", "responded"])


def test_two_arm_interim_goes_when_treatment_clearly_beats_control():
    r = modeling.fit_interim(_two_arm_df(26, 30, 8, 30), "responded", n_planned=80, tv=0.15, lrv=0.0,
                             framing="two_arm", group="arm", control="control")
    assert r.error is None and r.verdict["call"] == "GO"
    arms = {a["arm"]: a for a in r.arms}
    assert set(arms) == {"treatment", "control"} and arms["treatment"]["value"] > arms["control"]["value"]


def test_two_arm_interim_stops_for_futility_when_arms_are_equal():
    r = modeling.fit_interim(_two_arm_df(9, 45, 9, 45), "responded", n_planned=100, tv=0.15, lrv=0.0,
                             framing="two_arm", group="arm", control="control")
    assert r.error is None and r.verdict["call"] == "STOP"
    assert r.verdict["predictive_prob"] < 0.10


def test_two_arm_interim_reports_the_risk_difference_with_a_credible_interval():
    r = modeling.fit_interim(_two_arm_df(18, 40, 10, 40), "responded", n_planned=100, tv=0.15, lrv=0.0,
                             framing="two_arm", group="arm", control="control")
    t = r.terms[0]
    assert "difference" in t.name.lower()
    assert t.ci_low < t.estimate < t.ci_high
    assert t.estimate == pytest.approx(18 / 40 - 10 / 40, abs=0.02)


def test_two_arm_interim_without_a_lock_is_exploratory():
    r = modeling.fit_interim(_two_arm_df(18, 40, 10, 40), "responded", n_planned=100, tv=0.15, lrv=0.0,
                             framing="two_arm", group="arm", control="control")
    assert r.prespec["status"] == "EXPLORATORY"
    assert any("not pre-specified" in i.lower() for i in r.issues)


def test_two_arm_interim_infers_control_when_not_named():
    # 'control' is recognized by name even without the control= argument
    r = modeling.fit_interim(_two_arm_df(20, 40, 12, 40), "responded", n_planned=100, tv=0.15, lrv=0.0,
                             framing="two_arm", group="arm")
    assert r.error is None
    assert next(a for a in r.arms if a["is_baseline"])["arm"] == "control"


def test_two_arm_interim_rejects_a_single_arm_cohort():
    df = pd.DataFrame({"arm": ["treatment"] * 20, "responded": [1] * 12 + [0] * 8})
    r = modeling.fit_interim(df, "responded", n_planned=100, tv=0.15, lrv=0.0,
                             framing="two_arm", group="arm", control="control")
    assert r.error is not None and "two arms" in r.error.lower()


def test_two_arm_interim_rejects_more_observed_than_planned_per_arm():
    r = modeling.fit_interim(_two_arm_df(30, 60, 20, 60), "responded", n_planned=100, tv=0.15, lrv=0.0,
                             framing="two_arm", group="arm", control="control")
    assert r.error is not None and "planned" in r.error.lower()


def test_two_arm_interim_accepts_a_negative_lrv_non_inferiority_floor():
    # a risk-difference LRV may be negative (a non-inferiority-style floor); it must NOT be rejected as
    # out of [0,1], and a clearly-better treatment against a small negative floor should still be able to GO
    r = modeling.fit_interim(_two_arm_df(28, 30, 8, 30), "responded", n_planned=80, tv=0.15, lrv=-0.05,
                             framing="two_arm", group="arm", control="control")
    assert r.error is None and r.verdict["call"] in ("GO", "CONSIDER", "STOP")


def test_two_arm_interim_no_false_grid_binned_caveat_at_common_size():
    # n_planned=200 -> 100 planned/arm, but only ~40 observed/arm -> remaining grid 61x61=3721 < cap,
    # so the PPoS is EXACT and the grid-binned caveat must NOT appear (regression: it keyed on planned n)
    r = modeling.fit_interim(_two_arm_df(18, 40, 10, 40), "responded", n_planned=200, tv=0.15, lrv=0.0,
                             framing="two_arm", group="arm", control="control")
    assert r.error is None
    assert not any("grid-binned" in i.lower() for i in r.issues)


def test_two_arm_interim_grid_binned_caveat_fires_when_thinning(monkeypatch):
    # force the cap low so the remaining grid exceeds it -> the caveat SHOULD appear
    from agent import bayes
    monkeypatch.setattr(bayes, "MAX_ENUM_DIFF", 100)
    r = modeling.fit_interim(_two_arm_df(18, 40, 10, 40), "responded", n_planned=200, tv=0.15, lrv=0.0,
                             framing="two_arm", group="arm", control="control")
    assert r.error is None
    assert any("grid-binned" in i.lower() for i in r.issues)


def test_two_arm_interim_lower_is_better_adverse_event_rate():
    # lower is better: treatment has FEWER events than control -> a benefit. theta is always
    # (treatment - control); with higher_is_better=False the FAVORABLE direction is negative, so a
    # hoped-for 15pp reduction is tv=-0.15 (a minimum floor of "no worse than control" is lrv=0.0).
    # (tv=+0.15 is rejected by validation: with higher_is_better=False the TV must not exceed the LRV,
    # matching the sign convention already exercised in tests/test_bayes.py's lower-is-better rule.)
    r = modeling.fit_interim(_two_arm_df(6, 40, 18, 40), "responded", n_planned=100, tv=-0.15, lrv=0.0,
                             higher_is_better=False, framing="two_arm", group="arm", control="control")
    assert r.error is None and r.verdict["call"] in ("GO", "CONSIDER", "STOP")
    # treatment event rate (6/40=15%) is well below control (18/40=45%) -> favorable -> not a STOP
    assert r.verdict["call"] in ("GO", "CONSIDER")
