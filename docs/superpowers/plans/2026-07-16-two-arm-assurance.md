# Two-arm design-stage assurance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `modeling.calc_assurance` to a binary two-arm (treatment vs control) framing: the probability a randomized trial ends in GO before it runs, deciding on the risk difference.

**Architecture:** Two new pure functions in `agent/bayes.py` (`assurance_diff`, `operating_characteristics_diff`) computing the exact mixed weighting of `go_grid_diff` (treatment prior-predictive beta-binomial × control sampling binomial). A `framing="two_arm"` branch in `calc_assurance` (`_calc_assurance_two_arm` helper + `_sensitivity_diff`) that emits the same `ModelResult` shape as single-arm assurance, so rendering is reused. Routing gains the two-arm keys. `control_rate` joins `prespec.LOCKED_FIELDS` (None-skipped, backward-compatible).

**Tech Stack:** Python 3.12, numpy, scipy, pandas. pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-07-16-two-arm-assurance-design.md`

## Global Constraints

- No new dependencies (numpy + scipy only). No Monte Carlo in shipped code (closed-form / deterministic; MC only in a test as a cross-check).
- Never raise into the app: `calc_assurance` and its helpers return `ModelResult(..., error=str(e))`.
- Caveats are deterministic, appended to `ModelResult.issues`.
- Effect measure = absolute risk difference rate_t − rate_c; TV/LRV on that scale, may be 0 or negative (validated in `[-1, 1]`).
- 1:1 allocation; `n_planned` is the TOTAL, each arm plans `n_planned // 2`.
- The analysis decision (`go_grid_diff`) uses a **vague `Beta(1,1)`** control prior; the assurance/OC weighting fixes control at the known `control_rate` (binomial sampling only).
- `under_powered ≡ power < 0.80` (power = GO rate at true risk difference = TV), the OC-grounded robustness signal.
- Line length 120 (`.venv/bin/ruff check .`). Tests keyless. Commit style: NO `Co-Authored-By` trailer. Coverage gate `fail_under = 60`.
- Run the full suite `.venv/bin/pytest -q -p no:warnings` before each commit.

---

### Task 1: Two-arm assurance + operating characteristics (`agent/bayes.py`)

Pure numpy/scipy. Reuses the shipped `go_grid_diff`, `Prior`, `DecisionRule`, and module-level `stats`/`np`.

**Files:**
- Modify: `agent/bayes.py` (append)
- Test: `tests/test_bayes.py` (append)

**Interfaces:**
- Consumes: `go_grid_diff(prior_t, prior_c, n_t, n_c, rule)`, `Prior`, `DecisionRule` (existing).
- Produces:
  - `assurance_diff(prior_t, control_rate, n_planned_t, n_planned_c, rule) -> float`
  - `operating_characteristics_diff(prior_t, control_rate, n_planned_t, n_planned_c, rule, grid=None) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bayes.py`:

```python
# ── two-arm design-stage assurance ────────────────────────────────────────────────────────────────
ADIFF_RULE = bayes.DecisionRule(tv=0.15, lrv=0.0)      # a 15-point benefit hoped for; any benefit is the floor


def test_assurance_diff_matches_a_brute_force_simulation():
    """The shipped code computes the exact mixed sum; the TEST simulates, reusing the shipped GO grid."""
    prior_t = bayes.Prior("informed", "beta", (12.0, 10.0), "")   # treatment prior, mean ~0.55
    control_rate, npt, npc = 0.35, 60, 60
    exact = bayes.assurance_diff(prior_t, control_rate, npt, npc, ADIFF_RULE)

    go = bayes.go_grid_diff(prior_t, bayes.Prior("Vague", "beta", (1.0, 1.0), ""), npt, npc, ADIFF_RULE)
    rng = np.random.default_rng(0)
    theta_t = rng.beta(12.0, 10.0, 300_000)                       # treatment truth ~ prior
    s_t = rng.binomial(npt, theta_t)                             # treatment count | theta_t
    s_c = rng.binomial(npc, control_rate, 300_000)               # control count at the known rate
    sim = float(np.mean(go[s_t, s_c]))
    assert exact == pytest.approx(sim, abs=0.005)


def test_assurance_diff_high_when_treatment_beats_control_low_when_not():
    strong = bayes.assurance_diff(bayes.Prior("t", "beta", (30.0, 10.0), ""), 0.30, 80, 80, ADIFF_RULE)
    none = bayes.assurance_diff(bayes.Prior("t", "beta", (9.0, 21.0), ""), 0.30, 80, 80, ADIFF_RULE)
    assert strong > 0.8      # treatment prior mean 0.75 vs control 0.30 -> well past a 15-point benefit
    assert none < 0.2        # treatment prior mean 0.30 == control -> no benefit


def test_assurance_diff_rejects_a_non_binary_prior_and_bad_control_rate():
    with pytest.raises(ValueError):
        bayes.assurance_diff(bayes.Prior("n", "normal", (0.5, 0.1), ""), 0.3, 40, 40, ADIFF_RULE)
    with pytest.raises(ValueError):
        bayes.assurance_diff(bayes.Prior("t", "beta", (5.0, 5.0), ""), 1.4, 40, 40, ADIFF_RULE)


def test_operating_characteristics_diff_rises_and_brackets_type_i_and_power():
    prior_t = bayes.Prior("t", "beta", (12.0, 10.0), "")
    control_rate = 0.35
    oc = bayes.operating_characteristics_diff(prior_t, control_rate, 80, 80, ADIFF_RULE)
    rates = [r["go_rate"] for r in oc]
    assert rates == sorted(rates)                                # GO rate rises with the true difference
    # type I error = GO rate at difference = LRV; power = GO rate at difference = TV
    def _at(diff):
        return min(oc, key=lambda r: abs(r["theta"] - diff))["go_rate"]
    assert _at(ADIFF_RULE.lrv) < _at(ADIFF_RULE.tv)
    assert all("theta_t" in r and "theta" in r for r in oc)     # theta = risk difference, theta_t = trt rate
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_bayes.py -q -p no:warnings -k "assurance_diff or operating_characteristics_diff"`
Expected: FAIL with `AttributeError: module 'agent.bayes' has no attribute 'assurance_diff'`

- [ ] **Step 3: Write the implementation**

Append to `agent/bayes.py`:

```python
# ── two-arm design-stage assurance ────────────────────────────────────────────────────────────────
def assurance_diff(prior_t: Prior, control_rate: float, n_planned_t: int, n_planned_c: int,
                   rule: DecisionRule) -> float:
    """P(a randomized trial ends in GO), before it runs. The treatment arm is averaged over its
    prior-predictive (beta-binomial); the control responds at the known control_rate and contributes
    sampling variability only (binomial). The analysis decision uses a vague control prior, exactly as
    the shipped two-arm interim decides. Exact -- no Monte Carlo.

        assurance = SUM_{s_t, s_c} BetaBinom(s_t; n_t, prior_t) * Binom(s_c; n_c, control_rate) * go[s_t, s_c]
    """
    if prior_t.kind != "beta":
        raise ValueError("two-arm assurance supports a binary endpoint only")
    if not 0.0 <= control_rate <= 1.0:
        raise ValueError("control_rate must be between 0 and 1")
    go = go_grid_diff(prior_t, Prior("Vague", "beta", (1.0, 1.0), ""), n_planned_t, n_planned_c, rule)
    w_t = stats.betabinom.pmf(np.arange(n_planned_t + 1), n_planned_t, *prior_t.params)
    w_c = stats.binom.pmf(np.arange(n_planned_c + 1), n_planned_c, control_rate)
    return float(w_t @ go @ w_c)


def operating_characteristics_diff(prior_t: Prior, control_rate: float, n_planned_t: int,
                                   n_planned_c: int, rule: DecisionRule, grid=None) -> list[dict]:
    """The GO rate at each TRUE risk difference, with control fixed at control_rate. `theta` carries the
    risk difference (treatment rate - control_rate) so the existing OC chart plots it correctly; `theta_t`
    is the treatment rate. Type I error = GO rate at difference = LRV; power = GO rate at difference = TV."""
    if prior_t.kind != "beta":
        raise ValueError("two-arm operating characteristics support a binary endpoint only")
    go = go_grid_diff(prior_t, Prior("Vague", "beta", (1.0, 1.0), ""), n_planned_t, n_planned_c, rule)
    w_c = stats.binom.pmf(np.arange(n_planned_c + 1), n_planned_c, control_rate)
    if grid is None:
        span = abs(rule.tv - rule.lrv) or 0.1
        base = np.clip(np.linspace(control_rate + rule.lrv - 2 * span,
                                   control_rate + rule.tv + 2 * span, 99), 0.0, 1.0)
        grid = np.union1d(base, np.clip([control_rate + rule.lrv, control_rate + rule.tv], 0.0, 1.0))
    xs = np.arange(n_planned_t + 1)
    out = []
    for theta_t in np.asarray(grid, dtype=float):
        w_t = stats.binom.pmf(xs, n_planned_t, theta_t)
        out.append({"theta": float(theta_t - control_rate), "theta_t": float(theta_t),
                    "go_rate": float(w_t @ go @ w_c)})
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_bayes.py -q -p no:warnings`
Expected: PASS (all, including the 4 new two-arm assurance tests)

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check agent/bayes.py tests/test_bayes.py
git add agent/bayes.py tests/test_bayes.py
git commit -m "Add two-arm design-stage assurance and operating characteristics

assurance_diff is the exact probability a randomized trial ends in GO before it
runs: the treatment arm averaged over its prior-predictive (beta-binomial), the
control fixed at a known rate with binomial sampling, weighting the shipped
go_grid_diff decision grid (which decides with a vague control prior, as the
interim does). operating_characteristics_diff gives the GO rate over the true
risk difference. No Monte Carlo; cross-checked against a brute-force simulation."
```

---

### Task 2: `calc_assurance` two-arm branch (`agent/modeling.py`, `agent/prespec.py`)

**Files:**
- Modify: `agent/prespec.py` (`LOCKED_FIELDS` += `"control_rate"`)
- Modify: `agent/modeling.py` (`calc_assurance` signature + dispatch; add `_sensitivity_diff` and `_calc_assurance_two_arm`)
- Test: `tests/test_modeling.py` (append)

**Interfaces:**
- Consumes: `assurance_diff`, `operating_characteristics_diff` (Task 1); `go_grid_diff`, `prob_exceeds`, `decide`, `DecisionRule`, `Prior`, `_build_prior`, `_prespec.create_lock`, `ModelResult` (existing).
- Produces: `calc_assurance(..., control_rate=None)` handling `framing="two_arm"`; `_sensitivity_diff(prior_t, control_rate, n_t, n_c, rule) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_modeling.py`:

```python
# ── Bayesian go/no-go: two-arm design-stage assurance ─────────────────────────────────────────────
def test_two_arm_assurance_go_for_a_strong_expected_effect():
    # treatment expected ~70% (Phase I 14/20), control 35%, big benefit -> GO
    r = modeling.calc_assurance(framing="two_arm", n_planned=200, tv=0.15, lrv=0.0,
                                control_rate=0.35, prior_successes=14, prior_n=20)
    assert r.error is None and r.verdict["call"] == "GO"
    assert 0.0 <= r.verdict["assurance"] <= 1.0
    assert r.robustness["framing"] == "two_arm"


def test_two_arm_assurance_flags_underpowered_and_keeps_the_panel():
    # a small trial with a modest expected benefit -> power at the TV below 80%
    r = modeling.calc_assurance(framing="two_arm", n_planned=40, tv=0.15, lrv=0.0,
                                control_rate=0.35, prior_successes=6, prior_n=20)
    assert r.error is None
    assert r.robustness["under_powered"] is True
    assert "under-powered" in " ".join(r.issues).lower()
    assert len(r.robustness["panel"]) == 4                       # treatment-prior sensitivity panel


def test_two_arm_assurance_emits_the_oc_and_planning_curve():
    r = modeling.calc_assurance(framing="two_arm", n_planned=200, tv=0.15, lrv=0.0,
                                control_rate=0.35, prior_successes=14, prior_n=20)
    assert len(r.robustness["oc"]) > 10 and all("theta" in row for row in r.robustness["oc"])
    assert len(r.series) > 5 and all("assurance" in p for p in r.series)   # assurance-vs-n planning curve


def test_two_arm_assurance_lock_round_trips():
    from agent import prespec
    r = modeling.calc_assurance(framing="two_arm", n_planned=200, tv=0.15, lrv=0.0,
                                control_rate=0.35, prior_successes=14, prior_n=20)
    lock = r.prespec["lock"]
    assert prespec.verify(lock, lock["params"])["status"] == "PRE-SPECIFIED"
    assert lock["params"].get("control_rate") == 0.35           # control_rate is captured in the lock


def test_two_arm_assurance_requires_a_valid_control_rate():
    r = modeling.calc_assurance(framing="two_arm", n_planned=200, tv=0.15, lrv=0.0,
                                prior_successes=14, prior_n=20)
    assert r.error is not None and "control" in r.error.lower()
    r2 = modeling.calc_assurance(framing="two_arm", n_planned=200, tv=0.15, lrv=0.0,
                                 control_rate=1.4, prior_successes=14, prior_n=20)
    assert r2.error is not None and "control" in r2.error.lower()


def test_two_arm_assurance_accepts_a_negative_lrv():
    # a risk-difference LRV may be negative (a non-inferiority-style floor) and must not be rejected
    r = modeling.calc_assurance(framing="two_arm", n_planned=200, tv=0.15, lrv=-0.05,
                                control_rate=0.35, prior_successes=14, prior_n=20)
    assert r.error is None and r.verdict["call"] in ("GO", "CONSIDER", "STOP")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_modeling.py -q -p no:warnings -k "two_arm_assurance"`
Expected: FAIL (`calc_assurance` has no `control_rate` parameter → `TypeError`).

- [ ] **Step 3a: Add `control_rate` to the lock fields**

In `agent/prespec.py`, in `LOCKED_FIELDS`, add `"control_rate"`. The tuple currently is:

```python
LOCKED_FIELDS: tuple[str, ...] = (
    "endpoint_type", "framing", "n_planned",
    "tv", "lrv", "gate_tv", "gate_lrv", "stop_lrv", "higher_is_better",
    "prior_a", "prior_b", "prior_mu", "prior_sd",
)
```

Replace it with:

```python
LOCKED_FIELDS: tuple[str, ...] = (
    "endpoint_type", "framing", "n_planned",
    "tv", "lrv", "gate_tv", "gate_lrv", "stop_lrv", "higher_is_better",
    "prior_a", "prior_b", "prior_mu", "prior_sd", "control_rate",
)
```

(`canonical` already skips `None` values, so single-arm/interim locks — which have no `control_rate` — hash identically to before; existing prespec tests are unaffected.)

- [ ] **Step 3b: Extend the `calc_assurance` signature and dispatch**

In `agent/modeling.py`, change the `calc_assurance` signature to add `control_rate=None` (place it with the other keyword args, after `sd`):

```python
def calc_assurance(endpoint_type: str = "proportion", framing: str = "single_arm",
                   n_planned=None, tv=None, lrv=None,
                   gate_tv: float = 0.80, gate_lrv: float = 0.90, stop_lrv: float = 0.10,
                   higher_is_better: bool = True,
                   prior_successes=None, prior_n=None, prior_a=None, prior_b=None,
                   prior_mu=None, prior_sd=None, sd=None, anchor=None, control_rate=None) -> ModelResult:
```

Then, inside the `try:` block, immediately AFTER the `if n_planned <= 0:` guard
(`return _err("the planned sample size must be positive.")`) and BEFORE the
`if endpoint_type == "proportion" and not (0 <= tv <= 1 and 0 <= lrv <= 1):` range check, insert the
dispatch (it must precede the single-arm `[0,1]` range check because two-arm TV/LRV are risk differences
that may be negative):

```python
        if framing == "two_arm":
            return _calc_assurance_two_arm(endpoint_type, n_planned, tv, lrv, gate_tv, gate_lrv,
                                           stop_lrv, higher_is_better, prior_successes, prior_n,
                                           prior_a, prior_b, control_rate, anchor)
```

- [ ] **Step 3c: Add `_sensitivity_diff` and `_calc_assurance_two_arm`**

In `agent/modeling.py`, add these two functions immediately AFTER `calc_assurance` ends:

```python
def _sensitivity_diff(prior_t, control_rate, n_t, n_c, rule) -> list[dict]:
    """Two-arm prior-sensitivity panel: vary the TREATMENT prior (informed, vague, skeptical at
    control+LRV, enthusiastic at control+TV), holding the control rate fixed, and report each prior's
    assurance and its prior-only difference verdict. The control rate does not vary."""
    ess = 10.0
    skept_mean = min(max(control_rate + rule.lrv, 1e-6), 1 - 1e-6)
    enth_mean = min(max(control_rate + rule.tv, 1e-6), 1 - 1e-6)
    panel_priors = [
        prior_t,
        _bayes.Prior("Vague", "beta", (1.0, 1.0), "Uniform Beta(1,1) on the treatment rate."),
        _bayes.Prior("Skeptical", "beta", (skept_mean * ess, (1 - skept_mean) * ess),
                     f"Treatment centred at control + LRV ({skept_mean:g}); ESS {ess:g}."),
        _bayes.Prior("Enthusiastic", "beta", (enth_mean * ess, (1 - enth_mean) * ess),
                     f"Treatment centred at control + TV ({enth_mean:g}); ESS {ess:g}."),
    ]
    rows = []
    for p in panel_priors:
        a, b = p.params
        thr_tv = min(max(control_rate + rule.tv, 0.0), 1.0)
        thr_lrv = min(max(control_rate + rule.lrv, 0.0), 1.0)
        p_tv = _bayes.prob_exceeds("beta", a, b, thr_tv, rule.higher_is_better)
        p_lrv = _bayes.prob_exceeds("beta", a, b, thr_lrv, rule.higher_is_better)
        call, _ = _bayes.decide(float(p_tv), float(p_lrv), rule)
        rows.append({"prior": p.name, "params": [round(float(v), 3) for v in p.params],
                     "assurance": round(_bayes.assurance_diff(p, control_rate, n_t, n_c, rule), 4),
                     "call": call, "provenance": p.provenance})
    return rows


def _calc_assurance_two_arm(endpoint_type, n_planned, tv, lrv, gate_tv, gate_lrv, stop_lrv,
                            higher_is_better, prior_successes, prior_n, prior_a, prior_b,
                            control_rate, anchor) -> ModelResult:
    """Design-stage assurance for a randomized two-arm trial: the probability it ends in GO before it
    runs, deciding on the risk difference against a known control rate."""
    def _err(msg):
        return ModelResult("assurance", "go/no-go", 0, "probability of success", error=msg)
    try:
        if endpoint_type != "proportion":
            return _err("two-arm assurance currently supports a binary endpoint only.")
        if control_rate is None or not (0.0 <= float(control_rate) <= 1.0):
            return _err("a two-arm assurance needs a known control response rate between 0 and 1.")
        control_rate = float(control_rate)
        if not (-1.0 <= tv <= 1.0 and -1.0 <= lrv <= 1.0):
            return _err("the two-arm TV and LRV are risk differences and must be between -1 and 1 "
                        "(express a 15-point benefit as 0.15).")
        if higher_is_better and lrv > tv:
            return _err("the LRV must not exceed the TV.")
        if not higher_is_better and tv > lrv:
            return _err("with a lower-is-better endpoint the TV must not exceed the LRV.")

        rule = _bayes.DecisionRule(tv=tv, lrv=lrv, gate_tv=float(gate_tv), gate_lrv=float(gate_lrv),
                                   stop_lrv=float(stop_lrv), higher_is_better=bool(higher_is_better))
        prior_t = _build_prior("proportion", tv, lrv, prior_successes, prior_n, prior_a, prior_b,
                               None, None)
        n_t = n_c = n_planned // 2

        # prior-only design verdict: shift the threshold onto the treatment-rate scale by the control
        a_t, b_t = prior_t.params
        p_tv = float(_bayes.prob_exceeds("beta", a_t, b_t, min(max(control_rate + tv, 0.0), 1.0),
                                         higher_is_better))
        p_lrv = float(_bayes.prob_exceeds("beta", a_t, b_t, min(max(control_rate + lrv, 0.0), 1.0),
                                          higher_is_better))
        call, reason = _bayes.decide(p_tv, p_lrv, rule)

        assur = _bayes.assurance_diff(prior_t, control_rate, n_t, n_c, rule)
        oc = _bayes.operating_characteristics_diff(prior_t, control_rate, n_t, n_c, rule)

        def _oc_at(diff):                                        # GO rate at a given TRUE risk difference
            return min(oc, key=lambda r: abs(r["theta"] - diff))["go_rate"]
        t1, power = _oc_at(lrv), _oc_at(tv)
        under_powered = power < 0.80
        panel = _sensitivity_diff(prior_t, control_rate, n_t, n_c, rule)

        params = {"endpoint_type": "proportion", "framing": "two_arm", "n_planned": n_planned,
                  "tv": tv, "lrv": lrv, "gate_tv": gate_tv, "gate_lrv": gate_lrv,
                  "stop_lrv": stop_lrv, "higher_is_better": higher_is_better,
                  "prior_a": a_t, "prior_b": b_t, "prior_mu": None, "prior_sd": None,
                  "control_rate": control_rate}
        lock = _prespec.create_lock(params, oc, anchor=anchor)

        mr = ModelResult("assurance", "go/no-go", n_planned, "probability of success",
                         fit_stat=f"assurance={assur:.1%} · n={n_planned:,} (1:1) · "
                                  f"TV={tv:g} / LRV={lrv:g} vs control {control_rate:.0%}",
                         note="Design-stage two-arm Bayesian go/no-go on the risk difference. Assurance "
                              "averages the treatment arm over the prior and the control over its known "
                              "rate; it is not a prediction about any one trial. Synthetic data.")
        mr.verdict = {"call": call, "reason": reason, "assurance": round(assur, 4),
                      "power": round(power, 4)}
        mr.series = [{"n": int(nn),
                      "assurance": round(_bayes.assurance_diff(prior_t, control_rate,
                                                               int(nn) // 2, int(nn) // 2, rule), 4)}
                     for nn in np.unique(np.linspace(20, max(40, n_planned * 2), 20).astype(int))]
        mr.robustness = {"panel": panel, "under_powered": under_powered, "oc": oc, "framing": "two_arm",
                         "type_i_error": round(t1, 4), "power": round(power, 4)}
        mr.prespec = {"status": "PRE-SPECIFIED", "lock": lock, "drift": []}

        assur_vals = [r["assurance"] for r in panel]
        skept = next((r["assurance"] for r in panel if r["prior"] == "Skeptical"), min(assur_vals))
        issues = [_prespec.caveat({"status": "PRE-SPECIFIED", "drift": []}),
                  f"Treatment prior: {prior_t.provenance} Control fixed at {control_rate:.0%}.",
                  f"Prior sensitivity: across the four defensible treatment priors the assurance ranges "
                  f"from {min(assur_vals):.0%} to {max(assur_vals):.0%} (see the panel) -- a skeptic "
                  f"(treatment at control + LRV) expects {skept:.0%}, your prior expects {assur:.0%}."]
        if under_powered:
            issues.append(f"UNDER-POWERED: power at the TV is only {power:.0%}, below the conventional "
                          "80%. Even if the true benefit equals the Target Value, this design reaches GO "
                          f"only {power:.0%} of the time -- the binding limitation here. Increase n.")
        else:
            issues.append(f"Adequately powered: power at the TV is {power:.0%} (at or above the "
                          "conventional 80%): the design can reliably detect a benefit at the Target Value.")
        issues.append(f"Operating characteristics: type I error {t1:.1%} (GO when the true benefit is only "
                      f"at the LRV) and power {power:.1%} (GO when it is at the TV).")
        issues.append("Two-arm design-stage decision support on the risk difference; not a regulatory "
                      "submission analysis.")
        mr.issues = issues
        return mr
    except Exception as e:  # noqa: BLE001 — never raise into the app
        return _err(str(e))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_modeling.py tests/test_prespec.py -q -p no:warnings -k "two_arm_assurance or lock or prespec or verify"`
Expected: PASS (the new two-arm assurance tests and the existing prespec tests).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check agent/modeling.py agent/prespec.py tests/test_modeling.py
git add agent/modeling.py agent/prespec.py tests/test_modeling.py
git commit -m "Add the two-arm branch to modeling.calc_assurance

Design-stage assurance for a randomized trial: a treatment prior plus a known
control rate, deciding on the risk difference. Emits the same ModelResult shape
as single-arm assurance (verdict, assurance, assurance-vs-n planning curve, OC
over the risk difference, OC-grounded under-powered flag, treatment-prior
sensitivity panel, pre-specification lock), so the existing rendering is reused.
control_rate joins the locked fields (None-skipped, backward-compatible)."
```

---

### Task 3: Routing (`agent/agent.py`)

**Files:**
- Modify: `agent/agent.py` (`_ASSURANCE_KEYS`; the `assurance` router description)
- Test: `tests/test_agent.py` (append)

**Interfaces:**
- Consumes: `modeling.calc_assurance(..., framing, control_rate)` (Task 2).
- Produces: `_run_assurance` forwarding `framing`/`control_rate`; the router teaches the two-arm framing.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent.py`:

```python
def test_run_assurance_two_arm_takes_the_no_data_path(monkeypatch):
    """Two-arm assurance is a DESIGN calculation: it must never touch SQL."""
    def _boom(*a, **k):
        raise AssertionError("assurance must not run SQL")
    monkeypatch.setattr(agent, "run_query", _boom)
    monkeypatch.setattr(agent, "_interpret_model", lambda *a, **k: "**Findings**\nok")

    spec = {"model_type": "assurance", "framing": "two_arm", "n_planned": 200, "tv": 0.15, "lrv": 0.0,
            "control_rate": 0.35, "prior_successes": 14, "prior_n": 20, "hypothesis": "h"}
    res = agent._run_assurance("Will the randomized trial succeed?", spec, agent.AgentResult(question="q"))
    assert res.model["model_type"] == "assurance"
    assert res.model["robustness"]["framing"] == "two_arm"
    assert res.model["verdict"]["call"] in ("GO", "CONSIDER", "STOP")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_agent.py -q -p no:warnings -k two_arm`
Expected: FAIL — `_ASSURANCE_KEYS` omits `control_rate`, so `calc_assurance` gets no control rate and returns an error (`res.model["verdict"]` empty → KeyError, or the framing assertion fails).

- [ ] **Step 3a: Add the two-arm keys**

In `agent/agent.py`, replace `_ASSURANCE_KEYS`:

```python
_ASSURANCE_KEYS = ("endpoint_type", "framing", "n_planned", "tv", "lrv", "gate_tv", "gate_lrv",
                   "stop_lrv", "higher_is_better", "prior_successes", "prior_n", "prior_a",
                   "prior_b", "prior_mu", "prior_sd", "sd", "anchor")
```

with:

```python
_ASSURANCE_KEYS = ("endpoint_type", "framing", "n_planned", "tv", "lrv", "gate_tv", "gate_lrv",
                   "stop_lrv", "higher_is_better", "prior_successes", "prior_n", "prior_a",
                   "prior_b", "prior_mu", "prior_sd", "sd", "anchor", "control_rate")
```

- [ ] **Step 3b: Teach the router the two-arm framing**

In `agent/agent.py`, in the `assurance` model-type description inside `_route`, find the line ending
`"as proportions (30% -> 0.30). Optional: \`higher_is_better\` (false for an adverse-event rate).\n"`
and insert a new string literal immediately AFTER it (before the `"  'interim'     an INTERIM ..."` line):

```python
        "For a RANDOMIZED / two-arm design (treatment vs a concurrent control or placebo), set "
        "`framing`='two_arm', `control_rate`=the known control response rate (e.g. standard of care "
        "responds 35% -> 0.35), and express `tv`/`lrv` as RISK DIFFERENCES treatment-minus-control (a "
        "15-point benefit hoped for -> tv 0.15; any benefit is the floor -> lrv 0). Supply the treatment "
        "prior from Phase I as `prior_successes` + `prior_n`. n_planned is the TOTAL across both arms.\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_agent.py -q -p no:warnings`
Expected: PASS

- [ ] **Step 5: Lint, run the full suite, and commit**

```bash
.venv/bin/ruff check agent/agent.py tests/test_agent.py
.venv/bin/pytest -q -p no:warnings
git add agent/agent.py tests/test_agent.py
git commit -m "Route the two-arm design-stage assurance through the agent

_ASSURANCE_KEYS gains control_rate; the router description teaches the two-arm
framing (a treatment prior + a known control rate, TV/LRV as risk differences).
Two-arm assurance takes the same no-data design path as single-arm assurance."
```

---

## Final verification (after Task 3)

- [ ] **Run everything**

```bash
.venv/bin/pytest -q -p no:warnings          # expect all green
.venv/bin/ruff check .                      # expect "All checks passed!"
.venv/bin/pytest --cov=agent -q -p no:warnings | tail -3   # coverage >= 60%
```

- [ ] **Drive the real app**

```bash
.venv/bin/streamlit run app.py --server.headless=true --server.port=8603
```

Ask: `"Phase I showed the new drug responding in 14 of 20 patients. Control (standard of care) responds about 35%. If we plan a randomized trial of 200 total, is it worth running — we want at least a 15-point benefit, and any benefit is the floor?"`
Expect: an `assurance` model with `framing=two_arm`, a GO / CONSIDER / STOP badge, the assurance-vs-n curve, the operating-characteristics curve over the risk difference, a four-row treatment-prior sensitivity table, a working lock download, and (for this well-separated design) no under-powered caveat. Confirm 0 console errors.

Then ask the same with a **40-patient** total: expect the **UNDER-POWERED** caveat (power at the TV below 80%).

- [ ] **Update the docs**

In `README.md`, add a model-families row for the two-arm design-stage assurance, and extend the Bayesian go/no-go paragraph to note that a controlled trial can now be both planned (two-arm assurance) and monitored (two-arm interim). Bump the unit-test count. Commit.

Update `CONCEPTS.md` §26 to note two-arm assurance if that section is present locally (git-ignored — edit but do not commit).

---

## Self-review notes

**Spec coverage.** §2 the two control roles → Task 1 (`go_grid_diff` with a vague control; binomial control weighting). §3 the two bayes functions → Task 1. §4 the `calc_assurance` branch (validation, prior-only verdict, computation, ModelResult) → Task 2. §5 robustness (`under_powered` + `_sensitivity_diff`) → Task 2. §6 routing → Task 3. §7 rendering reuse → no code change (same ModelResult shape); verified in Final verification (drive the app). §8 error handling → Task 2 (tested: control_rate, negative LRV, endpoint). §9 test plan → distributed. §10 references → in the spec.

**Type consistency.** `assurance_diff(prior_t, control_rate, n_planned_t, n_planned_c, rule)` and `operating_characteristics_diff(..., grid=None)` defined in Task 1, called with those args in Task 2. OC rows carry `theta` (risk difference) + `theta_t` + `go_rate`. `robustness` uses `under_powered` (matching the merged single-arm fix) and `panel`/`oc`/`type_i_error`/`power`/`framing`. `control_rate` added to `prespec.LOCKED_FIELDS` (Task 2 Step 3a) and to the lock params (Task 2 Step 3c) and `_ASSURANCE_KEYS` (Task 3).

**Rendering reuse confirmed.** Two-arm assurance sets `model_type="assurance"`, so `app.py`/`report.py`/`charts.py` render it exactly like single-arm assurance (verdict badge, `assurance_curve_chart` on `series`, `oc_curve_chart` on `robustness.oc` reading `theta`/`go_rate`, the prior table, the lock download). No rendering code change is required; the Final-verification app drive confirms it.

**Deferred, flagged.** Unequal allocation; a control prior (vs point rate); continuous endpoint. Binary two-arm design-stage only, symmetric with the shipped two-arm interim.
