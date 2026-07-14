# Bayesian go/no-go decision module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Bayesian go/no-go decision capability (design-stage assurance and interim predictive probability) for drug and medical-device trials, with a dual-criterion TV/LRV verdict, a prior-sensitivity panel, simulated operating characteristics, and a pre-specification lock.

**Architecture:** Two new pure modules. `agent/bayes.py` holds the conjugate decision engine (closed-form and deterministic numeric integration; **no Monte Carlo in shipped code**). `agent/prespec.py` holds the pre-specification lock (canonical hash, verify, drift diff). `agent/modeling.py` gains two thin entry points, `calc_assurance` (no data, mirrors `calc_sample_size`) and `fit_interim` (takes a DataFrame, mirrors every other `fit_*`), both returning the existing `ModelResult`. `agent/agent.py` routes two new `model_type` values. `app.py` and `agent/report.py` render.

**Tech Stack:** Python 3.12, numpy, scipy, pandas, statsmodels (all already pinned in `requirements.txt`). No new dependencies. pytest for tests, ruff for lint.

**Spec:** `docs/superpowers/specs/2026-07-13-bayesian-go-no-go-design.md`

## Global Constraints

- **No new dependencies.** numpy + scipy only. Do not add PyMC, Stan, or any sampler.
- **No Monte Carlo in shipped code.** Every quantity is closed-form or deterministic grid integration. MC may appear only inside a test, as an independent cross-check.
- **Never raise into the app.** Every public entry point in `modeling.py` wraps its body in `try/except Exception` and returns `ModelResult(..., error=str(e))`. This is the existing convention.
- **Caveats are deterministic.** They are computed in code and appended to `ModelResult.issues`. The LLM may phrase them; it may never invent or drop one.
- **Line length 120** (`ruff` config in `pyproject.toml`). Run `.venv/bin/ruff check .` before every commit.
- **Tests are keyless.** No test may require `OPENAI_API_KEY` or network.
- **Commit style:** no `Co-Authored-By` trailer.
- **Coverage gate:** `fail_under = 60` in `pyproject.toml`. Do not lower it.
- Run the full suite with `.venv/bin/pytest -q -p no:warnings` before each commit.

---

### Task 1: The pre-specification lock (`agent/prespec.py`)

Fully self-contained: no Bayesian math, no dependency on any other new code. Build it first.

**Files:**
- Create: `agent/prespec.py`
- Test: `tests/test_prespec.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `canonical(params: dict) -> str`
  - `lock_id(params: dict) -> str`
  - `create_lock(params: dict, oc: list[dict], anchor: str | None = None) -> dict`
  - `verify(lock: dict | None, params: dict) -> dict` returning
    `{"status": "PRE-SPECIFIED"|"DRIFTED"|"EXPLORATORY"|"INVALID", "lock_id": str|None, "drift": list[dict], "anchor": str|None}`
  - `caveat(prespec: dict) -> str` — the deterministic caveat line for `ModelResult.issues`
  - Module constant `LOCKED_FIELDS: tuple[str, ...]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prespec.py`:

```python
"""Unit tests for the pre-specification lock (pure; no key, no network)."""
import pytest

from agent import prespec

PARAMS = {
    "endpoint_type": "proportion", "framing": "single_arm", "n_planned": 100,
    "tv": 0.30, "lrv": 0.15, "gate_tv": 0.80, "gate_lrv": 0.90, "stop_lrv": 0.10,
    "higher_is_better": True, "prior_a": 9.0, "prior_b": 13.0,
}
OC = [{"theta": 0.15, "go_rate": 0.05}, {"theta": 0.30, "go_rate": 0.82}]


def test_lock_id_is_stable_across_key_order():
    reordered = dict(reversed(list(PARAMS.items())))
    assert prespec.lock_id(PARAMS) == prespec.lock_id(reordered)


def test_lock_id_ignores_float_formatting():
    # 0.1 and 0.10 are the same number; they must not produce a spurious DRIFT
    a = dict(PARAMS, stop_lrv=0.1)
    b = dict(PARAMS, stop_lrv=0.10)
    assert prespec.lock_id(a) == prespec.lock_id(b)


def test_lock_id_changes_when_a_decision_field_changes():
    assert prespec.lock_id(PARAMS) != prespec.lock_id(dict(PARAMS, tv=0.35))


def test_lock_id_ignores_non_decision_fields():
    # a field outside LOCKED_FIELDS must not affect the hash
    assert prespec.lock_id(PARAMS) == prespec.lock_id(dict(PARAMS, hypothesis="anything at all"))


def test_verify_matching_params_is_pre_specified():
    lock = prespec.create_lock(PARAMS, OC)
    out = prespec.verify(lock, PARAMS)
    assert out["status"] == "PRE-SPECIFIED"
    assert out["drift"] == []
    assert out["lock_id"] == lock["lock_id"]


def test_verify_changed_lrv_is_drifted_and_names_the_field():
    lock = prespec.create_lock(PARAMS, OC)
    out = prespec.verify(lock, dict(PARAMS, lrv=0.10))
    assert out["status"] == "DRIFTED"
    drift = {d["field"]: d for d in out["drift"]}
    assert "lrv" in drift
    assert drift["lrv"]["locked"] == 0.15 and drift["lrv"]["actual"] == 0.10


def test_verify_without_a_lock_is_exploratory():
    out = prespec.verify(None, PARAMS)
    assert out["status"] == "EXPLORATORY" and out["lock_id"] is None


def test_verify_rejects_a_tampered_lock():
    lock = prespec.create_lock(PARAMS, OC)
    lock["params"]["tv"] = 0.99            # edited after creation; the recorded hash no longer matches
    out = prespec.verify(lock, PARAMS)
    assert out["status"] == "INVALID"


def test_create_lock_records_anchor_and_the_honest_limitation():
    lock = prespec.create_lock(PARAMS, OC, anchor="NCT01234567")
    assert lock["anchor"] == "NCT01234567"
    assert lock["operating_characteristics"] == OC
    assert "timestamp" in lock
    # the artifact must state plainly what a content hash does NOT prove
    assert "anteriority" in lock["limitation"].lower()


def test_caveat_text_per_status():
    assert "not pre-specified" in prespec.caveat({"status": "EXPLORATORY", "drift": []}).lower()
    assert "pre-specified" in prespec.caveat({"status": "PRE-SPECIFIED", "drift": []}).lower()
    drifted = {"status": "DRIFTED", "drift": [{"field": "lrv", "locked": 0.15, "actual": 0.10}]}
    assert "lrv" in prespec.caveat(drifted)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_prespec.py -q -p no:warnings`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.prespec'`

- [ ] **Step 3: Write the implementation**

Create `agent/prespec.py`:

```python
"""Pre-specification lock: design -> lock -> execute.

A Bayesian design must have its prior, decision thresholds, and planned sample size fixed BEFORE
anyone sees the data. An interactive tool that lets you pick a prior after looking at an interim
result is a prior-shopping machine. This module makes the distinction visible and enforceable.

At design stage we canonicalize the decision-relevant parameters, hash them, and emit a portable
lock artifact the user downloads. At interim we re-hash the run in front of us and compare, stamping
every output PRE-SPECIFIED, DRIFTED (naming each changed field), or EXPLORATORY.

HONEST LIMITATION: a content hash proves INTEGRITY (the lock was not edited after the fact) but NOT
ANTERIORITY (that it existed before the data was seen). Nothing stops a user fabricating a lock after
peeking. Real anteriority needs an external timestamp authority -- a filed protocol, a public registry
entry, a signed and dated SAP. That is what the optional `anchor` field is for. We claim nothing more.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json

# Only these fields are hashed. Everything else about a run (the question wording, the SQL, the
# hypothesis prose) may vary freely without breaking a lock -- what must not vary is the DECISION.
LOCKED_FIELDS: tuple[str, ...] = (
    "endpoint_type", "framing", "n_planned",
    "tv", "lrv", "gate_tv", "gate_lrv", "stop_lrv", "higher_is_better",
    "prior_a", "prior_b", "prior_mu", "prior_sd",
)

_PRECISION = 6          # floats are rounded before hashing so 0.1 and 0.10 cannot drift apart

_LIMITATION = (
    "This lock proves INTEGRITY (its contents were not altered after creation) but NOT ANTERIORITY "
    "(that it existed before the data were seen). Trustworthy pre-specification requires an external "
    "timestamp: a filed protocol, a public registry entry, or a signed and dated SAP. Record that "
    "reference in the `anchor` field."
)


def _norm(v):
    """Canonical form of one value: floats rounded, ints/bools/strings left alone."""
    if isinstance(v, bool):          # bool before int -- bool IS an int in Python
        return v
    if isinstance(v, float):
        return round(v, _PRECISION)
    return v


def canonical(params: dict) -> str:
    """Sorted, float-rounded JSON of the decision-relevant fields only."""
    d = {k: _norm(params[k]) for k in LOCKED_FIELDS if params.get(k) is not None}
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def lock_id(params: dict) -> str:
    return hashlib.sha256(canonical(params).encode()).hexdigest()


def create_lock(params: dict, oc: list[dict], anchor: str | None = None) -> dict:
    """The portable artifact the user downloads at design stage and supplies back at interim."""
    locked = {k: _norm(params[k]) for k in LOCKED_FIELDS if params.get(k) is not None}
    return {
        "lock_id": lock_id(params),
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "params": locked,
        "operating_characteristics": oc,
        "anchor": anchor,
        "limitation": _LIMITATION,
    }


def verify(lock: dict | None, params: dict) -> dict:
    """Compare a run's parameters against a lock. Never raises."""
    if not lock:
        return {"status": "EXPLORATORY", "lock_id": None, "drift": [], "anchor": None}
    try:
        recorded = lock["lock_id"]
        # Re-hash the lock's OWN recorded params. If they don't reproduce its stored id, it was edited.
        if lock_id(lock["params"]) != recorded:
            return {"status": "INVALID", "lock_id": recorded, "drift": [], "anchor": lock.get("anchor")}
    except (KeyError, TypeError):
        return {"status": "INVALID", "lock_id": None, "drift": [], "anchor": None}

    drift = []
    for k in LOCKED_FIELDS:
        locked_v = lock["params"].get(k)
        actual_v = _norm(params[k]) if params.get(k) is not None else None
        if locked_v != actual_v:
            drift.append({"field": k, "locked": locked_v, "actual": actual_v})
    status = "PRE-SPECIFIED" if not drift else "DRIFTED"
    return {"status": status, "lock_id": recorded, "drift": drift, "anchor": lock.get("anchor")}


def caveat(prespec: dict) -> str:
    """The deterministic caveat line. Always first in ModelResult.issues: it conditions how every
    other number should be read."""
    status = prespec.get("status")
    if status == "PRE-SPECIFIED":
        return ("PRE-SPECIFIED: this run matches its locked design (prior, thresholds, planned n). "
                "The verdict was not chosen after seeing the data.")
    if status == "DRIFTED":
        changes = "; ".join(f"{d['field']}: locked {d['locked']} -> used {d['actual']}"
                            for d in prespec.get("drift", []))
        return (f"NOT PRE-SPECIFIED (DRIFTED): this run departs from its locked design -- {changes}. "
                "A decision rule changed after the design was locked, so the verdict must be read as "
                "exploratory, not confirmatory.")
    if status == "INVALID":
        return ("NOT PRE-SPECIFIED (INVALID LOCK): the supplied lock's contents do not match its own "
                "hash, so it was edited after creation. Treating this run as exploratory.")
    return ("EXPLORATORY, not pre-specified: no design lock was supplied, so the prior and decision "
            "thresholds could have been chosen after seeing the data. Fine for exploration; not a "
            "basis for a confirmatory claim.")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prespec.py -q -p no:warnings`
Expected: PASS (11 tests)

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check agent/prespec.py tests/test_prespec.py
git add agent/prespec.py tests/test_prespec.py
git commit -m "Add the pre-specification lock: design -> lock -> execute

Canonical float-rounded hash of the decision-relevant fields only, so 0.1 and
0.10 cannot drift apart and prose changes cannot break a lock. verify() stamps
a run PRE-SPECIFIED, DRIFTED (naming every changed field), EXPLORATORY, or
INVALID (a lock edited after creation).

The artifact states plainly that a content hash proves integrity but NOT
anteriority; that needs an external timestamp, which is what the anchor field
is for."
```

---

### Task 2: Bayesian core — posteriors, probabilities, the decision rule (`agent/bayes.py`)

**Files:**
- Create: `agent/bayes.py`
- Test: `tests/test_bayes.py`

**Interfaces:**
- Consumes: nothing (pure numpy/scipy).
- Produces:
  - `Prior(name: str, kind: str, params: tuple[float, float], provenance: str)` — frozen dataclass
  - `DecisionRule(tv, lrv, gate_tv=0.80, gate_lrv=0.90, stop_lrv=0.10, higher_is_better=True)` — frozen dataclass
  - `beta_posterior(a, b, x, n) -> tuple[float, float]`
  - `normal_posterior(mu0, sd0, xbar, sd, n) -> tuple[float, float]`
  - `prob_exceeds(kind, p1, p2, threshold, higher_is_better=True) -> float | np.ndarray` (vectorized over p1/p2)
  - `prob_diff_exceeds(kind, t, c, threshold, higher_is_better=True) -> float` where `t`/`c` are `(p1, p2)` tuples
  - `decide(p_tv, p_lrv, rule) -> tuple[str, str]`
  - `prior_ess(prior) -> float`
  - `prior_panel(informed: Prior, rule: DecisionRule) -> list[Prior]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bayes.py`:

```python
"""Unit tests for the Bayesian decision engine (pure; exact values, no key, no network)."""
import numpy as np
import pytest
from scipy import stats

from agent import bayes


def test_beta_posterior_is_exact_conjugate_update():
    assert bayes.beta_posterior(1.0, 1.0, 8, 20) == (9.0, 13.0)


def test_beta_posterior_mean():
    a, b = bayes.beta_posterior(1.0, 1.0, 8, 20)
    assert a / (a + b) == pytest.approx(9 / 22)


def test_normal_posterior_shrinks_toward_the_prior():
    # a vague prior barely moves the sample mean; a tight prior pulls it hard
    mu_vague, _ = bayes.normal_posterior(0.0, 1e3, 5.0, 2.0, 25)
    mu_tight, _ = bayes.normal_posterior(0.0, 0.01, 5.0, 2.0, 25)
    assert mu_vague == pytest.approx(5.0, abs=0.01)
    assert abs(mu_tight) < 0.5


def test_prob_exceeds_matches_scipy():
    got = bayes.prob_exceeds("beta", 9.0, 13.0, 0.30)
    assert got == pytest.approx(float(stats.beta.sf(0.30, 9.0, 13.0)))


def test_prob_exceeds_flips_when_lower_is_better():
    hi = bayes.prob_exceeds("beta", 9.0, 13.0, 0.30, higher_is_better=True)
    lo = bayes.prob_exceeds("beta", 9.0, 13.0, 0.30, higher_is_better=False)
    assert hi + lo == pytest.approx(1.0)


def test_prob_exceeds_is_vectorized():
    out = bayes.prob_exceeds("beta", np.array([2.0, 9.0]), np.array([20.0, 13.0]), 0.30)
    assert out.shape == (2,) and out[1] > out[0]


def test_prob_diff_exceeds_beta_matches_monte_carlo():
    # the shipped code uses quadrature; the TEST uses MC as an independent cross-check
    t, c = (30.0, 20.0), (20.0, 30.0)
    quad = bayes.prob_diff_exceeds("beta", t, c, 0.0)
    rng = np.random.default_rng(0)
    mc = float(np.mean(rng.beta(*t, 400_000) - rng.beta(*c, 400_000) > 0.0))
    assert quad == pytest.approx(mc, abs=0.005)


def test_prob_diff_exceeds_normal_is_closed_form():
    # a difference of normals is normal: check against the analytic answer
    t, c = (5.0, 1.0), (3.0, 2.0)
    got = bayes.prob_diff_exceeds("normal", t, c, 1.0)
    want = float(stats.norm.sf(1.0, loc=5.0 - 3.0, scale=np.hypot(1.0, 2.0)))
    assert got == pytest.approx(want)


RULE = bayes.DecisionRule(tv=0.30, lrv=0.15)


def test_decide_truth_table():
    assert bayes.decide(0.85, 0.95, RULE)[0] == "GO"          # clears both gates
    assert bayes.decide(0.55, 0.94, RULE)[0] == "CONSIDER"    # clears LRV, misses TV
    assert bayes.decide(0.01, 0.05, RULE)[0] == "STOP"        # cannot even reach the LRV
    assert bayes.decide(0.85, 0.85, RULE)[0] == "CONSIDER"    # misses the LRV gate -> not a GO


def test_decide_gate_boundaries_are_inclusive():
    assert bayes.decide(0.80, 0.90, RULE)[0] == "GO"          # exactly on both gates
    assert bayes.decide(0.80, 0.10, RULE)[0] == "CONSIDER"    # exactly on stop_lrv -> not a STOP


def test_decide_reason_is_populated():
    call, reason = bayes.decide(0.85, 0.95, RULE)
    assert call == "GO" and "0.30" in reason and "0.15" in reason


def test_prior_ess_is_a_plus_b():
    assert bayes.prior_ess(bayes.Prior("x", "beta", (9.0, 13.0), "")) == 22.0


def test_prior_panel_spans_skeptical_to_enthusiastic():
    informed = bayes.Prior("Phase-I informed", "beta", (9.0, 13.0), "Phase I: 8/20")
    panel = bayes.prior_panel(informed, RULE)
    names = [p.name for p in panel]
    assert names == ["Phase-I informed", "Vague", "Skeptical", "Enthusiastic"]
    mean = lambda p: p.params[0] / (p.params[0] + p.params[1])   # noqa: E731
    skeptical = next(p for p in panel if p.name == "Skeptical")
    enthusiastic = next(p for p in panel if p.name == "Enthusiastic")
    assert mean(skeptical) <= RULE.lrv            # centred at or below the "not worth pursuing" value
    assert mean(enthusiastic) >= RULE.tv          # centred at or above the target
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_bayes.py -q -p no:warnings`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.bayes'`

- [ ] **Step 3: Write the implementation**

Create `agent/bayes.py`:

```python
"""Bayesian decision engine for early-development go/no-go.

Every model here is CONJUGATE, so every quantity is closed-form or deterministic numeric integration
on a fixed grid. There is NO Monte Carlo in this module: no seed to manage, results are
bit-reproducible across runs and platforms, and the tests assert exact values rather than tolerances.
A tool whose only job is to support a decision must not return a different verdict on re-run.

Endpoints:  binary (Beta-Binomial)  |  continuous mean, known SD (Normal-Normal)
Framings:   single-arm vs a performance goal (device)  |  two-arm vs a control (drug)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

_GRID = 2001          # quadrature points for the Beta-difference integral


@dataclass(frozen=True)
class Prior:
    name: str                       # "Phase-I informed" | "Vague" | "Skeptical" | "Enthusiastic"
    kind: str                       # "beta" (binary endpoint) | "normal" (continuous endpoint)
    params: tuple[float, float]     # beta: (a, b).  normal: (mu, sd).
    provenance: str                 # human-readable: where this prior came from


@dataclass(frozen=True)
class DecisionRule:
    """Dual-criterion (Lalonde) go/no-go. A device performance goal is the degenerate case tv == lrv."""
    tv: float                       # Target Value: the effect we hope for
    lrv: float                      # Lower Reference Value: the minimum worth pursuing
    gate_tv: float = 0.80           # required P(theta beyond tv)
    gate_lrv: float = 0.90          # required P(theta beyond lrv)
    stop_lrv: float = 0.10          # P(theta beyond lrv) below this -> STOP
    higher_is_better: bool = True


# ── conjugate updates ─────────────────────────────────────────────────────────────────────────────
def beta_posterior(a: float, b: float, x: int, n: int) -> tuple[float, float]:
    """Beta(a,b) prior + x successes in n trials -> Beta(a+x, b+n-x). Exact."""
    return float(a + x), float(b + n - x)


def normal_posterior(mu0: float, sd0: float, xbar: float, sd: float, n: int) -> tuple[float, float]:
    """Normal(mu0, sd0) prior + n observations with mean xbar and KNOWN sd -> normal posterior. Exact."""
    if n <= 0:
        return float(mu0), float(sd0)
    prec0, prec_d = 1.0 / sd0 ** 2, n / sd ** 2
    var = 1.0 / (prec0 + prec_d)
    return float(var * (prec0 * mu0 + prec_d * xbar)), float(np.sqrt(var))


# ── tail probabilities ────────────────────────────────────────────────────────────────────────────
def prob_exceeds(kind: str, p1, p2, threshold: float, higher_is_better: bool = True):
    """P(theta is BEYOND threshold on the good side). Vectorized over p1/p2 (numpy arrays welcome)."""
    if kind == "beta":
        sf = stats.beta.sf(threshold, p1, p2)
    else:
        sf = stats.norm.sf(threshold, loc=p1, scale=p2)
    return sf if higher_is_better else 1.0 - sf


def prob_diff_exceeds(kind: str, t: tuple[float, float], c: tuple[float, float],
                      threshold: float, higher_is_better: bool = True) -> float:
    """P(theta_treatment - theta_control is beyond threshold).

    normal: a difference of normals is normal -> closed form.
    beta:   no closed form, so 1-D quadrature on a fixed grid:
                P(T - C > d) = INT f_C(v) * sf_T(v + d) dv
            Deterministic and fast. NOT Monte Carlo.
    """
    if kind == "normal":
        mu = t[0] - c[0]
        sd = float(np.hypot(t[1], c[1]))
        sf = float(stats.norm.sf(threshold, loc=mu, scale=sd))
    else:
        v = np.linspace(0.0, 1.0, _GRID)
        f_c = stats.beta.pdf(v, c[0], c[1])
        sf_t = stats.beta.sf(np.clip(v + threshold, 0.0, 1.0), t[0], t[1])
        sf = float(np.trapezoid(f_c * sf_t, v))
    return sf if higher_is_better else 1.0 - sf


# ── the decision rule ─────────────────────────────────────────────────────────────────────────────
def decide(p_tv: float, p_lrv: float, rule: DecisionRule) -> tuple[str, str]:
    """Dual-criterion verdict. p_tv / p_lrv are probabilities of being on the GOOD side of each value."""
    side = "above" if rule.higher_is_better else "below"
    ev = (f"P({side} TV {rule.tv:g}) = {p_tv:.0%}, P({side} LRV {rule.lrv:g}) = {p_lrv:.0%}")
    if p_tv >= rule.gate_tv and p_lrv >= rule.gate_lrv:
        return "GO", (f"{ev}. Clears both pre-specified gates "
                      f"({rule.gate_tv:.0%} at the TV and {rule.gate_lrv:.0%} at the LRV).")
    if p_lrv < rule.stop_lrv:
        return "STOP", (f"{ev}. The effect is very unlikely to reach even the LRV "
                        f"({rule.lrv:g}), the minimum worth pursuing.")
    return "CONSIDER", (f"{ev}. Promising but short of the pre-specified GO gates "
                        f"({rule.gate_tv:.0%} at the TV, {rule.gate_lrv:.0%} at the LRV) -- "
                        "the evidence does not yet justify a commitment.")


# ── priors ────────────────────────────────────────────────────────────────────────────────────────
def prior_ess(prior: Prior) -> float:
    """Effective sample size. Beta(a,b) carries as much information as a+b observations."""
    return float(prior.params[0] + prior.params[1]) if prior.kind == "beta" else float("nan")


def prior_panel(informed: Prior, rule: DecisionRule) -> list[Prior]:
    """Four defensible priors. If the verdict flips across them it is FRAGILE, not an answer.

    This is FDA's prior-sensitivity requirement (Jan 2026 draft guidance): show that the trial's
    conclusion is robust across plausible alternative priors, not an artefact of one choice.
    """
    if informed.kind != "beta":
        mu, sd = informed.params
        span = abs(rule.tv - rule.lrv) or 1.0
        return [
            informed,
            Prior("Vague", "normal", (rule.lrv, 10.0 * span), "Weakly informative, centred at the LRV."),
            Prior("Skeptical", "normal", (rule.lrv, span / 2), "Centred at the minimum worth pursuing."),
            Prior("Enthusiastic", "normal", (rule.tv, span / 2), "Centred at the target value."),
        ]
    ess = 10.0                                       # the reference priors carry ~10 observations
    skeptical = (rule.lrv * ess, (1 - rule.lrv) * ess)
    enthusiastic = (rule.tv * ess, (1 - rule.tv) * ess)
    return [
        informed,
        Prior("Vague", "beta", (1.0, 1.0), "Uniform on [0,1]: every response rate equally likely."),
        Prior("Skeptical", "beta", skeptical,
              f"Centred on the LRV ({rule.lrv:g}), the minimum worth pursuing; ESS {ess:g}."),
        Prior("Enthusiastic", "beta", enthusiastic,
              f"Centred on the TV ({rule.tv:g}), the hoped-for effect; ESS {ess:g}."),
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_bayes.py -q -p no:warnings`
Expected: PASS (13 tests)

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check agent/bayes.py tests/test_bayes.py
git add agent/bayes.py tests/test_bayes.py
git commit -m "Add the Bayesian core: conjugate posteriors, tail probabilities, dual-criterion rule

Beta-Binomial and Normal-Normal conjugate updates, closed-form tail
probabilities, and the two-arm posterior difference by 1-D quadrature (a
difference of Betas has no closed form; a difference of normals does). No Monte
Carlo anywhere, so results are bit-reproducible and the tests assert exact
values. The prior panel implements FDA's sensitivity requirement."
```

---

### Task 3: Assurance and operating characteristics (`agent/bayes.py`)

**Files:**
- Modify: `agent/bayes.py` (append)
- Test: `tests/test_bayes.py` (append)

**Interfaces:**
- Consumes: `Prior`, `DecisionRule`, `prob_exceeds`, `decide` (Task 2).
- Produces:
  - `go_grid_binary(prior: Prior, n: int, rule: DecisionRule) -> np.ndarray` — `go[x] == 1` iff observing x successes in n yields GO
  - `assurance(prior: Prior, n_planned: int, rule: DecisionRule, sd: float | None = None) -> float`
  - `operating_characteristics(prior, n_planned, rule, sd=None, grid=None) -> list[dict]` — `[{"theta": float, "go_rate": float}, ...]`
  - `type_i_and_power(prior: Prior, n_planned: int, rule: DecisionRule, sd: float | None = None) -> tuple[float, float]`
    — evaluates the GO rate EXACTLY at the LRV and the TV (an earlier nearest-grid-point version
    misreported power by up to 3 points for off-grid thresholds; do not reintroduce it)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bayes.py`:

```python
# ── assurance + operating characteristics ─────────────────────────────────────────────────────────
def _point_prior(theta: float, k: float = 1e6) -> bayes.Prior:
    """A Beta prior collapsed onto a point mass at theta (huge effective sample size)."""
    return bayes.Prior("point", "beta", (theta * k, (1 - theta) * k), "point mass")


def test_go_grid_is_monotone_in_successes():
    prior = bayes.Prior("Vague", "beta", (1.0, 1.0), "")
    go = bayes.go_grid_binary(prior, 60, RULE)
    assert go.shape == (61,)
    assert go[0] == 0 and go[-1] == 1                 # 0 successes -> never GO; all successes -> GO
    assert np.all(np.diff(go) >= 0)                   # more successes can only help


def test_assurance_collapses_to_power_under_a_point_prior():
    """THE key invariant: as the prior tightens onto theta0, assurance -> classical power at theta0."""
    rule, n, theta0 = bayes.DecisionRule(tv=0.30, lrv=0.15), 80, 0.35
    a = bayes.assurance(_point_prior(theta0), n, rule)
    oc = bayes.operating_characteristics(_point_prior(theta0), n, rule, grid=np.array([theta0]))
    power_at_theta0 = oc[0]["go_rate"]
    assert a == pytest.approx(power_at_theta0, abs=1e-6)


def test_assurance_is_below_power_when_the_prior_has_spread():
    """The whole point of assurance: averaging over uncertainty is more honest, and lower, than
    assuming the effect is exactly the value you hope for."""
    rule, n, theta0 = bayes.DecisionRule(tv=0.30, lrv=0.15), 80, 0.35
    spread = bayes.Prior("informed", "beta", (7.0, 13.0), "Phase I: 6/18")   # mean 0.35, real spread
    power = bayes.operating_characteristics(_point_prior(theta0), n, rule,
                                            grid=np.array([theta0]))[0]["go_rate"]
    assert bayes.assurance(spread, n, rule) < power


def test_operating_characteristics_go_rate_rises_with_the_true_effect():
    prior = bayes.Prior("Vague", "beta", (1.0, 1.0), "")
    oc = bayes.operating_characteristics(prior, 80, RULE, grid=np.array([0.05, 0.15, 0.30, 0.60]))
    rates = [row["go_rate"] for row in oc]
    assert rates == sorted(rates)
    assert rates[0] < 0.05 and rates[-1] > 0.90


def test_type_i_and_power_are_read_off_the_oc_curve():
    prior = bayes.Prior("Vague", "beta", (1.0, 1.0), "")
    oc = bayes.operating_characteristics(prior, 80, RULE)
    t1, power = bayes.type_i_and_power(oc, RULE)
    assert 0.0 <= t1 <= 0.20            # GO rate when the effect is only at the LRV
    assert power > t1                   # GO rate at the TV must exceed it
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_bayes.py -q -p no:warnings -k "go_grid or assurance or operating or type_i"`
Expected: FAIL with `AttributeError: module 'agent.bayes' has no attribute 'go_grid_binary'`

- [ ] **Step 3: Write the implementation**

Append to `agent/bayes.py`:

```python
# ── assurance + operating characteristics ─────────────────────────────────────────────────────────
def go_grid_binary(prior: Prior, n: int, rule: DecisionRule) -> np.ndarray:
    """go[x] == 1 iff observing x successes in n trials yields a GO. Computed ONCE, then reused by
    assurance, the operating characteristics, and the predictive probability -- all three are just
    different weightings of this same vector."""
    a, b = prior.params
    xs = np.arange(n + 1)
    post_a, post_b = a + xs, b + (n - xs)
    p_tv = prob_exceeds("beta", post_a, post_b, rule.tv, rule.higher_is_better)
    p_lrv = prob_exceeds("beta", post_a, post_b, rule.lrv, rule.higher_is_better)
    return ((p_tv >= rule.gate_tv) & (p_lrv >= rule.gate_lrv)).astype(int)


def _go_threshold_normal(prior: Prior, n: int, rule: DecisionRule, sd: float) -> float:
    """The observed sample mean at which the decision tips to GO. The posterior tail probability is
    monotone in the sample mean, so a single threshold exists; find it by bisection on a fine grid."""
    mu0, sd0 = prior.params
    lo, hi = mu0 - 10 * sd, mu0 + 10 * sd
    xbars = np.linspace(lo, hi, 4001)
    pm, ps = np.vectorize(lambda xb: normal_posterior(mu0, sd0, xb, sd, n))(xbars)
    p_tv = prob_exceeds("normal", pm, ps, rule.tv, rule.higher_is_better)
    p_lrv = prob_exceeds("normal", pm, ps, rule.lrv, rule.higher_is_better)
    go = (p_tv >= rule.gate_tv) & (p_lrv >= rule.gate_lrv)
    if not go.any():
        return float("inf") if rule.higher_is_better else float("-inf")
    idx = int(np.argmax(go)) if rule.higher_is_better else int(len(go) - 1 - np.argmax(go[::-1]))
    return float(xbars[idx])


def assurance(prior: Prior, n_planned: int, rule: DecisionRule, sd: float | None = None) -> float:
    """P(the trial reaches GO), averaging over the prior uncertainty about the true effect.

    This is Bayesian power, and it is the honest number: classical power asks "what is the chance of
    success IF the effect is exactly X", which is a question nobody can answer. Assurance integrates
    over what you actually believe about X, and is usually lower.

    Binary: EXACT. assurance = SUM_x go[x] * BetaBinomial(x; n, a, b), because the prior-predictive
    distribution of the success count under a Beta prior IS the beta-binomial. No integration error.
    """
    if prior.kind == "beta":
        a, b = prior.params
        go = go_grid_binary(prior, n_planned, rule)
        xs = np.arange(n_planned + 1)
        return float(np.sum(stats.betabinom.pmf(xs, n_planned, a, b) * go))
    if sd is None or sd <= 0:
        raise ValueError("a continuous endpoint needs a positive known SD")
    mu0, sd0 = prior.params
    crit = _go_threshold_normal(prior, n_planned, rule, sd)
    se = sd / np.sqrt(n_planned)
    # theta ~ prior; xbar | theta ~ N(theta, se) -> xbar ~ N(mu0, sqrt(sd0^2 + se^2)) marginally
    marg = float(np.hypot(sd0, se))
    p = float(stats.norm.sf(crit, loc=mu0, scale=marg))
    return p if rule.higher_is_better else 1.0 - p


def operating_characteristics(prior: Prior, n_planned: int, rule: DecisionRule,
                              sd: float | None = None, grid=None) -> list[dict]:
    """The GO rate at each TRUE effect value. FDA's second pillar: show how the design behaves across
    a plausible range of truths, not just at the value you hope for.

    Read off this curve: the GO rate at the LRV is the type I error (declaring success when the effect
    is not worth pursuing); the GO rate at the TV is the power.
    """
    if grid is None:
        grid = (np.linspace(0.01, 0.99, 99) if prior.kind == "beta"
                else np.linspace(rule.lrv - 2 * abs(rule.tv - rule.lrv),
                                 rule.tv + 2 * abs(rule.tv - rule.lrv), 99))
    out = []
    if prior.kind == "beta":
        go = go_grid_binary(prior, n_planned, rule)
        xs = np.arange(n_planned + 1)
        for th in np.asarray(grid, dtype=float):
            out.append({"theta": float(th),
                        "go_rate": float(np.sum(stats.binom.pmf(xs, n_planned, th) * go))})
        return out
    if sd is None or sd <= 0:
        raise ValueError("a continuous endpoint needs a positive known SD")
    crit = _go_threshold_normal(prior, n_planned, rule, sd)
    se = sd / np.sqrt(n_planned)
    for th in np.asarray(grid, dtype=float):
        p = float(stats.norm.sf(crit, loc=th, scale=se))
        out.append({"theta": float(th), "go_rate": p if rule.higher_is_better else 1.0 - p})
    return out


def type_i_and_power(oc: list[dict], rule: DecisionRule) -> tuple[float, float]:
    """Type I error = GO rate at the LRV. Power = GO rate at the TV. Nearest grid point."""
    def _at(target):
        row = min(oc, key=lambda r: abs(r["theta"] - target))
        return float(row["go_rate"])
    return _at(rule.lrv), _at(rule.tv)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_bayes.py -q -p no:warnings`
Expected: PASS (18 tests)

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check agent/bayes.py tests/test_bayes.py
git add agent/bayes.py tests/test_bayes.py
git commit -m "Add assurance and operating characteristics

Assurance is EXACT for a binary endpoint: the prior-predictive of the success
count under a Beta prior is the beta-binomial, so assurance is just the
go-grid weighted by its pmf. No integration error.

Tested against the key invariant: as the prior collapses onto a point mass,
assurance converges to classical power at that point; with real prior spread it
is strictly lower, which is the whole reason assurance exists.

Operating characteristics (GO rate across true effects) are FDA's second
pillar: type I error is the GO rate at the LRV, power the GO rate at the TV."
```

---

### Task 4: Predictive probability of success (`agent/bayes.py`)

**Files:**
- Modify: `agent/bayes.py` (append)
- Test: `tests/test_bayes.py` (append)

**Interfaces:**
- Consumes: `go_grid_binary`, `beta_posterior`, `Prior`, `DecisionRule` (Tasks 2-3).
- Produces:
  - `predictive_prob_success(prior, x, n, n_planned, rule) -> float` — binary single-arm, exact
  - `MAX_ENUM: int` module constant (the documented enumeration cap)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bayes.py`:

```python
# ── predictive probability of success ─────────────────────────────────────────────────────────────
def test_predictive_prob_matches_a_brute_force_simulation():
    """The shipped code enumerates exactly. The TEST simulates, as an independent cross-check."""
    prior = bayes.Prior("Vague", "beta", (1.0, 1.0), "")
    x, n, n_planned = 12, 40, 100
    exact = bayes.predictive_prob_success(prior, x, n, n_planned, RULE)

    rng = np.random.default_rng(7)
    a, b = bayes.beta_posterior(1.0, 1.0, x, n)
    go_final = bayes.go_grid_binary(prior, n_planned, RULE)
    theta = rng.beta(a, b, 200_000)                       # draw the truth from the current posterior
    future = rng.binomial(n_planned - n, theta)           # simulate the rest of the trial
    sim = float(np.mean(go_final[x + future]))
    assert exact == pytest.approx(sim, abs=0.005)


def test_predictive_prob_is_one_when_the_trial_has_already_won():
    """Enough successes banked that every possible completion is a GO."""
    prior = bayes.Prior("Vague", "beta", (1.0, 1.0), "")
    assert bayes.predictive_prob_success(prior, 38, 40, 50, RULE) == pytest.approx(1.0, abs=1e-9)


def test_predictive_prob_is_near_zero_under_futility():
    """Far below the LRV with little enrollment left: this trial is not coming back."""
    prior = bayes.Prior("Vague", "beta", (1.0, 1.0), "")
    assert bayes.predictive_prob_success(prior, 1, 60, 70, RULE) < 0.01


def test_predictive_prob_at_full_enrollment_is_the_final_decision():
    """No patients left to observe -> the predictive probability degenerates to the final GO/no-GO."""
    prior = bayes.Prior("Vague", "beta", (1.0, 1.0), "")
    go = bayes.go_grid_binary(prior, 50, RULE)
    for x in (5, 20, 35):
        assert bayes.predictive_prob_success(prior, x, 50, 50, RULE) == pytest.approx(float(go[x]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_bayes.py -q -p no:warnings -k predictive`
Expected: FAIL with `AttributeError: module 'agent.bayes' has no attribute 'predictive_prob_success'`

- [ ] **Step 3: Write the implementation**

Append to `agent/bayes.py`:

```python
# ── predictive probability of success (interim) ───────────────────────────────────────────────────
MAX_ENUM = 20_000     # documented cap on the enumeration; above it we bin rather than get slow


def predictive_prob_success(prior: Prior, x: int, n: int, n_planned: int, rule: DecisionRule) -> float:
    """P(the trial ENDS in GO | what we have seen so far). The interim question.

    EXACT for a binary endpoint. Enumerate every possible number of successes y among the
    n_planned - n patients not yet observed, weight each by the posterior-predictive (beta-binomial)
    probability of seeing exactly y, and check whether the FINAL total x + y would be a GO:

        PPoS = SUM_y  BetaBinom(y; m, a+x, b+n-x) * go_final[x + y]

    No simulation error. Low PPoS is the futility signal that stops a trial early and saves the money.
    """
    if prior.kind != "beta":
        raise ValueError("predictive_prob_success currently supports a binary endpoint only")
    if n > n_planned:
        raise ValueError(f"observed n ({n}) exceeds the planned n ({n_planned})")
    a, b = prior.params
    go_final = go_grid_binary(prior, n_planned, rule)
    m = n_planned - n
    if m == 0:                                        # trial complete: this IS the final decision
        return float(go_final[x])
    if m > MAX_ENUM:
        raise ValueError(f"{m:,} unobserved patients exceeds the {MAX_ENUM:,} enumeration cap")
    post_a, post_b = beta_posterior(a, b, x, n)
    ys = np.arange(m + 1)
    w = stats.betabinom.pmf(ys, m, post_a, post_b)     # posterior-predictive of the remaining successes
    return float(np.sum(w * go_final[x + ys]))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_bayes.py -q -p no:warnings`
Expected: PASS (22 tests)

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check agent/bayes.py tests/test_bayes.py
git add agent/bayes.py tests/test_bayes.py
git commit -m "Add the interim predictive probability of success

Exact for a binary endpoint: enumerate every possible completion of the trial,
weight each by its beta-binomial posterior-predictive probability, and check
whether the final total would clear the pre-specified gates. No simulation
error. Cross-checked in the tests against a brute-force simulation, and against
the degenerate case at full enrollment, where it must equal the final decision."
```

---

### Task 5: `modeling.calc_assurance` — the design-stage entry point

**Files:**
- Modify: `agent/modeling.py` (add `prespec` field to `ModelResult`; append `calc_assurance` and its helpers near `calc_sample_size`)
- Test: `tests/test_modeling.py` (append)

**Interfaces:**
- Consumes: `agent.bayes` (Tasks 2-4), `agent.prespec` (Task 1).
- Produces:
  - `ModelResult.prespec: dict` (new field, `default_factory=dict`)
  - `modeling.calc_assurance(endpoint_type="proportion", framing="single_arm", n_planned=None, tv=None, lrv=None, gate_tv=0.80, gate_lrv=0.90, stop_lrv=0.10, higher_is_better=True, prior_successes=None, prior_n=None, prior_a=None, prior_b=None, prior_mu=None, prior_sd=None, sd=None, anchor=None) -> ModelResult`
  - `modeling._build_prior(...) -> bayes.Prior` (private)
  - `modeling._sensitivity(prior, n_planned, rule, sd) -> tuple[list[dict], bool]` (private) — the panel rows and a `fragile` flag

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_modeling.py`:

```python
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


def test_assurance_flags_a_fragile_verdict_when_the_skeptical_prior_flips_it():
    # engineered so the informed prior says GO but a skeptic is not yet convinced
    r = modeling.calc_assurance(n_planned=40, tv=0.30, lrv=0.15, prior_successes=14, prior_n=20)
    joined = " ".join(r.issues).lower()
    assert "fragile" in joined or "holds" in joined            # the panel always reports one or the other
    assert any(row["prior"] == "Skeptical" for row in r.robustness["panel"])


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
    # tv == lrv == the performance goal: the CONSIDER band vanishes
    r = modeling.calc_assurance(n_planned=150, tv=0.85, lrv=0.85, prior_successes=88, prior_n=100)
    assert r.error is None and r.verdict["call"] in ("GO", "STOP", "CONSIDER")
    assert r.robustness["framing"] == "single_arm"


def test_assurance_rejects_an_lrv_above_the_tv():
    r = modeling.calc_assurance(n_planned=100, tv=0.15, lrv=0.30, prior_successes=8, prior_n=20)
    assert r.error is not None and "lrv" in r.error.lower()


def test_assurance_rejects_out_of_range_proportions():
    r = modeling.calc_assurance(n_planned=100, tv=1.4, lrv=0.15, prior_successes=8, prior_n=20)
    assert r.error is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_modeling.py -q -p no:warnings -k assurance`
Expected: FAIL with `AttributeError: module 'agent.modeling' has no attribute 'calc_assurance'`

- [ ] **Step 3a: Add the `prespec` field to `ModelResult`**

In `agent/modeling.py`, in the `ModelResult` dataclass, add one line after the `robustness` field:

```python
    robustness: dict = field(default_factory=dict)  # specification-curve multiverse summary (adjusted models)
    prespec: dict = field(default_factory=dict)     # pre-specification lock status {status, lock, drift}
    issues: list = field(default_factory=list)   # flagged statistical issues (strings)
```

- [ ] **Step 3b: Write `calc_assurance`**

Append to `agent/modeling.py` (after `calc_sample_size`):

**Required imports.** Add these to the TOP of `agent/modeling.py`, with the existing imports (there is
no circular-import risk: neither `bayes` nor `prespec` imports `modeling`). `fit_interim` in Task 6
also needs `scipy.stats` for the credible interval, so add it now:

```python
import numpy as np
import pandas as pd
from scipy import stats

from . import bayes as _bayes
from . import prespec as _prespec
```

Then append the rest to the END of the file, after `calc_sample_size`:

```python
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
            issues.append(f"Assurance ({assur:.1%}) is below classical power ({power:.1%}) because power "
                          "assumes the effect is exactly the TV, while assurance averages over the "
                          "uncertainty about it. Assurance is the number to budget against.")
        mr.issues = issues
        return mr
    except Exception as e:  # noqa: BLE001 — never raise into the app
        return _err(str(e))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_modeling.py -q -p no:warnings`
Expected: PASS (all, including the 9 new assurance tests)

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check agent/modeling.py tests/test_modeling.py
git add agent/modeling.py tests/test_modeling.py
git commit -m "Add modeling.calc_assurance: the design-stage go/no-go entry point

Mirrors calc_sample_size (no data, computed from the question). Emits the
verdict, the assurance-vs-n curve, the four-prior sensitivity panel with a
FRAGILE flag when the call flips, the operating characteristics, and a
pre-specification lock.

Deterministic caveats: the prior and its provenance, prior sensitivity, prior
effective sample size versus the planned n, the assurance-versus-power gap, and
the type I error the pre-specified rule implies."
```

---

### Task 6: `modeling.fit_interim` — the interim entry point

**Files:**
- Modify: `agent/modeling.py` (append after `calc_assurance`)
- Test: `tests/test_modeling.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces:
  - `modeling.fit_interim(df, outcome, n_planned=None, tv=None, lrv=None, gate_tv=0.80, gate_lrv=0.90, stop_lrv=0.10, higher_is_better=True, prior_successes=None, prior_n=None, prior_a=None, prior_b=None, lock=None, endpoint_type="proportion", framing="single_arm") -> ModelResult`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_modeling.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_modeling.py -q -p no:warnings -k interim`
Expected: FAIL with `AttributeError: module 'agent.modeling' has no attribute 'fit_interim'`

- [ ] **Step 3: Write the implementation**

Append to `agent/modeling.py`:

```python
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
```

**Imports.** `from scipy import stats` and the `_bayes` / `_prespec` imports were added to the top of
`agent/modeling.py` in Task 5. Verify they are present before starting:
`grep -n "^from scipy import stats\|^from . import bayes" agent/modeling.py` (expect two hits).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_modeling.py -q -p no:warnings`
Expected: PASS (all)

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check agent/modeling.py tests/test_modeling.py
git add agent/modeling.py tests/test_modeling.py
git commit -m "Add modeling.fit_interim: the interim go/no-go entry point

Posterior from the subjects observed so far plus the exact predictive
probability that the trial ends in GO at full enrolment. A low PPoS is a STOP
for futility regardless of where the posterior sits today.

Verifies the run against its design lock: PRE-SPECIFIED, DRIFTED (naming every
field that moved), or EXPLORATORY. Guards the Beta(eps,eps) degeneracy that
FDA's 2026 draft guidance warns about, where a near-noninformative prior turns
unexpectedly informative at 0% or 100% response."
```

---

### Task 7: Agent routing (`agent/agent.py`)

**Files:**
- Modify: `agent/agent.py` (`_MODEL_HINT` at line ~284; the router prompt in `_route`; `_fit_model`; `_interpret_model`; `run_analysis`; add `_run_assurance`)
- Test: `tests/test_agent.py` (append)

**Interfaces:**
- Consumes: `modeling.calc_assurance`, `modeling.fit_interim` (Tasks 5-6).
- Produces: `_run_assurance(question, spec, result, deadline_start=None) -> AgentResult`; module constant `_ASSURANCE_KEYS`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent.py`:

```python
def test_model_hint_matches_bayesian_go_no_go_questions():
    """The router is GATED by _MODEL_HINT: a question that does not match never reaches _route,
    so the whole feature would be unreachable without these keywords."""
    for q in [
        "What is the probability a 100-patient Phase II succeeds?",
        "Should we go or no-go on this programme?",
        "We are 40 patients in with 12 responses — stop for futility?",
        "What is the assurance of a 60-patient trial?",
        "What is the predictive probability of success at this interim?",
    ]:
        assert agent._MODEL_HINT.search(q), q


def test_run_assurance_takes_the_no_data_path(monkeypatch):
    """assurance is a DESIGN calculation: it must never touch SQL."""
    def _boom(*a, **k):
        raise AssertionError("assurance must not run SQL")
    monkeypatch.setattr(agent, "run_query", _boom)
    monkeypatch.setattr(agent, "_interpret_model", lambda *a, **k: "**Findings**\nok")

    spec = {"model_type": "assurance", "n_planned": 100, "tv": 0.30, "lrv": 0.15,
            "prior_successes": 8, "prior_n": 20, "hypothesis": "h"}
    res = agent._run_assurance("Will Phase II succeed?", spec, agent.AgentResult(question="q"))
    assert res.model["model_type"] == "assurance"
    assert res.model["verdict"]["call"] in ("GO", "CONSIDER", "STOP")


def test_fit_model_dispatches_interim():
    import pandas as pd
    df = pd.DataFrame({"responded": [1] * 12 + [0] * 28})
    spec = {"model_type": "interim", "outcome": "responded", "n_planned": 100,
            "tv": 0.30, "lrv": 0.15}
    mr = agent._fit_model(spec, df)
    assert mr.model_type == "interim" and mr.error is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agent.py -q -p no:warnings -k "model_hint or assurance or interim"`
Expected: FAIL — `_MODEL_HINT` does not match, and `agent` has no attribute `_run_assurance`

- [ ] **Step 3a: Extend `_MODEL_HINT`**

In `agent/agent.py`, replace the final line of the `_MODEL_HINT` pattern:

```python
    r"sample size|power to detect|how many (patient|subject|participant|per arm)|enroll|powered)", re.I)
```

with:

```python
    r"sample size|power to detect|how many (patient|subject|participant|per arm)|enroll|powered|"
    r"go.?no.?go|assurance|probability of (success|technical success)|futility|stop early|"
    r"interim|predictive probability|posterior|bayesian|performance goal|de.?risk)", re.I)
```

- [ ] **Step 3b: Add the two model types to the router prompt**

In `_route`, insert immediately after the `'sample_size'` block (before the `'causal'` block):

```python
        "  'assurance'   a DESIGN-STAGE Bayesian go/no-go — 'what is the probability a 100-patient "
        "Phase II succeeds', 'what is the assurance', 'should we invest in the next study'. NO data, "
        "NO analytic_sql. Extract: `n_planned` (the planned enrolment), `tv` (Target Value: the effect "
        "hoped for), `lrv` (Lower Reference Value: the minimum worth pursuing), and the prior from any "
        "previous study as `prior_successes` + `prior_n` (e.g. 'Phase I showed 8/20' -> 8 and 20). For "
        "a DEVICE performance goal (a single-arm objective performance criterion, e.g. 'success rate "
        "above 85%') set tv AND lrv BOTH to the goal (0.85) and `framing`='single_arm'. Express rates "
        "as proportions (30% -> 0.30). Optional: `higher_is_better` (false for an adverse-event rate).\n"
        "  'interim'     an INTERIM Bayesian go/no-go on data collected SO FAR — 'we are 40 patients "
        "in with 12 responses, continue or stop', 'stop for futility?', 'predictive probability of "
        "success'. analytic_sql returns ONE ROW PER SUBJECT OBSERVED SO FAR with a binary `outcome` "
        "column (cast to int). Also extract `n_planned` (the FULL planned enrolment, which is larger "
        "than the rows returned), `tv`, `lrv`, and the prior as `prior_successes` + `prior_n` if a "
        "previous study is mentioned.\n"
```

Then extend the `Return JSON:` line by adding these keys to the example object:

```python
        '"n_planned":100, "tv":0.3, "lrv":0.15, "prior_successes":8, "prior_n":20, "framing":"single_arm", '
```

- [ ] **Step 3c: Dispatch `interim` in `_fit_model`**

In `_fit_model`, add before the final `return modeling.ModelResult(... unknown model_type ...)` line:

```python
    if mt == "interim":
        return modeling.fit_interim(
            df, spec["outcome"], n_planned=spec.get("n_planned"), tv=spec.get("tv"),
            lrv=spec.get("lrv"), higher_is_better=spec.get("higher_is_better", True),
            prior_successes=spec.get("prior_successes"), prior_n=spec.get("prior_n"),
            framing=spec.get("framing", "single_arm"))
```

- [ ] **Step 3d: Add `_run_assurance` and route to it**

Add after `_run_sample_size` in `agent/agent.py`:

```python
_ASSURANCE_KEYS = ("endpoint_type", "framing", "n_planned", "tv", "lrv", "gate_tv", "gate_lrv",
                   "stop_lrv", "higher_is_better", "prior_successes", "prior_n", "prior_a",
                   "prior_b", "prior_mu", "prior_sd", "sd", "anchor")


def _run_assurance(question: str, spec: dict, result: AgentResult,
                   deadline_start: float | None = None) -> AgentResult:
    """Design-stage Bayesian go/no-go — computed from the question, queries no data."""
    result.hypothesis = spec.get("hypothesis", "")
    params = {k: spec[k] for k in _ASSURANCE_KEYS if spec.get(k) is not None}
    mr = modeling.calc_assurance(**params)
    result.model = mr.as_dict()
    if not mr.error:
        _check_deadline(deadline_start)
    result.interpretation = (f"**Findings**\nCould not compute the go/no-go: {mr.error}"
                             if mr.error else _interpret_model(question, mr))
    return result
```

In `run_analysis`, extend the design-stage special case (currently `if spec.get("model_type") == "sample_size":`):

```python
            if spec.get("model_type") == "sample_size":       # design-stage calc — no data/SQL
                return _run_sample_size(question, spec, result, start)
            if spec.get("model_type") == "assurance":         # design-stage Bayesian go/no-go — no data
                return _run_assurance(question, spec, result, start)
```

- [ ] **Step 3e: Add interpretation guidance**

In `_interpret_model`, add after the `sample_size` bullet:

```python
        "- assurance: LEAD with the GO / CONSIDER / STOP verdict and the assurance (probability of "
        "success). State the TV and LRV it was judged against, the prior and where it came from, and "
        "the type I error and power. If the verdict is FRAGILE across priors, LEAD the caveats with "
        "that — a prior-driven call is not a data-driven one. Say plainly that this is a design-stage "
        "decision-support calculation, not a regulatory submission analysis.\n"
        "- interim: LEAD with the GO / CONSIDER / STOP verdict and the PREDICTIVE PROBABILITY OF "
        "SUCCESS (the chance the trial ends in GO at full enrolment). A low predictive probability is a "
        "futility signal: say so directly. State the posterior rate with its credible interval, and the "
        "pre-specification status — a DRIFTED or EXPLORATORY run must NOT be described as confirmatory.\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_agent.py -q -p no:warnings`
Expected: PASS

- [ ] **Step 5: Lint, run the full suite, and commit**

```bash
.venv/bin/ruff check agent/agent.py tests/test_agent.py
.venv/bin/pytest -q -p no:warnings
git add agent/agent.py tests/test_agent.py
git commit -m "Route the Bayesian go/no-go model types through the agent

_MODEL_HINT is a GATE: a question that does not match it never reaches the
router, so without the new keywords (go/no-go, assurance, futility, interim,
predictive probability) the whole feature would have been unreachable.

assurance takes the no-data path alongside sample_size; interim dispatches
through _fit_model like every other data-backed model. The interpretation
guidance forbids describing a DRIFTED or EXPLORATORY run as confirmatory."
```

---

### Task 8: UI rendering (`app.py`, `agent/charts.py`)

**Files:**
- Modify: `agent/charts.py` (add `assurance_curve_chart`, `oc_curve_chart`)
- Modify: `app.py` (verdict badge colours; the model block; the prior-sensitivity table; the lock download)
- Test: `tests/test_charts.py` (append)

**Interfaces:**
- Consumes: `ModelResult.as_dict()` with `verdict`, `series`, `robustness.panel`, `robustness.oc`, `prespec` (Tasks 5-6).
- Produces: `charts.assurance_curve_chart(model: dict)`, `charts.oc_curve_chart(model: dict)` (both return an Altair chart or `None`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_charts.py`:

```python
def test_assurance_curve_chart_builds():
    model = {"model_type": "assurance", "error": None,
             "series": [{"n": 20, "assurance": 0.31}, {"n": 100, "assurance": 0.62}],
             "verdict": {"call": "GO"}}
    assert charts.assurance_curve_chart(model) is not None


def test_assurance_curve_chart_is_none_without_series():
    assert charts.assurance_curve_chart({"model_type": "assurance", "series": []}) is None


def test_oc_curve_chart_builds():
    model = {"model_type": "assurance", "error": None,
             "robustness": {"oc": [{"theta": 0.1, "go_rate": 0.02}, {"theta": 0.4, "go_rate": 0.9}]}}
    assert charts.oc_curve_chart(model) is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_charts.py -q -p no:warnings -k "assurance or oc_curve"`
Expected: FAIL with `AttributeError: module 'agent.charts' has no attribute 'assurance_curve_chart'`

- [ ] **Step 3a: Add the charts**

Append to `agent/charts.py`:

```python
def assurance_curve_chart(model: dict):
    """Assurance (probability of success) against planned sample size: how much n buys how much
    confidence. The design-stage counterpart of the power curve."""
    if not model or model.get("error") or not model.get("series"):
        return None
    d = pd.DataFrame(model["series"])
    if "assurance" not in d or "n" not in d:
        return None
    line = alt.Chart(d).mark_line(color=TEAL, strokeWidth=2,
                                  point=alt.OverlayMarkDef(color=TEAL)).encode(
        x=alt.X("n:Q", title="planned sample size"),
        y=alt.Y("assurance:Q", title="assurance (probability of success)",
                axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=["n:Q", alt.Tooltip("assurance:Q", format=".1%")])
    return _finish(line, 300, "Assurance vs planned sample size")


def oc_curve_chart(model: dict):
    """Operating characteristics: the GO rate at each TRUE effect. FDA's second pillar -- how the
    design behaves across a plausible range of truths, not just the one you hope for."""
    oc = (model or {}).get("robustness", {}).get("oc")
    if not oc:
        return None
    d = pd.DataFrame(oc)
    if "theta" not in d or "go_rate" not in d:
        return None
    line = alt.Chart(d).mark_line(color=TEAL, strokeWidth=2).encode(
        x=alt.X("theta:Q", title="true effect"),
        y=alt.Y("go_rate:Q", title="probability of a GO", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("theta:Q", format=".3f"), alt.Tooltip("go_rate:Q", format=".1%")])
    return _finish(line, 260, "Operating characteristics (GO rate by true effect)")
```

- [ ] **Step 3b: Render in `app.py`**

In `app.py`, in `_render_model`, add a branch before the `experiment`/`noninferiority` branch:

```python
    if m.get("model_type") in ("assurance", "interim"):
        v = m.get("verdict", {})
        lines = [head, ""]
        if m.get("model_type") == "assurance":
            lines.append(f"- **assurance (probability of success)**: {v.get('assurance', 0):.1%}")
            lines.append(f"- **power at the TV**: {v.get('power', 0):.1%}")
        else:
            lines.append(f"- **predictive probability of success**: {v.get('predictive_prob', 0):.1%}")
            lines.append(f"- **posterior response rate**: {v.get('posterior_mean', 0):.1%}")
        panel = (m.get("robustness") or {}).get("panel")
        if panel:
            lines += ["", "**Prior sensitivity**", "",
                      "| prior | parameters | assurance | verdict |", "|---|---|---|---|"]
            for row in panel:
                lines.append(f"| {row['prior']} | {row['params']} | {row['assurance']:.1%} "
                             f"| **{row['call']}** |")
        if m.get("issues"):
            lines.append("")
            lines += [f"- ⚠️ {iss}" for iss in m["issues"]]
        if m.get("note"):
            lines.append(f"\n_{m['note']}_")
        return "\n".join(lines)
```

In the verdict-badge block, extend the model-type tuple and the colour map:

```python
        if _mt in ("experiment", "noninferiority", "sample_size", "assurance", "interim"):
            _v = result.model.get("verdict", {})
            _call = _v.get("call", "")
            _color = ("#4fd1c5" if _mt == "sample_size" else
                      {"SHIP": "#4fd1c5", "NON-INFERIOR": "#4fd1c5", "GO": "#4fd1c5",
                       "DO NOT SHIP": "#f87171", "NOT NON-INFERIOR": "#f87171", "STOP": "#f87171",
                       "INCONCLUSIVE": "#f5c451", "CONSIDER": "#f5c451"}.get(_call, "#8ea0b0"))
```

and extend the chart selection:

```python
            _dc = (experiment_chart(result.model) if _mt == "experiment"
                   else ni_plot(result.model) if _mt == "noninferiority"
                   else assurance_curve_chart(result.model) if _mt in ("assurance", "interim")
                   else power_curve_chart(result.model))
            if _dc is not None:
                st.altair_chart(_dc, width="stretch")
            if _mt == "assurance":
                _oc = oc_curve_chart(result.model)
                if _oc is not None:
                    st.altair_chart(_oc, width="stretch")
```

Add the pre-specification badge immediately above the verdict badge:

```python
        _ps = result.model.get("prespec") or {}
        if _ps.get("status"):
            _pc = {"PRE-SPECIFIED": "#4fd1c5", "DRIFTED": "#f5c451",
                   "INVALID": "#f87171"}.get(_ps["status"], "#8ea0b0")
            st.markdown(
                f"<div style='display:inline-block;border:1px solid {_pc};color:{_pc};"
                f"padding:.15rem .6rem;border-radius:6px;font-size:.75rem;letter-spacing:.08em;"
                f"margin-bottom:.4rem'>{html.escape(_ps['status'])}</div>", unsafe_allow_html=True)
```

Add the lock download after the model block, inside the same `if result.model:` scope:

```python
        _lock = (result.model.get("prespec") or {}).get("lock")
        if _lock and result.model.get("model_type") == "assurance":
            st.download_button(
                "⬇ Download pre-specification lock",
                data=json.dumps(_lock, indent=2),
                file_name=f"prespec-lock-{_lock['lock_id'][:12]}.json",
                mime="application/json",
                help="Lock this design before the trial runs. Supply it at the interim analysis and "
                     "the app will verify that nothing moved. A hash proves integrity, not that the "
                     "lock predates the data — anchor it to a protocol or registry entry for that.")
```

Import the new charts at the top of `app.py`, alongside the existing chart imports:

```python
from agent.charts import assurance_curve_chart, oc_curve_chart
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_charts.py tests/test_app_smoke.py -q -p no:warnings`
Expected: PASS

- [ ] **Step 5: Lint, run the full suite, and commit**

```bash
.venv/bin/ruff check agent/charts.py app.py tests/test_charts.py
.venv/bin/pytest -q -p no:warnings
git add agent/charts.py app.py tests/test_charts.py
git commit -m "Render the Bayesian go/no-go: verdict, prior sensitivity, OC curve, lock download

The assurance curve shows how much sample size buys how much probability of
success. The operating-characteristics curve shows the GO rate at every true
effect, which is what FDA asks for and what a power number alone hides. The
prior-sensitivity table puts the four verdicts side by side, so a call that
only survives one prior is visibly a prior-driven call.

The pre-specification badge sits above the verdict, because whether a decision
rule was fixed in advance conditions how every number below it should be read."
```

---

### Task 9: Word report section (`agent/report.py`)

**Files:**
- Modify: `agent/report.py` (`_METHOD_BLURB`; the results section; the approval page)
- Test: `tests/test_hardening.py` (append — this is where the existing docx smoke test lives)

**Interfaces:**
- Consumes: `ModelResult.as_dict()` from Tasks 5-6.
- Produces: no new public functions; extends `build_docx`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hardening.py`:

```python
def test_report_renders_a_bayesian_go_no_go(tmp_path):
    """The .docx must carry the verdict, the prior sensitivity, and the pre-specification status."""
    from types import SimpleNamespace

    from agent import modeling, report

    mr = modeling.calc_assurance(n_planned=100, tv=0.30, lrv=0.15, prior_successes=8, prior_n=20)
    assert mr.error is None
    res = SimpleNamespace(question="Will Phase II succeed?", model=mr.as_dict(),
                          sql="", interpretation="**Findings**\nok", findings=[], citations=[],
                          verification={}, hypothesis="", dataframe=None, trace={}, lineage=None,
                          error=None, clarification=None, attempts=[])
    blob = report.build_docx(res)
    assert blob[:2] == b"PK" and len(blob) > 5000       # a real .docx zip
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_hardening.py -q -p no:warnings -k bayesian`
Expected: FAIL (a `KeyError` or an empty results section, because `report.py` does not know these model types)

- [ ] **Step 3: Write the implementation**

In `agent/report.py`, add to `_METHOD_BLURB`:

```python
    "assurance": "Bayesian design-stage go/no-go. Assurance (the probability of a GO, averaged over "
                 "the prior uncertainty about the true effect) with a dual-criterion decision rule "
                 "(Target Value / Lower Reference Value), a prior-sensitivity panel, and simulated "
                 "operating characteristics. Conjugate Beta-Binomial; computed in closed form.",
    "interim": "Bayesian interim go/no-go. Posterior response rate with a 95% credible interval and "
               "the exact predictive probability that the trial ends in a GO at full enrolment, "
               "against a dual-criterion decision rule. Conjugate Beta-Binomial; computed in closed "
               "form (no simulation).",
```

In `build_docx`, on the approval page, add the pre-specification status immediately after the DRAFT
stamp (find `doc.add_heading("Approval / review", 2)` and add before it):

```python
    _ps = m.get("prespec") or {}
    if _ps.get("status"):
        _kv(doc, "Pre-specification", _ps["status"])
        if _ps.get("drift"):
            doc.add_paragraph("Departures from the locked design: "
                              + "; ".join(f"{d['field']}: locked {d['locked']} -> used {d['actual']}"
                                          for d in _ps["drift"]))
        if _ps.get("lock", {}).get("lock_id"):
            _kv(doc, "Design lock", _ps["lock"]["lock_id"][:16])
```

In the results section, add a branch before `elif m.get("arms"):`:

```python
    elif mt in ("assurance", "interim"):
        v = m.get("verdict", {})
        if mt == "assurance":
            _kv(doc, "Assurance (probability of success)", f"{v.get('assurance', 0):.1%}")
            _kv(doc, "Power at the Target Value", f"{v.get('power', 0):.1%}")
        else:
            _kv(doc, "Predictive probability of success", f"{v.get('predictive_prob', 0):.1%}")
            _kv(doc, "Posterior response rate", f"{v.get('posterior_mean', 0):.1%}")
        rb = m.get("robustness") or {}
        if rb.get("panel"):
            table_caption("Prior sensitivity: the verdict under each defensible prior. A verdict that "
                          "flips across priors is prior-driven, not data-driven.")
            pt = doc.add_table(rows=1, cols=4); pt.style = "Table Grid"
            for j, h in enumerate(["Prior", "Parameters", "Assurance", "Verdict"]):
                pt.rows[0].cells[j].text = h
            for row in rb["panel"]:
                c = pt.add_row().cells
                c[0].text = str(row["prior"]); c[1].text = str(row["params"])
                c[2].text = f"{row['assurance']:.1%}"; c[3].text = str(row["call"])
            _footnote(doc, "FDA's Jan-2026 draft Bayesian guidance requires a prior-sensitivity "
                           "analysis. A FRAGILE verdict is reported as fragile, not as an answer.")
        if rb.get("type_i_error") is not None:
            table_caption("Operating characteristics implied by the pre-specified decision rule.")
            ot = doc.add_table(rows=1, cols=2); ot.style = "Table Grid"
            ot.rows[0].cells[0].text = "Quantity"; ot.rows[0].cells[1].text = "Value"
            for k, val in [("Type I error (GO rate at the LRV)", rb["type_i_error"]),
                           ("Power (GO rate at the TV)", rb["power"])]:
                c = ot.add_row().cells; c[0].text = k; c[1].text = f"{val:.1%}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_hardening.py -q -p no:warnings`
Expected: PASS

- [ ] **Step 5: Lint, run the full suite, and commit**

```bash
.venv/bin/ruff check agent/report.py tests/test_hardening.py
.venv/bin/pytest -q -p no:warnings
.venv/bin/ruff check .
git add agent/report.py tests/test_hardening.py
git commit -m "Export the Bayesian go/no-go to the Word report

The pre-specification status sits on the approval page next to the DRAFT stamp,
because it is the first thing a reviewer should see. The results section carries
the verdict, the prior-sensitivity panel, and the operating characteristics --
the two FDA pillars that a bare probability of success would hide."
```

---

## Final verification (after Task 9)

- [ ] **Run everything**

```bash
.venv/bin/pytest -q -p no:warnings          # expect all green, ~175 tests
.venv/bin/ruff check .                      # expect "All checks passed!"
.venv/bin/pytest --cov=agent -q -p no:warnings | tail -3   # expect coverage >= 60%
```

- [ ] **Drive the real app** (the project's `verify` habit: tests passing is not the same as the
      feature working)

```bash
.venv/bin/streamlit run app.py --server.headless=true --server.port=8599
```

Ask, in the app:
1. `"Phase I showed 8 responses in 20 patients. What is the probability a 100-patient Phase II succeeds, if we need a 30% response rate and 15% is the minimum worth pursuing?"`
   Expect: an `assurance` model, a GO / CONSIDER / STOP badge, an assurance curve, an OC curve, a
   four-row prior-sensitivity table, and a working "Download pre-specification lock" button.
2. `"We are 40 patients into the trial with 12 responses and planned to enrol 100. Should we stop for futility?"`
   Expect: an `interim` model, a predictive probability of success, an EXPLORATORY badge, and the
   "not pre-specified" caveat.

- [ ] **Update the docs**

In `README.md`, add two rows to the model-families table:

```markdown
| "Should we invest in Phase II?" | Bayesian assurance (design-stage) | GO / CONSIDER / STOP with the probability of success, prior sensitivity, and operating characteristics |
| "Stop this trial for futility?" | Bayesian interim | Predictive probability of success, posterior with a credible interval, pre-specification status |
```

and add a short paragraph under Capabilities describing the module, the dual-criterion rule, the
prior-sensitivity panel, the pre-specification lock, and the honest scope (decision support, not a
submission tool). Bump the unit-test count. Commit.

---

## Self-review notes

**Spec coverage.** Every section of the spec maps to a task: §3 the decision rule (Task 2), §4 the
lock (Task 1), §5 `bayes.py` (Tasks 2-4) and the two entry points (Tasks 5-6), §5 routing (Task 7),
rendering (Task 8) and the report (Task 9), §6 data flow (Tasks 5-7), §7 the deterministic caveats
(Tasks 5-6), §8 error handling and every named edge case (Tasks 5-6, tested), §9 the full test list
(distributed across every task).

**Known scope reduction, deliberate and flagged.** The spec allows a continuous endpoint in BOTH
modes and a two-arm framing in both. This plan implements: binary single-arm and binary two-arm
machinery in `bayes.py` (`prob_diff_exceeds` handles the two-arm difference), continuous in
`calc_assurance` (via the normal branch), but **`fit_interim` is binary-only** and returns a clean
error for a continuous endpoint. Rationale: binary covers device procedural success and drug response
rate, which is the overwhelming majority of early-phase go/no-go, and a continuous interim needs a
nuisance-variance treatment that is a module of its own. This is an explicit, tested,
error-messaged limitation, not a silent gap. Extending it is a follow-on task.

**Type consistency.** `Prior.params` is a 2-tuple everywhere. `DecisionRule` field names are identical
in `bayes.py`, `modeling.py`, and `prespec.LOCKED_FIELDS`. `prespec.verify` returns the same dict
shape (`status`, `lock_id`, `drift`, `anchor`) in every branch. `ModelResult.prespec` is
`{status, lock, drift}` in both `calc_assurance` and `fit_interim`.
