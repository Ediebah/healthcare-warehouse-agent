# Bayesian go/no-go decision module — design

**Date:** 2026-07-13
**Status:** approved, ready for implementation planning
**Scope:** one feature. Dose-finding (BOIN/CRM), group-sequential designs, and Bayesian borrowing
from external controls are explicitly OUT of scope and are separate future modules.

## 1. Why

The platform today answers observational and descriptive questions (readmissions, cost, prevalence)
plus a solid inference layer (regression, survival, causal, A/B, non-inferiority). Early clinical
development asks a different question: *should we invest in the next study, and should we keep going
once it starts?* That is the go/no-go decision, and in early development it is usually Bayesian.

This module adds that capability for both drug and medical-device trials, and it is built to the
three pillars of [FDA's January 2026 draft guidance on Bayesian methodology](https://www.berryconsultants.com/resource/guide-to-the-draft-fda-bayesian-guidance-2026)
(CDER/CBER) and the [2010 CDRH device guidance](https://www.fda.gov/media/71512/download):

| FDA pillar | How this module satisfies it |
|---|---|
| Pre-specified decision criteria and thresholds | Dual-criterion TV/LRV gates (§3), enforced by the pre-specification lock (§4) |
| Operating characteristics evaluated by simulation | The OC simulator: type I error, power, and GO-rate across a grid of true effects |
| Justified priors with sensitivity analysis | The prior panel: four defensible priors, with the verdict flagged FRAGILE if it flips; plus prior effective-sample-size reporting |

FDA mandates no particular software, only that it be "reliable and adequately tested." Every model in
scope here is conjugate, so the engine is closed-form and deterministic rather than sampled (see §5).
That is a deliberate rigor claim, in the same family as "the statistical guardrail is deterministic;
the LLM may phrase a caveat but never invent or drop one."

### What this module is NOT

It is a **decision-support and design-exploration tool, not a submission-grade analysis system**, and
that holds even when a user uploads real trial data. Submission-readiness is about process, not
arithmetic: it requires validated software (IQ/OQ/PQ), an ALCOA+ audit trail, CDISC-conformant
datasets, independent double programming, and a pre-specified SAP with a defined estimand. This tool
has none of those, and real data changes none of it.

The deepest reason is the one the lock addresses. A Bayesian design must have its prior, thresholds,
and interim schedule fixed *before anyone sees the data*. An interactive tool that lets you choose a
prior after looking at the interim result is a prior-shopping machine. The lock exists to make that
distinction visible and enforceable rather than pretending it away.

The legitimate footprint is still large: internal go/no-go decisions (where the dual-criterion
framework predominantly lives anyway, and which never reach a regulator), design exploration before
the protocol is written, exploratory and hypothesis-generating analyses, and independent
sanity-checking of a primary analysis.

## 2. Scope

Two modes, two framings, two endpoint types. All combinations are supported.

**Modes**

- **Design-stage** (no data). "Phase I showed 8/20 responses. What is the probability a 100-patient
  Phase II succeeds?" Produces assurance / probability of success, and emits a lock.
- **Interim** (queries the warehouse or uploaded data). "We are 40 patients into the trial with 12
  responses. Continue or stop?" Produces the posterior and the predictive probability of eventual
  success, and verifies against a lock if one is supplied.

**Framings**

- **Single-arm vs a performance goal.** The standard medical-device shape (an objective performance
  criterion): is the success rate above a fixed threshold?
- **Two-arm vs a control.** The standard drug shape: is the treatment effect above a threshold?

**Endpoints**

- **Binary** (response rate, procedural success, freedom from complication).
- **Continuous mean** (change from baseline).

Time-to-event endpoints are out of scope: they need a different likelihood and would not be conjugate.

## 3. The decision rule (dual-criterion TV/LRV)

The industry-standard early-development framework. Two thresholds are pre-specified:

- **LRV** (Lower Reference Value): the minimum effect worth pursuing.
- **TV** (Target Value): the effect we hope for.

Given the posterior of the effect `θ`:

```
GO        P(θ > TV) >= gate_tv   AND  P(θ > LRV) >= gate_lrv     (defaults 0.80 / 0.90)
STOP      P(θ > LRV) < stop_lrv                                  (default 0.10)
CONSIDER  anything else
```

A **device performance goal is the degenerate case**: set `TV == LRV == the performance goal` and the
three-way verdict collapses to GO / NO-GO against that single threshold. No separate code path.

`higher_is_better=false` (an adverse-event or mortality endpoint, where lower is better) mirrors the
comparisons, following the convention `fit_noninferiority` already uses.

## 4. The pre-specification lock

The regulatory workflow is **design → lock → execute**, not "look at the data, then choose the rule."
The lock makes that workflow real in the tool.

**At design stage**, once the user settles on a prior, a decision rule, and a planned sample size, the
tool canonicalizes those decision-relevant parameters, hashes them (SHA-256 over a sorted, float-
rounded JSON form), stamps a timestamp, and emits a **portable lock artifact** the user downloads.
The lock records the prior and its provenance, the full `DecisionRule`, `n_planned`, the endpoint type
and framing, the interim schedule if any, the operating characteristics computed at lock time, and an
optional **external anchor** (a ClinicalTrials.gov ID, a protocol version, a signed SAP reference).

**At interim**, the user supplies the lock. The tool recomputes the hash from the parameters of the
run in front of it and compares. The result is a deterministic status carried on every output:

| Status | Meaning |
|---|---|
| `PRE-SPECIFIED` | The interim run matches the lock exactly. |
| `DRIFTED` | A lock was supplied but the run's parameters differ. The output names every changed field with its locked and actual value. The verdict is still shown, but stamped as not pre-specified. |
| `EXPLORATORY` | No lock supplied. The verdict is stamped "exploratory, not pre-specified." |

The lock is **optional by design**. An unlocked run still works, so the demo keeps its "ask anything"
character; it is simply, and visibly, labelled exploratory. That is the honest state of most real
analyses.

**Why the lock lives in a file the user holds, not on the server:** the deployed app runs on an
ephemeral filesystem (established the hard way: Streamlit Cloud wipes local state on every reboot), so
a server-side lock would silently vanish. A portable artifact is also truer to reality, where the SAP
is a document the sponsor holds, not a row in the analysis tool's database.

**Honest limitation, stated in the artifact itself.** A content hash proves *integrity* (the lock was
not altered after the fact) but **not anteriority** (that it existed before the data was seen). Nothing
stops a user from fabricating a lock after peeking. Real anteriority requires an external trusted
timestamp: a protocol filed with the agency, a public registry entry, a signed and dated SAP. That is
exactly what the optional external anchor field is for, and the spec claims nothing stronger. The lock
demonstrates and enforces the *workflow*; it does not, and cannot, substitute for the institutional
machinery that makes pre-specification credible.

## 5. Architecture

### New module: `agent/bayes.py`

The decision engine as pure functions. No LLM, no I/O, no `ModelResult`. Depends only on numpy and
scipy (both already pinned). Independently testable against textbook values. Kept out of
`modeling.py`, which is already ~1,300 lines.

```python
@dataclass(frozen=True)
class Prior:
    name: str                    # "Phase-I informed" | "Vague" | "Skeptical" | "Enthusiastic"
    kind: str                    # "beta" (binary endpoint) | "normal" (continuous endpoint)
    params: tuple[float, float]  # beta: (a, b).  normal: (mu, sd).
    provenance: str              # human-readable: where this prior came from

@dataclass(frozen=True)
class DecisionRule:
    tv: float                    # Target Value
    lrv: float                   # Lower Reference Value (== tv for a device performance goal)
    gate_tv: float = 0.80        # required P(theta > tv)
    gate_lrv: float = 0.90       # required P(theta > lrv)
    stop_lrv: float = 0.10       # P(theta > lrv) below this -> STOP
    higher_is_better: bool = True

beta_posterior(a, b, x, n) -> (a', b')                  # conjugate, exact
normal_posterior(mu0, sd0, xbar, sd, n) -> (mu', sd')   # conjugate, exact
prob_exceeds(post, threshold) -> float                  # single-arm, closed form (scipy sf)
prob_diff_exceeds(post_t, post_c, threshold) -> float   # two-arm.
                                                        #   normal: closed form (a difference of
                                                        #     normals is normal).
                                                        #   beta: no closed form, so 1-D quadrature
                                                        #     P(t - c > d) = INT f_c(v) * sf_t(v + d) dv
                                                        #     on a fixed grid. Deterministic, fast,
                                                        #     vectorizable. NOT Monte Carlo.
decide(p_tv, p_lrv, rule) -> (call, reason)             # the truth table in §3
predictive_prob_success(post, observed, n_planned, rule) -> float
                                                        # EXACT for a binary endpoint: enumerate every
                                                        # possible count of successes among the
                                                        # not-yet-observed patients (a double sum over
                                                        # both arms in the two-arm framing), weight
                                                        # each by the beta-binomial predictive pmf,
                                                        # apply the final decision rule to each.
                                                        # Continuous endpoint: the normal posterior-
                                                        # predictive, integrated on a grid.
                                                        # No simulation error either way.
assurance(prior, n_planned, rule) -> float              # integrate P(GO | theta) over the prior on a
                                                        # fine grid.
operating_characteristics(prior, n_planned, rule, grid) -> list[dict]
                                                        # for each true theta in the grid: the GO rate.
                                                        # GO rate at the null = type I error;
                                                        # at the TV = power.
prior_panel(informed, endpoint) -> list[Prior]          # the four defensible priors
prior_ess(prior) -> float                               # effective sample size (beta: a + b)
```

**No Monte Carlo anywhere in the module.** Every quantity is either closed-form or deterministic
numeric integration on a fixed grid. There is no random seed to manage, results are bit-reproducible
across runs and platforms, and the tests can assert exact values rather than tolerances. This also
means the module cannot silently produce a different verdict on re-run, which matters for a tool whose
whole purpose is to support a decision. (It is also the one genuine step this module takes toward
validatability: bit-reproducibility is a prerequisite for any validated environment.)

**Cost bound.** The binary two-arm predictive enumeration is `O((m_t + 1) * (m_c + 1))` quadratures,
where `m` is the number of patients not yet observed in each arm. It is vectorized in numpy and is
well under a second for trials of a few hundred per arm. `bayes.py` enforces a documented cap on the
enumeration size and, above it, bins the predictive distribution rather than silently getting slow.

### New module: `agent/prespec.py`

The lock, as pure functions. No Bayesian math, no I/O beyond serialization. Small (~100 lines), one
clear purpose, independently testable.

```python
canonical(params: dict) -> str        # sorted keys, floats rounded to a fixed precision so that
                                      # 0.1 and 0.10 cannot produce a spurious drift
lock_id(params: dict) -> str          # SHA-256 of canonical(params)
create_lock(params, oc, anchor=None) -> dict
                                      # the portable artifact: params + lock_id + timestamp +
                                      # operating characteristics at lock time + optional external
                                      # anchor + the honest-limitation notice from §4
verify(lock: dict, params: dict) -> Prespec
                                      # -> Prespec(status, lock_id, drift=[{field, locked, actual}])
                                      # Rejects a lock whose recorded lock_id does not match its own
                                      # contents (tampering) as INVALID.
```

`ModelResult` gains one field: `prespec: dict` (status, lock_id, drift, anchor), rendered everywhere
the verdict is.

### `agent/modeling.py` — two thin entry points

They call into `bayes.py` and package results into the existing `ModelResult` contract, exactly like
every other model family.

- `calc_assurance(**params) -> ModelResult` — design-stage, takes no data. Mirrors `calc_sample_size`.
  Attaches a freshly created lock to the result.
- `fit_interim(df, **params, lock=None) -> ModelResult` — interim, takes a DataFrame of subjects
  observed so far. Mirrors every other `fit_*`. Verifies against the lock when one is supplied.

Two functions rather than one because the *routing* genuinely differs (one needs SQL, one does not),
while the Bayesian core underneath is shared.

`ModelResult` fields used:

- `verdict = {"call": "GO"|"CONSIDER"|"STOP", "reason": str, ...}` — renders in the existing verdict card.
- `series` — the curve. Design-stage: assurance vs planned n. Interim: predictive probability vs enrollment.
- `terms` — the posterior summary (estimate + 95% credible interval).
- `issues` — the deterministic caveats (§7).
- `arms` — per-arm posterior summaries in the two-arm framing.
- `prespec` — the lock status (new).

### `agent/agent.py` — routing

Two new `model_type` values in the router prompt:

- `'assurance'` — design-stage. **No data, no `analytic_sql`.** Routed through a new `_run_assurance`,
  which mirrors the existing no-data `_run_sample_size` path (special-cased in `run_analysis`
  alongside `sample_size`). Extracted params: `endpoint_type` ("proportion"|"mean"), `framing`
  ("single_arm"|"two_arm"), `n_planned`, `tv`, `lrv`, `higher_is_better`, and the prior source
  (`prior_successes` + `prior_n` from a previous study, or explicit `prior_a`/`prior_b`, or none for
  a vague default).
- `'interim'` — data-driven. `analytic_sql` returns one row per subject observed so far, with the
  outcome column and, in the two-arm framing, the arm column. `_fit_model` gains an `interim` branch.
  Also needs `n_planned` (the full planned enrollment) to compute the predictive probability.

`_interpret_model` gains guidance to LEAD with the verdict and to state the pre-specification status,
as it already does for non-inferiority.

### `app.py` — rendering

Reuses the existing verdict card and series chart. New elements: the prior-sensitivity table, a
pre-specification status badge on the verdict card, a **"Download pre-specification lock"** button
after a design-stage run, and an optional lock upload in the interim path.

### `agent/report.py` — export

A new section rendering the decision, the pre-specified criteria, the priors and their provenance, the
sensitivity panel, and the operating characteristics. The **pre-specification status appears on the
title/approval page**, next to the existing DRAFT stamp, because it is the first thing a reviewer
should see.

## 6. Data flow

**Design-stage**

```
question -> _triage -> _route(model_type='assurance', params)   [no SQL]
         -> _run_assurance -> modeling.calc_assurance -> bayes core
         -> prespec.create_lock(params, oc)
         -> ModelResult(verdict, series, issues, prespec) -> _interpret_model
         -> UI (+ lock download) + report
```

**Interim**

```
question -> _triage -> _route(model_type='interim', analytic_sql)
         -> run_query (+ existing self-heal) -> _fit_model -> modeling.fit_interim -> bayes core
         -> prespec.verify(lock, params)  [PRE-SPECIFIED | DRIFTED | EXPLORATORY | INVALID]
         -> guardrails.analyze(df) also runs (this path HAS data)
         -> ModelResult(verdict, series, issues, prespec) -> _interpret_model -> UI + report
```

On the guardrail: the interim path gets the full deterministic guardrail for free, because it has a
DataFrame. The design-stage path has no DataFrame, so its caveats live in `ModelResult.issues`,
computed in code and un-droppable by the LLM. That is the same philosophy applied where no data
exists, and it matches how `calc_sample_size` already behaves.

## 7. Deterministic caveats (always emitted)

Computed in code. The LLM may phrase them but may never invent or drop one.

1. **Pre-specification status** (§4): PRE-SPECIFIED, DRIFTED (naming every changed field), or
   EXPLORATORY. This is the first caveat, because it conditions how every other number should be read.
2. **The prior, stated explicitly**, with its parameters and provenance ("Beta(9,13), from Phase I: 8
   responses in 20 patients").
3. **Prior sensitivity**: the verdict under each of the four priors, and whether the call HOLDS or is
   FRAGILE. A verdict that flips across defensible priors is reported as fragile, never as an answer.
4. **Prior effective sample size vs observed n.** If `prior_ess > n_observed`, flag that the prior is
   doing more work than the evidence. (This is FDA's ESS-quantification requirement.)
5. **Assurance vs classical power**, when they diverge: power assumes the effect is exactly one value;
   assurance averages over the uncertainty about it, and is usually the lower, more honest number.
6. **Operating characteristics**: the type I error and power implied by the pre-specified rule.
7. **Beta(ε,ε) degeneracy guard** (§8).
8. The standing **synthetic-data** note and the **not a pre-specified SAP / not regulatory-grade**
   statement, consistent with the existing ICH-E9 language in the report.

## 8. Error handling and edge cases

Every entry point wraps its body in `try/except` and returns `ModelResult(error=...)`. Nothing raises
into the app. This is the existing convention across `modeling.py`.

Fail-closed validation with actionable messages:

- `LRV > TV` (with `higher_is_better=true`) is contradictory: error explaining the ordering.
- Proportions outside [0, 1]; `n_planned <= 0`; non-positive prior parameters.
- **Interim with `n_observed > n_planned`**: not an interim. Error saying so, and pointing at the
  final-analysis framing.
- **Interim with `n_observed == n_planned`**: the trial is complete. The predictive probability is
  degenerate (it is just the posterior decision). Report the final decision and say so, rather than
  pretending to predict.
- **Zero or perfect responses with a near-noninformative prior.** FDA's 2026 draft guidance
  specifically warns that a Beta(ε,ε) prior becomes *unexpectedly informative* at 0 or 100% response.
  Guard: when the observed data is all-success or all-failure AND the prior is near-degenerate
  (`a + b < 1`), emit a loud caveat and report the verdict as unreliable. **This is a real trap for an
  early interim look and is a required test case.**
- **Tampered lock**: a lock whose recorded `lock_id` does not match a re-hash of its own contents is
  rejected as INVALID, and the run is treated as EXPLORATORY with a loud caveat.
- **Float-formatting drift**: canonicalization rounds floats to a fixed precision, so `0.1` and `0.10`
  cannot produce a spurious DRIFTED verdict. This is a required test case.
- Empty cohort: already caught upstream by the existing empty-cohort guard before `_fit_model` runs.

## 9. Testing

The project's culture is to assert ground truth, not the absence of a crash. This module is unusually
well suited to that, because closed-form Bayesian quantities have exact known values.

**`tests/test_bayes.py` (pure engine)**

- **Conjugacy**: Beta(1,1) updated with 8 successes in 20 gives exactly Beta(9,13); posterior mean is
  exactly 9/22.
- **`prob_exceeds`** agrees with `scipy.stats.beta.sf` to machine precision.
- **Predictive probability is exact**: cross-check the enumeration against a brute-force Monte Carlo
  simulation written *in the test only*. They must agree within MC error. This validates the exactness
  claim without putting a sampler in the shipped code.
- **`prob_diff_exceeds` quadrature is right**: cross-check the beta-difference quadrature against a
  large Monte Carlo draw (again, in the test only), and against the closed form in the normal case.
- **Assurance collapses to power** (the key invariant): as the prior tightens onto a point mass at
  θ₀, assurance converges to the classical power at θ₀. Textbook-checkable and a strong test that the
  integration is right.
- **Assurance < power** when the prior has genuine spread. This is the whole reason assurance exists.
- **`decide` truth table**, including the boundary values of every gate.
- **Prior panel**: the skeptical prior is centred at or below the null; the enthusiastic one above.
- **`prior_ess`**: Beta(9,13) has an effective sample size of 22.

**`tests/test_prespec.py` (the lock)**

- The same parameters always produce the same `lock_id`, regardless of dict key order.
- **`0.1` and `0.10` produce the same lock_id** (the float-canonicalization guard).
- Changing the TV changes the `lock_id`.
- `verify` with matching parameters returns PRE-SPECIFIED.
- `verify` with a changed LRV returns DRIFTED, and the drift list names `lrv` with both its locked and
  its actual value.
- No lock returns EXPLORATORY.
- A lock whose contents were edited after creation (so its recorded hash no longer matches) is rejected
  as INVALID.

**`tests/test_modeling.py` (additions)**

- `calc_assurance` recovers a planted answer; the verdict flips GO → STOP as the LRV is raised.
- `calc_assurance` attaches a valid lock that `prespec.verify` accepts.
- `fit_interim` with data well below the LRV yields STOP FOR FUTILITY.
- **Prior sensitivity**: construct a case where the skeptical prior flips the call, and assert the
  FRAGILE flag appears in `issues`. This tests the headline feature.
- **Drift is caught end to end**: lock a design at LRV=0.15, then run the interim at LRV=0.10, and
  assert the result is stamped DRIFTED with `lrv` named.
- **Device single-arm performance goal**: TV == LRV collapses cleanly to GO/NO-GO.
- **Beta(ε,ε) degeneracy**: all-success data with a near-degenerate prior emits the unreliability
  caveat.
- Error cases: LRV > TV, `n_observed > n_planned`, out-of-range proportions.

**`tests/test_agent.py` (additions)**

- A routed `assurance` spec reaches `_run_assurance` without touching SQL (monkeypatched LLM,
  following the existing `test_quality_agent` style).
- A routed `interim` spec dispatches correctly through `_fit_model`.

The existing `tests/test_app_smoke.py` already covers rendering end-to-end.

## 10. Out of scope (deliberately)

- Bayesian dose-finding (BOIN, CRM, mTPI). The natural *next* module.
- Group-sequential designs and alpha spending. A later module.
- Hierarchical borrowing from historical or external control arms. This one genuinely needs MCMC and
  will bring its own engine; the 2026 guidance devotes substantial attention to it (static vs dynamic
  discounting, drift, prior-data conflict), so it deserves its own spec.
- Time-to-event endpoints (non-conjugate).
- **Trusted anteriority.** The lock proves integrity, not that it predates the data (§4). Anchoring to
  an external timestamp authority is out of scope; the artifact carries an optional anchor field and
  says plainly what it does not prove.
- Any claim of being a regulatory submission tool. The module demonstrates method, on synthetic data.
