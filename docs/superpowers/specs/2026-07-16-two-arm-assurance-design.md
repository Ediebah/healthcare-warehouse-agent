# Two-arm design-stage Bayesian assurance — Design

**One-line:** Extend `modeling.calc_assurance` to a binary **two-arm** (treatment vs control) framing: the probability a randomized trial ends in GO before it runs, deciding on the **risk difference**, so a controlled trial can be *planned* as well as monitored at interim.

**Status:** design, approved, awaiting spec review.

## 1. Motivation and scope

The shipped `calc_assurance` answers the single-arm design question (a response rate against a fixed performance goal). The shipped two-arm `fit_interim` monitors a randomized trial mid-flight. Missing is the design-stage number for a randomized trial: *before it runs*, what is the probability this treatment-vs-control trial ends in GO? That is what tells a sponsor how many patients to enrol. This feature adds it, reusing the two-arm decision machinery (`go_grid_diff`, `prob_diff_exceeds`) and the OC-grounded robustness signal already in the module.

**In scope:** binary endpoint, two arms; a **treatment prior** (Phase-I data / Beta / vague fallback) plus a **known control rate** (a point value); decision on the absolute risk difference `d = rate_t − rate_c` with the dual-criterion rule; assurance, the assurance-vs-n planning curve, operating characteristics over the true risk difference, the OC-grounded `under_powered` robustness flag with a treatment-prior sensitivity panel, the pre-specification lock, and full rendering (reused from single-arm assurance).

**Out of scope (deferred, error-messaged):** unequal allocation (1:1 assumed); a control *prior* rather than a point rate; a continuous endpoint (binary only, matching the two-arm interim); a two-arm interim already exists and is untouched.

## 2. The two roles of the control rate (the crux)

- **In the analysis decision** (`go_grid_diff`, which decides GO at trial end): the control arm is analyzed with a **vague `Beta(1,1)`** prior on its observed data — identical to how the shipped two-arm interim decides. A real two-arm trial observes a control arm and analyses the difference; the control rate is *not* assumed known at analysis.
- **In the assurance / OC weighting** (the design belief): the control responds at the **known `control_rate`**, contributing **sampling** variability only (a plain Binomial), while the treatment count is averaged over its prior-predictive (beta-binomial). The known control rate is a *design assumption* about where control will land, not something the analysis treats as fixed.

So two-arm assurance is the exact mixed sum
`assurance = Σ_{s_t, s_c} BetaBinom(s_t; n_t, prior_t) · Binom(s_c; n_c, control_rate) · go_grid_diff[s_t, s_c]`.
Exact, deterministic, no Monte Carlo.

## 3. New `bayes.py` functions

```
assurance_diff(prior_t, control_rate, n_planned_t, n_planned_c, rule) -> float
operating_characteristics_diff(prior_t, control_rate, n_planned_t, n_planned_c, rule, grid=None) -> list[dict]
```

Both build the analysis decision grid once: `go = go_grid_diff(prior_t, Prior("Vague","beta",(1.0,1.0),...), n_planned_t, n_planned_c, rule)` — shape `(n_planned_t+1, n_planned_c+1)`.

- **`assurance_diff`**: `w_t = betabinom.pmf(arange(n_t+1), n_t, *prior_t.params)` (treatment prior-predictive); `w_c = binom.pmf(arange(n_c+1), n_c, control_rate)` (control at the known rate, sampling only); return `float(w_t @ go @ w_c)`. Requires `prior_t.kind == "beta"` (binary only) and `0 <= control_rate <= 1`, else `ValueError`.
- **`operating_characteristics_diff`**: fix control at `control_rate` (`w_c` as above). For each true treatment rate `θ_t` in the grid, `w_t = binom.pmf(arange(n_t+1), n_t, θ_t)`, `go_rate = float(w_t @ go @ w_c)`; append `{"theta": θ_t - control_rate, "theta_t": θ_t, "go_rate": go_rate}` — note `"theta"` carries the **risk difference** `θ_t − control_rate` (so the existing `oc_curve_chart`, which reads `theta`/`go_rate`, plots the curve against the risk difference), with `"theta_t"` kept for reference. Default grid: treatment rates spanning `[control_rate + lrv - 2·span, control_rate + tv + 2·span]` clipped to `[0,1]`, unioned with `{control_rate+lrv, control_rate+tv}` so type-I and power land exactly (`span = abs(tv-lrv) or 0.1`).
- **Type-I / power** are read off the OC as the GO rate at `θ_t = control_rate + lrv` and `θ_t = control_rate + tv`. Rather than a new `type_i_and_power_diff`, compute these two points directly in `calc_assurance` by calling `operating_characteristics_diff` with `grid=[control_rate+lrv, control_rate+tv]` (deterministic, exact, mirrors the single-arm `type_i_and_power` intent).

**Cost.** `go_grid_diff` is `O((n_t+1)(n_c+1))` quadratures, already vectorised to one matmul in the shipped code; the weightings are `O((n_t+1)(n_c+1))` numpy. For a planned trial of a few hundred per arm this is well under a second. No new cap needed (the design-stage grids are bounded by `n_planned//2`, and `go_grid_diff` already handles its own size).

## 4. `calc_assurance` two-arm branch (`modeling.py`)

`calc_assurance` gains `framing="two_arm"` and a `control_rate` parameter. When `framing == "two_arm"`:

- **Validation** (all via the existing `_err`, never raises): `control_rate` required and in `[0,1]`; TV/LRV are risk differences in `[-1,1]` (may be 0 or negative); `higher_is_better and lrv > tv` (or the reverse) rejected; `n_planned` positive; `endpoint_type` must be `proportion` (continuous two-arm assurance is out of scope → clean error).
- **Treatment prior**: `_build_prior("proportion", tv, lrv, prior_successes, prior_n, prior_a, prior_b, None, None)` — the existing helper (Phase-I data, a supplied Beta, or the vague fallback).
- **Allocation**: `n_planned` is the TOTAL; 1:1 → `n_t = n_c = n_planned // 2`.
- **Prior-only verdict** (design stage, no data): shift the threshold by the control rate — `p_tv = prob_exceeds("beta", a_t, b_t, clip(control_rate + tv, 0, 1), higher_is_better)`, `p_lrv` likewise with `lrv`; `call, reason = decide(p_tv, p_lrv, rule)`. (Mirrors the single-arm prior-only verdict, thresholds shifted onto the treatment-rate scale.)
- **Compute**: `assur = assurance_diff(...)`; `oc = operating_characteristics_diff(...)`; `t1, power` = the two GO-rate points at `control_rate+lrv` / `control_rate+tv`; `panel = _sensitivity_diff(...)` (see §5); `under_powered = power < 0.80`.
- **`ModelResult`** (same shape single-arm assurance emits, so rendering is free): `model_type="assurance"`; `verdict = {call, reason, assurance, power}`; `series` = assurance-vs-n planning curve (`assurance_diff` at a range of total n, split 1:1); `robustness = {"panel": panel, "under_powered": under_powered, "oc": oc, "framing": "two_arm", "type_i_error": t1, "power": power}`; `prespec` lock with `framing="two_arm"`, `control_rate` added to the locked params; `issues` = the deterministic caveats (prior + control-rate provenance; the descriptive prior-sensitivity spread; the OC-grounded under-powered/adequately-powered line; the type-I/power line; the assurance-vs-power line).
- `fit_stat` and `note` name the framing (treatment prior vs a control at `control_rate`, risk-difference TV/LRV).

## 5. Robustness: OC-grounded, treatment-prior panel

Reuse the just-merged OC-grounded signal (`under_powered = power < 0.80`). Add `_sensitivity_diff(prior_t, control_rate, n_t, n_c, rule)` — the two-arm analog of `_sensitivity`: for each treatment prior in a panel (informed, vague `Beta(1,1)`, skeptical centred at `control_rate+lrv`, enthusiastic centred at `control_rate+tv`, each ESS ~10 on the treatment-rate scale), report `assurance_diff` and the prior-only difference verdict, holding the control rate fixed. The caveat is descriptive (assurance spread across the panel), exactly as the single-arm fix now does. `_build_prior`'s panel logic is treatment-only here; construct the four treatment `Prior`s inline (the control rate does not vary).

## 6. Routing (`agent.py`)

`calc_assurance` is the design-stage, no-data path (`_run_assurance`). Extend:
- `_ASSURANCE_KEYS` gains `"framing"`, `"control_rate"` (treatment-prior keys `prior_successes`/`prior_n`/`prior_a`/`prior_b` already present).
- The `assurance` router description gains a two-arm clause: "For a RANDOMIZED / two-arm design (treatment vs a concurrent control), set `framing`='two_arm', `control_rate`=the known control response rate, and express `tv`/`lrv` as RISK DIFFERENCES treatment-minus-control; supply the treatment prior from Phase I as `prior_successes`+`prior_n`. n_planned is the TOTAL across both arms."
- `_MODEL_HINT` already matches assurance/go-no-go phrasing — no change.

## 7. Rendering (mostly reuse)

The two-arm assurance emits the same `ModelResult` fields single-arm assurance already renders — no new chart or app/report branch is required:
- `app.py` and `agent/report.py`: the verdict badge, assurance curve (`assurance_curve_chart`), OC curve (`oc_curve_chart`), prior-sensitivity table, lock download, and the docx assurance section all read the existing fields.
- The OC chart reads `theta`/`go_rate`; because `operating_characteristics_diff` emits `"theta"` = the risk difference (§3), the curve is correctly scaled around 0. The chart's axis title stays the generic "true effect" it already uses (renaming it to "true risk difference" is an optional one-line cosmetic follow-up, not required here).
- The prior-sensitivity table and caveats convey the treatment-prior-vs-known-control context in text.

## 8. Error handling and edge cases (tested)

- `control_rate` missing or outside `[0,1]` → clean error.
- TV/LRV outside `[-1,1]`, or `lrv > tv` (higher-is-better) → clean error.
- `control_rate + tv` (or `+lrv`) outside `[0,1]` → the shifted threshold is clipped to `[0,1]` (a threshold at/above 1 makes `prob_exceeds` ~0, at/below 0 makes it ~1 — correct limiting behaviour).
- `endpoint_type != "proportion"` in two-arm → clean "binary only" error.
- Everything returns `ModelResult(..., error=…)`; the branch is inside `calc_assurance`'s existing `try/except`.

## 9. Test plan (TDD, keyless, exact)

New tests in `tests/test_bayes.py` and `tests/test_modeling.py`:
- **`assurance_diff` vs Monte Carlo**: draw `θ_t` from the treatment prior, set `θ_c = control_rate`, simulate both arms' counts, decide via the shipped `go_grid_diff` reference; cross-check to ~0.005 (mirrors the interim's PPoS test).
- **Monotonicity**: `assurance_diff` rises as the treatment prior mean rises above `control_rate`; ~0 when the treatment prior sits below control by more than the LRV.
- **`operating_characteristics_diff`**: GO rate rises with the true risk difference; type-I (at `control_rate+lrv`) < power (at `control_rate+tv`).
- **`calc_assurance` two-arm end-to-end**: a strong expected treatment (prior mean well above `control_rate+tv`) → GO; an under-powered design → `under_powered is True` and the UNDER-POWERED caveat; a well-powered design → not flagged; the four-prior panel present; the lock round-trips to PRE-SPECIFIED with `framing="two_arm"`; errors on missing/out-of-range `control_rate`.
- **Routing**: a two-arm assurance spec dispatches through `_run_assurance` with `framing`, `control_rate`, and the treatment prior, taking the no-data path (no SQL).
- Full suite green, ruff clean, coverage ≥ 60%.
- **Drive the real app**: a two-arm design question ("we expect the new drug to respond ~55% from Phase I (11/20), control is 35%, plan 200 total, TV a 15-point benefit, LRV any benefit") → an assurance GO/CONSIDER/STOP badge, the assurance-vs-n curve, the OC curve over the risk difference, the prior panel, and the lock download; confirm the under-powered flag behaves.

## 10. References

Same grounding as the two-arm interim (spec `2026-07-15-two-arm-interim-design.md` §2): the risk-difference dual-criterion for a randomized binary trial (BOP2-DC / Zhao et al. 2023; Lalonde et al. 2007), assurance as Bayesian power (O'Hagan et al. 2005), and FDA's 2026 draft framing of a success criterion as `Pr(d > a) ≥ c` with simulated operating characteristics.
