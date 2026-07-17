# Assurance robustness: retire the degenerate FRAGILE flag — Design

**One-line:** Replace the always-on verdict-flip `fragile` flag in the design-stage assurance prior-sensitivity panel with an operating-characteristics-grounded robustness signal (`under_powered`), and make the prior-sensitivity caveat descriptive.

**Status:** design, approved, awaiting spec review.

## 1. Motivation

`modeling.calc_assurance`'s prior-sensitivity panel flags a verdict FRAGILE when the four priors' verdicts disagree (`len(set(calls)) > 1`). This is **degenerate for the two verdicts that matter**: the Skeptical prior (centred at the LRV) and the Enthusiastic prior (centred at the TV), each carrying only ~10 effective observations, cannot clear the 80% GO gate or fall below the 10% STOP threshold — both are mathematically pinned to CONSIDER. So every GO and every STOP necessarily differs from them and is flagged FRAGILE.

Verified empirically across a design sweep: the Skeptical and Enthusiastic priors produce **only CONSIDER**, ever; GO was fragile 23/23, STOP 11/11, CONSIDER 0/14. The reference priors' assurances are **fixed by the design geometry** (n, TV, LRV) and independent of the evidence — for TV 0.30 / LRV 0.15 the Skeptical prior gives 6% assurance whether Phase I was 16/20 or 2/20 — so no verdict- or skeptical-collapse comparison discriminates a sound design from a fragile one. The genuinely discriminating quantity is continuous and already computed: the **power at the TV** (the operating characteristic). The truly fragile design in the sweep — Phase I 8/20 → GO with only 28.9% power — is caught by power, not by the prior panel.

## 2. The change

**`_sensitivity(prior, n_planned, rule, sd)`** stops computing the verdict-flip bool. It returns only the panel rows (`list[dict]`), each still carrying its per-prior assurance and call — the prior-sensitivity table, retained as FDA-required context. Docstring reworded (drop "a verdict that flips is FRAGILE").

**`calc_assurance`:**
- `panel = _sensitivity(...)` (was `panel, fragile = ...`).
- `under_powered = power < 0.80` (power at the TV, already computed by `type_i_and_power`).
- `mr.robustness`: replace `"fragile": fragile` with `"under_powered": under_powered`. Panel, `oc`, `type_i_error`, `power`, `framing` unchanged.
- Issues: replace the `if fragile / else` branch with two deterministic lines:
  - **Prior sensitivity (descriptive):** report the assurance spread from the panel — the min and max across the four priors, naming the skeptical and informed values. No verdict-flip claim.
  - **OC-grounded robustness:** if `under_powered`, an UNDER-POWERED caveat stating power at the TV is below the conventional 80% and what that means (reaches GO only `power` of the time even when the true effect equals the TV — the binding limitation, more than the prior); else an "adequately powered" line.
  - The existing "assurance exceeds power at the TV" caveat (prior-inflation) is unchanged — it already covers that concern.

**Text-only edits (no behaviour change):**
- `bayes.prior_panel` docstring: the panel shows how assurance varies across defensible priors (prior-sensitivity), not "verdict flips → FRAGILE".
- `agent._interpret_model` assurance guidance: lead the caveats with UNDER-POWERED when power at the TV < 80%; report the assurance spread across priors as the prior-sensitivity analysis (drop the "FRAGILE across priors" instruction).
- `report.py` panel footnote: the panel is the prior-sensitivity analysis (how the probability of success varies across priors); the operating characteristics flag whether the design is adequately powered. Drop "A FRAGILE verdict is reported as fragile."

**Explicitly unchanged (a different feature — do NOT touch):** the specification-curve robustness `verdict` ("robust" / "mostly robust" / "fragile") for adjusted regression models (`modeling.specification_curve`, `_interpret_model`'s ROBUSTNESS-line guidance, `app.py`'s `{"robust","mostly robust","fragile"}` badge, `tests/test_robustness.py`). Those are a separate mechanism and are correct.

## 3. Threshold

`under_powered ≡ power < 0.80`. 0.80 is the conventional power bar; `power` here is the GO rate at the TV, already computed. `≥ 0.80` is adequate (not flagged). Works for every framing (single-arm, device performance goal where tv == lrv, and lower-is-better, since `power` is the GO rate at the TV in all cases).

## 4. No app.py change

`app.py` renders the panel table and the issues list (updated at source) for an assurance result; it never reads `robustness["fragile"]`. The only `app.py` "fragile" reference is the separate spec-curve verdict badge (`rb.get("verdict")`), which this change does not affect. `report.py` likewise reads `panel`, `type_i_error`, `power` — not the flag — so renaming `fragile` → `under_powered` is safe.

## 5. Testing

- Rewrite the one degenerate test (`test_assurance_flags_a_fragile_verdict_when_the_skeptical_prior_flips_it`) into:
  - `test_assurance_flags_an_underpowered_design`: an under-powered GO (Phase I 8/20 vs TV 0.30 / LRV 0.15 at n=100 → power ≈ 29%) → `robustness["under_powered"] is True` and an "under-powered" issue is present; the four-prior panel is still present.
  - `test_assurance_does_not_flag_a_well_powered_design`: an adequately-powered GO (Phase I 16/20 → power ≈ 89%) → `robustness["under_powered"] is False` and no "under-powered" issue; panel present.
- Confirm no other test asserts `robustness["fragile"]` (grep) — only that one test does.
- Full suite green, ruff clean, coverage ≥ 60%.
- Drive the real app on an under-powered design and confirm the UNDER-POWERED caveat renders and no false FRAGILE text appears.

## 6. Scope

Design-stage assurance only (`calc_assurance`). The interim path (`fit_interim`, single- and two-arm) never ran a prior panel and is untouched. No new dependencies; no Monte Carlo; never-raise contract preserved (the change is inside the existing `calc_assurance` try/except).
