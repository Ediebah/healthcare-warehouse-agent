# Assurance OC-grounded robustness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the degenerate verdict-flip `fragile` flag in `modeling.calc_assurance`'s prior-sensitivity panel with an operating-characteristics-grounded `under_powered` signal (power at the TV < 0.80) and a descriptive prior-sensitivity caveat.

**Architecture:** One contained change in `agent/modeling.py` (`_sensitivity` returns just the panel; `calc_assurance` computes `under_powered` from the already-computed power and rewrites two caveat lines), plus text-only edits in `agent/bayes.py`, `agent/agent.py`, `agent/report.py`, and one test rewrite in `tests/test_modeling.py`. The four-prior panel (FDA prior-sensitivity context) is retained.

**Tech Stack:** Python 3.12, numpy, scipy, pandas. pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-07-16-assurance-oc-robustness-design.md`

## Global Constraints

- No new dependencies. No Monte Carlo in shipped code.
- Never raise into the app (the change is inside `calc_assurance`'s existing `try/except`).
- Caveats are deterministic, appended to `ModelResult.issues`.
- **Do NOT touch the specification-curve robustness** (`verdict` = "robust"/"mostly robust"/"fragile") for regression models: `modeling.specification_curve`, the ROBUSTNESS-line guidance in `_interpret_model`, `app.py`'s `{"robust","mostly robust","fragile"}` badge, and `tests/test_robustness.py`. Different feature; leave it.
- `under_powered ≡ power < 0.80` (power at the TV, already computed by `type_i_and_power`).
- Line length 120 (`.venv/bin/ruff check .`). Tests keyless. Commit style: NO `Co-Authored-By` trailer. Coverage gate `fail_under = 60`.
- Run the full suite `.venv/bin/pytest -q -p no:warnings` before committing.

---

### Task 1: Retire the FRAGILE flag for OC-grounded robustness

**Files:**
- Modify: `agent/modeling.py` (`_sensitivity`; `calc_assurance`)
- Modify: `agent/bayes.py` (`prior_panel` docstring)
- Modify: `agent/agent.py` (`_interpret_model` assurance guidance)
- Modify: `agent/report.py` (panel footnote)
- Test: `tests/test_modeling.py` (rewrite one test into two)

**Interfaces:**
- `_sensitivity(prior, n_planned, rule, sd) -> list[dict]` (was `-> tuple[list[dict], bool]`).
- `ModelResult.robustness` for an assurance run gains `"under_powered": bool` and drops `"fragile"`.

- [ ] **Step 1: Rewrite the failing test**

In `tests/test_modeling.py`, REPLACE the existing test (currently at ~lines 337-343):

```python
def test_assurance_flags_a_fragile_verdict_when_the_skeptical_prior_flips_it():
    # engineered so the informed prior says GO but a skeptic is not yet convinced -> must be fragile
    r = modeling.calc_assurance(n_planned=40, tv=0.30, lrv=0.15, prior_successes=14, prior_n=20)
    joined = " ".join(r.issues).lower()
    assert r.robustness["fragile"] is True
    assert "fragile" in joined and "verdict holds across" not in joined
    assert any(row["prior"] == "Skeptical" for row in r.robustness["panel"])
```

with these TWO tests:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_modeling.py -q -p no:warnings -k "underpowered or well_powered"`
Expected: FAIL with `KeyError: 'under_powered'` (the robustness dict still has `fragile`, not `under_powered`).

- [ ] **Step 3a: `_sensitivity` returns just the panel**

In `agent/modeling.py`, REPLACE:

```python
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
```

with:

```python
def _sensitivity(prior, n_planned, rule, sd) -> list[dict]:
    """The prior-sensitivity panel: each defensible prior's assurance and its prior-only verdict. The
    assurance column shows how much the probability of success depends on the choice of prior."""
    rows = []
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
    return rows
```

- [ ] **Step 3b: `calc_assurance` — compute `under_powered`, update the robustness dict**

In `agent/modeling.py`, REPLACE:

```python
        panel, fragile = _sensitivity(prior, n_planned, rule, sd)
```

with:

```python
        panel = _sensitivity(prior, n_planned, rule, sd)
        under_powered = power < 0.80
```

Then REPLACE:

```python
        mr.robustness = {"panel": panel, "fragile": fragile, "oc": oc, "framing": framing,
                         "type_i_error": round(t1, 4), "power": round(power, 4)}
```

with:

```python
        mr.robustness = {"panel": panel, "under_powered": under_powered, "oc": oc, "framing": framing,
                         "type_i_error": round(t1, 4), "power": round(power, 4)}
```

- [ ] **Step 3c: `calc_assurance` — the two new caveat lines**

In `agent/modeling.py`, REPLACE:

```python
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
```

with:

```python
        assur_vals = [r["assurance"] for r in panel]
        skept = next((r["assurance"] for r in panel if r["prior"] == "Skeptical"), min(assur_vals))
        issues.append(f"Prior sensitivity: across the four defensible priors the assurance ranges from "
                      f"{min(assur_vals):.0%} to {max(assur_vals):.0%} (see the panel) -- a skeptic "
                      f"centred at the LRV expects {skept:.0%}, your prior expects {assur:.0%}.")
        if under_powered:
            issues.append(f"UNDER-POWERED: power at the TV is only {power:.0%}, below the conventional "
                          "80%. Even if the true effect equals the Target Value, this design reaches GO "
                          f"only {power:.0%} of the time -- the binding limitation here, more than the "
                          "choice of prior. Increase n or revisit the design.")
        else:
            issues.append(f"Adequately powered: power at the TV is {power:.0%} (at or above the "
                          "conventional 80%): the design can reliably detect an effect at the Target Value.")
        issues.append(f"Operating characteristics: type I error {t1:.1%} (the chance of a GO when the "
                      f"true effect is only at the LRV) and power {power:.1%} (the chance of a GO when "
                      f"it is at the TV).")
```

- [ ] **Step 3d: reword the `prior_panel` docstring (bayes.py)**

In `agent/bayes.py`, REPLACE:

```python
def prior_panel(informed: Prior, rule: DecisionRule) -> list[Prior]:
    """Four defensible priors. If the verdict flips across them it is FRAGILE, not an answer.

    This is FDA's prior-sensitivity requirement (Jan 2026 draft guidance): show that the trial's
    conclusion is robust across plausible alternative priors, not an artefact of one choice.
    """
```

with:

```python
def prior_panel(informed: Prior, rule: DecisionRule) -> list[Prior]:
    """Four defensible priors for a prior-sensitivity analysis: the assurance under each shows how much
    the probability of success depends on the choice of prior.

    This is FDA's prior-sensitivity requirement (Jan 2026 draft guidance): show how the trial's
    probability of success varies across plausible alternative priors, not just under one choice.
    """
```

- [ ] **Step 3e: reword the assurance interpretation guidance (agent.py)**

In `agent/agent.py`, in `_interpret_model`, REPLACE:

```python
        "the type I error and power. If the verdict is FRAGILE across priors, LEAD the caveats with "
        "that — a prior-driven call is not a data-driven one. Say plainly that this is a design-stage "
```

with:

```python
        "the type I error and power. If the design is UNDER-POWERED (power at the TV below 80%), LEAD "
        "the caveats with that; report the assurance spread across the priors as the prior-sensitivity "
        "analysis. Say plainly that this is a design-stage "
```

- [ ] **Step 3f: reword the report footnote (report.py)**

In `agent/report.py`, REPLACE:

```python
            _footnote(doc, "FDA's Jan-2026 draft Bayesian guidance requires a prior-sensitivity "
                           "analysis. A FRAGILE verdict is reported as fragile, not as an answer.")
```

with:

```python
            _footnote(doc, "FDA's Jan-2026 draft Bayesian guidance requires a prior-sensitivity "
                           "analysis: the panel shows how the probability of success varies across "
                           "defensible priors. The operating characteristics below flag whether the "
                           "design is adequately powered.")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_modeling.py -q -p no:warnings -k "assurance or interim"`
Expected: PASS (the two new tests plus the existing assurance/interim tests).

- [ ] **Step 5: Full suite, lint, commit**

```bash
.venv/bin/pytest -q -p no:warnings          # expect all green
.venv/bin/ruff check .                      # expect "All checks passed!"
git add agent/modeling.py agent/bayes.py agent/agent.py agent/report.py tests/test_modeling.py
git commit -m "Replace the degenerate assurance FRAGILE flag with an OC-grounded signal

The verdict-flip fragility flag was degenerate: the Skeptical (at LRV) and
Enthusiastic (at TV) reference priors are pinned to CONSIDER, so every GO and
STOP was flagged FRAGILE regardless of the evidence. Replace it with
under_powered = power at the TV < 0.80 -- the operating characteristic that
actually measures whether the design can detect the effect -- and make the
prior-sensitivity caveat descriptive (report the assurance spread). The
four-prior panel is retained as the FDA-required prior-sensitivity context. The
separate spec-curve robustness for regression models is untouched."
```

---

## Final verification (after Task 1)

- [ ] **Run everything**

```bash
.venv/bin/pytest -q -p no:warnings
.venv/bin/ruff check .
.venv/bin/pytest --cov=agent -q -p no:warnings | tail -3   # coverage >= 60%
```

- [ ] **Drive the real app**

```bash
.venv/bin/streamlit run app.py --server.headless=true --server.port=8602
```

Ask: `"Phase I showed 8 responses in 20 patients. What is the probability a 100-patient Phase II succeeds, if we need a 30% response rate and 15% is the minimum worth pursuing?"`
Expect: a GO badge, the four-prior sensitivity table, an **UNDER-POWERED** caveat (power at the TV ~29%), the descriptive prior-sensitivity spread line, and **no FRAGILE text**.

Then ask the same with `16 responses in 20 patients`: expect a GO with **no under-powered caveat** and no FRAGILE text.

- [ ] **Update the docs**

In `README.md`, if the Bayesian go/no-go paragraph mentions the FRAGILE / prior-sensitivity flag, adjust it to describe the operating-characteristics-grounded robustness signal. Commit if changed. (CONCEPTS.md §26, if present, is git-ignored — edit but do not commit.)

---

## Self-review notes

**Spec coverage.** §2 the change → Task 1 Steps 3a-3f. §3 threshold (`power < 0.80`) → Step 3b. §4 no app.py change → confirmed (no app.py edit in the plan). §5 testing → Step 1 (two tests) + Final verification. §6 scope (calc_assurance only) → all edits are in the assurance path or its text.

**Type consistency.** `_sensitivity` now returns `list[dict]` and is called as `panel = _sensitivity(...)` in Step 3b. `robustness["under_powered"]` set in Step 3b, asserted in Step 1. `robustness["fragile"]` removed everywhere (only consumer was the rewritten test — verified by grep: no other `.py` reads `robustness["fragile"]`).

**Blast radius confirmed.** The assurance `fragile` touched exactly: `modeling._sensitivity`/`calc_assurance`, `bayes.prior_panel` docstring, `agent._interpret_model` guidance, `report.py` footnote, and one `test_modeling.py` test. The spec-curve `fragile` (regression multiverse) is a separate mechanism and is explicitly out of scope.
