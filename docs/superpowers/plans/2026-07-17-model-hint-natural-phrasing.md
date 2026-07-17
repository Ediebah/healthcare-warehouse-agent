# Broaden `_MODEL_HINT` for natural trial-decision phrasing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add decision-context phrases to `agent._MODEL_HINT` so natural go/no-go questions ("is it worth running?", "continue or stop?", "should we keep going?") reach the router, guarded by a negative test against false positives.

**Architecture:** A single regex edit to the `_MODEL_HINT` alternation in `agent/agent.py`, plus a positive test (natural phrasings match) and a negative test (ordinary aggregations do not). `_MODEL_HINT` is a pre-filter — `_route` still adjudicates model-vs-aggregate after a match, and a non-model result falls through to the aggregation path — so the only cost of a false positive is one extra `_route` call.

**Tech Stack:** Python 3.12, `re`. pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-07-17-model-hint-natural-phrasing-design.md`

## Global Constraints

- No new dependencies. No change to `_route`, `_run_assurance`, or any model path — only the `_MODEL_HINT` regex.
- Line length 120 (`.venv/bin/ruff check .`). Tests keyless (regex-only — no LLM, no network). Commit style: NO `Co-Authored-By` trailer. Coverage gate `fail_under = 60`.
- Run the full suite `.venv/bin/pytest -q -p no:warnings` before committing.
- **Negative-test correction (verified during planning):** the spec §4 listed "How many patients continue treatment?" and "How many patients keep their appointments?" as negatives, but both already match the EXISTING `how many (patient|...)` pattern, so they do not isolate the new phrases. This plan uses genuinely clean negatives instead (verified: they match neither the existing pattern nor the additions).

---

### Task 1: Broaden `_MODEL_HINT` with decision-context phrases

**Files:**
- Modify: `agent/agent.py` (the `_MODEL_HINT` regex, ~line 284-292)
- Test: `tests/test_agent.py` (append two tests)

**Interfaces:**
- Consumes: nothing. Produces: a broader `agent._MODEL_HINT` compiled regex (same object name, same type).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent.py`:

```python
def test_model_hint_matches_natural_trial_decision_phrasing():
    """Natural go/no-go phrasing (no explicit trigger word) must reach the router, not fall through
    to a 'clarification needed' response."""
    for q in [
        "Is it worth running this trial?",
        "Continue or stop?",
        "Should we keep going?",
        "Is the drug programme worth pursuing?",
        "Should we go ahead with the Phase II?",
        "Should we continue the trial or stop for futility?",
    ]:
        assert agent._MODEL_HINT.search(q), q


def test_model_hint_does_not_match_ordinary_aggregations():
    """The decision-context phrases must not misfire on ordinary aggregation questions with incidental
    words. Each of these matches NEITHER the existing pattern NOR the new phrases."""
    for q in [
        "Which patients continue treatment the longest?",   # 'continue treatment', not 'continue the trial'
        "What's the running total of encounter costs?",     # 'running total', not 'worth running'
        "Which conditions are worth investigating?",        # 'worth investigating', not 'worth running/pursuing'
        "Invest more nurses in the ICU?",                   # 'invest more ... in', not the adjacent 'invest in'
        "List the top procedures by total cost.",
    ]:
        assert agent._MODEL_HINT.search(q) is None, q
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agent.py -q -p no:warnings -k "natural_trial or ordinary_aggregations"`
Expected: `test_model_hint_matches_natural_trial_decision_phrasing` FAILS (the natural phrasings do not match the current pattern — e.g. "Is it worth running this trial?" returns None). `test_model_hint_does_not_match_ordinary_aggregations` PASSES already (nothing added yet), which is fine — it is the guard that must stay green after the change.

- [ ] **Step 3: Broaden the regex**

In `agent/agent.py`, the `_MODEL_HINT` pattern currently ends:

```python
    r"go.?(or.?)?no.?go|assurance|(probability|chance|likelihood)[^.?!]{0,60}succe|futility|stop early|"
    r"interim|predictive probability|posterior|bayesian|performance goal|de.?risk)", re.I)
```

Replace ONLY the final line (`r"interim|...de.?risk)", re.I)`) with:

```python
    r"interim|predictive probability|posterior|bayesian|performance goal|de.?risk|"
    r"worth (running|pursuing|continuing)|worth the (investment|trial|study)|"
    r"continue or stop|stop or continue|keep going|"
    r"continue the (trial|study|program|programme|drug|arm)|"
    r"should we (continue|proceed|invest|keep going)|go ahead with|invest in)", re.I)
```

(The additions are new alternatives appended inside the same `\b(...)` group; the closing `)`, `re.I)` moves to the last new line. Do not change any earlier line of the pattern.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_agent.py -q -p no:warnings`
Expected: PASS — both new tests, plus the existing `test_model_hint_matches_bayesian_go_no_go_questions` (the earlier trigger-word phrasings still match).

- [ ] **Step 5: Full suite, lint, commit**

```bash
.venv/bin/pytest -q -p no:warnings          # expect all green
.venv/bin/ruff check .                      # expect "All checks passed!"
git add agent/agent.py tests/test_agent.py
git commit -m "Broaden _MODEL_HINT so natural trial-decision phrasing routes

Natural go/no-go questions -- 'is it worth running?', 'continue or stop?',
'should we keep going?' -- missed the routing gate and fell through to a
clarification response. Add decision-context phrases (worth running/pursuing,
continue or stop, keep going, continue the trial, should we continue/proceed/
invest, go ahead with, invest in). Safe because _MODEL_HINT is a pre-filter:
_route still adjudicates model-vs-aggregate, so a false positive costs one extra
call and falls through to the aggregation path. A negative test locks in the
false-positive protection (ordinary aggregations with incidental 'continue',
'running', 'worth', 'invest' do not match)."
```

---

## Final verification (after Task 1)

- [ ] **Run everything**

```bash
.venv/bin/pytest -q -p no:warnings
.venv/bin/ruff check .
.venv/bin/pytest --cov=agent -q -p no:warnings | tail -3   # coverage >= 60%
```

- [ ] **Drive the real app** (optional but recommended — the phrasing that motivated this)

```bash
.venv/bin/streamlit run app.py --server.headless=true --server.port=8604
```

Ask: `"We are 40 patients into the trial with 12 responses and planned to enrol 100. Continue or stop?"`
Expect: it routes to a Bayesian **interim** model (not a "clarification needed" response) — the natural "continue or stop?" phrasing now reaches the router. Confirm 0 console errors.

---

## Self-review notes

**Spec coverage.** §3 the regex additions → Task 1 Step 3 (verified: all six positive phrasings match). §4 false-positive protection → the negative test in Step 1 (with the spec's two invalid negatives replaced by clean ones, noted in Global Constraints). §5 testing → Step 1 (positive + negative) + Final verification. §6 scope (single regex edit, off `main`) → this plan touches only `_MODEL_HINT` and the tests.

**Placeholder scan.** None — the exact regex and both test bodies are given verbatim.

**Type consistency.** `_MODEL_HINT` stays the same compiled-regex object (`re.compile(..., re.I)`); no signature or name change, so `run_analysis`'s `_MODEL_HINT.search(question)` call site is unaffected.
