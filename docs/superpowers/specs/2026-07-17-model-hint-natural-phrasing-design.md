# Broaden `_MODEL_HINT` for natural trial-decision phrasing — Design

**One-line:** Add decision-context phrases to `agent._MODEL_HINT` so natural go/no-go questions ("is it worth running?", "continue or stop?", "should we keep going?") reach the router, without misrouting ordinary aggregation questions.

**Status:** design, approved, awaiting spec review.

## 1. Motivation

`_MODEL_HINT` is the regex pre-filter that decides whether a question reaches `_route` (the inferential-model router) at all. Live drives of the two-arm interim and two-arm assurance showed natural phrasings — "is it worth running?", "continue or stop?" — miss the gate and fall through to a "clarification needed" response, even though the question is clearly a trial-decision question. The feature works with trigger words ("assurance", "futility", "predictive probability") but not with everyday phrasing.

## 2. Why broadening is safe

`_MODEL_HINT` is only a **pre-filter**. On a match, `_route` (an LLM) still classifies the question as a model or an aggregation. If `_route` returns `mode: aggregate` (or a model type with no `analytic_sql`), `run_analysis` **falls through** to the normal triage/aggregation path (`agent/agent.py` — the `if _MODEL_HINT.search(question):` block returns only for `sample_size`, `assurance`, or a real model spec; otherwise execution continues to `_triage`). So a false-positive match costs **one extra `_route` LLM call**, not a misrouted answer. This bounds the downside and justifies moderate breadth.

## 3. The change

Append one alternation group of multi-word, decision-context phrases to the `_MODEL_HINT` pattern (they attach to the existing `\b(...)` alternation). Each phrase requires a decision word-pairing, so it does not fire on ordinary questions that merely contain "continue", "stop", "worth", or "running":

- `worth (running|pursuing|continuing)`
- `worth the (investment|trial|study)`
- `continue or stop` | `stop or continue`
- `keep going`
- `continue the (trial|study|program|programme|drug|arm)`
- `should we (continue|proceed|invest|keep going)`
- `go ahead with`
- `invest in`

No other code changes. `_route` and the downstream flow are unchanged.

## 4. False-positive protection (the guard test)

Add a negative test asserting that ordinary aggregation questions with incidental words do **not** match, so future broadening cannot silently regress the precision:

- "How many patients continue treatment?" (has "continue treatment", not "continue the trial"/"continue or stop")
- "What's the running total of encounter costs?" ("running total", not "worth running")
- "Which conditions are worth investigating?" ("worth investigating", not "worth running/pursuing/continuing")
- "How many patients keep their appointments?" ("keep their", not "keep going")

The `invest in` phrase requires literal adjacency, so "invest more nurses in the ICU" does not match; "invest in the next study" does (intended — it is a design question). `invest in` is unlikely in an aggregation over this healthcare warehouse, and a false positive only costs one extra `_route` call, so it is kept.

## 5. Testing

- **Positive** — extend the existing `test_model_hint_matches_bayesian_go_no_go_questions` (or add a sibling) with: "Is it worth running this trial?", "Continue or stop?", "Should we keep going?", "Is the drug programme worth pursuing?", "Should we go ahead with the Phase II?"
- **Negative** — a new `test_model_hint_does_not_match_ordinary_aggregations` with the §4 questions, asserting `_MODEL_HINT.search(q) is None`.
- Full suite green, ruff clean. Keyless tests (these only exercise the regex — no LLM, no network).

## 6. Scope

A single regex edit plus two tests. No change to `_route`, `_run_assurance`, or any model path. Independent of the pending two-arm assurance PR (branches off `main`).
