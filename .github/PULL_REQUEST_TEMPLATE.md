## What this changes

<!-- A short summary of the change and why it is needed. -->

## Checklist

- [ ] `pytest` passes locally (the full unit-test suite, no key needed)
- [ ] `ruff check .` is clean
- [ ] If the warehouse changed, `dbt build` passes (all data tests)
- [ ] Tests added or updated for the change
- [ ] No real patient data, PHI, or secrets added anywhere
- [ ] The statistical guardrail stays deterministic (no LLM-invented numbers, no dropped caveats)

## Notes for the reviewer

<!-- Anything worth calling out: trade-offs, follow-ups, screenshots. -->
