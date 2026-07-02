"""Observability summary over the run traces + query audit log (logs/*.jsonl).

Turns the raw traces the agent persists into the numbers you'd actually watch in production:
run count, error rate, latency p50/p95, tokens, spend, and how many warehouse queries were audited.

Run:  .venv/bin/python -m agent.observe
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "logs" / "traces.jsonl"
AUDIT = ROOT / "logs" / "audit.jsonl"


def _load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> int:
    traces, audit = _load(TRACES), _load(AUDIT)
    if not traces:
        print("No traces yet — run some analyses first (agent.agent / the app), then re-run.")
        return 0
    lat = [t.get("latency_ms", 0) for t in traces]
    toks = [t.get("prompt_tokens", 0) + t.get("completion_tokens", 0) for t in traces]
    cost = sum(t.get("est_cost_usd", 0.0) for t in traces)
    errs = sum(1 for t in traces if t.get("error"))

    print("── agent observability ──────────────────────────")
    print(f"  runs              : {len(traces)}")
    print(f"  error rate        : {errs}/{len(traces)} = {errs/len(traces):.0%}")
    print(f"  latency ms        : p50={_pct(lat,0.5):.0f}  p95={_pct(lat,0.95):.0f}  max={max(lat):.0f}")
    print(f"  tokens / run      : avg={sum(toks)//len(toks):,}")
    print(f"  est. spend        : ${cost:.3f}  (avg ${cost/len(traces):.4f}/run)")
    print(f"  queries audited   : {len(audit)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
