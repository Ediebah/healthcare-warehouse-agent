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


def load_traces() -> list[dict]:
    return _load(TRACES)


def load_audit() -> list[dict]:
    return _load(AUDIT)


def summary(traces: list[dict] | None = None, audit: list[dict] | None = None) -> dict:
    """The production numbers you'd watch: run count, error/success rate, latency p50/p95, tokens, spend."""
    traces = load_traces() if traces is None else traces
    audit = load_audit() if audit is None else audit
    s = {"runs": len(traces), "audited": len(audit)}
    if not traces:
        return s
    lat = [t.get("latency_ms", 0) for t in traces]
    toks = [t.get("prompt_tokens", 0) + t.get("completion_tokens", 0) for t in traces]
    cost = sum(t.get("est_cost_usd", 0.0) for t in traces)
    errs = sum(1 for t in traces if t.get("error"))
    s.update({
        "errors": errs, "error_rate": errs / len(traces), "success_rate": 1 - errs / len(traces),
        "p50_ms": _pct(lat, 0.5), "p95_ms": _pct(lat, 0.95), "max_ms": max(lat) if lat else 0,
        "avg_tokens": sum(toks) // max(1, len(toks)), "total_tokens": sum(toks),
        "spend_usd": cost, "avg_cost_usd": cost / len(traces),
    })
    return s


def main() -> int:
    s = summary()
    if not s["runs"]:
        print("No traces yet — run some analyses first (agent.agent / the app), then re-run.")
        return 0
    print("── agent observability ──────────────────────────")
    print(f"  runs              : {s['runs']}")
    print(f"  error rate        : {s['errors']}/{s['runs']} = {s['error_rate']:.0%}")
    print(f"  latency ms        : p50={s['p50_ms']:.0f}  p95={s['p95_ms']:.0f}  max={s['max_ms']:.0f}")
    print(f"  tokens / run      : avg={s['avg_tokens']:,}")
    print(f"  est. spend        : ${s['spend_usd']:.3f}  (avg ${s['avg_cost_usd']:.4f}/run)")
    print(f"  queries audited   : {s['audited']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
