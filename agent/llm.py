"""Thin OpenAI wrapper + lightweight cost/latency tracing.

The client is created lazily so this imports cleanly with no API key. Every call records tokens,
latency, and an estimated cost into a per-run trace the agent can attach to its result.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
_client = None

# rough USD per 1K tokens (input, output) — for an order-of-magnitude cost estimate
_COST = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
}
_TRACE: list[dict] = []


class LLMError(Exception):
    pass


def reset_trace() -> None:
    _TRACE.clear()


def trace_summary() -> dict:
    ci, co = _COST.get(MODEL, (0.0, 0.0))
    pt = sum(t["prompt_tokens"] for t in _TRACE)
    ct = sum(t["completion_tokens"] for t in _TRACE)
    return {
        "calls": len(_TRACE),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "latency_ms": round(sum(t["ms"] for t in _TRACE)),
        "est_cost_usd": round(pt / 1000 * ci + ct / 1000 * co, 4),
    }


def _get_client():
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMError(
                "OPENAI_API_KEY is not set. Copy agent/.env.example to agent/.env and add your key "
                "(then restart the app if it was already running)."
            )
        from openai import OpenAI
        _client = OpenAI()
    return _client


def complete(system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
    kwargs = dict(
        model=MODEL,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    t0 = time.perf_counter()
    resp = _get_client().chat.completions.create(**kwargs)
    ms = (time.perf_counter() - t0) * 1000
    u = getattr(resp, "usage", None)
    _TRACE.append({
        "ms": ms,
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
    })
    return resp.choices[0].message.content or ""


def complete_json(system: str, user: str, temperature: float = 0.0) -> dict:
    raw = complete(system, user, json_mode=True, temperature=temperature)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"Model did not return valid JSON: {e}\n---\n{raw[:500]}") from e
