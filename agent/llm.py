"""Thin OpenAI wrapper. Loads agent/.env, exposes a single `complete()` helper.

The client is created lazily so this module imports cleanly even with no API key set
(the deterministic parts of the agent and the tests don't need one).
"""
from __future__ import annotations
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
_client = None


class LLMError(Exception):
    pass


def _get_client():
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMError(
                "OPENAI_API_KEY is not set. Copy agent/.env.example to agent/.env and add your key."
            )
        from openai import OpenAI
        _client = OpenAI()
    return _client


def complete(system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
    """One chat completion. Returns the assistant text (a JSON string if json_mode)."""
    kwargs = dict(
        model=MODEL,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def complete_json(system: str, user: str, temperature: float = 0.0) -> dict:
    """complete() in JSON mode, parsed. Raises LLMError on unparseable output."""
    raw = complete(system, user, json_mode=True, temperature=temperature)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"Model did not return valid JSON: {e}\n---\n{raw[:500]}") from e
