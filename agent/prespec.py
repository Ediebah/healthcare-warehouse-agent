"""Pre-specification lock: design -> lock -> execute.

A Bayesian design must have its prior, decision thresholds, and planned sample size fixed BEFORE
anyone sees the data. An interactive tool that lets you pick a prior after looking at an interim
result is a prior-shopping machine. This module makes the distinction visible and enforceable.

At design stage we canonicalize the decision-relevant parameters, hash them, and emit a portable
lock artifact the user downloads. At interim we re-hash the run in front of us and compare, stamping
every output PRE-SPECIFIED, DRIFTED (naming each changed field), or EXPLORATORY.

HONEST LIMITATION: a content hash proves INTEGRITY (the lock was not edited after the fact) but NOT
ANTERIORITY (that it existed before the data was seen). Nothing stops a user fabricating a lock after
peeking. Real anteriority needs an external timestamp authority -- a filed protocol, a public registry
entry, a signed and dated SAP. That is what the optional `anchor` field is for. We claim nothing more.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json

# Only these fields are hashed. Everything else about a run (the question wording, the SQL, the
# hypothesis prose) may vary freely without breaking a lock -- what must not vary is the DECISION.
LOCKED_FIELDS: tuple[str, ...] = (
    "endpoint_type", "framing", "n_planned",
    "tv", "lrv", "gate_tv", "gate_lrv", "stop_lrv", "higher_is_better",
    "prior_a", "prior_b", "prior_mu", "prior_sd",
)

_PRECISION = 6          # floats are rounded before hashing so 0.1 and 0.10 cannot drift apart

_LIMITATION = (
    "This lock proves INTEGRITY (its contents were not altered after creation) but NOT ANTERIORITY "
    "(that it existed before the data were seen). Trustworthy pre-specification requires an external "
    "timestamp: a filed protocol, a public registry entry, or a signed and dated SAP. Record that "
    "reference in the `anchor` field."
)


def _norm(v):
    """Canonical form of one value: floats rounded, ints/bools/strings left alone."""
    if isinstance(v, bool):          # bool before int -- bool IS an int in Python
        return v
    if isinstance(v, float):
        return round(v, _PRECISION)
    return v


def canonical(params: dict) -> str:
    """Sorted, float-rounded JSON of the decision-relevant fields only."""
    d = {k: _norm(params[k]) for k in LOCKED_FIELDS if params.get(k) is not None}
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def lock_id(params: dict) -> str:
    return hashlib.sha256(canonical(params).encode()).hexdigest()


def create_lock(params: dict, oc: list[dict], anchor: str | None = None) -> dict:
    """The portable artifact the user downloads at design stage and supplies back at interim."""
    locked = {k: _norm(params[k]) for k in LOCKED_FIELDS if params.get(k) is not None}
    return {
        "lock_id": lock_id(params),
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "params": locked,
        "operating_characteristics": oc,
        "anchor": anchor,
        "limitation": _LIMITATION,
    }


def verify(lock: dict | None, params: dict) -> dict:
    """Compare a run's parameters against a lock. Never raises."""
    if not lock:
        return {"status": "EXPLORATORY", "lock_id": None, "drift": [], "anchor": None}
    try:
        recorded = lock["lock_id"]
        # Re-hash the lock's OWN recorded params. If they don't reproduce its stored id, it was edited.
        if lock_id(lock["params"]) != recorded:
            return {"status": "INVALID", "lock_id": recorded, "drift": [], "anchor": lock.get("anchor")}
    except (KeyError, TypeError):
        return {"status": "INVALID", "lock_id": None, "drift": [], "anchor": None}

    drift = []
    for k in LOCKED_FIELDS:
        locked_v = lock["params"].get(k)
        actual_v = _norm(params[k]) if params.get(k) is not None else None
        if locked_v != actual_v:
            drift.append({"field": k, "locked": locked_v, "actual": actual_v})
    status = "PRE-SPECIFIED" if not drift else "DRIFTED"
    return {"status": status, "lock_id": recorded, "drift": drift, "anchor": lock.get("anchor")}


def caveat(prespec: dict) -> str:
    """The deterministic caveat line. Always first in ModelResult.issues: it conditions how every
    other number should be read."""
    status = prespec.get("status")
    if status == "PRE-SPECIFIED":
        return ("PRE-SPECIFIED: this run matches its locked design (prior, thresholds, planned n). "
                "The verdict was not chosen after seeing the data.")
    if status == "DRIFTED":
        changes = "; ".join(f"{d['field']}: locked {d['locked']} -> used {d['actual']}"
                            for d in prespec.get("drift", []))
        return (f"NOT PRE-SPECIFIED (DRIFTED): this run departs from its locked design -- {changes}. "
                "A decision rule changed after the design was locked, so the verdict must be read as "
                "exploratory, not confirmatory.")
    if status == "INVALID":
        return ("NOT PRE-SPECIFIED (INVALID LOCK): the supplied lock's contents do not match its own "
                "hash, so it was edited after creation. Treating this run as exploratory.")
    return ("EXPLORATORY, not pre-specified: no design lock was supplied, so the prior and decision "
            "thresholds could have been chosen after seeing the data. Fine for exploration; not a "
            "basis for a confirmatory claim.")
