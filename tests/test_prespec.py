"""Unit tests for the pre-specification lock (pure; no key, no network)."""
from agent import prespec

PARAMS = {
    "endpoint_type": "proportion", "framing": "single_arm", "n_planned": 100,
    "tv": 0.30, "lrv": 0.15, "gate_tv": 0.80, "gate_lrv": 0.90, "stop_lrv": 0.10,
    "higher_is_better": True, "prior_a": 9.0, "prior_b": 13.0,
}
OC = [{"theta": 0.15, "go_rate": 0.05}, {"theta": 0.30, "go_rate": 0.82}]


def test_lock_id_is_stable_across_key_order():
    reordered = dict(reversed(list(PARAMS.items())))
    assert prespec.lock_id(PARAMS) == prespec.lock_id(reordered)


def test_lock_id_ignores_float_formatting():
    # 0.1 and 0.10 are the same number; they must not produce a spurious DRIFT
    a = dict(PARAMS, stop_lrv=0.1)
    b = dict(PARAMS, stop_lrv=0.10)
    assert prespec.lock_id(a) == prespec.lock_id(b)


def test_lock_id_changes_when_a_decision_field_changes():
    assert prespec.lock_id(PARAMS) != prespec.lock_id(dict(PARAMS, tv=0.35))


def test_lock_id_ignores_non_decision_fields():
    # a field outside LOCKED_FIELDS must not affect the hash
    assert prespec.lock_id(PARAMS) == prespec.lock_id(dict(PARAMS, hypothesis="anything at all"))


def test_verify_matching_params_is_pre_specified():
    lock = prespec.create_lock(PARAMS, OC)
    out = prespec.verify(lock, PARAMS)
    assert out["status"] == "PRE-SPECIFIED"
    assert out["drift"] == []
    assert out["lock_id"] == lock["lock_id"]


def test_verify_changed_lrv_is_drifted_and_names_the_field():
    lock = prespec.create_lock(PARAMS, OC)
    out = prespec.verify(lock, dict(PARAMS, lrv=0.10))
    assert out["status"] == "DRIFTED"
    drift = {d["field"]: d for d in out["drift"]}
    assert "lrv" in drift
    assert drift["lrv"]["locked"] == 0.15 and drift["lrv"]["actual"] == 0.10


def test_verify_without_a_lock_is_exploratory():
    out = prespec.verify(None, PARAMS)
    assert out["status"] == "EXPLORATORY" and out["lock_id"] is None


def test_verify_rejects_a_tampered_lock():
    lock = prespec.create_lock(PARAMS, OC)
    lock["params"]["tv"] = 0.99            # edited after creation; the recorded hash no longer matches
    out = prespec.verify(lock, PARAMS)
    assert out["status"] == "INVALID"


def test_verify_degrades_gracefully_when_params_is_none():
    out = prespec.verify({"lock_id": "x", "params": None}, PARAMS)
    assert out["status"] == "INVALID"


def test_verify_degrades_gracefully_when_params_is_a_string():
    out = prespec.verify({"lock_id": "x", "params": "garbage"}, PARAMS)
    assert out["status"] == "INVALID"


def test_verify_degrades_gracefully_when_params_is_a_list():
    out = prespec.verify({"lock_id": "x", "params": [1, 2, 3]}, PARAMS)
    assert out["status"] == "INVALID"


def test_verify_degrades_gracefully_when_params_is_an_int():
    out = prespec.verify({"lock_id": "x", "params": 12345}, PARAMS)
    assert out["status"] == "INVALID"


def test_verify_preserves_lock_id_when_params_is_missing_entirely():
    out = prespec.verify({"lock_id": "x"}, PARAMS)
    assert out["status"] == "INVALID"
    assert out["lock_id"] == "x"


def test_create_lock_records_anchor_and_the_honest_limitation():
    lock = prespec.create_lock(PARAMS, OC, anchor="NCT01234567")
    assert lock["anchor"] == "NCT01234567"
    assert lock["operating_characteristics"] == OC
    assert "timestamp" in lock
    # the artifact must state plainly what a content hash does NOT prove
    assert "anteriority" in lock["limitation"].lower()


def test_caveat_text_per_status():
    assert "not pre-specified" in prespec.caveat({"status": "EXPLORATORY", "drift": []}).lower()
    assert "pre-specified" in prespec.caveat({"status": "PRE-SPECIFIED", "drift": []}).lower()
    drifted = {"status": "DRIFTED", "drift": [{"field": "lrv", "locked": 0.15, "actual": 0.10}]}
    assert "lrv" in prespec.caveat(drifted)


# --- verify() hard contract: it must NEVER raise, on ANY input ------------------------------------

GOOD_LOCK = prespec.create_lock(PARAMS, OC)


def _circular_lock():
    """A structurally valid lock whose params contain a self-reference. json.dumps() (inside
    lock_id()'s re-hash) rejects this with ValueError, not KeyError/TypeError."""
    circ: dict = {}
    circ["self"] = circ
    return {"lock_id": "x", "params": {"tv": circ}}


def _well_formed_invalid(out):
    return (
        isinstance(out, dict)
        and out.get("status") == "INVALID"
        and "lock_id" in out
        and isinstance(out.get("drift"), list)
        and "anchor" in out
    )


# label, lock, params -- every case here must come back as a well-formed INVALID result. None of
# them may raise, regardless of which argument (or which field inside it) is the garbage one.
ADVERSARIAL_CASES = [
    ("valid lock, params=None", GOOD_LOCK, None),
    ("valid lock, params is a string", GOOD_LOCK, "garbage"),
    ("valid lock, params is a list", GOOD_LOCK, [1, 2, 3]),
    ("valid lock, params is an int", GOOD_LOCK, 42),
    ("lock's own params is None, run params is also garbage", {"lock_id": "x", "params": None}, "garbage"),
    ("circular reference inside lock params", _circular_lock(), PARAMS),
    ("lock itself is a string, not a dict", "not a dict", PARAMS),
    ("lock itself is a list, not a dict", [1, 2, 3], PARAMS),
    ("lock_id is the wrong type", {"lock_id": [1, 2], "params": {"tv": 1}}, PARAMS),
]


def test_verify_never_raises_on_adversarial_inputs():
    for label, lock, params in ADVERSARIAL_CASES:
        try:
            out = prespec.verify(lock, params)
        except Exception as e:  # noqa: BLE001 -- the entire point of this test is that nothing escapes
            raise AssertionError(f"verify() raised on case {label!r}: {e!r}") from e
        assert _well_formed_invalid(out), f"case {label!r} produced a malformed result: {out!r}"


def test_verify_with_no_lock_at_all_stays_exploratory_even_if_params_is_garbage():
    # lock=None means "no lock was supplied" -- a legitimate, distinct case from a garbage lock. It
    # must stay EXPLORATORY, not get reclassified as INVALID, no matter what params holds.
    for garbage in (None, "garbage", [1, 2, 3], 42):
        out = prespec.verify(None, garbage)
        assert out["status"] == "EXPLORATORY"
        assert out["lock_id"] is None


def test_verify_still_rejects_a_tampered_lock_after_the_hardening():
    # The structural fix (a catch-all safety net) must not swallow genuine tamper detection into a
    # generic result -- a lock whose params were edited after creation must still be caught by name.
    lock = prespec.create_lock(PARAMS, OC)
    lock["params"]["tv"] = 0.99            # edited after creation; the recorded hash no longer matches
    out = prespec.verify(lock, PARAMS)
    assert out["status"] == "INVALID"
    assert out["lock_id"] == lock["lock_id"]
