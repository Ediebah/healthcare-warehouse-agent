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
