"""Hermetic tests for the data-quality agent (detect → diagnose → propose-fix).

No API key, no network: the warehouse reads go against throwaway temp DuckDBs, and every LLM path is
either monkeypatched with a canned response or exercised with the key removed (graceful-degrade path).
"""
import shutil
from pathlib import Path

import duckdb
import pytest

from agent import quality_agent as qa
from agent import warehouse as W


# ─────────────────────────────── fixtures ───────────────────────────────
@pytest.fixture
def defect_db():
    """The agent's own planted-defect demo DB (dup PK, orphan FK, null + out-of-domain gender)."""
    path = qa.make_demo_db()
    yield path
    shutil.rmtree(Path(path).parent, ignore_errors=True)


@pytest.fixture
def healthy_db(tmp_path):
    """A clean warehouse that satisfies every check — no dups, orphans, nulls, bad values; metric in band."""
    path = tmp_path / "healthy.duckdb"
    con = duckdb.connect(str(path))
    try:
        con.execute("create table dim_patient (patient_id varchar, gender varchar, age integer)")
        con.execute("insert into dim_patient select 'P' || lpad(cast(i as varchar), 3, '0'), "
                    "case when i % 2 = 0 then 'F' else 'M' end, 20 + i * 5 from range(1, 11) t(i)")
        con.execute("create table fct_encounters (encounter_id varchar, patient_id varchar)")
        con.execute("insert into fct_encounters select 'E' || lpad(cast(i as varchar), 4, '0'), "
                    "'P' || lpad(cast((i % 10) + 1 as varchar), 3, '0') from range(0, 30) t(i)")
        con.execute("create table fct_medications (medication_order_id varchar, days_supplied integer)")
        con.execute("insert into fct_medications select 'M' || lpad(cast(i as varchar), 4, '0'), 30 "
                    "from range(0, 20) t(i)")
        con.execute("create table mart_readmissions "
                    "(index_encounter_id varchar, is_30d_readmission boolean, days_to_next_admission integer)")
        con.execute("insert into mart_readmissions select 'E' || lpad(cast(i as varchar), 4, '0'), (i < 9), "
                    "case when i < 9 then 15 else null end from range(0, 100) t(i)")
    finally:
        con.close()
    return str(path)


def _by_name(db):
    return {r.check.name: r for r in qa.detect(db)}


# ─────────────────────────────── DETECT ───────────────────────────────
def test_detect_flags_every_planted_defect(defect_db):
    r = _by_name(defect_db)
    assert not r["Primary-key uniqueness"].passed
    assert not r["Referential integrity"].passed
    assert not r["Completeness"].passed
    assert not r["Accepted values"].passed
    assert not r["Age in human range"].passed            # planted negative age
    assert not r["Non-negative medication supply"].passed   # planted negative days_supplied
    assert not r["Non-negative readmission gap"].passed  # planted negative days_to_next_admission
    assert r["Metric in band"].passed               # readmission rate is left in-band → this one passes


def test_detect_surfaces_the_offending_rows(defect_db):
    r = _by_name(defect_db)
    # uniqueness evidence names the duplicated key and its multiplicity
    dup = r["Primary-key uniqueness"].evidence
    assert "P001" in set(dup["patient_id"]) and int(dup[dup["patient_id"] == "P001"]["n_rows"].iloc[0]) == 2
    # referential evidence names the orphan encounter → ghost patient
    orph = r["Referential integrity"].evidence
    assert "GHOST" in set(orph["patient_id"]) and "E9999" in set(orph["encounter_id"])


def test_healthy_data_passes_all_checks(healthy_db):
    results = qa.detect(healthy_db)
    assert results and all(res.passed for res in results)
    report = qa.run(healthy_db)
    assert report["ok"]
    assert report["n_failed"] == 0 and report["n_errored"] == 0
    assert report["n_passed"] == report["n_checks"] == len(qa.CHECKS)


# ─────────────────────────────── DIAGNOSE ───────────────────────────────
def test_diagnose_names_the_specific_violation(defect_db):
    r = _by_name(defect_db)
    dx_pk = qa.diagnose(r["Primary-key uniqueness"])
    assert "patient_id" in dx_pk and "P001" in dx_pk and "not unique" in dx_pk.lower()

    dx_ref = qa.diagnose(r["Referential integrity"])
    assert "orphan" in dx_ref.lower() and "GHOST" in dx_ref

    dx_comp = qa.diagnose(r["Completeness"])
    assert "gender" in dx_comp and "13.3%" in dx_comp        # 2 of 15 rows null

    dx_av = qa.diagnose(r["Accepted values"])
    assert "'U'" in dx_av and "{M, F}" in dx_av

    dx_range = qa.diagnose(r["Age in human range"])
    assert "age" in dx_range.lower() and "-3" in dx_range and "range" in dx_range.lower()


# ─────────────────────────────── PROPOSE FIX (monkeypatched LLM) ───────────────────────────────
def test_propose_fix_returns_a_validated_readonly_sql(defect_db, monkeypatch):
    canned = {
        "fix_sql": ("WITH deduped AS (SELECT patient_id, MIN(gender) AS gender "
                    "FROM dim_patient GROUP BY patient_id) SELECT * FROM deduped"),
        "explanation": "Collapse duplicate patient_id rows to one row per key with a dedup CTE.",
        "confidence": "high",
    }
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")            # satisfy the key gate (no real call is made)
    monkeypatch.setattr(qa.llm, "complete_json", lambda *a, **k: canned)

    r = _by_name(defect_db)["Primary-key uniqueness"]
    fix = qa.propose_fix(r, qa.diagnose(r))

    assert fix["fix"] is not None and fix["note"] is None
    assert fix["confidence"] == "high" and fix["explanation"]
    # the returned fix genuinely passed the hardened read-only validator (idempotent on clean SQL)
    assert W.validate(fix["fix"]) == fix["fix"]


def test_propose_fix_rejects_unsafe_sql_and_retries_once(defect_db, monkeypatch):
    calls = {"n": 0}

    def unsafe(*_a, **_k):
        calls["n"] += 1
        return {"fix_sql": "DELETE FROM dim_patient WHERE 1=1", "explanation": "x", "confidence": "low"}

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(qa.llm, "complete_json", unsafe)

    r = _by_name(defect_db)["Primary-key uniqueness"]
    fix = qa.propose_fix(r, qa.diagnose(r))

    assert fix["fix"] is None                                   # a write was never surfaced as a "fix"
    assert calls["n"] == 2                                      # initial attempt + exactly one retry
    assert "validation" in fix["note"].lower()


def test_propose_fix_degrades_gracefully_without_key(defect_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def must_not_call(*_a, **_k):
        raise AssertionError("the LLM must not be called when no key is configured")

    monkeypatch.setattr(qa.llm, "complete_json", must_not_call)

    r = _by_name(defect_db)["Referential integrity"]
    dx = qa.diagnose(r)
    fix = qa.propose_fix(r, dx)

    assert fix["fix"] is None
    assert "LLM unavailable" in fix["note"]
    assert fix["diagnosis"] == dx                               # diagnosis is still returned for humans


# ─────────────────────────────── run() + safety ───────────────────────────────
def test_run_report_diagnoses_and_proposes_for_each_failure(defect_db, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(qa.llm, "complete_json", lambda *a, **k: {
        "fix_sql": "SELECT DISTINCT patient_id FROM dim_patient", "explanation": "dedup", "confidence": "medium"})

    report = qa.run(defect_db)
    assert not report["ok"]
    assert report["n_failed"] >= 4 and report["n_passed"] >= 1
    for c in report["checks"]:
        if not c["passed"]:
            assert c["diagnosis"] and "proposal" in c
            assert c["proposal"]["fix"] is not None


def test_make_demo_db_is_a_throwaway_never_the_real_warehouse(defect_db):
    assert Path(defect_db).exists()


# ─────────────────────────────── pre-flight gate ───────────────────────────────
def test_preflight_passes_a_healthy_warehouse(healthy_db):
    h = qa.preflight(healthy_db, force=True)
    assert h["healthy"] and not h["blocking"] and h["n_failed"] == 0


def test_preflight_blocks_on_critical_defects(defect_db):
    h = qa.preflight(defect_db, force=True)
    assert not h["healthy"] and h["blocking"]
    crit = {f["name"] for f in h["failures"] if f["severity"] == "critical"}
    assert "Primary-key uniqueness" in crit and "Referential integrity" in crit


def test_run_analysis_is_gated_by_a_broken_warehouse(defect_db, monkeypatch):
    # a critical-integrity failure must block the analysis BEFORE any LLM call (no corrupt metrics out)
    from agent import agent
    monkeypatch.setattr(agent.llm, "complete", lambda *a, **k: pytest.fail("LLM called past the gate"))
    monkeypatch.setattr(agent.llm, "complete_json", lambda *a, **k: pytest.fail("LLM called past the gate"))
    r = agent.run_analysis("What is the 30-day readmission rate?", db_path=defect_db)
    assert r.data_health and r.data_health["blocking"]
    assert "Data-health gate" in (r.clarification or "")
    assert "dq_demo_" in defect_db                              # isolated temp dir
    assert Path(defect_db).resolve() != W.DB_PATH.resolve()    # never the project warehouse
