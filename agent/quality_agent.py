"""Data-quality agent — closes the detect → diagnose → propose-fix loop over the warehouse.

The self-heal story in `pipeline_healer.py` fixes a *query*; this agent watches the *data*. It runs a
battery of declarative CHECK specs (primary-key uniqueness, referential integrity, completeness,
metric-in-band, accepted-values) read-only against the DuckDB warehouse, then for every failure:

    1. DETECT   run the check's SQL via `warehouse.run_query` (hardened read-only) → offending rows/evidence
    2. DIAGNOSE turn the evidence into a concrete, human-readable finding (which ids duplicate, how many
                orphans, which column is how-null, the out-of-band value) — deterministic, no LLM
    3. PROPOSE  ask the LLM for a corrected/guarded remediation SQL (dedup CTE / orphan filter / coalesce /
                whitelist), VALIDATE it with `warehouse.validate` (must be read-only safe), and return it as
                a PROPOSAL for human review [LLM]

SAFETY (sandboxed ethos, human-in-the-loop):
  * Every warehouse read goes through `warehouse.run_query` (single SELECT/WITH, external access off).
  * The agent NEVER issues DDL/DML against the real warehouse — it only reads.
  * A generated fix is STATICALLY VALIDATED (read-only safe) and RETURNED AS A PROPOSAL — never auto-applied.
  * The `--demo` path writes planted defects to a THROWAWAY temp DuckDB only, never the real warehouse.
  * No OPENAI_API_KEY → the deterministic detect+diagnose still run; propose_fix degrades to fix=None.

Run:
    python -m agent.quality_agent            # audit the real warehouse (usually all-green)
    python -m agent.quality_agent --demo     # planted-defect temp DB → the full loop is visible end-to-end
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from . import llm, warehouse


# ─────────────────────────────── check specifications ───────────────────────────────
@dataclass(frozen=True)
class Check:
    """A declarative data-quality rule. `sql` is a read-only query that surfaces the OFFENDING evidence
    (violating rows for row-based kinds, or a one-row summary for completeness/metric_band)."""
    name: str
    kind: str                       # unique | referential | completeness | metric_band | accepted_values | range
    severity: str                   # critical | high | medium
    description: str
    sql: str
    table: str = ""
    column: str = ""
    grain: str = ""
    threshold: float | None = None            # completeness: max tolerated null rate
    band: tuple[float, float] | None = None   # metric_band: (low, high) inclusive
    accepted: tuple[str, ...] | None = None    # accepted_values: the allowed set


# The battery. Each SQL is a plain SELECT/WITH — it passes `warehouse.validate` and runs read-only.
CHECKS: list[Check] = [
    Check(
        name="Primary-key uniqueness",
        kind="unique",
        severity="critical",
        description="dim_patient.patient_id must be unique — one row per patient.",
        table="dim_patient", column="patient_id", grain="one row per patient_id",
        sql=(
            "select patient_id, count(*) as n_rows "
            "from dim_patient "
            "group by patient_id "
            "having count(*) > 1 "
            "order by n_rows desc, patient_id"
        ),
    ),
    Check(
        name="Referential integrity",
        kind="referential",
        severity="critical",
        description="Every fct_encounters.patient_id must resolve to a dim_patient row — no orphan encounters.",
        table="fct_encounters", column="patient_id", grain="fct_encounters → dim_patient",
        sql=(
            "select e.encounter_id, e.patient_id "
            "from fct_encounters e "
            "left join dim_patient p on e.patient_id = p.patient_id "
            "where p.patient_id is null "
            "order by e.encounter_id"
        ),
    ),
    Check(
        name="Completeness",
        kind="completeness",
        severity="high",
        description="dim_patient.gender null rate must stay under 5%.",
        table="dim_patient", column="gender", threshold=0.05,
        sql=(
            "select "
            "sum((gender is null)::int) as null_count, "
            "count(*) as total_rows, "
            "avg((gender is null)::int) as null_rate "
            "from dim_patient"
        ),
    ),
    Check(
        name="Metric in band",
        kind="metric_band",
        severity="medium",
        description="30-day readmission rate should sit in the clinically plausible 3–25% band.",
        table="mart_readmissions", column="is_30d_readmission", band=(0.03, 0.25),
        sql="select avg(cast(is_30d_readmission as int)) as readmission_rate from mart_readmissions",
    ),
    Check(
        name="Accepted values",
        kind="accepted_values",
        severity="high",
        description="dim_patient.gender must be one of {M, F}.",
        table="dim_patient", column="gender", accepted=("M", "F"),
        sql=(
            "select gender as value, count(*) as n_rows "
            "from dim_patient "
            "where gender is not null and gender not in ('M', 'F') "
            "group by gender "
            "order by n_rows desc, value"
        ),
    ),
    Check(
        name="Age in human range",
        kind="range",
        severity="high",
        description="dim_patient.age must sit within [0, 120]. A fixed as_of_date once measured newborns "
                    "generated after it as NEGATIVE ages.",
        table="dim_patient", column="age", band=(0, 120),
        sql=(
            "select patient_id, age as value "
            "from dim_patient "
            "where age < 0 or age > 120 "
            "order by value limit 50"
        ),
    ),
    Check(
        name="Non-negative medication supply",
        kind="range",
        severity="medium",
        description="fct_medications.days_supplied must sit within [0, 36525]. A negative span means the "
                    "source dispense_stop precedes dispense_start.",
        table="fct_medications", column="days_supplied", band=(0, 36525),
        sql=(
            "select medication_order_id, days_supplied as value "
            "from fct_medications "
            "where days_supplied < 0 or days_supplied > 36525 "
            "order by value limit 50"
        ),
    ),
    Check(
        name="Non-negative readmission gap",
        kind="range",
        severity="medium",
        description="mart_readmissions.days_to_next_admission must sit within [0, 36525]. A negative gap "
                    "means overlapping encounters were treated as a sequential readmission.",
        table="mart_readmissions", column="days_to_next_admission", band=(0, 36525),
        sql=(
            "select index_encounter_id, days_to_next_admission as value "
            "from mart_readmissions "
            "where days_to_next_admission < 0 or days_to_next_admission > 36525 "
            "order by value limit 50"
        ),
    ),
]


@dataclass
class CheckResult:
    check: Check
    passed: bool
    evidence: pd.DataFrame          # offending rows (row-based) or a one-row summary (completeness/metric)
    summary: str                    # short human line for the report
    error: str | None = None        # set if the check query itself failed to run


def _accepted_set(check: Check) -> str:
    return "{" + ", ".join(check.accepted or ()) + "}"


def _evaluate(check: Check, df: pd.DataFrame) -> tuple[bool, str]:
    """Decide pass/fail + a one-line summary from a check's result frame."""
    kind = check.kind
    if kind == "unique":
        n = len(df)
        return n == 0, "no duplicate keys" if n == 0 else f"{n} duplicated {check.column} value(s)"
    if kind == "referential":
        n = len(df)
        return n == 0, "no orphan rows" if n == 0 else f"{n} orphan row(s) in {check.table}"
    if kind == "accepted_values":
        n = len(df)
        rows = int(df["n_rows"].sum()) if n else 0
        return n == 0, ("all values within the accepted set" if n == 0
                        else f"{rows} row(s) across {n} value(s) outside {_accepted_set(check)}")
    if kind == "completeness":
        rate = float(df.iloc[0]["null_rate"] or 0.0)
        thr = check.threshold or 0.0
        return rate <= thr, f"{check.column} null rate {rate:.1%} (threshold {thr:.0%})"
    if kind == "metric_band":
        raw = df.iloc[0, 0]
        val = float(raw) if raw is not None else float("nan")
        lo, hi = check.band or (0.0, 1.0)
        ok = (not math.isnan(val)) and lo <= val <= hi
        return ok, f"value {val:.1%} (band {lo:.0%}–{hi:.0%})"
    if kind == "range":
        n = len(df)
        lo, hi = check.band or (0.0, float("inf"))
        return n == 0, (f"all {check.column} within [{lo:g}, {hi:g}]" if n == 0
                        else f"{n} row(s) with {check.column} outside [{lo:g}, {hi:g}]")
    return True, "no rule"


# ─────────────────────────────── 1. DETECT ───────────────────────────────
def detect(db_path: str | Path | None = None) -> list[CheckResult]:
    """Run every CHECK read-only against the warehouse (or `db_path`) and return a result per check.

    Each result carries `passed` + the offending `evidence` (violating rows, or a one-row summary for
    completeness/metric_band). A check whose query fails to run is returned with `error` set and
    passed=False rather than crashing the whole sweep."""
    results: list[CheckResult] = []
    for check in CHECKS:
        try:
            df = warehouse.run_query(check.sql, db_path=db_path)
        except warehouse.QueryError as e:
            results.append(CheckResult(check, False, pd.DataFrame(),
                                       f"check could not run: {e}", error=str(e)))
            continue
        passed, summary = _evaluate(check, df)
        results.append(CheckResult(check, passed, df, summary))
    return results


# ─────────────────────────────── 2. DIAGNOSE ───────────────────────────────
def diagnose(result: CheckResult) -> str:
    """A concrete, human-readable diagnosis of a FAILING check — naming the specific offending evidence."""
    check, df = result.check, result.evidence
    if result.error:
        return f"{check.name}: the check query failed to run — {result.error}"
    if result.passed:
        return f"{check.name}: no violations found ({result.summary})."

    if check.kind == "unique":
        n_keys = len(df)
        surplus = int(df["n_rows"].sum() - n_keys)
        ex = "; ".join(f"{check.column}={row[check.column]!r} appears {int(row['n_rows'])}×"
                       for _, row in df.head(3).iterrows())
        return (f"{check.table}.{check.column} is NOT unique: {n_keys} value(s) are duplicated, "
                f"{surplus} surplus row(s) beyond the one-row-per-{check.column} grain. e.g. {ex}. "
                f"The primary-key grain is violated — joins on {check.column} will fan out.")
    if check.kind == "referential":
        n = len(df)
        ex = "; ".join(f"encounter {row['encounter_id']!r} → patient {row['patient_id']!r}"
                       for _, row in df.head(3).iterrows())
        return (f"{check.table} has {n} orphan row(s): {check.column} has no matching dim_patient row. "
                f"e.g. {ex}. These events reference patients absent from the dimension "
                f"(late/missing dimension load or a bad key).")
    if check.kind == "completeness":
        row = df.iloc[0]
        nc, total, rate = int(row["null_count"]), int(row["total_rows"]), float(row["null_rate"])
        return (f"{check.table}.{check.column} is {rate:.1%} null ({nc:,} of {total:,} rows) — "
                f"above the {check.threshold:.0%} completeness threshold. Downstream group-bys / "
                f"filters on {check.column} will silently drop or miscount these rows.")
    if check.kind == "metric_band":
        val = float(df.iloc[0, 0])
        lo, hi = check.band or (0.0, 1.0)
        side = "below" if val < lo else "above"
        return (f"30-day readmission rate is {val:.1%}, {side} the plausible {lo:.0%}–{hi:.0%} band. "
                f"A swing this large usually signals a modeling / date-window bug (e.g. self-joins or a "
                f"broken 30-day window) rather than a real clinical shift — investigate before reporting.")
    if check.kind == "accepted_values":
        n_vals = len(df)
        n_rows = int(df["n_rows"].sum())
        ex = ", ".join(f"{row['value']!r} ({int(row['n_rows'])} row(s))" for _, row in df.head(5).iterrows())
        return (f"{check.table}.{check.column} holds {n_rows} row(s) with {n_vals} value(s) outside the "
                f"accepted set {_accepted_set(check)}: {ex}. Likely un-normalized source codes that must "
                f"be mapped or filtered upstream.")
    if check.kind == "range":
        n = len(df)
        lo, hi = check.band or (0.0, float("inf"))
        idcol = df.columns[0]
        ex = "; ".join(f"{idcol}={row[idcol]!r} → {check.column}={row['value']:g}"
                       for _, row in df.head(3).iterrows())
        return (f"{check.table}.{check.column} has {n} row(s) outside the plausible range "
                f"[{lo:g}, {hi:g}]: e.g. {ex}. An impossible value (a negative age/span, a future date) "
                f"signals a source or derivation bug — guard it in the model, don't publish it.")
    return f"{check.name}: {result.summary}"


# ─────────────────────────────── 3. PROPOSE FIX (LLM) ───────────────────────────────
_FIX_SYSTEM = (
    "You are a data-reliability / analytics engineer remediating a data-quality defect in a dbt-modeled "
    "DuckDB warehouse. Propose ONE corrected, GUARDED query that returns a clean version of the affected "
    "table — pick the right pattern for the defect: a dedup CTE using row_number() over a partition, a "
    "join/filter that drops orphan rows, a coalesce() that fills a nullable column, a whitelist filter "
    "for accepted values, or a range guard (a CASE/WHERE that nulls or excludes out-of-bounds numerics).\n"
    "HARD CONSTRAINT: the SQL MUST be a SINGLE READ-ONLY statement beginning with SELECT or WITH. No "
    "INSERT/UPDATE/DELETE/CREATE/ALTER/DROP/COPY/ATTACH/PRAGMA, no semicolons, no file-reading functions. "
    "It is a PROPOSAL for a human to fold into the staging model — it is NEVER executed against the warehouse."
)


def _llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _fix_user_prompt(check: Check, diagnosis: str, evidence: pd.DataFrame) -> str:
    sample = evidence.head(5).to_csv(index=False).strip() or "(one-row summary — see diagnosis)"
    return (
        f"FAILING CHECK: {check.name}  (kind={check.kind}, severity={check.severity})\n"
        f"TABLE: {check.table}   COLUMN: {check.column}   GRAIN: {check.grain or 'n/a'}\n"
        f"DIAGNOSIS: {diagnosis}\n"
        f"OFFENDING EVIDENCE (sample):\n{sample}\n\n"
        "Propose a corrected read-only SELECT/WITH over the affected table that resolves the violation "
        "(so re-running the check would pass), plus a short plain-English explanation and your confidence.\n"
        'Return JSON: {"fix_sql": "SELECT ...", "explanation": "one or two sentences", '
        '"confidence": "high|medium|low"}.'
    )


def propose_fix(result: CheckResult, diagnosis: str) -> dict:
    """Generate a PROPOSED remediation for a failing check and validate it is read-only safe.

    Returns a dict: {check, kind, severity, diagnosis, fix, explanation, confidence, note}. `fix` is the
    validated, read-only SQL string (a PROPOSAL for human review — never auto-applied) or None. Degrades
    gracefully with fix=None + a note when the LLM is unavailable, or when the generated SQL fails the
    read-only safety validator twice."""
    check = result.check
    out = {"check": check.name, "kind": check.kind, "severity": check.severity,
           "diagnosis": diagnosis, "fix": None, "explanation": None, "confidence": None, "note": None}

    if not _llm_available():
        out["note"] = "LLM unavailable — set OPENAI_API_KEY to generate a proposed fix"
        return out

    user = _fix_user_prompt(check, diagnosis, result.evidence)
    hint, last_err = "", ""
    for _ in range(2):                                   # initial attempt + one validation-driven retry
        try:
            payload = llm.complete_json(_FIX_SYSTEM, user + hint)
        except llm.LLMError as e:                        # no key / bad JSON / API failure → graceful degrade
            out["note"] = f"LLM unavailable — {e}"
            return out
        fix_sql = str(payload.get("fix_sql") or payload.get("sql") or "").strip()
        try:
            safe = warehouse.validate(fix_sql)           # STATIC read-only safety gate before we surface it
        except warehouse.QueryError as e:
            last_err = str(e)
            hint = (f"\n\nYour previous SQL was REJECTED by the read-only safety validator: {e}. "
                    "Return ONE read-only SELECT or WITH statement only — no writes, DDL, semicolons, "
                    "or file functions.")
            continue
        out["fix"] = safe
        out["explanation"] = str(payload.get("explanation") or "").strip() or None
        out["confidence"] = str(payload.get("confidence") or "low").strip().lower()
        return out

    out["note"] = f"generated fix failed read-only validation after 2 attempts ({last_err})"
    return out


# ─────────────────────────────── orchestration ───────────────────────────────
def run(db_path: str | Path | None = None) -> dict:
    """Run the full loop and return a report dict. For each failing check it attaches a `diagnosis`
    and a `proposal` (from propose_fix); passing checks are recorded as-is."""
    results = detect(db_path)
    report: dict = {"ok": True, "n_checks": len(results), "n_passed": 0, "n_failed": 0,
                    "n_errored": 0, "llm_available": _llm_available(), "checks": []}
    for r in results:
        entry: dict = {"name": r.check.name, "kind": r.check.kind, "severity": r.check.severity,
                       "description": r.check.description, "passed": r.passed, "summary": r.summary}
        if r.error:
            report["ok"] = False
            report["n_errored"] += 1
            entry["error"] = r.error
        elif r.passed:
            report["n_passed"] += 1
        else:
            report["ok"] = False
            report["n_failed"] += 1
            dx = diagnose(r)
            entry["diagnosis"] = dx
            entry["proposal"] = propose_fix(r, dx)
        report["checks"].append(entry)
    return report


# ─────────────────────────────── self-contained demo ───────────────────────────────
def make_demo_db() -> str:
    """Create a THROWAWAY temp DuckDB seeded with deliberately-planted data-quality defects, so the full
    detect → diagnose → propose-fix loop is demonstrable even when the real warehouse is healthy.

    Writes ONLY to a fresh temp path — never the real warehouse. Returns the db path as a string.
    Planted: a duplicate primary key, an orphan FK encounter, null genders (completeness), an
    out-of-domain gender code (accepted-values), and out-of-range numerics — a negative age, a
    negative medication supply, and a negative readmission gap (range). The readmission metric is
    left IN band so the report also shows a healthy check. Caller owns cleanup (remove the parent dir)."""
    db_path = Path(tempfile.mkdtemp(prefix="dq_demo_")) / "healthcare_defects.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        # dim_patient: 10 clean patients P001..P010, gender alternating M/F, plausible ages 25..70
        con.execute("create table dim_patient (patient_id varchar, gender varchar, age integer)")
        con.execute(
            "insert into dim_patient "
            "select 'P' || lpad(cast(i as varchar), 3, '0'), "
            "case when i % 2 = 0 then 'F' else 'M' end, "
            "20 + i * 5 "
            "from range(1, 11) t(i)"
        )
        con.execute("insert into dim_patient values ('P001', 'M', 40)")                  # PLANT: duplicate PK
        con.execute("insert into dim_patient values ('P011', null, 33), ('P012', null, 44)")  # PLANT: null gender
        con.execute("insert into dim_patient values ('P013', 'U', 51)")                   # PLANT: bad gender code
        con.execute("insert into dim_patient values ('P014', 'F', -3)")                   # PLANT: negative age (range)

        # fct_encounters: 30 valid encounters spread across the real patients
        con.execute("create table fct_encounters (encounter_id varchar, patient_id varchar)")
        con.execute(
            "insert into fct_encounters "
            "select 'E' || lpad(cast(i as varchar), 4, '0'), "
            "'P' || lpad(cast((i % 10) + 1 as varchar), 3, '0') "
            "from range(0, 30) t(i)"
        )
        con.execute("insert into fct_encounters values ('E9999', 'GHOST')")           # PLANT: orphan FK

        # fct_medications: 20 valid 30-day supplies
        con.execute("create table fct_medications (medication_order_id varchar, days_supplied integer)")
        con.execute(
            "insert into fct_medications "
            "select 'M' || lpad(cast(i as varchar), 4, '0'), 30 from range(0, 20) t(i)"
        )
        con.execute("insert into fct_medications values ('M9999', -6)")               # PLANT: negative supply (range)

        # mart_readmissions: 9 of 100 readmitted → ~9% (inside the 3–25% band → that check PASSES)
        con.execute("create table mart_readmissions "
                    "(index_encounter_id varchar, is_30d_readmission boolean, days_to_next_admission integer)")
        con.execute("insert into mart_readmissions "
                    "select 'E' || lpad(cast(i as varchar), 4, '0'), (i < 9), "
                    "case when i < 9 then 15 else null end from range(0, 100) t(i)")
        con.execute("insert into mart_readmissions values ('E9998', false, -4)")      # PLANT: negative gap (range)
    finally:
        con.close()
    return str(db_path)


# ─────────────────────────────── CLI report ───────────────────────────────
_SEV = {"critical": "critical", "high": "high", "medium": "medium"}


def _print_report(report: dict) -> None:
    print("▶ DATA-QUALITY AGENT — detect · diagnose · propose fix\n")
    for c in report["checks"]:
        mark = "✓ PASS" if c["passed"] else ("⚠ ERROR" if c.get("error") else "✗ FAIL")
        print(f"  {mark:8} {c['name']:24} {c['summary']}")

    failing = [c for c in report["checks"] if not c["passed"]]
    if failing:
        print("\n  ── failing checks · diagnosis + proposed fix " + "─" * 26)
        for c in failing:
            print(f"\n  ✗ {c['name']}  [{_SEV.get(c['severity'], c['severity'])}]")
            if c.get("error"):
                print(f"      error     : {c['error']}")
                continue
            print(f"      diagnose  : {c['diagnosis']}")
            p = c.get("proposal", {})
            if p.get("fix"):
                print(f"      propose   : fix (confidence: {p.get('confidence', '?')}) — "
                      f"PROPOSAL for human review, not auto-applied:")
                for line in p["fix"].splitlines():
                    print(f"                    {line}")
                if p.get("explanation"):
                    print(f"      rationale : {p['explanation']}")
            else:
                print(f"      propose   : {p.get('note', 'no fix generated')}")

    print()
    if report["ok"]:
        print(f"  ✓ all {report['n_checks']} checks passed — warehouse is healthy.")
    else:
        bits = [f"{report['n_failed']} failed"]
        if report["n_errored"]:
            bits.append(f"{report['n_errored']} errored")
        print(f"  {', '.join(bits)} of {report['n_checks']} checks · {report['n_passed']} passed.")
        print("  Fixes above are PROPOSALS for human review — the agent never writes to the warehouse.")
    if not report["llm_available"]:
        print("  (No OPENAI_API_KEY: detect + diagnose ran deterministically; fix generation was skipped.)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m agent.quality_agent",
        description="Detect → diagnose → propose-fix data-quality checks over the DuckDB warehouse.")
    ap.add_argument("--demo", action="store_true",
                    help="run against a throwaway temp DB seeded with planted defects (shows the full loop)")
    ap.add_argument("--db", default=None,
                    help="path to a DuckDB warehouse to audit (default: the project warehouse)")
    args = ap.parse_args(argv)

    demo_path = None
    try:
        if args.demo:
            demo_path = make_demo_db()
            db = demo_path
            print(f"(demo) planted defects into a throwaway warehouse: {demo_path}\n")
        else:
            db = args.db
        _print_report(run(db_path=db))
        return 0
    finally:
        if demo_path:                                    # scrub the throwaway temp DB
            shutil.rmtree(Path(demo_path).parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
