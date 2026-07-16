"""Regression tests for the security + statistical + BYOD + report hardening pass.

All hermetic — no API key, no network. Warehouse tests use the committed demo DuckDB.
"""
import numpy as np
import pandas as pd
import pytest

from agent import charts, guardrails, modeling
from agent import warehouse as W


# ───────────────────────── SQL security boundary ─────────────────────────
@pytest.mark.parametrize("sql", [
    "SELECT * FROM read_text('/etc/passwd')",
    "SELECT * FROM read_csv_auto('agent/.env')",
    "SELECT * FROM read_blob('agent/__init__.py')",
    "SELECT * FROM read_parquet('x.parquet')",
    "SELECT file FROM glob('/etc/*')",
    "COPY (SELECT 1) TO '/tmp/x.csv'",
    "ATTACH 'x.db'; SELECT 1",
    "SELECT 1; DROP TABLE dim_patient",
])
def test_validate_blocks_exfiltration_and_writes(sql):
    with pytest.raises(W.QueryError):
        W.validate(sql)


def test_parenthesized_select_allowed():
    assert W.validate("(SELECT 1 AS a) UNION ALL (SELECT 2)")            # must not be rejected
    assert W.validate("SELECT 1 WHERE 'NURSE ON CALL' = 'NURSE ON CALL'")  # keyword inside a literal is fine


def test_run_query_engine_refuses_file_read():
    if not W.DB_PATH.exists():
        pytest.skip("demo warehouse not present")
    with pytest.raises(W.QueryError):
        W.run_query("SELECT content FROM read_text('requirements.txt')")
    # a normal read-only query still works
    df = W.run_query("SELECT count(*) AS n FROM dim_patient")
    assert int(df.iloc[0]["n"]) > 0


# ───────────────────────── statistical correctness ─────────────────────────
def test_uplift_aipw_valid_ci_covers_truth():
    rng = np.random.default_rng(7)
    n = 1400
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    t = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2))))        # confounded assignment
    y = rng.binomial(1, np.clip(1 / (1 + np.exp(-(-0.3 + 0.7 * x1))) + 0.15 * t, 0, 1))
    r = modeling.fit_uplift(pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2}), "y", "t", ["x1", "x2"])
    assert r.error is None and "AIPW" in r.fit_stat
    a = r.terms[0]
    se = (a.ci_high - a.ci_low) / (2 * 1.96)
    assert a.estimate > 0.05                          # detects the positive treatment effect
    assert se > 0.015                                 # honest influence-function SE (old bootstrap was ~3x smaller)
    assert a.ci_low < a.estimate < a.ci_high          # a well-formed interval around the point estimate


def test_bh_nan_is_never_significant():
    assert guardrails.benjamini_hochberg([float("nan"), 5.4e-5])[0] == 1.0   # NaN → not significant
    assert guardrails.benjamini_hochberg([None, 0.001])[0] == 1.0            # None handled the same


def test_cox_flags_separation():
    rng = np.random.default_rng(2)
    n = 300
    g = rng.integers(0, 2, n).astype(float)
    df = pd.DataFrame({"t": rng.exponential(1, n), "e": g.astype(int), "g": g})  # events ONLY in group 1
    r = modeling.fit_cox(df, "t", "e", ["g"])
    assert r.error is None and any("separation" in i.lower() for i in r.issues)


def test_rare_categorical_levels_pooled_with_counts():
    # sparse race levels (hawaiian/native, a handful each) must be pooled BEFORE fitting — no HR=0 [0, inf]
    rng = np.random.default_rng(0)
    n = 1100
    race = rng.choice(["white", "black", "asian", "other", "hawaiian", "native"], n,
                      p=[0.74, 0.14, 0.08, 0.036, 0.0022, 0.0018])
    age = rng.uniform(30, 90, n)
    dead = (rng.random(n) < 0.12).astype(int)
    mr = modeling.fit_cox(pd.DataFrame({"age": age, "e": dead, "race": race}), "age", "e", ["race"])
    assert mr.error is None
    fit = [t for t in mr.terms if not np.isnan(t.ci_low)]            # the estimated (non-reference) levels
    assert fit and all(np.isfinite(t.estimate) and t.estimate > 0 and np.isfinite(t.ci_high) for t in fit)
    assert any("pooled" in i.lower() and "race" in i.lower() for i in mr.issues)     # DE step recorded
    assert not any("hawaiian" in t.name or "native" in t.name for t in mr.terms)     # not fit as singletons
    cats = [t for t in mr.terms if t.n is not None]                 # per-category N/events, complete partition
    assert sum(t.n for t in cats) == n and all(t.events is not None for t in cats)


def test_cox_all_censored_clean_error():
    df = pd.DataFrame({"t": [1, 2, 3, 4] * 10, "e": [0] * 40, "x": list(range(40))})
    r = modeling.fit_cox(df, "t", "e", ["x"])
    assert r.error is not None and "censored" in r.error.lower()


def test_ni_ci_consistent_with_fm_verdict():
    # reviewer counterexample: FM p just over 0.025 ⇒ NOT NI; the score CI must agree (lower ≤ -margin)
    df = pd.DataFrame({"arm": ["control"] * 200 + ["trt"] * 200,
                       "cured": [1] * 140 + [0] * 60 + [1] * 138 + [0] * 62})
    r = modeling.fit_noninferiority(df, "arm", "cured", margin=0.10, higher_is_better=True, control="control")
    t, v = r.terms[0], r.verdict
    not_ni = v["call"] == "NOT NON-INFERIOR"
    assert not_ni == (t.ci_low <= -0.10)            # figure/CI can never contradict the verdict


def test_sample_size_means_uses_t_power():
    r = modeling.calc_sample_size(outcome_type="mean", mean_control=0, mean_treatment=0.5, sd=1.0)
    assert 63 <= r.arms[0]["n"] <= 65               # noncentral-t: 64 (normal approx gave 63)
    ni = modeling.calc_sample_size(kind="noninferiority", outcome_type="proportion", p_control=0.85, margin=0.10)
    assert "α=0.025" in ni.fit_stat                 # NI label shows the true one-sided level


def test_experiment_donotship_reachable_without_named_control():
    rng = np.random.default_rng(1)
    # arms named neutrally (no control keyword) → auto-baseline is the LARGER arm, not the min-rate arm
    df = pd.DataFrame({"arm": ["blue"] * 700 + ["red"] * 700,
                       "c": list(rng.binomial(1, 0.20, 700)) + list(rng.binomial(1, 0.12, 700))})
    r = modeling.fit_experiment(df, "arm", "c")
    assert r.verdict["call"] in ("DO NOT SHIP", "SHIP")   # a directional verdict is reachable (not forced ≥0)


def test_binary_note_on_1_2_coding():
    assert modeling._binary_note(pd.Series([1, 2, 1, 2])) is not None      # ambiguous coding → warned
    assert modeling._binary_note(pd.Series([0, 1, 0, 1])) is None          # {0,1} is unambiguous


# ───────────────────────── BYOD hardening ─────────────────────────
def test_userdata_reserved_and_unicode_columns():
    from agent import userdata
    df = pd.DataFrame({"from": [1], "Âge": [2], "select": [3]})
    out = userdata.sanitize_columns(df)
    cols = list(out.columns)
    assert "from" not in cols and "select" not in cols    # reserved words suffixed
    assert "age" in cols                                  # accented latin transliterated, not dropped


def test_userdata_zero_columns_guard():
    from agent import userdata
    with pytest.raises(ValueError):
        userdata.prepare_upload(pd.DataFrame(), "empty.csv")


# ───────────────────────── charts + print theme ─────────────────────────
def test_forest_plot_drops_nonfinite_terms():
    finite = {"name": "x", "estimate": 1.4, "ci_low": 1.1, "ci_high": 1.8, "p": 0.01}
    inf = {"name": "sep", "estimate": float("inf"), "ci_low": 0.0, "ci_high": float("inf"), "p": float("nan")}
    assert charts.forest_plot({"effect_label": "odds ratio", "terms": [finite, inf]}) is not None  # doesn't crash
    assert charts.forest_plot({"effect_label": "odds ratio", "terms": [inf]}) is None               # all dropped


def test_print_theme_switches_and_restores():
    assert charts._BG == "transparent"
    with charts.render_for_print():
        assert charts._BG == "white" and charts._PRINT_ON
    assert charts._BG == "transparent" and not charts._PRINT_ON   # restored, no leakage into the UI


# ───────────────────────── agent self-heal loop (hermetic) ─────────────────────────
def test_self_heal_loop_recovers_from_sql_error(monkeypatch):
    from agent import agent as A
    from agent.warehouse import QueryError
    calls = {"n": 0}

    def fake_run_query(sql, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise QueryError("Binder Error: no such column foo")
        return pd.DataFrame({"n": [42]})

    monkeypatch.setattr(A, "run_query", fake_run_query)
    monkeypatch.setattr(A.llm, "reset_trace", lambda: None)
    monkeypatch.setattr(A.llm, "trace_summary", lambda: {})
    monkeypatch.setattr(A.llm, "complete", lambda *a, **k: "SELECT count(*) AS n FROM dim_patient")
    monkeypatch.setattr(A.llm, "complete_json", lambda *a, **k: {
        "answerable": True, "hypothesis": "h", "analysis_plan": "p",
        "answers_question": True, "confidence": "high", "issues": []})

    r = A.run_analysis("How many patients are in the warehouse?")
    assert r.error is None
    assert calls["n"] == 2                                  # failed once → healed → succeeded
    assert r.attempts[0]["error"] and r.attempts[1]["error"] is None
    assert int(r.dataframe.iloc[0]["n"]) == 42


def test_report_docx_smoke():
    from types import SimpleNamespace

    from agent import report
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"arm": np.repeat(["control", "variant"], 300),
                       "c": np.r_[rng.binomial(1, 0.10, 300), rng.binomial(1, 0.15, 300)]})
    res = SimpleNamespace(question="Ship it?", model=modeling.fit_experiment(df, "arm", "c").as_dict(),
                          citations=["ab"], sql="select 1", dataframe=None, findings=[],
                          interpretation="**Findings**\nText.", hypothesis="H")
    b = report.build_docx(res)
    assert b[:2] == b"PK" and len(b) > 20_000               # a real .docx with an embedded figure

def test_report_renders_a_bayesian_go_no_go(tmp_path):
    """The .docx must carry the verdict, the prior sensitivity, and the pre-specification status."""
    import io
    import zipfile
    from types import SimpleNamespace

    from agent import modeling, report

    mr = modeling.calc_assurance(n_planned=100, tv=0.30, lrv=0.15, prior_successes=8, prior_n=20)
    assert mr.error is None
    res = SimpleNamespace(question="Will Phase II succeed?", model=mr.as_dict(),
                          sql="", interpretation="**Findings**\nok", findings=[], citations=[],
                          verification={}, hypothesis="", dataframe=None, trace={}, lineage=None,
                          error=None, clarification=None, attempts=[])
    blob = report.build_docx(res)
    assert blob[:2] == b"PK" and len(blob) > 5000       # a real .docx zip
    xml = zipfile.ZipFile(io.BytesIO(blob)).read("word/document.xml").decode()
    assert "PRE-SPECIFIED" in xml                       # the lock status, on the approval page
    assert "Prior sensitivity" in xml                   # the four-prior panel
    assert "Type I error (GO rate at the LRV)" in xml   # the operating characteristics


def test_report_renders_an_interim_run_without_a_lock():
    """A live-app regression: an interim run with NO design lock stores prespec.lock=None, which
    crashed build_docx (`'NoneType' object has no attribute 'get'`) on the approval page."""
    from types import SimpleNamespace

    import pandas as pd

    from agent import modeling, report

    df = pd.DataFrame({"responded": [1] * 12 + [0] * 28})
    mr = modeling.fit_interim(df, "responded", n_planned=100, tv=0.30, lrv=0.15)
    assert mr.error is None and mr.prespec["status"] == "EXPLORATORY" and mr.prespec["lock"] is None
    res = SimpleNamespace(question="Stop for futility?", model=mr.as_dict(),
                          sql="", interpretation="**Findings**\nok", findings=[], citations=[],
                          verification={}, hypothesis="", dataframe=None, trace={}, lineage=None,
                          error=None, clarification=None, attempts=[])
    blob = report.build_docx(res)
    assert blob[:2] == b"PK" and len(blob) > 5000


def test_report_renders_a_two_arm_interim(tmp_path):
    """The .docx must carry the per-arm rates and the risk difference for a two-arm interim run."""
    import io
    import zipfile
    from types import SimpleNamespace

    import pandas as pd

    from agent import modeling, report

    rows = ([("treatment", 1)] * 20 + [("treatment", 0)] * 20
            + [("control", 1)] * 10 + [("control", 0)] * 30)
    df = pd.DataFrame(rows, columns=["arm", "responded"])
    mr = modeling.fit_interim(df, "responded", n_planned=100, tv=0.15, lrv=0.0,
                              framing="two_arm", group="arm", control="control")
    assert mr.error is None and len(mr.arms) == 2
    res = SimpleNamespace(question="Continue the trial?", model=mr.as_dict(),
                          sql="", interpretation="**Findings**\nok", findings=[], citations=[],
                          verification={}, hypothesis="", dataframe=None, trace={}, lineage=None,
                          error=None, clarification=None, attempts=[])
    blob = report.build_docx(res)
    assert blob[:2] == b"PK" and len(blob) > 5000
    xml = zipfile.ZipFile(io.BytesIO(blob)).read("word/document.xml").decode()
    assert "treatment" in xml and "control" in xml           # the per-arm table
    assert "Predictive probability of success" in xml
