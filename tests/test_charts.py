"""Unit tests for the deterministic chart/KPI logic (no key needed)."""
import pandas as pd

from agent import charts


def _prevalence_df():
    return pd.DataFrame({"age_group": ["18-39", "65-74"],
                         "patients_with_condition": [12, 62],
                         "total_patients_in_age_group": [309, 120],
                         "prevalence_pct": [3.88, 51.67]})


def test_pick_measure_prefers_rate_over_denominator():
    assert charts._pick_measure(_prevalence_df()) == "prevalence_pct"


def test_build_chart_none_for_single_value():
    assert charts.build_chart(pd.DataFrame({"n": [1139]})) is None


def test_build_chart_layers_for_category_measure():
    ch = charts.build_chart(_prevalence_df())
    assert ch is not None and len(ch.to_dict().get("layer", [])) >= 2


def test_kpi_cards_shape():
    cards = charts.kpi_cards(_prevalence_df())
    assert len(cards) == 3 and cards[0]["label"].startswith("highest")


def test_kpi_formats_fraction_rate_as_percent():
    # a rate stored as a fraction (0-1) must render as a percentage, not '0.1%' (the readmission-KPI bug)
    df = pd.DataFrame({"age_group": ["0-17", "75+"], "readmission_rate": [0.0, 0.1443]})
    cards = charts.kpi_cards(df)
    blob = " ".join(c.get("value", "") + c.get("sub", "") for c in cards)
    assert "14.4%" in blob and "0.1%" not in blob


def test_kpi_percent_column_left_unscaled():
    # a column already in percent units (>1.5) must NOT be multiplied again
    df = pd.DataFrame({"grp": ["a", "b"], "prevalence_pct": [3.88, 51.67]})
    blob = " ".join(c.get("sub", "") for c in charts.kpi_cards(df))
    assert "51.7%" in blob and "3.9%" in blob


def test_add_ci_computes_bounds():
    d = _prevalence_df().copy()
    assert charts._add_ci(d, "prevalence_pct") is True
    assert "_ci_lo" in d and "_ci_hi" in d
    assert (d["_ci_lo"] <= d["prevalence_pct"]).all() and (d["prevalence_pct"] <= d["_ci_hi"]).all()


def test_assurance_curve_chart_builds():
    model = {"model_type": "assurance", "error": None,
             "series": [{"n": 20, "assurance": 0.31}, {"n": 100, "assurance": 0.62}],
             "verdict": {"call": "GO"}}
    assert charts.assurance_curve_chart(model) is not None


def test_assurance_curve_chart_is_none_without_series():
    assert charts.assurance_curve_chart({"model_type": "assurance", "series": []}) is None


def test_oc_curve_chart_builds():
    model = {"model_type": "assurance", "error": None,
             "robustness": {"oc": [{"theta": 0.1, "go_rate": 0.02}, {"theta": 0.4, "go_rate": 0.9}]}}
    assert charts.oc_curve_chart(model) is not None
