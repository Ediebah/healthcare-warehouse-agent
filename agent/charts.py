"""Industry-grade visualization: KPI cards + an annotated, story-telling chart.

Deterministic (no LLM) so the visualization is reliable. From the result's shape we produce:
  * kpi_cards(df)  — 1-3 headline metric cards (the extremes + spread) for instant reading.
  * build_chart(df) — a layered Altair chart: sorted bars, direct value labels, the extreme
                      highlighted, and — the differentiator — Wilson 95% CI error bars whenever the
                      result carries a numerator + denominator, so uncertainty is shown, not hidden.

Themed to the clinical-teal palette.
"""
from __future__ import annotations

import re
import threading

import altair as alt
import numpy as np
import pandas as pd

from .guardrails import wilson_ci

# Two palettes: SCREEN (dark clinical-teal, for the Streamlit UI) and PRINT (light, for the .docx export
# where charts sit on a WHITE page). Builders read the module-level names below; `render_for_print()`
# rebinds them to the light palette for the duration of a docx render so every title, axis label, value
# label and CI whisker stays legible on white (the on-screen dark theme rendered near-invisibly there).
_SCREEN = dict(teal="#4fd1c5", teal_hi="#8af7ea", teal_dim="#2c6f68", muted="#8ea0b0",
               grid="#1a2531", domain="#20303f", ink="#cfe0ec", bg="transparent", title="#cfe0ec")
_PRINT = dict(teal="#0d9488", teal_hi="#0f766e", teal_dim="#5eead4", muted="#475569",
              grid="#e5e7eb", domain="#94a3b8", ink="#0f172a", bg="white", title="#0f172a")

TEAL = _SCREEN["teal"]
TEAL_HI = _SCREEN["teal_hi"]
TEAL_DIM = _SCREEN["teal_dim"]
MUTED = _SCREEN["muted"]
GRID = _SCREEN["grid"]
DOMAIN = _SCREEN["domain"]
INK = _SCREEN["ink"]
_BG = _SCREEN["bg"]
_TITLE = _SCREEN["title"]
_PRINT_ON = False


def _apply_palette(p: dict) -> None:
    global TEAL, TEAL_HI, TEAL_DIM, MUTED, GRID, DOMAIN, INK, _BG, _TITLE, _PRINT_ON
    TEAL, TEAL_HI, TEAL_DIM = p["teal"], p["teal_hi"], p["teal_dim"]
    MUTED, GRID, DOMAIN, INK = p["muted"], p["grid"], p["domain"], p["ink"]
    _BG, _TITLE, _PRINT_ON = p["bg"], p["title"], p is _PRINT


# The palette is MODULE STATE, and Streamlit serves each session on its own thread in one process:
# without a lock, session B's docx export exiting mid-way through session A's would restore the
# screen palette under A's feet (dark-ink figures on a white page, or vice versa). The lock
# serializes exports; a concurrent on-screen rebuild can still catch the print palette briefly,
# which is the residual cost of the palette being global.
_PRINT_LOCK = threading.Lock()


class render_for_print:
    """Context manager: switch the chart palette to the light PRINT theme (white background, dark ink,
    subtle light gridlines) so figures rendered to PNG for the .docx are legible on a white page.
    Restores the SCREEN palette on exit. Exports are serialized by a process-wide lock — two
    concurrent sessions' exports cannot interleave the swap/restore."""
    def __enter__(self):
        _PRINT_LOCK.acquire()
        _apply_palette(_PRINT)
        return self

    def __exit__(self, *exc):
        _apply_palette(_SCREEN)
        _PRINT_LOCK.release()
        return False

_TIERS = [
    re.compile(r"(rate|pct|percent|prevalence|proportion|ratio)", re.I),
    re.compile(r"(avg|mean|median)", re.I),
    re.compile(r"(cost|amount|charge|price|expense|income|revenue)", re.I),
    re.compile(r"(count|num|total|sum|_n$|^n$)", re.I),
]
_ID = re.compile(r"(_id$|^id$|code$|zip|latitude|longitude|_seq$)", re.I)
_TIME = re.compile(r"(date|year|month|day|_at$|start|stop|time)", re.I)
_NUMER = re.compile(r"(patients_with|_with_|numerator|cases|events|readmit|readmiss|affected|positive)", re.I)
_DENOM = re.compile(r"(total|denom|cohort|sample|population|_size|(^|_)n($|_)|num_patients|total_patients)", re.I)
_PCTISH = re.compile(r"(pct|percent|rate|prevalence|proportion)", re.I)
_MONEY = re.compile(r"(cost|amount|charge|price|expense|income|revenue)", re.I)


def _numeric(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _cats(df):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def _pick_measure(df):
    nums = [c for c in _numeric(df) if not _ID.search(str(c))]
    if not nums:
        return None
    for tier in _TIERS:
        hit = [c for c in nums if tier.search(str(c))]
        if hit:
            return hit[0]
    return nums[0]


def _pct_scale(series) -> float:
    """For a rate/proportion column, return 100 when the values are stored as a fraction (0-1) rather than
    an already-scaled percentage (0-100), so 0.1443 renders as '14.4%' — not '0.1%'."""
    m = pd.to_numeric(pd.Series(series), errors="coerce").abs().max()
    return 100.0 if (m == m and m <= 1.5) else 1.0


def _fmt(col, v, pct_scale: float = 1.0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    c = str(col).lower()
    if _PCTISH.search(c):
        return f"{v * pct_scale:.1f}%"
    if _MONEY.search(c):
        return f"${v:,.0f}"
    return f"{int(v):,}" if float(v).is_integer() else f"{v:,.1f}"


def kpi_cards(df: pd.DataFrame, question: str = "") -> list[dict]:
    """1-3 headline cards. For a category × measure result: the top, the bottom, and the spread."""
    if df is None or len(df) == 0:
        return []
    df = df.reset_index(drop=True)                       # guarantee a unique index (idxmax → one row)
    y = _pick_measure(df)
    if y is None or df[y].dropna().empty:                # no measure, or all values missing
        return []
    psc = _pct_scale(df[y])                              # render a fraction rate (0-1) as a percentage
    cats = [c for c in _cats(df) if not _ID.search(str(c))]
    if len(df) == 1:
        return [{"label": str(y).replace("_", " "), "value": _fmt(y, df.iloc[0][y], psc), "sub": ""}]
    if not cats:
        col = df[y].dropna()
        return [{"label": f"max {y}".replace("_", " "), "value": _fmt(y, col.max(), psc), "sub": ""},
                {"label": f"median {y}".replace("_", " "), "value": _fmt(y, col.median(), psc), "sub": ""}]
    x = cats[0]
    valid = df.dropna(subset=[y])                        # idxmax/idxmin need non-NA
    top = valid.loc[valid[y].idxmax()]
    bot = valid.loc[valid[y].idxmin()]
    return [
        {"label": f"highest {x}".replace("_", " "), "value": str(top[x]), "sub": _fmt(y, top[y], psc)},
        {"label": f"lowest {x}".replace("_", " "), "value": str(bot[x]), "sub": _fmt(y, bot[y], psc)},
        {"label": "spread", "value": _fmt(y, top[y] - bot[y], psc), "sub": "high − low"},
    ]


def _add_ci(d: pd.DataFrame, y: str):
    """Add Wilson 95% CI columns (in the measure's units) if a numerator + denominator are present."""
    numer = [c for c in _numeric(d) if _NUMER.search(str(c))]
    denom = [c for c in _numeric(d) if _DENOM.search(str(c))]
    if not numer or not denom:
        return False
    kcol, ncol = numer[0], max(denom, key=lambda c: d[c].sum())
    ymax = float(d[y].max())
    if _PCTISH.search(str(y)) and ymax <= 100:
        scale = 100.0 if ymax > 1.5 else 1.0             # percentage (0-100) vs fraction (0-1)
    elif ymax <= 1.5:
        scale = 1.0                                      # a bare proportion
    else:
        return False                                     # a count/cost — a Wilson CI in these units is meaningless
    los, his = [], []
    for _, row in d.iterrows():
        n = int(row[ncol]) if np.isfinite(row[ncol]) else 0    # finite guard (inf → int() OverflowError)
        k = int(row[kcol]) if np.isfinite(row[kcol]) else 0
        lo, hi = wilson_ci(k, n) if n > 0 else (np.nan, np.nan)
        los.append(lo * scale)
        his.append(hi * scale)
    d["_ci_lo"], d["_ci_hi"] = los, his
    return not d["_ci_lo"].isna().all()


def build_chart(df: pd.DataFrame, question: str = ""):
    """Layered Altair chart: bars + value labels + highlighted extreme + Wilson CI error bars."""
    if df is None or len(df) < 2:
        return None
    y = _pick_measure(df)
    if y is None:
        return None

    time_cols = [c for c in df.columns if _TIME.search(str(c)) or pd.api.types.is_datetime64_any_dtype(df[c])]
    cats = [c for c in _cats(df) if not _ID.search(str(c))]

    if time_cols:                                            # time series → line
        x = time_cols[0]
        d = df[[x, y]].dropna()
        is_dt = pd.api.types.is_datetime64_any_dtype(df[x])
        chart = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(color=TEAL), color=TEAL).encode(
            x=alt.X(f"{x}:{'T' if is_dt else 'O'}", title=str(x)),
            y=alt.Y(f"{y}:Q", title=str(y)),
            tooltip=list(d.columns),
        )
        return _finish(chart, 300)

    if cats:                                                 # category × measure → annotated bars
        x = cats[0]
        keep = [c for c in df.columns if c == x or pd.api.types.is_numeric_dtype(df[c])]
        d = df[keep].dropna(subset=[y]).sort_values(y, ascending=False).head(15).copy()
        top_val = d[y].max()
        d["_top"] = d[y] == top_val
        psc = _pct_scale(d[y])
        d["_label"] = d[y].map(lambda v: _fmt(y, v, psc))
        has_ci = _add_ci(d, y)

        base = alt.Chart(d).encode(y=alt.Y(f"{x}:N", sort="-x", title=None))
        bars = base.mark_bar().encode(
            x=alt.X(f"{y}:Q", title=str(y).replace("_", " ")),
            color=alt.condition("datum._top", alt.value(TEAL_HI), alt.value(TEAL)),
            tooltip=[c for c in d.columns if not c.startswith("_")],
        )
        layers = [bars]
        if has_ci:                                           # Wilson 95% CI error bars
            layers.append(base.mark_rule(color=INK, opacity=0.8).encode(
                x=alt.X("_ci_lo:Q"), x2="_ci_hi:Q"))
            layers.append(base.mark_tick(color=INK, thickness=2, size=8).encode(x="_ci_lo:Q"))
            layers.append(base.mark_tick(color=INK, thickness=2, size=8).encode(x="_ci_hi:Q"))
        layers.append(base.mark_text(align="left", dx=5, color=MUTED, fontSize=11).encode(
            x=alt.X(f"{y}:Q"), text="_label:N"))
        chart = alt.layer(*layers)
        title = f"{str(y).replace('_', ' ')} by {str(x).replace('_', ' ')}"
        if has_ci:
            title += "  ·  bars = estimate, whiskers = 95% CI"
        return _finish(chart, min(400, 70 + 30 * len(d)), title)

    return None


def forest_plot(model: dict):
    """Forest plot of a model's effect estimates: point + 95% CI whiskers, null reference line.
    Log x-axis for ratios (OR/HR), linear for coefficients. Colored by significance."""
    if not model or model.get("error"):
        return None
    label = model.get("effect_label", "estimate")
    is_ratio = "ratio" in label
    null = 1.0 if is_ratio else 0.0

    def _ok(t):                                    # drop non-finite estimates (separation → OR/HR=inf);
        if not (np.isfinite(t["ci_low"]) and np.isfinite(t["ci_high"]) and np.isfinite(t["estimate"])):
            return False                           # they're flagged in the text, and would wreck the axis
        return (t["ci_low"] > 0 and t["ci_high"] > 0) if is_ratio else True   # log axis needs positive bounds
    terms = [t for t in model.get("terms", []) if _ok(t)]
    if not terms:
        return None

    def _sig(t):                                   # p<0.05, or (no p, e.g. AIPW ATE) CI excludes the null
        if t["p"] is not None and t["p"] == t["p"]:
            return "significant" if t["p"] < 0.05 else "n.s."
        return "significant" if (t["ci_low"] > null or t["ci_high"] < null) else "n.s."

    def _vlabel(t):                                # the numeric estimate a clinical reader must be able to read
        j = "–" if is_ratio else " to "
        return f"{t['estimate']:.2f} ({t['ci_low']:.2f}{j}{t['ci_high']:.2f})"

    d = pd.DataFrame([{"term": t["name"], "est": t["estimate"], "lo": t["ci_low"], "hi": t["ci_high"],
                       "sig": _sig(t), "vlabel": _vlabel(t)} for t in terms])
    xscale = alt.Scale(type="log", nice=False, padding=8) if is_ratio else alt.Scale(zero=False, padding=8)
    order = list(d["term"])
    base = alt.Chart(d).encode(y=alt.Y("term:N", sort=order, title=None,
                                       axis=alt.Axis(labelLimit=200, ticks=False, domain=False)))
    xaxis = alt.Axis(grid=False, title=f"{label}  (95% CI)")     # no gridlines → a clean, uncluttered forest
    ci = base.mark_rule(color=MUTED, strokeWidth=1.5).encode(x=alt.X("lo:Q", scale=xscale, axis=xaxis), x2="hi:Q")
    cap_lo = base.mark_tick(color=MUTED, thickness=1.5, size=7).encode(x=alt.X("lo:Q", scale=xscale))
    cap_hi = base.mark_tick(color=MUTED, thickness=1.5, size=7).encode(x=alt.X("hi:Q", scale=xscale))
    pts = base.mark_point(filled=True, size=95).encode(
        x=alt.X("est:Q", scale=xscale),
        color=alt.Color("sig:N", scale=alt.Scale(domain=["significant", "n.s."], range=[TEAL, "#94a3b8"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["term", "est", "lo", "hi"])
    vals = base.mark_text(align="center", dy=-12, fontSize=10, color=INK).encode(
        x=alt.X("est:Q", scale=xscale), text="vlabel:N")
    ref = alt.Chart(pd.DataFrame({"x": [null]})).mark_rule(
        color="#e0a83e", strokeDash=[5, 4], strokeWidth=1.5).encode(x="x:Q")
    chart = alt.layer(ref, ci, cap_lo, cap_hi, pts, vals).resolve_scale(x="shared")
    return _finish(chart, 64 + 48 * len(d), f"Forest plot — {label} (dashed line = no effect)", width=640)


def survival_plot(km: list):
    """Kaplan-Meier survival curves (step lines + optional 95% CI bands), by group."""
    if not km:
        return None
    d = pd.DataFrame(km)
    colors = alt.Scale(range=[TEAL, "#8ab4f8", "#f5c451", "#f87171", "#a78bfa"])
    base = alt.Chart(d)
    line = base.mark_line(interpolate="step-after", strokeWidth=2).encode(
        x=alt.X("time:Q", title="time"),
        y=alt.Y("survival:Q", title="survival probability", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("group:N", scale=colors,
                        legend=alt.Legend(title=None, orient="top", labelColor=MUTED)))
    layers = [line]
    if "ci_low" in d and d["ci_low"].notna().any():
        band = base.mark_area(opacity=0.12, interpolate="step-after").encode(
            x="time:Q", y="ci_low:Q", y2="ci_high:Q", color=alt.Color("group:N", scale=colors, legend=None))
        layers = [band, line]
    return _finish(alt.layer(*layers), 340, "Kaplan-Meier survival curve")


def importance_chart(model: dict):
    """Horizontal bars of random-forest permutation importances (most predictive feature on top)."""
    if not model or model.get("error"):
        return None
    terms = model.get("terms", [])
    if not terms:
        return None
    d = pd.DataFrame([{"feature": t["name"], "importance": t["estimate"]} for t in terms])
    d = d.sort_values("importance", ascending=False)
    order = list(d["feature"])
    base = alt.Chart(d).encode(y=alt.Y("feature:N", sort=order, title=None))
    bars = base.mark_bar(color=TEAL, cornerRadiusEnd=3, height={"band": 0.7}).encode(
        x=alt.X("importance:Q", title="permutation importance (drop in model skill)"),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")])
    labels = base.mark_text(align="left", dx=4, color=INK, fontSize=11).encode(
        x="importance:Q", text=alt.Text("importance:Q", format=".3f"))
    return _finish(alt.layer(bars, labels), min(460, 110 + 46 * len(d)),
                   "Feature importance — what most predicts the outcome")


def forecast_chart(series: list):
    """Time-series history (solid) + forecast (dashed) with a 95% prediction band."""
    if not series:
        return None
    d = pd.DataFrame(series)
    d["time"] = pd.to_datetime(d["time"])
    hist, fc = d[d["kind"] == "history"], d[d["kind"] == "forecast"]
    if len(hist):                                  # bridge the last observed point into the forecast line
        fc = pd.concat([hist.tail(1).assign(kind="forecast"), fc], ignore_index=True)
    layers = []
    if len(fc) and fc["lower"].notna().any():
        layers.append(alt.Chart(fc).mark_area(opacity=0.15, color=TEAL).encode(
            x=alt.X("time:T", title=None), y=alt.Y("lower:Q", title="value"), y2="upper:Q"))
    layers.append(alt.Chart(hist).mark_line(color=TEAL, strokeWidth=2).encode(
        x="time:T", y=alt.Y("value:Q", title="value"), tooltip=["time:T", "value:Q"]))
    layers.append(alt.Chart(fc).mark_line(color=TEAL_HI, strokeWidth=2, strokeDash=[5, 4]).encode(
        x="time:T", y="value:Q", tooltip=["time:T", "value:Q"]))
    return _finish(alt.layer(*layers), 340,
                   "Forecast — history (solid) + projection (dashed) with 95% band")


def experiment_chart(model: dict):
    """A/B outcome by variant — bars with 95% CI whiskers; winner highlighted, control dimmed."""
    if not model or model.get("error") or not model.get("arms"):
        return None
    binm = "conversion" in model.get("effect_label", "")
    d = pd.DataFrame(model["arms"])

    def _role(r):
        return "winner" if r["is_winner"] else ("control" if r["is_baseline"] else "variant")

    d["role"] = d.apply(_role, axis=1)
    fmt = ".0%" if binm else ".2f"
    y_title = "conversion rate" if binm else f"mean {model.get('outcome', 'value')}"
    order = list(d["arm"])
    present = [r for r in ("winner", "variant", "control") if r in set(d["role"])]   # no phantom legend rows
    palette = {"winner": TEAL, "variant": "#7c93a8", "control": "#334759"}
    base = alt.Chart(d).encode(x=alt.X("arm:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)))
    bars = base.mark_bar(cornerRadiusEnd=3, size=54).encode(
        y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=fmt)),
        color=alt.Color("role:N", scale=alt.Scale(domain=present, range=[palette[r] for r in present]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["arm", alt.Tooltip("value:Q", format=fmt), "n:Q"])
    err = base.mark_rule(color=INK, strokeWidth=1.5).encode(y=alt.Y("ci_low:Q", title=y_title), y2="ci_high:Q")
    cap_lo = base.mark_tick(color=INK, thickness=1.5, size=14, orient="horizontal").encode(y="ci_low:Q")
    cap_hi = base.mark_tick(color=INK, thickness=1.5, size=14, orient="horizontal").encode(y="ci_high:Q")
    labels = base.mark_text(dy=-8, color=INK, fontSize=13, fontWeight="bold").encode(
        y="ci_high:Q", text=alt.Text("value:Q", format=fmt))
    return _finish(alt.layer(bars, err, cap_lo, cap_hi, labels), 300, "Outcome by variant (95% CI)",
                   width=min(560, 200 + 120 * len(d)))


def power_curve_chart(model: dict):
    """Sample size (per arm) vs statistical power, with the chosen power/n marked."""
    if not model or model.get("error") or not model.get("series"):
        return None
    d = pd.DataFrame(model["series"])
    if "power" not in d or "n" not in d:
        return None
    chosen = (model.get("verdict") or {}).get("power")
    line = alt.Chart(d).mark_line(color=TEAL, strokeWidth=2, point=alt.OverlayMarkDef(color=TEAL)).encode(
        x=alt.X("power:Q", title="statistical power", axis=alt.Axis(format="%")),
        y=alt.Y("n:Q", title="n per arm"),
        tooltip=[alt.Tooltip("power:Q", format=".0%"), "n:Q"])
    layers = [line]
    if chosen is not None:
        rule = alt.Chart(pd.DataFrame({"power": [chosen]})).mark_rule(
            color="#f5c451", strokeDash=[5, 4]).encode(x="power:Q")
        layers.append(rule)
    return _finish(alt.layer(*layers), 300, "Sample size vs power")


def ni_plot(model: dict):
    """Non-inferiority plot: the treatment−control effect with 95% CI, against the NI margin and zero.
    Green point if non-inferior, red if not; gold dashed line = the margin, grey dotted = no difference."""
    if not model or model.get("error") or not model.get("terms"):
        return None
    t = model["terms"][0]
    v = model.get("verdict", {})
    margin = v.get("margin")
    if margin is None:
        return None
    refline = -margin if v.get("higher_is_better", True) else margin
    ni = v.get("call") == "NON-INFERIOR"
    d = pd.DataFrame([{"label": t["name"], "est": t["estimate"], "lo": t["ci_low"], "hi": t["ci_high"]}])
    base = alt.Chart(d).encode(y=alt.Y("label:N", title=None))
    x = alt.X("lo:Q", title="difference: treatment − control  (95% CI)", scale=alt.Scale(zero=False))
    ci = base.mark_rule(color=MUTED, strokeWidth=2).encode(x=x, x2="hi:Q")
    cap_lo = base.mark_tick(color=MUTED, thickness=2, size=10).encode(x="lo:Q")
    cap_hi = base.mark_tick(color=MUTED, thickness=2, size=10).encode(x="hi:Q")
    pt = base.mark_point(filled=True, size=180, color=TEAL if ni else "#f87171").encode(
        x="est:Q", tooltip=["label", "est", "lo", "hi"])
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        color=MUTED, strokeDash=[2, 3]).encode(x="x:Q")
    marg = alt.Chart(pd.DataFrame({"x": [refline], "t": ["NI margin"]}))
    marg_line = marg.mark_rule(color="#f5c451", strokeWidth=2, strokeDash=[6, 3]).encode(x="x:Q")
    marg_txt = marg.mark_text(color="#f5c451", dy=-70, fontSize=11).encode(x="x:Q", text="t:N")
    chart = alt.layer(marg_line, zero, ci, cap_lo, cap_hi, pt, marg_txt).resolve_scale(x="shared")
    return _finish(chart, 190, "Non-inferiority — effect vs the margin (gold) and no-difference (grey)")


def radar_chart(df: pd.DataFrame, question: str = ""):
    """Radar/spider chart comparing a few entities across several metrics (each axis min-max
    normalized so scales are comparable). Appropriate ONLY for 2-6 entities × ≥3 numeric measures."""
    if df is None:
        return None
    cats = [c for c in _cats(df) if not _ID.search(str(c))]
    measures = [c for c in _numeric(df) if not _ID.search(str(c))]
    if not cats or len(measures) < 3 or not (2 <= len(df) <= 6):
        return None
    import plotly.graph_objects as go
    cat = cats[0]
    d = df.head(6).reset_index(drop=True)
    measures = measures[:7]
    norm = d[measures].astype(float).copy()
    for m in measures:
        lo, hi = norm[m].min(), norm[m].max()
        norm[m] = 0.5 if hi == lo else (norm[m] - lo) / (hi - lo)
    axes = [m.replace("_", " ") for m in measures]
    palette = [TEAL, "#8ab4f8", "#f5c451", "#f87171", "#a78bfa", "#5eead4"]
    fig = go.Figure()
    for i in range(len(d)):
        vals = [float(norm.iloc[i][m]) for m in measures]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=axes + [axes[0]], fill="toself",
            name=str(d.iloc[i][cat]), line={"color": palette[i % len(palette)]}, opacity=0.7))
    fig.update_layout(
        polar={"bgcolor": "#131c27",
               "radialaxis": {"visible": True, "range": [0, 1], "showticklabels": False, "gridcolor": DOMAIN},
               "angularaxis": {"gridcolor": DOMAIN, "tickfont": {"color": MUTED, "size": 11}}},
        paper_bgcolor="rgba(0,0,0,0)", font={"color": MUTED},
        legend={"font": {"color": INK}}, margin={"l": 60, "r": 60, "t": 20, "b": 20}, height=380)
    return fig


def _finish(chart, height: int, title: str = "", width=None):
    props = {"height": height, "background": _BG}
    if _PRINT_ON:
        # explicit width → a landscape figure. Without it Word scales the narrow PNG to 6in wide and
        # stretches the height proportionally, producing the tall, whitespace-heavy figures.
        props["width"] = width or 660
    c = chart.properties(**props)
    if title and not _PRINT_ON:                     # in the docx the numbered Word caption names the figure
        c = c.properties(title=title)
    return (c.configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=DOMAIN,
                             labelFontSize=12, titleFontSize=13, titleFontWeight="normal")
            .configure_view(strokeWidth=0)
            .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFontSize=12)
            .configure_title(color=_TITLE, fontSize=14, anchor="start", font="IBM Plex Sans"))


def assurance_curve_chart(model: dict):
    """Assurance (probability of success) against planned sample size: how much n buys how much
    confidence. The design-stage counterpart of the power curve."""
    if not model or model.get("error") or not model.get("series"):
        return None
    d = pd.DataFrame(model["series"])
    if "assurance" not in d or "n" not in d:
        return None
    line = alt.Chart(d).mark_line(color=TEAL, strokeWidth=2,
                                  point=alt.OverlayMarkDef(color=TEAL)).encode(
        x=alt.X("n:Q", title="planned sample size"),
        y=alt.Y("assurance:Q", title="assurance (probability of success)",
                axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=["n:Q", alt.Tooltip("assurance:Q", format=".1%")])
    return _finish(line, 300, "Assurance vs planned sample size")


def oc_curve_chart(model: dict):
    """Operating characteristics: the GO rate at each TRUE effect. FDA's second pillar -- how the
    design behaves across a plausible range of truths, not just the one you hope for."""
    oc = (model or {}).get("robustness", {}).get("oc")
    if not oc:
        return None
    d = pd.DataFrame(oc)
    if "theta" not in d or "go_rate" not in d:
        return None
    line = alt.Chart(d).mark_line(color=TEAL, strokeWidth=2).encode(
        x=alt.X("theta:Q", title="true effect"),
        y=alt.Y("go_rate:Q", title="probability of a GO", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("theta:Q", format=".3f"), alt.Tooltip("go_rate:Q", format=".1%")])
    return _finish(line, 260, "Operating characteristics (GO rate by true effect)")
