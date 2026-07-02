"""Auto-generate a sensible Altair chart from an agent result DataFrame.

Deterministic picker (no LLM): choose a measure column and an axis from the result's shape —
a time column → line chart; a category column → horizontal bar sorted by the measure. Returns
None when a chart wouldn't help (single value, only ids, etc.). Themed to the app's palette.
"""
from __future__ import annotations
import re

import altair as alt
import pandas as pd

TEAL = "#4fd1c5"
_MUTED = "#8ea0b0"
_GRID = "#1a2531"
_DOMAIN = "#20303f"

# measure preference, most-interesting-to-chart first (a rate beats its denominator)
_TIERS = [
    re.compile(r"(rate|pct|percent|prevalence|proportion|ratio)", re.I),
    re.compile(r"(avg|mean|median)", re.I),
    re.compile(r"(cost|amount|charge|price|expense|income|revenue)", re.I),
    re.compile(r"(count|num|total|sum|_n$|^n$)", re.I),   # counts/denominators last
]
_ID = re.compile(r"(_id$|^id$|code$|zip|latitude|longitude|_seq$)", re.I)
_TIME = re.compile(r"(date|year|month|day|_at$|start|stop|time)", re.I)


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


def _theme(chart, rows: int, horizontal: bool):
    h = min(380, 70 + 26 * rows) if horizontal else 300
    return (chart.properties(height=h, background="transparent")
            .configure_axis(labelColor=_MUTED, titleColor=_MUTED, gridColor=_GRID, domainColor=_DOMAIN)
            .configure_view(strokeWidth=0))


def build_chart(df: pd.DataFrame, question: str = ""):
    """Return an Altair chart for the result, or None if a chart wouldn't add anything."""
    if df is None or len(df) < 2:
        return None
    y = _pick_measure(df)
    if y is None:
        return None

    time_cols = [c for c in df.columns
                 if _TIME.search(str(c)) or pd.api.types.is_datetime64_any_dtype(df[c])]
    cats = [c for c in _cats(df) if not _ID.search(str(c))]

    if time_cols:                                   # time series → line
        x = time_cols[0]
        d = df[[x, y]].dropna()
        is_dt = pd.api.types.is_datetime64_any_dtype(df[x])
        chart = alt.Chart(d).mark_line(point=True, color=TEAL).encode(
            x=alt.X(f"{x}:{'T' if is_dt else 'O'}", title=str(x)),
            y=alt.Y(f"{y}:Q", title=str(y)),
            tooltip=list(d.columns),
        )
        return _theme(chart, len(d), horizontal=False)

    if cats:                                        # category × measure → sorted horizontal bar
        x = cats[0]
        d = df[[x, y]].dropna().sort_values(y, ascending=False).head(15)
        chart = alt.Chart(d).mark_bar(color=TEAL).encode(
            x=alt.X(f"{y}:Q", title=str(y)),
            y=alt.Y(f"{x}:N", sort="-x", title=None),
            tooltip=list(d.columns),
        )
        return _theme(chart, len(d), horizontal=True)

    return None
