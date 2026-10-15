"""Panel 1 - Inflation vs the Fed.

Question: did interest rates keep up with inflation?

The chart compares the interest rate the US Federal Reserve actually set (the
effective fed funds rate) with the rate a simple policy rule - the Taylor
rule - says it "should" have set, given inflation and unemployment at the
time. When the fed funds line is below the rule, policy was looser than the
rule suggests (the Fed was "behind the curve"); above it, policy was tighter.

The module follows the same layout as every panel:
1. constants
2. data loading
3. model and analysis functions (pandas/NumPy only, no Streamlit, unit tested)
4. chart builder (returns a Plotly figure, no Streamlit)
5. render() - the Streamlit layout: controls, key numbers, chart, insight
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme, ui
from dashboard.data import filter_dates, load_fred, load_recessions, recession_periods

NUMBER = 1
SLUG = "inflation"
TITLE = "Inflation vs the Fed"
OWNER = "Team member 1"  # panel owner: put your name here
QUESTION = "Did interest rates keep up with inflation?"

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# Data: FRED series IDs (each one lives at https://fred.stlouisfed.org/series/<ID>)
SNAPSHOT_NAME = "inflation_macro"  # saved as data/inflation_macro.csv
DATA_START = "1985-01-01"
FED_FUNDS = "FEDFUNDS"     # effective federal funds rate, monthly average, %
UNEMPLOYMENT = "UNRATE"    # unemployment rate, %
NATURAL_RATE = "NROU"      # CBO natural rate of unemployment, % (quarterly)

# The inflation gauges the user can pick from: label -> FRED price index ID
INFLATION_MEASURES = {
    "Headline CPI": "CPIAUCSL",  # all items, what households feel
    "Core CPI": "CPILFESL",      # without food and energy (less noisy)
    "Core PCE": "PCEPILFE",      # the Fed's preferred inflation gauge
}
DEFAULT_MEASURE = "Core PCE"
FRED_SERIES = list(INFLATION_MEASURES.values()) + [FED_FUNDS, UNEMPLOYMENT, NATURAL_RATE]

# Model: Okun's law says each 1 point of unemployment above its natural rate
# costs roughly 2% of output, so output gap = -2 x (unemployment - natural rate).
OKUN_COEFFICIENT = 2.0

# Ready-made rule settings. r_star = neutral real interest rate (%),
# pi_target = inflation target (%), a = weight on inflation above target,
# b = weight on the output gap.
RULE_PRESETS = {
    "Taylor (1993)": {"r_star": 2.0, "pi_target": 2.0, "a": 0.5, "b": 0.5},
    "Balanced approach": {"r_star": 2.0, "pi_target": 2.0, "a": 0.5, "b": 1.0},
}
CUSTOM = "Custom"
PRESET_OPTIONS = list(RULE_PRESETS) + [CUSTOM]
DEFAULT_PRESET = "Taylor (1993)"

# One slider per rule parameter (the keys match the preset dictionaries above)
# Labels are short so the four sliders fit side by side; the help text gives the full meaning.
RULE_SLIDERS = {
    "r_star": {"label": "Neutral rate r\\*", "min": 0.0, "max": 4.0, "step": 0.25, "format": "%.2f%%",
               "help": "Neutral real interest rate r*: the real rate that neither speeds up nor "
                       "slows down the economy."},
    "pi_target": {"label": "Target π\\*", "min": 0.0, "max": 4.0, "step": 0.25, "format": "%.2f%%",
                  "help": "Inflation target π*: the inflation rate the Fed aims for (officially 2%)."},
    "a": {"label": "Weight a (π)", "min": 0.0, "max": 2.0, "step": 0.1, "format": "%.1f",
          "help": "Inflation weight a: how strongly the rule raises rates when inflation π is above target."},
    "b": {"label": "Weight b (gap)", "min": 0.0, "max": 2.0, "step": 0.1, "format": "%.1f",
          "help": "Output-gap weight b: how strongly the rule responds to the economy running hot or cold."},
}

DEFAULT_START_YEAR = 2000
CLOSE_TO_RULE = 0.25  # gaps smaller than this (percentage points) count as "in line with the rule"

# Widget keys (all start with the slug so they are unique across the dashboard)
KEY_MEASURE = f"{SLUG}_measure"
KEY_PRESET = f"{SLUG}_preset"
KEY_PERIOD = f"{SLUG}_period"
KEY_FLOOR = f"{SLUG}_floor"
KEY_SHOW_INFLATION = f"{SLUG}_show_inflation"


def slider_key(parameter: str) -> str:
    """Session-state key of the slider for one rule parameter, e.g. 'inflation_r_star'."""
    return f"{SLUG}_{parameter}"


# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------

def prepare_monthly_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw FRED download into a clean table with exactly one row per month.

    * NROU is quarterly and also contains CBO *projections* for years ahead,
      so the table is cut at the last month that has real (monthly) data.
    * Every calendar month gets a row (``asfreq``), so "12 rows back" is
      always "one year back" when we compute year-on-year inflation.
    * The quarterly NROU value is carried forward to the next two months.
    """
    monthly_columns = [column for column in raw.columns if column != NATURAL_RATE]
    has_actual_data = raw[monthly_columns].notna().any(axis=1)  # True/False for each row
    rows_with_data = raw[has_actual_data]
    if rows_with_data.empty:
        return rows_with_data.copy()

    last_actual_month = rows_with_data.index[-1]
    monthly = raw[raw.index <= last_actual_month].copy()
    monthly = monthly.asfreq("MS")  # "MS" = month start, the date FRED uses
    monthly[NATURAL_RATE] = monthly[NATURAL_RATE].ffill()
    return monthly


def load_data(refresh: bool = False) -> pd.DataFrame:
    """Monthly FRED data for this panel (from data/inflation_macro.csv unless refresh=True)."""
    raw = load_fred(SNAPSHOT_NAME, FRED_SERIES, start=DATA_START, refresh=refresh)
    return prepare_monthly_data(raw)


def refresh_data() -> None:
    """Download fresh data for this panel (used by the refresh button and scripts/update_data.py)."""
    load_data(refresh=True)


# ---------------------------------------------------------------------------
# 3. Model and analysis (no Streamlit here, so it can be unit tested)
# ---------------------------------------------------------------------------

def yoy_inflation(price_index: pd.Series) -> pd.Series:
    """Year-on-year inflation in % from a monthly price index.

    pct_change(12) compares each month with the same month one year earlier.
    fill_method=None keeps a missing month missing instead of quietly
    re-using an old value.
    """
    return price_index.pct_change(periods=12, fill_method=None) * 100


def okun_output_gap(unemployment: float | pd.Series, natural_rate: float | pd.Series,
                    okun_coefficient: float = OKUN_COEFFICIENT) -> float | pd.Series:
    """Output gap in % of potential output, estimated from unemployment (Okun's law).

    Unemployment above its natural rate means the economy is producing less
    than it could, so the gap is negative. Works on numbers or pandas Series.
    """
    return -okun_coefficient * (unemployment - natural_rate)


def taylor_rule(inflation: float | pd.Series, output_gap: float | pd.Series,
                r_star: float, pi_target: float, a: float, b: float,
                floor_at_zero: bool = False) -> float | pd.Series:
    """Interest rate (%) the Taylor rule prescribes.

        rate = r* + inflation + a x (inflation - target) + b x output gap

    r* + inflation is the "neutral" nominal rate. The rule then adds a bit
    more when inflation is above target and when the economy runs hot.
    With ``floor_at_zero`` the result never goes below 0% (it is floored at
    zero), because a central bank can't really set rates much below zero
    (the "zero lower bound"). Works on single numbers or on whole pandas
    Series at once.
    """
    rate = r_star + inflation + a * (inflation - pi_target) + b * output_gap
    if floor_at_zero:
        rate = np.maximum(rate, 0.0)  # NaN (missing data) stays NaN
    return rate


def match_preset(params: dict) -> str:
    """Name of the preset whose values equal ``params``, or "Custom" if none does."""
    for name, preset in RULE_PRESETS.items():
        all_equal = True
        for parameter, preset_value in preset.items():
            if abs(params[parameter] - preset_value) > 1e-6:  # tolerance for float rounding
                all_equal = False
        if all_equal:
            return name
    return CUSTOM


def build_model_table(data: pd.DataFrame, measure: str, params: dict,
                      floor_at_zero: bool = False) -> pd.DataFrame:
    """One row per month with everything the chart needs.

    Columns: fed_funds, inflation, unemployment, natural_rate, output_gap,
    rule_rate (Taylor rule) and policy_gap (fed funds minus rule, in
    percentage points; negative = looser than the rule).
    """
    inflation = yoy_inflation(data[INFLATION_MEASURES[measure]])
    output_gap = okun_output_gap(data[UNEMPLOYMENT], data[NATURAL_RATE])
    rule_rate = taylor_rule(inflation, output_gap, floor_at_zero=floor_at_zero, **params)

    table = pd.DataFrame({
        "fed_funds": data[FED_FUNDS],
        "inflation": inflation,
        "unemployment": data[UNEMPLOYMENT],
        "natural_rate": data[NATURAL_RATE],
        "output_gap": output_gap,
        "rule_rate": rule_rate,
    })
    table["policy_gap"] = table["fed_funds"] - table["rule_rate"]
    return table


def latest_value(series: pd.Series) -> tuple[pd.Timestamp, float] | None:
    """(month, value) of the last non-missing value, or None if there is none."""
    valid = series.dropna()
    if valid.empty:
        return None
    return valid.index[-1], float(valid.iloc[-1])


def change_over_year(series: pd.Series, month: pd.Timestamp) -> float | None:
    """Change in ``series`` from 12 months before ``month`` to ``month`` (None if unknown)."""
    year_earlier = month - pd.DateOffset(years=1)
    if month not in series.index or year_earlier not in series.index:
        return None
    change = series[month] - series[year_earlier]
    if pd.isna(change):
        return None
    return float(change)


def largest_gaps(policy_gap: pd.Series, min_gap: float = 0.0) -> dict:
    """Months when the Fed was furthest below and furthest above the rule.

    Returns {"below": (month, gap) or None, "above": (month, gap) or None}.
    "below" is None if the Fed was never below the rule in the data given
    (and the same for "above"). With ``min_gap`` a gap only counts if it is
    at least that big, e.g. min_gap=0.25 ignores a tiny -0.1 pp.
    """
    gaps = policy_gap.dropna()
    result = {"below": None, "above": None}
    if gaps.empty:
        return result

    lowest_month = gaps.idxmin()
    lowest_gap = float(gaps[lowest_month])
    if lowest_gap < 0 and abs(lowest_gap) >= min_gap:
        result["below"] = (lowest_month, lowest_gap)

    highest_month = gaps.idxmax()
    highest_gap = float(gaps[highest_month])
    if highest_gap > 0 and highest_gap >= min_gap:
        result["above"] = (highest_month, highest_gap)
    return result


def missing_months(series: pd.Series) -> list[pd.Timestamp]:
    """Months with no value *between* the first and last available value (gaps in the line)."""
    valid = series.dropna()
    if valid.empty:
        return []
    inside = series[(series.index >= valid.index[0]) & (series.index <= valid.index[-1])]
    return list(inside[inside.isna()].index)


def month_label(month: pd.Timestamp) -> str:
    """'Jun 2022' style label for a month."""
    return month.strftime("%b %Y")


def rule_name(preset: str) -> str:
    """How the chosen rule is called in sentences, e.g. 'the Taylor (1993) rule'."""
    if preset == CUSTOM:
        return "your custom rule"
    return f"the {preset} rule"


def describe_latest_gap(table: pd.DataFrame, preset: str) -> str:
    """First insight sentence: how the latest fed funds rate compares with the rule."""
    complete = table.dropna(subset=["fed_funds", "rule_rate"])
    if complete.empty:
        return "There isn't enough data in this period to compare the Fed with the rule."

    month = complete.index[-1]
    fed = complete.loc[month, "fed_funds"]
    rule = complete.loc[month, "rule_rate"]
    gap = fed - rule
    opening = f"In {month_label(month)} the fed funds rate ({fed:.2f}%) was"
    if gap <= -CLOSE_TO_RULE:
        return (f"{opening} {abs(gap):.2f} pp below the {rule:.2f}% that {rule_name(preset)} "
                "suggests, so policy was looser than the rule.")
    if gap >= CLOSE_TO_RULE:
        return (f"{opening} {gap:.2f} pp above the {rule:.2f}% that {rule_name(preset)} "
                "suggests, so policy was tighter than the rule.")
    return f"{opening} in line with {rule_name(preset)} ({rule:.2f}%)."


def describe_extremes(table: pd.DataFrame, measure: str, period_label: str) -> str:
    """Second insight sentence: when in the period the Fed was furthest below/above the rule.

    Gaps smaller than CLOSE_TO_RULE are treated as "in line", so a tiny +0.05 pp
    doesn't get reported as the Fed being "above the rule".
    """
    if table["policy_gap"].dropna().empty:
        return f"Over {period_label} there is no month with both the fed funds rate and the rule."

    extremes = largest_gaps(table["policy_gap"], min_gap=CLOSE_TO_RULE)
    below = extremes["below"]
    above = extremes["above"]

    if below is not None:
        month, gap = below
        rule = table.loc[month, "rule_rate"]
        fed = table.loc[month, "fed_funds"]
        inflation = table.loc[month, "inflation"]
        below_text = (f"furthest behind the rule in {month_label(month)} ({gap:+.2f} pp: "
                      f"the rule called for {rule:.2f}% but the fed funds rate was {fed:.2f}%")
        # Being behind the rule does NOT always mean rates were below inflation: the rule
        # also asks for more when the economy runs hot (output gap). So check that month.
        if fed < inflation:
            below_text += f", below {measure} inflation of {inflation:.2f}%"
        elif fed > inflation:
            below_text += f", still above {measure} inflation of {inflation:.2f}%"
        below_text += ")"
    if above is not None:
        month, gap = above
        above_text = f"furthest above it in {month_label(month)} ({gap:+.2f} pp"
        if table.loc[month, "rule_rate"] < 0:
            above_text += ", when the rule called for a negative rate"
        above_text += ")"

    if below is not None and above is not None:
        return f"Over {period_label} the Fed was {below_text} and {above_text}."
    if below is not None:
        return (f"Over {period_label} the Fed was {below_text}, "
                f"and never more than {CLOSE_TO_RULE:.2f} pp above it.")
    if above is not None:
        return (f"Over {period_label} the Fed was never more than {CLOSE_TO_RULE:.2f} pp below "
                f"the rule, and was {above_text}.")
    return f"Over {period_label} the Fed stayed within {CLOSE_TO_RULE:.2f} pp of the rule."


def write_insight(table: pd.DataFrame, measure: str, preset: str, period_label: str) -> str:
    """The 1-2 sentence story shown under the chart, computed from the current data."""
    return describe_latest_gap(table, preset) + " " + describe_extremes(table, measure, period_label)


def pick_highlight(table: pd.DataFrame) -> tuple[pd.Timestamp, float] | None:
    """The month to mark on the chart: the biggest gap between the Fed and the rule.

    Compares the month furthest below the rule with the month furthest above
    it and returns the one further away (in either direction). Gaps under
    CLOSE_TO_RULE are ignored, so None means there is nothing worth marking.
    """
    extremes = largest_gaps(table["policy_gap"], min_gap=CLOSE_TO_RULE)
    below = extremes["below"]
    above = extremes["above"]
    if below is None:
        return above  # may also be None
    if above is None:
        return below
    if abs(above[1]) > abs(below[1]):
        return above
    return below  # on a tie, "behind the rule" is the panel's main question


def make_display_table(table: pd.DataFrame, measure: str) -> pd.DataFrame:
    """The plotted numbers with readable column names, newest month first."""
    # Chart lines first, then the model inputs behind the rule
    readable_names = {
        "fed_funds": "Fed funds rate (%)",
        "rule_rate": "Taylor-rule rate (%)",
        "policy_gap": "Fed minus rule (pp)",
        "inflation": f"{measure} inflation (% YoY)",
        "unemployment": "Unemployment (%)",
        "natural_rate": "Natural rate of unemployment (%)",
        "output_gap": "Output gap (% of potential)",
    }
    display = table[list(readable_names)].rename(columns=readable_names)
    display = display.round(2).sort_index(ascending=False)
    display.index = display.index.strftime("%Y-%m")
    display.index.name = "Month"
    return display


# ---------------------------------------------------------------------------
# 4. Chart (returns a Plotly figure, no Streamlit here)
# ---------------------------------------------------------------------------

def add_gap_marker(fig: go.Figure, table: pd.DataFrame, month: pd.Timestamp, gap: float,
                   y_range: list[float]) -> None:
    """Mark the gap between the fed funds rate and the rule at ``month``.

    A dotted line joins the two lines. The label sits in the empty strip that
    y_axis_range leaves at the top of the chart (or at the bottom when the rule
    is below the Fed), with a thin leader line down (or up) to the rule.
    """
    fed = table.loc[month, "fed_funds"]
    rule = table.loc[month, "rule_rate"]
    fig.add_shape(type="line", x0=month, x1=month, y0=fed, y1=rule,
                  line=dict(color=theme.INK_SECONDARY, width=1, dash="dot"))

    # Near the left or right edge, anchor the text so it doesn't run off the chart
    share_of_period = (month - table.index[0]) / (table.index[-1] - table.index[0])
    if share_of_period < 0.2:
        x_anchor = "left"
    elif share_of_period > 0.8:
        x_anchor = "right"
    else:
        x_anchor = "center"

    if rule > fed:
        label_y, y_anchor = y_range[1], "top"      # label in the strip at the top
    else:
        label_y, y_anchor = y_range[0], "bottom"   # label in the strip at the bottom
    fig.add_annotation(
        x=month, y=rule,                 # the leader line points here...
        ax=0, ay=label_y, ayref="y",     # ...from the label, straight above/below
        showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=theme.INK_MUTED,
        text=f"<b>{gap:+.2f} pp</b> Fed vs rule ({month_label(month)})",
        xanchor=x_anchor, yanchor=y_anchor,
        font=dict(size=11, color=theme.INK_SECONDARY),
    )


def recessions_in_window(recessions: list, start: pd.Timestamp, end: pd.Timestamp) -> list:
    """The (start, end) recession periods that overlap the chart window start..end.

    Strict < and > so a recession that only touches the edge of the window
    (which would be a band of zero width) is left out.
    """
    visible = []
    for recession_start, recession_end in recessions:
        if recession_start < end and recession_end > start:
            visible.append((recession_start, recession_end))
    return visible


def y_axis_range(lines: list[pd.Series], room_above: bool, room_below: bool) -> list[float]:
    """Y-axis range that fits every line, plus extra empty space for a label above or below."""
    values = pd.concat(lines).dropna()
    if values.empty:
        return [-1.0, 1.0]
    low = min(float(values.min()), 0.0)  # always show the 0% line
    high = max(float(values.max()), 0.0)
    span = max(high - low, 1.0)
    padding = 0.05 * span
    label_space = 0.12 * span
    top = high + padding + (label_space if room_above else 0.0)
    bottom = low - padding - (label_space if room_below else 0.0)
    return [bottom, top]


def build_chart(table: pd.DataFrame, measure: str, preset: str, show_inflation: bool,
                recessions: list, highlight: tuple[pd.Timestamp, float] | None = None) -> go.Figure:
    """Line chart: fed funds rate vs the Taylor-rule rate (and inflation), all in % on one axis."""
    fig = go.Figure()

    # Fed funds: what the Fed actually did (solid blue)
    fig.add_trace(go.Scatter(
        x=table.index, y=table["fed_funds"], name="Fed funds rate", mode="lines",
        line=dict(color=theme.BLUE, width=2),
        hovertemplate="Fed funds rate: %{y:.2f}%<extra></extra>",
    ))
    # Rule: what the model prescribes (dashed orange, because it is a model, not data)
    fig.add_trace(go.Scatter(
        x=table.index, y=table["rule_rate"], name=f"Rule: {preset}", mode="lines",
        line=dict(color=theme.ORANGE, width=2, dash="dash"),
        customdata=table["policy_gap"],
        hovertemplate="Rule: %{y:.2f}% (Fed %{customdata:+.2f} pp vs rule)<extra></extra>",
    ))
    visible_lines = [table["fed_funds"], table["rule_rate"]]
    # Inflation keeps its aqua colour even when it is the only optional line
    if show_inflation:
        fig.add_trace(go.Scatter(
            x=table.index, y=table["inflation"], name=f"{measure} inflation", mode="lines",
            line=dict(color=theme.AQUA, width=2),
            hovertemplate=f"{measure} inflation: %{{y:.2f}}% YoY<extra></extra>",
        ))
        visible_lines.append(table["inflation"])

    start, end = table.index[0], table.index[-1]
    visible_recessions = recessions_in_window(recessions, start, end)
    if visible_recessions:
        theme.add_recession_bands(fig, visible_recessions, start, end)
        # Empty trace so the grey bands get a legend entry (only when a band is drawn)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], name="US recession", mode="markers", hoverinfo="skip",
            marker=dict(symbol="square", size=12, color=theme.RECESSION_FILL,
                        line=dict(color=theme.AXIS, width=1)),
        ))

    fig.add_hline(y=0, line_width=1, line_color=theme.INK_MUTED)  # hairline at 0%

    # Leave an empty strip for the gap label: at the top if the Fed was below the rule
    # (the rule is the upper end of the gap), at the bottom if it was above
    show_marker = highlight is not None and len(table) > 1
    label_above = show_marker and highlight[1] < 0
    label_below = show_marker and highlight[1] > 0
    y_range = y_axis_range(visible_lines, label_above, label_below)
    if show_marker:
        add_gap_marker(fig, table, highlight[0], highlight[1], y_range)

    theme.style_figure(fig, y_title="Percent per year")
    # automargin makes room for the tick labels so they aren't cut off
    fig.update_xaxes(range=[start, end], hoverformat="%b %Y", automargin=True)
    fig.update_yaxes(ticksuffix="%", automargin=True, range=y_range)
    fig.update_layout(legend=dict(traceorder="normal"), margin=dict(t=44))
    return fig


# ---------------------------------------------------------------------------
# 5. Streamlit layout
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_data() -> pd.DataFrame:
    """load_data() cached, so the CSV is read once and not on every click."""
    return load_data()


@st.cache_data(show_spinner=False)
def _cached_recession_periods() -> list:
    """NBER recession (start, end) dates, cached."""
    return recession_periods(load_recessions())


def _apply_preset() -> None:
    """Callback: when a preset is picked, move the four sliders to its values."""
    preset = st.session_state[KEY_PRESET]
    if preset in RULE_PRESETS:
        for parameter, value in RULE_PRESETS[preset].items():
            st.session_state[slider_key(parameter)] = value


def _current_params() -> dict:
    """The four rule parameters as currently set on the sliders."""
    params = {}
    for parameter in RULE_SLIDERS:
        params[parameter] = float(st.session_state[slider_key(parameter)])
    return params


def _on_slider_change() -> None:
    """Callback: when a slider moves, show which preset (if any) the values now match."""
    st.session_state[KEY_PRESET] = match_preset(_current_params())


def _init_rule_state() -> None:
    """Put default preset and slider values in session_state the first time the panel runs."""
    if KEY_PRESET not in st.session_state:
        st.session_state[KEY_PRESET] = DEFAULT_PRESET
    # Sliders start at the chosen preset's values (the default preset if "Custom")
    preset = st.session_state[KEY_PRESET]
    defaults = RULE_PRESETS.get(preset, RULE_PRESETS[DEFAULT_PRESET])
    for parameter in RULE_SLIDERS:
        if slider_key(parameter) not in st.session_state:
            st.session_state[slider_key(parameter)] = defaults[parameter]


def _draw_controls(years: list[int]) -> dict:
    """Draw all the controls (3 compact rows) and return the chosen values."""
    _init_rule_state()

    row1 = st.columns([1, 1, 1.4])
    with row1[0]:
        measure = st.selectbox(
            "Inflation measure", list(INFLATION_MEASURES),
            index=list(INFLATION_MEASURES).index(DEFAULT_MEASURE), key=KEY_MEASURE,
            help="Core measures leave out food and energy. Core PCE is the gauge the Fed targets.",
        )
    with row1[1]:
        preset = st.selectbox(
            "Policy rule", PRESET_OPTIONS, key=KEY_PRESET, on_change=_apply_preset,
            help="Pick a published rule, or move the sliders to build your own (Custom).",
        )
    with row1[2]:
        default_start = min(max(DEFAULT_START_YEAR, years[0]), years[-1])
        period = st.select_slider("Period", options=years, value=(default_start, years[-1]), key=KEY_PERIOD)

    row2 = st.columns(len(RULE_SLIDERS))
    for column, parameter in zip(row2, RULE_SLIDERS):
        spec = RULE_SLIDERS[parameter]
        with column:
            st.slider(spec["label"], min_value=spec["min"], max_value=spec["max"], step=spec["step"],
                      format=spec["format"], key=slider_key(parameter), on_change=_on_slider_change,
                      help=spec["help"])

    row3 = st.columns(2)
    with row3[0]:
        floor_at_zero = st.toggle(
            "Floor at zero", value=False, key=KEY_FLOOR,
            help="The Fed can't cut much below 0% (the zero lower bound), so the rule never goes below 0%.",
        )
    with row3[1]:
        show_inflation = st.toggle("Show inflation", value=True, key=KEY_SHOW_INFLATION)

    start_year, end_year = min(period), max(period)  # safe even if they arrive reversed
    return {
        "measure": measure,
        "preset": preset,
        "params": _current_params(),
        "start_year": start_year,
        "end_year": end_year,
        "floor_at_zero": floor_at_zero,
        "show_inflation": show_inflation,
    }


def _draw_kpis(window: pd.DataFrame, full_table: pd.DataFrame, choices: dict) -> None:
    """Three key numbers - inflation, the fed funds rate and the rule rate - all for the SAME month.

    That month is the latest one with both the fed funds rate and the rule, so
    "Fed vs rule" is exactly the fed funds KPI minus the rule KPI. (Core PCE is
    published about a month after the fed funds rate, so a newer fed funds
    value can exist; it is mentioned in the help text.)
    """
    col1, col2, col3 = st.columns(3)
    complete = window.dropna(subset=["fed_funds", "rule_rate"])
    if complete.empty:
        col1.metric("Inflation", "n/a")
        col2.metric("Fed funds", "n/a")
        col3.metric("Rule rate", "n/a")
        return

    month = complete.index[-1]
    when = month_label(month)
    measure = choices["measure"]
    target = choices["params"]["pi_target"]
    inflation_now = complete.loc[month, "inflation"]  # never missing when the rule exists
    fed = complete.loc[month, "fed_funds"]
    rule = complete.loc[month, "rule_rate"]
    gap = complete.loc[month, "policy_gap"]

    # Labels and deltas are kept short so they fit when the panel is only ~550 px wide
    with col1:
        st.metric(f"Inflation, {when}", f"{inflation_now:.2f}%",
                  delta=f"{inflation_now - target:+.2f} pp vs target", delta_color="inverse",
                  help=f"{measure} inflation: year-on-year change in the price index. The delta compares "
                       f"it with the {target:.2f}% inflation target (red = above target).")

    with col2:
        change = change_over_year(full_table["fed_funds"], month)
        delta = None if change is None else f"{change:+.2f} pp vs 1y ago"
        help_text = ("Effective federal funds rate, monthly average, for the same month as the other "
                     "two numbers. The delta is the change over the previous 12 months.")
        newest = latest_value(window["fed_funds"])
        if newest is not None and newest[0] > month:
            help_text += f" Newer value: {newest[1]:.2f}% in {month_label(newest[0])}."
        st.metric(f"Fed funds, {when}", f"{fed:.2f}%", delta=delta, delta_color="off", help=help_text)

    with col3:
        st.metric(f"Rule rate, {when}", f"{rule:.2f}%",
                  delta=f"{gap:+.2f} pp Fed vs rule", delta_color="off",
                  help=f"The rate {rule_name(choices['preset'])} prescribes. Delta: fed funds "
                       f"({fed:.2f}%) minus the rule ({rule:.2f}%); negative = policy looser than the rule.")


def _method_markdown(data: pd.DataFrame, choices: dict) -> str:
    """Text for the 'Method & data sources' expander (uses the current settings)."""
    params = choices["params"]
    measure = choices["measure"]
    floor_text = "on (the rule never goes below 0%)" if choices["floor_at_zero"] else "off"

    gaps = []
    for column in [INFLATION_MEASURES[measure], FED_FUNDS, UNEMPLOYMENT]:
        for month in missing_months(data[column]):
            gaps.append(f"{column} {month_label(month)}")
    if gaps:
        gaps_text = "Months missing in the source data (small breaks in the lines): " + ", ".join(gaps) + "."
    else:
        gaps_text = "No months are missing in the source data."

    links = []
    for series_id in FRED_SERIES:
        links.append(f"[{series_id}](https://fred.stlouisfed.org/series/{series_id})")

    return f"""
**Taylor rule.** A simple rule of thumb for where the Fed's interest rate "should" be:

$$i = r^* + \\pi + a(\\pi - \\pi^*) + b \\times \\text{{gap}}$$

- $\\pi$ = {measure} inflation over the last 12 months (`pct_change(12)` of the price index).
- Current settings: $r^*$ = {params["r_star"]:.2f}%, $\\pi^*$ = {params["pi_target"]:.2f}%,
  a = {params["a"]:.1f}, b = {params["b"]:.1f}; floor at zero is {floor_text}.
- **Output gap** from Okun's law: gap = −{OKUN_COEFFICIENT:.0f} × (unemployment − natural rate).
  The natural rate (CBO estimate) is quarterly, so each value is carried forward to the next two months;
  CBO projections beyond the latest real data are dropped.
- **Fed vs rule** = fed funds rate − rule rate, in percentage points (pp). Negative means
  policy was looser than the rule suggests ("behind the curve").

**Sources:** FRED, Federal Reserve Bank of St. Louis: {", ".join(links)}.
Grey bands: NBER US recessions ([USREC](https://fred.stlouisfed.org/series/USREC)). {gaps_text}
"""


def render() -> None:
    """Draw the whole panel inside the current Streamlit container."""
    ui.panel_header(NUMBER, TITLE, OWNER, QUESTION)

    try:
        data = _cached_data()
    except Exception as exc:  # no snapshot and no internet, or a broken CSV
        ui.no_data_message("Inflation and interest-rate", exc)
        return
    if data.empty:
        ui.no_data_message("Inflation and interest-rate", ValueError("the data file has no rows"))
        return

    try:
        recessions = _cached_recession_periods()
    except Exception:  # shading is nice to have, the panel works without it
        recessions = []

    # Year-on-year inflation needs one year of history, so the first full year is skipped
    years = list(range(data.index[0].year + 1, data.index[-1].year + 1))
    if len(years) < 2:
        st.info("Not enough history in the data file to draw this panel. Try refreshing the data.")
        return

    choices = _draw_controls(years)

    # Compute on the full history first (YoY needs the year before the window), then cut the window
    full_table = build_model_table(data, choices["measure"], choices["params"], choices["floor_at_zero"])
    window = filter_dates(full_table, f"{choices['start_year']}-01-01", f"{choices['end_year']}-12-31")
    # Empty months stay in the table, so the lines show a break there instead of a jump
    if window.dropna(how="all").empty:
        st.info("There is no data for the selected period. Try a wider range of years.")
        return

    _draw_kpis(window, full_table, choices)

    highlight = pick_highlight(window)
    fig = build_chart(window, choices["measure"], choices["preset"], choices["show_inflation"],
                      recessions, highlight)
    ui.show_chart(fig, key=f"{SLUG}_chart")

    if choices["start_year"] == choices["end_year"]:
        period_label = str(choices["start_year"])
    else:
        period_label = f"{choices['start_year']}–{choices['end_year']}"
    ui.insight(write_insight(window, choices["measure"], choices["preset"], period_label))

    ui.details(_method_markdown(data, choices), make_display_table(window, choices["measure"]),
               key=SLUG, file_name=f"{SLUG}_taylor_rule.csv")
