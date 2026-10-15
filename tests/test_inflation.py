"""Tests for Panel 1 - Inflation vs the Fed (no internet needed).

The model tests use tiny made-up data with answers worked out by hand.
The AppTest tests run the real panel on the CSV snapshot in data/ and move
every control to check nothing crashes.
"""

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import theme
from dashboard.data import DataUnavailableError
from dashboard.panels import inflation

TAYLOR_1993 = {"r_star": 2.0, "pi_target": 2.0, "a": 0.5, "b": 0.5}


def months(n, start="2020-01-01"):
    """n consecutive month-start dates."""
    return pd.date_range(start, periods=n, freq="MS")


# ---------------------------------------------------------------------------
# Taylor rule
# ---------------------------------------------------------------------------

def test_taylor_rule_hand_computed():
    # 2 + 4 + 0.5 x (4 - 2) + 0.5 x 1 = 2 + 4 + 1 + 0.5 = 7.5
    assert inflation.taylor_rule(4.0, 1.0, **TAYLOR_1993) == pytest.approx(7.5)


def test_taylor_rule_on_target_with_no_gap_gives_neutral_rate():
    # Inflation on target and no output gap -> r* + pi = 2 + 2 = 4
    assert inflation.taylor_rule(2.0, 0.0, **TAYLOR_1993) == pytest.approx(4.0)


def test_taylor_rule_works_on_whole_series():
    pi = pd.Series([1.0, 2.0, 6.0])
    gap = pd.Series([-2.0, 0.0, 2.0])
    balanced = inflation.RULE_PRESETS["Balanced approach"]
    # 2 + 1 + 0.5 x (-1) + 1 x (-2) = 0.5 | 2 + 2 + 0 + 0 = 4 | 2 + 6 + 0.5 x 4 + 1 x 2 = 12
    result = inflation.taylor_rule(pi, gap, **balanced)
    assert list(result) == pytest.approx([0.5, 4.0, 12.0])


def test_zero_lower_bound_floor():
    # Deep recession: 2 + 0 + 0.5 x (0 - 2) + 0.5 x (-10) = -4
    assert inflation.taylor_rule(0.0, -10.0, **TAYLOR_1993) == pytest.approx(-4.0)
    assert inflation.taylor_rule(0.0, -10.0, **TAYLOR_1993, floor_at_zero=True) == 0.0
    # The floor leaves positive rates alone and keeps missing values missing
    pi = pd.Series([0.0, 4.0, np.nan])
    gap = pd.Series([-10.0, 1.0, 0.0])
    floored = inflation.taylor_rule(pi, gap, **TAYLOR_1993, floor_at_zero=True)
    assert floored.iloc[0] == 0.0
    assert floored.iloc[1] == pytest.approx(7.5)
    assert np.isnan(floored.iloc[2])


# ---------------------------------------------------------------------------
# Inflation and output gap
# ---------------------------------------------------------------------------

def test_yoy_inflation():
    # Price index 100 for a year, then 103 -> 3% inflation; the first year has no year-ago value
    prices = pd.Series([100.0] * 12 + [103.0] * 12, index=months(24))
    yoy = inflation.yoy_inflation(prices)
    assert yoy.iloc[:12].isna().all()
    assert yoy.iloc[12:].tolist() == pytest.approx([3.0] * 12)


def test_yoy_inflation_keeps_missing_months_missing():
    prices = pd.Series([100.0] * 12 + [110.0, np.nan, 110.0], index=months(15))
    yoy = inflation.yoy_inflation(prices)
    assert yoy.iloc[12] == pytest.approx(10.0)
    assert np.isnan(yoy.iloc[13])  # not filled with an old value
    assert yoy.iloc[14] == pytest.approx(10.0)


def test_okun_output_gap():
    # Unemployment 1 point above the natural rate -> output 2% below potential
    assert inflation.okun_output_gap(5.0, 4.0) == pytest.approx(-2.0)
    # Unemployment below the natural rate -> economy running hot (positive gap)
    assert inflation.okun_output_gap(4.0, 4.5) == pytest.approx(1.0)


def raw_fred_frame():
    """Six months of monthly data plus quarterly NROU that runs on into 'projections'."""
    index = pd.DatetimeIndex(["2024-01-01", "2024-02-01", "2024-04-01", "2024-05-01",
                              "2024-06-01", "2024-07-01", "2024-10-01"])
    raw = pd.DataFrame(index=index)
    for series_id in inflation.INFLATION_MEASURES.values():
        raw[series_id] = [100.0, 101.0, 102.0, 103.0, 104.0, np.nan, np.nan]
    raw["FEDFUNDS"] = [5.0, 5.0, 5.0, 5.0, 5.0, np.nan, np.nan]
    raw["UNRATE"] = [5.0, 5.0, 5.0, 5.0, 5.0, np.nan, np.nan]
    # Quarterly: Jan and Apr are real estimates, Jul and Oct are CBO projections
    raw["NROU"] = [4.0, np.nan, 4.5, np.nan, np.nan, 4.6, 4.7]
    return raw  # note: March is missing completely


def test_prepare_monthly_data_cuts_projections_and_fills_quarterly_nrou():
    monthly = inflation.prepare_monthly_data(raw_fred_frame())
    # Ends at the last month with real monthly data (June), not at the projections
    assert monthly.index[-1] == pd.Timestamp("2024-06-01")
    # Every month has a row, including the missing March
    assert list(monthly.index) == list(months(6, "2024-01-01"))
    assert np.isnan(monthly.loc["2024-03-01", "FEDFUNDS"])
    # Quarterly NROU carried forward: Jan value for Jan-Mar, Apr value for Apr-Jun
    assert monthly["NROU"].tolist() == [4.0, 4.0, 4.0, 4.5, 4.5, 4.5]


def test_okun_gap_with_forward_filled_nrou():
    monthly = inflation.prepare_monthly_data(raw_fred_frame())
    gap = inflation.okun_output_gap(monthly["UNRATE"], monthly["NROU"])
    # -2 x (5 - 4) = -2 for Jan-Feb; March has no unemployment; -2 x (5 - 4.5) = -1 for Apr-Jun
    assert gap.loc["2024-01-01"] == pytest.approx(-2.0)
    assert gap.loc["2024-02-01"] == pytest.approx(-2.0)
    assert np.isnan(gap.loc["2024-03-01"])
    assert gap.loc["2024-04-01":"2024-06-01"].tolist() == pytest.approx([-1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------
# Full model table
# ---------------------------------------------------------------------------

def synthetic_data(n_months=13):
    """Monthly data where every number in the last month is easy to check by hand."""
    data = pd.DataFrame(index=months(n_months, "2023-01-01"))
    cpi = [100.0] * 12 + [104.0] * (n_months - 12)
    data["CPIAUCSL"] = cpi          # headline: 4% inflation in month 13
    data["CPILFESL"] = cpi
    data["PCEPILFE"] = [100.0] * 12 + [103.0] * (n_months - 12)  # core PCE: 3%
    data["FEDFUNDS"] = 5.0
    data["UNRATE"] = 4.5
    data["NROU"] = 5.0              # gap = -2 x (4.5 - 5) = +1
    return data


def test_build_model_table_hand_computed():
    table = inflation.build_model_table(synthetic_data(), "Headline CPI", TAYLOR_1993)
    last = table.iloc[-1]
    assert last["inflation"] == pytest.approx(4.0)
    assert last["output_gap"] == pytest.approx(1.0)
    assert last["rule_rate"] == pytest.approx(7.5)       # same numbers as the hand-computed rule
    assert last["policy_gap"] == pytest.approx(5.0 - 7.5)  # Fed 2.5 pp below the rule
    # The first year has no inflation yet, so no rule either
    assert table["rule_rate"].iloc[:12].isna().all()


def test_build_model_table_uses_the_chosen_measure():
    table = inflation.build_model_table(synthetic_data(), "Core PCE", TAYLOR_1993)
    # 2 + 3 + 0.5 x (3 - 2) + 0.5 x 1 = 6
    assert table["rule_rate"].iloc[-1] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def test_preset_values():
    assert inflation.RULE_PRESETS["Taylor (1993)"] == TAYLOR_1993
    assert inflation.RULE_PRESETS["Balanced approach"] == {"r_star": 2.0, "pi_target": 2.0, "a": 0.5, "b": 1.0}
    assert inflation.PRESET_OPTIONS == ["Taylor (1993)", "Balanced approach", "Custom"]
    # Every preset sets exactly the parameters that have sliders
    for preset in inflation.RULE_PRESETS.values():
        assert set(preset) == set(inflation.RULE_SLIDERS)


def test_match_preset():
    assert inflation.match_preset(TAYLOR_1993) == "Taylor (1993)"
    assert inflation.match_preset({**TAYLOR_1993, "b": 1.0}) == "Balanced approach"
    assert inflation.match_preset({**TAYLOR_1993, "b": 0.7}) == "Custom"
    # Slider arithmetic can give values like 0.30000000000000004 - still a match
    assert inflation.match_preset({**TAYLOR_1993, "a": 0.1 + 0.4}) == "Taylor (1993)"


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def test_largest_gaps():
    gaps = pd.Series([0.5, -3.0, 1.2, np.nan, -1.0], index=months(5))
    result = inflation.largest_gaps(gaps)
    assert result["below"] == (pd.Timestamp("2020-02-01"), -3.0)
    assert result["above"] == (pd.Timestamp("2020-03-01"), 1.2)


def test_largest_gaps_handles_no_event():
    never_below = inflation.largest_gaps(pd.Series([0.5, 2.0], index=months(2)))
    assert never_below["below"] is None
    assert never_below["above"] == (pd.Timestamp("2020-02-01"), 2.0)

    nothing = inflation.largest_gaps(pd.Series([np.nan, np.nan], index=months(2)))
    assert nothing == {"below": None, "above": None}


def test_largest_gaps_ignores_gaps_smaller_than_min_gap():
    gaps = pd.Series([-0.1, 0.3, -2.0], index=months(3))
    result = inflation.largest_gaps(gaps, min_gap=0.25)
    assert result["below"] == (pd.Timestamp("2020-03-01"), -2.0)
    assert result["above"] == (pd.Timestamp("2020-02-01"), 0.3)
    # -0.1 and +0.2 are both too small to count
    tiny = inflation.largest_gaps(pd.Series([-0.1, 0.2], index=months(2)), min_gap=0.25)
    assert tiny == {"below": None, "above": None}


def test_latest_value_and_change_over_year():
    series = pd.Series([1.0] + [np.nan] * 11 + [3.5, np.nan], index=months(14))
    assert inflation.latest_value(series) == (pd.Timestamp("2021-01-01"), 3.5)
    assert inflation.latest_value(pd.Series([np.nan], index=months(1))) is None
    assert inflation.change_over_year(series, pd.Timestamp("2021-01-01")) == pytest.approx(2.5)
    assert inflation.change_over_year(series, pd.Timestamp("2020-06-01")) is None  # no year-ago month


def test_missing_months_only_counts_gaps_inside_the_data():
    series = pd.Series([np.nan, 1.0, np.nan, 2.0, np.nan], index=months(5))
    assert inflation.missing_months(series) == [pd.Timestamp("2020-03-01")]


def gap_table(fed, rule, inflation_rate=3.0):
    """A small model table with chosen fed funds and rule rates."""
    index = months(len(fed))
    table = pd.DataFrame({"fed_funds": fed, "rule_rate": rule}, index=index)
    table["inflation"] = inflation_rate
    table["policy_gap"] = table["fed_funds"] - table["rule_rate"]
    return table


def test_insight_below_and_above_the_rule():
    table = gap_table(fed=[1.0, 4.0, 2.0], rule=[5.0, 2.0, 3.0])  # inflation 3% every month
    text = inflation.write_insight(table, "Core PCE", "Taylor (1993)", "2020")
    assert "In Mar 2020 the fed funds rate (2.00%) was 1.00 pp below the 3.00%" in text
    assert "looser than the rule" in text
    # Jan: fed funds 1% is below 3% inflation, so the sentence may say so
    assert ("furthest behind the rule in Jan 2020 (-4.00 pp: the rule called for 5.00% but the "
            "fed funds rate was 1.00%, below Core PCE inflation of 3.00%)") in text
    assert "furthest above it in Feb 2020 (+2.00 pp)" in text


def test_insight_behind_the_rule_but_above_inflation():
    # Regression test: the Fed can be behind the rule even with rates ABOVE inflation
    # (e.g. a hot economy). The text must not claim rates fell short of inflation.
    table = gap_table(fed=[6.0, 6.0], rule=[8.0, 6.1], inflation_rate=3.0)
    text = inflation.write_insight(table, "Core PCE", "Taylor (1993)", "2020")
    assert ("furthest behind the rule in Jan 2020 (-2.00 pp: the rule called for 8.00% but the "
            "fed funds rate was 6.00%, still above Core PCE inflation of 3.00%)") in text
    assert "below Core PCE inflation" not in text
    assert "only" not in text


def test_insight_latest_above_the_rule_and_never_clearly_above():
    # Latest month: Fed 0.5 pp above the rule -> "tighter" branch
    above = inflation.write_insight(gap_table(fed=[1.0, 5.0], rule=[4.0, 4.5]),
                                    "Core PCE", "Taylor (1993)", "2020")
    assert "In Feb 2020 the fed funds rate (5.00%) was 0.50 pp above the 4.50%" in above
    assert "tighter than the rule" in above

    # Clearly below once (-3 pp) and never clearly above (-0.1 pp only)
    below_only = inflation.write_insight(gap_table(fed=[1.0, 2.0], rule=[4.0, 2.1]),
                                         "Core PCE", "Taylor (1993)", "2020")
    assert "furthest behind the rule in Jan 2020 (-3.00 pp" in below_only
    assert "and never more than 0.25 pp above it." in below_only


def test_insight_never_below_and_negative_rule():
    table = gap_table(fed=[0.1, 3.0], rule=[-2.0, 2.9])
    text = inflation.write_insight(table, "Core CPI", "Custom", "2020")
    assert "in line with your custom rule" in text  # 0.1 pp gap counts as in line
    assert "never more than 0.25 pp below the rule" in text
    assert "furthest above it in Jan 2020 (+2.10 pp, when the rule called for a negative rate)" in text


def test_insight_when_the_fed_tracks_the_rule():
    table = gap_table(fed=[3.0, 3.1], rule=[3.1, 2.95])
    text = inflation.write_insight(table, "Core PCE", "Taylor (1993)", "2020")
    assert "stayed within 0.25 pp of the rule" in text


def test_insight_without_any_complete_month():
    table = gap_table(fed=[np.nan, np.nan], rule=[1.0, np.nan])
    text = inflation.write_insight(table, "Core PCE", "Taylor (1993)", "2020")
    assert "isn't enough data" in text


def test_pick_highlight_marks_the_biggest_gap_either_way():
    # -4 below vs +2 above -> the shortfall is bigger
    assert inflation.pick_highlight(gap_table([1.0, 4.0], [5.0, 2.0]))[1] == pytest.approx(-4.0)
    # Regression test: +3 above vs -0.7 below -> the bigger "above" gap is marked
    month, gap = inflation.pick_highlight(gap_table([4.0, 1.0], [1.0, 1.7]))
    assert month == pd.Timestamp("2020-01-01")
    assert gap == pytest.approx(3.0)
    # Only above the rule
    assert inflation.pick_highlight(gap_table([4.0, 3.0], [2.0, 3.0]))[1] == pytest.approx(2.0)
    assert inflation.pick_highlight(gap_table([3.0], [3.1])) is None  # too small to point out


def test_make_display_table():
    table = inflation.build_model_table(synthetic_data(14), "Core PCE", TAYLOR_1993)
    display = inflation.make_display_table(table, "Core PCE")
    assert display.index[0] == "2024-02"  # newest month first
    assert display.index.name == "Month"
    assert "Taylor-rule rate (%)" in display.columns
    assert "Core PCE inflation (% YoY)" in display.columns


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def test_chart_colours_are_fixed_per_series():
    table = inflation.build_model_table(synthetic_data(24), "Core PCE", TAYLOR_1993)
    recessions = [(pd.Timestamp("2023-03-01"), pd.Timestamp("2023-06-01"))]

    with_inflation = inflation.build_chart(table, "Core PCE", "Taylor (1993)", True, recessions)
    colours = {trace.name: trace.line.color for trace in with_inflation.data if trace.mode == "lines"}
    assert colours == {"Fed funds rate": theme.BLUE, "Rule: Taylor (1993)": theme.ORANGE,
                       "Core PCE inflation": theme.AQUA}
    assert "US recession" in [trace.name for trace in with_inflation.data]

    # Hiding inflation must not repaint the other lines
    without = inflation.build_chart(table, "Core PCE", "Taylor (1993)", False, [])
    colours = {trace.name: trace.line.color for trace in without.data}
    assert colours == {"Fed funds rate": theme.BLUE, "Rule: Taylor (1993)": theme.ORANGE}
    # One y-axis only
    assert "yaxis2" not in without.layout.to_plotly_json()


def test_recessions_in_window():
    recessions = [(pd.Timestamp("2001-03-01"), pd.Timestamp("2001-12-01")),
                  (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-07-01")),
                  (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-05-01"))]
    window = inflation.recessions_in_window(recessions, pd.Timestamp("2009-01-01"), pd.Timestamp("2019-12-01"))
    assert window == [recessions[1]]  # only the 2008-09 recession overlaps
    # A recession that ends exactly where the window starts doesn't overlap (zero-width band)
    assert inflation.recessions_in_window(recessions, pd.Timestamp("2009-07-01"), pd.Timestamp("2019-12-01")) == []


def test_no_recession_legend_without_a_recession_band():
    # Regression test: 'US recession' must not be in the legend when no band is drawn
    table = inflation.build_model_table(synthetic_data(24), "Core PCE", TAYLOR_1993)  # 2023-2024
    outside = [(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-05-01"))]
    fig = inflation.build_chart(table, "Core PCE", "Taylor (1993)", True, outside)
    assert "US recession" not in [trace.name for trace in fig.data]
    assert [shape for shape in fig.layout.shapes if shape.type == "rect"] == []

    inside = [(pd.Timestamp("2023-03-01"), pd.Timestamp("2023-06-01"))]
    fig = inflation.build_chart(table, "Core PCE", "Taylor (1993)", True, outside + inside)
    assert "US recession" in [trace.name for trace in fig.data]
    assert len([shape for shape in fig.layout.shapes if shape.type == "rect"]) == 1


# ---------------------------------------------------------------------------
# The panel in a (headless) Streamlit app
# ---------------------------------------------------------------------------

def _panel_app():
    from dashboard.panels import inflation

    inflation.render()


def run_panel():
    at = AppTest.from_function(_panel_app, default_timeout=120)
    at.run()
    assert_no_errors(at)
    return at


def assert_no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_panel_renders_from_snapshot():
    at = run_panel()
    assert len(at.metric) == 3
    assert at.selectbox(key="inflation_preset").value == "Taylor (1993)"
    assert at.slider(key="inflation_b").value == 0.5
    widgets = list(at.selectbox) + list(at.slider) + list(at.select_slider) + list(at.toggle)
    assert len(widgets) == 9
    for widget in widgets:
        assert widget.key.startswith("inflation_")
    assert "pp" in at.info[0].value  # the insight is shown


def test_kpis_use_one_month_so_the_numbers_add_up():
    # Regression test: core PCE lags the fed funds rate by a month. All three KPIs must
    # refer to the same month, so "Fed vs rule" = fed funds KPI - rule KPI on screen.
    at = run_panel()
    inflation_kpi, fed_kpi, rule_kpi = at.metric
    month = rule_kpi.label.split(", ")[1]
    assert inflation_kpi.label.endswith(month)
    assert fed_kpi.label.endswith(month)

    fed = float(fed_kpi.value.rstrip("%"))
    rule = float(rule_kpi.value.rstrip("%"))
    gap = float(rule_kpi.delta.split(" pp")[0])
    assert gap == pytest.approx(fed - rule, abs=0.011)  # equal up to rounding to 2 decimals
    assert f"In {month} the fed funds rate ({fed_kpi.value})" in at.info[0].value


def test_every_control_can_be_changed():
    at = run_panel()

    for measure in ["Headline CPI", "Core CPI", "Core PCE"]:
        at.selectbox(key="inflation_measure").set_value(measure).run()
        assert_no_errors(at)

    # Picking a preset moves the sliders
    at.selectbox(key="inflation_preset").set_value("Balanced approach").run()
    assert_no_errors(at)
    assert at.slider(key="inflation_b").value == 1.0

    # Moving a slider switches the preset to Custom
    at.slider(key="inflation_r_star").set_value(0.5).run()
    assert_no_errors(at)
    assert at.selectbox(key="inflation_preset").value == "Custom"
    at.slider(key="inflation_pi_target").set_value(3.0).run()
    at.slider(key="inflation_a").set_value(1.2).run()
    at.slider(key="inflation_b").set_value(1.5).run()
    assert_no_errors(at)

    at.select_slider(key="inflation_period").set_value((2019, 2026)).run()
    assert_no_errors(at)
    at.toggle(key="inflation_floor").set_value(True).run()
    assert_no_errors(at)
    at.toggle(key="inflation_show_inflation").set_value(False).run()
    assert_no_errors(at)

    # Back to a preset: sliders reset to its values
    at.selectbox(key="inflation_preset").set_value("Taylor (1993)").run()
    assert_no_errors(at)
    assert at.slider(key="inflation_r_star").value == 2.0
    assert at.slider(key="inflation_b").value == 0.5


def test_slider_matching_a_preset_selects_it():
    at = run_panel()
    at.slider(key="inflation_b").set_value(1.0).run()  # Taylor (1993) with b = 1 is Balanced approach
    assert at.selectbox(key="inflation_preset").value == "Balanced approach"


def test_extreme_control_values():
    at = run_panel()
    for parameter in ["r_star", "pi_target", "a", "b"]:
        at.slider(key=f"inflation_{parameter}").set_value(0.0)
    at.select_slider(key="inflation_period").set_value((2026, 2026))
    at.toggle(key="inflation_floor").set_value(True)
    at.run()
    assert_no_errors(at)

    at.slider(key="inflation_r_star").set_value(4.0)
    at.slider(key="inflation_pi_target").set_value(4.0)
    at.slider(key="inflation_a").set_value(2.0)
    at.slider(key="inflation_b").set_value(2.0)
    at.select_slider(key="inflation_period").set_value((1986, 1986))
    at.run()
    assert_no_errors(at)
    assert len(at.metric) == 3


def test_panel_without_data_shows_a_message(monkeypatch):
    def no_data():
        raise DataUnavailableError("no snapshot")

    monkeypatch.setattr(inflation, "_cached_data", no_data)
    at = AppTest.from_function(_panel_app, default_timeout=120)
    at.run()
    assert not at.exception
    assert "isn't available" in at.warning[0].value


def test_panel_with_short_gappy_data(monkeypatch):
    data = synthetic_data(30)
    data.loc["2024-03-01", ["CPIAUCSL", "FEDFUNDS"]] = np.nan  # a missing month
    monkeypatch.setattr(inflation, "_cached_data", lambda: data)
    at = AppTest.from_function(_panel_app, default_timeout=120)
    at.run()
    assert_no_errors(at)
    at.selectbox(key="inflation_measure").set_value("Headline CPI").run()
    assert_no_errors(at)
