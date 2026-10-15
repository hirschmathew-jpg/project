"""Tests for Panel 3 - Sector Winners & Losers (no internet needed).

The maths is tested on small made-up price series where the right answer
can be worked out by hand. The Streamlit tests (AppTest) draw the panel from
the saved snapshot in data/ and click through every control.
"""

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import theme
from dashboard.panels import sectors

BENCH = sectors.BENCHMARK


def prices_from_returns(daily_returns: dict, start: str = "2024-01-01", first_price: float = 100.0) -> pd.DataFrame:
    """Build a price table (business days) whose day-to-day returns are exactly ``daily_returns``."""
    columns = {}
    for ticker, returns in daily_returns.items():
        growth_factors = [1.0] + [1 + r for r in returns]  # +1% -> multiply by 1.01
        growth = np.cumprod(growth_factors)
        columns[ticker] = first_price * growth
    length = len(next(iter(columns.values())))
    index = pd.bdate_range(start, periods=length)
    return pd.DataFrame(columns, index=index)


# ---------------------------------------------------------------------------
# Measures (hand-computed answers)
# ---------------------------------------------------------------------------

def test_total_return_constant_growth():
    prices = prices_from_returns({"A": [0.01] * 10})
    assert sectors.total_return(prices)["A"] == pytest.approx(1.01 ** 10 - 1)


def test_constant_growth_has_no_risk():
    prices = prices_from_returns({"A": [0.01] * 10})
    returns = sectors.daily_returns(prices)
    assert len(returns) == 10  # the first day has no return
    assert sectors.annualised_volatility(returns)["A"] == pytest.approx(0.0, abs=1e-12)
    assert sectors.max_drawdown(prices)["A"] == 0.0
    # Zero volatility: the Sharpe ratio is undefined (NaN), not a division-by-zero error
    assert np.isnan(sectors.sharpe_ratio(returns, 0.02)["A"])


def test_annualised_volatility_hand_computed():
    prices = prices_from_returns({"A": [0.01, -0.01, 0.01, -0.01]})
    returns = sectors.daily_returns(prices)
    # Mean 0, squared deviations 4 x 0.0001, sample variance = 0.0004 / 3
    expected = np.sqrt(0.0004 / 3) * np.sqrt(252)
    assert sectors.annualised_volatility(returns)["A"] == pytest.approx(expected)


def test_sharpe_ratio_hand_computed():
    prices = prices_from_returns({"A": [0.02, 0.0, 0.02, 0.0]})
    returns = sectors.daily_returns(prices)
    # Daily risk-free rate = 2.52% / 252 = 0.0001; mean excess return = 0.01 - 0.0001 = 0.0099
    expected = (0.0099 * 252) / (np.sqrt(0.0004 / 3) * np.sqrt(252))
    assert sectors.sharpe_ratio(returns, 0.0252)["A"] == pytest.approx(expected)
    # A higher risk-free rate lowers the Sharpe ratio
    assert sectors.sharpe_ratio(returns, 0.05)["A"] < expected


def test_max_drawdown_known_path():
    index = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [100.0, 120.0, 90.0, 130.0, 104.0]}, index=index)
    # Running max 100, 120, 120, 130, 130 -> worst fall 90 / 120 - 1 = -25%
    assert sectors.max_drawdown(prices)["A"] == pytest.approx(-0.25)


def test_beta_of_two_times_leveraged_copy_is_two():
    rng = np.random.default_rng(seed=1)
    market = rng.normal(0.0005, 0.01, size=250)
    prices = prices_from_returns({BENCH: market, "LEV": 2 * market, "HALF": 0.5 * market})
    returns = sectors.daily_returns(prices)
    betas = sectors.beta(returns, returns[BENCH])
    assert betas[BENCH] == pytest.approx(1.0)
    assert betas["LEV"] == pytest.approx(2.0)
    assert betas["HALF"] == pytest.approx(0.5)


def test_beta_with_flat_benchmark_is_nan():
    prices = prices_from_returns({BENCH: [0.0] * 6, "A": [0.01, -0.02, 0.01, 0.0, 0.03, -0.01]})
    returns = sectors.daily_returns(prices)
    assert sectors.beta(returns, returns[BENCH]).isna().all()


def test_compute_metrics_table():
    rng = np.random.default_rng(seed=2)
    market = rng.normal(0.0, 0.01, size=60)
    prices = prices_from_returns({BENCH: market, "XLK": 2 * market})
    table = sectors.compute_metrics(prices, risk_free_annual=0.0)
    assert list(table.columns) == list(sectors.METRICS)
    assert list(table.index) == [BENCH, "XLK"]
    assert table.loc["XLK", sectors.BETA] == pytest.approx(2.0)
    # Twice the daily moves -> twice the volatility
    assert table.loc["XLK", sectors.VOLATILITY] == pytest.approx(2 * table.loc[BENCH, sectors.VOLATILITY])


def test_compute_metrics_without_benchmark_gives_nan_beta():
    prices = prices_from_returns({"XLK": [0.01, -0.01, 0.02, 0.0, 0.01, -0.005]})
    assert np.isnan(sectors.compute_metrics(prices).loc["XLK", sectors.BETA])


# ---------------------------------------------------------------------------
# Periods and windows
# ---------------------------------------------------------------------------

def test_period_dates_presets():
    last = pd.Timestamp("2026-10-14")
    assert sectors.period_dates("Year to date", last) == (pd.Timestamp("2025-12-31"), last)
    assert sectors.period_dates("Last 12 months", last) == (pd.Timestamp("2025-10-14"), last)
    assert sectors.period_dates("Last 3 years", last) == (pd.Timestamp("2023-10-14"), last)
    assert sectors.period_dates("Last 5 years", last) == (pd.Timestamp("2021-10-14"), last)
    assert sectors.period_dates("2022 rate-hike sell-off", last) == (
        pd.Timestamp("2021-12-31"), pd.Timestamp("2022-12-31"))
    covid = sectors.period_dates("COVID crash (19 Feb – 23 Mar 2020)", last)
    assert covid == (pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"))


def test_every_period_option_has_dates_and_a_phrase():
    for period in sectors.PERIODS:
        start, end = sectors.period_dates(period, "2026-10-14", "2020-01-01", "2021-01-01")
        assert start <= end
        if period != sectors.CUSTOM:
            assert period in sectors.PERIOD_PHRASES


def test_custom_period_swaps_reversed_dates():
    start, end = sectors.period_dates("Custom", "2026-10-14", datetime.date(2021, 6, 1), datetime.date(2020, 1, 1))
    assert start == pd.Timestamp("2020-01-01")
    assert end == pd.Timestamp("2021-06-01")


def test_period_dates_rejects_bad_input():
    with pytest.raises(ValueError):
        sectors.period_dates("Next week", "2026-10-14")
    with pytest.raises(ValueError):
        sectors.period_dates("Custom", "2026-10-14")


def test_year_to_date_near_year_start():
    # Data up to the first trading day of 2026: YTD = last close of 2025 + 2 Jan 2026
    prices = pd.DataFrame({BENCH: [100.0, 101.0, 102.0, 103.0]},
                          index=pd.DatetimeIndex(["2025-12-29", "2025-12-30", "2025-12-31", "2026-01-02"]))
    start, end = sectors.period_dates("Year to date", prices.index[-1])
    window = sectors.window_prices(prices, start, end)
    assert list(window.index) == [pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-02")]
    assert sectors.total_return(window)[BENCH] == pytest.approx(103 / 102 - 1)
    # Only 1 trading day of returns: render() shows a message instead of a chart
    assert len(window) - 1 < sectors.MIN_TRADING_DAYS


def test_window_starts_at_last_close_before_a_holiday():
    # 31 Dec 2022 was a Saturday: the 2023 return is measured from Friday 30 Dec
    prices = pd.DataFrame({BENCH: [100.0, 110.0, 121.0]},
                          index=pd.DatetimeIndex(["2022-12-30", "2023-01-03", "2023-01-04"]))
    window = sectors.window_prices(prices, "2022-12-31", "2023-12-31")
    assert window.index[0] == pd.Timestamp("2022-12-30")
    assert sectors.total_return(window)[BENCH] == pytest.approx(0.21)


def test_window_before_the_data_starts_at_first_price():
    prices = prices_from_returns({BENCH: [0.01] * 5}, start="2024-01-01")
    window = sectors.window_prices(prices, "2000-01-01", "2024-01-03")
    assert window.index[0] == prices.index[0]
    assert len(window) == 3
    assert sectors.window_prices(prices, "1990-01-01", "1999-01-01").empty


def test_tickers_missing_data_are_excluded():
    prices = prices_from_returns({BENCH: [0.01] * 5, "XLK": [0.02] * 5, "XLC": [0.01] * 5})
    prices.loc[prices.index[:2], "XLC"] = np.nan  # XLC only starts trading on day 3
    complete, excluded = sectors.split_by_data_coverage(prices)
    assert complete == [BENCH, "XLK"]
    assert excluded == ["XLC"]
    # Once the window starts after XLC's first price it is included again
    later = sectors.window_prices(prices, prices.index[2], prices.index[-1])
    assert sectors.split_by_data_coverage(later) == ([BENCH, "XLK", "XLC"], [])


# ---------------------------------------------------------------------------
# Ranking, relative mode, wording
# ---------------------------------------------------------------------------

def test_rank_sectors_best_first():
    values = pd.Series({BENCH: 0.10, "XLK": 0.30, "XLE": -0.20, "XLU": 0.05, "XLC": np.nan})
    ranked = sectors.rank_sectors(values, sectors.TOTAL_RETURN)
    assert list(ranked.index) == ["XLK", "XLU", "XLE"]  # S&P 500 and the missing value left out
    # Volatility: less risk is "better", so the lowest comes first
    vol = sectors.rank_sectors(pd.Series({BENCH: 0.2, "XLK": 0.3, "XLP": 0.1}), sectors.VOLATILITY)
    assert list(vol.index) == ["XLP", "XLK"]
    # Drawdown: -10% is a smaller fall than -40%
    dd = sectors.rank_sectors(pd.Series({"XLE": -0.4, "XLP": -0.1}), sectors.DRAWDOWN)
    assert list(dd.index) == ["XLP", "XLE"]


def test_relative_mode_and_beating_the_benchmark():
    values = pd.Series({"XLK": 0.30, "XLE": -0.20, "XLU": 0.05})
    relative = sectors.relative_to_benchmark(values, 0.10)
    assert relative["XLK"] == pytest.approx(0.20)
    assert relative["XLE"] == pytest.approx(-0.30)
    assert list(sectors.beats_benchmark(values, 0.10, sectors.TOTAL_RETURN)) == [True, False, False]
    # For volatility, beating the S&P 500 means being LESS volatile
    assert list(sectors.beats_benchmark(values, 0.10, sectors.VOLATILITY)) == [False, True, True]


def test_format_value():
    assert sectors.format_value(0.1234, sectors.TOTAL_RETURN) == "+12.3%"
    assert sectors.format_value(-0.05, sectors.TOTAL_RETURN) == "-5.0%"
    assert sectors.format_value(0.18, sectors.VOLATILITY) == "18.0%"
    assert sectors.format_value(-0.337, sectors.DRAWDOWN) == "-33.7%"
    assert sectors.format_value(0.041, sectors.TOTAL_RETURN, relative=True) == "+4.1 pp"
    assert sectors.format_value(0.8512, sectors.SHARPE) == "0.85"
    assert sectors.format_value(0.25, sectors.BETA, relative=True) == "+0.25"
    assert sectors.format_value(np.nan, sectors.BETA) == "n/a"


def test_count_phrase():
    assert sectors.count_phrase(0, 11) == "none of the 11 sectors"
    assert sectors.count_phrase(3, 11) == "3 of the 11 sectors"
    assert sectors.count_phrase(9, 9) == "all 9 sectors"


def test_ranking_insight_for_returns():
    ranked = pd.Series({"XLE": 0.64, "XLU": 0.01, "XLC": -0.38})
    text = sectors.ranking_insight(sectors.TOTAL_RETURN, ranked, -0.18, "the 2022 rate-hike sell-off")
    assert text == ("Over the 2022 rate-hike sell-off, Energy led with +64.0% and Communication Services "
                    "lagged with -38.0%; 2 of the 3 sectors beat the S&P 500 (-18.0%).")


def test_ranking_insight_when_every_sector_fell_or_rose():
    fell = pd.Series({"XLP": -0.24, "XLE": -0.56})
    text = sectors.ranking_insight(sectors.TOTAL_RETURN, fell, -0.34, "the COVID crash")
    assert "every sector lost money: Consumer Staples fell least (-24.0%) and Energy most (-56.0%)" in text
    assert "1 of the 2 sectors beat the S&P 500 (-34.0%)" in text
    rose = pd.Series({"XLK": 0.5, "XLU": 0.1})
    assert "every sector made money" in sectors.ranking_insight(sectors.TOTAL_RETURN, rose, 0.2, "x")


def test_ranking_insight_for_risk_measures():
    vol = pd.Series({"XLP": 0.12, "XLE": 0.30})
    text = sectors.ranking_insight(sectors.VOLATILITY, vol, 0.15, "the last 5 years")
    assert "Consumer Staples was the calmest sector (12.0% a year)" in text
    assert "1 of the 2 sectors had lower volatility than the S&P 500 (15.0%)" in text
    betas = pd.Series({"XLP": 0.5, "XLK": 1.3})
    text = sectors.ranking_insight(sectors.BETA, betas, 1.0, "the year to date")
    assert "Technology moved most with the market (beta 1.30)" in text
    assert text.endswith("had a beta below 1, so moved less than the market.")


def test_growth_of_100_and_weekly_thinning():
    rng = np.random.default_rng(seed=3)
    prices = prices_from_returns({BENCH: rng.normal(0.0004, 0.01, 800), "XLK": rng.normal(0.0006, 0.015, 800)})
    growth = sectors.growth_of_100(prices)
    assert (growth.iloc[0] == 100).all()
    assert growth["XLK"].iloc[-1] == pytest.approx(100 * (1 + sectors.total_return(prices)["XLK"]))

    weekly = sectors.thin_to_weekly(growth)
    assert len(weekly) < len(growth) / 4
    assert weekly.index[0] == growth.index[0]    # still starts at $100
    assert weekly.index[-1] == growth.index[-1]  # still ends on the last price
    assert weekly.index.isin(growth.index).all()  # only real trading days, no invented dates


def test_growth_insight_wording():
    index = pd.bdate_range("2022-01-03", periods=3)
    growth = pd.DataFrame({BENCH: [100.0, 95.0, 90.0], "XLE": [100.0, 120.0, 150.0]}, index=index)
    text = sectors.growth_insight(growth, "XLE", -0.10)
    assert "$100 put into Energy at the close on 3 Jan 2022 was worth $150.00 on 5 Jan 2022" in text
    assert "beat the index by $60.00" in text
    assert "at worst 10.0% below an earlier high" in text
    assert "trailed the index by $60.00" in sectors.growth_insight(growth.rename(
        columns={BENCH: "XLE", "XLE": BENCH}), "XLE", -0.1)
    assert "never fell below an earlier high" in sectors.growth_insight(growth, "XLE", 0.0)


def test_markdown_safe_escapes_dollars():
    assert sectors.markdown_safe("$100 became $150") == "\\$100 became \\$150"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def test_ranking_chart_sorted_with_diverging_colours():
    ranked = pd.Series({"XLE": 0.64, "XLU": 0.01, "XLC": -0.38})
    fig = sectors.build_ranking_chart(ranked, -0.18, sectors.TOTAL_RETURN)
    bars = fig.data[0]
    assert list(bars.y) == ["Energy", "Utilities", "Communication Services"]
    assert list(bars.x) == pytest.approx([64.0, 1.0, -38.0])  # plotted in % units
    assert list(bars.marker.color) == [theme.POSITIVE, theme.POSITIVE, theme.NEGATIVE]
    assert len(fig.data) == 1  # the S&P 500 is a reference line, not a bar
    assert fig.layout.shapes[-1].x0 == pytest.approx(-18.0)
    labels = [a.text for a in fig.layout.annotations]
    assert "S&P 500: -18.0%" in labels and "+64.0%" in labels


def test_ranking_chart_relative_mode():
    ranked = pd.Series({"XLE": 0.64, "XLU": 0.01, "XLC": -0.38})
    fig = sectors.build_ranking_chart(ranked, -0.18, sectors.TOTAL_RETURN, relative=True)
    assert list(fig.data[0].x) == pytest.approx([82.0, 19.0, -20.0])
    assert fig.layout.shapes[-1].x0 == 0.0  # the S&P 500 line moves to zero
    legend_names = [trace.name for trace in fig.data[1:]]
    assert legend_names == ["Beat the S&P 500", "Trailed the S&P 500"]
    assert fig.layout.xaxis.title.text == "Difference from the S&P 500's total return (percentage points)"
    # Only colours that are actually used get a legend entry
    all_beat = sectors.build_ranking_chart(ranked, -0.5, sectors.TOTAL_RETURN, relative=True)
    assert [trace.name for trace in all_beat.data[1:]] == ["Beat the S&P 500"]


def test_value_axis_range_leaves_room_for_labels():
    assert sectors.value_axis_range([10.0, 30.0]) == pytest.approx([0.0, 36.0])
    assert sectors.value_axis_range([-20.0, 10.0]) == pytest.approx([-26.0, 16.0])
    assert sectors.value_axis_range([0.0, 0.0]) == [-1.0, 1.0]  # never a zero-width axis


def test_end_labels_are_pushed_apart_when_close():
    index = pd.bdate_range("2024-01-01", periods=3)
    growth = pd.DataFrame({BENCH: [100.0, 120.0, 110.0], "XLK": [100.0, 90.0, 111.0]}, index=index)
    fig = go.Figure()
    sectors.add_end_labels(fig, growth, [BENCH, "XLK"])
    shifts = {a.text.split()[0]: a.yshift for a in fig.layout.annotations}
    assert shifts == {BENCH: -8, "XLK": 8}  # XLK ends higher, so its label moves up


def test_bar_colours_by_measure():
    values = pd.Series({"XLP": 0.1, "XLE": 0.3})
    assert sectors.bar_colours(values, 0.2, sectors.VOLATILITY, False) == [theme.BLUE, theme.BLUE]
    assert sectors.bar_colours(-values, -0.2, sectors.DRAWDOWN, False) == [theme.NEGATIVE, theme.NEGATIVE]
    # Relative volatility: less volatile than the S&P 500 counts as "beating" it (blue)
    assert sectors.bar_colours(values, 0.2, sectors.VOLATILITY, True) == [theme.POSITIVE, theme.NEGATIVE]


def test_growth_chart_emphasis_pattern():
    rng = np.random.default_rng(seed=4)
    returns = {ticker: rng.normal(0, 0.01, 30) for ticker in [BENCH, "XLK", "XLE", "XLU"]}
    growth = sectors.growth_of_100(prices_from_returns(returns))
    fig = sectors.build_growth_chart(growth, "XLE", recessions=[])
    by_name = {trace.name: trace for trace in fig.data}
    assert by_name["Energy (XLE)"].line.color == theme.BLUE
    assert by_name["Energy (XLE)"].line.width == 2.5
    assert by_name["S&P 500 (SPY)"].line.color == theme.INK_SECONDARY
    for name in ["Technology (XLK)", "Utilities (XLU)"]:
        assert by_name[name].line.color == theme.AXIS
        assert by_name[name].showlegend is False
    assert fig.data[-1].name == "Energy (XLE)"  # drawn last, on top
    end_labels = [a.text for a in fig.layout.annotations]
    assert any(text.startswith("XLE $") for text in end_labels)
    assert any(text.startswith("SPY $") for text in end_labels)


def test_growth_chart_shades_overlapping_recessions_only():
    growth = sectors.growth_of_100(prices_from_returns({BENCH: [0.01] * 30, "XLK": [0.0] * 30}, start="2020-02-03"))
    covid = [(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-01"))]
    old = [(pd.Timestamp("2008-01-01"), pd.Timestamp("2009-07-01"))]
    shaded = sectors.build_growth_chart(growth, "XLK", covid)
    assert "US recession" in [trace.name for trace in shaded.data]
    unshaded = sectors.build_growth_chart(growth, "XLK", old)
    assert "US recession" not in [trace.name for trace in unshaded.data]


# ---------------------------------------------------------------------------
# The snapshot shipped in data/
# ---------------------------------------------------------------------------

def test_snapshot_is_shipped_and_complete():
    prices = sectors.load_data()
    assert list(prices.columns) == sectors.TICKERS
    assert prices.index.is_monotonic_increasing
    assert prices.index[0] <= pd.Timestamp("2007-01-31")
    assert prices[BENCH].notna().all()
    # The two young ETFs only have prices from their launch dates
    assert prices["XLC"].first_valid_index() >= pd.Timestamp("2018-06-01")
    assert prices["XLRE"].first_valid_index() >= pd.Timestamp("2015-10-01")


# ---------------------------------------------------------------------------
# The Streamlit panel (AppTest clicks through every control)
# ---------------------------------------------------------------------------

def _panel_app():
    from dashboard.panels import sectors

    sectors.render()


def _no_data_app():
    from dashboard.data import DataUnavailableError
    from dashboard.panels import sectors

    def broken_load(refresh=False):
        raise DataUnavailableError("no snapshot and no internet")

    original = sectors.load_data
    sectors.load_data = broken_load
    sectors._cached_prices.clear()
    try:
        sectors.render()
    finally:
        sectors.load_data = original
        sectors._cached_prices.clear()


def run_panel():
    at = AppTest.from_function(_panel_app, default_timeout=120)
    at.run()
    assert_clean(at)
    return at


def assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def texts(elements) -> str:
    return " ".join(str(element.value) for element in elements)


def test_panel_default_view():
    at = run_panel()
    assert len(at.metric) == 3
    assert at.metric[0].label == "S&P 500 total return"
    assert at.metric[1].label == "Best: Energy"  # 2022: energy was the only big winner
    assert len(at.get("plotly_chart")) == 1
    assert "Over the 2022 rate-hike sell-off" in texts(at.info)
    assert "Measured from the close on 31 Dec 2021" in texts(at.caption)
    for key in ["period", "metric", "view", "relative"]:
        assert at.session_state[f"sectors_{key}"] is not None


def test_every_period_measure_and_view():
    at = run_panel()
    for period in sectors.PERIODS:
        if period == sectors.CUSTOM:
            continue
        at.selectbox(key="sectors_period").set_value(period).run()
        for metric in sectors.METRICS:
            at.selectbox(key="sectors_metric").set_value(metric).run()
            assert_clean(at)
            assert len(at.metric) == 3, (period, metric)
        at.radio(key="sectors_view").set_value(sectors.GROWTH).run()
        assert_clean(at)
        assert "$100 put into" in texts(at.info).replace("\\$", "$")
        at.radio(key="sectors_view").set_value(sectors.RANKING).run()
        assert_clean(at)


def test_relative_toggle_for_every_measure():
    at = run_panel()
    at.toggle(key="sectors_relative").set_value(True).run()
    for metric in sectors.METRICS:
        at.selectbox(key="sectors_metric").set_value(metric).run()
        assert_clean(at)
        assert len(at.get("plotly_chart")) == 1


def test_risk_free_slider_only_for_sharpe():
    at = run_panel()
    assert not at.slider  # hidden unless the Sharpe ratio is chosen
    at.selectbox(key="sectors_metric").set_value(sectors.SHARPE).run()
    assert len(at.slider) == 1
    low = at.metric[0].value
    at.slider(key="sectors_risk_free").set_value(5.0).run()
    assert_clean(at)
    assert at.metric[0].value != low  # a higher risk-free rate changes the S&P 500's Sharpe ratio


def test_growth_view_highlight_every_sector():
    at = run_panel()
    at.radio(key="sectors_view").set_value(sectors.GROWTH).run()
    for ticker in sectors.SECTORS:
        at.selectbox(key="sectors_highlight").set_value(ticker).run()
        assert_clean(at)
        assert sectors.SECTORS[ticker] in texts(at.info)


def test_growth_view_falls_back_when_highlight_has_no_data():
    at = run_panel()
    at.selectbox(key="sectors_period").set_value("Global financial crisis (Oct 2007 – Mar 2009)").run()
    at.radio(key="sectors_view").set_value(sectors.GROWTH).run()
    at.selectbox(key="sectors_highlight").set_value("XLC").run()
    assert_clean(at)
    captions = texts(at.caption)
    assert "Communication Services (XLC)" in captions  # listed as excluded
    assert "is shown in blue instead" in captions


def test_custom_dates_including_reversed_and_too_short():
    at = run_panel()
    at.selectbox(key="sectors_period").set_value(sectors.CUSTOM).run()
    assert_clean(at)
    assert len(at.date_input) == 1

    # Start after end: the dates are swapped, not an error
    at.date_input(key="sectors_dates").set_value((datetime.date(2021, 6, 30), datetime.date(2020, 6, 30))).run()
    assert_clean(at)
    assert "30 Jun 2020" in texts(at.caption)

    # Only two trading days: a friendly message instead of a chart
    at.date_input(key="sectors_dates").set_value((datetime.date(2026, 10, 12), datetime.date(2026, 10, 14))).run()
    assert_clean(at)
    assert "too few" in texts(at.info)
    assert not at.get("plotly_chart")

    # A weekend-only window: no trading days at all
    at.date_input(key="sectors_dates").set_value((datetime.date(2026, 10, 10), datetime.date(2026, 10, 11))).run()
    assert_clean(at)
    assert "too few" in texts(at.info)

    # Before XLC launched: it is left out and the caption says so
    at.date_input(key="sectors_dates").set_value((datetime.date(2016, 1, 4), datetime.date(2017, 12, 29))).run()
    assert_clean(at)
    assert "Communication Services (XLC)" in texts(at.caption)


def test_panel_without_data_shows_a_message():
    at = AppTest.from_function(_no_data_app, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Sector price data isn't available" in texts(at.warning)
