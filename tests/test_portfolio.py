"""Tests for Panel 4 - Portfolio Scenario Lab (no internet needed).

The model tests use tiny made-up data with answers worked out by hand.
The AppTest tests run the real panel on the CSV snapshot in data/ and move
every control to check nothing crashes.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import theme
from dashboard.data import DataUnavailableError
from dashboard.panels import portfolio

TWO_ASSETS = ["A", "B"]


def no_change_scenario(tickers):
    """A scenario that changes nothing (like "Historical"), for any list of tickers."""
    return {
        "drift_pp": {ticker: 0.0 for ticker in tickers},
        "vol_mult": {ticker: 1.0 for ticker in tickers},
        "shock_pct": {ticker: 0.0 for ticker in tickers},
    }


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def test_weights_that_add_to_100_are_unchanged():
    weights = portfolio.normalise_weights({"SPY": 50, "AGG": 30, "GLD": 10, "BTC-USD": 10})
    assert weights == pytest.approx({"SPY": 0.5, "AGG": 0.3, "GLD": 0.1, "BTC-USD": 0.1})


def test_weights_are_rescaled_to_100_percent():
    # 60 + 60 = 120, so each becomes 60 / 120 = 0.5
    assert portfolio.normalise_weights({"SPY": 60, "AGG": 60}) == pytest.approx({"SPY": 0.5, "AGG": 0.5})
    # 10 + 0 + 30 = 40 -> 0.25, 0, 0.75
    weights = portfolio.normalise_weights({"SPY": 10, "AGG": 0, "GLD": 30})
    assert weights == pytest.approx({"SPY": 0.25, "AGG": 0.0, "GLD": 0.75})
    assert sum(weights.values()) == pytest.approx(1.0)


def test_all_zero_weights_give_none():
    assert portfolio.normalise_weights({"SPY": 0, "AGG": 0, "GLD": 0, "BTC-USD": 0}) is None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def test_zero_volatility_equals_deterministic_compounding():
    # One asset growing exactly 1% a month with no randomness: value = 1000 x 1.01^t
    mean = np.array([math.log(1.01)])
    cov = np.zeros((1, 1))
    values = portfolio.simulate_portfolio(mean, cov, np.array([1.0]), months=12, start_value=1000,
                                          n_sims=5, seed=1)
    assert values.shape == (5, 13)
    expected = 1000 * 1.01 ** np.arange(13)
    for row in values:
        assert row == pytest.approx(expected)
    assert values[0, -1] == pytest.approx(1126.825, abs=1e-3)  # 1000 x 1.01^12


def test_monthly_rebalancing_with_two_assets():
    # A grows 2% (log) a month, B is flat. Rebalanced 60/40 every month, so each month the
    # portfolio grows by 0.6 x e^0.02 + 0.4 x 1 = 1.012121...
    mean = np.array([0.02, 0.0])
    cov = np.zeros((2, 2))
    values = portfolio.simulate_portfolio(mean, cov, np.array([0.6, 0.4]), months=24, start_value=100,
                                          n_sims=3, seed=7)
    monthly_growth = 0.6 * math.exp(0.02) + 0.4
    assert monthly_growth == pytest.approx(1.012121, abs=1e-6)
    assert values[:, -1] == pytest.approx(np.full(3, 100 * monthly_growth ** 24))


def test_same_seed_gives_same_results_and_new_seed_differs():
    mean = np.array([0.005, 0.002])
    cov = np.array([[0.0020, 0.0003], [0.0003, 0.0004]])
    weights = np.array([0.7, 0.3])
    first = portfolio.simulate_portfolio(mean, cov, weights, 36, 10_000, 200, seed=42)
    again = portfolio.simulate_portfolio(mean, cov, weights, 36, 10_000, 200, seed=42)
    other = portfolio.simulate_portfolio(mean, cov, weights, 36, 10_000, 200, seed=43)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_one_off_shock_hits_month_one_only():
    # No drift, no volatility: only the shock moves the value.
    # Month 1: A falls 50%, B unchanged, 50/50 weights -> 0.5 x 0.5 + 0.5 x 1 = 0.75
    mean = np.zeros(2)
    cov = np.zeros((2, 2))
    shocks = np.array([-0.5, 0.0])
    values = portfolio.simulate_portfolio(mean, cov, np.array([0.5, 0.5]), months=6, start_value=1000,
                                          n_sims=2, seed=0, shocks=shocks)
    assert values[0, 1] == pytest.approx(750)
    assert values[0, -1] == pytest.approx(750)  # flat afterwards: the shock is not repeated


def test_percentiles_are_in_order_every_month():
    mean = np.array([0.006, 0.002, 0.004, 0.02])
    cov = np.diag([0.002, 0.0003, 0.002, 0.03])
    values = portfolio.simulate_portfolio(mean, cov, np.array([0.5, 0.3, 0.1, 0.1]), 60, 10_000, 500, seed=3)
    paths = portfolio.percentile_paths(values)
    assert list(paths.columns) == ["p5", "p25", "p50", "p75", "p95"]
    assert len(paths) == 61
    assert (paths.iloc[0] == 10_000).all()  # everyone starts at the starting amount
    assert (paths["p5"] <= paths["p25"]).all()
    assert (paths["p25"] <= paths["p50"]).all()
    assert (paths["p50"] <= paths["p75"]).all()
    assert (paths["p75"] <= paths["p95"]).all()
    assert paths["p95"].iloc[-1] > paths["p5"].iloc[-1]  # the fan really opens up


def test_outcome_stats_hand_computed():
    # Four simulations over 1 year, ending at 90, 110, 121 and 100 (start 100)
    values = np.array([[100.0, 90.0], [100.0, 110.0], [100.0, 121.0], [100.0, 100.0]])
    stats = portfolio.outcome_stats(values, start_value=100, years=1)
    assert stats["median"] == pytest.approx(105.0)              # middle of 100 and 110
    assert stats["chance_of_loss"] == pytest.approx(0.25)       # only 90 < 100 (100 is not a loss)
    assert stats["median_growth"] == pytest.approx(0.05)        # median of -10%, 0%, +10%, +21%
    assert stats["p5"] == pytest.approx(91.5)                   # 90 + 0.15 x (100 - 90)
    assert stats["n_sims"] == 4


def test_median_growth_is_annualised():
    # Every simulation doubles in 10 years -> 2 ** (1/10) - 1 = 7.18% a year
    values = np.array([[1.0, 2.0], [1.0, 2.0]])
    assert portfolio.outcome_stats(values, 1.0, 10)["median_growth"] == pytest.approx(2 ** 0.1 - 1)


def test_histogram_shares_and_top_cut():
    final = np.arange(1, 1001, dtype=float)  # 1, 2, ..., 1000
    bins = portfolio.histogram_table(final, n_bins=10, top_cut=99)
    assert len(bins) == 10
    assert bins["from"].iloc[0] == 1
    assert bins["to"].iloc[-1] == pytest.approx(np.percentile(final, 99))
    # Shares are % of ALL simulations, so the values above the cut are missing (about 1%)
    assert bins["share_pct"].sum() == pytest.approx(99.0, abs=0.2)


def test_histogram_when_every_value_is_the_same():
    bins = portfolio.histogram_table(np.full(50, 500.0), n_bins=5)
    assert bins["share_pct"].sum() == pytest.approx(100.0)
    assert bins["from"].iloc[0] < 500 < bins["to"].iloc[-1]
    log_bins = portfolio.histogram_table(np.full(50, 500.0), n_bins=5, log_scale=True)
    assert log_bins["share_pct"].sum() == pytest.approx(100.0)


def skewed_outcomes():
    """Final values like a bitcoin-heavy portfolio: lognormal, the best are ~100x the worst."""
    rng = np.random.default_rng(0)
    return 10_000 * np.exp(rng.normal(0.5, 1.5, size=5000))


def test_log_scale_histogram_spreads_out_skewed_outcomes():
    final = skewed_outcomes()
    # Regression: equal-width bins put most simulations into the first bar
    linear = portfolio.histogram_table(final)
    assert linear["share_pct"].iloc[0] > 50
    # Log-scale bins: every bin covers the same ratio, nothing is cut off, no bar dominates
    bins = portfolio.histogram_table(final, log_scale=True)
    assert len(bins) == portfolio.HISTOGRAM_BINS
    assert bins["from"].iloc[0] == pytest.approx(final.min())
    assert bins["to"].iloc[-1] == pytest.approx(final.max())
    ratios = bins["to"] / bins["from"]
    assert ratios.to_numpy() == pytest.approx(np.full(len(bins), ratios.iloc[0]))
    assert bins["share_pct"].sum() == pytest.approx(100.0)
    assert bins["share_pct"].max() < 25


def test_use_log_scale_only_for_very_spread_out_outcomes():
    # 5% best case 4x the 5% worst case (a typical mixed portfolio): linear axis
    assert not portfolio.use_log_scale({"p5": 15_000, "p95": 60_000})
    # 300x (all bitcoin over 10 years): log axis
    assert portfolio.use_log_scale({"p5": 2_000, "p95": 600_000})
    assert not portfolio.use_log_scale({"p5": 1_000, "p95": 10_000})  # exactly 10x stays linear


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def test_apply_scenario_maths():
    mean = np.array([0.01, 0.02])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    scenario = {"drift_pp": {"A": 12.0, "B": -6.0}, "vol_mult": {"A": 2.0, "B": 1.0},
                "shock_pct": {"A": 0.0, "B": 0.0}}
    new_mean, new_cov = portfolio.apply_scenario(mean, cov, scenario, TWO_ASSETS)
    # +12 pp a year = +0.12 / 12 = +0.01 a month; -6 pp a year = -0.005 a month
    assert new_mean == pytest.approx([0.02, 0.015])
    # Volatility of A doubles: var(A) x 4, cov(A, B) x 2, var(B) unchanged
    assert new_cov == pytest.approx(np.array([[0.16, 0.02], [0.02, 0.09]]))
    # Correlation is unchanged: 0.01 / (0.2 x 0.3) = 0.02 / (0.4 x 0.3)
    assert new_cov[0, 1] / math.sqrt(new_cov[0, 0] * new_cov[1, 1]) == pytest.approx(0.01 / (0.2 * 0.3))


def test_no_change_scenario_leaves_inputs_alone():
    mean = np.array([0.01, 0.02])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    new_mean, new_cov = portfolio.apply_scenario(mean, cov, no_change_scenario(TWO_ASSETS), TWO_ASSETS)
    assert new_mean == pytest.approx(mean)
    assert new_cov == pytest.approx(cov)


def test_scenario_shocks_are_decimals():
    scenario = portfolio.SCENARIOS["Stock market crash"]
    shocks = portfolio.scenario_shocks(scenario, portfolio.TICKERS)
    assert shocks == pytest.approx([-0.30, 0.03, 0.05, -0.45])


def test_every_scenario_is_complete_and_sensible():
    assert portfolio.DEFAULT_SCENARIO in portfolio.SCENARIOS
    for name, scenario in portfolio.SCENARIOS.items():
        assert scenario["summary"], name
        for field in ["drift_pp", "vol_mult", "shock_pct"]:
            assert set(scenario[field]) == set(portfolio.TICKERS), (name, field)
        for ticker in portfolio.TICKERS:
            assert scenario["vol_mult"][ticker] > 0
            assert scenario["shock_pct"][ticker] > -100  # an asset can't lose more than everything
    historical = portfolio.SCENARIOS["Historical"]
    assert historical == {"summary": historical["summary"], **no_change_scenario(portfolio.TICKERS)}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def btc_style_prices(end="2024-03-31"):
    """Daily prices: bitcoin (B) trades every day, the ETF (S) only on weekdays."""
    days = pd.date_range("2024-01-01", end, freq="D")
    btc = pd.Series(np.linspace(100, 200, len(days)), index=days)
    etf = pd.Series(np.linspace(50, 60, len(days)), index=days)
    etf[etf.index.dayofweek >= 5] = np.nan  # no ETF prices on Saturdays and Sundays
    btc[pd.Timestamp("2024-01-31")] = 100.0
    btc[pd.Timestamp("2024-02-29")] = 110.0
    btc[pd.Timestamp("2024-03-31")] = 99.0  # a Sunday: bitcoin still trades
    etf[pd.Timestamp("2024-01-31")] = 50.0   # Wednesday
    etf[pd.Timestamp("2024-02-29")] = 55.0   # Thursday
    etf[pd.Timestamp("2024-03-29")] = 52.0   # Friday, the ETF's last trading day in March
    return pd.DataFrame({"S": etf, "B": btc}).loc[:end]


def test_monthly_log_returns_with_seven_day_bitcoin_data():
    returns = portfolio.monthly_log_returns(btc_style_prices())
    # January has no previous month, so the first return is February's
    assert list(returns.index) == [pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-31")]
    assert returns.loc["2024-02-29", "B"] == pytest.approx(math.log(110 / 100))
    assert returns.loc["2024-03-31", "B"] == pytest.approx(math.log(99 / 110))
    assert returns.loc["2024-02-29", "S"] == pytest.approx(math.log(55 / 50))
    # The ETF's March price is its last weekday price (Fri 29 Mar), not a missing Sunday
    assert returns.loc["2024-03-31", "S"] == pytest.approx(math.log(52 / 55))


def test_unfinished_last_month_is_dropped():
    # Data stops on 10 April: April is not over, so it must not count as a monthly return
    returns = portfolio.monthly_log_returns(btc_style_prices(end="2024-04-10"))
    assert returns.index[-1] == pd.Timestamp("2024-03-31")
    # Data stopping on the last Friday of a month (ETF only) still counts as a full month
    etf_only = btc_style_prices()[["S"]].loc[:"2024-03-29"]
    assert portfolio.monthly_log_returns(etf_only).index[-1] == pd.Timestamp("2024-03-31")
    # Data stopping on Sunday 31 March: bitcoin AND the ETF (Fri 29 Mar) have finished March
    assert portfolio.monthly_log_returns(btc_style_prices()).index[-1] == pd.Timestamp("2024-03-31")


def test_month_is_unfinished_until_every_asset_reaches_its_last_weekday():
    # Regression: data ending Wed 27 March (bitcoin trades every day, so the last row is a
    # bitcoin date) used to count March as finished because it was within 3 days of the 31st.
    # The ETF still has Thu 28 and Fri 29 March to trade, so March must be dropped.
    returns = portfolio.monthly_log_returns(btc_style_prices(end="2024-03-27"))
    assert returns.index[-1] == pd.Timestamp("2024-02-29")
    # Bitcoin is up to date (Sun 31 Mar) but the ETF's Friday price is missing: still unfinished
    prices = btc_style_prices()
    prices.loc["2024-03-29", "S"] = np.nan
    assert portfolio.monthly_log_returns(prices).index[-1] == pd.Timestamp("2024-02-29")


def test_only_months_where_every_asset_has_data_are_used():
    prices = btc_style_prices()
    prices.loc[:"2024-02-14", "B"] = np.nan  # bitcoin "starts" in mid-February
    returns = portfolio.monthly_log_returns(prices)
    # Feb needs a January bitcoin price to make a return, so March is the first complete month
    assert list(returns.index) == [pd.Timestamp("2024-03-31")]
    assert not returns.isna().any().any()


def test_estimation_window_keeps_the_last_n_years():
    months = pd.date_range("2011-10-31", periods=180, freq="ME")  # 15 years of month ends
    returns = pd.DataFrame({"A": np.arange(180) / 1000}, index=months)
    last_5 = portfolio.estimation_window(returns, 5)
    assert len(last_5) == 60
    assert last_5.index[0] == months[-60]
    assert last_5.index[-1] == months[-1]
    assert len(portfolio.estimation_window(returns, 10)) == 120
    assert len(portfolio.estimation_window(returns, None)) == 180
    assert portfolio.estimation_window(returns.iloc[:0], 5).empty


def test_estimate_inputs():
    returns = pd.DataFrame({"A": [0.01, 0.03], "B": [0.02, 0.00]})
    mean, cov = portfolio.estimate_inputs(returns)
    assert mean == pytest.approx([0.02, 0.01])
    # Sample variance of A: ((0.01-0.02)^2 + (0.03-0.02)^2) / (2-1) = 0.0002; cov(A, B) = -0.0002
    assert cov == pytest.approx(np.array([[0.0002, -0.0002], [-0.0002, 0.0002]]))


def test_snapshot_is_shipped_with_all_four_assets():
    prices = portfolio.load_data()
    assert list(prices.columns) == portfolio.TICKERS
    returns = portfolio.monthly_log_returns(prices)
    assert len(returns) >= 100                       # bitcoin data since 2014
    assert returns.index[0] == pd.Timestamp("2014-10-31")
    assert not returns.isna().any().any()


# ---------------------------------------------------------------------------
# Words and numbers
# ---------------------------------------------------------------------------

def test_format_helpers():
    assert portfolio.format_money(12_345.6) == "$12,346"
    assert portfolio.format_money(2_345_678) == "$2.35M"
    assert portfolio.format_money(1_165_280_000) == "$1.17B"     # not "$1,165.28M"
    assert portfolio.format_money(2.6e13) == "$26.00T"           # not "$26,000,000.00M"
    assert portfolio.format_money_short(500) == "$500"
    assert portfolio.format_money_short(20_000) == "$20k"
    assert portfolio.format_money_short(1_500_000) == "$1.5M"
    assert portfolio.format_money_short(2e9) == "$2B"            # not d3's "$2G"
    assert portfolio.format_money_short(1e12) == "$1T"
    assert portfolio.format_change(28_591, 10_000) == "+186% vs start"
    assert portfolio.format_change(8_207, 10_000) == "-18% vs start"
    assert portfolio.format_change(245_973, 10_000) == "×25 vs start"  # not "+2360% vs start"
    assert portfolio.format_chance(0) == "0%"
    assert portfolio.format_chance(0.0002) == "<0.1%"
    assert portfolio.format_chance(0.0123) == "1.2%"
    assert portfolio.years_label(1) == "1 year"
    assert portfolio.years_label(30) == "30 years"
    assert portfolio.escape_dollars("$10 to $20") == "\\$10 to \\$20"


def test_log_axis_ticks_are_round_and_not_too_many():
    # A narrow range gets 1-2-5 steps
    assert portfolio.log_axis_ticks(8_000, 300_000) == [10_000, 20_000, 50_000, 100_000, 200_000]
    # Too many 1-2-5 steps: 1-3 steps instead
    assert portfolio.log_axis_ticks(1_500, 700_000) == [3_000, 10_000, 30_000, 100_000, 300_000]
    # Many powers of ten: powers of ten only, and every second one for a huge range
    assert portfolio.log_axis_ticks(150, 15_000_000) == [1e3, 1e4, 1e5, 1e6, 1e7]
    huge = portfolio.log_axis_ticks(10, 1e12)
    assert huge == [10, 1e3, 1e5, 1e7, 1e9, 1e11]
    assert len(huge) <= 7


def test_axis_position():
    assert portfolio.axis_position(10_000, log_scale=False) == 10_000
    assert portfolio.axis_position(10_000, log_scale=True) == pytest.approx(4.0)


def test_describe_mix():
    assert portfolio.describe_mix({"SPY": 0.5, "AGG": 0.3, "GLD": 0.1, "BTC-USD": 0.1}) == \
        "50% US stocks, 30% US bonds, 10% gold and 10% bitcoin"
    assert portfolio.describe_mix({"SPY": 0.0, "AGG": 0.0, "GLD": 0.0, "BTC-USD": 1.0}) == "100% bitcoin"
    assert portfolio.describe_mix({"SPY": 0.5, "AGG": 0.5, "GLD": 0.0, "BTC-USD": 0.0}) == \
        "50% US stocks and 50% US bonds"


def make_stats(median, p5, chance, growth=0.05, n_sims=1000):
    return {"median": median, "p5": p5, "p95": median * 2, "chance_of_loss": chance,
            "median_growth": growth, "n_sims": n_sims}


def test_insight_covers_gain_loss_and_no_loss_cases():
    weights = {"SPY": 1.0, "AGG": 0.0, "GLD": 0.0, "BTC-USD": 0.0}
    corr = pd.DataFrame(np.eye(2), index=TWO_ASSETS, columns=TWO_ASSETS)

    no_loss = portfolio.write_insight("Fan chart", make_stats(20_000, 12_000, 0.0), weights,
                                      "Goldilocks", 10, 10_000, corr, "2021-2026")
    assert "grows $10,000 to $20,000 after 10 years (+5.0% a year)" in no_loss
    assert "none of the 1,000 simulations loses money" in no_loss

    small_loss = portfolio.describe_downside(make_stats(20_000, 11_000, 0.012), 10_000)
    assert "still worth $11,000" in small_loss and "1.2%" in small_loss

    big_loss = portfolio.write_insight(portfolio.VIEW_OUTCOMES, make_stats(9_000, 6_000, 0.6, growth=-0.02),
                                       weights, "Inflation shock", 5, 10_000, corr, "2021-2026")
    assert "shrinks $10,000 to $9,000" in big_loss
    assert "-2.0% a year" in big_loss
    assert "worth just $6,000, a loss of 40%" in big_loss and "60.0%" in big_loss

    # Regression: "a 18% loss" was bad grammar; a tiny loss doesn't read "a loss of 0%"
    assert "a loss of 18%" in portfolio.describe_downside(make_stats(20_000, 8_207, 0.09), 10_000)
    assert "a loss of less than 1%" in portfolio.describe_downside(make_stats(20_000, 9_980, 0.05), 10_000)


def test_high_return_caution_names_the_fastest_asset_held():
    yearly = {"SPY": 14.2, "AGG": 1.1, "GLD": 11.1, "BTC-USD": 49.2}
    with_bitcoin = {"SPY": 0.5, "AGG": 0.3, "GLD": 0.1, "BTC-USD": 0.1}
    text = portfolio.high_return_caution(yearly, with_bitcoin, "Oct 2016 – Sep 2026")
    assert "bitcoin averaged +49% a year over Oct 2016 – Sep 2026" in text
    assert "very unlikely to last" in text
    # No bitcoin in the portfolio, or no asset above the limit: no caution
    without_bitcoin = {"SPY": 0.6, "AGG": 0.4, "GLD": 0.0, "BTC-USD": 0.0}
    assert portfolio.high_return_caution(yearly, without_bitcoin, "Oct 2016 – Sep 2026") == ""
    calm = {"SPY": 12.8, "AGG": -0.6, "GLD": 16.8, "BTC-USD": 12.9}
    assert portfolio.high_return_caution(calm, with_bitcoin, "Oct 2021 – Sep 2026") == ""


def test_average_yearly_returns():
    returns = pd.DataFrame({"A": [0.01, 0.03], "B": [0.0, -0.01]})
    averages = portfolio.average_yearly_returns(returns)
    assert averages["A"] == pytest.approx(24.0)  # 0.02 a month x 12 = 24% a year
    assert averages["B"] == pytest.approx(-6.0)


def test_correlation_insight_names_the_extreme_pairs():
    corr = pd.DataFrame([[1.0, 0.6, -0.2], [0.6, 1.0, 0.1], [-0.2, 0.1, 1.0]],
                        index=["SPY", "AGG", "GLD"], columns=["SPY", "AGG", "GLD"])
    extremes = portfolio.extreme_correlations(corr)
    assert extremes["highest"] == ("SPY", "AGG", 0.6)
    assert extremes["lowest"] == ("SPY", "GLD", -0.2)
    text = portfolio.describe_correlations(corr, "Oct 2021 – Sep 2026")
    assert "US stocks and US bonds moved together the most (0.60)" in text
    assert "US stocks and gold the least (-0.20)" in text
    # A matrix with nothing usable gives a polite sentence, not an error
    empty = pd.DataFrame(np.nan, index=TWO_ASSETS, columns=TWO_ASSETS)
    assert portfolio.extreme_correlations(empty) == {}
    assert "can't be measured" in portfolio.describe_correlations(empty, "2020")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def example_values():
    mean = np.array([0.006, 0.001, 0.004, 0.01])
    cov = np.diag([0.002, 0.0003, 0.002, 0.03])
    return portfolio.simulate_portfolio(mean, cov, np.array([0.5, 0.3, 0.1, 0.1]), 24, 10_000, 300, seed=5)


def test_fan_chart_bands_colours_and_single_axis():
    paths = portfolio.percentile_paths(example_values())
    paths.index = pd.date_range("2026-09-30", periods=len(paths), freq="ME")
    fig = portfolio.build_fan_chart(paths, 10_000)
    names = [trace.name for trace in fig.data if trace.showlegend is not False]
    assert names == ["Middle 90% of outcomes", "Middle 50% of outcomes", "Median", "Start $10,000"]
    fills = [trace.fillcolor for trace in fig.data if trace.fill == "tonexty"]
    assert fills == [theme.SEQUENTIAL_SCALE[0][1], theme.SEQUENTIAL_SCALE[1][1]]
    median = [trace for trace in fig.data if trace.name == "Median"][0]
    assert median.line.width == 2
    assert "yaxis2" not in fig.layout.to_plotly_json()
    assert fig.layout.yaxis.type != "log"
    assert fig.layout.yaxis.tickformat is None  # Plotly's own labels: $20k, $1M, $2B (never "$2G")


def test_fan_chart_on_a_log_scale():
    paths = portfolio.percentile_paths(example_values())
    paths.index = pd.date_range("2026-09-30", periods=len(paths), freq="ME")
    fig = portfolio.build_fan_chart(paths, 10_000, log_scale=True)
    assert fig.layout.yaxis.type == "log"
    assert "log scale" in fig.layout.yaxis.title.text
    assert all(label.startswith("$") for label in fig.layout.yaxis.ticktext)
    # On a log axis Plotly places the median label at log10(value)
    median_label = fig.layout.annotations[0]
    assert median_label.y == pytest.approx(math.log10(paths["p50"].iloc[-1]))


def test_histogram_chart_has_three_labelled_reference_lines():
    values = example_values()
    stats = portfolio.outcome_stats(values, 10_000, 2)
    fig = portfolio.build_histogram(portfolio.histogram_table(values[:, -1]), stats, 10_000, 2)
    assert len(fig.layout.shapes) == 3
    labels = " ".join(annotation.text for annotation in fig.layout.annotations)
    assert "Median" in labels and "Start $10,000" in labels and "5% worst case" in labels
    assert fig.data[0].marker.color == theme.BLUE
    assert fig.layout.xaxis.type != "log"
    assert fig.layout.yaxis.range[1] <= 100  # never a tick above 100% of simulations


def test_histogram_on_a_log_scale_keeps_reference_lines_apart():
    final = skewed_outcomes()
    values = np.column_stack([np.full(len(final), 10_000.0), final])
    stats = portfolio.outcome_stats(values, 10_000, 10)
    assert portfolio.use_log_scale(stats)
    bins = portfolio.histogram_table(final, log_scale=True)
    fig = portfolio.build_histogram(bins, stats, 10_000, 10, log_scale=True)
    assert fig.layout.xaxis.type == "log"
    assert "log scale" in fig.layout.xaxis.title.text
    # Lines (shapes) use dollars; their labels (annotations) use log10(dollars) on a log axis
    assert fig.layout.shapes[1].x0 == 10_000
    assert fig.layout.annotations[1].x == pytest.approx(4.0)
    # The x range (in log10 units) covers every bin and the starting amount
    low, high = fig.layout.xaxis.range
    assert low < math.log10(bins["from"].iloc[0]) and high > math.log10(bins["to"].iloc[-1])


def test_correlation_heatmap_labels_every_cell():
    corr = pd.DataFrame(np.eye(4) * 0.5 + 0.5, index=portfolio.TICKERS, columns=portfolio.TICKERS)
    fig = portfolio.build_correlation_heatmap(corr, "Oct 2021 – Sep 2026")
    heatmap = fig.data[0]
    assert heatmap.zmin == -1 and heatmap.zmax == 1
    assert len(fig.layout.annotations) == 16
    assert fig.layout.annotations[0].text == "1.00"


# ---------------------------------------------------------------------------
# The panel in a (headless) Streamlit app
# ---------------------------------------------------------------------------

def _panel_app():
    from dashboard.panels import portfolio

    portfolio.render()


def assert_no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def run_panel():
    at = AppTest.from_function(_panel_app, default_timeout=120)
    at.run()
    assert_no_errors(at)
    return at


def test_panel_renders_from_snapshot():
    at = run_panel()
    assert len(at.metric) == 3
    assert at.metric[0].label == "Median after 10 years"
    assert at.selectbox(key="portfolio_scenario").value == "Historical"
    assert at.selectbox(key="portfolio_window").value == "Last 5 years"
    assert at.slider(key="portfolio_weight_SPY").value == 50
    assert at.number_input(key="portfolio_start").value == 10_000
    assert at.select_slider(key="portfolio_simulations").value == 2000
    assert at.radio(key="portfolio_view").value == "Fan chart"
    widgets = (list(at.slider) + list(at.selectbox) + list(at.number_input) + list(at.select_slider)
               + list(at.radio) + list(at.button))
    assert len(widgets) == 11
    for widget in widgets:
        assert widget.key.startswith("portfolio_")
    assert "median simulation" in at.info[0].value  # the insight is shown
    assert any("Illustrative assumptions" in caption.value for caption in at.caption)


def test_results_are_the_same_on_every_rerun():
    at = run_panel()
    first = [metric.value for metric in at.metric]
    at.run()
    assert [metric.value for metric in at.metric] == first


def test_every_control_and_view_can_be_changed():
    at = run_panel()

    for ticker, weight in [("SPY", 100), ("AGG", 0), ("GLD", 35), ("BTC-USD", 5)]:
        at.slider(key=f"portfolio_weight_{ticker}").set_value(weight).run()
        assert_no_errors(at)
    assert any("rescaled" in caption.value for caption in at.caption)  # 140% -> rescaled to 100%

    for scenario in portfolio.SCENARIOS:
        at.selectbox(key="portfolio_scenario").set_value(scenario).run()
        assert_no_errors(at)
        assert len(at.metric) == 3

    for window in portfolio.ESTIMATION_WINDOWS:
        at.selectbox(key="portfolio_window").set_value(window).run()
        assert_no_errors(at)

    for years in [1, 30, 7]:
        at.slider(key="portfolio_horizon").set_value(years).run()
        assert_no_errors(at)
    assert at.metric[0].label == "Median after 7 years"

    at.number_input(key="portfolio_start").set_value(250_000).run()
    assert_no_errors(at)

    for n_sims in [500, 5000, 1000]:
        at.select_slider(key="portfolio_simulations").set_value(n_sims).run()
        assert_no_errors(at)

    for view in portfolio.VIEWS:
        at.radio(key="portfolio_view").set_value(view).run()
        assert_no_errors(at)
        assert len(at.metric) == 3
    assert "moved together the most" in at.info[0].value  # correlation insight


def chart_spec(at):
    """The Plotly figure (as a dict) that the panel drew in an AppTest run."""
    return json.loads(at.get("plotly_chart")[0].proto.spec)


def test_bitcoin_heavy_portfolio_gets_log_scales_and_a_caution():
    at = run_panel()
    assert chart_spec(at)["layout"]["yaxis"].get("type") != "log"  # default mix: linear axis
    for ticker in ["SPY", "AGG", "GLD"]:
        at.slider(key=f"portfolio_weight_{ticker}").set_value(0)
    at.slider(key="portfolio_weight_BTC-USD").set_value(100)
    at.selectbox(key="portfolio_window").set_value("Last 10 years")
    at.slider(key="portfolio_horizon").set_value(30)
    at.run()
    assert_no_errors(at)
    assert chart_spec(at)["layout"]["yaxis"]["type"] == "log"
    assert at.metric[0].value.endswith("B")              # billions, not "$25,277.36M"
    assert at.metric[0].delta.startswith("×")            # a multiple, not "+252773518%"
    assert any("Caution:" in caption.value and "bitcoin averaged" in caption.value for caption in at.caption)
    at.radio(key="portfolio_view").set_value(portfolio.VIEW_OUTCOMES).run()
    assert_no_errors(at)
    assert chart_spec(at)["layout"]["xaxis"]["type"] == "log"


def test_new_random_draw_changes_the_seed_and_the_results():
    at = run_panel()
    assert at.session_state["portfolio_seed"] == portfolio.DEFAULT_SEED
    before = [metric.value for metric in at.metric]
    at.button(key="portfolio_new_draw").click().run()
    assert_no_errors(at)
    assert at.session_state["portfolio_seed"] == portfolio.DEFAULT_SEED + 1
    assert [metric.value for metric in at.metric] != before
    assert any("Random draw #2" in caption.value for caption in at.caption)


def test_all_zero_weights_show_a_friendly_message():
    at = run_panel()
    for ticker in portfolio.TICKERS:
        at.slider(key=f"portfolio_weight_{ticker}").set_value(0)
    at.run()
    assert_no_errors(at)
    assert len(at.metric) == 0
    assert "All the weights are 0%" in at.info[0].value

    # Back to one asset: the panel works again, with a 100% bitcoin portfolio
    at.slider(key="portfolio_weight_BTC-USD").set_value(5).run()
    assert_no_errors(at)
    assert len(at.metric) == 3
    assert "100% bitcoin" in at.info[0].value


def test_extreme_combination_of_controls():
    at = run_panel()
    at.slider(key="portfolio_weight_SPY").set_value(0)
    at.slider(key="portfolio_weight_AGG").set_value(0)
    at.slider(key="portfolio_weight_GLD").set_value(0)
    at.slider(key="portfolio_weight_BTC-USD").set_value(100)
    at.selectbox(key="portfolio_scenario").set_value("Crypto winter")
    at.selectbox(key="portfolio_window").set_value("All data")
    at.slider(key="portfolio_horizon").set_value(30)
    at.number_input(key="portfolio_start").set_value(portfolio.MAX_START_AMOUNT)
    at.select_slider(key="portfolio_simulations").set_value(5000)
    at.radio(key="portfolio_view").set_value(portfolio.VIEW_OUTCOMES)
    at.run()
    assert_no_errors(at)
    assert len(at.metric) == 3


def test_panel_without_data_shows_a_message(monkeypatch):
    def no_data():
        raise DataUnavailableError("no snapshot")

    monkeypatch.setattr(portfolio, "_cached_returns", no_data)
    at = AppTest.from_function(_panel_app, default_timeout=120)
    at.run()
    assert not at.exception
    assert "isn't available" in at.warning[0].value


def test_panel_with_too_little_history(monkeypatch):
    months = pd.date_range("2025-01-31", periods=12, freq="ME")
    short = pd.DataFrame(0.01, index=months, columns=portfolio.TICKERS)
    monkeypatch.setattr(portfolio, "_cached_returns", lambda: short)
    at = AppTest.from_function(_panel_app, default_timeout=120)
    at.run()
    assert_no_errors(at)
    assert "Only 12 months" in at.info[0].value
    at.selectbox(key="portfolio_window").set_value("All data").run()
    assert_no_errors(at)


def test_window_options_are_short_enough_to_fit():
    at = run_panel()
    # "All data" is shown with its start month, short enough for a half-width drop-down
    assert at.selectbox(key="portfolio_window").options == ["Last 5 years", "Last 10 years", "Since Oct 2014"]
    assert at.radio(key="portfolio_view").options == ["Fan chart", "Outcome distribution", "Correlations"]


def test_panel_with_no_complete_months(monkeypatch):
    # e.g. a failed bitcoin download: no month has prices for all four assets
    empty = pd.DataFrame(columns=portfolio.TICKERS, index=pd.DatetimeIndex([]), dtype=float)
    monkeypatch.setattr(portfolio, "_cached_returns", lambda: empty)
    at = AppTest.from_function(_panel_app, default_timeout=120)
    at.run()
    assert_no_errors(at)
    assert "Only 0 months" in at.info[0].value
