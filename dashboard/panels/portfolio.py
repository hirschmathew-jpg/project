"""Panel 4 - Portfolio Scenario Lab.

Question: what could this mean for an investor's portfolio?

The panel simulates thousands of possible futures for a portfolio of four
assets - US stocks, US bonds, gold and bitcoin - with a Monte Carlo model:

1. Measure how the assets behaved in the past (the "estimation window"):
   the average monthly return of each asset, how much it jumped around
   (volatility) and how the assets moved together (covariance/correlation).
2. Tilt those numbers with a scenario, e.g. an inflation shock or a crash.
   The scenarios are illustrative assumptions, not forecasts.
3. Draw random monthly returns with those averages, volatilities and
   correlations, month after month, for thousands of possible futures.
4. Grow the portfolio through each future, rebalancing back to the target
   weights every month, and summarise the spread of final values.

Module layout: constants -> data loading -> model and analysis functions
(NumPy/pandas only, no Streamlit, tested in tests/test_portfolio.py) ->
chart builders (return Plotly figures, no Streamlit) -> render().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme, ui
from dashboard.data import load_prices

NUMBER = 4
SLUG = "portfolio"
TITLE = "Portfolio Scenario Lab"
OWNER = "Team member 4"  # panel owner: put your name here
QUESTION = "What could this mean for an investor's portfolio?"

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# Data: Yahoo Finance tickers (each lives at https://finance.yahoo.com/quote/<TICKER>)
SNAPSHOT_NAME = "portfolio_prices"  # saved as data/portfolio_prices.csv
DATA_START = "2010-01-01"
# Names as they appear in the middle of a sentence ("10% gold and 10% bitcoin")
ASSET_NAMES = {
    "SPY": "US stocks",    # SPDR S&P 500 ETF: the 500 largest US companies
    "AGG": "US bonds",     # iShares Core US Aggregate Bond ETF: government and company bonds
    "GLD": "gold",         # SPDR Gold Shares: tracks the gold price
    "BTC-USD": "bitcoin",  # bitcoin in US dollars (trades 7 days a week, data from Sep 2014)
}
TICKERS = list(ASSET_NAMES)
SHORT_LABELS = {"SPY": "Stocks (SPY)", "AGG": "Bonds (AGG)", "GLD": "Gold (GLD)", "BTC-USD": "Bitcoin (BTC)"}
DEFAULT_WEIGHTS = {"SPY": 50, "AGG": 30, "GLD": 10, "BTC-USD": 10}  # % of the portfolio

# Scenarios: ILLUSTRATIVE ASSUMPTIONS, NOT FORECASTS. Each one says, for every asset:
#   drift_pp  - change to the average yearly (log) return, in percentage points
#   vol_mult  - volatility multiplier (1.0 = as in history, 1.5 = 50% bigger swings)
#   shock_pct - one-off gain or loss in the first simulated month, in %
SCENARIOS = {
    "Historical": {
        "summary": "Returns, volatility and correlations exactly as in the estimation window.",
        "drift_pp":  {"SPY": 0.0, "AGG": 0.0, "GLD": 0.0, "BTC-USD": 0.0},
        "vol_mult":  {"SPY": 1.0, "AGG": 1.0, "GLD": 1.0, "BTC-USD": 1.0},
        "shock_pct": {"SPY": 0.0, "AGG": 0.0, "GLD": 0.0, "BTC-USD": 0.0},
    },
    "Inflation shock": {
        "summary": ("Inflation flares up again and rates rise: stocks, bonds and bitcoin earn less, "
                    "gold earns more, and every asset swings more."),
        "drift_pp":  {"SPY": -6.0, "AGG": -3.0, "GLD": 4.0, "BTC-USD": -8.0},
        "vol_mult":  {"SPY": 1.3, "AGG": 1.5, "GLD": 1.2, "BTC-USD": 1.3},
        "shock_pct": {"SPY": 0.0, "AGG": 0.0, "GLD": 0.0, "BTC-USD": 0.0},
    },
    "Stock market crash": {
        "summary": ("A sudden crash in month 1 (stocks −30%, bitcoin −45%, bonds +3%, gold +5% "
                    "as safe havens), then history repeats."),
        "drift_pp":  {"SPY": 0.0, "AGG": 0.0, "GLD": 0.0, "BTC-USD": 0.0},
        "vol_mult":  {"SPY": 1.0, "AGG": 1.0, "GLD": 1.0, "BTC-USD": 1.0},
        "shock_pct": {"SPY": -30.0, "AGG": 3.0, "GLD": 5.0, "BTC-USD": -45.0},
    },
    "Crypto winter": {
        "summary": ("Bitcoin falls 65% in month 1 (about its 2022 drop) and then grows "
                    "20 points a year less than in history. Other assets as in history."),
        "drift_pp":  {"SPY": 0.0, "AGG": 0.0, "GLD": 0.0, "BTC-USD": -20.0},
        "vol_mult":  {"SPY": 1.0, "AGG": 1.0, "GLD": 1.0, "BTC-USD": 1.0},
        "shock_pct": {"SPY": 0.0, "AGG": 0.0, "GLD": 0.0, "BTC-USD": -65.0},
    },
    "Goldilocks": {
        "summary": ("Steady growth with calm inflation: stocks, bonds and bitcoin earn a little more "
                    "and every asset swings 20% less."),
        "drift_pp":  {"SPY": 2.0, "AGG": 1.0, "GLD": 0.0, "BTC-USD": 2.0},
        "vol_mult":  {"SPY": 0.8, "AGG": 0.8, "GLD": 0.8, "BTC-USD": 0.8},
        "shock_pct": {"SPY": 0.0, "AGG": 0.0, "GLD": 0.0, "BTC-USD": 0.0},
    },
}
DEFAULT_SCENARIO = "Historical"

# Estimation window: label -> number of years of history (None = all of it).
# The default covers the post-2021 inflation and rate-hike era of panels 1-3.
ESTIMATION_WINDOWS = {"Last 5 years": 5, "Last 10 years": 10, "All data": None}
DEFAULT_WINDOW = "Last 5 years"
MIN_MONTHS = 24  # need at least 2 years of monthly returns for sensible estimates
# An asset that averaged more than this (% a year) in the window gets a caution note:
# the simulation assumes the average carries on, which is very unlikely for, e.g., +49% a year.
HIGH_RETURN_WARNING = 25

DEFAULT_HORIZON_YEARS = 10
MAX_HORIZON_YEARS = 30
DEFAULT_START_AMOUNT = 10_000
MAX_START_AMOUNT = 1_000_000
SIMULATION_CHOICES = [500, 1000, 2000, 5000]
DEFAULT_SIMULATIONS = 2000
DEFAULT_SEED = 42  # fixed, so the results are the same every time the page reruns

PERCENTILES = [5, 25, 50, 75, 95]
HISTOGRAM_BINS = 30
HISTOGRAM_TOP_CUT = 99  # on a linear axis the histogram stops at the 99th percentile (long right tail)
# The fan chart and histogram switch to a log scale when the 5% best case is more than
# 10 times the 5% worst case (see use_log_scale)
LOG_SCALE_RATIO = 10

VIEW_FAN = "Fan chart"
VIEW_OUTCOMES = "Outcome distribution"
VIEW_CORRELATIONS = "Correlations"
VIEWS = [VIEW_FAN, VIEW_OUTCOMES, VIEW_CORRELATIONS]

# Colours (all from the shared theme). The fan uses steps of the single-hue blue ramp:
# light = wide 5-95% band, mid = 25-75% band, dark = the median line.
OUTER_BAND_COLOUR = theme.SEQUENTIAL_SCALE[0][1]
INNER_BAND_COLOUR = theme.SEQUENTIAL_SCALE[1][1]
MEDIAN_COLOUR = theme.SEQUENTIAL_SCALE[4][1]

# Widget keys (all start with the slug so they are unique across the dashboard)
KEY_SCENARIO = f"{SLUG}_scenario"
KEY_WINDOW = f"{SLUG}_window"
KEY_HORIZON = f"{SLUG}_horizon"
KEY_START = f"{SLUG}_start"
KEY_SIMULATIONS = f"{SLUG}_simulations"
KEY_VIEW = f"{SLUG}_view"
KEY_NEW_DRAW = f"{SLUG}_new_draw"
KEY_SEED = f"{SLUG}_seed"  # not a widget: the random seed kept in session_state


def weight_key(ticker: str) -> str:
    """Session-state key of the weight slider for one asset, e.g. 'portfolio_weight_SPY'."""
    return f"{SLUG}_weight_{ticker}"


# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------

def load_data(refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted close prices of the four assets (data/portfolio_prices.csv unless refresh=True).

    "Adjusted" means dividends and splits are included, so price changes are total returns.
    """
    return load_prices(SNAPSHOT_NAME, TICKERS, start=DATA_START, refresh=refresh)


def refresh_data() -> None:
    """Download fresh data for this panel (used by the refresh button and scripts/update_data.py)."""
    load_data(refresh=True)


# ---------------------------------------------------------------------------
# 3. Model and analysis (no Streamlit here, so it can be unit tested)
# ---------------------------------------------------------------------------

def month_end_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Last available price of every month for each asset.

    Bitcoin trades 7 days a week but the ETFs only on weekdays, so on a
    weekend the ETF columns are empty. resample("ME").last() takes the last
    price that exists in each month, for each column separately.
    The newest month is dropped if it is not finished yet (data ending
    mid-month would give a half-month "monthly" return).
    """
    monthly = daily_prices.resample("ME").last()
    if monthly.empty:
        return monthly
    # The newest month counts as finished once EVERY asset has a price on the month's
    # last weekday (e.g. Fri 29 Mar 2024, as Sun 31 Mar is not a trading day for the ETFs).
    # Each asset is checked on its own: bitcoin trades every day, so the last date in
    # the table can be a Sunday even though the ETFs stopped on Friday.
    last_weekday = pd.offsets.BMonthEnd().rollback(monthly.index[-1])
    for ticker in daily_prices.columns:
        last_price_date = daily_prices[ticker].last_valid_index()
        if last_price_date is not None and last_price_date < last_weekday:
            return monthly.iloc[:-1]  # this asset hasn't reached the end of the month yet
    return monthly


def monthly_log_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Monthly log returns, keeping only months where every asset has data.

    log return = ln(price this month / price last month). Log returns add up
    over time (two months of +10% and -10% in logs give 0), which makes them
    the natural thing to model with a normal distribution.
    """
    monthly = month_end_prices(daily_prices)
    log_returns = np.log(monthly / monthly.shift(1))
    return log_returns.dropna()  # drops the first month and months before bitcoin existed


def estimation_window(returns: pd.DataFrame, years: int | None) -> pd.DataFrame:
    """The last ``years`` years of monthly returns (all of them if ``years`` is None)."""
    if years is None or returns.empty:
        return returns
    cutoff = returns.index[-1] - pd.DateOffset(years=years)
    return returns[returns.index > cutoff]


def estimate_inputs(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector and covariance matrix of monthly log returns (one row/column per asset)."""
    mean_monthly = returns.mean().to_numpy()
    cov_monthly = returns.cov().to_numpy()
    return mean_monthly, cov_monthly


def average_yearly_returns(returns: pd.DataFrame) -> dict[str, float]:
    """Average yearly log return of each asset in %, e.g. {"SPY": 12.8, ...} (monthly mean x 12)."""
    averages = {}
    for ticker in returns.columns:
        averages[ticker] = float(returns[ticker].mean() * 12 * 100)
    return averages


def apply_scenario(mean_monthly: np.ndarray, cov_monthly: np.ndarray, scenario: dict,
                   tickers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Tilt the historical inputs with a scenario's drift shifts and volatility multipliers.

    * Drift: the scenario gives a change in the yearly return in percentage
      points, so it is divided by 100 (to a decimal) and by 12 (to a month).
    * Volatility: multiplying asset i's volatility by m_i multiplies the
      covariance of assets i and j by m_i x m_j (the variance by m_i squared).
      Correlations stay the same. np.outer builds that table of m_i x m_j.
    """
    drift_shift = np.array([scenario["drift_pp"][ticker] for ticker in tickers]) / 100 / 12
    vol_multiplier = np.array([scenario["vol_mult"][ticker] for ticker in tickers])
    new_mean = mean_monthly + drift_shift
    new_cov = cov_monthly * np.outer(vol_multiplier, vol_multiplier)
    return new_mean, new_cov


def scenario_shocks(scenario: dict, tickers: list[str]) -> np.ndarray:
    """One-off first-month shocks as decimals, e.g. -30% -> -0.30."""
    return np.array([scenario["shock_pct"][ticker] for ticker in tickers]) / 100


def normalise_weights(raw_weights: dict[str, float]) -> dict[str, float] | None:
    """Scale the weights so they add up to 1 (100%). Returns None if they are all zero.

    Example: {"SPY": 60, "AGG": 60} -> {"SPY": 0.5, "AGG": 0.5}.
    """
    total = sum(raw_weights.values())
    if total <= 0:
        return None
    weights = {}
    for ticker, weight in raw_weights.items():
        weights[ticker] = weight / total
    return weights


def simulate_portfolio(mean_monthly: np.ndarray, cov_monthly: np.ndarray, weights: np.ndarray,
                       months: int, start_value: float, n_sims: int, seed: int,
                       shocks: np.ndarray | None = None) -> np.ndarray:
    """Monte Carlo simulation of a portfolio rebalanced to fixed weights every month.

    Each month, for every simulation at once (NumPy vectorisation):
      1. draw correlated log returns for all assets from a multivariate normal
         distribution with the given mean vector and covariance matrix,
      2. turn them into growth factors exp(r), e.g. exp(0.02) = 1.0202 (+2.02%),
      3. in month 1 only, apply any one-off scenario shock: growth x (1 + shock),
      4. portfolio growth = sum of weight_i x growth_i (rebalanced to target weights),
      5. value this month = value last month x portfolio growth.

    Returns an array of shape (n_sims, months + 1); column 0 is the starting value.
    The same ``seed`` always gives exactly the same results.
    """
    rng = np.random.default_rng(seed)
    if shocks is None:
        shocks = np.zeros(len(weights))

    values = np.empty((n_sims, months + 1))
    values[:, 0] = start_value
    for month in range(1, months + 1):
        asset_log_returns = rng.multivariate_normal(mean_monthly, cov_monthly, size=n_sims)
        asset_growth = np.exp(asset_log_returns)       # shape (n_sims, n_assets)
        if month == 1:
            asset_growth = asset_growth * (1 + shocks)  # one-off shock as a simple return
        portfolio_growth = asset_growth @ weights       # weighted sum across the assets
        values[:, month] = values[:, month - 1] * portfolio_growth
    return values


def percentile_paths(values: np.ndarray, percentiles: list[int] = PERCENTILES) -> pd.DataFrame:
    """Percentiles of portfolio value at every month: one column per percentile ("p5", "p50"...).

    The 5th percentile is the value that 5% of simulations end below.
    """
    table = {}
    for p in percentiles:
        table[f"p{p}"] = np.percentile(values, p, axis=0)  # across simulations, month by month
    return pd.DataFrame(table)


def outcome_stats(values: np.ndarray, start_value: float, years: int) -> dict:
    """Headline numbers about the final portfolio values.

    * median: the middle outcome (half of the simulations end above it)
    * p5 / p95: the 5% worst-case and 5% best-case values
    * chance_of_loss: share of simulations that end below the starting amount (0 to 1)
    * median_growth: median annualised growth rate, (final / start) ** (1 / years) - 1
    """
    final = values[:, -1]
    growth_rates = (final / start_value) ** (1 / years) - 1
    return {
        "median": float(np.median(final)),
        "p5": float(np.percentile(final, 5)),
        "p95": float(np.percentile(final, 95)),
        "chance_of_loss": float(np.mean(final < start_value)),
        "median_growth": float(np.median(growth_rates)),
        "n_sims": len(final),
    }


def use_log_scale(stats: dict) -> bool:
    """True when the outcomes are so spread out that a normal (linear) money axis would squash them.

    Compounding multiplies money, so with a volatile asset such as bitcoin the lucky
    simulations can end 100 times higher than the unlucky ones. On a linear axis most
    simulations would then be squashed together near $0. On a log scale every x10
    step takes the same space ($1k -> $10k looks as big as $10k -> $100k).
    """
    if stats["p5"] <= 0:
        return False
    return stats["p95"] / stats["p5"] > LOG_SCALE_RATIO


def histogram_table(final_values: np.ndarray, n_bins: int = HISTOGRAM_BINS,
                    top_cut: float = HISTOGRAM_TOP_CUT, log_scale: bool = False) -> pd.DataFrame:
    """Histogram of final values: columns 'from', 'to' and 'share_pct' (% of ALL simulations).

    * Linear scale: equally wide bins from the lowest value up to the ``top_cut``
      percentile, because a few lucky simulations end far to the right.
    * Log scale: every bin covers the same RATIO (e.g. $1,000-$1,500 and
      $10,000-$15,000), made with np.geomspace. On a log axis these bins look
      equally wide, and the whole range fits, so no simulation is left out.
    """
    lowest = float(np.min(final_values))
    if log_scale:
        highest = float(np.max(final_values))
    else:
        highest = float(np.percentile(final_values, top_cut))
    if highest - lowest < 1e-9:  # every simulation ended at the same value
        lowest, highest = lowest * 0.95, lowest * 1.05
    if log_scale:
        edges = np.geomspace(lowest, highest, n_bins + 1)
    else:
        edges = np.linspace(lowest, highest, n_bins + 1)
    counts, edges = np.histogram(final_values, bins=edges)
    return pd.DataFrame({
        "from": edges[:-1],
        "to": edges[1:],
        "share_pct": counts / len(final_values) * 100,
    })


def extreme_correlations(corr: pd.DataFrame) -> dict:
    """The most and the least correlated pair of different assets.

    Returns {"highest": (asset_a, asset_b, value), "lowest": (...)}, or {} if
    the matrix has no usable pairs (e.g. missing values).
    """
    pairs = []
    names = list(corr.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):  # j > i: each pair once, skip the diagonal
            value = corr.iloc[i, j]
            if not pd.isna(value):
                pairs.append((names[i], names[j], float(value)))
    if not pairs:
        return {}
    highest = pairs[0]
    lowest = pairs[0]
    for pair in pairs:
        if pair[2] > highest[2]:
            highest = pair
        if pair[2] < lowest[2]:
            lowest = pair
    return {"highest": highest, "lowest": lowest}


def format_money(value: float) -> str:
    """'$12,345', or '$1.23M' / '$4.56B' / '$7.89T' for very big amounts (keeps the KPI cards readable)."""
    if abs(value) >= 1e12:
        return f"${value / 1e12:,.2f}T"
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.2f}M"
    return f"${value:,.0f}"


def format_money_short(value: float) -> str:
    """Short axis label for a round amount: $500, $20k, $1.5M, $2B, $1T, $10,000T."""
    if abs(value) >= 1e15:
        # Beyond $1,000T keep counting in trillions with commas: ':g' below would
        # switch to scientific notation ('$1e+06T') for such huge numbers
        return f"${value / 1e12:,.0f}T"
    sizes = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")]
    for size, letter in sizes:
        if abs(value) >= size:
            return f"${value / size:g}{letter}"  # :g drops needless zeros: 20.0 -> '20'
    return f"${value:g}"


def format_change(value: float, start_value: float) -> str:
    """KPI delta versus the starting amount: '+186% vs start' or '-18% vs start'.

    Once the value is 10 times the start or more, a multiple is easier to read
    than a huge percentage: '×25 vs start' instead of '+2400% vs start'.
    """
    multiple = value / start_value
    if multiple >= 10:
        return f"×{multiple:,.0f} vs start"
    return f"{(multiple - 1) * 100:+.0f}% vs start"


def log_axis_ticks(low: float, high: float, max_ticks: int = 7) -> list[float]:
    """Round dollar values to label a log-scale axis that runs from ``low`` to ``high``.

    Tries steps of 1-2-5 ($1k, $2k, $5k, $10k...), then 1-3 ($1k, $3k, $10k...), then
    powers of ten only ($1k, $10k, $100k...), and keeps the first one that gives at
    most ``max_ticks`` labels (more would overlap on a half-width chart). For a huge
    range it labels every second power of ten.
    """
    first_power = int(np.floor(np.log10(low)))
    last_power = int(np.ceil(np.log10(high)))
    for multipliers in [[1, 2, 5], [1, 3], [1]]:
        ticks = []
        for power in range(first_power, last_power + 1):
            for multiplier in multipliers:
                value = multiplier * 10.0 ** power
                if low <= value <= high:
                    ticks.append(value)
        if len(ticks) <= max_ticks:
            return ticks
    return ticks[::2]  # still too many powers of ten: keep every second one


def format_chance(share: float) -> str:
    """A probability between 0 and 1 as a percentage: 0.0123 -> '1.2%'.

    A tiny chance that is not zero (e.g. 1 simulation in 5,000) shows as '<0.1%', not '0.0%'.
    """
    if share == 0:
        return "0%"
    if share < 0.001:
        return "<0.1%"
    return f"{share * 100:.1f}%"


def years_label(years: int) -> str:
    """'1 year' or '10 years'."""
    if years == 1:
        return "1 year"
    return f"{years} years"


def describe_mix(weights: dict[str, float]) -> str:
    """Portfolio in words, e.g. '50% US stocks, 30% US bonds and 20% gold' (zero weights left out)."""
    parts = []
    for ticker, weight in weights.items():
        if weight > 0:
            parts.append(f"{weight * 100:.0f}% {ASSET_NAMES.get(ticker, ticker)}")
    if not parts:
        return "an empty portfolio"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def describe_outcome(stats: dict, weights: dict[str, float], scenario: str, years: int,
                     start_value: float) -> str:
    """First insight sentence: the median outcome and its yearly growth rate."""
    if stats["median"] >= start_value:
        verb = "grows"
    else:
        verb = "shrinks"
    return (f"With {describe_mix(weights)} in the \"{scenario}\" scenario, the median simulation {verb} "
            f"{format_money(start_value)} to {format_money(stats['median'])} after {years_label(years)} "
            f"({stats['median_growth'] * 100:+.1f}% a year).")


def describe_downside(stats: dict, start_value: float) -> str:
    """Second insight sentence: how bad the bad outcomes are.

    The "5% worst case" is the 5th percentile: 1 in 20 simulations ends below it.
    """
    worst = format_money(stats["p5"])
    chance = format_chance(stats["chance_of_loss"])
    if stats["chance_of_loss"] == 0:
        return (f"Even the 5% worst case ({worst}) is above the starting amount: none of the "
                f"{stats['n_sims']:,} simulations loses money.")
    if stats["p5"] >= start_value:
        return (f"In the 5% worst case the portfolio is still worth {worst} (1 in 20 simulations "
                f"ends lower), and {chance} of the simulations lose money.")
    loss = (1 - stats["p5"] / start_value) * 100
    if loss < 1:
        loss_text = "less than 1%"
    else:
        loss_text = f"{loss:.0f}%"
    return (f"In the 5% worst case the portfolio is worth just {worst}, a loss of {loss_text} (1 in 20 "
            f"simulations ends even lower), and {chance} of all simulations lose money.")


def describe_correlations(corr: pd.DataFrame, period: str) -> str:
    """Insight sentence for the correlation view: the most and least linked pairs."""
    extremes = extreme_correlations(corr)
    if not extremes:
        return f"Correlations can't be measured over {period}."
    a1, b1, high = extremes["highest"]
    a2, b2, low = extremes["lowest"]
    return (f"Over {period}, {ASSET_NAMES.get(a1, a1)} and {ASSET_NAMES.get(b1, b1)} moved together "
            f"the most ({high:.2f}) and {ASSET_NAMES.get(a2, a2)} and {ASSET_NAMES.get(b2, b2)} the least "
            f"({low:.2f}); low correlations are what make a mix of assets less risky than its parts.")


def write_insight(view: str, stats: dict, weights: dict[str, float], scenario: str, years: int,
                  start_value: float, corr: pd.DataFrame, period: str) -> str:
    """The 1-2 sentence story under the chart, built from the current numbers."""
    first = describe_outcome(stats, weights, scenario, years, start_value)
    if view == VIEW_CORRELATIONS:
        return first + " " + describe_correlations(corr, period)
    return first + " " + describe_downside(stats, start_value)


def high_return_caution(yearly_returns: dict[str, float], weights: dict[str, float], period: str) -> str:
    """A caution note when an asset in the portfolio grew unusually fast in the estimation window.

    ``yearly_returns`` holds each asset's average yearly log return in % (from
    average_yearly_returns). The simulation builds on these averages, so if bitcoin
    averaged +49% a year, $10,000 can grow to billions over 30 years. Returns '' when
    no asset with a weight above 0 averaged more than HIGH_RETURN_WARNING % a year.
    """
    fastest = None
    for ticker, weight in weights.items():
        if weight > 0 and yearly_returns[ticker] > HIGH_RETURN_WARNING:
            if fastest is None or yearly_returns[ticker] > yearly_returns[fastest]:
                fastest = ticker
    if fastest is None:
        return ""
    return (f"**Caution:** {ASSET_NAMES[fastest]} averaged {yearly_returns[fastest]:+.0f}% a year over {period} "
            f"and the simulation builds on that average. Returns like that are very unlikely to last, "
            f"so treat these results as an illustration, not a forecast.")


def escape_dollars(text: str) -> str:
    """Streamlit reads $...$ as a maths formula, so "$" signs in plain text need a backslash."""
    return text.replace("$", "\\$")


def month_label(month: pd.Timestamp) -> str:
    """'Sep 2026' style label."""
    return month.strftime("%b %Y")


def axis_position(value: float, log_scale: bool) -> float:
    """Where a dollar value sits in Plotly's axis units: log10(value) on a log axis, else the value.

    Plotly wants axis ranges and annotation (label) positions in these units on a log
    axis, while data points, tick values and shapes (lines) use the plain dollar value.
    """
    if log_scale:
        return float(np.log10(value))
    return float(value)


# ---------------------------------------------------------------------------
# 4. Charts (return Plotly figures, no Streamlit here)
# ---------------------------------------------------------------------------

def style_money_axis(fig: go.Figure, axis: str, low: float, high: float, log_scale: bool) -> None:
    """Dollar labels on the x or y axis ('x' or 'y'); on a log scale with round ticks from log_axis_ticks."""
    if log_scale:
        ticks = log_axis_ticks(low, high)
        labels = []
        for tick in ticks:
            labels.append(format_money_short(tick))
        settings = dict(type="log", tickvals=ticks, ticktext=labels, automargin=True)
    else:
        # Plotly writes big numbers as $20k, $1.5M, $2B by itself
        settings = dict(tickprefix="$", separatethousands=True, automargin=True)
    if axis == "x":
        fig.update_xaxes(**settings)
    else:
        fig.update_yaxes(**settings)


def build_fan_chart(paths: pd.DataFrame, start_value: float, log_scale: bool = False) -> go.Figure:
    """Fan chart: shaded percentile bands of portfolio value over time, with the median line.

    ``paths`` is percentile_paths() output with a date index (month ends).
    Bands are drawn as a lower edge (no fill) followed by an upper edge that
    fills down to it (fill="tonexty"). ``log_scale`` puts the value axis on a
    log scale, for very spread-out outcomes (see use_log_scale).
    """
    fig = go.Figure()
    dates = paths.index

    bands = [
        # (lower, upper, legend name, colour)
        ("p5", "p95", "Middle 90% of outcomes", OUTER_BAND_COLOUR),
        ("p25", "p75", "Middle 50% of outcomes", INNER_BAND_COLOUR),
    ]
    for lower, upper, name, colour in bands:
        fig.add_trace(go.Scatter(
            x=dates, y=paths[lower], mode="lines", line=dict(width=0),
            showlegend=False, legendgroup=name, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=paths[upper], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=colour, name=name, legendgroup=name, hoverinfo="skip",
        ))

    # The median line carries the hover text for all five percentiles
    band_values = np.column_stack([paths["p5"], paths["p25"], paths["p75"], paths["p95"]])
    fig.add_trace(go.Scatter(
        x=dates, y=paths["p50"], mode="lines", name="Median", line=dict(color=MEDIAN_COLOUR, width=2),
        customdata=band_values,
        hovertemplate=("Median: $%{y:,.0f}<br>"
                       "Middle 50%: $%{customdata[1]:,.0f} – $%{customdata[2]:,.0f}<br>"
                       "Middle 90%: $%{customdata[0]:,.0f} – $%{customdata[3]:,.0f}<extra></extra>"),
    ))

    # Muted reference line at the starting amount (in the legend, so no label can collide with the fan)
    fig.add_trace(go.Scatter(
        x=[dates[0], dates[-1]], y=[start_value, start_value], mode="lines",
        name=f"Start {format_money(start_value)}", line=dict(color=theme.INK_MUTED, width=1),
        hoverinfo="skip",
    ))
    # Direct label for the final median value (the first KPI)
    final_median = float(paths["p50"].iloc[-1])
    fig.add_annotation(
        x=dates[-1], y=axis_position(final_median, log_scale), text=f"<b>{format_money(final_median)}</b>",
        showarrow=False, xanchor="right", yanchor="bottom", yshift=3,
        font=dict(size=12, color=theme.INK),
    )

    if log_scale:
        y_title = "Portfolio value ($, log scale)"
    else:
        y_title = "Portfolio value ($)"
    theme.style_figure(fig, x_title=f"Simulated future, starting from the end of {month_label(dates[0])}",
                       y_title=y_title)
    fig.update_xaxes(range=[dates[0], dates[-1]], hoverformat="%b %Y", automargin=True)
    lowest = min(float(paths["p5"].min()), start_value)
    style_money_axis(fig, "y", lowest, float(paths["p95"].max()), log_scale)
    # Top margin: room for every legend row below Plotly's toolbar
    fig.update_layout(margin=dict(t=theme.legend_top_margin(theme.legend_names(fig))))
    return fig


def build_histogram(bins: pd.DataFrame, stats: dict, start_value: float, years: int,
                    log_scale: bool = False) -> go.Figure:
    """Histogram of final portfolio values with labelled lines at the start, median and 5% worst case.

    ``bins`` is histogram_table() output; with ``log_scale`` the bins must be the
    log-scale ones and the dollar axis is drawn on a log scale.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        # Each bar starts at its bin's 'from' value (offset=0) and is 92% of the bin wide,
        # leaving a small gap. This works on both a linear and a log axis.
        x=bins["from"], width=(bins["to"] - bins["from"]) * 0.92, offset=0, y=bins["share_pct"],
        marker=dict(color=theme.BLUE, line=dict(width=0)), name="Simulations",
        customdata=np.column_stack([bins["from"], bins["to"]]),
        hovertemplate="$%{customdata[0]:,.0f} – $%{customdata[1]:,.0f}<br>"
                      "%{y:.1f}% of simulations<extra></extra>",
    ))

    # The x-axis must show the bins AND the starting amount (all simulations may end above it)
    x_low = min(float(bins["from"].iloc[0]), start_value)
    x_high = max(float(bins["to"].iloc[-1]), start_value)
    axis_low = axis_position(x_low, log_scale)
    axis_high = axis_position(x_high, log_scale)
    padding = 0.03 * (axis_high - axis_low)
    x_range = [axis_low - padding, axis_high + padding]

    # Reference lines, each labelled on its own row at the top so labels never overlap
    reference_lines = [
        # (value, label, colour, label height as a share of the plot height)
        (stats["median"], f"Median {format_money(stats['median'])}", MEDIAN_COLOUR, 0.99),
        (start_value, f"Start {format_money(start_value)}", theme.INK, 0.89),
        (stats["p5"], f"5% worst case {format_money(stats['p5'])}", theme.RED, 0.79),
    ]
    for value, label, colour, label_height in reference_lines:
        fig.add_shape(type="line", x0=value, x1=value, y0=0, y1=label_height, yref="paper",
                      line=dict(color=colour, width=2), layer="above")
        position = axis_position(value, log_scale)
        share_of_axis = (position - x_range[0]) / (x_range[1] - x_range[0])
        if share_of_axis < 0.7:
            x_anchor, x_shift = "left", 4    # label to the right of the line
        else:
            x_anchor, x_shift = "right", -4  # near the right edge: label to the left
        fig.add_annotation(
            x=position, y=label_height, yref="paper", text=label, showarrow=False,
            xanchor=x_anchor, yanchor="middle", xshift=x_shift,
            font=dict(size=11, color=theme.INK), bgcolor="rgba(252,252,251,0.85)",
        )

    # Leave the top ~30% of the plot free for the labels (but never show ticks above 100%)
    tallest_bar = max(float(bins["share_pct"].max()), 1.0)
    y_top = min(tallest_bar * 1.45, 100)
    if log_scale:
        x_title = f"Portfolio value after {years_label(years)} ($, log scale)"
    else:
        x_title = (f"Portfolio value after {years_label(years)} ($) · the luckiest {100 - HISTOGRAM_TOP_CUT}% "
                   f"(above {format_money(bins['to'].iloc[-1])}) not shown")
    theme.style_figure(fig, x_title=x_title, y_title="Share of simulations (%)", hovermode="closest")
    style_money_axis(fig, "x", x_low, x_high, log_scale)
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=[0, y_top], ticksuffix="%", automargin=True)
    fig.update_layout(showlegend=False, bargap=0)
    return fig


def build_correlation_heatmap(corr: pd.DataFrame, period: str) -> go.Figure:
    """4 x 4 heatmap of monthly return correlations, every cell labelled with its value.

    +1 = always move together, 0 = unrelated, -1 = always move in opposite directions.
    """
    labels = []
    for ticker in corr.columns:
        labels.append(SHORT_LABELS.get(ticker, ticker))

    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=labels, y=labels,
        colorscale=theme.DIVERGING_SCALE, zmin=-1, zmax=1, xgap=2, ygap=2,
        colorbar=dict(title=dict(text="Correlation", side="right"), thickness=12, len=0.9,
                      tickvals=[-1, -0.5, 0, 0.5, 1], outlinewidth=0),
        hovertemplate="%{y} and %{x}: %{z:.2f}<extra></extra>",
    ))

    # Label every cell (a 4 x 4 matrix is small enough). Light text on the dark cells.
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = corr.iloc[row, col]
            if pd.isna(value):
                text, colour = "n/a", theme.INK_MUTED
            else:
                text = f"{value:.2f}"
                colour = theme.SURFACE if abs(value) >= 0.8 else theme.INK  # light text on dark cells
            fig.add_annotation(x=labels[col], y=labels[row], text=text, showarrow=False,
                               font=dict(size=13, color=colour))

    theme.style_figure(fig, title=f"Correlation of monthly returns, {period}", hovermode="closest")
    fig.update_xaxes(showline=False, ticks="", automargin=True)
    fig.update_yaxes(showgrid=False, ticks="", autorange="reversed", automargin=True)
    fig.update_layout(margin=dict(t=44))
    return fig


# ---------------------------------------------------------------------------
# 5. Streamlit layout
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_returns() -> pd.DataFrame:
    """Monthly log returns from the saved prices, cached so the CSV is read once, not on every click."""
    return monthly_log_returns(load_data())


def _new_random_draw() -> None:
    """Callback for the button: use the next random seed, so the simulations are redrawn."""
    st.session_state[KEY_SEED] = st.session_state.get(KEY_SEED, DEFAULT_SEED) + 1


def _window_label(option: str, returns: pd.DataFrame) -> str:
    """How an estimation-window option is shown: 'All data' becomes e.g. 'Since Oct 2014' (short, so it fits)."""
    if ESTIMATION_WINDOWS[option] is None and not returns.empty:
        return f"Since {month_label(returns.index[0])}"
    return option


def _draw_controls(returns: pd.DataFrame) -> dict:
    """Draw the controls in three compact rows and return the chosen values."""
    # Row 1: one weight slider per asset
    raw_weights = {}
    for column, ticker in zip(st.columns(len(TICKERS)), TICKERS):
        with column:
            raw_weights[ticker] = st.slider(
                SHORT_LABELS[ticker], min_value=0, max_value=100, value=DEFAULT_WEIGHTS[ticker], step=5,
                format="%d%%", key=weight_key(ticker),
                help=f"Share of the portfolio in {ASSET_NAMES[ticker]} ({ticker}). "
                     "Weights are rescaled to add up to 100%.",
            )

    # Row 2: scenario, estimation window, horizon, starting amount. The two drop-downs get the
    # most room so "Stock market crash" and "Since Oct 2014" fit in a half-width panel.
    col_scenario, col_window, col_horizon, col_start = st.columns([1.65, 1.35, 0.8, 0.8])
    with col_scenario:
        scenario = st.selectbox(
            "Scenario", list(SCENARIOS), index=list(SCENARIOS).index(DEFAULT_SCENARIO), key=KEY_SCENARIO,
            help="Illustrative what-if assumptions applied on top of the history, not forecasts.",
        )
    with col_window:
        # format_func only changes how an option is shown ("All data" -> "Since Oct 2014");
        # the value returned is still the dictionary key, e.g. "All data"
        window = st.selectbox(
            "Estimated from", list(ESTIMATION_WINDOWS), index=list(ESTIMATION_WINDOWS).index(DEFAULT_WINDOW),
            key=KEY_WINDOW, format_func=lambda option: _window_label(option, returns),
            help="The estimation window: which past months are used to measure average returns, "
                 "volatility and correlations.",
        )
    with col_horizon:
        horizon = st.slider("Years ahead", min_value=1, max_value=MAX_HORIZON_YEARS,
                            value=DEFAULT_HORIZON_YEARS, step=1, key=KEY_HORIZON)
    with col_start:
        # Invested at the end of the last month with data (the fan chart starts there)
        start_amount = st.number_input("Invested ($)", min_value=1_000, max_value=MAX_START_AMOUNT,
                                       value=DEFAULT_START_AMOUNT, step=1_000, key=KEY_START)

    # Row 3: which chart, how many simulations, new random draw. The view options are
    # stacked (not side by side) so they fit next to the other two controls in half a screen.
    col_view, col_sims, col_button = st.columns([1.45, 1.1, 1.05], vertical_alignment="center")
    with col_view:
        view = st.radio("View", VIEWS, key=KEY_VIEW,
                        help="Fan chart: range of values over time. Outcome distribution: spread of the "
                             "final values. Correlations: how the assets moved together in the estimation window.")
    with col_sims:
        n_sims = st.select_slider("Simulations", options=SIMULATION_CHOICES, value=DEFAULT_SIMULATIONS,
                                  key=KEY_SIMULATIONS)
    with col_button:
        st.button("New random draw", key=KEY_NEW_DRAW, on_click=_new_random_draw,
                  help="Re-run the simulations with fresh random numbers.")

    return {
        "raw_weights": raw_weights,
        "scenario": scenario,
        "window": window,
        "years": int(horizon),
        "start_value": float(start_amount),
        "view": view,
        "n_sims": int(n_sims),
    }


def _draw_assumption_caption(choices: dict, weights: dict[str, float], seed: int, caution: str) -> None:
    """One caption under the controls: rescaled weights (if needed), the scenario assumptions
    and, if there is one, the caution note from high_return_caution()."""
    lines = []
    total = sum(choices["raw_weights"].values())
    if total != 100:
        rescaled = []
        for ticker, weight in weights.items():
            if weight > 0:
                rescaled.append(f"{ASSET_NAMES[ticker]} {weight * 100:.1f}%")
        lines.append(f"Weights add up to {total}%, so they were rescaled to 100%: " + ", ".join(rescaled) + ".")
    scenario = choices["scenario"]
    lines.append(f"**{scenario}:** {SCENARIOS[scenario]['summary']} "
                 f"Illustrative assumptions, not a forecast. Random draw #{seed - DEFAULT_SEED + 1}.")
    if caution:
        lines.append(caution)
    st.caption("  \n".join(lines))


def _draw_kpis(stats: dict, start_value: float, years: int) -> None:
    """Three key numbers: median final value, chance of losing money, 5% worst case."""
    col_median, col_loss, col_worst = st.columns(3)
    with col_median:
        st.metric(f"Median after {years_label(years)}", format_money(stats["median"]),
                  delta=format_change(stats["median"], start_value),
                  help="Half of the simulations end above this value and half below it.")
    with col_loss:
        st.metric("Chance of a loss", format_chance(stats["chance_of_loss"]),
                  help=f"Share of the {stats['n_sims']:,} simulations that end below the starting amount.")
    with col_worst:
        st.metric("5% worst case", format_money(stats["p5"]), delta=format_change(stats["p5"], start_value),
                  help="5th percentile: only 1 in 20 simulations ends lower than this.")


def _paths_table(paths: pd.DataFrame) -> pd.DataFrame:
    """Percentile paths with readable column names for the 'View the data' expander."""
    readable_names = {"p5": "5th percentile ($)", "p25": "25th percentile ($)", "p50": "Median ($)",
                      "p75": "75th percentile ($)", "p95": "95th percentile ($)"}
    table = paths.rename(columns=readable_names).round(0)
    table.insert(0, "Years from start", np.arange(len(table)) / 12)
    table["Years from start"] = table["Years from start"].round(2)
    table.index = table.index.strftime("%Y-%m")
    table.index.name = "Month"
    return table


def _histogram_display_table(bins: pd.DataFrame) -> pd.DataFrame:
    """Histogram bins with readable column names."""
    table = pd.DataFrame({
        "From ($)": bins["from"].round(0),
        "To ($)": bins["to"].round(0),
        "Share of simulations (%)": bins["share_pct"].round(2),
    })
    table.index.name = "Bin"
    return table


def _correlation_display_table(corr: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix with readable asset names."""
    names = {}
    for ticker in corr.columns:
        names[ticker] = SHORT_LABELS.get(ticker, ticker)
    return corr.rename(index=names, columns=names).round(2)


def _method_markdown(window_returns: pd.DataFrame, choices: dict, period: str) -> str:
    """Text for the 'Method & data sources' expander, with the current inputs in a table."""
    scenario = SCENARIOS[choices["scenario"]]
    yearly_returns = average_yearly_returns(window_returns)  # monthly mean x 12
    rows = []
    for ticker in TICKERS:
        yearly_volatility = window_returns[ticker].std() * np.sqrt(12) * 100  # monthly std x sqrt(12)
        rows.append(
            f"| [{SHORT_LABELS[ticker]}](https://finance.yahoo.com/quote/{ticker}) "
            f"| {yearly_returns[ticker]:+.1f}% | {yearly_volatility:.1f}% "
            f"| {scenario['drift_pp'][ticker]:+.1f} pp | ×{scenario['vol_mult'][ticker]:.1f} "
            f"| {scenario['shock_pct'][ticker]:+.0f}% |"
        )
    table = "\n".join(rows)
    return f"""
**Monte Carlo simulation.** Monthly log returns $r = \\ln(P_t / P_{{t-1}})$ of the four assets are
measured over the estimation window ({period}, {len(window_returns)} months with data for every asset;
month-end prices, bitcoin's weekend prices included). Their mean vector $\\mu$ and covariance matrix
$\\Sigma$ describe the "typical" month. The scenario then shifts $\\mu$ by (drift / 12) and scales $\\Sigma$
by $m_i m_j$ (volatility multipliers; correlations unchanged). Each simulation draws
{choices['years'] * 12} months of correlated returns from a multivariate normal distribution and
rebalances to the target weights every month:

$$V_t = V_{{t-1}} \\times \\sum_i w_i \\, e^{{r_{{i,t}}}}$$

One-off shocks multiply month 1 by $(1 + \\text{{shock}}_i)$. The KPIs are the median, the 5th percentile
and the share of the {choices['n_sims']:,} final values below the starting amount. When the 5% best case
is more than {LOG_SCALE_RATIO} times the 5% worst case, the fan chart and the outcome distribution use a
log scale (each ×10 step takes the same space), because compounding makes outcomes very lopsided.

| Asset | Avg. log return / yr | Volatility / yr | Scenario drift | Vol. × | Month-1 shock |
|---|---|---|---|---|---|
{table}

Past returns, especially bitcoin's, are not a forecast: the averages are measured from only a few years
and move a lot with the window chosen. Scenarios are illustrative assumptions. Values are in nominal
dollars (not adjusted for inflation), before fees and taxes.

**Sources:** Yahoo Finance daily adjusted close prices (dividends included) via `yfinance`:
[SPY](https://finance.yahoo.com/quote/SPY), [AGG](https://finance.yahoo.com/quote/AGG),
[GLD](https://finance.yahoo.com/quote/GLD), [BTC-USD](https://finance.yahoo.com/quote/BTC-USD).
"""


def render() -> None:
    """Draw the whole panel inside the current Streamlit container."""
    ui.panel_header(NUMBER, TITLE, OWNER, QUESTION)

    try:
        returns = _cached_returns()
    except Exception as exc:  # no snapshot and no internet, or a broken CSV
        ui.no_data_message("Asset price", exc)
        return

    if KEY_SEED not in st.session_state:
        st.session_state[KEY_SEED] = DEFAULT_SEED
    choices = _draw_controls(returns)
    seed = st.session_state[KEY_SEED]

    weights = normalise_weights(choices["raw_weights"])
    if weights is None:
        st.info("All the weights are 0%. Move at least one slider above 0% to build a portfolio.")
        return

    window_returns = estimation_window(returns, ESTIMATION_WINDOWS[choices["window"]])
    if len(window_returns) < MIN_MONTHS:
        st.info(f"Only {len(window_returns)} months have prices for all four assets in this window "
                f"(at least {MIN_MONTHS} are needed). Try a longer estimation window or refresh the data.")
        return
    period = f"{month_label(window_returns.index[0])} – {month_label(window_returns.index[-1])}"

    caution = high_return_caution(average_yearly_returns(window_returns), weights, period)
    _draw_assumption_caption(choices, weights, seed, caution)

    # Model: estimate -> apply scenario -> simulate -> summarise
    scenario = SCENARIOS[choices["scenario"]]
    mean_monthly, cov_monthly = estimate_inputs(window_returns)
    mean_monthly, cov_monthly = apply_scenario(mean_monthly, cov_monthly, scenario, TICKERS)
    if not (np.isfinite(mean_monthly).all() and np.isfinite(cov_monthly).all()):
        st.info("The saved prices contain gaps, so the model inputs can't be estimated. Try refreshing the data.")
        return
    weight_array = np.array([weights[ticker] for ticker in TICKERS])
    months = choices["years"] * 12
    values = simulate_portfolio(mean_monthly, cov_monthly, weight_array, months, choices["start_value"],
                                choices["n_sims"], seed, shocks=scenario_shocks(scenario, TICKERS))
    stats = outcome_stats(values, choices["start_value"], choices["years"])
    corr = window_returns.corr()

    _draw_kpis(stats, choices["start_value"], choices["years"])

    view = choices["view"]
    log_scale = use_log_scale(stats)  # very spread-out outcomes (e.g. lots of bitcoin) need a log scale
    if view == VIEW_FAN:
        paths = percentile_paths(values)
        # Month 0 is the last month with data; each column after it is one month later
        paths.index = pd.date_range(window_returns.index[-1], periods=months + 1, freq="ME")
        fig = build_fan_chart(paths, choices["start_value"], log_scale)
        table = _paths_table(paths)
        file_name = f"{SLUG}_fan_chart.csv"
    elif view == VIEW_OUTCOMES:
        bins = histogram_table(values[:, -1], log_scale=log_scale)
        fig = build_histogram(bins, stats, choices["start_value"], choices["years"], log_scale)
        table = _histogram_display_table(bins)
        file_name = f"{SLUG}_outcomes.csv"
    else:
        fig = build_correlation_heatmap(corr, period)
        table = _correlation_display_table(corr)
        file_name = f"{SLUG}_correlations.csv"
    ui.show_chart(fig, key=f"{SLUG}_chart")

    insight_text = write_insight(view, stats, weights, choices["scenario"], choices["years"],
                                 choices["start_value"], corr, period)
    ui.insight(escape_dollars(insight_text))
    ui.details(_method_markdown(window_returns, choices, period), table, key=SLUG, file_name=file_name)
