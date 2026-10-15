"""Panel 3 - Sector Winners & Losers.

Question: which parts of the stock market won and lost?

The S&P 500 index is split into 11 sectors (technology, energy, banks and so
on). Each sector can be bought as an exchange-traded fund (ETF): XLK, for
example, holds the technology companies of the S&P 500. Comparing these 11
sector ETFs with SPY, an ETF that holds the whole S&P 500, shows which parts
of the market led and which lagged in any period - for example in 2022, when
the Fed raised interest rates quickly to fight inflation.

Two views:
* Ranking        - a sorted bar chart of one measure (return, risk, ...) per sector
* Growth of $100 - how $100 invested at the start grew in each sector

The module follows the same layout as every panel:
1. constants
2. data loading
3. analysis functions (pandas/NumPy only, no Streamlit, unit tested)
4. chart builders (return Plotly figures, no Streamlit)
5. render() - the Streamlit layout: controls, key numbers, chart, insight
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme, ui
from dashboard.data import filter_dates, load_prices, load_recessions, recession_periods, recessions_in_window

NUMBER = 3
SLUG = "sectors"
TITLE = "Sector Winners & Losers"
OWNER = "Team member 3"  # panel owner: put your name here
QUESTION = "Which parts of the stock market won and lost?"

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# Data: daily prices from Yahoo Finance (https://finance.yahoo.com/quote/<TICKER>)
SNAPSHOT_NAME = "sectors_prices"  # saved as data/sectors_prices.csv
DATA_START = "2007-01-01"

BENCHMARK = "SPY"           # SPDR S&P 500 ETF: the whole market, our yardstick
BENCHMARK_NAME = "S&P 500"

# The 11 SPDR sector ETFs: Yahoo Finance ticker -> sector name
SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",              # only trades from October 2015
    "XLC": "Communication Services",    # only trades from June 2018
}
TICKERS = [BENCHMARK] + list(SECTORS)
# Shorter names for tight spaces (the key-number titles); the others are short already
SHORT_NAMES = {"XLY": "Discretionary", "XLP": "Staples", "XLC": "Comm. Services"}

TRADING_DAYS_PER_YEAR = 252  # used to turn daily numbers into yearly ones
MIN_TRADING_DAYS = 5         # fewer daily returns than this is too little to measure anything
MAX_DAILY_POINTS = 600       # longer line charts are thinned to one point per week
# A volatility or variance below this counts as zero: rounding errors in
# floating-point maths leave tiny numbers like 1e-17 where the exact answer is 0.
NEAR_ZERO = 1e-10

# --- Periods ---------------------------------------------------------------
YEAR_TO_DATE = "Year to date"
CUSTOM = "Custom"
# Look-back periods: label -> number of years before the latest price
LOOKBACK_YEARS = {"Last 12 months": 1, "Last 3 years": 3, "Last 5 years": 5}
# Historical episodes: label -> (start, end). The start is the day of the
# starting price: the last close of 2021, or the S&P 500's peak close before a crash.
EPISODES = {
    "2022 rate-hike sell-off": ("2021-12-31", "2022-12-31"),
    "COVID crash (19 Feb – 23 Mar 2020)": ("2020-02-19", "2020-03-23"),
    "Global financial crisis (Oct 2007 – Mar 2009)": ("2007-10-09", "2009-03-09"),
}
PERIODS = [YEAR_TO_DATE] + list(LOOKBACK_YEARS) + list(EPISODES) + [CUSTOM]
DEFAULT_PERIOD = "2022 rate-hike sell-off"
CUSTOM_DEFAULT_START = datetime.date(2020, 1, 1)

# How each period is named inside a sentence ("Over the last 12 months, ...")
PERIOD_PHRASES = {
    YEAR_TO_DATE: "the year to date",
    "Last 12 months": "the last 12 months",
    "Last 3 years": "the last 3 years",
    "Last 5 years": "the last 5 years",
    "2022 rate-hike sell-off": "the 2022 rate-hike sell-off",
    "COVID crash (19 Feb – 23 Mar 2020)": "the COVID crash",
    "Global financial crisis (Oct 2007 – Mar 2009)": "the global financial crisis",
}

# --- Measures ----------------------------------------------------------------
TOTAL_RETURN = "Total return"
VOLATILITY = "Annualised volatility"
SHARPE = "Sharpe ratio"
DRAWDOWN = "Maximum drawdown"
BETA = "Beta vs S&P 500"

# Everything the panel needs to know about each measure, in one place:
#   percent          - stored as a fraction (0.12) and shown as a percentage (12.0%)
#   signed           - always write the + or - sign in front of the number (+12.3%)
#   bar_colour       - "sign" (blue if >= 0, red if < 0), "blue" or "red"
#   higher_is_better - sort order of the ranking (best at the top)
#   best_label / worst_label - KPI titles for the top and bottom sector
#   beat_phrase      - what "beating the S&P 500" means for this measure
#   story            - first half of the insight sentence
#   legend           - bar legend in "Relative to S&P 500" mode (beat, trailed)
#   delta_color      - Streamlit colour for KPI changes ("normal" = up is green)
METRICS = {
    TOTAL_RETURN: {
        "short": "total return",
        "percent": True,
        "signed": True,  # a gain and a loss look different: +12.3% and -12.3%
        "bar_colour": "sign",
        "higher_is_better": True,
        "best_label": "Best",
        "worst_label": "Worst",
        "beat_phrase": "beat the S&P 500",
        "story": "{best} led with {best_value} and {worst} lagged with {worst_value}",
        "legend": ("Beat the S&P 500", "Trailed the S&P 500"),
        "delta_color": "normal",
        "help": "Price change over the period with dividends reinvested: last price / first price − 1.",
    },
    VOLATILITY: {
        "short": "volatility",
        "percent": True,
        "signed": False,
        "bar_colour": "blue",
        "higher_is_better": False,  # less risk ranks higher
        "best_label": "Calmest",
        "worst_label": "Riskiest",
        "beat_phrase": "had lower volatility than the S&P 500",
        "story": "{best} was the calmest sector ({best_value} a year) and {worst} "
                 "the most volatile ({worst_value})",
        "legend": ("Less volatile than the S&P 500", "More volatile than the S&P 500"),
        "delta_color": "inverse",
        "help": "How much daily returns swing, scaled to a year: std(daily returns) × √252.",
    },
    SHARPE: {
        "short": "Sharpe ratio",
        "percent": False,
        "signed": False,
        "bar_colour": "sign",
        "higher_is_better": True,
        "best_label": "Best",
        "worst_label": "Worst",
        "beat_phrase": "had a higher Sharpe ratio than the S&P 500",
        "story": "{best} had the best return for its risk (Sharpe {best_value}) and "
                 "{worst} the worst ({worst_value})",
        "legend": ("Higher Sharpe than the S&P 500", "Lower Sharpe than the S&P 500"),
        "delta_color": "normal",
        "help": "Return above the risk-free rate per unit of volatility (both per year).",
    },
    DRAWDOWN: {
        "short": "max drawdown",
        "percent": True,
        "signed": False,  # always 0 or negative, so the - sign shows anyway
        "bar_colour": "red",
        "higher_is_better": True,  # -10% is a smaller fall than -30%
        "best_label": "Best",
        "worst_label": "Worst",
        "beat_phrase": "fell less than the S&P 500 at their worst",
        "story": "{worst} had the deepest fall from a peak ({worst_value}) and {best} "
                 "the smallest ({best_value})",
        "legend": ("Smaller fall than the S&P 500", "Deeper fall than the S&P 500"),
        "delta_color": "normal",
        "help": "Largest fall from a previous high inside the period: min(price / running max − 1).",
    },
    BETA: {
        "short": "beta",
        "percent": False,
        "signed": False,
        "bar_colour": "blue",
        "higher_is_better": False,  # less market risk ranks higher
        "best_label": "Lowest",
        "worst_label": "Highest",
        # Beta measures how strongly a sector follows the S&P 500, NOT how much it
        # swings: in 2022 Energy had a beta of 0.64 but was the most volatile sector.
        "beat_phrase": "had a beta below 1, so were less sensitive to S&P 500 moves",
        "story": "{worst} was the most sensitive to the market (beta {worst_value}) and "
                 "{best} the least (beta {best_value})",
        "legend": ("Beta below 1 (less market-sensitive)", "Beta above 1 (more market-sensitive)"),
        "delta_color": "off",
        "help": "How much a sector tended to move when the S&P 500 moved 1%: "
                "cov(sector, S&P 500) / var(S&P 500). It measures sensitivity to the market, "
                "not the size of the sector's own swings (that is volatility).",
    },
}
DEFAULT_RISK_FREE = 2.0  # % per year, starting value of the risk-free rate slider

# --- Views --------------------------------------------------------------------
RANKING = "Ranking"
GROWTH = "Growth of $100"
VIEWS = [RANKING, GROWTH]
DEFAULT_HIGHLIGHT = "XLK"

# Widget keys (all start with the slug so they are unique across the dashboard)
KEY_PERIOD = f"{SLUG}_period"
KEY_DATES = f"{SLUG}_dates"
KEY_METRIC = f"{SLUG}_metric"
KEY_VIEW = f"{SLUG}_view"
KEY_RISK_FREE = f"{SLUG}_risk_free"
KEY_RELATIVE = f"{SLUG}_relative"
KEY_HIGHLIGHT = f"{SLUG}_highlight"


# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------

def load_data(refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted close prices of SPY and the 11 sector ETFs, one column per ticker.

    "Adjusted" prices include dividends, so the % change of the price is the
    total return an investor actually got. ETFs that didn't exist yet have
    empty (NaN) prices: XLRE before October 2015 and XLC before June 2018.
    """
    return load_prices(SNAPSHOT_NAME, TICKERS, start=DATA_START, refresh=refresh)


def refresh_data() -> None:
    """Download fresh data for this panel (used by the refresh button and scripts/update_data.py)."""
    load_data(refresh=True)


# ---------------------------------------------------------------------------
# 3. Analysis (no Streamlit here, so it can be unit tested)
# ---------------------------------------------------------------------------

def sector_name(ticker: str) -> str:
    """'Technology' for 'XLK', 'S&P 500' for 'SPY'."""
    if ticker == BENCHMARK:
        return BENCHMARK_NAME
    return SECTORS.get(ticker, ticker)


def short_name(ticker: str) -> str:
    """Short sector name for tight spaces, e.g. 'Staples' for 'Consumer Staples'."""
    return SHORT_NAMES.get(ticker, sector_name(ticker))


def ticker_label(ticker: str) -> str:
    """Name and ticker together, e.g. 'Technology (XLK)'."""
    return f"{sector_name(ticker)} ({ticker})"


def day_label(day) -> str:
    """'9 Oct 2007' style label (written this way because '%-d' doesn't work on Windows)."""
    day = pd.Timestamp(day)
    return f"{day.day} {day.strftime('%b %Y')}"


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Prices ready for the panel: no half-finished last day and no gaps.

    * Days without an S&P 500 price are dropped. A download made during US
      trading hours can end with a day on which only some ETFs have a price.
    * A single missing price inside a ticker's history (a gap in the download)
      is filled with the previous close (``ffill``), i.e. "no change that day",
      so one bad row can't knock a sector out of a whole period.
    ``ffill`` only fills forward from a ticker's first real price, so the empty
    prices *before* an ETF started trading (XLC before June 2018) stay NaN.
    """
    has_benchmark_price = prices[BENCHMARK].notna()
    return prices[has_benchmark_price].ffill()


def period_dates(period: str, last_date, custom_start=None,
                 custom_end=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last date of the chosen period.

    ``last_date`` is the date of the newest price in the data. The first date
    is the day of the *starting* price: Year to date starts on 31 December of
    the previous year, so January's first trading day counts as part of the year.
    For Custom, the two dates are swapped if they arrive the wrong way round.
    """
    last_date = pd.Timestamp(last_date)
    if period == YEAR_TO_DATE:
        return pd.Timestamp(last_date.year - 1, 12, 31), last_date
    if period in LOOKBACK_YEARS:
        start = last_date - pd.DateOffset(years=LOOKBACK_YEARS[period])
        return start, last_date
    if period in EPISODES:
        start, end = EPISODES[period]
        return pd.Timestamp(start), pd.Timestamp(end)
    if period == CUSTOM:
        if custom_start is None or custom_end is None:
            raise ValueError("Custom period needs a start and an end date")
        first, second = pd.Timestamp(custom_start), pd.Timestamp(custom_end)
        return min(first, second), max(first, second)
    raise ValueError(f"Unknown period: {period}")


def window_prices(prices: pd.DataFrame, start, end) -> pd.DataFrame:
    """Prices from ``start`` to ``end``, beginning with the last close on or before ``start``.

    Returns are measured from one closing price to another. If ``start`` is a
    weekend or holiday (31 Dec 2022 was a Saturday), the window starts at the
    trading day before it. If ``start`` is before the data begins, the window
    starts at the first price we have.
    """
    trading_days_up_to_start = prices.index[prices.index <= pd.Timestamp(start)]
    if len(trading_days_up_to_start) > 0:
        start = trading_days_up_to_start[-1]
    return filter_dates(prices, start, end)


def split_by_data_coverage(window: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split the tickers into (complete, excluded).

    A ticker is excluded when it has no price on the window's first day. That
    happens when an ETF started trading inside the window (XLC in June 2018):
    without a starting price its return can't be compared fairly with the others.
    The window should come from clean_prices, so a ticker with a first price
    has a price on every later day too (gaps were filled).
    """
    first_day = window.iloc[0]
    complete = []
    excluded = []
    for ticker in window.columns:
        if pd.notna(first_day[ticker]):
            complete.append(ticker)
        else:
            excluded.append(ticker)
    return complete, excluded


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Day-to-day % change of each price, as fractions (0.01 = +1%). The first row is dropped."""
    return prices.pct_change(fill_method=None).iloc[1:]


def total_return(prices: pd.DataFrame) -> pd.Series:
    """Total return over the window: last price / first price - 1."""
    return prices.iloc[-1] / prices.iloc[0] - 1


def annualised_volatility(returns: pd.DataFrame) -> pd.Series:
    """Annualised volatility: sample standard deviation of daily returns x sqrt(252).

    Daily swings are scaled up to a year with the square root of the number
    of trading days, because variance (not standard deviation) adds up over time.
    """
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: pd.DataFrame, risk_free_annual: float) -> pd.Series:
    """Sharpe ratio: (mean daily excess return x 252) / (std of daily returns x sqrt(252)).

    The excess return is the return above a safe investment (cash or Treasury
    bills). ``risk_free_annual`` is a fraction per year (0.02 = 2%), split
    evenly over the trading days: daily rate = risk_free_annual / 252.
    A ticker whose price never wiggled has (practically) zero volatility; its
    ratio is undefined and left as NaN instead of dividing by zero.
    """
    daily_risk_free = risk_free_annual / TRADING_DAYS_PER_YEAR
    annual_excess_return = (returns - daily_risk_free).mean() * TRADING_DAYS_PER_YEAR
    volatility = annualised_volatility(returns)
    usable_volatility = volatility.where(volatility > NEAR_ZERO)  # NaN where it is ~0
    return annual_excess_return / usable_volatility


def max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """Maximum drawdown: the worst fall from a previous high, min(price / running max - 1).

    ``cummax`` gives the highest price so far on each day. 0 means the price
    never fell below an earlier high; -0.25 means it was at one point 25% below one.
    """
    running_max = prices.cummax()
    drawdowns = prices / running_max - 1
    return drawdowns.min()


def beta(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    """Beta of every column against the benchmark: cov(column, benchmark) / var(benchmark).

    Beta 1.2 means the sector tended to move 1.2% when the S&P 500 moved 1%.
    The benchmark itself has beta 1 by definition.
    """
    benchmark_variance = benchmark_returns.var()
    betas = {}
    for ticker in returns.columns:
        if benchmark_variance > NEAR_ZERO:
            betas[ticker] = returns[ticker].cov(benchmark_returns) / benchmark_variance
        else:  # a flat benchmark: beta can't be measured
            betas[ticker] = np.nan
    return pd.Series(betas)


def compute_metrics(prices: pd.DataFrame, risk_free_annual: float = 0.0) -> pd.DataFrame:
    """All five measures for every ticker in ``prices`` (rows = tickers, columns = measures).

    ``prices`` should only contain tickers with a price on every day of the
    window (see split_by_data_coverage), including the benchmark for beta.
    """
    returns = daily_returns(prices)
    table = pd.DataFrame({
        TOTAL_RETURN: total_return(prices),
        VOLATILITY: annualised_volatility(returns),
        SHARPE: sharpe_ratio(returns, risk_free_annual),
        DRAWDOWN: max_drawdown(prices),
    })
    if BENCHMARK in returns.columns:
        table[BETA] = beta(returns, returns[BENCHMARK])
    else:
        table[BETA] = np.nan
    return table


def rank_sectors(values: pd.Series, metric: str) -> pd.Series:
    """The sectors' values sorted best first (the benchmark and missing values left out)."""
    sectors_only = values.drop(BENCHMARK, errors="ignore").dropna()
    best_first = METRICS[metric]["higher_is_better"]
    return sectors_only.sort_values(ascending=not best_first)


def relative_to_benchmark(values: pd.Series, benchmark_value: float) -> pd.Series:
    """Each value minus the S&P 500's value (e.g. sector return - S&P 500 return)."""
    return values - benchmark_value


def beats_benchmark(values: pd.Series, benchmark_value: float, metric: str) -> pd.Series:
    """True for each sector that did better than the S&P 500 on this measure.

    "Better" means higher for return, Sharpe and drawdown, and lower for
    volatility and beta (less risk).
    """
    if METRICS[metric]["higher_is_better"]:
        return values > benchmark_value
    return values < benchmark_value


def format_value(value: float, metric: str, relative: bool = False) -> str:
    """A measure as text: '+12.3%', '18.0%', '0.85', or '+4.1 pp' in relative mode.

    pp = percentage points, the right unit for the difference between two percentages.
    """
    if pd.isna(value):
        return "n/a"
    spec = METRICS[metric]
    if relative:  # a difference from the S&P 500 always shows its sign
        if spec["percent"]:
            return f"{value * 100:+.1f} pp"
        return f"{value:+.2f}"
    if spec["percent"]:
        if spec["signed"]:
            return f"{value * 100:+.1f}%"
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def count_phrase(count: int, total: int) -> str:
    """'none of the 11 sectors', '3 of the 11 sectors' or 'all 11 sectors'."""
    if count == 0:
        return f"none of the {total} sectors"
    if count == total:
        return f"all {total} sectors"
    return f"{count} of the {total} sectors"


def all_sharpe_ratios_negative(metric: str, ranked: pd.Series) -> bool:
    """True when the measure is the Sharpe ratio and every sector's ratio is below zero.

    Below zero the Sharpe ratio stops being a fair ranking: a loss divided by
    a BIGGER volatility gives a number closer to zero, so the riskiest sector
    looks "best". (COVID crash: Technology lost more than Health Care and was
    riskier, yet had the higher Sharpe ratio.) The panel then words things differently.
    """
    return metric == SHARPE and ranked.max() < 0


def kpi_labels(metric: str, ranked: pd.Series) -> tuple[str, str]:
    """Titles of the top and bottom sector's key numbers, e.g. ('Best', 'Worst')."""
    if all_sharpe_ratios_negative(metric, ranked):
        return "Highest", "Lowest"  # not "best": see all_sharpe_ratios_negative
    return METRICS[metric]["best_label"], METRICS[metric]["worst_label"]


def leader_story(metric: str, ranked: pd.Series) -> str:
    """First half of the ranking insight: who was best and who was worst."""
    best, worst = ranked.index[0], ranked.index[-1]
    best_value = format_value(ranked[best], metric)
    worst_value = format_value(ranked[worst], metric)
    if len(ranked) == 1:
        return f"{sector_name(best)} is the only sector with data ({best_value})"
    # "Led with -24.2%" sounds wrong, so returns get their own wording when all fell or all rose
    if metric == TOTAL_RETURN and ranked[best] < 0:
        return (f"every sector lost money: {sector_name(best)} fell least ({best_value}) "
                f"and {sector_name(worst)} most ({worst_value})")
    if metric == TOTAL_RETURN and ranked[worst] > 0:
        return (f"every sector made money: {sector_name(best)} led with {best_value} "
                f"and {sector_name(worst)} gained least ({worst_value})")
    if all_sharpe_ratios_negative(metric, ranked):
        return (f"every sector earned less than the risk-free rate, so all Sharpe ratios were "
                f"negative: {sector_name(best)} was least negative ({best_value}) and "
                f"{sector_name(worst)} most negative ({worst_value})")
    # A negative beta means the sector tended to go UP on days the S&P 500 went down
    if metric == BETA and ranked[best] < 0:
        return (f"{sector_name(worst)} was the most sensitive to the market (beta {worst_value}) "
                f"and {sector_name(best)} even tended to move against it (beta {best_value})")
    return METRICS[metric]["story"].format(best=sector_name(best), best_value=best_value,
                                           worst=sector_name(worst), worst_value=worst_value)


def ranking_insight(metric: str, ranked: pd.Series, benchmark_value: float, period_phrase: str) -> str:
    """Insight for the Ranking view: leader, laggard and how many sectors beat the S&P 500.

    ``ranked`` is the output of rank_sectors (best first, absolute values).
    """
    story = f"Over {period_phrase}, {leader_story(metric, ranked)}"
    if all_sharpe_ratios_negative(metric, ranked):
        # Counting who "beat" the S&P 500 would repeat the unfair ranking, so warn instead
        return (story + ". Below zero, a riskier sector gets a ratio closer to zero, so this "
                "ranking isn't a fair test: compare total returns instead.")
    winners = int(beats_benchmark(ranked, benchmark_value, metric).sum())
    text = f"{story}; {count_phrase(winners, len(ranked))} {METRICS[metric]['beat_phrase']}"
    if metric == BETA:
        return text + "."  # the S&P 500's beta is always 1, no need to repeat it
    return text + f" ({format_value(benchmark_value, metric)})."


def growth_of_100(prices: pd.DataFrame) -> pd.DataFrame:
    """What $100 invested on the first day of the window is worth on every later day."""
    return 100 * prices / prices.iloc[0]


def thin_to_weekly(growth: pd.DataFrame) -> pd.DataFrame:
    """Keep the first day plus the last trading day of every week.

    A chart about 550 pixels wide can't show thousands of daily points, so
    long periods are drawn from weekly points (faster, and looks the same).
    The first day stays so every line still starts at exactly $100.
    """
    weeks = growth.index.to_period("W")
    last_day_of_each_week = growth.groupby(weeks).tail(1)
    keep = growth.index.isin(last_day_of_each_week.index)
    keep[0] = True
    return growth[keep]


def growth_insight(growth: pd.DataFrame, highlight: str, highlight_drawdown: float) -> str:
    """Insight for the Growth view: where $100 in the highlighted sector ended vs the S&P 500."""
    # Round to whole cents BEFORE subtracting, so the gap in the sentence is
    # exactly the difference of the two amounts the reader sees next to it
    sector_end = round(float(growth[highlight].iloc[-1]), 2)
    benchmark_end = round(float(growth[BENCHMARK].iloc[-1]), 2)
    difference = sector_end - benchmark_end  # dollars per $100, i.e. percentage points

    text = (f"$100 put into {sector_name(highlight)} at the close on {day_label(growth.index[0])} "
            f"was worth ${sector_end:,.2f} on {day_label(growth.index[-1])}, against "
            f"${benchmark_end:,.2f} in the S&P 500")
    if abs(difference) < 0.5:
        text += ", roughly the same."
    elif difference > 0:
        text += f", so it beat the index by ${difference:,.2f}."
    else:
        text += f", so it trailed the index by ${-difference:,.2f}."

    if highlight_drawdown < 0:
        text += f" Along the way it was at worst {abs(highlight_drawdown):.1%} below an earlier high."
    else:
        text += " It never fell below an earlier high along the way."
    return text


def markdown_safe(text: str) -> str:
    """Escape $ signs so Streamlit doesn't read '$100 ... $245' as a maths formula."""
    return text.replace("$", "\\$")


# ---------------------------------------------------------------------------
# 4. Charts (return Plotly figures, no Streamlit here)
# ---------------------------------------------------------------------------

def to_chart_units(values, metric: str):
    """Percent measures are stored as fractions (0.12) but plotted as percentages (12)."""
    if METRICS[metric]["percent"]:
        return values * 100
    return values


def bar_colours(values: pd.Series, benchmark_value: float, metric: str, relative: bool) -> list[str]:
    """One colour per bar.

    Relative mode: blue if the sector did better than the S&P 500, red if worse.
    Otherwise it depends on the measure: returns and Sharpe are blue above
    zero and red below, drawdowns are always red (they are losses), and
    volatility and beta are plain blue (they are sizes, not good or bad).
    """
    style = METRICS[metric]["bar_colour"]
    better = beats_benchmark(values, benchmark_value, metric)
    colours = []
    for ticker, value in values.items():
        if relative:
            colours.append(theme.POSITIVE if better[ticker] else theme.NEGATIVE)
        elif style == "sign":
            colours.append(theme.POSITIVE if value >= 0 else theme.NEGATIVE)
        elif style == "red":
            colours.append(theme.NEGATIVE)
        else:
            colours.append(theme.BLUE)
    return colours


def value_axis_range(plotted: list[float]) -> list[float]:
    """X-axis range that fits every bar, the zero line, the S&P 500 line and the value labels."""
    low = min(min(plotted), 0.0)
    high = max(max(plotted), 0.0)
    span = high - low
    if span == 0:  # every bar is exactly 0: show a small range around zero
        return [-1.0, 1.0]
    room_for_labels = 0.2 * span  # the value labels sit just past the end of each bar
    if low < 0:
        low -= room_for_labels
    if high > 0:
        high += room_for_labels
    else:
        # Every bar is negative (e.g. drawdowns): an axis ending exactly at 0 would
        # put the '0%' tick label on the chart's right edge, where Plotly hides it
        high = 0.06 * span
    return [low, high]


def add_value_labels(fig: go.Figure, names: list[str], values: list[float], labels: list[str]) -> None:
    """Write each bar's value just past the end of the bar.

    The labels are annotations (not bar text) so they can have a background
    in the plot's own colour: where the dashed S&P 500 line runs behind a
    label, the label stays readable instead of being crossed out.
    """
    for name, value, label in zip(names, values, labels):
        if value >= 0:
            anchor, shift = "left", 4    # positive bar: label to the right of its end
        else:
            anchor, shift = "right", -4  # negative bar: label to the left of its end
        fig.add_annotation(x=value, y=name, text=label, showarrow=False, xanchor=anchor, xshift=shift,
                           bgcolor=theme.PLOT_BACKGROUND, borderpad=1,
                           font=dict(size=11, color=theme.INK_SECONDARY))


def add_benchmark_line(fig: go.Figure, x: float, text: str, x_range: list[float]) -> None:
    """Dashed vertical line for the S&P 500, labelled above the plot.

    layer="below" draws it behind the bars, so a bar is never hidden by it.
    Near the left or right edge the label is anchored so it isn't cut off.
    """
    fig.add_vline(x=x, line_width=1.5, line_dash="dash", line_color=theme.INK_MUTED, layer="below")
    share_of_axis = (x - x_range[0]) / (x_range[1] - x_range[0])  # 0 = left edge, 1 = right edge
    if share_of_axis < 0.15:
        anchor = "left"
    elif share_of_axis > 0.85:
        anchor = "right"
    else:
        anchor = "center"
    fig.add_annotation(x=x, y=1, yref="paper", xanchor=anchor, yanchor="bottom", yshift=2,
                       showarrow=False, text=text, font=dict(size=11, color=theme.INK_SECONDARY))


def build_ranking_chart(ranked: pd.Series, benchmark_value: float, metric: str,
                        relative: bool = False) -> go.Figure:
    """Horizontal bars, best sector at the top, with the S&P 500 as a vertical reference line.

    ``ranked`` holds absolute values sorted best first (from rank_sectors).
    In relative mode the bars show each sector minus the S&P 500, so the
    S&P 500 line sits at zero.
    """
    if relative:
        shown = relative_to_benchmark(ranked, benchmark_value)
        reference = 0.0
        reference_text = "S&P 500"
        axis_title = f"Difference from the S&P 500's {METRICS[metric]['short']}"
    else:
        shown = ranked
        reference = benchmark_value
        reference_text = f"S&P 500: {format_value(benchmark_value, metric)}"
        axis_title = metric
    if METRICS[metric]["percent"]:
        axis_title += " (percentage points)" if relative else " (%)"

    names = [sector_name(ticker) for ticker in shown.index]
    bar_lengths = list(to_chart_units(shown, metric))
    labels = [format_value(value, metric, relative) for value in shown]
    hover_info = []
    for ticker in shown.index:
        difference = ranked[ticker] - benchmark_value
        hover_info.append([ticker_label(ticker), format_value(ranked[ticker], metric),
                           format_value(difference, metric, relative=True)])

    fig = go.Figure(go.Bar(
        x=bar_lengths, y=names, orientation="h",
        marker=dict(color=bar_colours(ranked, benchmark_value, metric, relative), line_width=0),
        customdata=hover_info, showlegend=False,
        hovertemplate=(f"<b>%{{customdata[0]}}</b><br>{metric}: %{{customdata[1]}}"
                       "<br>vs S&P 500: %{customdata[2]}<extra></extra>"),
    ))
    add_value_labels(fig, names, bar_lengths, labels)

    # Legend for the colours used in relative mode (empty traces that only draw a legend key)
    if relative:
        beat_text, trail_text = METRICS[metric]["legend"]
        winners = int(beats_benchmark(ranked, benchmark_value, metric).sum())
        if winners > 0:
            fig.add_trace(go.Bar(x=[None], y=[None], name=beat_text, marker_color=theme.POSITIVE,
                                 orientation="h", hoverinfo="skip"))
        if winners < len(ranked):
            fig.add_trace(go.Bar(x=[None], y=[None], name=trail_text, marker_color=theme.NEGATIVE,
                                 orientation="h", hoverinfo="skip"))
    else:
        # Zero line (in relative mode the S&P 500 line is already at zero)
        fig.add_vline(x=0, line_width=1, line_color=theme.AXIS, layer="below")

    reference_x = float(to_chart_units(reference, metric))
    x_range = value_axis_range(bar_lengths + [reference_x])
    add_benchmark_line(fig, reference_x, reference_text, x_range)

    theme.style_figure(fig, x_title=axis_title, hovermode="closest")
    fig.update_xaxes(range=x_range, showgrid=True,
                     gridcolor=theme.GRID, zeroline=False, automargin=True,
                     ticksuffix="%" if METRICS[metric]["percent"] and not relative else "")
    # Categories top to bottom in the order given (best first); no gridlines between bars
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticks="")
    # Extra top margin keeps the S&P 500 label clear of Plotly's toolbar
    fig.update_layout(bargap=0.35, barmode="overlay", barcornerradius=3, margin=dict(t=48),
                      legend=dict(orientation="h", y=-0.2, yanchor="top"))
    return fig


def add_end_labels(fig: go.Figure, growth: pd.DataFrame, tickers: list[str]) -> None:
    """Write the final dollar value just right of the end of each line in ``tickers``.

    If two labels would overlap (end values close together) they are nudged apart.
    """
    end_values = {}
    shifts = {}  # how many pixels to move each label up (+) or down (-)
    for ticker in tickers:
        end_values[ticker] = growth[ticker].iloc[-1]
        shifts[ticker] = 0

    # Two labels closer than 7% of the chart's height would overlap: push them apart
    if len(tickers) == 2:
        first, second = tickers
        chart_height_in_dollars = growth.max().max() - growth.min().min()
        if abs(end_values[first] - end_values[second]) < 0.07 * chart_height_in_dollars:
            if end_values[first] >= end_values[second]:
                shifts[first], shifts[second] = 8, -8
            else:
                shifts[first], shifts[second] = -8, 8

    for ticker in tickers:
        fig.add_annotation(
            x=growth.index[-1], y=end_values[ticker], text=f"{ticker} ${end_values[ticker]:,.0f}",
            showarrow=False, xanchor="left", xshift=6, yshift=shifts[ticker],
            font=dict(size=11, color=theme.INK_SECONDARY),
        )


def build_growth_chart(growth: pd.DataFrame, highlight: str, recessions: list | None = None) -> go.Figure:
    """Growth of $100: every sector as a thin grey line, one highlighted in blue, the S&P 500 in dark grey.

    Highlighting one sector instead of giving all 11 their own colour keeps the
    chart readable (more than ~8 colours can't be told apart).
    """
    fig = go.Figure()
    # 1) The other sectors: context only, so light grey, no legend, no hover
    for ticker in growth.columns:
        if ticker in (BENCHMARK, highlight):
            continue
        fig.add_trace(go.Scatter(
            x=growth.index, y=growth[ticker], name=ticker_label(ticker), mode="lines",
            line=dict(color=theme.AXIS, width=1), hoverinfo="skip", showlegend=False,
        ))
    # 2) The S&P 500 benchmark in dark grey
    fig.add_trace(go.Scatter(
        x=growth.index, y=growth[BENCHMARK], name=ticker_label(BENCHMARK), mode="lines",
        line=dict(color=theme.INK_SECONDARY, width=2), legendrank=2,
        hovertemplate="S&P 500: $%{y:,.2f}<extra></extra>",
    ))
    # 3) The highlighted sector last, so it is drawn on top
    end_labels = [BENCHMARK]
    if highlight in growth.columns:
        fig.add_trace(go.Scatter(
            x=growth.index, y=growth[highlight], name=ticker_label(highlight), mode="lines",
            line=dict(color=theme.BLUE, width=2.5), legendrank=1,  # first in the legend
            hovertemplate=f"{sector_name(highlight)}: $%{{y:,.2f}}<extra></extra>",
        ))
        end_labels.append(highlight)

    start, end = growth.index[0], growth.index[-1]
    if recessions is None:
        recessions = []
    visible_recessions = recessions_in_window(recessions, start, end)
    if visible_recessions:  # legend entry only when a grey band is actually drawn
        theme.add_recession_bands(fig, visible_recessions, start, end)
        theme.add_recession_legend_entry(fig)

    fig.add_hline(y=100, line_width=1, line_color=theme.INK_MUTED)  # the starting $100
    add_end_labels(fig, growth, end_labels)

    theme.style_figure(fig, y_title="Value of $100 invested ($)")
    fig.update_xaxes(range=[start, end], hoverformat="%d %b %Y", automargin=True)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", automargin=True)
    # Legend under the chart: at the top it would sit under Plotly's toolbar,
    # which covers it whenever the mouse is over the chart. y=-0.15 leaves room
    # for two lines of date labels (short periods add the year on a second line).
    # Right margin: room for the end labels ("XLK $245")
    fig.update_layout(margin=dict(r=72),
                      legend=dict(orientation="h", y=-0.15, yanchor="top", x=0))
    return fig


# ---------------------------------------------------------------------------
# 5. Streamlit layout
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_prices() -> pd.DataFrame:
    """load_data() + clean_prices(), cached so the CSV is read once and not on every click."""
    return clean_prices(load_data())


@st.cache_data(show_spinner=False)
def _cached_recession_periods() -> list:
    """NBER recession (start, end) dates, cached."""
    return recession_periods(load_recessions())


def _custom_default_dates(first_day: datetime.date, last_day: datetime.date) -> tuple:
    """Starting value of the Custom dates picker: 1 Jan 2020 (or the first day of data) to the last day."""
    default_start = min(max(CUSTOM_DEFAULT_START, first_day), last_day)
    return default_start, last_day


def _remember_hidden_controls(first_day: datetime.date, last_day: datetime.date) -> None:
    """Keep the values of the controls that are only shown some of the time.

    Streamlit forgets a widget's value on a rerun where the widget isn't drawn
    (e.g. the Relative toggle while the Growth view is shown), and the widget
    would come back with its default. Copying every value into st.session_state
    ourselves on each run keeps it. These widgets are then created without
    value=/index= and take their value from session_state.
    """
    defaults = {
        KEY_METRIC: TOTAL_RETURN,
        KEY_RELATIVE: False,
        KEY_RISK_FREE: DEFAULT_RISK_FREE,
        KEY_HIGHLIGHT: DEFAULT_HIGHLIGHT,
        KEY_DATES: _custom_default_dates(first_day, last_day),
    }
    for key, default in defaults.items():
        st.session_state[key] = st.session_state.get(key, default)


def _draw_controls(first_date: pd.Timestamp, last_date: pd.Timestamp) -> dict:
    """Draw the controls (2 or 3 rows) and return the chosen values.

    Row 1: period and view. Row 2: the options of the chosen view (Ranking:
    the measure and the Relative toggle; Growth of $100: the highlighted
    sector). Row 3 only appears when it is needed: custom dates (Custom
    period) and the risk-free rate (Sharpe ratio only).
    """
    first_day, last_day = first_date.date(), last_date.date()
    _remember_hidden_controls(first_day, last_day)

    row1 = st.columns(2)
    with row1[0]:
        period = st.selectbox("Period", PERIODS, index=PERIODS.index(DEFAULT_PERIOD), key=KEY_PERIOD)
    with row1[1]:
        view = st.radio("View", VIEWS, horizontal=True, key=KEY_VIEW)

    # Values used when a control isn't shown. The Growth view's lines show what
    # $100 became, i.e. the total return, so its key numbers show total return too.
    choices = {"period": period, "view": view, "metric": TOTAL_RETURN, "relative": False,
               "highlight": DEFAULT_HIGHLIGHT, "custom_dates": None,
               "risk_free": st.session_state[KEY_RISK_FREE] / 100}

    # vertical_alignment="bottom" lines the toggle up with the Measure box, not its label
    row2 = st.columns(2, vertical_alignment="bottom")
    if view == RANKING:
        with row2[0]:
            choices["metric"] = st.selectbox("Measure", list(METRICS), key=KEY_METRIC,
                                             help="What the bars rank and the key numbers show.")
        with row2[1]:
            choices["relative"] = st.toggle("Relative to S&P 500", key=KEY_RELATIVE,
                                            help="Show each sector minus the S&P 500 (e.g. return "
                                                 "difference in percentage points).")
    else:
        with row2[0]:
            choices["highlight"] = st.selectbox("Highlight sector", list(SECTORS), format_func=sector_name,
                                                key=KEY_HIGHLIGHT, help="The sector drawn in blue.")

    if period == CUSTOM or choices["metric"] == SHARPE:
        row3 = st.columns(2)
        with row3[0]:
            if period == CUSTOM:
                choices["custom_dates"] = st.date_input("Custom dates", min_value=first_day,
                                                        max_value=last_day, key=KEY_DATES,
                                                        format="DD/MM/YYYY")
        with row3[1]:
            if choices["metric"] == SHARPE:
                risk_free = st.slider("Risk-free rate (per year)", min_value=0.0, max_value=6.0,
                                      step=0.25, format="%.2f%%", key=KEY_RISK_FREE,
                                      help="Return on a safe investment such as 3-month US Treasury "
                                           "bills. The Sharpe ratio rewards only the return above it.")
                choices["risk_free"] = risk_free / 100
    return choices


def _custom_range(dates) -> tuple:
    """(start, end) from the date picker, or (None, None) until both dates are picked.

    The range picker gives a tuple: () before any click, (start,) after the
    first click (older Streamlit versions rerun the app at that moment), and
    (start, end) once both dates are picked.
    """
    if not isinstance(dates, (list, tuple)):
        return None, None
    picked = [day for day in dates if day is not None]
    if len(picked) < 2:
        return None, None
    return picked[0], picked[1]


def _draw_kpis(ranked: pd.Series, benchmark_value: float, metric: str, sector_count: int) -> None:
    """Three key numbers: the S&P 500's value, the best sector and the worst sector."""
    spec = METRICS[metric]
    best, worst = ranked.index[0], ranked.index[-1]
    best_label, worst_label = kpi_labels(metric, ranked)
    delta_color = spec["delta_color"]
    if all_sharpe_ratios_negative(metric, ranked):
        delta_color = "off"  # a less negative Sharpe ratio isn't clearly "good" (no green/red)
    winners = int(beats_benchmark(ranked, benchmark_value, metric).sum())
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"S&P 500 {spec['short']}", format_value(benchmark_value, metric),
                  help=f"{spec['help']} {count_phrase(winners, sector_count).capitalize()} "
                       f"{spec['beat_phrase']} in this period.")
    for column, label, ticker in [(col2, best_label, best), (col3, worst_label, worst)]:
        difference = ranked[ticker] - benchmark_value
        with column:
            st.metric(f"{label}: {short_name(ticker)}", format_value(ranked[ticker], metric),
                      delta=f"{format_value(difference, metric, relative=True)} vs S&P 500",
                      delta_color=delta_color, help=f"{ticker_label(ticker)}. {spec['help']}")


def _window_caption(window: pd.DataFrame, excluded: list[str]) -> str:
    """Which prices were used, and which sectors were left out for missing data."""
    trading_days = len(window) - 1
    text = (f"Measured from the close on {day_label(window.index[0])} to {day_label(window.index[-1])} "
            f"({trading_days} trading days).")
    if excluded:
        names = []
        for ticker in excluded:
            names.append(ticker_label(ticker))
        text += " Not shown, no price at the start of the period: " + ", ".join(names) + "."
    return text


def _metrics_table(metrics: pd.DataFrame, ranked: pd.Series) -> pd.DataFrame:
    """Every measure for the S&P 500 and each sector (best first), in readable units."""
    order = [BENCHMARK] + list(ranked.index)
    table = pd.DataFrame(index=[ticker_label(ticker) for ticker in order])
    table.index.name = "Sector"
    for metric in METRICS:
        values = metrics.loc[order, metric]
        if METRICS[metric]["percent"]:
            table[f"{metric} (%)"] = (values * 100).round(2).to_numpy()
        else:
            table[metric] = values.round(2).to_numpy()
    return table


def _growth_table(growth: pd.DataFrame) -> pd.DataFrame:
    """The plotted growth-of-$100 values, one column per ticker."""
    table = growth.round(2)
    table.columns = [ticker_label(ticker) for ticker in growth.columns]
    table.index = table.index.strftime("%Y-%m-%d")
    table.index.name = "Date"
    return table


# The formulas use LaTeX maths (shown nicely by st.markdown). It is a raw string (r"...")
# so the backslashes reach Streamlit unchanged.
METHOD_FORMULAS = r"""
**Data.** Daily adjusted closing prices (dividends reinvested) of SPY, which tracks the S&P 500,
and the 11 SPDR sector ETFs. A period starts at the close on its first day, so every later
trading day counts. $P_t$ = price on day $t$, $r_t = P_t / P_{t-1} - 1$ = daily return,
252 trading days per year.

- **Total return** $= P_{\text{last}} / P_{\text{first}} - 1$
- **Annualised volatility** $\sigma = \text{std}(r_t) \times \sqrt{252}$ (sample standard deviation)
- **Sharpe ratio** $= \dfrac{\text{mean}(r_t - r_f) \times 252}{\text{std}(r_t) \times \sqrt{252}}$,
  with daily risk-free rate $r_f$ = annual rate ÷ 252
- **Maximum drawdown** $= \min_t \left( P_t / \max_{s \le t} P_s - 1 \right)$, the worst fall from an earlier high
- **Beta** $\beta = \text{cov}(r_{\text{sector}}, r_{\text{S\&P}}) \,/\, \text{var}(r_{\text{S\&P}})$:
  how strongly a sector follows the S&P 500, not how much it swings (a sector can have a low
  beta and still be very volatile, like Energy in 2022)
- **Relative to S&P 500** = sector value − S&P 500 value (percentage points, pp, for % measures)

Bars are sorted best first: highest return, Sharpe ratio and drawdown (smallest fall),
lowest volatility and beta. When excess returns are negative, a more volatile sector gets a
*less* negative Sharpe ratio, so a ranking of negative Sharpe ratios is not a fair test of
risk-adjusted performance. A sector without a price at the start of the period is left out
(XLRE only trades from Oct 2015 and XLC from Jun 2018); a single missing day inside the data
is filled with the previous close. The Growth of \$100 view and its key numbers use total
return. Periods longer than about 2½ years are drawn with one point per week.
"""


def _method_markdown(choices: dict) -> str:
    """Text for the 'Method & data sources' expander: formulas, current settings, sources."""
    links = []
    for ticker in TICKERS:
        links.append(f"[{ticker}](https://finance.yahoo.com/quote/{ticker})")
    settings = f"Risk-free rate used for the Sharpe ratio: {choices['risk_free'] * 100:.2f}% a year."
    sources = ("**Sources:** Yahoo Finance via yfinance: " + ", ".join(links) + ". "
               "Grey bands: NBER US recessions ([USREC](https://fred.stlouisfed.org/series/USREC)).")
    return METHOD_FORMULAS + "\n" + settings + "\n\n" + sources


def render() -> None:
    """Draw the whole panel inside the current Streamlit container."""
    ui.panel_header(NUMBER, TITLE, OWNER, QUESTION)

    try:
        prices = _cached_prices()
    except Exception as exc:  # no snapshot and no internet, or a broken CSV
        ui.no_data_message("Sector price", exc)
        return
    if prices.empty:  # clean_prices keeps only days with an S&P 500 price
        ui.no_data_message("Sector price", ValueError("the data file has no S&P 500 prices"))
        return

    choices = _draw_controls(prices.index[0], prices.index[-1])
    metric = choices["metric"]

    # 1) Cut out the chosen period
    if choices["period"] == CUSTOM:
        custom_start, custom_end = _custom_range(choices["custom_dates"])
        if custom_start is None:
            st.info("Pick both a start and an end date in the calendar.")
            return
        first_day, last_day = min(custom_start, custom_end), max(custom_start, custom_end)
        period_phrase = f"{day_label(first_day)} – {day_label(last_day)}"
    else:
        custom_start, custom_end = None, None
        period_phrase = PERIOD_PHRASES[choices["period"]]
    start, end = period_dates(choices["period"], prices.index[-1], custom_start, custom_end)
    window = window_prices(prices, start, end)
    if len(window) - 1 < MIN_TRADING_DAYS:
        st.info(f"This period has {max(len(window) - 1, 0)} trading days of prices, too few to compare "
                f"sectors. Pick a period of at least {MIN_TRADING_DAYS} trading days.")
        return

    # 2) Leave out the ETFs without a starting price (launched inside the window)
    complete, excluded = split_by_data_coverage(window)
    if BENCHMARK not in complete or len(complete) < 2:
        st.info("The S&P 500 or every sector is missing prices in this period. Try another period.")
        return
    window = window[complete]

    # 3) Measures for every ticker, ranked best first
    metrics = compute_metrics(window, choices["risk_free"])
    benchmark_value = metrics.loc[BENCHMARK, metric]
    ranked = rank_sectors(metrics[metric], metric)
    if ranked.empty or pd.isna(benchmark_value):
        st.info(f"The {metric.lower()} can't be measured in this period (prices didn't move). "
                "Try a longer period.")
        return

    _draw_kpis(ranked, benchmark_value, metric, len(ranked))

    # 4) Chart, caption, insight and data table for the chosen view
    caption = _window_caption(window, excluded)
    if choices["view"] == RANKING:
        fig = build_ranking_chart(ranked, benchmark_value, metric, choices["relative"])
        insight = ranking_insight(metric, ranked, benchmark_value, period_phrase)
        table = _metrics_table(metrics, ranked)
        file_name = f"{SLUG}_ranking.csv"
    else:
        highlight = choices["highlight"]
        if highlight not in window.columns:
            # e.g. Communication Services before 2018: highlight the best performer instead
            best_return = rank_sectors(metrics[TOTAL_RETURN], TOTAL_RETURN).index[0]
            caption += (f" {sector_name(highlight)} can't be highlighted for this period, so the "
                        f"best performer, {sector_name(best_return)}, is shown in blue instead.")
            highlight = best_return
        growth = growth_of_100(window)
        if len(growth) > MAX_DAILY_POINTS:
            growth = thin_to_weekly(growth)
        try:
            recessions = _cached_recession_periods()
        except Exception:  # shading is nice to have, the chart works without it
            recessions = []
        fig = build_growth_chart(growth, highlight, recessions)
        insight = growth_insight(growth, highlight, metrics.loc[highlight, DRAWDOWN])
        table = _growth_table(growth)
        file_name = f"{SLUG}_growth_of_100.csv"

    ui.show_chart(fig, key=f"{SLUG}_chart")
    st.caption(markdown_safe(caption))
    ui.insight(markdown_safe(insight))
    ui.details(_method_markdown(choices), table, key=SLUG, file_name=file_name)
