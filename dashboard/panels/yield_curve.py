"""Panel 2 - The Yield Curve.

Question: what is the bond market signalling about recessions?

The yield curve plots the interest rate (yield) the US government pays to
borrow for different lengths of time, from 3 months to 30 years. Normally
long-term rates are higher than short-term rates, because lenders want extra
pay for tying their money up for longer. When short-term rates rise ABOVE
long-term rates the curve is "inverted": investors expect the central bank to
cut rates in future, usually because they expect the economy to weaken.
Inversions have come before most US recessions, which is why markets watch
the "spread" (10-year yield minus 2-year yield, or minus 3-month yield).

The panel has two views:

* Curve snapshots - the whole curve on a few chosen days, to compare its shape.
* Spread history  - the spread over time, with inversions in red, recessions
  shaded grey, and whether a recession followed each inversion (and how soon).

Module layout: constants -> data loading -> analysis functions (pure, no
Streamlit, tested in tests/test_yield_curve.py) -> chart builders -> render().
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme, ui
from dashboard.data import (
    DataUnavailableError,
    filter_dates,
    load_fred,
    load_recessions,
    recession_periods,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMBER = 2
SLUG = "yield_curve"
TITLE = "The Yield Curve"
OWNER = "Team member 2"  # panel owner: put your name here
QUESTION = "What is the bond market signalling about recessions?"

# FRED series ID -> short maturity label, shortest maturity first.
# DGS = "market yield on US Treasury securities at constant maturity" (daily, % per year)
MATURITIES = {
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
    "DGS3": "3Y",
    "DGS5": "5Y",
    "DGS7": "7Y",
    "DGS10": "10Y",
    "DGS20": "20Y",
    "DGS30": "30Y",
}
MATURITY_LABELS = list(MATURITIES.values())  # the x-axis order of the curve chart
DATASET_NAME = "yield_curve_treasury"  # saved as data/yield_curve_treasury.csv
START_DATE = "1990-01-01"

# The two spreads the panel can show: name -> (long maturity, short maturity)
SPREADS = {
    "10Y - 2Y": ("10Y", "2Y"),
    "10Y - 3M": ("10Y", "3M"),
}

# Inversion episode rule (see find_inversion_episodes for the full explanation)
MIN_DAYS_DEFAULT = 10  # an episode needs at least this many inverted trading days
MAX_GAP_DAYS = 63      # inversions at most ~3 months (63 trading days) apart are one episode

# Lead-time rule (see episode_outcome). A recession only counts as "following" an
# inversion if it begins within this many months of the inversion's start.
# Without a limit, a recession many years later would still count as a hit,
# and an inversion could never be called a false alarm.
MAX_LEAD_DEFAULT = 30

# The three possible outcomes of an inversion episode
FOLLOWED = "followed"          # a recession began within the time limit
NO_RECESSION = "no recession"  # the time limit has passed without one: a false alarm
TOO_SOON = "too soon"          # no recession yet, but the time limit hasn't passed

FLAT_LIMIT_BP = 25  # a spread between 0 and 25 bp is described as a "flat" curve

# Named moments for the snapshot view. Fixed dates are snapped to the nearest
# earlier trading day; the others are worked out from the data.
LATEST = "Latest"
YEAR_AGO = "1 year ago"
DEEPEST = "Deepest inversion"
CUSTOM = "Custom date"
FIXED_MOMENTS = {
    "Pre-COVID (Jan 2020)": "2020-01-31",
    "Financial crisis (Dec 2008)": "2008-12-31",
    "Dot-com peak (Mar 2000)": "2000-03-10",
}
NAMED_MOMENTS = [LATEST, YEAR_AGO, DEEPEST] + list(FIXED_MOMENTS)  # the multiselect options
MOMENT_NAMES = NAMED_MOMENTS + [CUSTOM]  # custom date comes from its own date picker
DEFAULT_MOMENTS = [LATEST, YEAR_AGO, DEEPEST]
MAX_MOMENTS = 4
CUSTOM_DATE_DEFAULT = dt.date(2019, 8, 28)  # the short 2019 inversion before the COVID recession

# Each moment keeps the same colour whatever else is selected (theme order)
MOMENT_COLOURS = {}
for position, moment_name in enumerate(MOMENT_NAMES):
    MOMENT_COLOURS[moment_name] = theme.SERIES[position]

# Period choices for the spread history: label -> number of years (None = all data)
PERIODS = {"1Y": 1, "2Y": 2, "5Y": 5, "10Y": 10, "20Y": 20, "All": None}
WEEKLY_AFTER_YEARS = 5  # windows longer than this are plotted with weekly values

SNAPSHOT_VIEW = "Curve snapshots"
HISTORY_VIEW = "Spread history"

# Space (px) above the plot: room for the legend rows below Plotly's toolbar
TOP_MARGIN = 56
LEGEND_ROW_HEIGHT = 22


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(refresh: bool = False) -> pd.DataFrame:
    """Daily Treasury yields (% per year), one column per FRED series ID.

    Reads data/yield_curve_treasury.csv, or downloads it from FRED when
    ``refresh`` is True or the file doesn't exist yet.
    """
    return load_fred(DATASET_NAME, list(MATURITIES), start=START_DATE, refresh=refresh)


def refresh_data() -> None:
    """Download fresh data for this panel (used by the refresh button and scripts/update_data.py)."""
    load_data(refresh=True)


def tidy_yields(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename the FRED columns to maturity labels (DGS10 -> 10Y) and drop empty days."""
    yields = raw.rename(columns=MATURITIES)
    yields = yields[MATURITY_LABELS]
    return yields.dropna(how="all")


# ---------------------------------------------------------------------------
# Analysis functions (no Streamlit here, so they can be unit tested)
# ---------------------------------------------------------------------------

def compute_spread_bp(yields: pd.DataFrame, long: str = "10Y", short: str = "2Y") -> pd.Series:
    """Long yield minus short yield, in basis points (1 bp = 0.01 percentage points).

    A negative spread means the curve is inverted: borrowing for a short time
    costs more than borrowing for a long time. Days where either yield is
    missing are dropped.
    """
    spread = (yields[long] - yields[short]) * 100
    spread = spread.dropna()
    spread.name = f"{long} - {short} spread (bp)"
    return spread


def snap_to_previous_date(dates: pd.DatetimeIndex, target) -> pd.Timestamp | None:
    """The last date in ``dates`` on or before ``target`` (None if there is none).

    Markets are closed at weekends and on holidays, so a date someone picks may
    have no data. We then use the most recent trading day before it, which is
    what an investor on that day would actually have seen.
    """
    if target is None:
        return None
    target = pd.Timestamp(target)
    earlier_dates = dates[dates <= target]
    if len(earlier_dates) == 0:
        return None
    return earlier_dates.max()


def describe_curve(spread_bp: float) -> str:
    """One-word description of the curve's shape from its spread in basis points."""
    if spread_bp < 0:
        return "inverted"
    if spread_bp < FLAT_LIMIT_BP:
        return "flat"
    return "upward-sloping"


def find_inversion_runs(spread_bp: pd.Series) -> list[tuple[int, int]]:
    """Positions (first, last) of every run of consecutive days with a negative spread.

    Positions are row numbers (0, 1, 2, ...), so one step = one trading day.
    """
    is_inverted = spread_bp.to_numpy() < 0  # NumPy: one True/False per day
    runs = []
    run_start = None
    for position, inverted_today in enumerate(is_inverted):
        if inverted_today and run_start is None:
            run_start = position  # an inversion begins
        elif not inverted_today and run_start is not None:
            runs.append((run_start, position - 1))  # it ended yesterday
            run_start = None
    if run_start is not None:  # still inverted on the last day of data
        runs.append((run_start, len(is_inverted) - 1))
    return runs


def merge_close_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Join runs that are separated by ``max_gap`` trading days or fewer.

    The spread often pops back above zero for a few days in the middle of an
    inversion. Those short breaks don't mean the signal has gone away, so the
    runs on either side are treated as one episode.
    """
    merged: list[tuple[int, int]] = []
    for first, last in runs:
        if merged:
            previous_first, previous_last = merged[-1]
            gap = first - previous_last - 1  # non-inverted trading days in between
            if gap <= max_gap:
                merged[-1] = (previous_first, last)
                continue
        merged.append((first, last))
    return merged


def find_inversion_episodes(spread_bp: pd.Series, min_days: int = MIN_DAYS_DEFAULT,
                            max_gap: int = MAX_GAP_DAYS) -> list[dict]:
    """Find the inversion episodes in a daily spread series.

    Rule, step by step:

    1. A trading day is "inverted" when the spread is below zero.
    2. Consecutive inverted days form a run.
    3. Runs separated by ``max_gap`` trading days or fewer are merged into one
       episode (a short break doesn't end the signal).
    4. An episode is kept only if it contains at least ``min_days`` inverted
       days, so a one-day blip isn't counted as a recession warning.

    Returns a list of dicts, oldest first, with the keys ``start``, ``end``,
    ``inverted_days``, ``deepest_bp``, ``deepest_date`` and ``ongoing`` (True
    when the episode is still running on the last day of data).
    """
    spread_bp = spread_bp.dropna()
    is_inverted = spread_bp.to_numpy() < 0
    runs = merge_close_runs(find_inversion_runs(spread_bp), max_gap)

    episodes = []
    for first, last in runs:
        inverted_days = int(is_inverted[first:last + 1].sum())  # True counts as 1
        if inverted_days < min_days:
            continue  # too short to count as a signal
        episode_spread = spread_bp.iloc[first:last + 1]
        episodes.append({
            "start": spread_bp.index[first],
            "end": spread_bp.index[last],
            "inverted_days": inverted_days,
            "deepest_bp": float(episode_spread.min()),
            "deepest_date": episode_spread.idxmin(),
            "ongoing": last == len(spread_bp) - 1,
        })
    return episodes


def months_between(earlier: pd.Timestamp, later: pd.Timestamp) -> int:
    """Number of calendar months from the month of ``earlier`` to the month of ``later``.

    Only the year and month count, not the day: 15 Mar -> 1 Aug = 5 (March to
    August), and 20 Dec 2007 -> 1 Jan 2008 = 1. Recessions are dated by month,
    so we measure lead times in calendar months too.
    """
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def next_recession_start(start: pd.Timestamp, recession_starts: list[pd.Timestamp]) -> pd.Timestamp | None:
    """The start date of the first recession that begins after ``start`` (None if none has yet)."""
    for recession_start in sorted(recession_starts):
        if recession_start > start:
            return recession_start
    return None


def episode_outcome(lead_months: int | None, months_since_start: int,
                    max_lead: int = MAX_LEAD_DEFAULT) -> str:
    """Was an inversion followed by a recession within ``max_lead`` months?

    * FOLLOWED: the next recession began at most ``max_lead`` months after the
      inversion started (exactly ``max_lead`` still counts).
    * NO_RECESSION: more than ``max_lead`` months have passed and no recession
      began in that time - a false alarm, even if one came later.
    * TOO_SOON: no recession yet, but ``max_lead`` months haven't passed, so we
      can't judge it yet.
    """
    if lead_months is not None:
        if lead_months <= max_lead:
            return FOLLOWED
        return NO_RECESSION  # the next recession came, but too late to count
    if months_since_start > max_lead:
        return NO_RECESSION  # the time limit has passed with no recession at all
    return TOO_SOON


def add_lead_times(episodes: list[dict], recession_starts: list[pd.Timestamp],
                   latest_date: pd.Timestamp, max_lead: int = MAX_LEAD_DEFAULT) -> list[dict]:
    """Add what happened after each inversion episode (``latest_date`` = last day of data).

    New keys in each episode dict:

    * ``recession_start`` - start of the next recession (None if none since)
    * ``lead_months`` - calendar months from the episode's start to it (or None)
    * ``months_since_start`` - calendar months from the episode's start to ``latest_date``
    * ``outcome`` - FOLLOWED, NO_RECESSION or TOO_SOON (see episode_outcome)
    * ``first_warning`` - True for the FIRST episode followed by a given recession

    Sometimes two inversions come before the same recession (e.g. the 10Y-3M
    inverted in 2019 and again in Jan 2020, before the COVID recession). Only
    the first one is a ``first_warning``, so the recession isn't counted twice.
    """
    # Only recessions that had begun by the last day of data can count
    known_starts = []
    for recession_start in recession_starts:
        if recession_start <= latest_date:
            known_starts.append(recession_start)

    warned_recessions = []  # recessions an earlier episode already warned about
    for episode in episodes:
        recession_start = next_recession_start(episode["start"], known_starts)
        lead_months = None
        if recession_start is not None:
            lead_months = months_between(episode["start"], recession_start)
        months_since_start = months_between(episode["start"], latest_date)
        outcome = episode_outcome(lead_months, months_since_start, max_lead)

        first_warning = outcome == FOLLOWED and recession_start not in warned_recessions
        if first_warning:
            warned_recessions.append(recession_start)

        episode["recession_start"] = recession_start
        episode["lead_months"] = lead_months
        episode["months_since_start"] = months_since_start
        episode["outcome"] = outcome
        episode["first_warning"] = first_warning
    return episodes


def summarise_lead_times(episodes: list[dict]) -> dict:
    """Count the episodes' outcomes and summarise the lead times to recession.

    Returns a dict with ``episodes`` (how many), ``followed``, ``no_recession``
    and ``too_soon`` (how many episodes had each outcome), ``recessions`` (how
    many different recessions followed an inversion) and, when there is at
    least one, the ``mean``, ``min`` and ``max`` lead in months. Lead times are
    measured from the FIRST inversion before each recession, so every recession
    counts once.
    """
    counts = {FOLLOWED: 0, NO_RECESSION: 0, TOO_SOON: 0}
    leads = []
    for episode in episodes:
        counts[episode["outcome"]] += 1
        if episode["first_warning"]:
            leads.append(episode["lead_months"])
    summary = {
        "episodes": len(episodes),
        "followed": counts[FOLLOWED],
        "no_recession": counts[NO_RECESSION],
        "too_soon": counts[TOO_SOON],
        "recessions": len(leads),
        "mean": None,
        "min": None,
        "max": None,
    }
    if leads:
        summary["mean"] = float(np.mean(leads))
        summary["min"] = int(np.min(leads))
        summary["max"] = int(np.max(leads))
    return summary


def moment_dates(yields: pd.DataFrame, moments: list[str], spread_bp: pd.Series,
                 custom_date=None) -> dict[str, pd.Timestamp]:
    """The trading day used for each chosen moment, in the fixed MOMENT_NAMES order.

    ``Deepest inversion`` is the day ``spread_bp`` was lowest. A moment with no
    data on or before its date is left out.
    """
    dates = yields.index
    latest = dates.max()
    chosen = {}
    for name in MOMENT_NAMES:
        if name not in moments:
            continue
        if name == LATEST:
            target = latest
        elif name == YEAR_AGO:
            target = latest - pd.DateOffset(years=1)
        elif name == DEEPEST:
            target = spread_bp.idxmin() if len(spread_bp) > 0 else None
        elif name == CUSTOM:
            target = custom_date
        else:
            target = FIXED_MOMENTS[name]
        snapped = snap_to_previous_date(dates, target)
        if snapped is not None:
            chosen[name] = snapped
    return chosen


def drop_same_day_moments(dates: dict[str, pd.Timestamp]) -> tuple[dict[str, pd.Timestamp], list[str]]:
    """Keep one moment per trading day, so the same curve isn't drawn twice.

    E.g. a custom date of today is the same day as 'Latest'. The moment that
    comes first in ``dates`` is kept. Returns (moments kept, names dropped).
    """
    kept = {}
    dropped = []
    for name, date in dates.items():
        if date in kept.values():
            dropped.append(name)
        else:
            kept[name] = date
    return kept, dropped


def last_n_years(frame: pd.DataFrame, years: int | None) -> pd.DataFrame:
    """The last ``years`` years of ``frame`` (all of it when ``years`` is None)."""
    if years is None or frame.empty:
        return frame
    start = frame.index.max() - pd.DateOffset(years=years)
    return filter_dates(frame, start=start)


def recessions_in_window(recessions: list[tuple[pd.Timestamp, pd.Timestamp]],
                         start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The (start, end) recession periods that overlap the window ``start``..``end``."""
    overlapping = []
    for recession_start, recession_end in recessions:
        if recession_start <= end and recession_end >= start:
            overlapping.append((recession_start, recession_end))
    return overlapping


def to_weekly(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per week, the last trading day of that week, so long charts stay fast.

    Rows keep their real dates (e.g. a Thursday before a Friday holiday).
    """
    weeks = pd.Grouper(freq="W-FRI")  # calendar weeks ending on Friday
    return frame.groupby(weeks).tail(1)


# ---------------------------------------------------------------------------
# Text helpers: insights, KPI help and the episode table
# ---------------------------------------------------------------------------

def format_date(date: pd.Timestamp) -> str:
    """A date like '3 Jul 2023'."""
    return f"{date.day} {date:%b %Y}"


def format_bp(value_bp: float) -> str:
    """A spread like '+42 bp' or '-108 bp'."""
    return f"{value_bp:+.0f} bp"


def short_moment_name(name: str) -> str:
    """'Dot-com peak (Mar 2000)' -> 'Dot-com peak' (the real date is shown next to it anyway)."""
    return name.split(" (")[0]


def plural(count: int, word: str) -> str:
    """'1 episode', '3 episodes'."""
    if count == 1:
        return f"{count} {word}"
    return f"{count} {word}s"


def was_or_were(count: int) -> str:
    """'was' for one thing, 'were' for several."""
    if count == 1:
        return "was"
    return "were"


def list_months(episodes: list[dict]) -> str:
    """The episodes' start months as text: 'May 1998' or 'May 1998 and Apr 2022'."""
    months = []
    for episode in episodes:
        months.append(f"{episode['start']:%b %Y}")
    if len(months) == 1:
        return months[0]
    return ", ".join(months[:-1]) + " and " + months[-1]


def outcome_text(episode: dict, max_lead: int) -> str:
    """What followed an episode, written so it can follow 'the inversion ...'.

    e.g. 'was followed by a recession 14 months after it began'.
    """
    if episode["outcome"] == FOLLOWED:
        return f"was followed by a recession {plural(episode['lead_months'], 'month')} after it began"
    if episode["outcome"] == NO_RECESSION:
        text = f"was not followed by a recession within {max_lead} months"
        if episode["lead_months"] is None:
            return text + f" ({episode['months_since_start']} months have passed since it began)"
        return text + f" (the next one began {episode['lead_months']} months after it)"
    return f"has not been followed by a recession so far, but it is too soon to judge with a {max_lead}-month limit"


def episode_containing(date: pd.Timestamp, episodes: list[dict]) -> dict | None:
    """The inversion episode that ``date`` falls in (None if it isn't in one)."""
    for episode in episodes:
        if episode["start"] <= date <= episode["end"]:
            return episode
    return None


def inversion_outcome_clause(date: pd.Timestamp, spread_bp: float, episodes: list[dict],
                             max_lead: int) -> str:
    """'; that inversion began on ... and was (not) followed ...' for an inverted curve.

    Empty text when the curve isn't inverted or the day isn't part of an
    episode (e.g. a dip too short to count).
    """
    if spread_bp >= 0:
        return ""
    episode = episode_containing(date, episodes)
    if episode is None:
        return ""
    return f"; that inversion began on {format_date(episode['start'])} and {outcome_text(episode, max_lead)}"


def curve_spreads(curves: dict[str, pd.Series], long: str, short: str) -> dict[str, float]:
    """The long - short spread (bp) of each curve, skipping curves where a yield is missing."""
    spreads = {}
    for name, curve in curves.items():
        if pd.notna(curve.get(long)) and pd.notna(curve.get(short)):
            spreads[name] = (curve[long] - curve[short]) * 100
    return spreads


def comparison_sentence(spreads: dict[str, float], dates: dict[str, pd.Timestamp], spread_name: str,
                        episodes: list[dict], max_lead: int) -> str:
    """How the first curve shown compares with the others (needs at least 2 curves)."""
    first_name = list(spreads)[0]
    lowest_name = min(spreads, key=spreads.get)
    highest_name = max(spreads, key=spreads.get)
    lowest = spreads[lowest_name]
    highest = spreads[highest_name]

    if round(highest) == round(lowest):
        return f"All the dates shown have the same {spread_name} spread."

    if lowest_name != first_name:
        where = f"{format_date(dates[lowest_name])} ({short_moment_name(lowest_name)}), at {format_bp(lowest)}"
        if lowest < 0:
            return (
                f"Of the dates shown it was most inverted on {where}, the classic recession warning"
                + inversion_outcome_clause(dates[lowest_name], lowest, episodes, max_lead) + "."
            )
        return f"Of the dates shown it was flattest on {where}."

    # The first curve has the lowest spread, so compare it with the highest one
    where = f"{format_date(dates[highest_name])} ({short_moment_name(highest_name)}), at {format_bp(highest)}"
    if highest < 0:
        return f"That is the most inverted of the dates shown; it was least inverted on {where}."
    return f"That is the lowest spread of the dates shown; it was steepest on {where}."


def snapshot_insight(curves: dict[str, pd.Series], dates: dict[str, pd.Timestamp], spread_name: str,
                     episodes: list[dict], max_lead: int = MAX_LEAD_DEFAULT) -> str:
    """One or two sentences about the curves in the snapshot chart.

    Sentence 1 describes the first curve. Sentence 2 compares it with the
    others. For the most inverted curve shown we also say whether a recession
    followed that inversion (``episodes`` need their outcomes, see add_lead_times).
    """
    long, short = SPREADS[spread_name]
    spreads = curve_spreads(curves, long, short)
    if not spreads:
        return f"The {long} or {short} yield is missing on the chosen dates, so the slope can't be measured."

    first_name = list(spreads)[0]
    first_spread = spreads[first_name]
    gap_word = "above" if first_spread >= 0 else "below"
    text = (
        f"On {format_date(dates[first_name])} ({short_moment_name(first_name)}) the curve was "
        f"{describe_curve(first_spread)}: the {long} yield was {abs(first_spread):.0f} bp "
        f"{gap_word} the {short}"
    )
    lowest_name = min(spreads, key=spreads.get)
    if lowest_name == first_name:
        # The first curve is the most inverted one shown, so its outcome goes here
        text += inversion_outcome_clause(dates[first_name], first_spread, episodes, max_lead)
    text += "."

    if len(spreads) >= 2:
        text += " " + comparison_sentence(spreads, dates, spread_name, episodes, max_lead)
    return text


def current_state_sentence(spread_bp: pd.Series, episodes: list[dict], spread_name: str,
                           min_days: int, max_lead: int) -> str:
    """Where the spread is now, and what happened after the latest inversion episode."""
    latest_date = spread_bp.index[-1]
    latest = spread_bp.iloc[-1]

    if episodes and episodes[-1]["ongoing"]:
        last = episodes[-1]
        # The episode can join several dips, so we don't say "inverted since <start>"
        return (
            f"The {spread_name} spread is {format_bp(latest)} on {format_date(latest_date)}: the curve is "
            f"inverted, in an episode that began on {format_date(last['start'])} and has "
            f"{plural(last['inverted_days'], 'inverted day')} so far (dips less than 3 months apart count "
            f"as one episode); it {outcome_text(last, max_lead)}."
        )

    if latest >= 0:
        state = "not inverted"
    else:
        # This dip may still grow into an episode, hence "so far"
        state = f"inverted, but so far too briefly to count under the {min_days}-day rule"
    text = f"The {spread_name} spread is {format_bp(latest)} on {format_date(latest_date)} ({state})"
    if not episodes:
        first_year = spread_bp.index[0].year
        return text + f", and it has not been inverted for {min_days}+ trading days since {first_year}."
    last = episodes[-1]
    return (
        text + f"; its last inversion, {format_date(last['start'])} to {format_date(last['end'])}, "
        f"{outcome_text(last, max_lead)}."
    )


def track_record_sentence(episodes: list[dict], first_year: int, max_lead: int) -> str:
    """What followed all the inversion episodes since ``first_year`` ('' if there are fewer than 2)."""
    if len(episodes) < 2:
        return ""  # the only episode is already described in the first sentence
    summary = summarise_lead_times(episodes)
    false_alarms = []
    too_soon = []
    for episode in episodes:
        if episode["outcome"] == NO_RECESSION:
            false_alarms.append(episode)
        elif episode["outcome"] == TOO_SOON:
            too_soon.append(episode)

    if summary["recessions"] == 0:
        text = (
            f"Since {first_year}, none of the {len(episodes)} inversion episodes "
            f"was followed by a recession within {max_lead} months"
        )
    else:
        text = f"Since {first_year}, {plural(summary['recessions'], 'recession')} began within {max_lead} months of an inversion"
        if summary["recessions"] == 1:
            text += f", {plural(summary['min'], 'month')} after the first warning"
        else:
            text += (
                f", on average {summary['mean']:.0f} months after the first warning "
                f"(range {summary['min']}-{summary['max']} months)"
            )
        if false_alarms:
            text += (
                f"; {plural(len(false_alarms), 'inversion')} ({list_months(false_alarms)}) "
                f"{was_or_were(len(false_alarms))} not followed by one within {max_lead} months"
            )
    if len(too_soon) == 1:
        text += f"; it is too soon to judge the one that began in {list_months(too_soon)}"
    elif len(too_soon) > 1:
        text += f"; it is too soon to judge the ones that began in {list_months(too_soon)}"
    return text + "."


def history_insight(spread_bp: pd.Series, episodes: list[dict], spread_name: str, min_days: int,
                    max_lead: int = MAX_LEAD_DEFAULT) -> str:
    """Two sentences: where the spread is now, and what followed past inversion episodes.

    ``episodes`` must already have their outcomes (see add_lead_times).
    """
    if spread_bp.empty:
        return f"There is no {spread_name} data to analyse."
    first = current_state_sentence(spread_bp, episodes, spread_name, min_days, max_lead)
    second = track_record_sentence(episodes, spread_bp.index[0].year, max_lead)
    if second == "":
        return first
    return first + " " + second


def lead_kpi_help(summary: dict, spread_name: str, min_days: int, max_lead: int, first_year: int) -> str:
    """The '?' help text of the 'Avg. lead to recession' KPI."""
    return (
        f"Calendar months from the first {spread_name} inversion ({min_days}+ inverted days) before a "
        f"recession to the month the recession began, counting recessions that began within {max_lead} "
        f"months. Since {first_year}: {plural(summary['recessions'], 'recession')} followed an inversion "
        f"this way; {plural(summary['no_recession'], 'inversion')} {was_or_were(summary['no_recession'])} "
        f"not followed by one within {max_lead} months."
    )


def episodes_markdown(episodes: list[dict], max_lead: int) -> str:
    """The inversion episodes as a Markdown table for the 'Method' expander."""
    if not episodes:
        return "_No inversion episodes match the current rule._"
    lines = [
        f"| Start | End | Inverted days | Deepest | Recession within {max_lead} months? |",
        "|---|---|---:|---:|---|",
    ]
    for episode in episodes:
        if episode["outcome"] == FOLLOWED:
            outcome = f"Yes, {plural(episode['lead_months'], 'month')} later"
            if not episode["first_warning"]:
                outcome += " (same recession as the inversion before)"
        elif episode["outcome"] == NO_RECESSION:
            outcome = "No"
            if episode["lead_months"] is not None:
                outcome += f" (the next began {episode['lead_months']} months later)"
        else:
            outcome = "Too soon to judge"
        end = "still inverted" if episode["ongoing"] else format_date(episode["end"])
        lines.append(
            f"| {format_date(episode['start'])} | {end} | {episode['inverted_days']} "
            f"| {format_bp(episode['deepest_bp'])} | {outcome} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chart builders (return Plotly figures, no Streamlit here)
# ---------------------------------------------------------------------------

def hex_to_rgba(hex_colour: str, alpha: float) -> str:
    """'#2a78d6' -> 'rgba(42, 120, 214, 0.3)' so a theme colour can be made see-through."""
    red = int(hex_colour[1:3], 16)
    green = int(hex_colour[3:5], 16)
    blue = int(hex_colour[5:7], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def build_snapshot_chart(curves: dict[str, pd.Series], dates: dict[str, pd.Timestamp]) -> go.Figure:
    """Line + marker chart of the yield curve on each chosen date (x = maturity)."""
    fig = go.Figure()
    for name, curve in curves.items():
        curve = curve.dropna()  # e.g. no 20Y yield before Oct 1993
        label = f"{short_moment_name(name)} · {format_date(dates[name])}"
        fig.add_trace(go.Scatter(
            x=list(curve.index),
            y=curve.to_numpy(),
            name=label,
            mode="lines+markers",
            line=dict(color=MOMENT_COLOURS[name], width=2),
            marker=dict(size=8, color=MOMENT_COLOURS[name], line=dict(color="white", width=1.5)),
            hovertemplate="%{y:.2f}%",
        ))
    theme.style_figure(fig, x_title="Time to maturity", y_title="Yield (% per year)")
    # Maturities are ordered categories (3M, 6M, ... 30Y), not numbers
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=MATURITY_LABELS,
                     automargin=True)
    # Same number of decimals on every tick (3.5%, 4.0%, ...); automargin leaves room for the labels
    fig.update_yaxes(tickformat=".1f", ticksuffix="%", automargin=True)
    # About two legend entries fit on one row at half-screen width (~500 px), so the
    # legend needs one row per two curves. The top margin grows with it, which keeps
    # the legend below Plotly's toolbar (shown at the top right on hover).
    legend_rows = max(1, (len(curves) + 1) // 2)
    fig.update_layout(showlegend=True, margin=dict(t=TOP_MARGIN + LEGEND_ROW_HEIGHT * (legend_rows - 1)))
    return fig


def split_by_sign(spread_bp: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split the spread into a normal (>= 0) part and an inverted (< 0) part.

    Each part is blank (NaN) where the other one applies, so each area only
    draws its own stretch. On the day next to a change of sign the part is set
    to 0, so the blue and red areas meet at the zero line without a gap.
    """
    is_normal = spread_bp >= 0
    is_inverted = spread_bp < 0
    near_normal = is_normal | is_normal.shift(1, fill_value=False) | is_normal.shift(-1, fill_value=False)
    near_inverted = is_inverted | is_inverted.shift(1, fill_value=False) | is_inverted.shift(-1, fill_value=False)
    normal_part = spread_bp.clip(lower=0).where(near_normal)
    inverted_part = spread_bp.clip(upper=0).where(near_inverted)
    return normal_part, inverted_part


def build_spread_chart(spread_bp: pd.Series, spread_name: str,
                       recessions: list[tuple[pd.Timestamp, pd.Timestamp]],
                       show_recessions: bool = True) -> go.Figure:
    """Area chart of the spread: blue above zero, red when inverted, grey recession bands."""
    long, short = SPREADS[spread_name]
    normal_part, inverted_part = split_by_sign(spread_bp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal_part.index, y=normal_part.to_numpy(), name=f"Normal: {long} above {short}",
        mode="lines", line=dict(color=theme.POSITIVE, width=2),
        fill="tozeroy", fillcolor=hex_to_rgba(theme.POSITIVE, 0.25), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=inverted_part.index, y=inverted_part.to_numpy(), name=f"Inverted: {long} below {short}",
        mode="lines", line=dict(color=theme.NEGATIVE, width=2),
        fill="tozeroy", fillcolor=hex_to_rgba(theme.NEGATIVE, 0.3), hoverinfo="skip",
    ))
    # An invisible line along the real spread gives one clean hover value per day
    fig.add_trace(go.Scatter(
        x=spread_bp.index, y=spread_bp.to_numpy(), name=f"{spread_name} spread",
        mode="lines", line=dict(color=theme.INK_SECONDARY, width=0),
        showlegend=False, hovertemplate="%{y:+.0f} bp",
    ))
    visible_recessions = []
    if show_recessions and not spread_bp.empty:
        visible_recessions = recessions_in_window(recessions, spread_bp.index.min(), spread_bp.index.max())
    if visible_recessions:
        theme.add_recession_bands(fig, visible_recessions, spread_bp.index.min(), spread_bp.index.max())
        # Legend entry for the grey bands (shapes don't appear in the legend on their own)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], name="US recession", mode="markers", hoverinfo="skip",
            marker=dict(symbol="square", size=13, color=theme.RECESSION_FILL),
        ))
    fig.add_hline(y=0, line_color=theme.INK_MUTED, line_width=1)
    theme.style_figure(fig, y_title=f"{spread_name} spread (bp)")
    fig.update_layout(margin=dict(t=TOP_MARGIN))
    if not spread_bp.empty:
        # Direct label for the latest value, in the right margin next to the last point.
        # x is in "paper" units (1 = right edge of the plot), so the label doesn't
        # stretch the date axis past the last day of data.
        fig.add_annotation(
            xref="paper", x=1, yref="y", y=spread_bp.iloc[-1], text=format_bp(spread_bp.iloc[-1]),
            showarrow=False, xanchor="left", xshift=4, font=dict(size=12, color=theme.INK),
        )
        fig.update_layout(margin=dict(r=56))
        years_shown = (spread_bp.index.max() - spread_bp.index.min()).days / 365.25
        if years_shown > 10:
            fig.update_xaxes(dtick="M60", tickformat="%Y")  # a year label every 5 years
    fig.update_xaxes(hoverformat="%d %b %Y", automargin=True)
    fig.update_yaxes(ticksuffix=" bp", automargin=True)  # automargin: leave room for tick labels
    return fig


# ---------------------------------------------------------------------------
# Streamlit panel
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_yields() -> pd.DataFrame:
    """Tidy yields, cached so the CSV is only read once per session."""
    return tidy_yields(load_data())


@st.cache_data(show_spinner=False)
def _cached_recessions() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of every NBER recession, cached."""
    return recession_periods(load_recessions())


def _remember_control_values(yields: pd.DataFrame) -> None:
    """Keep each view's control values when the user switches views.

    Streamlit forgets a widget's value on a rerun where the widget isn't drawn
    (e.g. the Spread history sliders while Curve snapshots is shown), and the
    widget would come back with its default. Copying every value into
    st.session_state ourselves on each run keeps it. The widgets are then
    created without value=/default= and take their value from session_state.
    """
    first_date = yields.index.min().date()
    last_date = yields.index.max().date()
    defaults = {
        f"{SLUG}_moments": list(DEFAULT_MOMENTS),
        f"{SLUG}_use_custom": False,
        f"{SLUG}_custom_date": min(max(CUSTOM_DATE_DEFAULT, first_date), last_date),
        f"{SLUG}_period": "All",
        f"{SLUG}_recessions": True,
        f"{SLUG}_min_days": MIN_DAYS_DEFAULT,
        f"{SLUG}_max_lead": MAX_LEAD_DEFAULT,
    }
    for key, default in defaults.items():
        st.session_state[key] = st.session_state.get(key, default)


def _episodes_with_outcomes(spread_bp: pd.Series, min_days: int, max_lead: int) -> list[dict]:
    """Inversion episodes of ``spread_bp`` with what followed each one (see add_lead_times)."""
    if spread_bp.empty:
        return []
    recession_starts = [start for start, _end in _cached_recessions()]
    episodes = find_inversion_episodes(spread_bp, min_days=min_days)
    return add_lead_times(episodes, recession_starts, spread_bp.index[-1], max_lead)


def _show_kpis(yields: pd.DataFrame, spread_bp: pd.Series, spread_name: str,
               episodes: list[dict], min_days: int, max_lead: int) -> None:
    """Three headline numbers: the spread now, the 10Y yield now, and the average lead to recession."""
    latest_date = yields.index.max()
    year_ago = snap_to_previous_date(yields.index, latest_date - pd.DateOffset(years=1))
    col_spread, col_yield, col_lead = st.columns(3)

    long, short = SPREADS[spread_name]
    if spread_bp.empty:
        col_spread.metric(f"{spread_name} spread", "n/a")
    else:
        spread_delta = None
        if year_ago is not None and year_ago in spread_bp.index:
            spread_delta = f"{spread_bp.iloc[-1] - spread_bp[year_ago]:+.0f} bp vs a year earlier"
        col_spread.metric(
            f"{spread_name} spread", format_bp(spread_bp.iloc[-1]), spread_delta, delta_color="off",
            help=f"{long} yield minus {short} yield on {format_date(spread_bp.index[-1])}. "
                 "Below zero = inverted curve.",
        )

    ten_year = yields["10Y"].dropna()
    if ten_year.empty:
        col_yield.metric("10-year yield", "n/a")
    else:
        yield_delta = None
        if year_ago is not None and year_ago in ten_year.index:
            yield_delta = f"{(ten_year.iloc[-1] - ten_year[year_ago]) * 100:+.0f} bp vs a year earlier"
        col_yield.metric(
            "10-year yield", f"{ten_year.iloc[-1]:.2f}%", yield_delta, delta_color="off",
            help=f"US 10-year Treasury yield (FRED DGS10) on {format_date(ten_year.index[-1])}.",
        )

    summary = summarise_lead_times(episodes)
    first_year = spread_bp.index[0].year if not spread_bp.empty else int(START_DATE[:4])
    lead_help = lead_kpi_help(summary, spread_name, min_days, max_lead, first_year)
    if summary["mean"] is None:
        col_lead.metric("Avg. lead to recession", "n/a", help=lead_help)
    else:
        col_lead.metric("Avg. lead to recession", f"{summary['mean']:.0f} months", help=lead_help)


def _data_sources_markdown() -> str:
    """The data-source paragraph shared by both views' method notes."""
    links = []
    for series_id in MATURITIES:
        links.append(f"[{series_id}](https://fred.stlouisfed.org/series/{series_id})")
    return (
        "**Data.** Daily US Treasury constant-maturity yields from FRED, "
        f"{', '.join(links)}, since {START_DATE[:4]}; recession months from "
        "[USREC](https://fred.stlouisfed.org/series/USREC) (NBER dates)."
    )


def _render_snapshots(yields: pd.DataFrame, spread_bp: pd.Series, spread_name: str) -> None:
    """Controls, chart, insight and details for the 'Curve snapshots' view."""
    first_date = yields.index.min().date()
    last_date = yields.index.max().date()
    moments = st.multiselect(
        f"Compare the curve on (up to {MAX_MOMENTS})", NAMED_MOMENTS,
        max_selections=MAX_MOMENTS, key=f"{SLUG}_moments",
        help=f"'{DEEPEST}' is the day the {spread_name} spread was lowest. Dates without data "
             "(weekends, holidays) use the trading day before.",
    )
    col_toggle, col_date = st.columns([1, 1.3], vertical_alignment="bottom")
    use_custom = col_toggle.toggle("Add a custom date", key=f"{SLUG}_use_custom")
    custom_date = col_date.date_input(
        "Custom date", min_value=first_date, max_value=last_date,
        disabled=not use_custom, key=f"{SLUG}_custom_date", format="DD/MM/YYYY",
    )

    chosen = list(moments)
    if use_custom and custom_date is not None:
        chosen.append(CUSTOM)
    dates = moment_dates(yields, chosen, spread_bp, custom_date if use_custom else None)
    dates, same_day = drop_same_day_moments(dates)
    if same_day:
        st.caption(f"{' and '.join(same_day)}: same trading day as another curve, so it is drawn once.")

    # The episode rule is set in the Spread history view; use the same values here
    # so the lead-time KPI is identical in both views.
    min_days = st.session_state[f"{SLUG}_min_days"]
    max_lead = st.session_state[f"{SLUG}_max_lead"]
    episodes = _episodes_with_outcomes(spread_bp, min_days, max_lead)
    _show_kpis(yields, spread_bp, spread_name, episodes, min_days, max_lead)

    if not dates:
        st.info("Pick at least one moment (or switch on a custom date) to draw the yield curve.")
        return

    curves = {}
    for name, date in dates.items():
        curves[name] = yields.loc[date, MATURITY_LABELS]
    ui.show_chart(build_snapshot_chart(curves, dates), key=f"{SLUG}_chart")
    ui.insight(snapshot_insight(curves, dates, spread_name, episodes, max_lead))

    table = pd.DataFrame(curves)
    table.columns = [f"{name} ({date:%Y-%m-%d})" for name, date in dates.items()]
    table.index.name = "Maturity"
    method = (
        _data_sources_markdown() + "\n\n"
        "**Chart.** Each line is the whole yield curve on one day: the annual yield on US government "
        "bonds for each time to maturity. A normal curve slopes up (lenders want more for lending "
        "longer); an inverted curve (short yields above long yields) means the market expects rates "
        "to be cut, usually because it expects the economy to weaken.\n\n"
        "**Dates.** If a chosen day has no data (weekend or holiday) the nearest earlier trading day "
        f"is used, and the legend shows that date. '{DEEPEST}' is the day the {spread_name} spread was "
        "at its minimum. The 20-year bond wasn't issued between 1987 and Oct 1993, so curves before "
        "then skip the 20Y point.\n\n"
        f"**Shape words.** Inverted = {spread_name} spread below 0; flat = 0 to {FLAT_LIMIT_BP} bp; "
        "upward-sloping = above that. Whether a recession followed an inversion uses the episode rule "
        f"of the Spread history view ({min_days}+ inverted days, recession within {max_lead} months)."
    )
    ui.details(method, table.round(2), key=SLUG, file_name=f"{SLUG}_snapshots.csv")


def _render_history(yields: pd.DataFrame, spread_bp: pd.Series, spread_name: str) -> None:
    """Controls, chart, insight and details for the 'Spread history' view."""
    col_period, col_toggle = st.columns([1.6, 1], vertical_alignment="bottom")
    period = col_period.select_slider("Period", options=list(PERIODS), key=f"{SLUG}_period")
    show_recessions = col_toggle.toggle("Shade recessions", key=f"{SLUG}_recessions",
                                        help="Grey bands = NBER US recessions.")
    col_days, col_lead = st.columns(2)
    min_days = col_days.slider(
        "Min. days inverted", min_value=1, max_value=60, key=f"{SLUG}_min_days",
        help=f"An inversion counts as an episode only if the spread is below zero on at least this many "
             f"trading days. Dips at most {MAX_GAP_DAYS} trading days (~3 months) apart are joined into one episode.",
    )
    max_lead = col_lead.slider(
        "Recession within (months)", min_value=6, max_value=48, step=6, key=f"{SLUG}_max_lead",
        help="An inversion counts as 'followed by a recession' only if one began within this many "
             "months of the inversion's start. Once that time has passed without one, it was a false alarm.",
    )

    episodes = _episodes_with_outcomes(spread_bp, min_days, max_lead)
    _show_kpis(yields, spread_bp, spread_name, episodes, min_days, max_lead)

    long, short = SPREADS[spread_name]
    frame = pd.DataFrame({
        f"{long} yield (%)": yields[long],
        f"{short} yield (%)": yields[short],
        f"{spread_name} spread (bp)": spread_bp,
    }).dropna()
    years = PERIODS[period]
    frame = last_n_years(frame, years)
    if frame.empty:
        st.info("There is no data in this period. Try a longer one.")
        return
    weekly = years is None or years > WEEKLY_AFTER_YEARS
    if weekly:
        frame = to_weekly(frame)
    plotted_spread = frame[f"{spread_name} spread (bp)"]

    ui.show_chart(build_spread_chart(plotted_spread, spread_name, _cached_recessions(), show_recessions),
                  key=f"{SLUG}_chart")
    ui.insight(history_insight(spread_bp, episodes, spread_name, min_days, max_lead))

    resolution = "weekly (last trading day of each week)" if weekly else "daily"
    method = (
        _data_sources_markdown() + "\n\n"
        f"**Spread.** {long} yield minus {short} yield, in basis points (100 bp = 1 percentage point). "
        "Blue = normal curve; red = inverted (short-term rates above long-term rates). "
        f"This chart shows {resolution} values; the episode rule always uses daily data.\n\n"
        "**Inversion episodes.** A trading day is inverted when the spread is below 0. Runs of "
        f"inverted days separated by at most {MAX_GAP_DAYS} trading days (~3 months) are merged, and an "
        f"episode is kept only if it has at least {plural(min_days, 'inverted day')}.\n\n"
        "**Lead time.** The number of calendar months from the month of the episode's first inverted day "
        "to the month the next NBER recession began (e.g. Mar -> Aug = 5). An episode counts as followed "
        f"by a recession only if that is at most {max_lead} months; once {max_lead} months have passed "
        "without one it was a false alarm, and before that it is too soon to judge. When two episodes "
        "come before the same recession, the average lead time uses only the first, so each recession "
        f"counts once. The data start in January {START_DATE[:4]}, so inversions before then are not "
        "counted.\n\n"
        + episodes_markdown(episodes, max_lead)
    )
    table = frame.copy()
    table[f"{spread_name} spread (bp)"] = table[f"{spread_name} spread (bp)"].round(0)
    table.index = table.index.strftime("%Y-%m-%d")  # show dates without a 00:00:00 time
    table.index.name = "Date"
    ui.details(method, table, key=SLUG, file_name=f"{SLUG}_spread_history.csv")


def render() -> None:
    """Draw the whole panel inside the current Streamlit container."""
    ui.panel_header(NUMBER, TITLE, OWNER, QUESTION)
    try:
        yields = _cached_yields()
        _cached_recessions()
    except (DataUnavailableError, OSError, ValueError, KeyError) as exc:
        ui.no_data_message("Treasury yield", exc)
        return

    _remember_control_values(yields)
    col_view, col_spread = st.columns([1.3, 1])
    view = col_view.radio("View", [SNAPSHOT_VIEW, HISTORY_VIEW], horizontal=True, key=f"{SLUG}_view")
    spread_name = col_spread.radio(
        "Spread", list(SPREADS), horizontal=True, key=f"{SLUG}_spread",
        help="10Y - 2Y is the most quoted recession signal; 10Y - 3M is the one the New York Fed uses.",
    )
    long, short = SPREADS[spread_name]
    spread_bp = compute_spread_bp(yields, long, short)

    if view == SNAPSHOT_VIEW:
        _render_snapshots(yields, spread_bp, spread_name)
    else:
        _render_history(yields, spread_bp, spread_name)
