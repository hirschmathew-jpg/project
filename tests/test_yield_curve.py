"""Tests for Panel 2 - The Yield Curve (no internet needed).

Unit tests use small made-up data where the right answer can be worked out
by hand. The AppTest tests render the panel from the CSV snapshot in data/
and change every control to check nothing crashes.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import theme
from dashboard.data import snapshot_path
from dashboard.panels import yield_curve as yc


# ---------------------------------------------------------------------------
# Helpers that build small synthetic data sets
# ---------------------------------------------------------------------------

def make_spread(values, start="2024-01-01") -> pd.Series:
    """A spread series (bp) on consecutive business days."""
    dates = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=dates, dtype=float)


def make_yields(dates, ten_year, two_year, three_month) -> pd.DataFrame:
    """A yields DataFrame with every maturity column; only 3M, 2Y and 10Y matter."""
    frame = pd.DataFrame(index=pd.DatetimeIndex(dates))
    for label in yc.MATURITY_LABELS:
        frame[label] = 3.0
    frame["10Y"] = ten_year
    frame["2Y"] = two_year
    frame["3M"] = three_month
    return frame


# ---------------------------------------------------------------------------
# Panel contract
# ---------------------------------------------------------------------------

def test_panel_constants_match_the_contract():
    assert yc.NUMBER == 2
    assert yc.SLUG == "yield_curve"
    assert yc.TITLE == "The Yield Curve"
    assert isinstance(yc.OWNER, str) and yc.OWNER.strip()  # the student puts their name here
    assert callable(yc.render) and callable(yc.refresh_data)


def test_snapshot_csv_is_shipped_and_recent():
    """The repo includes the data so the dashboard works offline."""
    path = snapshot_path(yc.DATASET_NAME)
    assert path.exists()
    assert path.stat().st_size < 1_500_000
    raw = yc.load_data()  # reads the CSV, no download
    assert list(raw.columns) == list(yc.MATURITIES)
    assert raw.index.min() < pd.Timestamp("1990-02-01")
    assert raw.index.max() > pd.Timestamp("2026-01-01")
    yields = yc.tidy_yields(raw)
    assert list(yields.columns) == yc.MATURITY_LABELS


# ---------------------------------------------------------------------------
# Spread and maturity ordering
# ---------------------------------------------------------------------------

def test_compute_spread_in_basis_points():
    dates = pd.bdate_range("2024-01-01", periods=3)
    yields = make_yields(dates, ten_year=[4.00, 3.50, np.nan], two_year=[3.50, 4.25, 4.0],
                         three_month=[5.0, 5.0, 5.0])
    spread = yc.compute_spread_bp(yields, "10Y", "2Y")
    # 4.00 - 3.50 = 0.50 points = +50 bp; 3.50 - 4.25 = -75 bp; the NaN day is dropped
    assert spread.tolist() == pytest.approx([50.0, -75.0])
    assert len(spread) == 2

    spread_3m = yc.compute_spread_bp(yields, "10Y", "3M")
    assert spread_3m.tolist() == pytest.approx([-100.0, -150.0])


def test_maturities_are_ordered_shortest_first():
    assert yc.MATURITY_LABELS == ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    # tidy_yields puts the columns in maturity order whatever order they arrive in
    raw = pd.DataFrame([[1.0] * 10], columns=list(reversed(list(yc.MATURITIES))),
                       index=pd.DatetimeIndex(["2024-01-02"]))
    assert list(yc.tidy_yields(raw).columns) == yc.MATURITY_LABELS


def test_snapshot_chart_uses_maturity_order_and_skips_missing_points():
    date = pd.Timestamp("1991-06-03")
    curve = pd.Series(np.arange(10, dtype=float), index=yc.MATURITY_LABELS)
    curve["20Y"] = np.nan  # the 20-year bond wasn't issued in 1991
    fig = yc.build_snapshot_chart({yc.CUSTOM: curve}, {yc.CUSTOM: date})
    assert list(fig.layout.xaxis.categoryarray) == yc.MATURITY_LABELS
    assert fig.layout.xaxis.type == "category"
    trace = fig.data[0]
    assert "20Y" not in list(trace.x)
    assert list(trace.x) == [m for m in yc.MATURITY_LABELS if m != "20Y"]
    assert "3 Jun 1991" in trace.name


def test_moment_colours_are_fixed_theme_colours():
    assert yc.MOMENT_COLOURS[yc.LATEST] == theme.SERIES[0]
    assert yc.MOMENT_COLOURS[yc.YEAR_AGO] == theme.SERIES[1]
    assert len(set(yc.MOMENT_COLOURS.values())) == len(yc.MOMENT_NAMES)
    # A curve keeps its colour when other curves are hidden
    curve = pd.Series(1.0, index=yc.MATURITY_LABELS)
    date = pd.Timestamp("2024-01-02")
    fig = yc.build_snapshot_chart({yc.DEEPEST: curve}, {yc.DEEPEST: date})
    assert fig.data[0].line.color == yc.MOMENT_COLOURS[yc.DEEPEST]


# ---------------------------------------------------------------------------
# Date snapping
# ---------------------------------------------------------------------------

def test_snap_to_previous_date():
    dates = pd.DatetimeIndex(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    # Exact match
    assert yc.snap_to_previous_date(dates, "2024-01-04") == pd.Timestamp("2024-01-04")
    # Sunday 7 Jan -> Friday 5 Jan
    assert yc.snap_to_previous_date(dates, dt.date(2024, 1, 7)) == pd.Timestamp("2024-01-05")
    # After the last date -> last date
    assert yc.snap_to_previous_date(dates, "2030-01-01") == pd.Timestamp("2024-01-08")
    # Before the first date, or no date at all -> None
    assert yc.snap_to_previous_date(dates, "2024-01-02") is None
    assert yc.snap_to_previous_date(dates, None) is None


def test_moment_dates_snaps_and_computes_named_moments():
    dates = pd.bdate_range("2019-01-01", "2020-03-31")
    ten = np.full(len(dates), 2.0)
    two = np.full(len(dates), 1.5)
    two[100] = 2.6  # one day where the 2Y is far above the 10Y (spread -60 bp)
    yields = make_yields(dates, ten, two, np.full(len(dates), 1.0))
    spread = yc.compute_spread_bp(yields, "10Y", "2Y")

    chosen = yc.moment_dates(
        yields,
        [yc.CUSTOM, yc.DEEPEST, yc.LATEST, yc.YEAR_AGO, "Pre-COVID (Jan 2020)", "Dot-com peak (Mar 2000)"],
        spread,
        custom_date=dt.date(2019, 6, 16),  # a Sunday
    )
    assert chosen[yc.LATEST] == pd.Timestamp("2020-03-31")
    assert chosen[yc.YEAR_AGO] == pd.Timestamp("2019-03-29")  # 31 Mar 2019 was a Sunday
    assert chosen[yc.DEEPEST] == dates[100]
    assert chosen["Pre-COVID (Jan 2020)"] == pd.Timestamp("2020-01-31")
    assert chosen[yc.CUSTOM] == pd.Timestamp("2019-06-14")
    # March 2000 is before this data starts, so that moment is left out
    assert "Dot-com peak (Mar 2000)" not in chosen
    # Results come back in the fixed MOMENT_NAMES order, not the order they were picked
    assert list(chosen) == [yc.LATEST, yc.YEAR_AGO, yc.DEEPEST, "Pre-COVID (Jan 2020)", yc.CUSTOM]


def test_describe_curve():
    assert yc.describe_curve(-1) == "inverted"
    assert yc.describe_curve(0) == "flat"
    assert yc.describe_curve(24.9) == "flat"
    assert yc.describe_curve(25) == "upward-sloping"


# ---------------------------------------------------------------------------
# Inversion episodes
# ---------------------------------------------------------------------------

def test_find_inversion_runs():
    spread = make_spread([5, -1, -2, 3, -4, 6, -1])
    # runs at positions 1-2, 4-4 and 6-6 (the last one runs to the end)
    assert yc.find_inversion_runs(spread) == [(1, 2), (4, 4), (6, 6)]
    assert yc.find_inversion_runs(make_spread([1, 2, 3])) == []
    assert yc.find_inversion_runs(make_spread([-1, -1])) == [(0, 1)]


def test_merge_close_runs():
    runs = [(0, 2), (5, 6), (20, 25)]
    # gaps: 5 - 2 - 1 = 2 days, then 20 - 6 - 1 = 13 days
    assert yc.merge_close_runs(runs, max_gap=2) == [(0, 6), (20, 25)]
    assert yc.merge_close_runs(runs, max_gap=1) == runs
    assert yc.merge_close_runs(runs, max_gap=13) == [(0, 25)]
    assert yc.merge_close_runs([], max_gap=5) == []


def test_episode_min_length_filter():
    # A 3-day dip and a 5-day dip, far apart
    values = [10] * 5 + [-5] * 3 + [10] * 30 + [-8, -9, -12, -9, -8] + [10] * 5
    spread = make_spread(values)
    episodes = yc.find_inversion_episodes(spread, min_days=4, max_gap=10)
    assert len(episodes) == 1  # the 3-day dip is too short
    only = episodes[0]
    assert only["start"] == spread.index[38]
    assert only["end"] == spread.index[42]
    assert only["inverted_days"] == 5
    assert only["deepest_bp"] == -12
    assert only["deepest_date"] == spread.index[40]
    assert only["ongoing"] is False

    # With min_days=3 both dips count
    assert len(yc.find_inversion_episodes(spread, min_days=3, max_gap=10)) == 2


def test_episode_gap_merging_counts_only_inverted_days():
    # -,-,-, +,+, -,-,-  : two 3-day dips with a 2-day break
    values = [5] * 3 + [-1, -2, -3] + [4, 4] + [-3, -2, -1] + [5] * 3
    spread = make_spread(values)
    merged = yc.find_inversion_episodes(spread, min_days=5, max_gap=2)
    assert len(merged) == 1
    assert merged[0]["start"] == spread.index[3]
    assert merged[0]["end"] == spread.index[10]
    assert merged[0]["inverted_days"] == 6  # the 2 positive days in between don't count
    # With a smaller allowed gap they stay separate, and each is too short for min_days=5
    assert yc.find_inversion_episodes(spread, min_days=5, max_gap=1) == []


def test_episode_running_to_the_end_of_the_data():
    spread = make_spread([3, 2, 1, -1, -2, -3, -4])
    episodes = yc.find_inversion_episodes(spread, min_days=2, max_gap=5)
    assert len(episodes) == 1
    assert episodes[0]["ongoing"] is True
    assert episodes[0]["end"] == spread.index[-1]
    assert episodes[0]["inverted_days"] == 4


def test_episodes_ignore_missing_values():
    spread = make_spread([1, -1, np.nan, -1, -1, 2])
    episodes = yc.find_inversion_episodes(spread, min_days=3, max_gap=0)
    # the NaN day is dropped, so the three inverted days are consecutive
    assert len(episodes) == 1
    assert episodes[0]["inverted_days"] == 3


# ---------------------------------------------------------------------------
# Lead times to recession
# ---------------------------------------------------------------------------

RECESSION_STARTS = [pd.Timestamp("2001-04-01"), pd.Timestamp("2008-01-01")]
DATA_END = pd.Timestamp("2026-10-13")


def test_months_between_counts_calendar_months():
    # Feb 2000 -> Apr 2001 = 14 calendar months
    assert yc.months_between(pd.Timestamp("2000-02-02"), pd.Timestamp("2001-04-01")) == 14
    # Only the month counts, not the day: 20 Dec -> 1 Jan is 1 (Dec -> Jan), 15 Mar -> 1 Aug is 5
    assert yc.months_between(pd.Timestamp("2007-12-20"), pd.Timestamp("2008-01-01")) == 1
    assert yc.months_between(pd.Timestamp("2021-03-15"), pd.Timestamp("2021-08-01")) == 5
    assert yc.months_between(pd.Timestamp("2020-05-15"), pd.Timestamp("2020-05-31")) == 0


def test_next_recession_start():
    assert yc.next_recession_start(pd.Timestamp("2000-02-02"), RECESSION_STARTS) == RECESSION_STARTS[0]
    # An inversion after 2001 points to the 2008 recession, not the earlier one
    assert yc.next_recession_start(pd.Timestamp("2006-01-15"), RECESSION_STARTS) == RECESSION_STARTS[1]
    # No recession after it (yet) -> None
    assert yc.next_recession_start(pd.Timestamp("2022-07-06"), RECESSION_STARTS) is None
    assert yc.next_recession_start(pd.Timestamp("2000-01-01"), []) is None


def test_episode_outcome_time_limit_boundary():
    # A lead of exactly the limit counts; one month more does not
    assert yc.episode_outcome(30, months_since_start=60, max_lead=30) == yc.FOLLOWED
    assert yc.episode_outcome(31, months_since_start=60, max_lead=30) == yc.NO_RECESSION
    assert yc.episode_outcome(5, months_since_start=60, max_lead=30) == yc.FOLLOWED


def test_episode_outcome_without_a_recession():
    # No recession and the limit has passed: a false alarm
    assert yc.episode_outcome(None, months_since_start=31, max_lead=30) == yc.NO_RECESSION
    # No recession yet, but the limit hasn't passed: too soon to judge
    assert yc.episode_outcome(None, months_since_start=30, max_lead=30) == yc.TOO_SOON
    assert yc.episode_outcome(None, months_since_start=4, max_lead=30) == yc.TOO_SOON


def make_episodes(*starts) -> list[dict]:
    """Episode dicts with just a start date (all add_lead_times needs)."""
    return [{"start": pd.Timestamp(start)} for start in starts]


def test_add_lead_times_and_summary():
    # 2000 -> Apr 2001 (14 months); 2006 -> Jan 2008 (24); 2022 -> none since (54 months on)
    episodes = yc.add_lead_times(make_episodes("2000-02-02", "2006-01-15", "2022-04-01"),
                                 RECESSION_STARTS, DATA_END, max_lead=30)
    assert [e["lead_months"] for e in episodes] == [14, 24, None]
    assert [e["outcome"] for e in episodes] == [yc.FOLLOWED, yc.FOLLOWED, yc.NO_RECESSION]
    assert episodes[2]["months_since_start"] == 54
    assert episodes[0]["recession_start"] == RECESSION_STARTS[0]

    summary = yc.summarise_lead_times(episodes)
    assert summary["episodes"] == 3
    assert summary["followed"] == 2
    assert summary["no_recession"] == 1
    assert summary["too_soon"] == 0
    assert summary["recessions"] == 2
    assert summary["mean"] == pytest.approx(19.0)  # (14 + 24) / 2
    assert summary["min"] == 14
    assert summary["max"] == 24


def test_recession_too_late_is_a_false_alarm():
    # Like the May 1998 inversion: the next recession (Apr 2001) began 35 months later
    episodes = yc.add_lead_times(make_episodes("1998-05-26", "2000-02-02"), RECESSION_STARTS, DATA_END, max_lead=30)
    assert episodes[0]["lead_months"] == 35
    assert episodes[0]["outcome"] == yc.NO_RECESSION
    assert episodes[1]["outcome"] == yc.FOLLOWED
    assert episodes[1]["first_warning"] is True  # 1998 didn't count, so 2000 is the first warning
    summary = yc.summarise_lead_times(episodes)
    assert summary["recessions"] == 1 and summary["mean"] == 14

    # With a longer limit the 1998 inversion counts, and it becomes the first warning
    episodes = yc.add_lead_times(make_episodes("1998-05-26", "2000-02-02"), RECESSION_STARTS, DATA_END, max_lead=36)
    assert [e["first_warning"] for e in episodes] == [True, False]
    assert yc.summarise_lead_times(episodes)["mean"] == 35


def test_two_inversions_before_one_recession_count_it_once():
    # Like the 10Y-3M in 2019 and Jan 2020, both before the COVID recession (Mar 2020)
    covid = [pd.Timestamp("2020-03-01")]
    episodes = yc.add_lead_times(make_episodes("2019-03-22", "2020-01-31"), covid, DATA_END)
    assert [e["lead_months"] for e in episodes] == [12, 2]
    assert [e["outcome"] for e in episodes] == [yc.FOLLOWED, yc.FOLLOWED]
    assert [e["first_warning"] for e in episodes] == [True, False]
    summary = yc.summarise_lead_times(episodes)
    assert summary["followed"] == 2
    assert summary["recessions"] == 1
    assert summary["mean"] == 12  # from the first warning only, not (12 + 2) / 2


def test_inversion_still_inside_the_time_limit_is_too_soon():
    episodes = yc.add_lead_times(make_episodes("2025-06-02"), RECESSION_STARTS, DATA_END, max_lead=30)
    assert episodes[0]["outcome"] == yc.TOO_SOON
    assert episodes[0]["months_since_start"] == 16
    summary = yc.summarise_lead_times(episodes)
    assert summary["too_soon"] == 1 and summary["mean"] is None


def test_recessions_after_the_last_day_of_data_are_ignored():
    # Looking at the data as it stood on 1 Mar 2007, the 2008 recession hadn't happened yet
    episodes = yc.add_lead_times(make_episodes("2006-07-17"), RECESSION_STARTS, pd.Timestamp("2007-03-01"))
    assert episodes[0]["lead_months"] is None
    assert episodes[0]["outcome"] == yc.TOO_SOON


def test_summary_when_no_recession_followed():
    episodes = yc.add_lead_times(make_episodes("2010-01-04"), RECESSION_STARTS, DATA_END)
    summary = yc.summarise_lead_times(episodes)
    assert summary["followed"] == 0
    assert summary["no_recession"] == 1
    assert summary["mean"] is None
    assert yc.summarise_lead_times([])["episodes"] == 0


# ---------------------------------------------------------------------------
# Insights (the text must describe "not yet" honestly)
# ---------------------------------------------------------------------------

def episodes_for(spread: pd.Series, recessions=RECESSION_STARTS, min_days=10, max_lead=30) -> list[dict]:
    """Find the episodes in ``spread`` and add their outcomes, as the panel does."""
    episodes = yc.find_inversion_episodes(spread, min_days=min_days)
    return yc.add_lead_times(episodes, recessions, spread.index[-1], max_lead)


def test_history_insight_inversion_not_followed_by_recession():
    # A 2022 inversion, data up to Nov 2024: no recession, and more than 6 months have passed
    spread = make_spread([50] * 5 + [-20] * 20 + [30] * 600, start="2022-07-01")
    episodes = episodes_for(spread, max_lead=6)
    text = yc.history_insight(spread, episodes, "10Y - 2Y", 10, max_lead=6)
    assert "+30 bp" in text
    assert "not inverted" in text
    assert "its last inversion, 8 Jul 2022 to 4 Aug 2022, was not followed by a recession within 6 months" in text
    assert "(28 months have passed since it began)" in text  # since it BEGAN, not since it ended
    assert "Since" not in text  # the only episode is already described


def test_history_insight_ongoing_inversion_and_track_record():
    old = make_spread([-10] * 15 + [20] * 100, start="2000-02-01")
    new = make_spread([20] * 5 + [-40] * 12, start="2022-07-01")
    spread = pd.concat([old, new])
    episodes = episodes_for(spread)
    text = yc.history_insight(spread, episodes, "10Y - 3M", 10)
    assert "the curve is inverted, in an episode that began on 8 Jul 2022 and has 12 inverted days so far" in text
    assert "too soon to judge" in text
    # The 2000 episode started Feb 2000 and the recession in Apr 2001: 14 months
    assert "Since 2000, 1 recession began within 30 months of an inversion, 14 months after the first warning" in text
    assert "range" not in text  # only one lead time, so no range
    assert "it is too soon to judge the one that began in Jul 2022" in text


def test_history_insight_ongoing_episode_made_of_several_dips():
    # 12 inverted days, a 20-day bounce, then 1 inverted day: the gap rule joins them into one episode
    spread = make_spread([30] * 5 + [-10] * 12 + [15] * 20 + [-3], start="2025-06-02")
    episodes = episodes_for(spread)
    assert len(episodes) == 1 and episodes[0]["ongoing"]
    text = yc.history_insight(spread, episodes, "10Y - 3M", 10)
    # It must not claim the curve has been inverted all the time since the episode began
    assert "inverted since" not in text
    assert "in an episode that began on 9 Jun 2025 and has 13 inverted days so far" in text
    assert "dips less than 3 months apart count as one episode" in text


def test_history_insight_track_record_lists_false_alarms():
    # 2000 (recession 14 months later), 2010 (no recession), 2015 (the last one, no recession)
    parts = [make_spread([-10] * 12 + [20] * 80, start=start) for start in ("2000-02-01", "2010-03-01", "2015-03-02")]
    later = make_spread([20] * 700, start="2016-01-01")  # data to 2018, so 30 months pass after 2015
    spread = pd.concat(parts + [later])
    episodes = episodes_for(spread, recessions=[pd.Timestamp("2001-04-01")])
    text = yc.history_insight(spread, episodes, "10Y - 2Y", 10)
    assert "its last inversion, 2 Mar 2015 to 17 Mar 2015, was not followed by a recession within 30 months" in text
    assert "Since 2000, 1 recession began within 30 months of an inversion, 14 months after the first warning" in text
    assert "2 inversions (Mar 2010 and Mar 2015) were not followed by one within 30 months" in text


def test_history_insight_averages_lead_times():
    parts = [make_spread([-10] * 12 + [20] * 80, start=start) for start in ("2000-02-01", "2006-01-02")]
    later = make_spread([20] * 600, start="2007-01-01")  # data past the Jan 2008 recession
    spread = pd.concat(parts + [later])
    episodes = episodes_for(spread)  # leads 14 and 24 months
    text = yc.history_insight(spread, episodes, "10Y - 2Y", 10)
    assert "its last inversion, 2 Jan 2006 to 17 Jan 2006, was followed by a recession 24 months after it began" in text
    assert "2 recessions began within 30 months of an inversion, on average 19 months after the first warning" in text
    assert "(range 14-24 months)" in text
    assert "not followed" not in text  # every inversion was followed by a recession


def test_history_insight_without_any_episode():
    spread = make_spread([40, 30, 20, -5])
    text = yc.history_insight(spread, [], "10Y - 2Y", 10)
    assert "-5 bp" in text
    assert "inverted, but so far too briefly to count under the 10-day rule" in text
    assert "has not been inverted for 10+ trading days" in text


def test_lead_kpi_help_grammar():
    one = {"recessions": 1, "no_recession": 1}
    text = yc.lead_kpi_help(one, "10Y - 2Y", 10, 30, 1990)
    assert "1 recession followed an inversion" in text
    assert "1 inversion was not followed by one within 30 months" in text
    several = {"recessions": 3, "no_recession": 2}
    text = yc.lead_kpi_help(several, "10Y - 2Y", 10, 30, 1990)
    assert "3 recessions followed" in text and "2 inversions were not followed" in text
    assert "10+ inverted days" in text and "Since 1990" in text


DEEPEST_DAY = pd.Timestamp("2023-07-03")


def inverted_episode() -> list[dict]:
    """One episode around DEEPEST_DAY that began in Apr 2022, with no recession since."""
    spread = make_spread([20] * 5 + [-50] * 400 + [20] * 600, start="2022-03-25")
    episodes = episodes_for(spread, recessions=[])
    assert episodes[0]["start"] <= DEEPEST_DAY <= episodes[0]["end"]
    return episodes


def test_snapshot_insight_mentions_the_most_inverted_curve_and_its_outcome():
    latest = pd.Series({"3M": 4.3, "2Y": 4.8, "10Y": 5.2})
    inverted = pd.Series({"3M": 5.4, "2Y": 4.9, "10Y": 3.8})
    dates = {yc.LATEST: pd.Timestamp("2026-10-13"), yc.DEEPEST: DEEPEST_DAY}
    text = yc.snapshot_insight({yc.LATEST: latest, yc.DEEPEST: inverted}, dates, "10Y - 2Y", inverted_episode())
    assert "upward-sloping" in text
    assert "40 bp above the 2Y" in text
    assert "most inverted on 3 Jul 2023 (Deepest inversion), at -110 bp" in text
    # The default view says honestly whether a recession followed
    assert "that inversion began on 1 Apr 2022 and was not followed by a recession within 30 months" in text


def test_snapshot_insight_when_the_first_curve_is_the_lowest():
    flat = pd.Series({"3M": 4.0, "2Y": 4.0, "10Y": 4.1})
    steep = pd.Series({"3M": 1.0, "2Y": 1.5, "10Y": 3.5})
    dates = {yc.LATEST: pd.Timestamp("2026-10-13"), "Financial crisis (Dec 2008)": pd.Timestamp("2008-12-31")}
    text = yc.snapshot_insight({yc.LATEST: flat, "Financial crisis (Dec 2008)": steep}, dates, "10Y - 2Y", [])
    assert "flat" in text
    assert "lowest spread of the dates shown" in text
    assert "steepest on 31 Dec 2008 (Financial crisis)" in text and "+200 bp" in text


def test_snapshot_insight_tie():
    # Two different days with the same spread: no "lowest ... steepest" contradiction
    curve_a = pd.Series({"3M": 4.0, "2Y": 4.0, "10Y": 4.42})
    curve_b = pd.Series({"3M": 3.0, "2Y": 3.5, "10Y": 3.92})
    dates = {yc.LATEST: pd.Timestamp("2026-10-13"), yc.YEAR_AGO: pd.Timestamp("2025-10-10")}
    text = yc.snapshot_insight({yc.LATEST: curve_a, yc.YEAR_AGO: curve_b}, dates, "10Y - 2Y", [])
    assert "All the dates shown have the same 10Y - 2Y spread." in text
    assert "steepest" not in text and "lowest" not in text


def test_snapshot_insight_when_every_curve_is_inverted():
    deepest = pd.Series({"3M": 5.4, "2Y": 4.9, "10Y": 3.82})
    milder = pd.Series({"3M": 5.0, "2Y": 4.9, "10Y": 4.01})
    dates = {yc.DEEPEST: DEEPEST_DAY, yc.CUSTOM: pd.Timestamp("2023-03-03")}
    text = yc.snapshot_insight({yc.DEEPEST: deepest, yc.CUSTOM: milder}, dates, "10Y - 2Y", inverted_episode())
    # An inverted curve is never called "steepest"
    assert "steepest" not in text
    assert "That is the most inverted of the dates shown; it was least inverted on 3 Mar 2023 (Custom date), at -89 bp." in text
    # The outcome is attached to the first (most inverted) curve
    assert "108 bp below the 2Y; that inversion began on 1 Apr 2022" in text


def test_snapshot_insight_single_curve_and_missing_yields():
    curve = pd.Series({"3M": 1.0, "2Y": 1.0, "10Y": 1.1})
    text = yc.snapshot_insight({yc.LATEST: curve}, {yc.LATEST: pd.Timestamp("2020-01-31")}, "10Y - 2Y", [])
    assert "flat" in text and "10 bp above" in text
    missing = pd.Series({"3M": np.nan, "2Y": 1.0, "10Y": 1.1})
    text = yc.snapshot_insight({yc.LATEST: missing}, {yc.LATEST: pd.Timestamp("2020-01-31")}, "10Y - 3M", [])
    assert "missing" in text


def test_drop_same_day_moments():
    day = pd.Timestamp("2026-10-13")
    dates = {yc.LATEST: day, yc.DEEPEST: pd.Timestamp("2023-07-03"), yc.CUSTOM: day}
    kept, dropped = yc.drop_same_day_moments(dates)
    assert kept == {yc.LATEST: day, yc.DEEPEST: pd.Timestamp("2023-07-03")}
    assert dropped == [yc.CUSTOM]
    assert yc.drop_same_day_moments({}) == ({}, [])


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def test_split_by_sign():
    spread = make_spread([10, 5, -3, -6, -2, 4])
    normal, inverted = yc.split_by_sign(spread)
    # Normal part: own values, 0 on an inverted day next to a normal day, blank otherwise
    assert normal.tolist()[:3] == [10, 5, 0]
    assert np.isnan(normal.iloc[3])  # -6 has inverted days on both sides
    assert normal.tolist()[4:] == [0, 4]
    # Inverted part: blank far from inversions, 0 next to them, own values when inverted
    assert np.isnan(inverted.iloc[0])
    assert inverted.tolist()[1:] == [0, -3, -6, -2, 0]


def test_spread_chart_has_one_axis_and_theme_colours():
    spread = make_spread(list(range(-20, 20)), start="2020-01-01")
    recessions = [(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01"))]
    fig = yc.build_spread_chart(spread, "10Y - 2Y", recessions, show_recessions=True)
    assert "yaxis2" not in fig.layout  # never a second y-axis
    assert fig.data[0].line.color == theme.POSITIVE
    assert fig.data[1].line.color == theme.NEGATIVE
    assert fig.data[0].line.width == 2 and fig.data[1].line.width == 2
    names = [trace.name for trace in fig.data]
    assert "US recession" in names
    assert len(fig.layout.shapes) >= 2  # recession band + zero line

    # A window with no recession in it gets no recession legend entry
    later = make_spread([10, 20, 30], start="2024-01-01")
    fig = yc.build_spread_chart(later, "10Y - 3M", recessions, show_recessions=True)
    assert "US recession" not in [trace.name for trace in fig.data]


def test_spread_chart_end_label_does_not_stretch_the_date_axis():
    spread = make_spread([10, 20, 42], start="2024-01-01")
    fig = yc.build_spread_chart(spread, "10Y - 2Y", [], show_recessions=True)
    label = fig.layout.annotations[0]
    assert label.text == "+42 bp"
    # In "paper" units (not dates), so Plotly doesn't widen the x-axis to fit the label
    assert label.xref == "paper" and label.x == 1
    assert label.y == 42


def test_spread_chart_long_windows_get_a_tick_every_five_years():
    long_spread = make_spread(np.ones(3000), start="1990-01-01")  # about 11.5 years
    fig = yc.build_spread_chart(long_spread, "10Y - 2Y", [], show_recessions=False)
    assert fig.layout.xaxis.dtick == "M60"
    short_spread = make_spread(np.ones(500), start="2020-01-01")  # about 2 years: Plotly picks the ticks
    fig = yc.build_spread_chart(short_spread, "10Y - 2Y", [], show_recessions=False)
    assert fig.layout.xaxis.dtick is None


def test_snapshot_chart_makes_room_for_more_legend_rows():
    curve = pd.Series(1.0, index=yc.MATURITY_LABELS)
    date = pd.Timestamp("2024-01-02")
    two = yc.build_snapshot_chart({yc.LATEST: curve, yc.YEAR_AGO: curve}, {yc.LATEST: date, yc.YEAR_AGO: date})
    names = yc.MOMENT_NAMES[:5]
    five = yc.build_snapshot_chart({name: curve for name in names}, {name: date for name in names})
    assert two.layout.margin.t == yc.TOP_MARGIN
    assert five.layout.margin.t == yc.TOP_MARGIN + 2 * yc.LEGEND_ROW_HEIGHT  # 3 legend rows
    assert five.layout.yaxis.tickformat == ".1f"  # 3.5%, 4.0%, ... the same decimals on every tick


def test_recessions_in_window():
    recessions = [(pd.Timestamp("2001-04-01"), pd.Timestamp("2001-12-01")),
                  (pd.Timestamp("2008-01-01"), pd.Timestamp("2009-07-01"))]
    found = yc.recessions_in_window(recessions, pd.Timestamp("2009-01-01"), pd.Timestamp("2015-01-01"))
    assert found == [recessions[1]]
    assert yc.recessions_in_window(recessions, pd.Timestamp("2015-01-01"), pd.Timestamp("2020-01-01")) == []


def test_last_n_years_and_weekly():
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    frame = pd.DataFrame({"spread": np.arange(len(dates), dtype=float)}, index=dates)
    window = yc.last_n_years(frame, 2)
    assert window.index.min() >= pd.Timestamp("2022-12-31")
    assert window.index.max() == dates.max()
    assert len(yc.last_n_years(frame, None)) == len(frame)
    weekly = yc.to_weekly(window)
    assert len(weekly) < len(window) / 4
    # Each weekly row is the last trading day of that week, with its real date
    assert weekly["spread"].iloc[-1] == frame["spread"].iloc[-1]
    assert weekly.index[-1] == dates.max()  # Tue 31 Dec 2024, not Fri 3 Jan 2025
    assert set(weekly.index) <= set(window.index)


def test_hex_to_rgba():
    assert yc.hex_to_rgba("#2a78d6", 0.25) == "rgba(42, 120, 214, 0.25)"


def test_episodes_markdown():
    episodes = [
        {"start": pd.Timestamp("1998-05-26"), "end": pd.Timestamp("1998-07-27"), "inverted_days": 27,
         "deepest_bp": -7.0, "ongoing": False},
        {"start": pd.Timestamp("2000-02-02"), "end": pd.Timestamp("2000-12-28"), "inverted_days": 227,
         "deepest_bp": -52.0, "ongoing": False},
        {"start": pd.Timestamp("2000-12-01"), "end": pd.Timestamp("2000-12-20"), "inverted_days": 12,
         "deepest_bp": -5.0, "ongoing": False},
        {"start": pd.Timestamp("2025-06-02"), "end": pd.Timestamp("2026-10-13"), "inverted_days": 300,
         "deepest_bp": -108.0, "ongoing": True},
    ]
    yc.add_lead_times(episodes, RECESSION_STARTS, DATA_END, max_lead=30)
    table = yc.episodes_markdown(episodes, 30)
    assert "Recession within 30 months?" in table
    assert "| 26 May 1998 | 27 Jul 1998 | 27 | -7 bp | No (the next began 35 months later) |" in table
    assert "| 2 Feb 2000 | 28 Dec 2000 | 227 | -52 bp | Yes, 14 months later |" in table
    assert "Yes, 4 months later (same recession as the inversion before)" in table
    assert "still inverted" in table and "Too soon to judge" in table
    assert "No inversion episodes" in yc.episodes_markdown([], 30)


# ---------------------------------------------------------------------------
# The real data: sanity checks of the finance (reads the shipped CSV)
# ---------------------------------------------------------------------------

def test_real_data_finds_known_inversions():
    yields = yc.tidy_yields(yc.load_data())
    spread = yc.compute_spread_bp(yields, "10Y", "2Y")
    starts = [episode["start"].year for episode in yc.find_inversion_episodes(spread)]
    # The 10Y-2Y curve inverted before the 2001 and 2008 recessions and again in 2022
    assert 2000 in starts
    assert 2005 in starts or 2006 in starts
    assert 2022 in starts


def test_real_data_track_record_before_2020():
    """Hand-checkable history (it can't change when the data is refreshed)."""
    yields = yc.tidy_yields(yc.load_data())
    spread = yc.compute_spread_bp(yields, "10Y", "2Y")
    recession_starts = [pd.Timestamp(d) for d in ("1990-08-01", "2001-04-01", "2008-01-01", "2020-03-01")]
    episodes = yc.find_inversion_episodes(spread, min_days=10)
    episodes = yc.add_lead_times(episodes, recession_starts, spread.index[-1], max_lead=30)
    before_2020 = {}
    for episode in episodes:
        if episode["start"].year < 2020:
            before_2020[episode["start"].year] = (episode["lead_months"], episode["outcome"])
    assert before_2020 == {
        1990: (5, yc.FOLLOWED),       # Mar 1990 -> Aug 1990 recession
        1998: (35, yc.NO_RECESSION),  # the next recession came 35 months later: too late to count
        2000: (14, yc.FOLLOWED),      # Feb 2000 -> Apr 2001
        2005: (25, yc.FOLLOWED),      # Dec 2005 -> Jan 2008
    }


# ---------------------------------------------------------------------------
# AppTest: render the panel and change every control
# ---------------------------------------------------------------------------

def _panel_app():
    from dashboard.panels import yield_curve

    yield_curve.render()


def _check(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.fixture
def app():
    at = AppTest.from_function(_panel_app, default_timeout=120)
    at.run()
    _check(at)
    return at


@pytest.fixture(scope="module")
def real_yields() -> pd.DataFrame:
    return yc.tidy_yields(yc.load_data())


def expected_lead_kpi(yields: pd.DataFrame, spread_name: str, min_days: int, max_lead: int) -> str:
    """The 'Avg. lead to recession' value worked out from the shipped CSVs."""
    from dashboard.data import load_recessions, recession_periods

    long, short = yc.SPREADS[spread_name]
    spread = yc.compute_spread_bp(yields, long, short)
    recession_starts = [start for start, _end in recession_periods(load_recessions())]
    episodes = yc.find_inversion_episodes(spread, min_days=min_days)
    episodes = yc.add_lead_times(episodes, recession_starts, spread.index[-1], max_lead)
    summary = yc.summarise_lead_times(episodes)
    if summary["mean"] is None:
        return "n/a"
    return f"{summary['mean']:.0f} months"


def test_app_default_view(app, real_yields):
    assert len(app.metric) == 3
    # Work out the KPIs by hand from the CSV: 10Y - 2Y now, and a year earlier
    latest_day = real_yields.index[-1]
    year_ago = real_yields.index[real_yields.index <= latest_day - pd.DateOffset(years=1)][-1]
    spread_now = (real_yields.loc[latest_day, "10Y"] - real_yields.loc[latest_day, "2Y"]) * 100
    spread_then = (real_yields.loc[year_ago, "10Y"] - real_yields.loc[year_ago, "2Y"]) * 100
    ten_now = real_yields.loc[latest_day, "10Y"]
    ten_then = real_yields.loc[year_ago, "10Y"]

    assert app.metric[0].label == "10Y - 2Y spread"
    assert app.metric[0].value == f"{spread_now:+.0f} bp"
    assert app.metric[0].delta == f"{spread_now - spread_then:+.0f} bp vs a year earlier"
    assert app.metric[1].value == f"{ten_now:.2f}%"
    assert app.metric[1].delta == f"{(ten_now - ten_then) * 100:+.0f} bp vs a year earlier"
    assert app.metric[2].value == expected_lead_kpi(real_yields, "10Y - 2Y", yc.MIN_DAYS_DEFAULT, yc.MAX_LEAD_DEFAULT)

    insight = app.info[0].value
    assert insight.startswith(f"On {yc.format_date(latest_day)} (Latest) the curve was")
    # The default view says whether a recession followed the deepest inversion
    assert "Deepest inversion" in insight and "that inversion began on" in insight


def test_app_snapshot_controls(app):
    moments = app.multiselect(key="yield_curve_moments")
    for option in yc.NAMED_MOMENTS:
        moments.set_value([option]).run()
        _check(app)

    moments.set_value(["Pre-COVID (Jan 2020)", "Financial crisis (Dec 2008)",
                       "Dot-com peak (Mar 2000)", yc.LATEST]).run()
    _check(app)

    # Custom date, including one before the 20-year bond existed
    app.toggle(key="yield_curve_use_custom").set_value(True).run()
    _check(app)
    app.date_input(key="yield_curve_custom_date").set_value(dt.date(1991, 6, 2)).run()
    _check(app)
    app.radio(key="yield_curve_spread").set_value("10Y - 3M").run()
    _check(app)
    assert app.metric[0].label == "10Y - 3M spread"

    # Nothing selected: a friendly prompt, not an error
    app.toggle(key="yield_curve_use_custom").set_value(False).run()
    app.multiselect(key="yield_curve_moments").set_value([]).run()
    _check(app)
    assert any("Pick at least one moment" in info.value for info in app.info)

    # Only the custom date selected
    app.toggle(key="yield_curve_use_custom").set_value(True).run()
    _check(app)
    assert len(app.metric) == 3


def test_app_custom_date_on_the_same_day_as_latest(app, real_yields):
    latest_day = real_yields.index[-1].date()
    app.multiselect(key="yield_curve_moments").set_value([yc.LATEST]).run()
    app.toggle(key="yield_curve_use_custom").set_value(True).run()
    app.date_input(key="yield_curve_custom_date").set_value(latest_day).run()
    _check(app)
    assert any("drawn once" in caption.value for caption in app.caption)
    insight = app.info[0].value
    assert "steepest" not in insight and "lowest" not in insight  # only one curve left to describe


def test_app_history_view_every_control(app, real_yields):
    app.radio(key="yield_curve_view").set_value(yc.HISTORY_VIEW).run()
    _check(app)
    for spread_name in yc.SPREADS:
        app.radio(key="yield_curve_spread").set_value(spread_name).run()
        for period in yc.PERIODS:
            app.select_slider(key="yield_curve_period").set_value(period).run()
            _check(app)
            assert len(app.metric) == 3
    app.radio(key="yield_curve_spread").set_value("10Y - 2Y").run()

    # The sliders change the episode rule, so the KPI and the insight must change with them
    for min_days in (1, 30, 60):
        app.slider(key="yield_curve_min_days").set_value(min_days).run()
        _check(app)
        assert app.metric[2].value == expected_lead_kpi(real_yields, "10Y - 2Y", min_days, yc.MAX_LEAD_DEFAULT)
    app.slider(key="yield_curve_min_days").set_value(30).run()
    # 30+ days: only 2000 (14 months) and 2005 (25 months) count, plus the 2022 false alarm
    assert app.metric[2].value == "20 months"
    assert "2 recessions began within 30 months of an inversion" in app.info[0].value
    assert "1 inversion (Apr 2022) was not followed by one within 30 months" in app.info[0].value

    for max_lead in (6, 36, 48):
        app.slider(key="yield_curve_max_lead").set_value(max_lead).run()
        _check(app)
        assert app.metric[2].value == expected_lead_kpi(real_yields, "10Y - 2Y", 30, max_lead)
        assert f"within {max_lead} months" in app.info[0].value

    app.toggle(key="yield_curve_recessions").set_value(False).run()
    _check(app)
    assert len(app.metric) == 3

    # Back to snapshots works too
    app.radio(key="yield_curve_view").set_value(yc.SNAPSHOT_VIEW).run()
    _check(app)


def test_app_switching_views_keeps_every_control(app):
    # Change the snapshot controls
    moments = [yc.LATEST, "Financial crisis (Dec 2008)"]
    app.multiselect(key="yield_curve_moments").set_value(moments).run()
    app.toggle(key="yield_curve_use_custom").set_value(True).run()
    app.date_input(key="yield_curve_custom_date").set_value(dt.date(2006, 8, 7)).run()
    # ... then the history controls
    app.radio(key="yield_curve_view").set_value(yc.HISTORY_VIEW).run()
    app.select_slider(key="yield_curve_period").set_value("5Y").run()
    app.slider(key="yield_curve_min_days").set_value(30).run()
    app.slider(key="yield_curve_max_lead").set_value(24).run()
    app.toggle(key="yield_curve_recessions").set_value(False).run()
    history_kpi = app.metric[2].value

    # The snapshot controls survived the trip to the history view
    app.radio(key="yield_curve_view").set_value(yc.SNAPSHOT_VIEW).run()
    _check(app)
    assert app.multiselect(key="yield_curve_moments").value == moments
    assert app.toggle(key="yield_curve_use_custom").value is True
    assert app.date_input(key="yield_curve_custom_date").value == dt.date(2006, 8, 7)
    # Both views use the same episode rule, so the lead-time KPI is the same
    assert app.metric[2].value == history_kpi

    # ... and the history controls survived the trip back
    app.radio(key="yield_curve_view").set_value(yc.HISTORY_VIEW).run()
    _check(app)
    assert app.select_slider(key="yield_curve_period").value == "5Y"
    assert app.slider(key="yield_curve_min_days").value == 30
    assert app.slider(key="yield_curve_max_lead").value == 24
    assert app.toggle(key="yield_curve_recessions").value is False
    assert not app.warning  # no "default value but also set via Session State" warning
