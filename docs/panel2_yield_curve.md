# Panel 2 · The Yield Curve

**Question:** *What is the bond market signalling about recessions?*
**Code:** `dashboard/panels/yield_curve.py` · **Tests:** `tests/test_yield_curve.py` · **Data:** `data/yield_curve_treasury.csv`

The panel links Panel 1 (what the Fed did with rates) to Panels 3-4 (what markets did). It shows the US Treasury yield curve and the gap between long-term and short-term rates. When short-term rates rise above long-term rates, the curve is **inverted**. Bond investors are then expecting rate cuts, usually because they expect the economy to weaken. Most US recessions since 1990 were preceded by an inversion. The panel checks every inversion: was it followed by a recession within a time limit (30 months by default)? If so, how many months later? If not, it was a false alarm. If the time limit hasn't passed yet, it is too soon to judge.

## Data

| What | Source | Series |
|---|---|---|
| Daily Treasury yields, constant maturity, % per year, from 1990 | FRED (St. Louis Fed) | [DGS3MO](https://fred.stlouisfed.org/series/DGS3MO), [DGS6MO](https://fred.stlouisfed.org/series/DGS6MO), [DGS1](https://fred.stlouisfed.org/series/DGS1), [DGS2](https://fred.stlouisfed.org/series/DGS2), [DGS3](https://fred.stlouisfed.org/series/DGS3), [DGS5](https://fred.stlouisfed.org/series/DGS5), [DGS7](https://fred.stlouisfed.org/series/DGS7), [DGS10](https://fred.stlouisfed.org/series/DGS10), [DGS20](https://fred.stlouisfed.org/series/DGS20), [DGS30](https://fred.stlouisfed.org/series/DGS30) |
| US recession months (1 = recession) | FRED / NBER | [USREC](https://fred.stlouisfed.org/series/USREC) (shared `data/recessions.csv`) |

`load_data()` calls the shared `load_fred()`, which reads the CSV snapshot so the panel works offline. `refresh_data()` downloads fresh data (the sidebar "Refresh data" button and `python scripts/update_data.py` both call it). DGS20 is empty before Oct 1993, when the 20-year bond was not issued. Those curves simply skip the 20Y point.

## Method

- **Spread (basis points):** `spread = (long yield − short yield) × 100`, where 1 bp = 0.01 percentage points. Example: 10Y 5.21% − 2Y 4.79% = 0.42 points = **+42 bp**. Below 0 means inverted. Curves are described as *inverted* (< 0), *flat* (0-25 bp) or *upward-sloping* (≥ 25 bp).
- **Inversion episodes** (`find_inversion_episodes`), on daily data:
  1. A day is inverted when the spread is below 0.
  2. Consecutive inverted days form a *run*.
  3. Runs separated by at most `MAX_GAP_DAYS` = 63 trading days (about 3 months) are merged, because a short bounce above zero does not end the signal.
  4. An episode is kept only if it has at least `min_days` inverted days (default 10), so one-day blips are ignored.
- **Lead time** (`months_between`): the number of calendar months from the month of an episode's first inverted day to the month the next recession began: `(year2 − year1) × 12 + (month2 − month1)`. Only the month counts, not the day (e.g. 15 Mar → 1 Aug = 5, Mar → Aug), because recessions are dated by month. Only recessions that had begun by the last day of data count.
- **Outcome of an episode** (`episode_outcome`), with a time limit of `max_lead` months (default `MAX_LEAD_DEFAULT` = 30, set with the "Recession within" slider):
  - *followed by a recession*: the next recession began at most `max_lead` months after the inversion started (exactly `max_lead` still counts);
  - *no recession* (a false alarm): more than `max_lead` months have passed with no recession starting. This includes a recession that came later than the limit. Example: the May 1998 inversion was followed by the April 2001 recession only 35 months later;
  - *too soon to judge*: no recession yet, but `max_lead` months haven't passed.

  Without a limit, any recession however many years later would count as a hit, and an inversion could never be called a false alarm.
- **Each recession counts once** (`add_lead_times`, `summarise_lead_times`): sometimes two inversions come before the same recession, e.g. the 10Y − 3M in 2019 and again in January 2020, both before the COVID recession. Only the first is the "first warning". The average lead time uses first warnings only, so a late repeat doesn't pull the average down.
- **Date snapping:** a chosen date with no data (weekend, holiday) is replaced by the nearest **earlier** trading day, which is what an investor saw that day. The legend always shows the date actually used. If two chosen moments land on the same trading day (e.g. a custom date of today and "Latest"), that curve is drawn once and a caption says so.
- **Speed:** windows longer than 5 years are plotted weekly, using the last trading day of each week (`groupby` + `tail(1)`). Episodes are always computed from daily data.

## Controls

| Control | Key | Effect |
|---|---|---|
| View | `yield_curve_view` | *Spread history* (area chart, shown first because it answers the question directly) or *Curve snapshots* (line + markers by maturity). |
| Spread | `yield_curve_spread` | 10Y − 2Y (most quoted) or 10Y − 3M (the New York Fed's choice). Drives the KPIs, the episodes, the insight and "Deepest inversion". |
| Compare the curve on | `yield_curve_moments` | Up to 4 moments: Latest, 1 year ago, Deepest inversion (the day the spread was lowest), Pre-COVID, Financial crisis, Dot-com peak. Each has a fixed colour. |
| Add a custom date + Custom date | `yield_curve_use_custom`, `yield_curve_custom_date` | Adds a fifth curve for any date since 1990. |
| Period | `yield_curve_period` | 1Y, 2Y, 5Y, 10Y, 20Y or All for the spread chart. |
| Shade recessions | `yield_curve_recessions` | Shows or hides the grey NBER recession bands. |
| Min. days inverted | `yield_curve_min_days` | Sensitivity analysis: how long an inversion must last to count (1-60 trading days). |
| Recession within (months) | `yield_curve_max_lead` | Sensitivity analysis: the time limit for "followed by a recession" (6-48 months). |

The two sliders set the episode rule. They change the episode table, the insight and the "Avg. lead to recession" KPI in **both** views. The snapshot view uses the values last set in the history view. Every control keeps its value when you switch views (`_remember_control_values` saves them in `st.session_state`, because Streamlit forgets the value of a widget that isn't drawn).

**KPIs:**
1. The current spread, with its change versus a year earlier (bp).
2. The current 10Y yield, with its change versus a year earlier (bp).
3. The average lead time to recession, counting each recession once from its first warning. Only recessions that began within the time limit count. Hovering the "?" shows how many recessions that is and how many inversions were false alarms.

## Code walkthrough

| Function | What it does |
|---|---|
| `load_data`, `refresh_data`, `tidy_yields` | Load the FRED snapshot; rename `DGS10` → `10Y` using the `MATURITIES` dict, in maturity order. |
| `compute_spread_bp` | Vectorised pandas subtraction × 100; drops days with a missing yield. |
| `snap_to_previous_date` | Boolean filter `dates[dates <= target]`, then `.max()`; `None` if nothing is earlier. |
| `find_inversion_runs` | NumPy `spread < 0`, then a `for` loop that records where each run starts and ends. |
| `merge_close_runs` | Loop that joins a run to the previous one when the gap is ≤ `max_gap`. |
| `find_inversion_episodes` | Applies the rule above; returns a **list of dicts** (start, end, inverted_days, deepest_bp, deepest_date, ongoing). |
| `months_between`, `next_recession_start` | Calendar-month arithmetic; a loop that finds the first recession after a date. |
| `episode_outcome` | The time-limit rule as an if/else: followed, no recession (false alarm) or too soon. |
| `add_lead_times` | Adds the next recession, lead time, outcome and "first warning" flag to every episode dict. It keeps a list of the recessions already warned about. |
| `summarise_lead_times` | Counts the outcomes with a dict, plus the mean, min and max lead with NumPy (first warnings only). |
| `moment_dates`, `drop_same_day_moments` | Turn the chosen moment names into real trading days (if/elif chain), then keep one moment per day. |
| `last_n_years`, `to_weekly` | Window the data and thin it to weekly. The recessions that overlap the window come from the shared `recessions_in_window` in `dashboard/data.py`. |
| `snapshot_insight` (+ `comparison_sentence`, `inversion_outcome_clause`) | The snapshot insight: the first curve's shape, how it compares (handling ties and all-inverted curves), and whether a recession followed the most inverted curve shown. |
| `history_insight` (= `current_state_sentence` + `track_record_sentence`) | The history insight: where the spread is now and what followed the last episode, then the track record since 1990 (recessions, average lead, false alarms). |
| `outcome_text`, `lead_kpi_help`, `episodes_markdown`, `plural`, `was_or_were`, `list_months` | Small text helpers, so the wording is correct for 1 or many and never hard-codes facts. |
| `split_by_sign`, `build_snapshot_chart`, `build_spread_chart` | Plotly figures. `split_by_sign` makes the blue (normal) and red (inverted) areas meet at zero. The shared `theme.legend_top_margin` leaves room for every legend row, so the legend never sits under Plotly's toolbar, and `theme.add_recession_legend_entry` gives the grey bands their legend entry. A single y-axis only, with colours from `dashboard/theme.py`. |
| `render` (+ `_remember_control_values`, `_render_snapshots`, `_render_history`, `_show_kpis`) | The only Streamlit code: controls → 3 KPIs → chart → insight → details. |

## Course concepts used

| Session | Concept | Where it is used |
|---|---|---|
| S3 | Lists, dicts, tuples | `MATURITIES`, `SPREADS`, `PERIODS`, `MOMENT_COLOURS`; episodes are a list of dicts; runs are tuples; outcome counts in a dict. |
| S4 | Loops and conditionals | Run detection, gap merging, the outcome rule, the moment if/elif chain, insight wording. |
| S5 | Functions | Docstrings, default and keyword arguments, small refactored helpers. |
| S6 | NumPy, FRED, CSV snapshots | Boolean arrays, `.sum()`, `np.mean`; FRED data saved as CSV snapshots. |
| S7 | pandas | DataFrames, `.loc`/`.iloc`, boolean filtering, `dropna`, `idxmin`. |
| S9 | Time series | `DateOffset`, `shift`, `clip`/`where`, `groupby(pd.Grouper(freq="W-FRI"))`. |
| S10 | Visualisation | Line, marker and area charts, a categorical axis, recession bands, hover templates. Drawn with Plotly instead of Matplotlib so they have hover and zoom in the web dashboard (see "Why Plotly" in the README). |
| Testing | pytest + AppTest | Robustness and accuracy (learning outcome 5). |

## Limitations

- The data start in 1990. The 1989 inversion before the 1990 recession is therefore missed, and the March 1990 episode is a short re-inversion, so its 5-month lead understates that cycle.
- The 30-month time limit is a judgement call, and the result depends on it. With the defaults, the 10Y − 2Y record is 3 recessions (1990, 2001, 2008), an average lead of 15 months (range 5-25), and 2 false alarms (1998 and 2022). With a 36-month limit the 1998 inversion counts instead, as the first warning of the 2001 recession, and the average rises to 22 months. The "Recession within" slider shows this sensitivity.
- The panel counts false alarms but not *missed* recessions. With the 10-day rule, the 10Y − 2Y did not invert before the 2020 COVID recession: the August 2019 dip lasted only 3 days. Only a handful of episodes exist, so the average is a description, not a forecast.
- `MAX_GAP_DAYS`, `FLAT_LIMIT_BP` and the default `min_days` are judgement calls too. The "Min. days inverted" slider shows how sensitive the results are to the minimum length.
- NBER dates recessions months later, and USREC marks the month *after* the peak as the first recession month. A recession that has started but not yet been announced is therefore not in the data, so a recent inversion can look "too soon to judge" or like a false alarm until the data catch up.

## Extension ideas

1. **Recession probability model:** fit a logistic regression of "recession within 12 months" on the 10Y − 3M spread, as the New York Fed does, and plot the probability.
2. **Real vs nominal:** add TIPS yields (FRED `DFII10`) and show breakeven inflation (`DGS10 − DFII10`). This links to Panel 1.
3. **Animated curve:** a monthly date slider that redraws the curve, so the 2022-2024 inversion can be watched forming and unwinding.

## Run it

```bash
python -m pytest -q tests/test_yield_curve.py          # unit tests + AppTest (no internet needed)
streamlit run scripts/preview_panel.py -- yield_curve  # this panel on its own
streamlit run app.py                                   # the full dashboard
```
