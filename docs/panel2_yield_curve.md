# Panel 2 · The Yield Curve

**Question:** *What is the bond market signalling about recessions?*
**Code:** `dashboard/panels/yield_curve.py` · **Tests:** `tests/test_yield_curve.py` · **Data:** `data/yield_curve_treasury.csv`

The panel links Panel 1 (what the Fed did with rates) to Panels 3-4 (what markets did). It shows the US Treasury yield curve and the gap between long-term and short-term rates. When short-term rates rise above long-term rates, the curve is **inverted**. Bond investors are then expecting rate cuts, usually because they expect the economy to weaken. Most US recessions since 1990 were preceded by an inversion. The panel measures how long each inversion came before a recession, and states honestly when no recession has followed (yet).

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
- **Lead time:** whole calendar months from an episode's first inverted day to the first month of the next recession: `(year2 − year1) × 12 + (month2 − month1)`. If no recession has started since, the lead time is `None`. The text then says "not yet" or "has not been followed by a recession so far", instead of pretending a recession came.
- **Date snapping:** a chosen date with no data (weekend, holiday) is replaced by the nearest **earlier** trading day, which is what an investor saw that day. The legend always shows the date actually used.
- **Speed:** windows longer than 5 years are plotted weekly, using the last trading day of each week (`groupby` + `tail(1)`). Episodes are always computed from daily data.

## Controls

| Control | Key | Effect |
|---|---|---|
| View | `yield_curve_view` | *Curve snapshots* (line + markers by maturity) or *Spread history* (area chart). |
| Spread | `yield_curve_spread` | 10Y − 2Y (most quoted) or 10Y − 3M (the New York Fed's choice). Drives the KPIs, the episodes, the insight and "Deepest inversion". |
| Compare the curve on | `yield_curve_moments` | Up to 4 moments: Latest, 1 year ago, Deepest inversion (the day the spread was lowest), Pre-COVID, Financial crisis, Dot-com peak. Each has a fixed colour. |
| Add a custom date + Custom date | `yield_curve_use_custom`, `yield_curve_custom_date` | Adds a fifth curve for any date since 1990. |
| Period | `yield_curve_period` | 1Y, 2Y, 5Y, 10Y, 20Y or All for the spread chart. |
| Min. inversion (trading days) | `yield_curve_min_days` | Sensitivity analysis: how long an inversion must last to count (1-60). Changes the episode table, the insight and the "Avg. lead to recession" KPI. |
| Recessions | `yield_curve_recessions` | Shows or hides the grey NBER recession bands. |

**KPIs:**
1. The current spread, with its change versus a year earlier.
2. The current 10Y yield, with its change in bp.
3. The average lead time from inversion to recession, over the episodes that were followed by one. Hovering the "?" shows how many episodes that is.

## Code walkthrough

| Function | What it does |
|---|---|
| `load_data`, `refresh_data`, `tidy_yields` | Load the FRED snapshot; rename `DGS10` → `10Y` using the `MATURITIES` dict, in maturity order. |
| `compute_spread_bp` | Vectorised pandas subtraction × 100; drops days with a missing yield. |
| `snap_to_previous_date` | Boolean filter `dates[dates <= target]`, then `.max()`; `None` if nothing is earlier. |
| `find_inversion_runs` | NumPy `spread < 0`, then a `for` loop that records where each run starts and ends. |
| `merge_close_runs` | Loop that joins a run to the previous one when the gap is ≤ `max_gap`. |
| `find_inversion_episodes` | Applies the rule above; returns a **list of dicts** (start, end, inverted_days, deepest_bp, deepest_date, ongoing). |
| `months_between`, `months_to_next_recession`, `add_lead_times`, `summarise_lead_times` | Lead-time arithmetic, plus count, mean, min and max with NumPy. |
| `moment_dates` | Turns the chosen moment names into real trading days (if/elif chain). |
| `last_n_years`, `to_weekly`, `recessions_in_window` | Window the data, thin it to weekly, and keep the recessions that overlap. |
| `snapshot_insight`, `history_insight`, `episodes_markdown` | Build the insight sentences and the episode table from the data, with no hard-coded facts. |
| `split_by_sign`, `build_snapshot_chart`, `build_spread_chart` | Plotly figures. `split_by_sign` makes the blue (normal) and red (inverted) areas meet at zero. A single y-axis only, with colours from `dashboard/theme.py`. |
| `render` (+ `_render_snapshots`, `_render_history`, `_show_kpis`) | The only Streamlit code: controls → 3 KPIs → chart → insight → details. |

## Course concepts used

| Session | Concept | Where it is used |
|---|---|---|
| S3 | Lists, dicts, tuples | `MATURITIES`, `SPREADS`, `PERIODS`, `MOMENT_COLOURS`; episodes are a list of dicts; runs are tuples. |
| S4 | Loops and conditionals | Run detection, gap merging, the moment if/elif chain, insight wording. |
| S5 | Functions | Docstrings, default and keyword arguments, small refactored helpers. |
| S6 | NumPy, FRED, CSV snapshots | Boolean arrays, `.sum()`, `np.mean`; FRED data saved as CSV snapshots. |
| S7 | pandas | DataFrames, `.loc`/`.iloc`, boolean filtering, `dropna`, `idxmin`. |
| S9 | Time series | `DateOffset`, `shift`, `clip`/`where`, `groupby(pd.Grouper(freq="W-FRI"))`. |
| S10 | Visualisation | Line, marker and area charts, a categorical axis, recession bands, hover templates. |
| Testing | pytest + AppTest | Robustness and accuracy (learning outcome 5). |

## Limitations

- The data start in 1990. The 1989 inversion before the 1990 recession is therefore missed, and the March 1990 episode is a short re-inversion, so its 5-month lead understates that cycle.
- Lead times are "to the *next* recession", however far away. The 1998 inversion counts with a 35-month lead, and the 1998 and 2000 inversions point to the same 2001 recession. Only a handful of episodes exist, so the average is a description, not a forecast.
- `MAX_GAP_DAYS`, `FLAT_LIMIT_BP` and the default `min_days` are judgement calls. The slider shows how sensitive the results are to the minimum length.
- NBER dates recessions months later, and USREC marks the month *after* the peak as the first recession month.

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
