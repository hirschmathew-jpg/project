# Panel 1 · Inflation vs the Fed

**Question:** did interest rates keep up with inflation?

The panel compares the interest rate the US Federal Reserve actually set (the effective fed funds rate) with the rate a simple policy rule, the **Taylor rule**, says it "should" have set, given inflation and unemployment at the time. When the blue fed funds line sits below the dashed orange rule line, policy was looser than the rule suggests (the Fed was "behind the curve"). With the data snapshot in the repo, the default view (Core PCE, 2000 to today) shows the main story: in March 2022 core PCE inflation was 5.6% but the fed funds rate was still 0.2%, almost 10 percentage points below the rule.

Code: `dashboard/panels/inflation.py` · Tests: `tests/test_inflation.py` · Data snapshot: `data/inflation_macro.csv`

## Data

All series are monthly from FRED (Federal Reserve Bank of St. Louis), downloaded from January 1985 through the shared helper `load_fred("inflation_macro", ...)`:

| FRED ID | What it is | Used for |
|---|---|---|
| [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) | Consumer Price Index, all items | "Headline CPI" inflation |
| [CPILFESL](https://fred.stlouisfed.org/series/CPILFESL) | CPI excluding food and energy | "Core CPI" inflation |
| [PCEPILFE](https://fred.stlouisfed.org/series/PCEPILFE) | Core PCE price index (the Fed's preferred gauge) | "Core PCE" inflation |
| [FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) | Effective federal funds rate, monthly average, % | The Fed's actual policy rate |
| [UNRATE](https://fred.stlouisfed.org/series/UNRATE) | Unemployment rate, % | Output gap |
| [NROU](https://fred.stlouisfed.org/series/NROU) | CBO natural rate of unemployment, % (quarterly) | Output gap |
| [USREC](https://fred.stlouisfed.org/series/USREC) | NBER recession indicator (shared file) | Grey recession bands |

Two cleaning steps happen in `prepare_monthly_data`. First, NROU is quarterly and also contains CBO *projections* up to ten years ahead, so the table is cut at the last month with real data and each quarterly value is carried forward (`ffill`) to the next two months. Second, `asfreq("MS")` gives every calendar month its own row, so "12 rows back" always means "one year back". Months that are missing at the source (for example October 2025 CPI and unemployment, which are missing in FRED) stay missing, so the lines show a small break instead of an invented value.

## Method

**Inflation** π is the year-on-year % change of the chosen price index: `pct_change(12) * 100`.

**Output gap** (how far output is above or below its full-capacity level), using Okun's law with `OKUN_COEFFICIENT = 2.0`:

    gap = −2 × (unemployment − natural rate)

Unemployment one point above its natural rate means output is roughly 2% below potential.

**Taylor rule** (Taylor, 1993):

    rule rate = r* + π + a × (π − π*) + b × gap

The term r* + π is the "neutral" nominal rate. The rule then adds more when inflation is above the target π* (weight a) and when the economy runs hot (weight b). Presets: **Taylor (1993)**: r* = 2, π* = 2, a = 0.5, b = 0.5. **Balanced approach**: the same but b = 1.0, which reacts more to unemployment. The optional zero floor applies `max(rate, 0)`, because a central bank cannot really cut much below 0% (the "zero lower bound").

**Fed vs rule** = fed funds rate − rule rate, in percentage points (pp). Negative means policy was looser than the rule.

Worked check, as used in the tests: π = 4, π* = 2, r* = 2, a = b = 0.5, gap = 1 gives 2 + 4 + 0.5 × 2 + 0.5 × 1 = **7.5%**.

## Controls

| Control | Effect |
|---|---|
| Inflation measure | Headline CPI, Core CPI or Core PCE feeds π (default Core PCE) |
| Policy rule | Picks a preset and moves the four sliders to its values |
| Period | Range of years shown (default 2000 to the latest year) |
| Neutral rate r\*, Target π\*, Weight a (π), Weight b (gap) | Rule parameters (r\*, inflation target, inflation weight, output-gap weight). Moving one switches the preset to "Custom", or to a preset whose values now match. Labels are short so the four sliders fit side by side; the help icons give the full meaning |
| Floor at zero | The rule's rate never goes below 0% (it is floored at zero) |
| Show inflation | Shows or hides the inflation line |

The three KPIs all refer to the **same month**: the latest month in the chosen period that has both the fed funds rate and the rule. They show inflation (red when above target), the fed funds rate (change versus a year earlier) and the rule rate (delta = Fed vs rule). Because the month is shared, the "Fed vs rule" delta is exactly the fed funds KPI minus the rule KPI. Core PCE comes out about a month after the fed funds rate, so a newer fed funds value can exist; the help icon on the fed funds KPI then gives it.

The insight sentence is built from the current data. It gives the latest gap, plus the months in the chosen period when the Fed was furthest below and furthest above the rule. Gaps under 0.25 pp count as "in line". Being behind the rule does not always mean rates were below inflation, because the rule also asks for more when unemployment is below its natural rate. So for the month furthest behind, the sentence gives the rule rate and the fed funds rate, then checks whether fed funds was below or still above inflation that month. Whichever of the two extremes is bigger (in either direction) is marked on the chart with a dotted line and a label. Grey recession bands, and their "US recession" legend entry, appear only when a recession falls inside the chosen period.

## Code walkthrough

| Function | What it does |
|---|---|
| `prepare_monthly_data(raw)` | Cuts NROU projections, gives one row per month, forward-fills NROU |
| `load_data(refresh)` / `refresh_data()` | Read the CSV snapshot, or download fresh data from FRED |
| `yoy_inflation(price_index)` | Year-on-year inflation in % |
| `okun_output_gap(u, u_star)` | Output gap from unemployment |
| `taylor_rule(...)` | The rule; works on single numbers and on whole Series (vectorised) |
| `match_preset(params)` | Loops over the presets dict to find one equal to the slider values |
| `build_model_table(data, measure, params, floor)` | One DataFrame with every input, the rule rate and the policy gap |
| `latest_value`, `change_over_year`, `largest_gaps`, `missing_months` | Small analysis helpers that return `None` when there is nothing to report (`largest_gaps(..., min_gap=0.25)` ignores tiny gaps) |
| `describe_latest_gap`, `describe_extremes`, `write_insight` | Build the insight sentences with if/elif branches for each case |
| `pick_highlight`, `make_display_table` | Month to mark on the chart (the bigger of the two extremes); table for the "View the data" expander |
| `y_axis_range`, `add_gap_marker`, `build_chart` | The Plotly figure: three lines on one % axis, recession bands (only those inside the window, picked by the shared `recessions_in_window` in `dashboard/data.py`), 0% hairline, gap label |
| `_draw_controls`, `_draw_kpis`, `render` | Streamlit layout. Presets and sliders are kept in sync with `st.session_state` and `on_change` callbacks |

## Course concepts used

- **S3 data structures:** dicts for measures, presets and slider settings; lists of series IDs; tuples for (month, value).
- **S4 control flow:** loops over presets and sliders; if/elif branches in the insight and edge cases.
- **S5 functions:** short functions with docstrings and type hints; `**params` (kwargs) unpacks a preset into `taylor_rule`; refactoring into pure functions plus a thin `render()`.
- **S6 NumPy / CSV / FRED:** `np.maximum` for the zero floor; CSV snapshots; FRED download.
- **S7 and S9 pandas:** DataFrames, `.loc` and boolean filtering, `pct_change`, `ffill`, `asfreq`, `idxmin`/`idxmax`, `dropna`, date indexes.
- **S10 visualisation:** the line chart from S10, drawn with Plotly instead of Matplotlib so it has hover and zoom in the web dashboard (see "Why Plotly" in the README): a single axis, consistent colours, a legend, hover text and annotations.

## Limitations

- The Taylor rule is a rule of thumb, not what the Fed actually follows. Results depend heavily on r\*, π\* and the weights, and r\* itself is uncertain and probably changes over time.
- The output gap comes from unemployment through a fixed Okun coefficient of 2. The true relationship varies, and with the Taylor (1993) settings it pushes the rule to about −8% in April 2020.
- The rule uses the latest revised data, not what the Fed knew at the time (real-time data), and it ignores forward guidance and quantitative easing.
- Core PCE is published about a month after CPI and the fed funds rate. The KPIs and the insight therefore use the latest month with every input, which can be a month before the newest fed funds value.

## Extension ideas

1. Add a "real fed funds rate" line (fed funds minus inflation) to answer "did rates keep up with inflation?" without any model.
2. Add a "Fed reaction lag" slider that compares the fed funds rate with the rule from *n* months earlier, to test whether the Fed simply reacted late.
3. Let r\* follow a published estimate over time (for example the Laubach-Williams r\* from the New York Fed) instead of a constant.

## Run and test

```bash
python -m pytest -q tests/test_inflation.py                   # unit tests + headless app tests (no internet)
streamlit run scripts/preview_panel.py -- inflation           # this panel on its own
streamlit run app.py                                          # the whole dashboard
python -c "from dashboard.panels import inflation; inflation.refresh_data()"   # download fresh data
```
