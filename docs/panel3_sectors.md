# Panel 3 · Sector Winners & Losers

**Question:** which parts of the stock market won and lost?

The S&P 500 is split into 11 sectors (technology, energy, financials and so on), and each sector can be bought as an exchange-traded fund (ETF). The panel compares those 11 sector ETFs with SPY, an ETF that holds the whole S&P 500. It can rank them on return or on risk, or show how $100 invested in each would have grown. The default period is the **2022 rate-hike sell-off**, which links the panel to the dashboard's story: in the year the Fed raised rates fastest, the S&P 500 lost 18.2% while Energy *gained* 64.3%. Communication Services fell 37.6%, and 7 of the 11 sectors held up better than the index (figures from the data snapshot in the repo).

Code: `dashboard/panels/sectors.py` · Tests: `tests/test_sectors.py` · Data snapshot: `data/sectors_prices.csv`

## Data

Daily **adjusted** closing prices from Yahoo Finance, downloaded from January 2007 with the shared helper `load_prices("sectors_prices", ...)` (yfinance with `auto_adjust=True`). Adjusted prices include dividends, so the % change of the price is the total return an investor received.

| Ticker | Fund | Ticker | Fund |
|---|---|---|---|
| [SPY](https://finance.yahoo.com/quote/SPY) | S&P 500 (the benchmark) | [XLP](https://finance.yahoo.com/quote/XLP) | Consumer Staples |
| [XLK](https://finance.yahoo.com/quote/XLK) | Technology | [XLI](https://finance.yahoo.com/quote/XLI) | Industrials |
| [XLF](https://finance.yahoo.com/quote/XLF) | Financials | [XLU](https://finance.yahoo.com/quote/XLU) | Utilities |
| [XLE](https://finance.yahoo.com/quote/XLE) | Energy | [XLB](https://finance.yahoo.com/quote/XLB) | Materials |
| [XLV](https://finance.yahoo.com/quote/XLV) | Health Care | [XLRE](https://finance.yahoo.com/quote/XLRE) | Real Estate (trades from Oct 2015) |
| [XLY](https://finance.yahoo.com/quote/XLY) | Consumer Discretionary | [XLC](https://finance.yahoo.com/quote/XLC) | Communication Services (from Jun 2018) |

The snapshot covers 3 Jan 2007 to 14 Oct 2026 (4,977 trading days, about 600 KB) with no gaps inside any ticker's history; it ends at the last full close because the download was made during US trading hours. A later refresh may not be so tidy, so `clean_prices` prepares the data before use: it drops days without an S&P 500 price (a half-finished last day) and fills a single missing price with the previous close (`ffill`). Prices before an ETF started trading stay empty, because `ffill` only fills forward from the first real price. Grey recession bands use the shared NBER indicator [USREC](https://fred.stlouisfed.org/series/USREC) from FRED.

## Method

**Choosing the window.** `period_dates` turns the Period choice into a start and an end date. `window_prices` then starts the window at the **last close on or before the start date**, because a return is measured from one closing price to the next. "Year to date" therefore starts at the close on 31 December of the previous year, and "2022" runs from 31 Dec 2021 to 30 Dec 2022. The crash episodes use the S&P 500's peak and trough closes (19 Feb to 23 Mar 2020; 9 Oct 2007 to 9 Mar 2009). A sector with **no price on the window's first day**, such as XLC before June 2018, has no starting price to measure from. It is left out, and the caption under the chart names it.

**Measures.** P = price, r = daily return = P(today) / P(yesterday) − 1, 252 trading days per year:

| Measure | Formula | Meaning |
|---|---|---|
| Total return | P(last) / P(first) − 1 | What $1 became, minus the $1 |
| Annualised volatility | std(r) × √252 | How much returns swing; √252 because variances add up over time |
| Sharpe ratio | mean(r − r_f) × 252 ÷ (std(r) × √252), with r_f = annual risk-free rate ÷ 252 | Return above cash per unit of risk |
| Maximum drawdown | min(P / running maximum of P − 1) | Worst fall from an earlier high (`cummax`) |
| Beta vs S&P 500 | cov(r_sector, r_SPY) ÷ var(r_SPY) | 1.2 means the sector tended to move 1.2% when the market moved 1% |

**Relative to S&P 500** = sector value − S&P 500 value (in percentage points, pp, for % measures). Bars are sorted **best first**: highest return, Sharpe ratio and drawdown (the smallest fall), and lowest volatility and beta (least risk). A sector "beats" the S&P 500 when it is better in that same sense.

Two measures need care when reading them:

- **Beta is sensitivity to the market, not the size of a sector's swings.** In 2022 Energy had a beta of only 0.64 but the highest volatility (35% against the S&P 500's 24%): its big moves had little to do with the index. A negative beta (Energy year to date in 2026: −0.52) means the sector tended to rise on days the market fell. The panel says "less sensitive to S&P 500 moves", never "moved less".
- **Negative Sharpe ratios can't be ranked fairly.** When a sector earns less than the risk-free rate, dividing that shortfall by a *bigger* volatility gives a number *closer to zero*. In the global financial crisis Energy had the highest Sharpe ratio (−0.57) because it was so volatile (56%), even though Consumer Staples lost far less (−28.5% against −47.9%). When every ratio is negative, the key numbers say "Highest/Lowest" instead of "Best/Worst" (with grey changes, not green/red), and the insight advises comparing total returns instead.

Worked checks used in the tests: prices 100 → 120 → 90 → 130 → 104 give a maximum drawdown of 90/120 − 1 = **−25%**. Daily returns of +1%, −1%, +1%, −1% give a volatility of √(0.0004/3) × √252 = **18.3%**. A copy of the market with exactly twice its daily moves has a beta of **2**.

## Controls

Row 1 holds the period and the view, row 2 the options of the chosen view, and row 3 appears only when needed (custom dates, risk-free rate).

| Control | Effect |
|---|---|
| Period | Year to date, last 12 months / 3 years / 5 years, three historical episodes, or Custom |
| View | **Ranking** (sorted bars) or **Growth of $100** (lines) |
| Measure | Ranking only: which of the five measures the bars rank and the three key numbers show |
| Relative to S&P 500 | Ranking only: bars show sector minus S&P 500; blue = better than the index, red = worse |
| Highlight sector | Growth only: the sector drawn in blue. If it has no data for the period, the best performer is used and the caption says so |
| Custom dates | Shown only for Custom. Until both dates are picked the panel asks for them; dates the wrong way round are swapped; fewer than 5 trading days shows a message |
| Risk-free rate | Shown only for the Sharpe ratio (0–6% a year, default 2%) |

Controls that are hidden keep their value: Streamlit forgets a widget's value on a run where it isn't drawn, so `_remember_hidden_controls` copies each value into `st.session_state` on every run. Switching to the Growth view and back keeps the measure, the Relative toggle and the risk-free rate.

**Charts.** *Ranking*: returns and Sharpe ratios are blue above zero and red below, drawdowns red (losses), volatility and beta plain blue (sizes, not good or bad); the S&P 500 is a dashed reference line, not a bar. When every bar is negative (drawdowns), the axis runs a little past 0 so the "0%" tick label stays visible. *Growth of $100*: other sectors are thin grey context lines, the highlighted sector blue and the S&P 500 dark grey, an emphasis pattern that avoids eleven hard-to-tell-apart colours. Its legend sits under the chart, where Plotly's toolbar can't cover it. Periods longer than about 2½ years use one point per week (`thin_to_weekly`).

**Key numbers and insight.** The KPIs show the S&P 500's value of the measure and the best and worst sector, with their difference from the S&P 500. In the Growth view they always show **total return**, because that is what the lines show ($100 became $100 × (1 + total return)). The insight is computed from the current data: leader, laggard and how many sectors beat the index. There is special wording for "every sector lost/made money", "none of the sectors", all-negative Sharpe ratios, and a negative beta ("even tended to move against it"). In the Growth view, the insight says what $100 in the highlighted sector became versus the S&P 500, and its worst fall along the way. The dollar gap is computed from the two amounts *after* rounding to cents, so it matches the numbers printed beside it.

## Code walkthrough

| Function | What it does |
|---|---|
| `load_data(refresh)` / `refresh_data()` | Read the CSV snapshot, or download fresh prices from Yahoo Finance |
| `sector_name`, `short_name`, `ticker_label`, `day_label` | Names and date labels for text and charts |
| `clean_prices(prices)` | Drops days without an S&P 500 price and fills single missing prices with `ffill` |
| `period_dates(period, last_date, ...)` | Start and end date for a Period choice (if/elif over the period dicts) |
| `window_prices(prices, start, end)` | Cuts the window, beginning at the last close on or before the start |
| `split_by_data_coverage(window)` | Loops over tickers and splits them into (complete, excluded) by whether they have a starting price |
| `daily_returns`, `total_return`, `annualised_volatility`, `sharpe_ratio`, `max_drawdown`, `beta` | The five measures, each for every ticker at once (vectorised pandas) |
| `compute_metrics(prices, risk_free)` | One table: rows = tickers, columns = measures |
| `rank_sectors`, `relative_to_benchmark`, `beats_benchmark` | Sort best first, subtract the S&P 500, compare with it |
| `all_sharpe_ratios_negative`, `kpi_labels` | Detect the "every Sharpe ratio below zero" case and pick the KPI titles for it |
| `format_value`, `count_phrase`, `leader_story`, `ranking_insight`, `growth_insight` | Turn numbers into text such as "+12.3%" and the insight sentences |
| `growth_of_100`, `thin_to_weekly` | The Growth view's data; weekly thinning with `groupby(...).tail(1)` |
| `bar_colours`, `value_axis_range`, `add_value_labels`, `add_benchmark_line`, `build_ranking_chart` | The bar chart: colours, axis room for labels, labels, S&P 500 line |
| `add_end_labels`, `build_growth_chart` | The line chart with emphasis colours, recession bands and end labels |
| `_remember_hidden_controls`, `_draw_controls`, `_custom_range`, `_draw_kpis`, `_window_caption`, `_metrics_table`, `_growth_table`, `_method_markdown`, `render` | Streamlit layout. `render` checks every edge case (no data, half-picked dates, too few days, missing tickers) before drawing |

All settings for a measure live in one dictionary, `METRICS`: its unit, whether it shows a +/− sign, colour rule, sort direction, KPI titles and sentence templates. Adding a sixth measure means adding one entry plus one function.

## Course concepts used

- **S3 data structures:** dicts (`SECTORS`, `EPISODES`, `LOOKBACK_YEARS`, `METRICS` as a dict of dicts), lists of tickers, tuples for (start, end).
- **S4 control flow:** `for` loops over tickers and bars; `if/elif/else` for periods, colours, number formats and insight wording.
- **S5 functions:** short pure functions with docstrings and type hints; sentence templates filled in with `str.format` keyword arguments; `render()` kept thin by refactoring into helpers.
- **S6 NumPy, CSV, yfinance:** `np.sqrt`, `np.nan`; CSV snapshot in `data/`; Yahoo Finance download through yfinance.
- **S7 pandas:** DataFrames, `.loc`/`.iloc`, boolean filtering, `std`, `mean`, `min`, `sort_values`, `dropna`.
- **S9 time series:** `pct_change`, `cummax`, `cov`/`var`, `ffill` for missing prices, `pd.DateOffset`, `to_period("W")` with `groupby`.
- **S10 visualisation:** bar and line charts on a single axis, diverging and emphasis colours, a reference line, annotations, hover text and recession shading. Drawn with Plotly instead of Matplotlib so they have hover and zoom in the web dashboard (see "Why Plotly" in the README).

## Limitations

- Sector membership changes over time. In 2018 Alphabet and Meta moved from Technology, and Netflix from Consumer Discretionary, into the new Communication Services sector; Real Estate became its own sector in 2016 after being part of Financials. Long comparisons across those dates mix different baskets of companies.
- The Sharpe ratio uses one constant risk-free rate, but real Treasury bill yields were near 0% in 2009–2015 and 2020–2021 and around 5% in 2023–2024.
- Volatility scaled by √252 assumes daily returns are independent. Beta from daily data in a single window can change a lot from one period to the next.
- Sector ETFs are weighted by company size, so a few very large companies can drive a whole sector (for example XLK). ETF fees (about 0.1% a year) are included in the prices.
- Past performance in a chosen window says nothing certain about the future, and window choice changes the ranking.

## Extension ideas

1. Replace the risk-free slider with the actual average 3-month Treasury bill yield over the window, from FRED series [DTB3](https://fred.stlouisfed.org/series/DTB3) via `load_fred`.
2. Add a "calendar-year returns" heatmap (sectors × years, `groupby(index.year)`) to show how leadership rotates from year to year.
3. Add a correlation matrix of sector daily returns to show which sectors diversify each other, or compare with the equal-weight S&P 500 (RSP).

## Run and test

```bash
python -m pytest -q tests/test_sectors.py                     # unit tests + headless app tests (no internet)
streamlit run scripts/preview_panel.py -- sectors             # this panel on its own
streamlit run app.py                                          # the whole dashboard
python -c "from dashboard.panels import sectors; sectors.refresh_data()"   # download fresh prices
```
