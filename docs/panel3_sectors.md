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

The snapshot covers 3 Jan 2007 to 14 Oct 2026 (4,977 trading days, about 600 KB) with no gaps inside any ticker's history; it ends at the last full close because the download was made during US trading hours. Grey recession bands use the shared NBER indicator [USREC](https://fred.stlouisfed.org/series/USREC) from FRED.

## Method

**Choosing the window.** `period_dates` turns the Period choice into a start and an end date. `window_prices` then starts the window at the **last close on or before the start date**, because a return is measured from one closing price to the next. "Year to date" therefore starts at the close on 31 December of the previous year, and "2022" runs from 31 Dec 2021 to 30 Dec 2022. The crash episodes use the S&P 500's peak and trough closes (19 Feb to 23 Mar 2020; 9 Oct 2007 to 9 Mar 2009). A sector without a price on **every** day of the window, such as XLC before June 2018, is left out, and the caption under the chart names it.

**Measures.** P = price, r = daily return = P(today) / P(yesterday) − 1, 252 trading days per year:

| Measure | Formula | Meaning |
|---|---|---|
| Total return | P(last) / P(first) − 1 | What $1 became, minus the $1 |
| Annualised volatility | std(r) × √252 | How much returns swing; √252 because variances add up over time |
| Sharpe ratio | mean(r − r_f) × 252 ÷ (std(r) × √252), with r_f = annual risk-free rate ÷ 252 | Return above cash per unit of risk |
| Maximum drawdown | min(P / running maximum of P − 1) | Worst fall from an earlier high (`cummax`) |
| Beta vs S&P 500 | cov(r_sector, r_SPY) ÷ var(r_SPY) | 1.2 means the sector tended to move 1.2% when the market moved 1% |

**Relative to S&P 500** = sector value − S&P 500 value (in percentage points, pp, for % measures). Bars are sorted **best first**: highest return, Sharpe ratio and drawdown (the smallest fall), and lowest volatility and beta (least risk). A sector "beats" the S&P 500 when it is better in that same sense.

Worked checks used in the tests: prices 100 → 120 → 90 → 130 → 104 give a maximum drawdown of 90/120 − 1 = **−25%**. Daily returns of +1%, −1%, +1%, −1% give a volatility of √(0.0004/3) × √252 = **18.3%**. A copy of the market with exactly twice its daily moves has a beta of **2**.

## Controls

| Control | Effect |
|---|---|
| Period | Year to date, last 12 months / 3 years / 5 years, three historical episodes, or Custom |
| Custom dates | Shown only for Custom. Dates the wrong way round are swapped; fewer than 5 trading days shows a message |
| Measure | Which of the five measures the bars rank and the three key numbers show |
| Risk-free rate | Shown only for the Sharpe ratio (0–6% a year, default 2%) |
| View | **Ranking** (sorted bars) or **Growth of $100** (lines) |
| Relative to S&P 500 | Ranking only: bars show sector minus S&P 500; blue = better than the index, red = worse |
| Highlight sector | Growth only: the sector drawn in blue. If it has no data for the period, the best performer is used and the caption says so |

**Charts.** *Ranking*: returns and Sharpe ratios are blue above zero and red below, drawdowns red (losses), volatility and beta plain blue (sizes, not good or bad); the S&P 500 is a dashed reference line, not a bar. *Growth of $100*: other sectors are thin grey context lines, the highlighted sector blue and the S&P 500 dark grey, an emphasis pattern that avoids eleven hard-to-tell-apart colours. Periods longer than about 2½ years use one point per week (`thin_to_weekly`).

**Key numbers and insight.** The KPIs show the S&P 500's value of the measure and the best and worst sector, with their difference from the S&P 500. The insight is computed from the current data: leader, laggard and how many sectors beat the index (with special wording for "every sector lost money" or "none of the sectors"); in the Growth view, what $100 in the highlighted sector became versus the S&P 500 and its worst fall along the way.

## Code walkthrough

| Function | What it does |
|---|---|
| `load_data(refresh)` / `refresh_data()` | Read the CSV snapshot, or download fresh prices from Yahoo Finance |
| `sector_name`, `short_name`, `ticker_label`, `day_label` | Names and date labels for text and charts |
| `period_dates(period, last_date, ...)` | Start and end date for a Period choice (if/elif over the period dicts) |
| `window_prices(prices, start, end)` | Cuts the window, beginning at the last close on or before the start |
| `split_by_data_coverage(window)` | Loops over tickers and splits them into (complete, excluded) |
| `daily_returns`, `total_return`, `annualised_volatility`, `sharpe_ratio`, `max_drawdown`, `beta` | The five measures, each for every ticker at once (vectorised pandas) |
| `compute_metrics(prices, risk_free)` | One table: rows = tickers, columns = measures |
| `rank_sectors`, `relative_to_benchmark`, `beats_benchmark` | Sort best first, subtract the S&P 500, compare with it |
| `format_value`, `count_phrase`, `leader_story`, `ranking_insight`, `growth_insight` | Turn numbers into text such as "+12.3%" and the insight sentences |
| `growth_of_100`, `thin_to_weekly` | The Growth view's data; weekly thinning with `groupby(...).tail(1)` |
| `bar_colours`, `value_axis_range`, `add_value_labels`, `add_benchmark_line`, `build_ranking_chart` | The bar chart: colours, axis room for labels, labels, S&P 500 line |
| `add_end_labels`, `build_growth_chart` | The line chart with emphasis colours, recession bands and end labels |
| `_draw_controls`, `_draw_kpis`, `_window_caption`, `_metrics_table`, `_growth_table`, `_method_markdown`, `render` | Streamlit layout. `render` checks every edge case (no data, too few days, missing tickers) before drawing |

All settings for a measure live in one dictionary, `METRICS`: its unit, colour rule, sort direction, KPI titles and sentence templates. Adding a sixth measure means adding one entry plus one function.

## Course concepts used

- **S3 data structures:** dicts (`SECTORS`, `EPISODES`, `LOOKBACK_YEARS`, `METRICS` as a dict of dicts), lists of tickers, tuples for (start, end).
- **S4 control flow:** `for` loops over tickers and bars; `if/elif/else` for periods, colours, number formats and insight wording.
- **S5 functions:** short pure functions with docstrings and type hints; sentence templates filled in with `str.format` keyword arguments; `render()` kept thin by refactoring into helpers.
- **S6 NumPy, CSV, yfinance:** `np.sqrt`, `np.nan`; CSV snapshot in `data/`; Yahoo Finance download through yfinance.
- **S7 pandas:** DataFrames, `.loc`/`.iloc`, boolean filtering, `std`, `mean`, `min`, `sort_values`, `dropna`.
- **S9 time series:** `pct_change`, `cummax`, `cov`/`var`, `pd.DateOffset`, `to_period("W")` with `groupby`.
- **S10 visualisation:** Plotly bar and line charts on a single axis, diverging and emphasis colours, a reference line, annotations, hover text and recession shading.

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
