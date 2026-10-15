# Panel 4 · Portfolio Scenario Lab

**Question:** what could this mean for an investor's portfolio?

Panels 1 to 3 look back at inflation, interest rates and markets. This panel looks forward. It simulates thousands of possible futures for a portfolio of US stocks, US bonds, gold and bitcoin with a **Monte Carlo model**, then shows the range of outcomes. The user sets the weights, picks a what-if scenario (inflation shock, stock market crash, crypto winter, goldilocks), chooses how much history the model learns from, and sees how the median outcome and the downside change. With the data snapshot in the repo, the default view (50/30/10/10, Historical scenario, last 5 years of data, 10 years ahead) turns $10,000 into a median of about $28,600 (+11.1% a year). The 5% worst case is about $14,500, and 0.9% of simulations lose money. Switching to "Inflation shock" cuts the median to about $20,400 and raises the chance of a loss to about 9%.

Code: `dashboard/panels/portfolio.py` · Tests: `tests/test_portfolio.py` · Data snapshot: `data/portfolio_prices.csv`

## Data

Daily adjusted close prices from Yahoo Finance (dividends and splits included, so price changes are total returns), downloaded from January 2010 through the shared helper `load_prices("portfolio_prices", ...)`:

| Ticker | What it is | Role in the portfolio |
|---|---|---|
| [SPY](https://finance.yahoo.com/quote/SPY) | SPDR S&P 500 ETF | US stocks |
| [AGG](https://finance.yahoo.com/quote/AGG) | iShares Core US Aggregate Bond ETF | US bonds |
| [GLD](https://finance.yahoo.com/quote/GLD) | SPDR Gold Shares | Gold |
| [BTC-USD](https://finance.yahoo.com/quote/BTC-USD) | Bitcoin in US dollars | Crypto |

Bitcoin trades 7 days a week and its Yahoo data starts in September 2014. The ETFs trade only on weekdays. `month_end_prices` therefore uses `resample("ME").last()`, which takes each asset's last available price in every month, so a weekend month-end is handled correctly. The newest month is dropped if the data stops mid-month, because a half-month return is not a monthly return. `monthly_log_returns` then keeps only months where **all four** assets have a return, which means October 2014 onwards.

## Method

1. **Monthly log returns:** r = ln(P_t / P_{t−1}). Log returns add up over time, which makes them a natural fit for a normal distribution.
2. **Estimation window:** the last 5 years (default), the last 10 years, or all data since October 2014. From it the model estimates the **mean vector** μ (average monthly log return of each asset) and the **covariance matrix** Σ (volatilities and how the assets move together).
3. **Scenario:** each scenario in the `SCENARIOS` dict gives, per asset, a drift shift in percentage points a year, a volatility multiplier and an optional one-off shock in month 1. These are illustrative assumptions, not forecasts:
   - μ_new = μ + drift / 100 / 12 (yearly percentage points → monthly decimal)
   - Σ_new[i, j] = Σ[i, j] × m_i × m_j (volatility of asset i × m_i; correlations stay the same)
   - The drift shift moves each asset's typical (median) growth. The multiplier widens or narrows the spread around it.
4. **Simulation:** for each month and for every simulation at once, draw correlated log returns with `np.random.default_rng(seed).multivariate_normal(μ, Σ)`. The portfolio is **rebalanced to its target weights every month**:

       V_t = V_{t−1} × Σ_i w_i × exp(r_i,t)

   In month 1 each asset's growth is also multiplied by (1 + shock_i). The output is an array of shape (simulations, months + 1).
5. **Results:** percentiles 5/25/50/75/95 at every month (the fan), the median final value, the 5th percentile ("5% worst case": 1 in 20 simulations ends lower), the share of final values below the starting amount, and the median annualised growth rate (final / start)^(1/years) − 1.

Worked checks used in the tests: one asset growing exactly 1% a month with zero volatility gives 1000 × 1.01¹² = 1,126.83 after a year. A 60/40 mix of an asset growing 2% (log) a month and a flat one grows by 0.6 × e^0.02 + 0.4 = 1.01212 each month. A −50% shock to half the portfolio leaves 0.5 × 0.5 + 0.5 = 75%.

The seed starts at 42, so the numbers are identical on every rerun. Comparisons between scenarios also use the same random draws, so the differences come from the assumptions, not from luck. "New random draw" moves to the next seed.

## Controls

| Control | Effect |
|---|---|
| Stocks / Bonds / Gold / Bitcoin (0–100%, step 5) | Target weights. If they don't add up to 100% they are rescaled, and the caption says so. All zero shows a message instead of a chart |
| Scenario | Historical, Inflation shock, Stock market crash, Crypto winter, Goldilocks (assumptions shown in the caption and the Method table) |
| Estimated from | Estimation window: last 5 years, last 10 years, or all data since Oct 2014 |
| Horizon (years) | 1 to 30 years ahead (default 10) |
| Invested ($) | Starting amount (default $10,000) |
| View | Fan chart (percentile bands over time), Histogram (final values, top 1% cut off), Correlations (4 × 4 heatmap of the window's monthly return correlations) |
| Simulations | 500, 1,000, 2,000 (default) or 5,000 simulated futures |
| New random draw | Re-runs with a new random seed (stored in `st.session_state`) |

KPIs: median value after N years (delta = % vs start), chance of a loss, and the 5% worst case (delta = % vs start). The insight sentence is built from the current results. The first part gives the portfolio mix, the scenario, the median outcome and its yearly growth rate. The second part gives the downside, with separate wording for "no simulation loses money", "even the worst case gains" and "the worst case loses X%". In the Correlations view it names the most and least correlated pairs instead.

## Code walkthrough

| Function | What it does |
|---|---|
| `load_data(refresh)` / `refresh_data()` | Read the CSV snapshot, or download fresh prices from Yahoo Finance |
| `month_end_prices`, `monthly_log_returns` | Daily prices → month-end prices → monthly log returns for complete months only |
| `estimation_window(returns, years)` | Keeps the last *n* years of months (`None` = all) |
| `estimate_inputs(returns)` | Mean vector and covariance matrix (`DataFrame.mean`, `.cov`) |
| `apply_scenario`, `scenario_shocks` | Scenario maths: drift / 12, `np.outer` of the volatility multipliers, shocks as decimals |
| `normalise_weights(raw)` | Weights → fractions that add to 1, or `None` if all are zero |
| `simulate_portfolio(...)` | The Monte Carlo loop over months, vectorised over simulations |
| `percentile_paths`, `outcome_stats`, `histogram_table` | Summaries for the fan chart, KPIs and histogram |
| `extreme_correlations`, `describe_*`, `write_insight` | Nested loop over asset pairs; if/elif branches that build the insight text |
| `format_money`, `format_chance`, `years_label`, `escape_dollars` | Small formatting helpers (`$` must be escaped because Streamlit reads `$...$` as maths) |
| `build_fan_chart`, `build_histogram`, `build_correlation_heatmap` | Plotly figures: fill-between bands, labelled reference lines, annotated heatmap |
| `_draw_controls`, `_draw_kpis`, `_method_markdown`, `render` | Streamlit layout. The button uses an `on_click` callback to change the seed |

## Course concepts used

- **S3 data structures:** dicts for assets, weights, scenarios (a dict of dicts) and results; lists of percentiles and options; tuples for (asset, asset, correlation).
- **S4 control flow:** the month-by-month simulation loop; nested loops over heatmap cells and asset pairs; if/elif for the insight wording and edge cases (all-zero weights, too little data).
- **S5 functions:** short functions with docstrings and type hints; pure model functions separate from the Streamlit `render()`.
- **S6 NumPy, CSV, yfinance:** `default_rng`, `multivariate_normal`, `exp`, `@` (weighted sum), `percentile`, `median`, `outer`, `histogram`; arrays of shape (simulations, months); CSV snapshots; yfinance download.
- **S7 and S9 pandas:** `resample("ME").last()`, `shift`, `dropna`, date filtering with boolean masks, `mean`, `cov`, `corr`, `rename`, `date_range`.
- **S10 visualisation:** fan chart, histogram and heatmap in Plotly, with one y-axis, theme colours, legends, hover text and direct labels.

## Limitations

- Means estimated from 5 to 12 years of data are very uncertain. The standard error of a yearly stock return measured over 5 years is about 7 percentage points. Bitcoin's average ranges from about 13% a year (last 5 years) to about 49% (last 10 years), so "Historical" with a long window is very optimistic.
- Returns are drawn from a normal distribution with constant volatility and correlations. Real markets have fat tails (crashes are more common than the model expects), volatility clustering, and correlations that change. The stock–bond correlation turned positive in 2022, for example.
- Scenario numbers are illustrative assumptions chosen for teaching, not forecasts.
- Values are nominal (not adjusted for inflation) and ignore fees, taxes and trading costs. Monthly rebalancing is assumed to be free.

## Extension ideas

1. Add an "inflation-adjusted" toggle that divides values by projected CPI (panel 1's data) to show real purchasing power.
2. Add monthly contributions (e.g. $200 a month), a common question for savers.
3. Use bootstrapping (re-sampling real historical months with `rng.choice`) instead of the normal distribution, so crashes keep their real size, and compare the 5% worst case with the current model.

## Run and test

```bash
python -m pytest -q tests/test_portfolio.py                   # unit tests + headless app tests (no internet)
streamlit run scripts/preview_panel.py -- portfolio           # this panel on its own
streamlit run app.py                                          # the whole dashboard
python -c "from dashboard.panels import portfolio; portfolio.refresh_data()"   # download fresh prices
```
