# PCA and Kalman statistical arbitrage

A residual-reversal study comparing rolling PCA, Kalman-filtered exposures and inverse-volatility sizing across four 15-stock universes. All panels cover the same active trading period: **9 July 2018 to 31 December 2021**.

![Five strategy variants](artifacts/period_comparison/2017-2021.png)

## Results

Historical annualized Sharpe at 10 bps per unit of target-weight turnover:

| Universe | PCA | Fixed Kalman | Calibrated Kalman | Fixed + risk | Calibrated + risk |
|---|---:|---:|---:|---:|---:|
| Technology | 0.548 | 0.558 | 0.410 | 0.337 | 0.223 |
| Financials | 0.527 | 0.710 | 0.547 | 0.700 | 0.520 |
| Energy | 0.262 | 0.262 | 0.440 | 0.234 | 0.416 |
| UK equities | 0.798 | 0.798 | 0.740 | 0.753 | 0.702 |

Inputs cover 2017–2021. The initial model and signal-history windows are used before the displayed trading period; no inactive warmup is included in the graph or these metrics. All five curves in every panel share the same calendar bounds and vertical scale.

Calibration and risk sizing do not consistently improve the reference: they help energy, but weaken technology, financial and UK results. This historical period was retained after comparing date ranges; it is not independent validation or evidence of stability across market regimes.

PCA uses 126 past trading days, refits every 10 days and retains 55% explained variance. Calibration compares three process-noise scales on past predictive likelihood: 84 fitting days followed by 42 sequential validation days. The filter resets when the PCA basis changes.

The signal ranks five-day residual pressure scaled by 20-day residual volatility. Three stocks per side are selected every 20 days, after 252 residual observations of warmup. Risk sizing uses preceding 20-day stock volatility, 50% per side and a 25% target position cap. Positions are lagged one day before earning returns.

Accounting uses a constant daily target-weight approximation. Costs cover changes in those targets; holding drift and maintenance trades are omitted. Borrow fees, financing, short availability and market impact are excluded. Universes are hand-picked, and dollar neutrality does not imply factor neutrality.

## Run

The recorded results used Python 3.14.7 on Windows.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[dev]'
python -m equity_stat_arb.period_comparison
python -m pytest -q
```

The first run downloads adjusted prices directly from Yahoo Finance through `yfinance`. Later runs check and reuse the local snapshot. The cache can cover a wider history, but only the configured 2017–2021 observations enter this analysis. Fresh downloads may differ if the provider revises its history. Source settings, hashes, versions and evaluation bounds are in `artifacts/period_comparison/run.json`.

Start with [the executed notebook](notebooks/research.ipynb). All five variants and 0–20 bps costs are in `artifacts/period_comparison/metrics.csv`. Code is in `src/equity_stat_arb/`, settings in [config.json](config.json), and raw prices remain outside Git. Code is released under [MIT](LICENSE); market data has separate usage terms.

The earlier mixed-period calculation remains a separate numerical regression check: `python -m equity_stat_arb.pipeline --verify`. Its frozen input sources and assumptions are recorded in `artifacts/run.json`; those archived figures are not the matched-period results above.
