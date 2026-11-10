# PCA and Kalman statistical arbitrage

A residual-reversal study comparing rolling PCA, Kalman-filtered exposures and inverse-volatility sizing across four 15-stock universes. This implementation reproduces a frozen historical experiment with the same inputs, parameters and accounting conventions.

![Strategy comparison](artifacts/overview.png)

## Results

Historical annualized Sharpe at 10 bps per unit of target-weight turnover:

| Universe | Fixed Kalman / equal weights | Calibrated Kalman / risk sizing |
|---|---:|---:|
| Technology | 0.458 | 1.116 |
| Financials | 0.626 | 0.458 |
| Energy | 0.231 | 0.367 |
| UK equities | 0.707 | 0.617 |

Calibration and risk sizing improve technology and energy in this sample, but weaken financials and UK equities. All five variants and 0–20 bps costs are reported in the notebook. Technology covers 2022–2026; the other panels cover 2017–2021, so comparisons are within each universe rather than matched-period sector comparisons.

PCA uses the preceding 126 trading days, refits every 10 days and keeps enough components to explain 55% of training variance. Calibration compares three process-noise scales by sequential predictive likelihood: 84 past days fit PCA and observation noise, then 42 past days score the candidates. The filter restarts at each new PCA basis.

The signal ranks five-day residual pressure scaled by 20-day residual volatility. It selects three longs and three shorts every 20 days. Risk sizing uses preceding 20-day stock volatility, 50% per side and a 25% target position cap. Positions are lagged one day before calculating returns.

**Accounting is simplified:** daily target weights are treated as constant until the next signal rebalance. Costs cover changes in those targets; holding drift and the trades required to maintain weights are omitted. Headline statistics include the archived initial flat period; trading-period Sharpe is shown separately. These are previously inspected histories, not a new untouched holdout. Borrow fees, financing, short availability and market impact are excluded. Dollar neutrality does not imply factor neutrality.

## Run

The recorded results used Python 3.14.7 on Windows.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[dev]'
python -m equity_stat_arb.pipeline --verify
python -m pytest -q
```

The first run downloads four checksummed return panels; later runs use the cache. The implementation is independent; the input snapshots are preserved from [this pinned dataset source](https://github.com/b-enh/pca-stat-arb/tree/6a24304205318de080b019e74ac766e7cdb18ff6), rather than replaced with a fresh price download. Source URLs, dates, hashes and software versions are recorded in `artifacts/run.json`.

`--verify` checks all 80 reference rows across eight measures at an absolute tolerance of 1e-9, before exporting results. The notebook and GitHub checks run this verification. Passing it establishes reproduction of the archived calculation, not profitability or realistic execution.

Start with [the executed notebook](notebooks/research.ipynb). Reusable code is in `src/equity_stat_arb/`, settings in [config.json](config.json), and results in `artifacts/`. Raw data and environments are excluded from Git. Code is released under [MIT](LICENSE); data has separate usage terms.
