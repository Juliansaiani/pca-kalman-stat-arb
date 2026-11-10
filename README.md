# PCA and Kalman statistical arbitrage

An equity residual-reversal study comparing rolling PCA, Kalman-filtered exposures and inverse-volatility position sizing. It uses 60 stocks in four separate universes, with adjusted daily prices downloaded directly from Yahoo Finance.

![Strategy comparison](artifacts/overview.png)

## Results

Annualized Sharpe after 10 bps per unit of turnover. The full evaluation starts on 9 July 2018; the final reporting period is calendar year 2025.

| Universe | PCA, full period | Calibrated + risk, full period | PCA, 2025 | Calibrated + risk, 2025 |
|---|---:|---:|---:|---:|
| Technology | -0.481 | -0.577 | -1.760 | -1.975 |
| Financials | -0.050 | -0.089 | 1.110 | 0.760 |
| Energy | -0.120 | -0.171 | -0.740 | -0.711 |
| UK equities | 0.369 | 0.214 | -0.739 | -0.990 |

There is no consistent net edge across the four universes. UK equities are positive over the full sample but lose money in 2025. Calibration and risk sizing do not provide a general improvement. The notebook reports all five variants, including each addition separately, with drawdowns and costs from 0 to 20 bps.

PCA uses 126 past trading days and refits every 10 days. Kalman calibration scores 21 combinations of process and observation noise on past predictive likelihood: 84 days fit the factor basis and 42 days validate the filter. No trading profit is used in that choice. Each new PCA basis starts a fresh filter.

The signal ranks five-day residual pressure scaled by 20-day residual volatility. Three stocks are bought and three shorted every 20 days, with a one-observation execution lag. Risk sizing uses past 20-day stock volatility, 50% per side and a 25% target weight cap. Holdings drift between rebalances; costs include the resulting rebalance turnover.

The reporting split is pre-2023 development, 2023–2024 validation and a 2025 final historical test. Settings were fixed before this implementation evaluated 2025, but overlapping market histories had already been explored. This is not a pristine research holdout. The universes are hand-picked rather than point-in-time, and borrow fees, financing and market impact are omitted. Neutrality and position caps apply at rebalances, not continuously.

## Run

The recorded results used Python 3.14.7 on Windows.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[dev]'
python -m equity_stat_arb.pipeline
python -m pytest -q
```

The first run downloads 2017–2025 prices through `yfinance`; later runs use the local snapshot. Use `--refresh-data` to replace that snapshot deliberately. A refresh may change results if the provider revises adjusted prices. The source request, download time, file hashes and software versions are recorded in `artifacts/run.json`.

Start with [the executed notebook](notebooks/research.ipynb). Reusable code is in `src/equity_stat_arb/`, experiment settings in [config.json](config.json), and metrics, calibration choices and figures in `artifacts/`. Raw prices and environments are excluded from Git. Tests check causality, PCA recovery, batched filtering, capped allocations and holdings accounting.

Code: [MIT](LICENSE). Market data comes from [Yahoo Finance](https://finance.yahoo.com/) and has separate usage terms.

Also: [Order-book volatility forecasting](https://github.com/julian5422/order-book-volatility-forecasting).
