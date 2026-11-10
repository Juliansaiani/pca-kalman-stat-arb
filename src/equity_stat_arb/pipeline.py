"""Rebuild the equity study from the configured price snapshot."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_prices
from .models import fit_residual_models
from .portfolio import decision_weights, simulate


def score(returns):
    values = np.asarray(returns, dtype=float)
    if not len(values):
        return {'days': 0, 'sharpe': np.nan, 'cagr': np.nan, 'max_drawdown': np.nan}
    wealth = np.r_[1.0, np.cumprod(1 + values)]
    standard_deviation = values.std(ddof=1) if len(values) > 1 else 0
    return {'days': len(values),
            'sharpe': values.mean() / standard_deviation * np.sqrt(252) if standard_deviation > 0 else np.nan,
            'cagr': wealth[-1] ** (252 / len(values)) - 1,
            'max_drawdown': (wealth / np.maximum.accumulate(wealth) - 1).min()}


def analyze(returns, config, trading_start=None):
    residuals, diagnostics = fit_residual_models(returns, config['model'])
    portfolio = config['portfolio']
    aligned = returns.loc[next(iter(residuals.values())).index]
    variants = {'PCA / equal': ('PCA', False),
                'Fixed Kalman / equal': ('Fixed Kalman', False),
                'Calibrated Kalman / equal': ('Calibrated Kalman', False),
                'Fixed Kalman / inverse vol': ('Fixed Kalman', True),
                'Calibrated Kalman / inverse vol': ('Calibrated Kalman', True)}
    start = trading_start or aligned.index[portfolio['residual_history'] + 1]
    metrics, daily, holding_frames = [], [], []
    for variant, (model, risk_sizing) in variants.items():
        orders = decision_weights(residuals[model], aligned, portfolio, risk_sizing)
        for cost in config['cost_bps']:
            result, holdings = simulate(aligned, orders, cost)
            result = result.loc[start:]
            periods = {'development': result.index < config['evaluation']['validation_start'],
                       'validation': (result.index >= config['evaluation']['validation_start']) & (result.index < config['evaluation']['holdout_start']),
                       'holdout': result.index >= config['evaluation']['holdout_start'],
                       'all': np.ones(len(result), dtype=bool)}
            for period, mask in periods.items():
                sub = result.loc[mask]
                metrics.append({'variant': variant, 'cost_bps': cost, 'period': period,
                                **score(sub.net_return), 'average_daily_turnover': sub.turnover.mean(),
                                'average_gross_exposure': sub.gross_exposure.mean()})
            daily.append(result.reset_index().rename(columns={result.index.name or 'index': 'date'}).assign(variant=variant, cost_bps=cost))
            if cost == 10:
                holding_frames.append(holdings.loc[start:].reset_index().rename(columns={holdings.index.name or 'index': 'date'}).assign(variant=variant))
    return pd.DataFrame(metrics), pd.concat(daily, ignore_index=True), diagnostics, pd.concat(holding_frames, ignore_index=True)


def figures(daily, output, test_start):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    chosen = daily[daily.cost_bps == 10]
    names = chosen.universe.unique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax, universe in zip(axes.flat, names):
        for variant, group in chosen[chosen.universe == universe].groupby('variant', sort=False):
            wealth = (1 + group.net_return).cumprod()
            ax.plot(pd.to_datetime(group.date), wealth, label=variant, lw=1.2)
        ax.axvline(pd.Timestamp(test_start), color='#777777', lw=.8, ls=':')
        ax.set_title(universe.replace('_', ' ').title())
        ax.set_ylabel('Growth of 1')
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=6.5)
    fig.suptitle('Equity residual reversal | 10 bps per unit turnover', fontsize=15)
    fig.tight_layout()
    fig.savefig(output / 'overview.png', dpi=150)
    plt.close(fig)


def run(root, refresh=False):
    root = Path(root).resolve()
    settings_path = root / 'config.json'
    config = json.loads(settings_path.read_text(encoding='utf-8'))
    prices, snapshot = load_prices(root, config['data'], refresh)
    panels = {name: frame.pct_change(fill_method=None).dropna() for name, frame in prices.items()}
    # Compare the same evaluation dates despite different exchange holidays/history starts.
    warmup = config['model']['lookback'] + config['portfolio']['residual_history'] + 1
    common_start = max(frame.index[warmup] for frame in panels.values())
    rows, days, calibrations, holdings = [], [], [], []
    for name, returns in panels.items():
        print(f'Evaluating {name}', flush=True)
        metrics, daily, diagnostics, held = analyze(returns, config, common_start)
        for frame in (metrics, daily, diagnostics, held):
            frame.insert(0, 'universe', name)
        rows.append(metrics); days.append(daily); calibrations.append(diagnostics); holdings.append(held)
    metrics = pd.concat(rows, ignore_index=True)
    daily = pd.concat(days, ignore_index=True)
    output = root / 'artifacts'
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / 'metrics.csv', index=False)
    pd.concat(calibrations, ignore_index=True).to_csv(output / 'calibration.csv', index=False)
    daily.to_parquet(output / 'daily_returns.parquet', index=False)
    pd.concat(holdings, ignore_index=True).to_parquet(output / 'holdings.parquet', index=False)
    figures(daily, output, config['evaluation']['holdout_start'])
    manifest = {'config': config, 'config_sha256': hashlib.sha256(settings_path.read_bytes()).hexdigest(),
                'python': platform.python_version(),
                'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'matplotlib', 'pyarrow', 'yfinance')},
                'snapshot': snapshot, 'evaluation_start': str(common_start.date()),
                'evaluation': 'Rolling models with past-only Q/R likelihood selection. Specifications fixed before this implementation first evaluated 2025. The historical test overlaps previously explored market periods and is not a pristine research holdout. All five variants reported without choosing a test winner.',
                'limitations': ['Current hand-picked universes, not point-in-time membership',
                                'Daily-close execution proxy, no intraday fills',
                                'No borrow availability, financing or market impact',
                                'Neutrality and caps apply at rebalances only',
                                'One-year holdout is too short to establish a persistent edge']}
    (output / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(metrics[(metrics.period == 'holdout') & (metrics.cost_bps == 10)][['universe', 'variant', 'days', 'sharpe', 'cagr', 'max_drawdown']].round(4).to_string(index=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--refresh-data', action='store_true')
    args = parser.parse_args()
    run(args.root, args.refresh_data)


if __name__ == '__main__':
    main()
