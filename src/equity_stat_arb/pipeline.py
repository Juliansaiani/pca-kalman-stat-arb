"""Reproduce the archived experiment with an independently implemented model."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_returns
from .models import fit_residual_models
from .portfolio import decision_weights, simulate


def score(returns):
    values = np.asarray(returns, dtype=float)
    if not len(values):
        return {'days': 0, 'sharpe': np.nan, 'cagr': np.nan, 'max_drawdown': np.nan}
    wealth = np.r_[1.0, np.cumprod(1 + values)]
    volatility = values.std(ddof=1) if len(values) > 1 else 0
    return {'days': len(values),
            'sharpe': values.mean() / volatility * np.sqrt(252) if volatility else np.nan,
            'cagr': wealth[-1] ** (252 / len(values)) - 1,
            'max_drawdown': (wealth / np.maximum.accumulate(wealth) - 1).min()}


def analyze(returns, config):
    residuals, diagnostics = fit_residual_models(returns, config['model'])
    aligned = returns.loc[residuals['PCA'].index]
    choices = {'PCA / equal': ('PCA', False), 'Fixed Kalman / equal': ('Fixed Kalman', False),
               'Calibrated Kalman / equal': ('Calibrated Kalman', False),
               'Fixed Kalman / inverse vol': ('Fixed Kalman', True),
               'Calibrated Kalman / inverse vol': ('Calibrated Kalman', True)}
    summaries, daily_frames, holdings = [], [], []
    for variant, (model, risk_sizing) in choices.items():
        orders = decision_weights(residuals[model], aligned, config['portfolio'], risk_sizing)
        for cost in config['cost_bps']:
            result, held = simulate(aligned, orders, cost)
            summaries.append({'variant': variant, 'cost_bps': cost, **score(result.net_return),
                              'trading_sample_sharpe': score(result.net_return.iloc[config['portfolio']['residual_history'] + 1:])['sharpe'],
                              'average_daily_turnover': result.turnover.mean(),
                              'max_weight': held.abs().to_numpy().max(),
                              'max_absolute_net_exposure': result.net_exposure.abs().max()})
            daily_frames.append(result.rename_axis('date').reset_index().assign(variant=variant, cost_bps=cost))
            if cost == 10:
                holdings.append(held.rename_axis('date').reset_index().assign(variant=variant))
    return (pd.DataFrame(summaries), pd.concat(daily_frames, ignore_index=True), diagnostics,
            pd.concat(holdings, ignore_index=True))


def verify_reference(actual, expected, tolerance=1e-9):
    keys = ['universe', 'variant', 'cost_bps']
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError('Reference comparison keys must be unique')
    compared = actual.merge(expected, on=keys, how='outer', suffixes=('_new', '_reference'), indicator=True)
    if not compared['_merge'].eq('both').all():
        raise AssertionError('Reference and computed result rows differ')
    columns = ['days', 'sharpe', 'cagr', 'max_drawdown', 'trading_sample_sharpe',
               'average_daily_turnover', 'max_weight', 'max_absolute_net_exposure']
    report = compared[keys].copy()
    for metric in columns:
        error = (compared[f'{metric}_new'] - compared[f'{metric}_reference']).abs()
        report[f'{metric}_absolute_error'] = error
        if not np.isfinite(error).all() or error.max() > tolerance:
            raise AssertionError(f'Reference mismatch: {metric}, largest error={error.max()}')
    return report


def figures(daily, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    selected = daily[daily.cost_bps == 10]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, universe in zip(axes.flat, selected.universe.unique()):
        for variant, group in selected[selected.universe == universe].groupby('variant', sort=False):
            ax.plot(pd.to_datetime(group.date), (1 + group.net_return).cumprod(), label=variant, lw=1.3)
        ax.set_title(universe.replace('_', ' ').title())
        ax.set_ylabel('Growth of 1')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=6.5)
    fig.suptitle('Frozen-sample residual reversal | 10 bps per unit turnover', fontsize=15)
    fig.tight_layout()
    fig.savefig(output / 'overview.png', dpi=150)
    plt.close(fig)


def run(root, verify=False):
    root = Path(root).resolve()
    settings_path = root / 'config.json'
    config = json.loads(settings_path.read_text(encoding='utf-8'))
    panels, sources = load_returns(root, config['data'])
    scores, frames, calibration, holdings = [], [], [], []
    for universe, observations in panels.items():
        print(f'Evaluating {universe}', flush=True)
        summary, daily, diagnostics, held = analyze(observations, config)
        for frame in (summary, daily, diagnostics, held):
            frame.insert(0, 'universe', universe)
        scores.append(summary); frames.append(daily); calibration.append(diagnostics); holdings.append(held)
    summary = pd.concat(scores, ignore_index=True)
    daily = pd.concat(frames, ignore_index=True)
    comparison = None
    if verify:
        reference = pd.read_csv(root / config['verification']['reference'])
        comparison = verify_reference(summary, reference, config['verification']['absolute_tolerance'])
        print(f'Reference verified: {len(comparison)} rows, 8 measures per row.', flush=True)
    output = root / 'artifacts'
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / 'metrics.csv', index=False)
    pd.concat(calibration, ignore_index=True).to_csv(output / 'calibration.csv', index=False)
    daily.to_parquet(output / 'daily_returns.parquet', index=False)
    pd.concat(holdings, ignore_index=True).to_parquet(output / 'holdings.parquet', index=False)
    if comparison is not None:
        comparison.to_csv(output / 'verification.csv', index=False)
    figures(daily, output)
    manifest = {'config': config, 'config_sha256': hashlib.sha256(settings_path.read_bytes()).hexdigest(),
                'python': platform.python_version(),
                'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'matplotlib', 'pyarrow')},
                'data': sources, 'reference_verified': comparison is not None,
                'evaluation': 'Previously inspected historical panels; no new holdout. Panel periods differ.',
                'accounting': 'Constant daily target weights with one-day lag. Costs on target-weight changes only; holding drift and maintenance turnover are omitted.',
                'limitations': ['Not a self-financing execution simulation',
                                'Stock-selection and survivorship bias possible',
                                'No borrow fees, financing, short availability or impact',
                                'Dollar neutral, not factor neutral',
                                'Headline metrics include the archived initial flat period']}
    (output / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(summary[summary.cost_bps == 10][['universe', 'variant', 'sharpe', 'cagr', 'max_drawdown']].round(6).to_string(index=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--verify', action='store_true', help='Fail if any frozen reference measure differs')
    args = parser.parse_args()
    run(args.root, args.verify)


if __name__ == '__main__':
    main()
