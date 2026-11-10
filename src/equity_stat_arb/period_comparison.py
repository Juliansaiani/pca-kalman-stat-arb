"""Matched-calendar historical study, plotted only while positions are active."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import analyze, score


def price_history(root, request):
    import yfinance as yf
    directory = root / 'data/period_comparison'
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    saved = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    query = {k: request[k] for k in ('download_start', 'download_end', 'universes')}
    if saved and saved['request'] != query:
        raise ValueError('Price request changed; use a new snapshot directory')
    panels, sources = {}, []
    for universe, tickers in request['universes'].items():
        path = directory / f'{universe}.csv'
        record = next((r for r in saved['files'] if r['universe'] == universe), None) if saved else None
        if record and path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError(f'Price checksum mismatch: {universe}')
            prices = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            print(f'Downloading matched-period prices: {universe}', flush=True)
            raw = yf.download(tickers, start=request['download_start'], end=request['download_end'],
                              auto_adjust=True, progress=False, threads=2, timeout=30)
            prices = raw['Close'].reindex(columns=tickers).sort_index()
            missing = prices.columns[prices.notna().mean() < .95].tolist()
            if missing:
                raise ValueError(f'Incomplete ticker histories: {missing}')
            prices = prices.dropna(how='any')
            if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
                raise ValueError(f'Invalid prices: {universe}')
            prices.to_csv(path, index_label='Date')
            prices = pd.read_csv(path, index_col=0, parse_dates=True)
            record = {'universe': universe, 'tickers': tickers,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'first_date': str(prices.index[0].date()), 'last_date': str(prices.index[-1].date()),
                      'observations': len(prices)}
        panels[universe] = prices
        sources.append(record)
        manifest_path.write_text(json.dumps({'provider': 'Yahoo Finance through yfinance',
                                             'auto_adjust': True, 'request': query, 'files': sources}, indent=2), encoding='utf-8')
    return panels, sources


def align_active_period(daily):
    """Choose shared calendar bounds after every universe has started trading."""
    at_10 = daily[daily.cost_bps == 10]
    active = at_10[at_10.gross_exposure > 0]
    first = active.groupby('universe').date.min()
    if len(first) != daily.universe.nunique():
        raise ValueError('Some universes never trade')
    latest_start = first.max()
    common_dates = None
    for _, group in at_10.groupby('universe'):
        dates = pd.DatetimeIndex(group.date.unique())
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    common_dates = common_dates[common_dates >= latest_start].sort_values()
    if common_dates.empty:
        raise ValueError('No shared active calendar interval')
    start, end = common_dates[0], common_dates[-1]
    cropped = daily[daily.date.between(start, end)].copy()
    # The same calendar bounds must occur in every plotted series.
    bounds = cropped.groupby(['universe', 'variant', 'cost_bps']).date.agg(['min', 'max'])
    if not bounds['min'].eq(start).all() or not bounds['max'].eq(end).all():
        raise AssertionError('Evaluation bounds differ between series')
    return cropped, start, end


def chart(daily, output, start, end):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    selected = daily[daily.cost_bps == 10].copy()
    labels = {'PCA / equal': 'PCA, equal weights',
              'Fixed Kalman / equal': 'Fixed Kalman, equal weights',
              'Calibrated Kalman / equal': 'Calibrated Kalman, equal weights',
              'Fixed Kalman / inverse vol': 'Fixed Kalman, risk sizing',
              'Calibrated Kalman / inverse vol': 'Calibrated Kalman, risk sizing'}
    wealth = selected.groupby(['universe', 'variant'], sort=False).net_return.transform(lambda s: (1 + s).cumprod())
    selected = selected.assign(wealth=wealth)
    low, high = min(1, selected.wealth.min()), max(1, selected.wealth.max())
    padding = max((high - low) * .06, .01)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, universe in zip(axes.flat, daily.universe.unique()):
        for variant, group in selected[selected.universe == universe].groupby('variant', sort=False):
            ax.plot(group.date, group.wealth, label=labels[variant], lw=1.6)
        ax.set_title({'financial': 'Financials', 'uk_external': 'UK equities'}.get(
            universe, universe.replace('_', ' ').title()))
        ax.set_ylabel('Growth of 1')
        ax.set_xlim(start, end)
        ax.set_ylim(low - padding, high + padding)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f'Active trading only | {start.date()} to {end.date()} | 10 bps', fontsize=14)
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run(root):
    root = Path(root).resolve()
    config = json.loads((root / 'config.json').read_text())
    request = config['period_comparison']
    prices, sources = price_history(root, request)
    rows, histories, periods = [], [], []
    output = root / 'artifacts/period_comparison'
    output.mkdir(parents=True, exist_ok=True)
    for sample, dates in request['samples'].items():
        sample_days = []
        for universe, frame in prices.items():
            # Restart each experiment with the same warmup rules in its own sample.
            observations = frame.loc[dates['start']:dates['end']].pct_change(fill_method=None).dropna()
            print(f'Evaluating {sample}: {universe}', flush=True)
            _, daily, _, _ = analyze(observations, config)
            daily.insert(0, 'universe', universe)
            sample_days.append(daily)
        active, start, end = align_active_period(pd.concat(sample_days, ignore_index=True))
        for (universe, variant, cost), group in active.groupby(['universe', 'variant', 'cost_bps'], sort=False):
            rows.append({'sample': sample, 'universe': universe, 'variant': variant, 'cost_bps': cost,
                         'start': str(start.date()), 'end': str(end.date()), **score(group.net_return),
                         'average_daily_turnover': group.turnover.mean()})
        periods.append({'sample': sample, 'start': str(start.date()), 'end': str(end.date()),
                        'zero_exposure_rows': int(active.gross_exposure.eq(0).sum())})
        chart(active, output / f'{sample}.png', start, end)
        histories.append(active.assign(sample=sample))
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'metrics.csv', index=False)
    pd.concat(histories, ignore_index=True).to_parquet(output / 'daily_returns.parquet', index=False)
    manifest = {'samples': periods, 'prices': sources, 'model': config['model'],
                'portfolio': config['portfolio'], 'cost_bps': config['cost_bps'],
                'python': platform.python_version(),
                'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'yfinance')},
                'price_request': {key: request[key] for key in ('download_start', 'download_end', 'universes')},
                'evaluation': 'Historical sample retained after comparing periods; not independent validation or evidence of stability across regimes.',
                'accounting': config['accounting'], 'source': 'Yahoo Finance through yfinance'}
    (output / 'run.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(summary[(summary.cost_bps == 10) & summary.variant.isin([
        'Fixed Kalman / equal', 'Calibrated Kalman / inverse vol'])][[
            'sample', 'universe', 'variant', 'sharpe', 'cagr', 'max_drawdown']].round(4).to_string(index=False))
    return manifest


if __name__ == '__main__':
    run(Path.cwd())
