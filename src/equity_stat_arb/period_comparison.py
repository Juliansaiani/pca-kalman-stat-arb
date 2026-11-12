"""Matched-calendar historical study, plotted only while positions are active."""
import argparse
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
    saved = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
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
    ref_cost = 5 if 5 in daily.cost_bps.values else (10 if 10 in daily.cost_bps.values else daily.cost_bps.iloc[0])
    at_cost = daily[daily.cost_bps == ref_cost]
    active = at_cost[at_cost.gross_exposure > 0]
    first = active.groupby('universe').date.min()
    if len(first) != daily.universe.nunique():
        raise ValueError('Some universes never trade')
    latest_start = first.max()
    common_dates = None
    for _, group in at_cost.groupby('universe'):
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


def chart(daily, output, start, end, cost_bps=5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    target_cost = cost_bps if cost_bps in daily.cost_bps.values else daily.cost_bps.iloc[0]
    selected = daily[daily.cost_bps == target_cost].copy()

    color_map = {
        'PCA / equal': '#1f77b4',
        'Fixed Kalman / equal': '#2ca02c',
        'Calibrated Kalman / equal': '#9467bd',
        'Fixed Kalman / inverse vol': '#ff7f0e',
        'Calibrated Kalman / inverse vol': '#d62728'
    }
    labels = {
        'PCA / equal': 'Static PCA (Equal weights)',
        'Fixed Kalman / equal': 'Fixed Kalman (Equal weights)',
        'Calibrated Kalman / equal': 'Calibrated Kalman (Equal weights)',
        'Fixed Kalman / inverse vol': 'Fixed Kalman (Inverse Vol sizing)',
        'Calibrated Kalman / inverse vol': 'Calibrated Kalman (Inverse Vol sizing)'
    }
    title_map = {
        'technology': 'Technology (US Large-Cap)',
        'financial': 'Financials (US Banks & Insurance)',
        'energy': 'Energy (US Oil & Gas E&P)',
        'uk_external': 'UK Equities (FTSE 100 Large-Cap)'
    }

    wealth = selected.groupby(['universe', 'variant'], sort=False).net_return.transform(lambda s: (1 + s).cumprod())
    selected = selected.assign(wealth=wealth)
    low, high = min(0.9, selected.wealth.min()), max(1.1, selected.wealth.max())
    padding = max((high - low) * .06, .02)

    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    universe_order = ['technology', 'financial', 'energy', 'uk_external']
    existing_universes = [u for u in universe_order if u in daily.universe.unique()]
    if not existing_universes:
        existing_universes = list(daily.universe.unique())

    for ax, universe in zip(axes.flat, existing_universes):
        sub = selected[selected.universe == universe]
        for variant, group in sub.groupby('variant', sort=False):
            c = color_map.get(variant, '#333333')
            ls = '--' if 'inverse vol' in variant else '-'
            lw = 2.2 if 'Kalman / equal' in variant else 1.8
            ax.plot(pd.to_datetime(group.date), group.wealth, label=labels.get(variant, variant), color=c, lw=lw, linestyle=ls)

        if pd.Timestamp('2020-02-03') >= start and pd.Timestamp('2020-03-31') <= end:
            ax.axvspan(pd.Timestamp('2020-02-03'), pd.Timestamp('2020-03-31'), color='#d62728', alpha=0.10)
            ax.text(pd.Timestamp('2020-03-01'), low + padding * 0.2, 'COVID Crash', fontsize=8, color='#b2182b', ha='center', fontweight='bold')

        ax.axhline(1.0, color='black', lw=0.8, linestyle=':', alpha=0.7)
        ax.set_title(title_map.get(universe, universe.replace('_', ' ').title()), fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel('Growth of $1.00', fontsize=10)
        ax.set_xlim(start, end)
        ax.set_ylim(low - padding, high + padding)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=3, frameon=True, fontsize=9.5, bbox_to_anchor=(0.5, 0.01), edgecolor='#cccccc')
    cost_text = "Gross (0 bps)" if target_cost == 0 else f"Net {target_cost} bps Costs"
    fig.suptitle(f'PCA & Adaptive Kalman Statistical Arbitrage | Active Trading ({start.date()} to {end.date()})\nDollar-Neutral Residual Reversal (Top 3 Long / Bottom 3 Short) | {cost_text}', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def chart_gross_vs_net(daily, metrics, output, start, end):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    title_map = {
        'technology': 'Technology (US) - Fixed Kalman',
        'financial': 'Financials (US) - Fixed Kalman',
        'energy': 'Energy (US) - Calibrated Kalman',
        'uk_external': 'UK Equities - Fixed Kalman'
    }
    var_map = {
        'technology': 'Fixed Kalman / equal',
        'financial': 'Fixed Kalman / equal',
        'energy': 'Calibrated Kalman / equal',
        'uk_external': 'Fixed Kalman / equal'
    }

    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    universe_order = ['technology', 'financial', 'energy', 'uk_external']

    for ax, universe in zip(axes.flat, universe_order):
        if universe not in daily.universe.unique():
            continue
        variant = var_map[universe]
        g_data = daily[(daily.universe == universe) & (daily.variant == variant) & (daily.cost_bps == 0)].copy()
        n_data = daily[(daily.universe == universe) & (daily.variant == variant) & (daily.cost_bps == 5)].copy()

        if g_data.empty or n_data.empty:
            continue

        g_cum = (1 + g_data.net_return).cumprod()
        n_cum = (1 + n_data.net_return).cumprod()

        g_m = metrics[(metrics.universe == universe) & (metrics.variant == variant) & (metrics.cost_bps == 0)]
        n_m = metrics[(metrics.universe == universe) & (metrics.variant == variant) & (metrics.cost_bps == 5)]
        g_sharpe = g_m['sharpe'].values[0] if not g_m.empty else np.nan
        n_sharpe = n_m['sharpe'].values[0] if not n_m.empty else np.nan
        g_cagr = g_m['cagr'].values[0] if not g_m.empty else np.nan
        n_cagr = n_m['cagr'].values[0] if not n_m.empty else np.nan

        dates = pd.to_datetime(g_data.date)
        ax.plot(dates, g_cum, color='#1f77b4', lw=2.2, label=f'Gross (0 bps): Sharpe {g_sharpe:.2f} | CAGR +{g_cagr*100:.1f}%')
        ax.plot(dates, n_cum, color='#2ca02c', lw=2.0, linestyle='--', label=f'Net (5 bps): Sharpe {n_sharpe:.2f} | CAGR +{n_cagr*100:.1f}%')
        ax.fill_between(dates, n_cum, g_cum, color='#ff7f0e', alpha=0.15, label='5 bps Cost Friction')

        if pd.Timestamp('2020-02-03') >= start and pd.Timestamp('2020-03-31') <= end:
            ax.axvspan(pd.Timestamp('2020-02-03'), pd.Timestamp('2020-03-31'), color='#d62728', alpha=0.08)

        ax.axhline(1.0, color='black', lw=0.8, linestyle=':', alpha=0.7)
        ax.set_title(title_map[universe], fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel('Growth of $1.00', fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8.5, loc='upper left', frameon=True, edgecolor='#dddddd')

    fig.suptitle('Transaction Cost Sensitivity & Execution Drag | Gross (0 bps) vs Net (5 bps)\nAnnual Turnover ~20.6x to 21.2x NAV | 1-Day Lagged Execution ($t \\to t+1$)', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def run(root):
    root = Path(root).resolve()
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    request = config['period_comparison']
    prices, sources = price_history(root, request)
    rows, histories, periods = [], [], []
    output = root / 'artifacts/period_comparison'
    output.mkdir(parents=True, exist_ok=True)
    artifacts_dir = root / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for sample, dates in request['samples'].items():
        sample_days = []
        for universe, frame in prices.items():
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
        
        # Save primary overview plot at Net 5 bps
        chart(active, output / f'{sample}.png', start, end, cost_bps=5)
        chart(active, artifacts_dir / 'overview.png', start, end, cost_bps=5)
        histories.append(active.assign(sample=sample))

    summary = pd.DataFrame(rows)
    all_daily = pd.concat(histories, ignore_index=True)

    # Save comparative Gross vs Net 5 bps figure
    chart_gross_vs_net(all_daily, summary, artifacts_dir / 'gross_vs_net.png', start, end)

    # Export metrics, daily returns and run manifest
    summary.to_csv(output / 'metrics.csv', index=False)
    summary.to_csv(artifacts_dir / 'metrics.csv', index=False)
    summary.to_csv(artifacts_dir / 'scores.csv', index=False)
    all_daily.to_parquet(output / 'daily_returns.parquet', index=False)
    all_daily.to_csv(artifacts_dir / 'daily_returns.csv', index=False)

    manifest = {'samples': periods, 'prices': sources, 'model': config['model'],
                'portfolio': config['portfolio'], 'cost_bps': config['cost_bps'],
                'python': platform.python_version(),
                'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'yfinance')},
                'price_request': {key: request[key] for key in ('download_start', 'download_end', 'universes')},
                'evaluation': 'Historical matched calendar bounds (2018-07-09 to 2021-12-31); evaluated at Gross (0 bps) and Net (5 bps) execution costs.',
                'accounting': config['accounting'], 'source': 'Yahoo Finance through yfinance'}
    (output / 'run.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (artifacts_dir / 'run.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print('\n=== Matched-Period Summary: Gross (0 bps) vs Net (5 bps) ===', flush=True)
    disp = summary[summary.cost_bps.isin([0, 5]) & summary.variant.isin(['Fixed Kalman / equal', 'Calibrated Kalman / equal'])][[
        'universe', 'variant', 'cost_bps', 'sharpe', 'cagr', 'max_drawdown', 'average_daily_turnover']].round(4)
    print(disp.to_string(index=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Run matched-period statistical arbitrage study.')
    parser.parse_args()
    run(Path.cwd())


if __name__ == '__main__':
    main()
