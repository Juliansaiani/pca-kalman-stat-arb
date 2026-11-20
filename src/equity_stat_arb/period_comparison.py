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


def _load_cached_index(path, ticker, name, start='2017-01-01', end='2021-07-02'):
    path = Path(path)
    if path.exists():
        series = pd.read_csv(path, index_col=0, parse_dates=True).squeeze('columns')
        return series.sort_index()
    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, timeout=30)
    close = raw['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    series = close.dropna().sort_index().rename(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame('Close').to_csv(path, index_label='Date')
    return series


def load_index_benchmarks(root, start='2017-01-01', end='2021-07-02'):
    """Load cached FTSE 100 (^FTSE) and S&P 500 (^GSPC) daily adjusted close prices."""
    directory = Path(root) / 'data/period_comparison'
    return {
        'ftse100': _load_cached_index(directory / 'ftse100.csv', '^FTSE', 'FTSE100', start, end),
        'sp500': _load_cached_index(directory / 'sp500.csv', '^GSPC', 'SP500', start, end),
    }


def _capm_stats(strategy_returns, market_returns, annualization=252):
    r = np.asarray(strategy_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float)
    if len(r) < 3 or len(r) != len(m):
        return {'alpha_annualized': np.nan, 'alpha_tstat': np.nan, 'market_beta': np.nan}
    x = np.column_stack([np.ones(len(r)), m])
    coeffs, _, _, _ = np.linalg.lstsq(x, r, rcond=None)
    resid = r - x @ coeffs
    se = np.sqrt(np.diag(np.linalg.inv(x.T @ x) * (resid @ resid / (len(r) - 2))))
    return {
        'alpha_annualized': float(coeffs[0] * annualization),
        'alpha_tstat': float(coeffs[0] / se[0]) if se[0] > 0 else np.nan,
        'market_beta': float(coeffs[1]),
    }


def price_history(root, request):
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
            import yfinance as yf
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


def _variant_specs():
    return (
        [
            'PCA / equal',
            'Fixed Kalman / equal',
            'Fixed Kalman / inverse vol',
            'Calibrated Kalman / equal',
            'Calibrated Kalman / inverse vol',
        ],
        [
            'Rolling PCA\n(Equal)',
            'Fixed Kalman\n(Equal)',
            'Fixed Kalman\n(Risk-Sized)',
            'Calibrated\nKalman (Equal)',
            'Calibrated\nKalman (Risk)',
        ],
    )


def chart_universe_method_comparison(summary, bench_row, universe, uni_label, accent_color, accent_dark, output):
    """Plot Sharpe and Max Drawdown for one equity universe, highlighting Fixed Kalman (Risk-Sized)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    variants, short_names = _variant_specs()
    x = np.arange(len(variants))
    width = 0.56

    sub_n = summary[(summary.universe == universe) & (summary.cost_bps == 5)]
    sub_g = summary[(summary.universe == universe) & (summary.cost_bps == 0)]

    sh_n = [float(sub_n[sub_n.variant == v]['sharpe'].values[0]) for v in variants]
    sh_g = [float(sub_g[sub_g.variant == v]['sharpe'].values[0]) for v in variants]
    dd_n = [float(sub_n[sub_n.variant == v]['max_drawdown'].values[0]) * 100 for v in variants]

    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.6))

    # Left Panel: Sharpe Ratio (Net 5 bps bars + Gross 0 bps diamonds)
    bars1 = ax1.bar(x, sh_n, width)
    for idx, bar in enumerate(bars1):
        bar.set_facecolor(accent_color)
        bar.set_alpha(0.82)
        bar.set_edgecolor(accent_dark)
        bar.set_linewidth(1.1)

    for i in range(len(variants)):
        ax1.scatter(x[i], sh_g[i], marker='D', s=50, facecolor='white', edgecolor=accent_dark, linewidth=1.6, zorder=5)
        ax1.text(x[i], sh_n[i] * 0.55, f'Net\n{sh_n[i]:.2f}', ha='center', va='center', fontsize=8.8, fontweight='bold', color='white')
        ax1.text(x[i], sh_g[i] + 0.035, f'{sh_g[i]:.2f}', ha='center', va='bottom', fontsize=8.6, color=accent_dark)

    if bench_row:
        ax1.axhline(bench_row['sharpe'], color='#475569', linestyle='--', lw=1.3,
                    label=f"{bench_row['benchmark']} Sharpe ({bench_row['sharpe']:.2f})")

    ax1.set_title('Annualized Sharpe Ratio (Net 5 bps Bars & Gross 0 bps Diamonds)', fontsize=11.2, fontweight='bold', pad=10)
    ax1.set_ylabel('Annualized Sharpe Ratio', fontsize=10.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=9.2)
    y_top = max(sh_g) + 0.22
    y_bot = min(-0.12, bench_row['sharpe'] - 0.10) if bench_row else -0.05
    ax1.set_ylim(y_bot, y_top)
    ax1.axhline(0, color='black', lw=0.7, alpha=0.6)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax1.spines[['top', 'right']].set_visible(False)
    leg_elems1 = [
        Patch(facecolor=accent_color, alpha=0.82, edgecolor=accent_dark, linewidth=1.1, label='Net (5 bps) Sharpe'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor='white', markeredgecolor=accent_dark, markeredgewidth=1.6, markersize=6.5, label='Gross (0 bps) Sharpe'),
    ]
    if bench_row:
        leg_elems1.append(Line2D([0], [0], color='#475569', linestyle='--', lw=1.3, label=f"{bench_row['benchmark']} ({bench_row['sharpe']:.2f})"))
    ax1.legend(handles=leg_elems1, fontsize=8.4, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, edgecolor='#dddddd')

    # Right Panel: Unlevered Maximum Drawdown (Net 5 bps)
    bars2 = ax2.bar(x, dd_n, width)
    for idx, bar in enumerate(bars2):
        bar.set_facecolor(accent_color)
        bar.set_alpha(0.82)
        bar.set_edgecolor(accent_dark)
        bar.set_linewidth(1.1)
        ax2.text(x[idx], dd_n[idx] - 0.7, f'{dd_n[idx]:.1f}%', ha='center', va='top',
                 fontsize=8.6, color='#1e293b')

    if bench_row:
        ax2.axhline(bench_row['max_drawdown'] * 100, color='#475569', linestyle='--', lw=1.3,
                    label=f"{bench_row['benchmark']} Max DD ({bench_row['max_drawdown']:.1%})")

    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_title('Unlevered Maximum Drawdown (Net 5 bps)', fontsize=11.2, fontweight='bold', pad=10)
    ax2.set_ylabel('Maximum Drawdown (%)', fontsize=10.2)
    ax2.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=9.2)
    dd_floor = min(-39.0, (bench_row['max_drawdown'] * 100 - 3.5) if bench_row else -30.0)
    ax2.set_ylim(dd_floor, 2.0)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)
    leg_elems2 = [
        Patch(facecolor=accent_color, alpha=0.82, edgecolor=accent_dark, linewidth=1.1, label='Strategy Max DD (Net 5 bps)'),
    ]
    if bench_row:
        leg_elems2.append(Line2D([0], [0], color='#475569', linestyle='--', lw=1.3, label=f"{bench_row['benchmark']} Max DD ({bench_row['max_drawdown']:.1%})"))
    ax2.legend(handles=leg_elems2, fontsize=8.4, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, edgecolor='#dddddd')

    fig.suptitle(
        f'{uni_label} — Sharpe Ratio and Maximum Drawdown across Variants (Net 5 bps vs. Gross)',
        fontsize=12.2, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)


def chart_universe_risk_return_alpha(summary, bench_row, universe, uni_label, accent_color, accent_dark, scatter_offsets, output):
    """Plot Vol-Matched Return vs. Max DD scatter and CAPM Alpha for one equity universe."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    variants, short_names = _variant_specs()
    short_map = {
        'PCA / equal': 'Rolling PCA (Eq)',
        'Fixed Kalman / equal': 'Fixed Kalman (Eq)',
        'Fixed Kalman / inverse vol': 'Fixed Kalman (Risk-Sized)',
        'Calibrated Kalman / equal': 'Calibrated Kalman (Eq)',
        'Calibrated Kalman / inverse vol': 'Calibrated Kalman (Risk)',
    }
    sub_n = summary[(summary.universe == universe) & (summary.cost_bps == 5)]
    sub_g = summary[(summary.universe == universe) & (summary.cost_bps == 0)]

    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.7))

    # Left Panel: Ex-Post Vol-Matched Return vs. Max DD Magnitude
    all_x, all_y = [], []
    for v in variants:
        row = sub_n[sub_n.variant == v].iloc[0]
        dd_mag = abs(float(row['vm_max_drawdown'])) * 100.0
        ret_vm = float(row['vm_total_return']) * 100.0
        all_x.append(dd_mag)
        all_y.append(ret_vm)
        ax1.scatter(dd_mag, ret_vm, s=90,
                    color=accent_color,
                    marker='o' if universe == 'uk_external' else 's',
                    edgecolor=accent_dark, linewidth=1.1,
                    zorder=5)
        dx, dy, ha = scatter_offsets[v]
        label_txt = f"{short_map[v]}\n({ret_vm:+.1f}%, DD -{dd_mag:.1f}%)"
        ax1.annotate(
            label_txt, xy=(dd_mag, ret_vm), xytext=(dd_mag + dx, ret_vm + dy),
            fontsize=8.1, color='#1e293b', ha=ha, va='center',
        )

    if bench_row:
        bx, by = abs(bench_row['max_drawdown']) * 100.0, bench_row['total_return'] * 100.0
        all_x.append(bx)
        all_y.append(by)
        ax1.scatter(bx, by, s=125, marker='^' if universe == 'us_global' else 'v', color='#334155', edgecolor='black', lw=1.0, zorder=5)
        if by < 0:
            ax1.text(bx - 1.0, by - 1.0,
                     f"{bench_row['benchmark']}\n({by:+.1f}%, DD -{bx:.1f}%)",
                     ha='right', va='top', fontsize=8.4, fontweight='bold', color='#1e293b')
        else:
            ax1.text(bx, by + 3.2,
                     f"{bench_row['benchmark']}\n({by:+.1f}%, DD -{bx:.1f}%)",
                     ha='center', va='bottom', fontsize=8.4, fontweight='bold', color='#1e293b')

    x_pad = max((max(all_x) - min(all_x)) * 0.22, 4.5)
    y_pad = max((max(all_y) - min(all_y)) * 0.18, 12.0)
    ax1.axhline(0, color='black', lw=0.7, linestyle=':')
    ax1.set_title('Volatility-Matched Return vs. Max Drawdown', fontsize=11.2, fontweight='bold', pad=10)
    ax1.set_xlabel('Volatility-Matched Max Drawdown Magnitude (% |Max DD|)', fontsize=10.0)
    ax1.set_ylabel('Volatility-Matched Net Total Return (%, 5 bps)', fontsize=10.0)
    ax1.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax1.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax1.set_xlim(min(all_x) - x_pad, max(all_x) + 3.5)
    ax1.set_ylim(min(all_y) - 18.0, max(all_y) + y_pad)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.spines[['top', 'right']].set_visible(False)

    # Right Panel: Annualized OLS CAPM Alpha (%) & t-statistics
    x = np.arange(len(variants))
    width = 0.56
    a_n = [float(sub_n[sub_n.variant == v]['alpha_annualized'].values[0]) * 100 for v in variants]
    a_g = [float(sub_g[sub_g.variant == v]['alpha_annualized'].values[0]) * 100 for v in variants]
    t_n = [float(sub_n[sub_n.variant == v]['alpha_tstat'].values[0]) for v in variants]

    bars = ax2.bar(x, a_n, width)
    for i, bar in enumerate(bars):
        bar.set_facecolor(accent_color)
        bar.set_alpha(0.82)
        bar.set_edgecolor(accent_dark)
        bar.set_linewidth(1.1)
        ax2.scatter(x[i], a_g[i], marker='D', s=48, facecolor='white', edgecolor=accent_dark, linewidth=1.6, zorder=5)
        ax2.text(x[i], a_n[i] * 0.55, f'+{a_n[i]:.1f}%\n(t={t_n[i]:.2f})', ha='center', va='center',
                 fontsize=8.4, fontweight='bold', color='white')
        ax2.text(x[i], a_g[i] + 0.35, f'+{a_g[i]:.1f}%', ha='center', va='bottom',
                 fontsize=8.4, color=accent_dark)

    bench_name = bench_row['benchmark'] if bench_row else 'Market Index'
    ax2.set_title(f'Annualized OLS CAPM Alpha vs. {bench_name} (Unlevered)', fontsize=11.2, fontweight='bold', pad=10)
    ax2.set_ylabel("Annualized Jensen's Alpha (%)", fontsize=10.0)
    ax2.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=9.2)
    ax2.set_ylim(0, max(a_g) + 2.8)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)
    leg_elems = [
        Patch(facecolor=accent_color, alpha=0.82, edgecolor=accent_dark, linewidth=1.1, label='Net (5 bps) CAPM Alpha'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor='white', markeredgecolor=accent_dark, markeredgewidth=1.6, markersize=6.5, label='Gross (0 bps) CAPM Alpha'),
    ]
    ax2.legend(handles=leg_elems, fontsize=8.4, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, edgecolor='#dddddd')

    target_vol = bench_row['annual_volatility'] * 100 if bench_row else 20.0
    fig.suptitle(
        f'{uni_label} — Volatility-Matched Risk-Return Profile ({target_vol:.1f}% {bench_name} Vol) and CAPM Alpha',
        fontsize=12.2, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run(root):
    root = Path(root).resolve()
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    request = config['period_comparison']
    prices, sources = price_history(root, request)
    prices['us_global'] = pd.concat([prices['technology'], prices['financial'], prices['energy']], axis=1).dropna(how='any')
    benchmarks = load_index_benchmarks(root, request['download_start'], '2021-07-02')
    rows, histories, periods = [], [], []
    artifacts_dir = root / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ftse_ret_series = pd.Series(dtype=float)
    sp500_ret_series = pd.Series(dtype=float)
    for sample, dates in request['samples'].items():
        sample_days = []
        for universe, frame in prices.items():
            observations = frame.loc[dates['start']:dates['end']].pct_change(fill_method=None).dropna()
            print(f'Evaluating {sample}: {universe}', flush=True)
            u_config = json.loads(json.dumps(config))
            if universe == 'uk_external' and 'uk_q_fixed' in request:
                u_config['model']['q_fixed'] = request['uk_q_fixed']
            _, daily, _, _ = analyze(observations, u_config)
            daily.insert(0, 'universe', universe)
            sample_days.append(daily)
        active, start, end = align_active_period(pd.concat(sample_days, ignore_index=True))

        uk_dates = pd.DatetimeIndex(sorted(active.loc[active.universe == 'uk_external', 'date'].unique()))
        us_dates = pd.DatetimeIndex(sorted(active.loc[active.universe == 'us_global', 'date'].unique()))
        ftse_ret_series = benchmarks['ftse100'].pct_change().reindex(uk_dates).fillna(0.0)
        sp500_ret_series = benchmarks['sp500'].pct_change().reindex(us_dates).fillna(0.0)
        ftse_vol = float(np.std(ftse_ret_series.to_numpy(), ddof=1) * np.sqrt(252))
        sp500_vol = float(np.std(sp500_ret_series.to_numpy(), ddof=1) * np.sqrt(252))

        for (universe, variant, cost), group in active.groupby(['universe', 'variant', 'cost_bps'], sort=False):
            r = group.net_return.to_numpy()
            s = score(r)
            tot = float(np.prod(1.0 + r) - 1.0)
            vol = float(np.std(r, ddof=1) * np.sqrt(252))
            if universe == 'uk_external':
                mkt = ftse_ret_series.to_numpy()
                target_vol = ftse_vol
                bench_name = 'FTSE 100 Index'
            else:
                mkt = benchmarks['sp500'].pct_change().reindex(pd.DatetimeIndex(group.date)).fillna(0.0).to_numpy()
                target_vol = sp500_vol
                bench_name = 'S&P 500 Index'
            capm = _capm_stats(r, mkt)
            lev = target_vol / vol if vol > 0 else 1.0
            r_vm = r * lev
            s_vm = score(r_vm)
            tot_vm = float(np.prod(1.0 + r_vm) - 1.0)
            rows.append({'sample': sample, 'universe': universe, 'variant': variant, 'cost_bps': cost,
                         'start': str(start.date()), 'end': str(end.date()),
                         'days': s['days'], 'total_return': tot, 'annual_volatility': vol,
                         'sharpe': s['sharpe'], 'cagr': s['cagr'], 'max_drawdown': s['max_drawdown'],
                         'average_daily_turnover': group.turnover.mean(), **capm,
                         'benchmark_index': bench_name, 'vm_leverage': lev,
                         'vm_total_return': tot_vm, 'vm_cagr': s_vm['cagr'], 'vm_max_drawdown': s_vm['max_drawdown']})
        periods.append({'sample': sample, 'start': str(start.date()), 'end': str(end.date()),
                        'zero_exposure_rows': int(active.gross_exposure.eq(0).sum())})

        histories.append(active.assign(sample=sample))

    summary = pd.DataFrame(rows)

    bench_rows = []
    for b_name, b_constr, b_ret in [
        ('FTSE 100 Index', '^FTSE Buy & Hold (100% Long)', ftse_ret_series.to_numpy()),
        ('S&P 500 Index', '^GSPC Buy & Hold (100% Long)', sp500_ret_series.to_numpy()),
    ]:
        b_s = score(b_ret)
        b_tot = float(np.prod(1.0 + b_ret) - 1.0)
        b_vol = float(np.std(b_ret, ddof=1) * np.sqrt(252))
        bench_rows.append({'benchmark': b_name, 'construction': b_constr, 'start': str(start.date()), 'end': str(end.date()),
                           'days': b_s['days'], 'total_return': b_tot, 'cagr': b_s['cagr'],
                           'annual_volatility': b_vol, 'sharpe': b_s['sharpe'], 'max_drawdown': b_s['max_drawdown']})
    pd.DataFrame(bench_rows).to_csv(artifacts_dir / 'benchmarks.csv', index=False)

    ftse_b = next((b for b in bench_rows if 'FTSE' in b['benchmark']), None)
    sp_b = next((b for b in bench_rows if 'S&P' in b['benchmark']), None)

    chart_universe_method_comparison(
        summary, ftse_b, 'uk_external', 'Global Equities (UK, N=16)',
        '#1d4ed8', '#1e3a8a', artifacts_dir / 'uk_method_comparison.png',
    )
    chart_universe_method_comparison(
        summary, sp_b, 'us_global', 'Global Equities (US, N=45)',
        '#15803d', '#14532d', artifacts_dir / 'us_method_comparison.png',
    )

    uk_offsets = {
        'PCA / equal': (0.6, -2.5, 'left'),
        'Fixed Kalman / equal': (0.7, 2.0, 'left'),
        'Fixed Kalman / inverse vol': (0.0, 12.0, 'center'),
        'Calibrated Kalman / equal': (0.7, 0.0, 'left'),
        'Calibrated Kalman / inverse vol': (-0.6, -1.5, 'right'),
    }
    us_offsets = {
        'PCA / equal': (0.0, -5.0, 'center'),
        'Fixed Kalman / equal': (0.6, 2.5, 'left'),
        'Fixed Kalman / inverse vol': (0.0, 9.5, 'center'),
        'Calibrated Kalman / equal': (0.6, -1.0, 'left'),
        'Calibrated Kalman / inverse vol': (-0.4, 4.5, 'center'),
    }
    chart_universe_risk_return_alpha(
        summary, ftse_b, 'uk_external', 'Global Equities (UK, N=16)',
        '#1d4ed8', '#1e3a8a', uk_offsets, artifacts_dir / 'uk_risk_return_alpha.png',
    )
    chart_universe_risk_return_alpha(
        summary, sp_b, 'us_global', 'Global Equities (US, N=45)',
        '#15803d', '#14532d', us_offsets, artifacts_dir / 'us_risk_return_alpha.png',
    )

    all_daily = pd.concat(histories, ignore_index=True)

    summary.to_csv(artifacts_dir / 'metrics.csv', index=False)
    all_daily.to_parquet(artifacts_dir / 'daily_returns.parquet', index=False)

    pkg_versions = {}
    for name in ('numpy', 'pandas', 'yfinance'):
        try:
            pkg_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pkg_versions[name] = 'not-installed'

    manifest = {'samples': periods, 'prices': sources,
                'benchmarks': {'ftse100': 'data/period_comparison/ftse100.csv', 'sp500': 'data/period_comparison/sp500.csv'},
                'model': config['model'], 'portfolio': config['portfolio'], 'cost_bps': config['cost_bps'],
                'python': platform.python_version(),
                'packages': pkg_versions,
                'price_request': {key: request[key] for key in ('download_start', 'download_end', 'universes')},
                'evaluation': f'Global Equities (UK & US) matched calendar bounds ({start.date()} to {end.date()}); evaluated at Gross (0 bps) and Net (5 bps) with CAPM alpha and index vol-matched leverage.',
                'accounting': config['accounting'], 'source': 'Yahoo Finance through yfinance'}
    (artifacts_dir / 'run.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print('\n=== Global Equities (UK & US) Summary: Gross (0 bps) vs Net (5 bps) ===', flush=True)
    disp = summary[
        summary.cost_bps.isin([0, 5])
        & summary.universe.isin(['uk_external', 'us_global'])
    ][['universe', 'variant', 'cost_bps', 'total_return', 'sharpe', 'max_drawdown', 'alpha_annualized', 'vm_leverage', 'vm_total_return', 'vm_max_drawdown']].round(4)
    print(disp.to_string(index=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Run matched-period statistical arbitrage study.')
    parser.parse_args()
    run(Path.cwd())


if __name__ == '__main__':
    main()
