"""Direct adjusted-price downloads with a local, checksummed snapshot."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(root: Path, settings: dict, refresh=False):
    import yfinance as yf
    cache = root / 'data'
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous['request'] != settings and not refresh:
        raise ValueError('Data request changed; pass --refresh-data for a new snapshot')
    frames, records = {}, []
    for name, tickers in settings['universes'].items():
        path = cache / f'{name}_prices.csv'
        old = next((r for r in previous['files'] if r['universe'] == name), None) if previous else None
        if path.exists() and old and not refresh:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != old['sha256']:
                raise ValueError(f'Cached price checksum mismatch: {name}')
            prices = pd.read_csv(path, index_col=0, parse_dates=True, float_precision='round_trip')
            record = old
        else:
            print(f'Downloading adjusted prices: {name}', flush=True)
            downloaded = yf.download(tickers, start=settings['start'], end=settings['end'],
                                     auto_adjust=True, progress=False, threads=2, timeout=30)
            prices = downloaded['Close']
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(tickers[0])
            prices = prices.reindex(columns=tickers).sort_index()
            coverage = prices.notna().mean()
            bad = coverage[coverage < .90].index.tolist()
            for ticker in bad:
                retry = yf.download([ticker], start=settings['start'], end=settings['end'],
                                    auto_adjust=True, progress=False, threads=False, timeout=30)
                if not retry.empty and 'Close' in retry:
                    close = retry['Close']
                    prices[ticker] = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
            coverage = prices.notna().mean()
            bad = coverage[coverage < .90].index.tolist()
            if bad:
                raise ValueError(f'Incomplete Yahoo history for {bad}; no silent universe substitution')
            # No price interpolation or long forward fills. Align on observed dates only.
            prices = prices.dropna(how='any')
            if len(prices) < 500 or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
                raise ValueError(f'Invalid adjusted-price history: {name}')
            prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
            prices.to_csv(path, index_label='Date')
            prices = pd.read_csv(path, index_col=0, parse_dates=True, float_precision='round_trip')
            record = {'universe': name, 'file': f'data/{path.name}',
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'downloaded_utc': datetime.now(timezone.utc).isoformat(),
                      'tickers': tickers, 'observations': len(prices),
                      'first_date': str(prices.index[0].date()), 'last_date': str(prices.index[-1].date()),
                      'minimum_raw_coverage': float(coverage.min())}
        frames[name] = prices
        records.append(record)
        # Preserve successful downloads if a later universe fails.
        manifest_path.write_text(json.dumps({'provider': 'Yahoo Finance through yfinance',
                                             'auto_adjust': True, 'request': settings,
                                             'files': records}, indent=2) + '\n', encoding='utf-8')
    manifest = {'provider': 'Yahoo Finance through yfinance', 'auto_adjust': True,
                'request': settings, 'files': records}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return frames, manifest
