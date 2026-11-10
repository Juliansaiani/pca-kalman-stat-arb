"""Load the frozen return observations used in the archived experiment."""
import hashlib
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


def load_returns(root: Path, settings: dict):
    directory = root / 'data' / 'returns'
    directory.mkdir(parents=True, exist_ok=True)
    panels, records = {}, []
    for universe, specification in settings['panels'].items():
        if not universe.replace('_', '').isalnum():
            raise ValueError('Invalid universe identifier')
        path = directory / f'{universe}.csv'
        address = ('https://raw.githubusercontent.com/' + settings['repository'] + '/'
                   + settings['revision'] + '/' + urllib.parse.quote(specification['path']))
        if path.exists():
            content = path.read_bytes()
        else:
            print(f'Downloading frozen returns: {universe}', flush=True)
            request = urllib.request.Request(address, headers={'User-Agent': 'equity-stat-arb/0.1'})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
        digest = hashlib.sha256(content).hexdigest()
        if digest != specification['sha256']:
            raise ValueError(f'Frozen data checksum mismatch: {universe}')
        if not path.exists():
            path.write_bytes(content)
        returns = pd.read_csv(path, index_col=0, parse_dates=True)
        if (returns.empty or not returns.index.is_unique or not returns.index.is_monotonic_increasing
                or not returns.columns.is_unique or not np.isfinite(returns.to_numpy()).all()):
            raise ValueError(f'Invalid return observations: {universe}')
        panels[universe] = returns
        records.append({'universe': universe, 'source_url': address, 'sha256': digest,
                        'stocks': len(returns.columns), 'observations': len(returns),
                        'start': str(returns.index[0].date()), 'end': str(returns.index[-1].date())})
    return panels, records
