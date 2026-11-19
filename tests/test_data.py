import hashlib
import io
import urllib.request

import pandas as pd
import pytest

from equity_stat_arb.data import load_returns


def test_pinned_download_and_corrupted_cache_check(tmp_path, monkeypatch):
    content = b'Date,A,B\n2020-01-01,0.01,-0.02\n2020-01-02,0.02,0.03\n'
    settings = {'repository': 'sample/data', 'revision': 'frozen',
                'panels': {'sample': {'path': 'returns.csv',
                                     'sha256': hashlib.sha256(content).hexdigest()}}}
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(content))
    loaded, provenance = load_returns(tmp_path, settings)
    assert loaded['sample'].shape == (2, 2)
    assert provenance[0]['sha256'] == hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *args, **kwargs: pytest.fail('Cache should be offline'))
    cached, _ = load_returns(tmp_path, settings)
    pd.testing.assert_frame_equal(loaded['sample'], cached['sample'])
    (tmp_path / 'data/returns/sample.csv').write_bytes(content + b'altered')
    with pytest.raises(ValueError, match='checksum'):
        load_returns(tmp_path, settings)
