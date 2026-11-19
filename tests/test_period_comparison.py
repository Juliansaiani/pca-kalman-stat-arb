import numpy as np
import pandas as pd
import pytest

from equity_stat_arb.period_comparison import align_active_period


def test_all_panels_share_actual_dates_and_initial_flat_rows_are_removed():
    dates = pd.bdate_range('2020-01-01', periods=20)
    frames = []
    for name, first, length in [('a', 2, 20), ('b', 5, 18), ('c', 3, 19), ('d', 4, 20)]:
        for variant in ['Fixed Kalman / equal', 'Calibrated Kalman / inverse vol']:
            frames.append(pd.DataFrame({'date': dates[:length], 'universe': name,
                                        'variant': variant, 'cost_bps': 10,
                                        'gross_exposure': np.r_[np.zeros(first), np.ones(length - first)],
                                        'net_return': .001}))
    cropped, start, end = align_active_period(pd.concat(frames, ignore_index=True))
    assert start == dates[5]
    assert end == dates[17]
    assert cropped.gross_exposure.gt(0).all()
    bounds = cropped.groupby(['universe', 'variant']).date.agg(['min', 'max'])
    assert bounds['min'].eq(start).all()
    assert bounds['max'].eq(end).all()


def test_disjoint_histories_cannot_be_presented_as_shared_period():
    data = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2023-01-01', '2023-01-02']),
                         'universe': ['old', 'old', 'new', 'new'], 'variant': 'model',
                         'cost_bps': 10, 'gross_exposure': 1, 'net_return': .001})
    with pytest.raises(ValueError, match='No shared'):
        align_active_period(data)


def test_universe_without_positions_fails_explicitly():
    data = pd.DataFrame({'date': pd.to_datetime(['2020-01-01'] * 2),
                         'universe': ['active', 'flat'], 'variant': 'model',
                         'cost_bps': 10, 'gross_exposure': [1, 0], 'net_return': .001})
    with pytest.raises(ValueError, match='never trade'):
        align_active_period(data)
