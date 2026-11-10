import numpy as np
import pandas as pd
import pytest

from equity_stat_arb.pipeline import analyze, verify_reference


def test_all_variants_and_costs_run_without_network():
    rng = np.random.default_rng(6)
    frame = pd.DataFrame(rng.normal(0, .01, (400, 8)),
                         index=pd.bdate_range('2017-01-01', periods=400, name='Date'),
                         columns=[f'S{i}' for i in range(8)])
    config = {
        'model': dict(lookback=60, refit_every=10, variance_fraction=.55, q_fixed=1e-4,
                      q_grid=[0, 1e-4], r_grid=[1], calibration_fraction=2 / 3),
        'portfolio': dict(residual_history=60, pressure_days=5, signal_vol_days=20,
                          rebalance_every=20, names_per_side=3, risk_window=20, position_cap=.25),
        'cost_bps': [0, 10],
    }
    scores, daily, diagnostics, holdings = analyze(frame, config)
    assert len(scores) == 5 * 2
    assert daily.variant.nunique() == 5
    assert scores.days.eq(len(frame) - config['model']['lookback']).all()
    assert np.isfinite(daily.net_return).all()
    assert (diagnostics.training_end < diagnostics.date).all()
    assert not holdings.empty
    for _, group in daily.groupby('variant'):
        free = group[group.cost_bps == 0]
        paid = group[group.cost_bps == 10]
        np.testing.assert_allclose(free.net_return.to_numpy() - paid.net_return.to_numpy(),
                                   paid.turnover.to_numpy() * .001, atol=1e-12)
    scores.insert(0, 'universe', 'synthetic')
    expected = scores.copy()
    report = verify_reference(scores, expected)
    assert len(report) == 10
    expected.loc[0, 'sharpe'] += .001
    with pytest.raises(AssertionError, match='sharpe'):
        verify_reference(scores, expected)
