import numpy as np
import pandas as pd

from equity_stat_arb.pipeline import analyze


def test_all_variants_and_periods_run_without_network():
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
        'evaluation': dict(validation_start='2018-01-01', holdout_start='2018-05-01'),
    }
    scores, daily, diagnostics, holdings = analyze(frame, config)
    assert len(scores) == 5 * 2 * 4
    assert daily.variant.nunique() == 5
    assert scores[scores.period == 'holdout'].days.gt(0).all()
    assert np.isfinite(daily.net_return).all()
    assert (diagnostics.training_end < diagnostics.date).all()
    assert not holdings.empty
    holdout = daily[daily.date >= '2018-05-01']
    for name in daily.variant.unique():
        expected = len(holdout[(holdout.variant == name) & (holdout.cost_bps == 10)])
        actual = scores[(scores.variant == name) & (scores.cost_bps == 10) & (scores.period == 'holdout')].iloc[0].days
        assert actual == expected
