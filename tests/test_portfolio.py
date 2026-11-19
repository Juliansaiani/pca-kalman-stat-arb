import numpy as np
import pandas as pd
import pytest

from equity_stat_arb.portfolio import capped_allocation, decision_weights, simulate
from equity_stat_arb.pipeline import score


def settings():
    return dict(pressure_days=5, signal_vol_days=20, residual_history=40,
                rebalance_every=20, names_per_side=3, risk_window=20, position_cap=.25)


def test_capped_weights_preserve_budget_and_handle_extreme_risk():
    weights = capped_allocation([10000, 2, 1])
    assert weights[0] == .25
    assert weights.sum() == pytest.approx(.5)
    assert weights.max() <= .25
    with pytest.raises(ValueError):
        capped_allocation([1], cap=.25)


def test_ranking_is_disjoint_and_risk_sizing_uses_strictly_past_prices():
    rng = np.random.default_rng(73)
    dates = pd.bdate_range('2020-01-01', periods=100)
    residuals = pd.DataFrame(rng.normal(size=(100, 8)), index=dates)
    raw = pd.DataFrame(rng.normal(size=(100, 8)) * np.linspace(.005, .03, 8), index=dates)
    orders = decision_weights(residuals, raw, settings(), True)
    changed = raw.copy()
    changed.iloc[40:] *= 100
    revised = decision_weights(residuals, changed, settings(), True)
    pd.testing.assert_series_equal(orders.iloc[40], revised.iloc[40])
    active = orders.dropna()
    np.testing.assert_allclose(active.sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(active.abs().sum(axis=1), 1)
    assert ((active > 0).sum(axis=1) == 3).all()
    assert ((active < 0).sum(axis=1) == 3).all()
    assert active.abs().to_numpy().max() <= .25 + 1e-12
    assert orders.iloc[:40].isna().all().all()


def test_execution_lag_and_target_costs_match_hand_calculation():
    index = pd.bdate_range('2020-01-01', periods=3)
    returns = pd.DataFrame([[0, 0], [.1, 0], [.1, 0]], index=index)
    orders = pd.DataFrame([[.5, -.5], [np.nan, np.nan], [np.nan, np.nan]], index=index)
    daily, held = simulate(returns, orders, 10)
    assert daily.iloc[0].net_return == 0
    assert daily.iloc[1].cost == pytest.approx(.001)
    assert daily.iloc[2].turnover == 0
    assert held.iloc[2, 0] == pytest.approx(.5)
    assert daily.iloc[-1].equity == pytest.approx(1.049 * 1.05)


def test_no_orders_stay_flat_and_first_loss_enters_drawdown():
    frame = pd.DataFrame(np.full((10, 2), .01))
    orders = frame * np.nan
    daily, _ = simulate(frame, orders, 10)
    assert daily.net_return.eq(0).all()
    assert np.isnan(score(daily.net_return)['sharpe'])
    assert score([-.1, .1])['max_drawdown'] == pytest.approx(-.1)
