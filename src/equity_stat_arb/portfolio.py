"""Residual ranks, capped risk sizing and constant-target-weight accounting."""
import numpy as np
import pandas as pd


def capped_allocation(inverse_risk, budget=.5, cap=.25):
    risk = np.asarray(inverse_risk, dtype=float)
    if (not np.isfinite(risk).all() or (risk <= 0).any()
            or len(risk) * cap < budget or budget <= 0 or cap <= 0):
        raise ValueError('Invalid allocation or infeasible cap')
    result = np.zeros(len(risk))
    available = np.ones(len(risk), dtype=bool)
    remaining = budget
    while available.any():
        proposed = remaining * risk[available] / risk[available].sum()
        indices = np.flatnonzero(available)
        clipped = proposed > cap
        if not clipped.any():
            result[indices] = proposed
            break
        fixed = indices[clipped]
        result[fixed] = cap
        available[fixed] = False
        remaining -= len(fixed) * cap
    return result


def decision_weights(residuals, raw_returns, config, risk_sizing=False):
    """NaN rows mean no new order; non-NaN rows specify target NAV weights."""
    if not residuals.index.equals(raw_returns.index) or not residuals.columns.equals(raw_returns.columns):
        raise ValueError('Residuals and returns must align')
    pressure = residuals.rolling(config['pressure_days']).sum()
    scale = residuals.rolling(config['signal_vol_days']).std().replace(0, np.nan)
    scores = -pressure / scale
    n_side = config['names_per_side']
    orders = np.full(residuals.shape, np.nan)
    for t in range(config['residual_history'], len(residuals), config['rebalance_every']):
        valid = np.flatnonzero(np.isfinite(scores.iloc[t].to_numpy()))
        weights = np.zeros(residuals.shape[1])
        if len(valid) >= 2 * n_side:
            ranking = valid[np.argsort(scores.iloc[t].to_numpy()[valid], kind='stable')]
            short, long = ranking[:n_side], ranking[-n_side:]
            if risk_sizing:
                if t < config['risk_window']:
                    raise ValueError('Insufficient past volatility history')
                vol = raw_returns.iloc[t - config['risk_window']:t].std().to_numpy()
                for sign, names in ((1, long), (-1, short)):
                    weights[names] = sign * capped_allocation(1 / np.maximum(vol[names], 1e-4), cap=config['position_cap'])
            else:
                weights[long], weights[short] = .5 / n_side, -.5 / n_side
        orders[t] = weights
    return pd.DataFrame(orders, index=residuals.index, columns=residuals.columns)


def simulate(returns, orders, cost_bps):
    """Reproduce the archived daily target-weight approximation.

    Targets carry forward, then shift one day. Costs count only changes in those
    targets. Weight-maintenance trades and share drift are deliberately omitted;
    this is a simplified research convention, not an execution simulation.
    """
    if not returns.index.equals(orders.index) or not returns.columns.equals(orders.columns):
        raise ValueError('Orders and returns must align')
    if not np.isfinite(cost_bps) or cost_bps < 0 or not np.isfinite(returns.to_numpy()).all():
        raise ValueError('Invalid cost or returns')
    carried = orders.ffill().shift(1).fillna(0.0).to_numpy()
    exposure = np.abs(carried).sum(axis=1)
    weights = np.divide(carried, exposure[:, None], out=np.zeros_like(carried),
                        where=exposure[:, None] > 0)
    traded = np.abs(np.diff(weights, axis=0, prepend=np.zeros((1, weights.shape[1])))).sum(axis=1)
    gross = (weights * returns.to_numpy()).sum(axis=1)
    fees = traded * cost_bps / 10000
    net = gross - fees
    if np.any(net <= -1):
        raise ValueError('Portfolio NAV became non-positive')
    daily = pd.DataFrame({'gross_return': gross, 'cost': fees, 'net_return': net,
                          'turnover': traded, 'equity': np.cumprod(1 + net),
                          'net_exposure': weights.sum(axis=1),
                          'gross_exposure': np.abs(weights).sum(axis=1)}, index=returns.index)
    return daily, pd.DataFrame(weights, index=returns.index, columns=returns.columns)
