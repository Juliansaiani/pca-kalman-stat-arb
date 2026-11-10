"""Residual ranks, capped risk sizing and holdings-based portfolio accounting."""
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
    """Apply orders one observation later; shares drift between scheduled trades.

    Costs are paid from cash. Borrow/financing and intraday execution are absent.
    Neutrality and caps apply at rebalancing, not continuously between trades.
    """
    if not returns.index.equals(orders.index) or not returns.columns.equals(orders.columns):
        raise ValueError('Orders and returns must align')
    if not np.isfinite(cost_bps) or cost_bps < 0 or not np.isfinite(returns.to_numpy()).all():
        raise ValueError('Invalid cost or returns')
    x, targets = returns.to_numpy(), orders.to_numpy()
    weights = np.zeros(x.shape[1])
    nav, history, held = 1.0, [], []
    for t, changes in enumerate(x):
        turnover = 0.0
        if t and np.isfinite(targets[t - 1]).all():
            turnover = float(np.abs(targets[t - 1] - weights).sum())
            weights = targets[t - 1].copy()
        held.append(weights.copy())
        gross = float(weights @ changes)
        fees = turnover * cost_bps / 10000
        net = gross - fees
        if 1 + net <= 0:
            raise ValueError('Portfolio NAV became non-positive')
        nav *= 1 + net
        history.append((gross, fees, net, turnover, nav, weights.sum(), np.abs(weights).sum()))
        weights = weights * (1 + changes) / (1 + net)
    daily = pd.DataFrame(history, index=returns.index,
                         columns=['gross_return', 'cost', 'net_return', 'turnover', 'equity', 'net_exposure', 'gross_exposure'])
    return daily, pd.DataFrame(held, index=returns.index, columns=returns.columns)
