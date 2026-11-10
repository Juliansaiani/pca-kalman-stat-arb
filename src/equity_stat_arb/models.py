"""Correlation PCA and batched Kalman innovation models."""
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


@dataclass
class Basis:
    center: np.ndarray
    scale: np.ndarray
    directions: np.ndarray
    noise: np.ndarray

    @classmethod
    def estimate(cls, returns, fraction):
        x = np.asarray(returns, dtype=float)
        center, scale = x.mean(axis=0), x.std(axis=0)
        if not np.isfinite(x).all() or np.any(scale <= 0) or not 0 < fraction <= 1:
            raise ValueError('PCA requires finite observations, positive scales and a valid variance fraction')
        z = (x - center) / scale
        eigenvalues, eigenvectors = np.linalg.eigh(z.T @ z / len(z))
        order = np.argsort(eigenvalues)[::-1]
        spectrum = np.maximum(eigenvalues[order], 0)
        k = min(np.searchsorted(spectrum.cumsum() / spectrum.sum(), fraction) + 1, len(spectrum))
        directions = eigenvectors[:, order[:k]]
        errors = z - (z @ directions) @ directions.T
        return cls(center, scale, directions, np.maximum(errors.var(axis=0), 1e-6))

    def standardize(self, returns):
        return (np.asarray(returns) - self.center) / self.scale

    def residuals(self, returns):
        z = self.standardize(returns)
        return z - (z @ self.directions) @ self.directions.T


def filter_paths(basis, observations, candidates):
    """Run all (Q,R) choices in parallel; return innovations and Gaussian losses.

    Covariances are independent per stock, while one parameter pair is selected
    for the whole universe. Q is relative to training residual variance; R is a
    multiplier of that variance. The saved error precedes the state update.
    """
    candidates = np.asarray(candidates, dtype=float)
    if (candidates.ndim != 2 or candidates.shape[1] != 2
            or not np.isfinite(candidates).all() or (candidates[:, 0] < 0).any()
            or (candidates[:, 1] <= 0).any()):
        raise ValueError('Invalid Kalman noise candidates')
    standardized = basis.standardize(observations)
    count, stocks = len(candidates), len(basis.center)
    dimension = basis.directions.shape[1] + 1
    state = np.repeat(np.column_stack((np.zeros(stocks), basis.directions))[None], count, axis=0)
    covariance = np.zeros((count, stocks, dimension, dimension))
    process = candidates[:, 0, None, None, None] * basis.noise[None, :, None, None] * np.eye(dimension)
    observation = candidates[:, 1, None] * basis.noise[None]
    losses = np.zeros(count)
    errors = np.empty((count, len(standardized), stocks))
    for t, z in enumerate(standardized):
        design = np.r_[1.0, z @ basis.directions]
        error = z[None] - state @ design
        predicted = covariance + process
        cross = predicted @ design
        variance = cross @ design + observation
        losses += .5 * np.sum(np.log(2 * np.pi * variance) + error**2 / variance, axis=1)
        errors[:, t] = error
        state += cross * (error / variance)[..., None]
        covariance = predicted - cross[..., :, None] * cross[..., None, :] / variance[..., None, None]
        covariance = .5 * (covariance + covariance.swapaxes(-1, -2))
    return errors, losses


def fit_residual_models(returns: pd.DataFrame, config: dict):
    """Each block uses only its preceding training window; bases reset per refit."""
    x = returns.to_numpy(dtype=float)
    window, step = config['lookback'], config['refit_every']
    if (window < 30 or step < 1 or len(x) <= window or not np.isfinite(x).all()
            or not returns.index.is_unique or not returns.index.is_monotonic_increasing):
        raise ValueError('Invalid rolling model inputs')
    buffers = {name: [] for name in ('PCA', 'Fixed Kalman', 'Calibrated Kalman')}
    dates, records = [], []
    pairs = list(product(config['q_grid'], config['r_grid']))
    for first in range(window, len(x), step):
        past = x[first - window:first]
        split = int(window * config['calibration_fraction'])
        if not 10 <= split < window - 5:
            raise ValueError('Calibration needs separate fitting and validation histories')
        calibration_basis = Basis.estimate(past[:split], config['variance_fraction'])
        _, losses = filter_paths(calibration_basis, past[split:], pairs)
        winner = int(np.argmin(losses))
        q, r = pairs[winner]
        production_basis = Basis.estimate(past, config['variance_fraction'])
        block = x[first:min(first + step, len(x))]
        errors, _ = filter_paths(production_basis, block, [(config['q_fixed'], 1), (q, r)])
        buffers['PCA'].append(production_basis.residuals(block))
        buffers['Fixed Kalman'].append(errors[0])
        buffers['Calibrated Kalman'].append(errors[1])
        dates.extend(returns.index[first:first + len(block)])
        records.append({'date': returns.index[first], 'training_end': returns.index[first - 1],
                        'fit_observations': split, 'validation_observations': window - split,
                        'factors': production_basis.directions.shape[1], 'q': q, 'r': r,
                        'validation_nll': losses[winner]})
    index = pd.DatetimeIndex(dates, name='Date')
    frames = {name: pd.DataFrame(np.concatenate(blocks), index=index, columns=returns.columns)
              for name, blocks in buffers.items()}
    return frames, pd.DataFrame(records)
