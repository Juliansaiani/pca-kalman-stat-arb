import numpy as np
import pandas as pd
import pytest

from equity_stat_arb.models import Basis, filter_paths, fit_residual_models


def sample_returns(n=320):
    rng = np.random.default_rng(37)
    factors = rng.normal(size=(n, 2))
    beta = rng.normal(size=(2, 8))
    values = .01 * (factors @ beta + rng.normal(0, .5, size=(n, 8)))
    return pd.DataFrame(values, index=pd.bdate_range('2017-01-01', periods=n),
                        columns=[f'S{i}' for i in range(8)])


def settings():
    return dict(lookback=60, refit_every=10, variance_fraction=.55,
                q_fixed=1e-4, q_grid=[0, 1e-4, .01], r_grid=[.5, 1, 2],
                calibration_fraction=2 / 3)


def test_pca_errors_are_orthogonal_to_factors():
    frame = sample_returns()
    basis = Basis.estimate(frame.iloc[:100], .55)
    errors = basis.residuals(frame.iloc[100:])
    np.testing.assert_allclose(errors @ basis.directions, 0, atol=1e-12)


def test_zero_process_noise_recovers_fixed_pca():
    frame = sample_returns()
    basis = Basis.estimate(frame.iloc[:100], .55)
    errors, _ = filter_paths(basis, frame.iloc[100:], [(0, 1)])
    np.testing.assert_allclose(errors[0], basis.residuals(frame.iloc[100:]), atol=1e-12)


def test_batched_calibration_matches_individual_filters():
    frame = sample_returns(150)
    basis = Basis.estimate(frame.iloc[:100], .55)
    candidates = [(0, 1), (1e-4, .5), (.01, 2)]
    errors, losses = filter_paths(basis, frame.iloc[100:], candidates)
    for i, pair in enumerate(candidates):
        single_error, single_loss = filter_paths(basis, frame.iloc[100:], [pair])
        np.testing.assert_allclose(errors[i], single_error[0], atol=1e-12)
        np.testing.assert_allclose(losses[i], single_loss[0], atol=1e-10)


def test_future_observations_do_not_change_past_models_or_calibration():
    frame = sample_returns()
    modified = frame.copy()
    modified.iloc[-40:] *= 10
    original, diagnostics = fit_residual_models(frame, settings())
    revised, other = fit_residual_models(modified, settings())
    for name in original:
        pd.testing.assert_frame_equal(original[name].iloc[:-40], revised[name].iloc[:-40])
    cutoff = frame.index[-40]
    pd.testing.assert_frame_equal(diagnostics[diagnostics.date < cutoff], other[other.date < cutoff])
    assert (diagnostics.training_end < diagnostics.date).all()


@pytest.mark.parametrize('pair', [[(-1, 1)], [(0, 0)], [(np.nan, 1)]])
def test_invalid_noise_rejected(pair):
    frame = sample_returns(100)
    with pytest.raises(ValueError):
        filter_paths(Basis.estimate(frame.iloc[:60], .55), frame.iloc[60:], pair)
