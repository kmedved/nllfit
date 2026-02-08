import numpy as np
from nllfit.metrics import (
    crps_gaussian,
    gaussian_nll,
    interval_coverage,
    interval_width,
    mae,
    pit_gaussian,
    rmse,
)


def test_gaussian_nll_sanity_constant_var():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    mu = np.zeros_like(y)
    var = np.ones_like(y)

    nll = gaussian_nll(y, mu, var, include_const=False)
    # Should be close to 0.5 * E[y^2] when var=1 and mu=0
    assert abs(nll - 0.5 * np.mean(y**2)) < 1e-6


def test_gaussian_nll_weighted_average():
    y = np.array([0.0, 1.0, 2.0])
    mu = np.zeros_like(y)
    var = np.ones_like(y)
    weights = np.array([1.0, 2.0, 1.0])

    per = 0.5 * (np.log(var) + (y - mu) ** 2 / var)
    expected = float(np.average(per, weights=weights))
    assert gaussian_nll(y, mu, var, sample_weight=weights) == expected


def test_rmse_basic():
    y = np.array([1.0, 2.0, 3.0])
    mu = np.array([1.0, 2.0, 3.0])
    assert rmse(y, mu) == 0.0


def test_rmse_weighted():
    y = np.array([0.0, 0.0])
    mu = np.array([1.0, 2.0])
    assert abs(rmse(y, mu) - np.sqrt(2.5)) < 1e-10
    # Zero weight on first sample: weighted RMSE is entirely determined by
    # the second sample (error=2.0). This is intentional — it exercises the
    # edge case where individual weights are zero but total sum is positive.
    w = np.array([0.0, 1.0])
    assert abs(rmse(y, mu, sample_weight=w) - 2.0) < 1e-10


def test_mae_basic():
    y = np.array([0.0, 0.0])
    mu = np.array([1.0, -3.0])
    assert abs(mae(y, mu) - 2.0) < 1e-10


def test_interval_coverage_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.5, 1.5, 3.5, 3.5])
    hi = np.array([1.5, 2.5, 3.5, 4.5])
    assert abs(interval_coverage(y, lo, hi) - 0.75) < 1e-10


def test_interval_width_basic():
    lo = np.array([0.0, 1.0])
    hi = np.array([2.0, 5.0])
    assert abs(interval_width(lo, hi) - 3.0) < 1e-10


def test_crps_gaussian_standard_normal():
    rng = np.random.default_rng(42)
    n = 100_000
    y = rng.normal(size=n)
    mu = np.zeros(n)
    var = np.ones(n)
    expected_crps = 1.0 / np.sqrt(np.pi)
    assert abs(crps_gaussian(y, mu, var) - expected_crps) < 0.01


def test_pit_gaussian_uniform():
    rng = np.random.default_rng(0)
    n = 50_000
    mu = rng.normal(size=n)
    var = np.exp(rng.normal(scale=0.3, size=n))
    y = mu + rng.normal(scale=np.sqrt(var))
    pit = pit_gaussian(y, mu, var)
    assert abs(pit.mean() - 0.5) < 0.01
    assert abs(pit.std() - 1.0 / np.sqrt(12)) < 0.01
