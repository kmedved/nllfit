import numpy as np
from nllfit.metrics import gaussian_nll


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
