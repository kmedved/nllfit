import numpy as np
import pytest

from nllfit.lognormal import LogNormalRegressor
from nllfit.metrics import log1p_normal_nll, lognormal_nll
from nllfit.types import HeteroscedasticPrediction


class _ConstantDistEstimator:
    def fit(self, X, y, **kwargs):
        self.mu_ = float(np.mean(y))
        self.var_ = float(np.var(y) + 1e-6)
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_)

    def predict_dist(self, X):
        mu = np.full(len(X), self.mu_)
        var = np.full(len(X), self.var_)
        return HeteroscedasticPrediction(mu=mu, var=var, log_var=np.log(var))


def test_lognormal_nll_domain():
    y = np.array([1.0, 0.0])
    mu = np.zeros_like(y)
    var = np.ones_like(y)
    with pytest.raises(ValueError, match="y > 0"):
        lognormal_nll(y, mu, var)


def test_log1p_normal_nll_domain():
    y = np.array([1.0, -0.1])
    mu = np.zeros_like(y)
    var = np.ones_like(y)
    with pytest.raises(ValueError, match="y >= 0"):
        log1p_normal_nll(y, mu, var)


def test_lognormal_nll_weighted_matches_replication():
    y = np.array([1.0, 2.0])
    mu = np.array([0.0, 0.0])
    var = np.array([1.0, 1.0])
    weights = np.array([1.0, 2.0])
    expanded_y = np.array([1.0, 2.0, 2.0])
    expanded_mu = np.array([0.0, 0.0, 0.0])
    expanded_var = np.array([1.0, 1.0, 1.0])
    assert lognormal_nll(y, mu, var, sample_weight=weights, include_const=True) == lognormal_nll(
        expanded_y,
        expanded_mu,
        expanded_var,
        include_const=True,
    )


def test_lognormal_regressor_quantiles_monotone():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = np.exp(rng.normal(size=20))

    model = LogNormalRegressor(_ConstantDistEstimator(), transform="log")
    model.fit(X, y)
    qs = model.predict_quantiles(X[:5], [0.1, 0.5, 0.9])
    assert np.all(qs[:, 0] < qs[:, 1])
    assert np.all(qs[:, 1] < qs[:, 2])


def test_lognormal_regressor_smoke():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = np.exp(rng.normal(size=30))

    model = LogNormalRegressor(_ConstantDistEstimator(), transform="log")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    lo, hi = model.predict_interval(X, alpha=0.2)
    assert np.all(hi > lo)
