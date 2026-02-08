import numpy as np

from nllfit.distributions import LaplaceWrapper
from nllfit.metrics import laplace_nll
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


def test_laplace_nll_simple():
    y = np.array([1.0, 1.0])
    mu = np.array([1.0, 1.0])
    var = np.array([2.0, 2.0])
    expected = float(np.log(2.0))
    np.testing.assert_allclose(laplace_nll(y, mu, var, include_const=True), expected, rtol=1e-12)


def test_laplace_nll_weighted_matches_replication():
    y = np.array([0.0, 1.0])
    mu = np.array([0.0, 0.0])
    var = np.array([2.0, 2.0])
    weights = np.array([1.0, 2.0])
    expanded_y = np.array([0.0, 1.0, 1.0])
    expanded_mu = np.array([0.0, 0.0, 0.0])
    expanded_var = np.array([2.0, 2.0, 2.0])
    np.testing.assert_allclose(
        laplace_nll(y, mu, var, sample_weight=weights, include_const=True),
        laplace_nll(expanded_y, expanded_mu, expanded_var, include_const=True),
        rtol=1e-12,
    )


def test_laplace_wrapper_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = rng.normal(size=50)

    wrapper = LaplaceWrapper(_ConstantDistEstimator())
    wrapper.fit(X, y)
    lo, hi = wrapper.predict_interval(X, alpha=0.2)

    assert lo.shape == y.shape
    assert hi.shape == y.shape
    assert np.all(hi > lo)
    assert np.isfinite(wrapper.nll(X, y))
