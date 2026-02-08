import numpy as np
import pytest

from nllfit.distributions import StudentTWrapper
from nllfit.metrics import student_t_nll
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


def test_student_t_nll_weighted_matches_replication():
    y = np.array([0.0, 1.0])
    mu = np.array([0.0, 0.0])
    var = np.array([1.0, 1.0])
    df = 5.0
    weights = np.array([1.0, 2.0])
    expanded_y = np.array([0.0, 1.0, 1.0])
    expanded_mu = np.array([0.0, 0.0, 0.0])
    expanded_var = np.array([1.0, 1.0, 1.0])
    assert student_t_nll(y, mu, var, df, sample_weight=weights, include_const=True) == student_t_nll(
        expanded_y,
        expanded_mu,
        expanded_var,
        df,
        include_const=True,
    )


def test_student_t_wrapper_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)

    wrapper = StudentTWrapper(_ConstantDistEstimator(), df=5.0)
    wrapper.fit(X, y)
    assert np.isfinite(wrapper.nll(X, y, include_const=True))


def test_student_t_quantiles_with_scipy():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=10)
    wrapper = StudentTWrapper(_ConstantDistEstimator(), df=5.0)
    wrapper.fit(X, y)
    qs = wrapper.predict_quantiles(X, [0.1, 0.9])
    assert qs.shape == (len(X), 2)
