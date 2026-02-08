import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from nllfit.conformal import SplitConformalRegressor
from nllfit.metrics import interval_coverage


class _SimpleEstimator:
    """Minimal estimator for testing."""

    def __init__(self):
        self._model = LinearRegression()

    def fit(self, X, y, **kw):
        self._model.fit(X, y, **kw)
        return self

    def predict(self, X):
        return self._model.predict(X)


def test_absolute_conformal_coverage():
    """Split conformal with absolute method should achieve ~(1-alpha) coverage."""
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, 0.5, -0.3]) + rng.normal(scale=0.5, size=n)

    alpha = 0.1
    cr = SplitConformalRegressor(
        _SimpleEstimator(),
        method="absolute",
        alpha=alpha,
        calibration_fraction=0.25,
    )

    cr.fit(X[:1500], y[:1500])
    lo, hi = cr.predict_interval(X[1500:])

    cov = interval_coverage(y[1500:], lo, hi)
    assert cov >= (1.0 - alpha) - 0.05, f"Coverage {cov:.3f} too low"


def test_absolute_conformal_with_explicit_cal():
    """Explicit calibration set should work."""
    rng = np.random.default_rng(1)
    n = 1000
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + rng.normal(scale=0.3, size=n)

    alpha = 0.1
    cr = SplitConformalRegressor(
        _SimpleEstimator(),
        method="absolute",
        alpha=alpha,
    )

    cr.fit(X[:600], y[:600], X_cal=X[600:800], y_cal=y[600:800])
    lo, hi = cr.predict_interval(X[800:])

    cov = interval_coverage(y[800:], lo, hi)
    assert cov >= (1.0 - alpha) - 0.05


def test_conformal_not_fitted_raises():
    cr = SplitConformalRegressor(_SimpleEstimator())
    with pytest.raises(RuntimeError, match="not been fitted"):
        cr.predict(np.zeros((5, 2)))


def test_conformal_normalized_requires_predict_dist():
    cr = SplitConformalRegressor(_SimpleEstimator(), method="normalized")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = rng.normal(size=100)
    with pytest.raises(TypeError, match="predict_dist"):
        cr.fit(X, y)


def test_conformal_q_is_positive():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 2))
    y = X[:, 0] + rng.normal(size=500)

    cr = SplitConformalRegressor(_SimpleEstimator(), method="absolute", alpha=0.1)
    cr.fit(X, y)
    assert cr.q_ > 0


def test_conformal_predict_dist_passthrough():
    """predict_dist should delegate to base estimator."""

    class _DistEstimator(_SimpleEstimator):
        def predict_dist(self, X):
            from types import SimpleNamespace
            mu = self.predict(X)
            return SimpleNamespace(mu=mu, var=np.ones_like(mu))

    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] + rng.normal(size=200)

    cr = SplitConformalRegressor(_DistEstimator(), method="absolute", alpha=0.1)
    cr.fit(X, y)
    dist = cr.predict_dist(X[:10])
    assert hasattr(dist, "var")
    assert len(dist.mu) == 10


def test_conformal_predict_dist_raises_without_base():
    """predict_dist should raise if base estimator lacks it."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] + rng.normal(size=200)

    cr = SplitConformalRegressor(_SimpleEstimator(), method="absolute", alpha=0.1)
    cr.fit(X, y)
    with pytest.raises(TypeError, match="predict_dist"):
        cr.predict_dist(X[:10])
