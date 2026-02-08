import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from nllfit.conformal import SplitConformalRegressor
from nllfit.conformal import _conformal_quantile
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


class _PerfectIntervalEstimator:
    """Estimator whose intervals perfectly cover y when X[:, 0] == y."""

    def fit(self, X, y, **kw):
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]

    def predict_interval(self, X, alpha=0.1):
        mu = self.predict(X)
        return mu - 0.5, mu + 0.5


class _DistEstimator(_SimpleEstimator):
    def predict_dist(self, X):
        from types import SimpleNamespace
        mu = self.predict(X)
        return SimpleNamespace(mu=mu, var=np.ones_like(mu))


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
    assert cov >= (1.0 - alpha) - 0.06, f"Coverage {cov:.3f} too low"


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

    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] + rng.normal(size=200)

    cr = SplitConformalRegressor(_DistEstimator(), method="absolute", alpha=0.1)
    cr.fit(X, y)
    dist = cr.predict_dist(X[:10])
    assert hasattr(dist, "var")
    assert len(dist.mu) == 10


def test_normalized_conformal_with_sample_weight():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] + rng.normal(size=200)
    weights = rng.uniform(0.5, 2.0, size=200)

    cr = SplitConformalRegressor(_DistEstimator(), method="normalized", alpha=0.1)
    cr.fit(X, y, sample_weight=weights)
    assert cr.q_ >= 0.0


def test_conformal_predict_dist_raises_without_base():
    """predict_dist should raise if base estimator lacks it."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] + rng.normal(size=200)

    cr = SplitConformalRegressor(_SimpleEstimator(), method="absolute", alpha=0.1)
    cr.fit(X, y)
    with pytest.raises(TypeError, match="predict_dist"):
        cr.predict_dist(X[:10])


def test_cqr_scores_nonnegative_and_q_zero():
    X = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    y = X[:, 0].copy()

    cr = SplitConformalRegressor(_PerfectIntervalEstimator(), method="cqr", alpha=0.1)
    cr.fit(X, y)
    assert cr.q_ == pytest.approx(0.0)

    lo_base, hi_base = cr.base_estimator.predict_interval(X, alpha=0.1)
    lo, hi = cr.predict_interval(X)
    np.testing.assert_allclose(lo, lo_base)
    np.testing.assert_allclose(hi, hi_base)


def test_conformal_quantile_order_statistic_ties():
    scores = np.array([0.0, 1.0, 1.0, 2.0])
    alpha = 0.2
    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    expected = np.sort(scores)[k - 1]
    assert _conformal_quantile(scores, alpha) == expected


def test_conformal_quantile_monotone_in_alpha():
    scores = np.array([0.1, 0.2, 0.5, 0.9])
    q_lo = _conformal_quantile(scores, 0.05)
    q_hi = _conformal_quantile(scores, 0.2)
    assert q_lo >= q_hi


def test_conformal_quantile_weighted_matches_unweighted_when_equal():
    scores = np.array([0.1, 0.3, 0.5, 0.8])
    weights = np.ones_like(scores)
    q_unweighted = _conformal_quantile(scores, 0.2)
    q_weighted = _conformal_quantile(scores, 0.2, sample_weight=weights)
    assert q_weighted == q_unweighted


def test_conformal_quantile_weighted_matches_expanded():
    scores = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 2.0, 1.0])
    expanded = np.array([0.0, 1.0, 1.0, 2.0])
    alpha = 0.2
    q_weighted = _conformal_quantile(scores, alpha, sample_weight=weights)
    q_expanded = _conformal_quantile(expanded, alpha)
    assert q_weighted == q_expanded


def test_conformal_quantile_weighted_boundary_replication():
    """Regression test: weighted quantile must match expansion for integer weights.

    This is the exact counterexample that caught the side='right' bug:
    scores=[0,1,2], weights=[1,2,1] expands to [0,1,1,2].
    At alpha=0.4, unweighted on expanded gives 1, not 2.
    """
    scores = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 2.0, 1.0])
    expanded = np.array([0.0, 1.0, 1.0, 2.0])
    alpha = 0.4
    q_weighted = _conformal_quantile(scores, alpha, sample_weight=weights)
    q_expanded = _conformal_quantile(expanded, alpha)
    assert q_weighted == q_expanded, (
        f"Weighted quantile {q_weighted} != expanded quantile {q_expanded} "
        f"for alpha={alpha}, weights={weights.tolist()}"
    )
