import numpy as np
from sklearn.linear_model import LinearRegression

from nllfit.oof import oof_mean_predictions, oof_squared_residuals


def test_oof_with_string_groups():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 2))
    y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=8)
    groups = np.array(["a", "a", "b", "b", "a", "a", "b", "b"], dtype=object)

    def model_factory():
        return LinearRegression()

    mu_oof = oof_mean_predictions(X, y, model_factory=model_factory, n_splits=5, groups=groups)
    res2 = oof_squared_residuals(X, y, model_factory=model_factory, n_splits=5, groups=groups)

    assert mu_oof.shape == y.shape
    assert res2.shape == y.shape


def test_oof_raises_on_too_few_splits():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 2))
    y = rng.normal(size=3)

    def model_factory():
        return LinearRegression()

    try:
        oof_mean_predictions(X, y, model_factory=model_factory, n_splits=1)
    except ValueError as exc:
        assert "Need at least 2 samples" in str(exc)
    else:
        raise AssertionError("Expected ValueError for too many splits.")
