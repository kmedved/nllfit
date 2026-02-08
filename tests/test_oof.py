import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from nllfit.oof import choose_oof_splitter, oof_mean_predictions, oof_squared_residuals


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


def test_choose_oof_splitter_clamps_groups():
    X = np.zeros((5, 1))
    groups = np.array(["a", "a", "b", "b", "a"], dtype=object)
    with pytest.warns(UserWarning, match="clamping"):
        splitter = choose_oof_splitter(X, n_splits=5, random_state=0, groups=groups)
    assert splitter.n_splits == 2


def test_choose_oof_splitter_clamps_samples():
    X = np.zeros((3, 1))
    with pytest.warns(UserWarning, match="clamping"):
        splitter = choose_oof_splitter(X, n_splits=5, random_state=0, groups=None)
    assert splitter.n_splits == 3
