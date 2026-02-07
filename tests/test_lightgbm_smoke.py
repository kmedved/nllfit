import numpy as np
import warnings
import pytest

pytest.importorskip("lightgbm")

import pandas as pd
from nllfit import TwoStageHeteroscedasticLightGBM
from nllfit.metrics import gaussian_nll


def _make_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    true_var = np.exp(0.3 + 0.7 * X["x1"].values)
    y = (2 * X["x1"] - X["x2"]).values + rng.normal(scale=np.sqrt(true_var))
    return X, y, true_var


def test_lightgbm_fit_predict_smoke():
    X, y, _ = _make_data()
    n = len(y)

    m = TwoStageHeteroscedasticLightGBM(
        n_iterations=1,
        n_oof_folds=3,
        variance_mode="auto",
        calibration_method="none",
    )
    m.fit(X, y)
    pred = m.predict_dist(X)

    assert pred.mu.shape == (n,)
    assert pred.var.shape == (n,)
    assert np.all(pred.var > 0)


def test_calibration_method_oof():
    """OOF calibration should produce scale near 1.0 for 1-iter."""
    X, y, _ = _make_data(n=1000)

    m = TwoStageHeteroscedasticLightGBM(
        n_iterations=1,
        n_oof_folds=5,
        calibration_method="oof",
    )
    m.fit(X, y)

    assert m.calibration_.source == "oof"
    # OOF cal at 1 iter should be very close to 1.0
    assert 0.8 < m.calibration_.scale < 1.2


def test_calibration_method_holdout():
    X, y, _ = _make_data(n=1000)

    m = TwoStageHeteroscedasticLightGBM(
        n_iterations=1,
        n_oof_folds=5,
        calibration_method="holdout",
        calibration_fraction=0.2,
    )
    m.fit(X, y)

    assert m.calibration_.source == "holdout"
    assert m.calibration_.scale > 0


def test_calibration_method_train_warns():
    X, y, _ = _make_data()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = TwoStageHeteroscedasticLightGBM(
            n_iterations=1,
            n_oof_folds=3,
            calibration_method="train",
        )
        m.fit(X, y)

    assert any("systematic variance shrinkage" in str(warning.message) for warning in w)
    assert m.calibration_.source == "train"
    # Train cal with flexible model should produce scale < 1
    assert m.calibration_.scale < 0.8


def test_calibration_method_none():
    X, y, _ = _make_data()

    m = TwoStageHeteroscedasticLightGBM(
        n_iterations=1,
        n_oof_folds=3,
        calibration_method="none",
    )
    m.fit(X, y)

    assert m.calibration_.source == "none"
    assert m.calibration_.scale == 1.0


def test_calibration_method_invalid_raises():
    with pytest.raises(ValueError, match="calibration_method must be"):
        m = TwoStageHeteroscedasticLightGBM(calibration_method="bad")
        m.fit(pd.DataFrame({"x": [1, 2, 3]}), np.array([1.0, 2.0, 3.0]))


def test_deprecated_calibrate_param_warns():
    X, y, _ = _make_data()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = TwoStageHeteroscedasticLightGBM(
            n_oof_folds=3,
            calibrate=False,
        )

    assert any("calibrate" in str(warning.message) and "deprecated" in str(warning.message) for warning in w)
    assert m.calibration_method == "none"


def test_two_iter_oof_cal_improves_nll():
    """2 iterations with OOF cal should be at least as good as 1 iter no cal."""
    X, y, _ = _make_data(n=2000, seed=42)

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    m1 = TwoStageHeteroscedasticLightGBM(
        n_iterations=1, n_oof_folds=5, calibration_method="none",
    )
    m1.fit(X_tr, y_tr)
    nll1 = m1.nll(X_te, y_te)

    m2 = TwoStageHeteroscedasticLightGBM(
        n_iterations=2, n_oof_folds=5, calibration_method="oof",
    )
    m2.fit(X_tr, y_tr)
    nll2 = m2.nll(X_te, y_te)

    # 2 iter OOF cal should not be significantly worse
    assert nll2 < nll1 + 0.1


def test_explicit_cal_data_overrides():
    """Passing X_cal/y_cal should use holdout calibration regardless of calibration_method."""
    X, y, _ = _make_data(n=1000)

    from sklearn.model_selection import train_test_split
    X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.2, random_state=0)

    m = TwoStageHeteroscedasticLightGBM(
        n_iterations=1, n_oof_folds=3, calibration_method="none",
    )
    m.fit(X_tr, y_tr, X_cal=X_cal, y_cal=y_cal)

    assert m.calibration_.source == "holdout"
    assert m.calibration_.scale > 0
