import numpy as np
import warnings
import pytest

pytest.importorskip("glum")

import pandas as pd
from nllfit import TwoStageHeteroscedasticGLUM


def _make_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    true_var = np.exp(0.3 + 0.7 * X["x1"].values)
    y = (2 * X["x1"] - X["x2"]).values + rng.normal(scale=np.sqrt(true_var))
    return X, y, true_var


def test_glum_fit_predict_smoke():
    X, y, _ = _make_data()
    n = len(y)

    m = TwoStageHeteroscedasticGLUM(calibration_method="none")
    m.fit(X, y)
    pred = m.predict_dist(X)

    assert pred.mu.shape == (n,)
    assert pred.var.shape == (n,)
    assert np.all(pred.var > 0)


def test_glum_calibration_method_holdout():
    X, y, _ = _make_data(n=1000)

    m = TwoStageHeteroscedasticGLUM(
        calibration_method="holdout",
        calibration_fraction=0.2,
    )
    m.fit(X, y)

    assert m.calibration_.source == "holdout"
    assert m.calibration_.scale > 0


def test_glum_calibration_method_train():
    """Train cal is safe for GLMs — scale should be near 1.0."""
    X, y, _ = _make_data(n=1000)

    m = TwoStageHeteroscedasticGLUM(calibration_method="train")
    m.fit(X, y)

    assert m.calibration_.source == "train"
    # GLMs don't overfit, so train scale should be near 1.0
    assert 0.8 < m.calibration_.scale < 1.2


def test_glum_calibration_method_none():
    X, y, _ = _make_data()

    m = TwoStageHeteroscedasticGLUM(calibration_method="none")
    m.fit(X, y)

    assert m.calibration_.source == "none"
    assert m.calibration_.scale == 1.0


def test_glum_calibration_method_oof_falls_back():
    """OOF cal not supported for GLM — should warn and fall back to none."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = TwoStageHeteroscedasticGLUM(calibration_method="oof")

    assert any("not supported for GLM" in str(warning.message) for warning in w)
    assert m.calibration_method == "none"


def test_glum_calibration_method_invalid_raises():
    with pytest.raises(ValueError, match="calibration_method must be"):
        m = TwoStageHeteroscedasticGLUM(calibration_method="bad")
        m.fit(pd.DataFrame({"x": [1, 2, 3]}), np.array([1.0, 2.0, 3.0]))


def test_glum_deprecated_calibrate_param():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = TwoStageHeteroscedasticGLUM(calibrate=False)

    assert any("deprecated" in str(warning.message) for warning in w)
    assert m.calibration_method == "none"


def test_glum_deprecated_calibrate_true_with_fraction():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = TwoStageHeteroscedasticGLUM(
            calibrate=True,
            calibration_fraction=0.2,
        )

    assert any("deprecated" in str(warning.message) for warning in w)
    assert m.calibration_method == "holdout"


def test_glum_explicit_cal_data_overrides():
    X, y, _ = _make_data(n=1000)

    from sklearn.model_selection import train_test_split
    X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.2, random_state=0)

    m = TwoStageHeteroscedasticGLUM(calibration_method="none")
    m.fit(X_tr, y_tr, X_cal=X_cal, y_cal=y_cal)

    assert m.calibration_.source == "holdout"
    assert m.calibration_.scale > 0
