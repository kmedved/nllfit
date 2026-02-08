import numpy as np
import pytest

from nllfit.splitting import infer_time, time_sort, calibration_split

pd = pytest.importorskip("pandas")


def test_time_sort_and_time_quantile_split():
    n = 100
    X = pd.DataFrame({"x": np.arange(n)})
    X["t"] = pd.date_range("2020-01-01", periods=n, freq="D")
    # scramble rows
    X = X.sample(frac=1.0, random_state=0).reset_index(drop=True)
    y = np.arange(n).astype(float)

    time = infer_time(X, time_col="t")
    Xs, ys, _, _, order = time_sort(X, y, None, None, time)
    assert order is not None
    assert np.all(np.diff(Xs["t"].values.astype("datetime64[ns]")) >= np.timedelta64(0, "ns"))

    X_tr, X_cal, y_tr, y_cal, *_ = calibration_split(
        Xs, ys, sample_weight=None, groups=None, time=time, calibration_fraction=0.2, random_state=0
    )
    assert len(y_cal) > 0 and len(y_tr) > 0


def test_calibration_split_weight_lengths():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = rng.normal(size=20)
    w = rng.uniform(0.1, 1.0, size=20)

    X_tr, X_cal, y_tr, y_cal, w_tr, w_cal, *_ = calibration_split(
        X, y, sample_weight=w, groups=None, time=infer_time(X), calibration_fraction=0.25, random_state=0
    )

    assert len(y_tr) == len(w_tr)
    assert len(y_cal) == len(w_cal)
