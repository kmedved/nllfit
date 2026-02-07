import numpy as np
import pytest

pytest.importorskip("lightgbm")

import pandas as pd
from hetero_nll import TwoStageHeteroscedasticLightGBM


def test_lightgbm_fit_predict_smoke():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    true_var = np.exp(0.3 + 0.7 * X["x1"].values)
    y = (2 * X["x1"] - X["x2"]).values + rng.normal(scale=np.sqrt(true_var))

    m = TwoStageHeteroscedasticLightGBM(
        n_oof_folds=3,
        variance_mode="auto",
        calibrate=False,
    )
    m.fit(X, y)
    pred = m.predict_dist(X)

    assert pred.mu.shape == (n,)
    assert pred.var.shape == (n,)
    assert np.all(pred.var > 0)
