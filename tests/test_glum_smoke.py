import numpy as np
import pytest

pytest.importorskip("glum")

import pandas as pd
from hetero_nll import TwoStageHeteroscedasticGLUM


def test_glum_fit_predict_smoke():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    true_var = np.exp(0.3 + 0.7 * X["x1"].values)
    y = (2 * X["x1"] - X["x2"]).values + rng.normal(scale=np.sqrt(true_var))

    m = TwoStageHeteroscedasticGLUM(calibrate=False)
    m.fit(X, y)
    pred = m.predict_dist(X)

    assert pred.mu.shape == (n,)
    assert pred.var.shape == (n,)
    assert np.all(pred.var > 0)
