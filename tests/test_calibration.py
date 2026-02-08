import numpy as np
from nllfit.calibration import fit_variance_scale, apply_variance_calibration
from nllfit.metrics import gaussian_nll


def test_variance_scale_closed_form_reduces_nll():
    rng = np.random.default_rng(0)
    n = 2000
    mu = rng.normal(size=n)
    true_var = np.exp(rng.normal(scale=0.2, size=n))
    y = mu + rng.normal(scale=np.sqrt(true_var))

    # Underconfident raw variance (too small)
    var_raw = true_var * 0.25

    before = gaussian_nll(y, mu, var_raw)
    cal = fit_variance_scale(y, mu, var_raw)
    var_cal = apply_variance_calibration(var_raw, cal)
    after = gaussian_nll(y, mu, var_cal)

    assert after <= before


def test_variance_scale_weighted():
    y = np.array([0.0, 0.0, 2.0, 2.0])
    mu = np.zeros_like(y)
    var_raw = np.ones_like(y)
    weights = np.array([1.0, 1.0, 3.0, 3.0])

    cal = fit_variance_scale(y, mu, var_raw, sample_weight=weights)
    assert cal.scale > 1.0
