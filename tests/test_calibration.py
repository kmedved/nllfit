import numpy as np
from hetero_nll.calibration import fit_variance_scale, apply_variance_calibration
from hetero_nll.metrics import gaussian_nll


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
