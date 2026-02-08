"""Prove that weighted training is consistent with row duplication."""
import numpy as np
import pytest


def test_weighted_nll_equals_duplicated():
    """Weighted gaussian_nll should equal unweighted NLL on duplicated rows."""
    from nllfit.metrics import gaussian_nll

    rng = np.random.default_rng(0)
    n = 50
    y_base = rng.normal(size=n)
    mu_base = rng.normal(size=n)
    var_base = np.exp(rng.normal(scale=0.3, size=n))

    counts = rng.integers(1, 5, size=n)

    nll_weighted = gaussian_nll(y_base, mu_base, var_base, sample_weight=counts.astype(float))

    y_dup = np.repeat(y_base, counts)
    mu_dup = np.repeat(mu_base, counts)
    var_dup = np.repeat(var_base, counts)
    nll_dup = gaussian_nll(y_dup, mu_dup, var_dup)

    np.testing.assert_allclose(nll_weighted, nll_dup, rtol=1e-10)


def test_weighted_crps_equals_duplicated():
    """Weighted CRPS should equal unweighted CRPS on duplicated rows."""
    from nllfit.metrics import crps_gaussian

    rng = np.random.default_rng(1)
    n = 50
    y_base = rng.normal(size=n)
    mu_base = rng.normal(size=n)
    var_base = np.exp(rng.normal(scale=0.3, size=n))
    counts = rng.integers(1, 5, size=n)

    crps_w = crps_gaussian(y_base, mu_base, var_base, sample_weight=counts.astype(float))

    y_dup = np.repeat(y_base, counts)
    mu_dup = np.repeat(mu_base, counts)
    var_dup = np.repeat(var_base, counts)
    crps_dup = crps_gaussian(y_dup, mu_dup, var_dup)

    np.testing.assert_allclose(crps_w, crps_dup, rtol=1e-10)


def test_weighted_lgbm_close_to_duplicated():
    """LightGBM with weights should produce similar predictions to duplicated rows.

    Not exact due to tree-building heuristics, but predictions should be close.
    """
    pytest.importorskip("lightgbm")
    from nllfit.estimators.lightgbm import TwoStageHeteroscedasticLightGBM

    rng = np.random.default_rng(0)
    n = 100
    X_base = rng.normal(size=(n, 3))
    y_base = X_base @ [1.0, 0.5, -0.3] + rng.normal(scale=0.5, size=n)
    counts = rng.integers(1, 4, size=n)

    params = dict(
        calibration_method="none",
        random_state=0,
        mean_params={"n_estimators": 200, "verbose": -1, "random_state": 0},
        var_params={"n_estimators": 200, "verbose": -1, "random_state": 0},
    )

    m_w = TwoStageHeteroscedasticLightGBM(**params)
    m_w.fit(X_base, y_base, sample_weight=counts.astype(float))
    pred_w = m_w.predict(X_base)
    dist_w = m_w.predict_dist(X_base)

    X_dup = np.repeat(X_base, counts, axis=0)
    y_dup = np.repeat(y_base, counts)
    m_d = TwoStageHeteroscedasticLightGBM(**params)
    m_d.fit(X_dup, y_dup)
    pred_d = m_d.predict(X_base)
    dist_d = m_d.predict_dist(X_base)

    corr_mu = np.corrcoef(pred_w, pred_d)[0, 1]
    assert corr_mu > 0.98, f"Weighted vs duplicated mean correlation = {corr_mu:.4f}"

    corr_var = np.corrcoef(dist_w.var, dist_d.var)[0, 1]
    assert corr_var > 0.90, f"Weighted vs duplicated variance correlation = {corr_var:.4f}"
