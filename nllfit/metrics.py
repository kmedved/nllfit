from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

from .validation import as_1d_float, validate_1d_same_length, validate_sample_weight

# NOTE: This module uses a polynomial approximation to erf (Abramowitz & Stegun,
# max error ~1.5e-7) to avoid requiring scipy. If scipy is already a dependency
# in your environment, you can replace _erf_approx with scipy.special.erf for
# full precision.

ArrayLike = Union[np.ndarray, "np.typing.ArrayLike"]


def gaussian_nll(
    y: ArrayLike,
    mu: ArrayLike,
    var: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    eps: float = 1e-12,
    include_const: bool = False,
) -> float:
    """Mean Gaussian negative log-likelihood.

    Per sample (up to constant):
        0.5 * (log(var_i) + (y_i - mu_i)^2 / var_i)

    If include_const=True, adds 0.5 * log(2*pi) per sample.

    Parameters
    ----------
    y, mu, var:
        1D arrays of targets, predicted mean, predicted variance.
    sample_weight:
        Optional 1D nonnegative weights (treated as frequency weights). Returned
        value is a weighted average.
    eps:
        Lower bound for variance clipping for numerical stability.
    include_const:
        Whether to include 0.5 * log(2*pi) constant.

    Returns
    -------
    float
        Average NLL.
    """
    y_ = as_1d_float("y", y)
    mu_ = as_1d_float("mu", mu)
    var_ = as_1d_float("var", var)
    validate_1d_same_length(y_, mu=mu_, var=var_)

    var_ = np.clip(var_, eps, np.inf)
    per = 0.5 * (np.log(var_) + (y_ - mu_) ** 2 / var_)

    if include_const:
        per = per + 0.5 * np.log(2.0 * np.pi)

    if sample_weight is None:
        return float(per.mean())

    w = validate_sample_weight(y_, sample_weight)
    return float(np.average(per, weights=w))


def laplace_nll(
    y: ArrayLike,
    mu: ArrayLike,
    var: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    eps: float = 1e-12,
    include_const: bool = False,
) -> float:
    """Mean Laplace negative log-likelihood (variance parameterization).

    For Laplace(mu, b): var = 2 * b^2, so b = sqrt(var / 2).

    Parameters
    ----------
    y, mu, var:
        1D arrays of targets, predicted mean, predicted variance.
    sample_weight:
        Optional 1D nonnegative weights (treated as frequency weights). Returned
        value is a weighted average.
    eps:
        Lower bound for variance clipping for numerical stability.
    include_const:
        Whether to include the log(2) constant (the log(b) term remains).
    """
    y_ = as_1d_float("y", y)
    mu_ = as_1d_float("mu", mu)
    var_ = as_1d_float("var", var)
    validate_1d_same_length(y_, mu=mu_, var=var_)

    var_ = np.clip(var_, eps, np.inf)
    b = np.sqrt(var_ / 2.0)
    b = np.clip(b, np.sqrt(eps / 2.0), np.inf)

    per = np.log(2.0 * b) + np.abs(y_ - mu_) / b
    if not include_const:
        per = per - np.log(2.0)

    if sample_weight is None:
        return float(per.mean())
    w = validate_sample_weight(y_, sample_weight)
    return float(np.average(per, weights=w))


def student_t_nll(
    y: ArrayLike,
    mu: ArrayLike,
    var: ArrayLike,
    df: float,
    *,
    sample_weight: Optional[ArrayLike] = None,
    eps: float = 1e-12,
    include_const: bool = False,
) -> float:
    """Mean Student-t negative log-likelihood (variance parameterization).

    For df > 2, var = scale^2 * df / (df - 2).
    """
    if df <= 2:
        raise ValueError("student_t_nll requires df > 2 when var is a variance parameter.")

    y_ = as_1d_float("y", y)
    mu_ = as_1d_float("mu", mu)
    var_ = as_1d_float("var", var)
    validate_1d_same_length(y_, mu=mu_, var=var_)

    scale2 = var_ * (df - 2.0) / df
    scale2 = np.clip(scale2, eps, np.inf)
    scale = np.sqrt(scale2)

    z2 = ((y_ - mu_) / scale) ** 2
    per = np.log(scale) + 0.5 * (df + 1.0) * np.log1p(z2 / df)

    if include_const:
        const = 0.5 * math.log(df * math.pi) + math.lgamma(df / 2.0) - math.lgamma((df + 1.0) / 2.0)
        per = per + const

    if sample_weight is None:
        return float(per.mean())
    w = validate_sample_weight(y_, sample_weight)
    return float(np.average(per, weights=w))


def lognormal_nll(
    y: ArrayLike,
    mu_log: ArrayLike,
    var_log: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    eps: float = 1e-12,
    include_const: bool = False,
) -> float:
    """Mean lognormal negative log-likelihood for log(y) ~ Normal(mu_log, var_log)."""
    y_ = as_1d_float("y", y)
    if np.any(y_ <= 0.0):
        raise ValueError("lognormal_nll requires y > 0.")
    mu_ = as_1d_float("mu_log", mu_log)
    var_ = as_1d_float("var_log", var_log)
    validate_1d_same_length(y_, mu_log=mu_, var_log=var_)

    var_ = np.clip(var_, eps, np.inf)
    t = np.log(y_)
    per = 0.5 * (np.log(var_) + (t - mu_) ** 2 / var_) + t
    if include_const:
        per = per + 0.5 * np.log(2.0 * np.pi)

    if sample_weight is None:
        return float(per.mean())
    w = validate_sample_weight(y_, sample_weight)
    return float(np.average(per, weights=w))


def log1p_normal_nll(
    y: ArrayLike,
    mu_log: ArrayLike,
    var_log: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    eps: float = 1e-12,
    include_const: bool = False,
) -> float:
    """Mean NLL for log1p(y) ~ Normal(mu_log, var_log)."""
    y_ = as_1d_float("y", y)
    if np.any(y_ < 0.0):
        raise ValueError("log1p_normal_nll requires y >= 0.")
    mu_ = as_1d_float("mu_log", mu_log)
    var_ = as_1d_float("var_log", var_log)
    validate_1d_same_length(y_, mu_log=mu_, var_log=var_)

    var_ = np.clip(var_, eps, np.inf)
    t = np.log1p(y_)
    per = 0.5 * (np.log(var_) + (t - mu_) ** 2 / var_) + np.log1p(y_)
    if include_const:
        per = per + 0.5 * np.log(2.0 * np.pi)

    if sample_weight is None:
        return float(per.mean())
    w = validate_sample_weight(y_, sample_weight)
    return float(np.average(per, weights=w))


def rmse(
    y: ArrayLike,
    mu: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Root mean squared error (optionally weighted)."""
    y_ = as_1d_float("y", y)
    mu_ = as_1d_float("mu", mu)
    validate_1d_same_length(y_, mu=mu_)
    sq = (y_ - mu_) ** 2
    if sample_weight is None:
        return float(np.sqrt(sq.mean()))
    w = validate_sample_weight(y_, sample_weight)
    return float(np.sqrt(np.average(sq, weights=w)))


def mae(
    y: ArrayLike,
    mu: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Mean absolute error (optionally weighted)."""
    y_ = as_1d_float("y", y)
    mu_ = as_1d_float("mu", mu)
    validate_1d_same_length(y_, mu=mu_)
    ae = np.abs(y_ - mu_)
    if sample_weight is None:
        return float(ae.mean())
    w = validate_sample_weight(y_, sample_weight)
    return float(np.average(ae, weights=w))


def interval_coverage(
    y: ArrayLike,
    lo: ArrayLike,
    hi: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Fraction of y values falling within [lo, hi] (optionally weighted)."""
    y_ = as_1d_float("y", y)
    lo_ = as_1d_float("lo", lo)
    hi_ = as_1d_float("hi", hi)
    validate_1d_same_length(y_, lo=lo_, hi=hi_)
    covered = ((y_ >= lo_) & (y_ <= hi_)).astype(float)
    if sample_weight is None:
        return float(covered.mean())
    w = validate_sample_weight(y_, sample_weight)
    return float(np.average(covered, weights=w))


def interval_width(
    lo: ArrayLike,
    hi: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Mean interval width (optionally weighted)."""
    lo_ = as_1d_float("lo", lo)
    hi_ = as_1d_float("hi", hi)
    validate_1d_same_length(lo_, hi=hi_)
    widths = hi_ - lo_
    if sample_weight is None:
        return float(widths.mean())
    w = validate_sample_weight(lo_, sample_weight)
    return float(np.average(widths, weights=w))


def crps_gaussian(
    y: ArrayLike,
    mu: ArrayLike,
    var: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    eps: float = 1e-12,
) -> float:
    """Closed-form CRPS for Gaussian predictive distributions.

    CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))

    where z = (y - mu) / sigma, Phi = standard normal CDF, phi = standard normal PDF.
    Lower is better.
    """
    y_ = as_1d_float("y", y)
    mu_ = as_1d_float("mu", mu)
    var_ = as_1d_float("var", var)
    validate_1d_same_length(y_, mu=mu_, var=var_)
    var_ = np.clip(var_, eps, np.inf)
    sigma = np.sqrt(var_)
    z = (y_ - mu_) / sigma

    phi_z = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    cdf_z = 0.5 * (1.0 + _erf_approx(z / np.sqrt(2.0)))

    per = sigma * (z * (2.0 * cdf_z - 1.0) + 2.0 * phi_z - 1.0 / np.sqrt(np.pi))

    if sample_weight is None:
        return float(per.mean())
    w = validate_sample_weight(y_, sample_weight)
    return float(np.average(per, weights=w))


def pit_gaussian(
    y: ArrayLike,
    mu: ArrayLike,
    var: ArrayLike,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Probability Integral Transform values for Gaussian predictions.

    Returns Phi((y - mu) / sigma) for each sample. Uniform(0,1) if
    calibrated.
    """
    y_ = as_1d_float("y", y)
    mu_ = as_1d_float("mu", mu)
    var_ = as_1d_float("var", var)
    validate_1d_same_length(y_, mu=mu_, var=var_)
    var_ = np.clip(var_, eps, np.inf)
    z = (y_ - mu_) / np.sqrt(var_)
    return 0.5 * (1.0 + _erf_approx(z / np.sqrt(2.0)))


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun approximation to erf (max error ~1.5e-7).

    Used to avoid a scipy dependency for CDF/CRPS computation.
    """
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = np.sign(x)
    x_abs = np.abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    result = 1.0 - poly * np.exp(-x_abs ** 2)
    return sign * result
