from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .types import VarianceCalibration
from .validation import as_1d_float, validate_1d_same_length, validate_sample_weight

ArrayLike = Union[np.ndarray, "np.typing.ArrayLike"]


def fit_variance_scale(
    y: ArrayLike,
    mu: ArrayLike,
    var_raw: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    eps: float = 1e-12,
    source: str = "holdout",
) -> VarianceCalibration:
    """Fit multiplicative variance scale minimizing Gaussian NLL.

    We calibrate var_cal = c * var_raw. sample_weight, when provided, is
    treated as frequency weights.

    Closed-form optimum:
        c* = E[(y-mu)^2 / var_raw]   (weighted average if sample_weight provided)

    Notes
    -----
    - This only rescales variance globally. It will not fix a misspecified
      variance *shape* function.
    """
    y_ = as_1d_float("y", y)
    mu_ = as_1d_float("mu", mu)
    var_ = as_1d_float("var_raw", var_raw)
    validate_1d_same_length(y_, mu=mu_, var_raw=var_)
    var_ = np.clip(var_, eps, np.inf)

    ratio = (y_ - mu_) ** 2 / var_

    if sample_weight is None:
        c = float(ratio.mean())
    else:
        w = validate_sample_weight(y_, sample_weight)
        c = float(np.average(ratio, weights=w))

    c = float(np.clip(c, eps, np.inf))
    return VarianceCalibration(scale=c, source=source)


def apply_variance_calibration(
    var_raw: ArrayLike,
    cal: VarianceCalibration,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply multiplicative calibration with clipping."""
    var_ = np.asarray(var_raw, dtype=float).reshape(-1)
    var_cal = cal.scale * var_
    return np.clip(var_cal, eps, np.inf)
