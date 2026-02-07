from __future__ import annotations

from typing import Optional, Union

import numpy as np

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
        Optional 1D nonnegative weights. Returned value is a weighted average.
    eps:
        Lower bound for variance clipping for numerical stability.
    include_const:
        Whether to include 0.5 * log(2*pi) constant.

    Returns
    -------
    float
        Average NLL.
    """
    y_ = np.asarray(y, dtype=float).reshape(-1)
    mu_ = np.asarray(mu, dtype=float).reshape(-1)
    var_ = np.asarray(var, dtype=float).reshape(-1)

    var_ = np.clip(var_, eps, np.inf)
    per = 0.5 * (np.log(var_) + (y_ - mu_) ** 2 / var_)

    if include_const:
        per = per + 0.5 * np.log(2.0 * np.pi)

    if sample_weight is None:
        return float(per.mean())

    w = np.asarray(sample_weight, dtype=float).reshape(-1)
    return float(np.average(per, weights=w))
