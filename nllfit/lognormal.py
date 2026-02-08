from __future__ import annotations

import copy
import importlib.util
from statistics import NormalDist
from typing import Dict, Literal, Optional, Sequence, Union

import numpy as np

if importlib.util.find_spec("pandas") is not None:
    import pandas as pd
else:  # pragma: no cover
    pd = None  # type: ignore

if importlib.util.find_spec("sklearn") is not None:
    from sklearn.base import BaseEstimator, RegressorMixin, clone
else:  # pragma: no cover
    BaseEstimator = object  # type: ignore
    RegressorMixin = object  # type: ignore
    clone = None  # type: ignore

from .metrics import log1p_normal_nll, lognormal_nll
from .types import LogNormalPrediction
from .validation import as_1d_float, validate_sample_weight

ArrayLike = Union[np.ndarray, "pd.DataFrame", "pd.Series"]
QuantileOutput = Literal["array", "dataframe", "dict"]


def _format_quantiles_output(
    arr: np.ndarray,
    qs: np.ndarray,
    *,
    output: QuantileOutput,
    column_prefix: str,
    X: ArrayLike,
) -> Union[np.ndarray, "pd.DataFrame", Dict[float, np.ndarray]]:
    if output == "array":
        return arr
    if output == "dict":
        return {float(q): arr[:, i].copy() for i, q in enumerate(qs)}
    if output == "dataframe":
        if pd is None:
            raise ImportError("pandas is required for output='dataframe'.")
        cols = [f"{column_prefix}{q:g}" for q in qs]
        idx = getattr(X, "index", None)
        return pd.DataFrame(arr, index=idx, columns=cols)
    raise ValueError(f"Unknown output={output!r}")


class LogNormalRegressor(BaseEstimator, RegressorMixin):
    """Fit a base estimator in log/log1p space and predict in original space."""

    def __init__(self, base_estimator, *, transform: str = "log", eps: float = 1e-12):
        self.base_estimator = base_estimator
        self.transform = transform
        self.eps = eps

    def fit(self, X: ArrayLike, y: np.ndarray, *, sample_weight=None, groups=None, **kwargs):
        y_ = as_1d_float("y", y)
        w_ = validate_sample_weight(y_, sample_weight)

        transform = self.transform.lower().strip()
        if transform == "log":
            if np.any(y_ <= 0.0):
                raise ValueError("transform='log' requires y > 0.")
            y_t = np.log(y_)
        elif transform == "log1p":
            if np.any(y_ < 0.0):
                raise ValueError("transform='log1p' requires y >= 0.")
            y_t = np.log1p(y_)
        else:
            raise ValueError("transform must be 'log' or 'log1p'.")

        if clone is not None and hasattr(self.base_estimator, "get_params"):
            self.base_estimator_ = clone(self.base_estimator)
        else:
            self.base_estimator_ = copy.deepcopy(self.base_estimator)

        fit_kw = dict(kwargs)
        if w_ is not None:
            fit_kw["sample_weight"] = w_
        if groups is not None:
            fit_kw["groups"] = groups
        self.base_estimator_.fit(X, y_t, **fit_kw)
        return self

    def predict_dist(self, X: ArrayLike) -> LogNormalPrediction:
        pred_t = self.base_estimator_.predict_dist(X)
        mu_log = np.asarray(pred_t.mu, dtype=float).ravel()
        var_log = np.clip(np.asarray(pred_t.var, dtype=float).ravel(), self.eps, np.inf)

        if self.transform.lower().strip() == "log":
            mean = np.exp(mu_log + 0.5 * var_log)
        else:
            mean = np.exp(mu_log + 0.5 * var_log) - 1.0

        var = (np.exp(var_log) - 1.0) * np.exp(2.0 * mu_log + var_log)
        var = np.clip(var, self.eps, np.inf)

        return LogNormalPrediction(
            mu=mean,
            var=var,
            log_var=np.log(var),
            mu_log=mu_log,
            var_log=var_log,
        )

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.predict_dist(X).mu

    def nll(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[np.ndarray] = None,
        include_const: bool = False,
    ) -> float:
        pred = self.predict_dist(X)
        if self.transform.lower().strip() == "log":
            return lognormal_nll(
                y,
                pred.mu_log,
                pred.var_log,
                sample_weight=sample_weight,
                eps=self.eps,
                include_const=include_const,
            )
        return log1p_normal_nll(
            y,
            pred.mu_log,
            pred.var_log,
            sample_weight=sample_weight,
            eps=self.eps,
            include_const=include_const,
        )

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[np.ndarray] = None) -> float:
        return -self.nll(X, y, sample_weight=sample_weight, include_const=False)

    def predict_quantiles(
        self,
        X: ArrayLike,
        quantiles: Sequence[float],
        *,
        output: QuantileOutput = "array",
        column_prefix: str = "q",
    ) -> Union[np.ndarray, "pd.DataFrame", Dict[float, np.ndarray]]:
        qs = np.asarray(list(quantiles), dtype=float).reshape(-1)
        if qs.size == 0:
            raise ValueError("quantiles must be non-empty.")
        if np.any(qs <= 0.0) or np.any(qs >= 1.0):
            raise ValueError("All quantiles must be strictly between 0 and 1.")

        pred = self.predict_dist(X)
        sd_log = np.sqrt(pred.var_log)
        z = np.array([NormalDist().inv_cdf(float(q)) for q in qs], dtype=float)
        q_log = pred.mu_log[:, None] + sd_log[:, None] * z[None, :]

        if self.transform.lower().strip() == "log":
            arr = np.exp(q_log)
        else:
            arr = np.exp(q_log) - 1.0

        return _format_quantiles_output(arr, qs, output=output, column_prefix=column_prefix, X=X)

    def predict_interval(self, X: ArrayLike, *, alpha: float = 0.1):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        qs = self.predict_quantiles(X, [alpha / 2.0, 1.0 - alpha / 2.0], output="array")
        return qs[:, 0], qs[:, 1]
