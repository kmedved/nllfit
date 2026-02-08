from __future__ import annotations

import copy
import importlib.util
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

from .metrics import laplace_nll, student_t_nll

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


def laplace_ppf(
    p: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("All quantiles must be strictly between 0 and 1.")
    var = np.clip(np.asarray(var, dtype=float), eps, np.inf)
    b = np.sqrt(var / 2.0)
    z = np.where(p < 0.5, np.log(2.0 * p), -np.log(2.0 * (1.0 - p)))
    return mu[..., None] + b[..., None] * z[None, :]


class LaplaceWrapper(BaseEstimator, RegressorMixin):
    """Wrap a mu/var estimator to use Laplace NLL and quantiles."""

    def __init__(self, base_estimator, *, eps: float = 1e-12):
        self.base_estimator = base_estimator
        self.eps = eps

    def fit(self, X: ArrayLike, y: np.ndarray, *, sample_weight=None, groups=None, **kwargs):
        if clone is not None and hasattr(self.base_estimator, "get_params"):
            self.base_estimator_ = clone(self.base_estimator)
        else:
            self.base_estimator_ = copy.deepcopy(self.base_estimator)
        fit_kw = dict(kwargs)
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight
        if groups is not None:
            fit_kw["groups"] = groups
        self.base_estimator_.fit(X, y, **fit_kw)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        return np.asarray(self.base_estimator_.predict(X), dtype=float)

    def predict_dist(self, X: ArrayLike):
        return self.base_estimator_.predict_dist(X)

    def nll(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[np.ndarray] = None,
        include_const: bool = False,
    ) -> float:
        pred = self.predict_dist(X)
        eps_eff = getattr(self.base_estimator_, "eps_", self.eps)
        return laplace_nll(
            y,
            pred.mu,
            pred.var,
            sample_weight=sample_weight,
            eps=eps_eff,
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
        pred = self.predict_dist(X)
        arr = laplace_ppf(qs, np.asarray(pred.mu, dtype=float), np.asarray(pred.var, dtype=float), eps=self.eps)
        return _format_quantiles_output(arr, qs, output=output, column_prefix=column_prefix, X=X)

    def predict_interval(self, X: ArrayLike, *, alpha: float = 0.1):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        qs = self.predict_quantiles(X, [alpha / 2.0, 1.0 - alpha / 2.0], output="array")
        return qs[:, 0], qs[:, 1]


class StudentTWrapper(BaseEstimator, RegressorMixin):
    """Wrap a mu/var estimator to use Student-t NLL and quantiles."""

    def __init__(self, base_estimator, df: float, *, eps: float = 1e-12):
        self.base_estimator = base_estimator
        self.df = df
        self.eps = eps

    def fit(self, X: ArrayLike, y: np.ndarray, *, sample_weight=None, groups=None, **kwargs):
        if clone is not None and hasattr(self.base_estimator, "get_params"):
            self.base_estimator_ = clone(self.base_estimator)
        else:
            self.base_estimator_ = copy.deepcopy(self.base_estimator)
        fit_kw = dict(kwargs)
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight
        if groups is not None:
            fit_kw["groups"] = groups
        self.base_estimator_.fit(X, y, **fit_kw)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        return np.asarray(self.base_estimator_.predict(X), dtype=float)

    def predict_dist(self, X: ArrayLike):
        return self.base_estimator_.predict_dist(X)

    def nll(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[np.ndarray] = None,
        include_const: bool = False,
    ) -> float:
        pred = self.predict_dist(X)
        eps_eff = getattr(self.base_estimator_, "eps_", self.eps)
        return student_t_nll(
            y,
            pred.mu,
            pred.var,
            self.df,
            sample_weight=sample_weight,
            eps=eps_eff,
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
        if self.df <= 2:
            raise ValueError("StudentTWrapper requires df > 2 when var is a variance parameter.")
        qs = np.asarray(list(quantiles), dtype=float).reshape(-1)
        if qs.size == 0:
            raise ValueError("quantiles must be non-empty.")
        if np.any(qs <= 0.0) or np.any(qs >= 1.0):
            raise ValueError("All quantiles must be strictly between 0 and 1.")

        if importlib.util.find_spec("scipy") is None:
            raise ImportError("scipy is required for Student-t quantiles.")
        from scipy.stats import t as student_t

        pred = self.predict_dist(X)
        var = np.clip(np.asarray(pred.var, dtype=float), self.eps, np.inf)
        scale2 = var * (self.df - 2.0) / self.df
        scale2 = np.clip(scale2, self.eps, np.inf)
        scale = np.sqrt(scale2)
        arr = student_t.ppf(qs[None, :], self.df, loc=np.asarray(pred.mu, dtype=float)[:, None], scale=scale[:, None])
        return _format_quantiles_output(arr, qs, output=output, column_prefix=column_prefix, X=X)

    def predict_interval(self, X: ArrayLike, *, alpha: float = 0.1):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        qs = self.predict_quantiles(X, [alpha / 2.0, 1.0 - alpha / 2.0], output="array")
        return qs[:, 0], qs[:, 1]
