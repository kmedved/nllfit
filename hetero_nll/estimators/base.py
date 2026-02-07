from __future__ import annotations

from statistics import NormalDist
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ..metrics import gaussian_nll
from ..types import HeteroscedasticPrediction

ArrayLike = Union[np.ndarray, "pd.DataFrame", "pd.Series"]
QuantileOutput = Literal["array", "dataframe", "dict"]


try:
    from sklearn.base import BaseEstimator, RegressorMixin
except Exception:  # pragma: no cover
    BaseEstimator = object  # type: ignore
    RegressorMixin = object  # type: ignore


class HeteroscedasticRegressor(BaseEstimator, RegressorMixin):
    """Base class for heteroscedastic regressors.

    Conventions:
      - predict(X) returns mean predictions (sklearn-compatible)
      - predict_dist(X) returns mu/var/log_var
      - score(X, y) returns negative NLL (higher is better)
    """

    eps_: float = 1e-12

    def predict_dist(self, X: ArrayLike) -> HeteroscedasticPrediction:  # pragma: no cover
        raise NotImplementedError

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
        return gaussian_nll(
            y,
            pred.mu,
            pred.var,
            sample_weight=sample_weight,
            eps=getattr(self, "eps_", 1e-12),
            include_const=include_const,
        )

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[np.ndarray] = None) -> float:
        """Return negative NLL (higher is better)."""
        return -self.nll(X, y, sample_weight=sample_weight, include_const=False)

    def predict_quantiles(
        self,
        X: ArrayLike,
        quantiles: Sequence[float],
        *,
        output: QuantileOutput = "array",
        column_prefix: str = "q",
    ) -> Union[np.ndarray, "pd.DataFrame", Dict[float, np.ndarray]]:
        """Gaussian predictive quantiles assuming y|x ~ Normal(mu(x), var(x)).

        Parameters
        ----------
        quantiles:
            Iterable of quantiles in (0, 1), e.g. [0.1, 0.5, 0.9].
        output:
            "array" -> ndarray (n, k)
            "dict" -> {q: ndarray(n,)}
            "dataframe" -> DataFrame (n, k) if pandas available
        column_prefix:
            Used for dataframe column naming like "q0.1".

        Returns
        -------
        ndarray / dict / DataFrame
        """
        qs = np.asarray(list(quantiles), dtype=float).reshape(-1)
        if qs.size == 0:
            raise ValueError("quantiles must be non-empty.")
        if np.any(qs <= 0.0) or np.any(qs >= 1.0):
            raise ValueError("All quantiles must be strictly between 0 and 1.")

        pred = self.predict_dist(X)
        sd = np.sqrt(pred.var)

        z = np.array([NormalDist().inv_cdf(float(q)) for q in qs], dtype=float)  # (k,)
        arr = pred.mu[:, None] + sd[:, None] * z[None, :]  # (n, k)

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

    def predict_interval(self, X: ArrayLike, *, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Central (1-alpha) Gaussian predictive interval."""
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        qs = self.predict_quantiles(X, [alpha / 2.0, 1.0 - alpha / 2.0], output="array")
        return qs[:, 0], qs[:, 1]
