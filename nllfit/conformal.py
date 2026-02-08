"""Split conformal prediction intervals.

Three methods:
  - absolute: distribution-free, constant-width intervals
  - normalized: locally adaptive using predicted std
  - cqr: conformalized quantile regression using base model intervals
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Union

import numpy as np

from .splitting import TimeInfo, calibration_split, infer_time, time_sort
from .validation import as_1d_float, validate_groups, validate_sample_weight

ArrayLike = Union[np.ndarray, "np.typing.ArrayLike"]


def _conformal_quantile(
    scores: np.ndarray,
    alpha: float,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Compute conformal quantile with finite-sample correction.

    Unweighted: k = ceil((n+1)*(1-alpha)); return k-th order statistic.
    Weighted: weighted quantile at level (1 - alpha) adjusted for finite sample,
    treating weights as frequency weights.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot compute conformal quantile on empty scores.")

    if sample_weight is not None:
        n_eff = float(np.sum(sample_weight))
        w = sample_weight / n_eff
        level = (1.0 - alpha) * (1.0 + 1.0 / n_eff)
        level = min(level, 1.0)
        order = np.argsort(scores)
        sorted_scores = scores[order]
        cum_w = np.cumsum(w[order])
        # side="left" picks the first index where cum_w >= level, which
        # matches the unweighted order-statistic rule (ceil((n+1)*(1-alpha)))
        # and preserves replication consistency: integer weights produce the
        # same quantile as expanding the dataset by those counts.
        idx = int(np.searchsorted(cum_w, level, side="left"))
        idx = min(idx, n - 1)
        return float(sorted_scores[idx])
    else:
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        sorted_scores = np.sort(scores)
        return float(sorted_scores[k - 1])


class SplitConformalRegressor:
    """Split conformal prediction intervals wrapping any base estimator.

    Parameters
    ----------
    base_estimator :
        A fitted or unfitted estimator. Must implement `.fit(X, y, **kw)`
        and `.predict(X)`. For method="normalized", must also implement
        `.predict_dist(X)` returning an object with `.var` attribute.
        For method="cqr", must implement `.predict_interval(X, alpha=...)`.
    method : {"absolute", "normalized", "cqr"}
        - absolute: |y - mu|, constant-width intervals
        - normalized: |y - mu| / sigma, locally adaptive
        - cqr: conformalized quantile regression
    alpha : float
        Miscoverage rate. Intervals target 1-alpha coverage.
    calibration_fraction : float
        Fraction of training data to hold out for calibration when no
        explicit calibration set is provided.
    time_col : str or None
        Column name for time-aware calibration splitting.
    calibration_random_state : int
        Random seed for calibration split.
    """

    def __init__(
        self,
        base_estimator: Any,
        *,
        method: Literal["absolute", "normalized", "cqr"] = "absolute",
        alpha: float = 0.1,
        calibration_fraction: float = 0.2,
        time_col: Optional[str] = None,
        calibration_random_state: int = 42,
    ):
        if method not in {"absolute", "normalized", "cqr"}:
            raise ValueError(f"method must be 'absolute', 'normalized', or 'cqr'. Got {method!r}.")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if not (0.0 < calibration_fraction < 1.0):
            raise ValueError("calibration_fraction must be in (0, 1).")

        self.base_estimator = base_estimator
        self.method = method
        self.alpha = alpha
        self.calibration_fraction = calibration_fraction
        self.time_col = time_col
        self.calibration_random_state = calibration_random_state

        self.q_: Optional[float] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: ArrayLike,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        X_cal: Optional[ArrayLike] = None,
        y_cal: Optional[np.ndarray] = None,
        sample_weight_cal: Optional[np.ndarray] = None,
    ) -> "SplitConformalRegressor":
        """Fit base estimator and compute conformal quantile.

        If X_cal/y_cal are provided, they are used as the calibration set
        directly (base estimator is trained on all of X/y). Otherwise,
        data is split internally using calibration_fraction.
        """
        y_all = as_1d_float("y", y)
        w_all = validate_sample_weight(y_all, sample_weight)
        g_all = validate_groups(y_all, groups)

        if X_cal is not None and y_cal is not None:
            X_tr, y_tr, w_tr = X, y_all, w_all
            X_c = X_cal
            y_c = as_1d_float("y_cal", y_cal)
            w_c = validate_sample_weight(y_c, sample_weight_cal)
        else:
            time = infer_time(X, time_col=self.time_col)
            Xs, ys, ws, gs, order = time_sort(X, y_all, w_all, g_all, time)
            if order is not None and time.values is not None:
                time = TimeInfo(kind=time.kind, values=np.asarray(time.values)[order], name=time.name)

            X_tr, X_c, y_tr, y_c, w_tr, w_c, _, _, _ = calibration_split(
                Xs,
                ys,
                sample_weight=ws,
                groups=gs,
                time=time,
                calibration_fraction=self.calibration_fraction,
                random_state=self.calibration_random_state,
            )

        fit_kw = {}
        if w_tr is not None:
            fit_kw["sample_weight"] = w_tr
        self.base_estimator.fit(X_tr, y_tr, **fit_kw)

        scores = self._compute_scores(X_c, y_c)
        self.q_ = _conformal_quantile(scores, self.alpha, sample_weight=w_c)
        self.is_fitted_ = True
        return self

    def _compute_scores(self, X: ArrayLike, y: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores on a dataset."""
        mu = np.asarray(self.base_estimator.predict(X), dtype=float).ravel()

        if self.method == "absolute":
            return np.abs(y - mu)

        elif self.method == "normalized":
            if not hasattr(self.base_estimator, "predict_dist"):
                raise TypeError(
                    "method='normalized' requires base_estimator to implement predict_dist()."
                )
            dist = self.base_estimator.predict_dist(X)
            var = np.asarray(dist.var, dtype=float).ravel()
            sigma = np.sqrt(np.clip(var, 1e-12, np.inf))
            return np.abs(y - mu) / sigma

        elif self.method == "cqr":
            if not hasattr(self.base_estimator, "predict_interval"):
                raise TypeError(
                    "method='cqr' requires base_estimator to implement predict_interval()."
                )
            lo, hi = self.base_estimator.predict_interval(X, alpha=self.alpha)
            lo = np.asarray(lo, dtype=float).ravel()
            hi = np.asarray(hi, dtype=float).ravel()
            return np.maximum.reduce([lo - y, y - hi, np.zeros_like(y)])

        else:
            raise ValueError(f"Unknown method {self.method!r}")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict mean (delegates to base estimator)."""
        self._check_fitted()
        return np.asarray(self.base_estimator.predict(X), dtype=float).ravel()

    def predict_interval(self, X: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Predict conformal intervals.

        Returns
        -------
        (lo, hi) : tuple of 1D arrays
        """
        self._check_fitted()
        mu = self.predict(X)

        if self.method == "absolute":
            return mu - self.q_, mu + self.q_

        elif self.method == "normalized":
            dist = self.base_estimator.predict_dist(X)
            var = np.asarray(dist.var, dtype=float).ravel()
            sigma = np.sqrt(np.clip(var, 1e-12, np.inf))
            return mu - self.q_ * sigma, mu + self.q_ * sigma

        elif self.method == "cqr":
            lo_base, hi_base = self.base_estimator.predict_interval(X, alpha=self.alpha)
            lo_base = np.asarray(lo_base, dtype=float).ravel()
            hi_base = np.asarray(hi_base, dtype=float).ravel()
            return lo_base - self.q_, hi_base + self.q_

        else:
            raise ValueError(f"Unknown method {self.method!r}")

    def predict_dist(self, X: ArrayLike):
        """Predict distributional parameters (delegates to base estimator).

        Raises TypeError if the base estimator does not implement predict_dist.
        """
        self._check_fitted()
        if not hasattr(self.base_estimator, "predict_dist"):
            raise TypeError(
                "Base estimator does not implement predict_dist()."
            )
        return self.base_estimator.predict_dist(X)

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("SplitConformalRegressor has not been fitted. Call .fit() first.")
