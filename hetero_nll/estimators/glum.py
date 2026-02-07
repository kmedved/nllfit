from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ..calibration import apply_variance_calibration, fit_variance_scale
from ..splitting import TimeInfo, calibration_split, infer_time, time_sort
from ..types import HeteroscedasticPrediction, VarianceCalibration
from .base import HeteroscedasticRegressor, ArrayLike


class TwoStageHeteroscedasticGLUM(HeteroscedasticRegressor):
    """Two-stage heteroscedastic Gaussian model using glum GLMs.

    Stage 1:
        Normal family, identity link for mean mu(x)
    Stage 2:
        Gamma family, log link for squared residuals -> variance proxy var(x)

    Parameters
    ----------
    alpha, l1_ratio:
        Elastic-net penalty parameters for glum.
    n_iterations:
        Alternating re-estimation iterations. In parametric GLM setting this
        corresponds to an IRLS-like alternating scheme.
    calibrate:
        Whether to fit a scalar variance multiplier on a holdout split (preferred)
        or on training data if no holdout is provided.
    calibration_fraction:
        If >0 and no explicit (X_cal, y_cal) provided, creates an internal
        calibration holdout using group/time-aware logic.
    time_col:
        Optional name of a datetime-like column in X to use as time ordering.
        If None, uses pandas time index if present.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        n_iterations: int = 1,
        calibrate: bool = True,
        calibration_fraction: float = 0.0,
        calibration_random_state: int = 123,
        time_col: Optional[str] = None,
        eps: float = 1e-12,
    ):
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.n_iterations = int(n_iterations)
        self.calibrate = bool(calibrate)
        self.calibration_fraction = float(calibration_fraction)
        self.calibration_random_state = int(calibration_random_state)
        self.time_col = time_col
        self.eps = float(eps)

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
    ) -> "TwoStageHeteroscedasticGLUM":
        try:
            from glum import GeneralizedLinearRegressor
        except Exception as e:  # pragma: no cover
            raise ImportError("glum is required to use TwoStageHeteroscedasticGLUM") from e

        y_all = np.asarray(y, dtype=float).reshape(-1)
        w_all = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
        g_all = None if groups is None else np.asarray(groups)

        eps_eff = max(self.eps, 1e-12 * max(float(np.var(y_all)), float(np.finfo(float).tiny)))

        # Time handling: infer + sort for time-aware splitting (does not affect model correctness)
        time = infer_time(X, time_col=self.time_col)
        Xs, ys, ws, gs, order = time_sort(X, y_all, w_all, None if g_all is None else np.asarray(g_all), time)

        # If we sorted, also reorder time values so any time-based split is aligned.
        if order is not None and time.values is not None:
            time = TimeInfo(kind=time.kind, values=np.asarray(time.values)[order], name=time.name)

        # Calibration split (optional)
        if self.calibrate and X_cal is None and y_cal is None and self.calibration_fraction > 0.0:
            X_tr, X_hold, y_tr, y_hold, w_tr, w_hold, _, _, cal_strategy = calibration_split(
                Xs,
                ys,
                sample_weight=ws,
                groups=gs,
                time=time,
                calibration_fraction=self.calibration_fraction,
                random_state=self.calibration_random_state,
            )
        else:
            X_tr, y_tr, w_tr = Xs, ys, ws
            X_hold, y_hold, w_hold = X_cal, y_cal, sample_weight_cal
            cal_strategy = "explicit" if (X_cal is not None and y_cal is not None) else "none"

        variance_tr = None

        for _ in range(self.n_iterations):
            mean_model = GeneralizedLinearRegressor(
                family="normal",
                link="identity",
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=True,
            )

            w_mean = None if w_tr is None else np.asarray(w_tr, dtype=float).copy()
            if variance_tr is not None:
                inv_var = 1.0 / np.clip(variance_tr, eps_eff, np.inf)
                w_mean = inv_var if w_mean is None else (w_mean * inv_var)

            mean_model.fit(X_tr, y_tr, sample_weight=w_mean)
            mu_tr = mean_model.predict(X_tr)

            res2 = np.maximum((y_tr - mu_tr) ** 2, eps_eff)

            var_model = GeneralizedLinearRegressor(
                family="gamma",
                link="log",
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=True,
            )
            var_model.fit(X_tr, res2, sample_weight=w_tr)
            variance_tr = np.clip(var_model.predict(X_tr), eps_eff, np.inf)

        self.mean_model_ = mean_model
        self.var_model_ = var_model
        self.eps_ = eps_eff

        # Calibration
        cal = VarianceCalibration(scale=1.0, source="none")
        if self.calibrate:
            if X_hold is not None and y_hold is not None:
                y_hold_ = np.asarray(y_hold, dtype=float).reshape(-1)
                mu_hold = self.mean_model_.predict(X_hold)
                var_hold_raw = np.clip(self.var_model_.predict(X_hold), eps_eff, np.inf)
                w_hold_ = None if w_hold is None else np.asarray(w_hold, dtype=float).reshape(-1)
                cal = fit_variance_scale(y_hold_, mu_hold, var_hold_raw, sample_weight=w_hold_, eps=eps_eff, source="holdout")
            else:
                mu_tr_final = self.mean_model_.predict(X_tr)
                var_tr_raw = np.clip(self.var_model_.predict(X_tr), eps_eff, np.inf)
                cal = fit_variance_scale(y_tr, mu_tr_final, var_tr_raw, sample_weight=w_tr, eps=eps_eff, source="train")

        self.calibration_ = cal
        self.calibration_strategy_ = cal_strategy
        return self

    def predict_dist(self, X: ArrayLike) -> HeteroscedasticPrediction:
        eps_eff = getattr(self, "eps_", self.eps)

        mu = np.asarray(self.mean_model_.predict(X), dtype=float)
        var_raw = np.asarray(self.var_model_.predict(X), dtype=float)
        var = apply_variance_calibration(var_raw, self.calibration_, eps=eps_eff)
        return HeteroscedasticPrediction(mu=mu, var=var, log_var=np.log(var))
