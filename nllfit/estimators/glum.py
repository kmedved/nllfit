from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ..calibration import apply_variance_calibration, fit_variance_scale
from ..splitting import TimeInfo, calibration_split, infer_time, time_sort
from ..types import HeteroscedasticPrediction, VarianceCalibration
from ..validation import as_1d_float, validate_groups, validate_sample_weight
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
    calibration_method:
        How to calibrate the variance scale after fitting:
          - "none" (default): No calibration. GLMs rarely overfit, so the
            in-sample variance estimate is usually adequate.
          - "holdout": Split off calibration_fraction of data and calibrate on
            held-out residuals. Use when you want explicit temporal holdout.
          - "oof": Not supported for GLM (GLMs don't use OOF residuals).
            If passed, falls back to "none" with a warning.
          - "train": Calibrate on in-sample residuals. Safe for GLMs (unlike
            flexible models) since they don't overfit.
    calibration_fraction:
        Fraction of data to hold out when calibration_method="holdout".
        Ignored for other calibration methods.
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
        calibration_method: str = "none",  # "none" | "holdout" | "train"
        calibration_fraction: float = 0.2,
        calibration_random_state: int = 123,
        time_col: Optional[str] = None,
        eps: float = 1e-12,
        # Deprecated parameters — will be removed in 0.3.0
        calibrate: Optional[bool] = None,
    ):
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.n_iterations = int(n_iterations)
        self.calibration_fraction = float(calibration_fraction)
        self.calibration_random_state = int(calibration_random_state)
        self.time_col = time_col
        self.eps = float(eps)
        self.calibrate = calibrate

        # Handle deprecated `calibrate` parameter
        if calibrate is not None:
            warnings.warn(
                "The `calibrate` parameter is deprecated and will be removed in v0.3.0. "
                "Use `calibration_method` instead: "
                "calibrate=False -> calibration_method='none', "
                "calibrate=True -> calibration_method='holdout' or 'train'.",
                FutureWarning,
                stacklevel=2,
            )
            if not calibrate:
                calibration_method = "none"
            elif calibration_method == "none":
                # User passed calibrate=True without specifying calibration_method;
                # map to legacy behavior: holdout if fraction > 0, else train
                if self.calibration_fraction > 0.0:
                    calibration_method = "holdout"
                else:
                    calibration_method = "train"

        if calibration_method == "oof":
            warnings.warn(
                "calibration_method='oof' is not supported for GLM (no OOF residuals). "
                "Falling back to calibration_method='none'.",
                UserWarning,
                stacklevel=2,
            )
            calibration_method = "none"

        self.calibration_method = str(calibration_method)

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

        y_all = as_1d_float("y", y)
        w_all = validate_sample_weight(y_all, sample_weight)
        g_all = validate_groups(y_all, groups)

        eps_eff = max(self.eps, 1e-12 * max(float(np.var(y_all)), float(np.finfo(float).tiny)))

        cal_method = self.calibration_method.lower().strip()
        if cal_method not in {"none", "holdout", "train"}:
            raise ValueError(
                f"calibration_method must be one of: 'none', 'holdout', 'train' for GLM. Got {cal_method!r}."
            )

        # Time handling: infer + sort for time-aware splitting (does not affect model correctness)
        time = infer_time(X, time_col=self.time_col)
        Xs, ys, ws, gs, order = time_sort(X, y_all, w_all, None if g_all is None else np.asarray(g_all), time)

        if order is not None and time.values is not None:
            time = TimeInfo(kind=time.kind, values=np.asarray(time.values)[order], name=time.name)

        # Calibration holdout split
        X_hold, y_hold, w_hold = None, None, None

        if X_cal is not None and y_cal is not None:
            # Explicit calibration data always takes priority
            X_tr, y_tr, w_tr = Xs, ys, ws
            X_hold = X_cal
            y_hold = as_1d_float("y_cal", y_cal)
            w_hold = validate_sample_weight(y_hold, sample_weight_cal)
            cal_method = "holdout"
            cal_strategy = "explicit"
        elif cal_method == "holdout" and self.calibration_fraction > 0.0:
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
            cal_strategy = cal_method  # "train" or "none"

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
        self.calibration_strategy_ = cal_strategy

        # Calibration
        cal = VarianceCalibration(scale=1.0, source="none")

        if cal_method == "holdout" and X_hold is not None and y_hold is not None:
            y_hold_ = np.asarray(y_hold, dtype=float).reshape(-1)
            mu_hold = self.mean_model_.predict(X_hold)
            var_hold_raw = np.clip(self.var_model_.predict(X_hold), eps_eff, np.inf)
            w_hold_ = None if w_hold is None else np.asarray(w_hold, dtype=float).reshape(-1)
            cal = fit_variance_scale(
                y_hold_, mu_hold, var_hold_raw,
                sample_weight=w_hold_, eps=eps_eff, source="holdout",
            )

        elif cal_method == "train":
            mu_tr_final = self.mean_model_.predict(X_tr)
            var_tr_raw = np.clip(self.var_model_.predict(X_tr), eps_eff, np.inf)
            cal = fit_variance_scale(
                y_tr, mu_tr_final, var_tr_raw,
                sample_weight=w_tr, eps=eps_eff, source="train",
            )

        self.calibration_ = cal
        return self

    def predict_dist(self, X: ArrayLike) -> HeteroscedasticPrediction:
        eps_eff = getattr(self, "eps_", self.eps)

        mu = np.asarray(self.mean_model_.predict(X), dtype=float)
        var_raw = np.asarray(self.var_model_.predict(X), dtype=float)
        var = apply_variance_calibration(var_raw, self.calibration_, eps=eps_eff)
        return HeteroscedasticPrediction(mu=mu, var=var, log_var=np.log(var))
