from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ..calibration import apply_variance_calibration, fit_variance_scale
from ..oof import choose_oof_splitter, oof_mean_predictions, oof_squared_residuals
from ..splitting import TimeInfo, calibration_split, infer_time, time_sort
from ..types import HeteroscedasticPrediction, VarianceCalibration
from ..validation import as_1d_float, validate_groups, validate_sample_weight
from .base import HeteroscedasticRegressor, ArrayLike


@dataclass(frozen=True)
class VarianceModelInfo:
    mode: str  # "gamma" | "log"
    objective: str
    details: str = ""


class TwoStageHeteroscedasticLightGBM(HeteroscedasticRegressor):
    """Two-stage heteroscedastic Gaussian model using LightGBM.

    Stage 1 (mean):
        LGBM regression

    Stage 2 (variance proxy):
        Fits either:
          - mode="gamma": objective="gamma" on squared residuals
          - mode="log":   objective="regression" on log(squared residuals)

        mode="auto" tries gamma first and falls back to log.

    Key features
    ------------
    - OOF residuals for Stage 2 (prevents variance leakage from overfitting).
    - Optional OOF caching across iterations for speed.
    - Group- and/or time-aware calibration holdout split.
    - Gaussian intervals + quantiles (via base class).

    Parameters
    ----------
    mean_params, var_params:
        LightGBM parameter dicts passed to LGBMRegressor for Stage 1 and Stage 2.
        If var_params is provided, it is used as a base dict and objective is set
        according to variance_mode.
    variance_mode:
        "auto" | "gamma" | "log"
    n_iterations:
        Number of mean/variance alternating iterations. 2 is recommended: the
        second iteration reweights the mean model by inverse predicted variance.
    n_oof_folds:
        If >1, use OOF residuals. If <=1, uses in-sample residuals (not recommended).
    oof_residuals_reuse:
        If True and n_iterations>1, compute OOF residuals once (based on the first
        iteration's mean weights) and reuse for later iterations. Faster but approximate.
    calibration_method:
        How to calibrate the variance scale after fitting:
          - "oof" (default): Calibrate using OOF mean predictions on the training
            set. No data is held out, no in-sample bias. Recommended.
          - "holdout": Split off calibration_fraction of data and calibrate on
            held-out residuals. Use when you want explicit temporal holdout
            (e.g., time-series with calibration_fraction > 0).
          - "train": Calibrate on in-sample residuals. NOT recommended for
            flexible models — causes systematic variance shrinkage.
          - "none": No calibration.
    calibration_fraction:
        Fraction of data to hold out when calibration_method="holdout".
        Ignored for other calibration methods.
    time_col:
        Optional name of a datetime-like column in X to use for time ordering/splitting.
        If None, uses pandas time index if present.
    """

    def __init__(
        self,
        *,
        mean_params: Optional[Dict[str, Any]] = None,
        var_params: Optional[Dict[str, Any]] = None,
        n_iterations: int = 2,
        n_oof_folds: int = 5,
        oof_splitter: Optional[Any] = None,
        oof_random_state: int = 42,
        variance_mode: str = "auto",  # "auto" | "gamma" | "log"
        oof_residuals_reuse: bool = True,
        calibration_method: str = "oof",  # "none" | "holdout" | "oof" | "train"
        calibration_fraction: float = 0.2,
        calibration_random_state: int = 123,
        time_col: Optional[str] = None,
        eps: float = 1e-12,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        # Deprecated parameters — will be removed in 0.3.0
        calibrate: Optional[bool] = None,
    ):
        self.mean_params = mean_params
        self.var_params = var_params
        self.n_iterations = int(n_iterations)
        self.n_oof_folds = int(n_oof_folds)
        self.oof_splitter = oof_splitter
        self.oof_random_state = int(oof_random_state)
        self.variance_mode = str(variance_mode)
        self.oof_residuals_reuse = bool(oof_residuals_reuse)
        self.calibration_fraction = float(calibration_fraction)
        self.calibration_random_state = int(calibration_random_state)
        self.time_col = time_col
        self.eps = float(eps)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.calibrate = calibrate

        # Handle deprecated `calibrate` parameter
        if calibrate is not None:
            warnings.warn(
                "The `calibrate` parameter is deprecated and will be removed in v0.3.0. "
                "Use `calibration_method` instead: "
                "calibrate=False -> calibration_method='none', "
                "calibrate=True -> calibration_method='oof' (recommended) or 'holdout'.",
                FutureWarning,
                stacklevel=2,
            )
            if not calibrate:
                calibration_method = "none"
            elif calibration_method == "oof":
                # User passed calibrate=True without specifying calibration_method;
                # map to legacy behavior: holdout if fraction > 0, else train
                if self.calibration_fraction > 0.0:
                    calibration_method = "holdout"
                else:
                    calibration_method = "train"

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
    ) -> "TwoStageHeteroscedasticLightGBM":
        try:
            import lightgbm as lgb
        except Exception as e:  # pragma: no cover
            raise ImportError("lightgbm is required to use TwoStageHeteroscedasticLightGBM") from e

        y_all = as_1d_float("y", y)
        w_all = validate_sample_weight(y_all, sample_weight)
        g_all = validate_groups(y_all, groups)

        eps_eff = max(self.eps, 1e-12 * max(float(np.var(y_all)), float(np.finfo(float).tiny)))

        cal_method = self.calibration_method.lower().strip()
        if cal_method not in {"none", "holdout", "oof", "train"}:
            raise ValueError(
                f"calibration_method must be one of: 'none', 'holdout', 'oof', 'train'. Got {cal_method!r}."
            )

        if cal_method == "train":
            warnings.warn(
                "calibration_method='train' calibrates on in-sample residuals, which causes "
                "systematic variance shrinkage with flexible models. Use 'oof' or 'holdout' instead.",
                UserWarning,
                stacklevel=2,
            )

        if cal_method == "oof" and (not self.n_oof_folds or self.n_oof_folds <= 1):
            raise ValueError("calibration_method='oof' requires n_oof_folds > 1.")

        need_oof = bool(self.n_oof_folds and self.n_oof_folds > 1) or (cal_method == "oof")

        # Time handling: infer + sort for time-aware splitting
        time = infer_time(X, time_col=self.time_col)
        Xs, ys, ws, gs, order = time_sort(X, y_all, w_all, None if g_all is None else np.asarray(g_all), time)

        if order is not None and time.values is not None:
            time = TimeInfo(kind=time.kind, values=np.asarray(time.values)[order], name=time.name)

        # Default params
        mean_params = dict(self.mean_params) if self.mean_params is not None else {
            "objective": "regression",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "reg_lambda": 0.0,
            "verbose": -1,
        }
        if self.random_state is not None and "random_state" not in mean_params:
            mean_params["random_state"] = self.random_state
        if self.n_jobs is not None and "n_jobs" not in mean_params:
            mean_params["n_jobs"] = self.n_jobs
        base_var_params = dict(self.var_params) if self.var_params is not None else {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 50,
            "reg_lambda": 1.0,
            "verbose": -1,
        }
        if self.random_state is not None and "random_state" not in base_var_params:
            base_var_params["random_state"] = self.random_state
        if self.n_jobs is not None and "n_jobs" not in base_var_params:
            base_var_params["n_jobs"] = self.n_jobs

        # Calibration holdout split (only for "holdout" method)
        X_hold, y_hold, w_hold = None, None, None

        if X_cal is not None and y_cal is not None:
            # Explicit calibration data always takes priority
            X_tr, y_tr, w_tr, g_tr = Xs, ys, ws, gs
            X_hold = X_cal
            y_hold = as_1d_float("y_cal", y_cal)
            w_hold = validate_sample_weight(y_hold, sample_weight_cal)
            cal_method = "holdout"  # explicit cal data overrides
            cal_strategy = "explicit"
        elif cal_method == "holdout" and self.calibration_fraction > 0.0:
            X_tr, X_hold, y_tr, y_hold, w_tr, w_hold, g_tr, g_hold, cal_strategy = calibration_split(
                Xs,
                ys,
                sample_weight=ws,
                groups=gs,
                time=time,
                calibration_fraction=self.calibration_fraction,
                random_state=self.calibration_random_state,
            )
        else:
            X_tr, y_tr, w_tr, g_tr = Xs, ys, ws, gs
            cal_strategy = cal_method  # "oof", "train", or "none"

        g_tr = None if g_tr is None else np.asarray(g_tr)

        # Splitter for OOF residuals
        splitter_obj = None
        if need_oof:
            if self.oof_splitter is not None:
                splitter_obj = self.oof_splitter
            elif g_tr is not None:
                splitter_obj = choose_oof_splitter(
                    X_tr,
                    n_splits=max(2, self.n_oof_folds),
                    random_state=self.oof_random_state,
                    splitter=None,
                    groups=g_tr,
                )
            elif time.kind != "none" and time.values is not None:
                from sklearn.model_selection import TimeSeriesSplit
                splitter_obj = TimeSeriesSplit(n_splits=max(2, self.n_oof_folds))
            else:
                splitter_obj = choose_oof_splitter(
                    X_tr,
                    n_splits=max(2, self.n_oof_folds),
                    random_state=self.oof_random_state,
                    splitter=None,
                    groups=None,
                )

        requested_mode = self.variance_mode.lower().strip()
        if requested_mode not in {"auto", "gamma", "log"}:
            raise ValueError("variance_mode must be one of: 'auto', 'gamma', 'log'.")

        chosen_mode: Optional[str] = None
        variance_tr: Optional[np.ndarray] = None
        cached_res2: Optional[np.ndarray] = None

        for it in range(self.n_iterations):
            # ---- Stage 1 mean ----
            mean_model = lgb.LGBMRegressor(**mean_params)

            w_mean = None if w_tr is None else np.asarray(w_tr, dtype=float).copy()
            if variance_tr is not None:
                inv_var = 1.0 / np.clip(variance_tr, eps_eff, np.inf)
                w_mean = inv_var if w_mean is None else (w_mean * inv_var)

            mean_model.fit(X_tr, y_tr, sample_weight=w_mean)

            # ---- Stage 2 target ----
            use_oof = bool(self.n_oof_folds and self.n_oof_folds > 1)

            if use_oof:
                if self.oof_residuals_reuse and cached_res2 is not None:
                    res2 = cached_res2
                else:
                    def mean_factory() -> Any:
                        return lgb.LGBMRegressor(**mean_params)

                    res2 = oof_squared_residuals(
                        X_tr,
                        y_tr,
                        model_factory=mean_factory,
                        n_splits=self.n_oof_folds,
                        sample_weight=w_mean,
                        splitter=splitter_obj,
                        groups=g_tr,
                        random_state=self.oof_random_state,
                        eps=eps_eff,
                    )
                    if self.oof_residuals_reuse:
                        cached_res2 = res2
            else:
                mu_in = mean_model.predict(X_tr)
                res2 = np.maximum((y_tr - mu_in) ** 2, eps_eff)

            # ---- Stage 2 variance model ----
            if chosen_mode is None:
                chosen_mode = "gamma" if requested_mode in {"auto", "gamma"} else "log"

            var_model: Any

            if chosen_mode == "gamma":
                gamma_params = dict(base_var_params)
                gamma_params["objective"] = "gamma"
                try:
                    var_model = lgb.LGBMRegressor(**gamma_params)
                    var_model.fit(X_tr, res2, sample_weight=w_tr)
                    variance_tr = np.clip(var_model.predict(X_tr), eps_eff, np.inf)
                    self.variance_model_info_ = VarianceModelInfo(mode="gamma", objective="gamma", details="ok")
                except Exception as e:
                    if requested_mode == "gamma":
                        raise
                    chosen_mode = "log"
                    self.variance_model_info_ = VarianceModelInfo(
                        mode="log", objective="regression",
                        details=f"gamma_failed: {type(e).__name__}",
                    )

            if chosen_mode == "log":
                log_params = dict(base_var_params)
                log_params["objective"] = "regression"
                log_target = np.log(np.maximum(res2, eps_eff))

                var_model = lgb.LGBMRegressor(**log_params)
                var_model.fit(X_tr, log_target, sample_weight=w_tr)
                log_var_pred_tr = var_model.predict(X_tr)
                variance_tr = np.clip(np.exp(log_var_pred_tr), eps_eff, np.inf)
                self.variance_model_info_ = VarianceModelInfo(mode="log", objective="regression", details="ok")

        self.mean_model_ = mean_model
        self.var_model_ = var_model
        self.variance_mode_ = chosen_mode or "log"
        self.eps_ = eps_eff
        self.calibration_strategy_ = cal_strategy

        # ---- Helper to get var predictions ----
        def _predict_var(X_in: ArrayLike) -> np.ndarray:
            if self.variance_mode_ == "gamma":
                return np.clip(self.var_model_.predict(X_in), eps_eff, np.inf)
            else:
                return np.clip(np.exp(self.var_model_.predict(X_in)), eps_eff, np.inf)

        # ---- Calibration ----
        cal = VarianceCalibration(scale=1.0, source="none")

        if cal_method == "holdout" and X_hold is not None and y_hold is not None:
            y_hold_ = np.asarray(y_hold, dtype=float).reshape(-1)
            mu_hold = self.mean_model_.predict(X_hold)
            var_hold_raw = _predict_var(X_hold)
            w_hold_ = None if w_hold is None else np.asarray(w_hold, dtype=float).reshape(-1)
            cal = fit_variance_scale(
                y_hold_, mu_hold, var_hold_raw,
                sample_weight=w_hold_, eps=eps_eff, source="holdout",
            )

        elif cal_method == "oof":
            # OOF mean predictions — avoids in-sample bias, no data loss
            def mean_factory_cal() -> Any:
                return lgb.LGBMRegressor(**mean_params)

            mu_oof = oof_mean_predictions(
                X_tr,
                y_tr,
                model_factory=mean_factory_cal,
                n_splits=self.n_oof_folds,
                sample_weight=w_mean,
                splitter=splitter_obj,
                groups=g_tr,
                random_state=self.oof_random_state,
            )
            var_tr_cal = _predict_var(X_tr)
            cal = fit_variance_scale(
                y_tr, mu_oof, var_tr_cal,
                sample_weight=w_tr, eps=eps_eff, source="oof",
            )

        elif cal_method == "train":
            mu_tr_final = self.mean_model_.predict(X_tr)
            var_tr_raw = _predict_var(X_tr)
            cal = fit_variance_scale(
                y_tr, mu_tr_final, var_tr_raw,
                sample_weight=w_tr, eps=eps_eff, source="train",
            )

        self.calibration_ = cal
        return self

    def predict_dist(self, X: ArrayLike) -> HeteroscedasticPrediction:
        eps_eff = getattr(self, "eps_", self.eps)

        mu = np.asarray(self.mean_model_.predict(X), dtype=float)

        if getattr(self, "variance_mode_", "gamma") == "gamma":
            var_raw = np.asarray(self.var_model_.predict(X), dtype=float)
        else:
            var_raw = np.exp(np.asarray(self.var_model_.predict(X), dtype=float))

        var = apply_variance_calibration(var_raw, self.calibration_, eps=eps_eff)
        return HeteroscedasticPrediction(mu=mu, var=var, log_var=np.log(var))
