from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from .validation import as_1d_float, validate_groups, validate_sample_weight

ArrayLike = Union[np.ndarray, "pd.DataFrame", "pd.Series"]


def _slice_rows(X: ArrayLike, idx: np.ndarray) -> Any:
    if pd is not None and hasattr(X, "iloc"):
        return X.iloc[idx]  # type: ignore[attr-defined]
    return np.asarray(X)[idx]


def choose_oof_splitter(
    X: ArrayLike,
    *,
    n_splits: int,
    random_state: int,
    splitter: Optional[Any] = None,
    groups: Optional[np.ndarray] = None,
) -> Any:
    """Choose a default splitter if none is provided.

    Priority:
      1) splitter arg
      2) GroupKFold if groups is not None
      3) TimeSeriesSplit if X has a time-like index
      4) KFold(shuffle=True)
    """
    if splitter is not None:
        return splitter

    if groups is not None:
        n_groups = len(np.unique(groups))
        if n_splits > n_groups:
            warnings.warn(
                f"n_oof_folds={n_splits} exceeds n_groups={n_groups}; "
                f"clamping to {n_groups}-fold GroupKFold.",
                UserWarning,
                stacklevel=2,
            )
            n_splits = n_groups
        if n_splits < 2:
            raise ValueError("Need at least 2 groups for GroupKFold.")
        from sklearn.model_selection import GroupKFold
        return GroupKFold(n_splits=n_splits)

    n_samples = len(X)
    if n_splits > n_samples:
        warnings.warn(
            f"n_oof_folds={n_splits} exceeds n_samples={n_samples}; "
            f"clamping to {n_samples}-fold KFold.",
            UserWarning,
            stacklevel=2,
        )
        n_splits = n_samples
    if n_splits < 2:
        raise ValueError("Need at least 2 samples for KFold.")

    if pd is not None and hasattr(X, "index"):
        idx = X.index  # type: ignore[attr-defined]
        if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
            from sklearn.model_selection import TimeSeriesSplit
            if n_splits >= n_samples:
                raise ValueError("Need at least n_splits < n_samples for TimeSeriesSplit.")
            return TimeSeriesSplit(n_splits=n_splits)

    from sklearn.model_selection import KFold
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def oof_squared_residuals(
    X: ArrayLike,
    y: np.ndarray,
    *,
    model_factory: Callable[[], Any],
    n_splits: int = 5,
    sample_weight: Optional[np.ndarray] = None,
    splitter: Optional[Any] = None,
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute out-of-fold squared residuals (y - mu_oof)^2."""
    y_ = as_1d_float("y", y)
    w_all = validate_sample_weight(y_, sample_weight)
    g_all = validate_groups(y_, groups)

    splitter_obj = choose_oof_splitter(
        X,
        n_splits=n_splits,
        random_state=random_state,
        splitter=splitter,
        groups=g_all,
    )

    mu_oof = np.empty(len(y_), dtype=float)

    try:
        split_iter = splitter_obj.split(X, y_, groups=g_all)
    except TypeError:
        split_iter = splitter_obj.split(X, y_)

    for tr_idx, va_idx in split_iter:
        model = model_factory()

        X_tr = _slice_rows(X, tr_idx)
        y_tr = y_[tr_idx]
        X_va = _slice_rows(X, va_idx)

        fit_kwargs: Dict[str, Any] = {}
        if w_all is not None:
            fit_kwargs["sample_weight"] = w_all[tr_idx]

        model.fit(X_tr, y_tr, **fit_kwargs)
        mu_oof[va_idx] = model.predict(X_va)

    res2 = (y_ - mu_oof) ** 2
    return np.maximum(res2, eps)


def oof_mean_predictions(
    X: ArrayLike,
    y: np.ndarray,
    *,
    model_factory: Callable[[], Any],
    n_splits: int = 5,
    sample_weight: Optional[np.ndarray] = None,
    splitter: Optional[Any] = None,
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> np.ndarray:
    """Compute out-of-fold mean predictions (for OOF calibration).

    Returns mu_oof: array of shape (n,) where each entry is the mean
    prediction from a model trained on all other folds.
    """
    y_ = as_1d_float("y", y)
    w_all = validate_sample_weight(y_, sample_weight)
    g_all = validate_groups(y_, groups)

    splitter_obj = choose_oof_splitter(
        X,
        n_splits=n_splits,
        random_state=random_state,
        splitter=splitter,
        groups=g_all,
    )

    mu_oof = np.empty(len(y_), dtype=float)

    try:
        split_iter = splitter_obj.split(X, y_, groups=g_all)
    except TypeError:
        split_iter = splitter_obj.split(X, y_)

    for tr_idx, va_idx in split_iter:
        model = model_factory()
        fit_kwargs: Dict[str, Any] = {}
        if w_all is not None:
            fit_kwargs["sample_weight"] = w_all[tr_idx]
        model.fit(_slice_rows(X, tr_idx), y_[tr_idx], **fit_kwargs)
        mu_oof[va_idx] = model.predict(_slice_rows(X, va_idx))

    return mu_oof
