from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from .validation import validate_groups, validate_sample_weight, validate_time_values

ArrayLike = Union[np.ndarray, "pd.DataFrame", "pd.Series"]


@dataclass(frozen=True)
class TimeInfo:
    """Time information inferred from X."""

    kind: str  # "none" | "index" | "column"
    values: Optional[np.ndarray]  # dtype datetime64[ns] or numeric
    name: Optional[str] = None


def infer_time(
    X: ArrayLike,
    *,
    time_col: Optional[str] = None,
) -> TimeInfo:
    """Infer time ordering values from X.

    Priority:
      1) If time_col is provided and X is a pandas DataFrame with that column,
         use that column.
      2) Else if X has a pandas time-like index, use the index.
      3) Else: no time info.

    Returns
    -------
    TimeInfo with .values as numpy array of time-like values (datetime64[ns])
    when possible.
    """
    if pd is None:
        return TimeInfo(kind="none", values=None, name=None)

    if time_col is not None and hasattr(X, "__getitem__"):
        try:
            col = X[time_col]  # type: ignore[index]
            if isinstance(col, (pd.Series, pd.Index)):
                values = np.asarray(col.values)
            else:
                values = np.asarray(col)
            return TimeInfo(kind="column", values=values, name=time_col)
        except Exception:
            pass

    if hasattr(X, "index"):
        idx = X.index  # type: ignore[attr-defined]
        if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
            return TimeInfo(kind="index", values=np.asarray(idx.values), name=getattr(idx, "name", None))

    return TimeInfo(kind="none", values=None, name=None)


def _to_time_numeric(t: np.ndarray) -> np.ndarray:
    """Convert time-like array to numeric for quantiles/cutoffs.

    Supports:
    - datetime64 / timedelta64 arrays
    - pandas datetime-like columns (including object columns that can be parsed)
    - numeric arrays
    """
    t = np.asarray(t)

    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]").astype("int64")

    if np.issubdtype(t.dtype, np.timedelta64):
        return t.astype("timedelta64[ns]").astype("int64")

    # Object dtype sometimes appears for datetime columns that were not normalized.
    if t.dtype == object and pd is not None:
        try:
            tt = pd.to_datetime(t)
            return np.asarray(tt.values).astype("datetime64[ns]").astype("int64")
        except Exception:
            pass

    return t.astype(float)


def time_sort(
    X: ArrayLike,
    y: np.ndarray,
    w: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    time: TimeInfo,
) -> Tuple[ArrayLike, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Sort rows by time values (or time index) if available.

    Returns sorted (X, y, w, groups, order). If no time, order is None.
    """
    if time.kind == "none" or time.values is None:
        return X, y, w, groups, None

    t = np.asarray(time.values)
    order = np.argsort(_to_time_numeric(t))

    # If already sorted, do nothing
    if np.all(order == np.arange(len(order))):
        return X, y, w, groups, order

    if pd is not None and hasattr(X, "iloc"):
        Xs = X.iloc[order]  # type: ignore[attr-defined]
    else:
        Xs = np.asarray(X)[order]

    ys = y[order]
    ws = None if w is None else w[order]
    gs = None if groups is None else groups[order]
    return Xs, ys, ws, gs, order


def calibration_split(
    X: ArrayLike,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    time: TimeInfo,
    calibration_fraction: float,
    random_state: int,
) -> Tuple[Any, Any, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
    """Create an internal calibration holdout split.

    Strategy:
      - If groups is provided:
          - If time is available: split by groups using group last time; last fraction of groups
            become calibration. (No group leakage.)
          - Else: GroupShuffleSplit
      - Else (no groups):
          - If time is available: split by time cutoff at (1-frac) quantile of time values.
          - Else: random train_test_split

    Returns
    -------
    (X_tr, X_cal, y_tr, y_cal, w_tr, w_cal, g_tr, g_cal, strategy_name)
    """
    if not (0.0 < calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be in (0, 1)")

    y = np.asarray(y)
    if y.ndim != 1:
        y = y.reshape(-1)

    w = validate_sample_weight(y, sample_weight)
    g = validate_groups(y, groups)
    if time.values is not None:
        validate_time_values(y, time.values)

    n = len(y)

    has_time = (time.kind != "none") and (time.values is not None)

    if g is not None:
        if has_time:
            # Group time-ordered split: last fraction of groups by last observed time.
            tnum = _to_time_numeric(np.asarray(time.values))
            # last time per group
            uniq = np.unique(g)
            last_t = np.empty(len(uniq), dtype=float)
            for i, ug in enumerate(uniq):
                last_t[i] = float(np.max(tnum[g == ug]))

            order = np.argsort(last_t)
            n_cal_groups = max(1, int(np.floor(len(uniq) * calibration_fraction)))
            cal_groups = set(uniq[order[-n_cal_groups:]].tolist())

            cal_mask = np.array([gi in cal_groups for gi in g], dtype=bool)
            tr_mask = ~cal_mask

            if tr_mask.sum() == 0 or cal_mask.sum() == 0:
                raise ValueError("Degenerate calibration split; adjust calibration_fraction.")

            if pd is not None and hasattr(X, "iloc"):
                X_tr = X.iloc[np.where(tr_mask)[0]]  # type: ignore[attr-defined]
                X_cal = X.iloc[np.where(cal_mask)[0]]  # type: ignore[attr-defined]
            else:
                Xa = np.asarray(X)
                X_tr = Xa[tr_mask]
                X_cal = Xa[cal_mask]

            y_tr, y_cal = y[tr_mask], y[cal_mask]
            w_tr = None if w is None else w[tr_mask]
            w_cal = None if w is None else w[cal_mask]
            g_tr, g_cal = g[tr_mask], g[cal_mask]
            return X_tr, X_cal, y_tr, y_cal, w_tr, w_cal, g_tr, g_cal, "group_time_ordered"

        # GroupShuffleSplit (random groups)
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=calibration_fraction, random_state=random_state)
        tr_idx, cal_idx = next(splitter.split(np.zeros(n), y, groups=g))

        if pd is not None and hasattr(X, "iloc"):
            X_tr = X.iloc[tr_idx]  # type: ignore[attr-defined]
            X_cal = X.iloc[cal_idx]  # type: ignore[attr-defined]
        else:
            Xa = np.asarray(X)
            X_tr = Xa[tr_idx]
            X_cal = Xa[cal_idx]

        y_tr, y_cal = y[tr_idx], y[cal_idx]
        w_tr = None if w is None else w[tr_idx]
        w_cal = None if w is None else w[cal_idx]
        g_tr, g_cal = g[tr_idx], g[cal_idx]
        return X_tr, X_cal, y_tr, y_cal, w_tr, w_cal, g_tr, g_cal, "group_shuffle"

    # No groups
    if has_time:
        tnum = _to_time_numeric(np.asarray(time.values))
        cutoff = np.quantile(tnum, 1.0 - calibration_fraction)
        cal_mask = tnum >= cutoff

        # Guarantee both sides non-empty (adjust cutoff if needed)
        if cal_mask.sum() == 0:
            cal_mask[np.argmax(tnum)] = True
        if (~cal_mask).sum() == 0:
            cal_mask[np.argmin(tnum)] = False

        tr_mask = ~cal_mask

        if pd is not None and hasattr(X, "iloc"):
            X_tr = X.iloc[np.where(tr_mask)[0]]  # type: ignore[attr-defined]
            X_cal = X.iloc[np.where(cal_mask)[0]]  # type: ignore[attr-defined]
        else:
            Xa = np.asarray(X)
            X_tr = Xa[tr_mask]
            X_cal = Xa[cal_mask]

        y_tr, y_cal = y[tr_mask], y[cal_mask]
        w_tr = None if w is None else w[tr_mask]
        w_cal = None if w is None else w[cal_mask]
        return X_tr, X_cal, y_tr, y_cal, w_tr, w_cal, None, None, "time_quantile"

    # Random split
    from sklearn.model_selection import train_test_split

    if w is None:
        X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=calibration_fraction, random_state=random_state)
        return X_tr, X_cal, y_tr, y_cal, None, None, None, None, "random"

    X_tr, X_cal, y_tr, y_cal, w_tr, w_cal = train_test_split(
        X, y, w, test_size=calibration_fraction, random_state=random_state
    )
    return X_tr, X_cal, y_tr, y_cal, w_tr, w_cal, None, None, "random"
