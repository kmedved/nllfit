from __future__ import annotations

from typing import Any, Optional

import numpy as np


def as_1d_float(name: str, arr: Any) -> np.ndarray:
    """Convert array-like input to a 1D float numpy array.

    Allows scalar and (n,1) column vectors (squeezed to 1D).
    Raises on arrays with more than one non-singleton dimension.
    """
    if arr is None:
        raise ValueError(f"{name} must not be None.")
    out = np.asarray(arr, dtype=float)
    if out.ndim == 0:
        out = out.reshape(1)
    elif out.ndim == 1:
        pass
    elif out.ndim == 2 and out.shape[1] == 1:
        out = out.ravel()
    else:
        raise ValueError(
            f"{name} must be 1D, got shape {out.shape}. "
            f"If passing a 2D array, ensure it has a single column."
        )
    return out


def validate_1d_same_length(y: np.ndarray, **arrays: Optional[np.ndarray]) -> None:
    """Ensure all provided arrays are 1D and match y length."""
    n = len(y)
    for name, arr in arrays.items():
        if arr is None:
            continue
        arr_np = np.asarray(arr)
        if arr_np.ndim != 1:
            arr_np = arr_np.reshape(-1)
        if len(arr_np) != n:
            raise ValueError(f"{name} must have length {n}, got {len(arr_np)}.")


def validate_sample_weight(
    y: np.ndarray,
    sample_weight: Optional[Any],
    *,
    allow_none: bool = True,
    require_positive_sum: bool = True,
) -> Optional[np.ndarray]:
    """Validate sample weights and return a 1D float array if provided.

    Parameters
    ----------
    y : array of shape (n,)
        Reference array for length validation.
    sample_weight : array-like or None
        Weights to validate.
    allow_none : bool
        If True, None is accepted and returned as-is.
    require_positive_sum : bool
        If True (default), raise if sum(weights) <= 0.
    """
    if sample_weight is None:
        if allow_none:
            return None
        raise ValueError("sample_weight must not be None.")

    w = as_1d_float("sample_weight", sample_weight)
    validate_1d_same_length(y, sample_weight=w)

    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight must be finite.")
    if np.any(w < 0):
        raise ValueError("sample_weight must be nonnegative.")
    if require_positive_sum and float(np.sum(w)) <= 0.0:
        raise ValueError("sample_weight must have a positive sum.")

    return w


def validate_groups(y: np.ndarray, groups: Optional[Any]) -> Optional[np.ndarray]:
    """Validate groups array and return a 1D numpy array if provided."""
    if groups is None:
        return None
    g = np.asarray(groups)
    if g.ndim == 2 and g.shape[1] == 1:
        g = g.ravel()
    elif g.ndim != 1:
        raise ValueError(f"groups must be 1D, got shape {g.shape}.")
    validate_1d_same_length(y, groups=g)
    return g


def validate_time_values(y: np.ndarray, time_values: Optional[Any]) -> Optional[np.ndarray]:
    """Validate time values length and return a 1D numpy array if provided."""
    if time_values is None:
        return None
    t = np.asarray(time_values)
    if t.ndim == 2 and t.shape[1] == 1:
        t = t.ravel()
    elif t.ndim != 1:
        raise ValueError(f"time_values must be 1D, got shape {t.shape}.")
    validate_1d_same_length(y, time_values=t)
    return t
