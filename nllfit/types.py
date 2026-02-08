from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class HeteroscedasticPrediction(NamedTuple):
    """Distributional prediction for y|x ~ Normal(mu, var)."""

    mu: np.ndarray
    var: np.ndarray
    log_var: np.ndarray


class LogNormalPrediction(NamedTuple):
    """Prediction for lognormal/log1p-normal distributions."""

    mu: np.ndarray
    var: np.ndarray
    log_var: np.ndarray
    mu_log: np.ndarray
    var_log: np.ndarray


@dataclass(frozen=True)
class VarianceCalibration:
    """Multiplicative variance calibration: var_cal = scale * var_raw."""

    scale: float = 1.0
    source: str = "none"  # "holdout" | "oof" | "train" | "none"
