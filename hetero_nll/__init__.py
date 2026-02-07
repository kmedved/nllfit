"""Heteroscedastic Gaussian models (two-stage) with Gaussian NLL evaluation.

Core exports:
- TwoStageHeteroscedasticGLUM
- TwoStageHeteroscedasticLightGBM
- gaussian_nll
- HeteroscedasticPrediction
"""

from ._version import __version__
from .types import HeteroscedasticPrediction, VarianceCalibration
from .metrics import gaussian_nll
from .estimators.glum import TwoStageHeteroscedasticGLUM
from .estimators.lightgbm import TwoStageHeteroscedasticLightGBM

__all__ = [
    "__version__",
    "HeteroscedasticPrediction",
    "VarianceCalibration",
    "gaussian_nll",
    "TwoStageHeteroscedasticGLUM",
    "TwoStageHeteroscedasticLightGBM",
]
