"""nllfit — two-stage heteroscedastic regression with Gaussian NLL evaluation.

Core exports:
- TwoStageHeteroscedasticGLUM
- TwoStageHeteroscedasticLightGBM
- gaussian_nll
- HeteroscedasticPrediction
"""

from ._version import __version__
from .types import HeteroscedasticPrediction, LogNormalPrediction, VarianceCalibration
from .conformal import SplitConformalRegressor
from .metrics import gaussian_nll, laplace_nll, lognormal_nll, student_t_nll
from .distributions import LaplaceWrapper, StudentTWrapper
from .estimators.glum import TwoStageHeteroscedasticGLUM
from .estimators.lightgbm import TwoStageHeteroscedasticLightGBM
from .lognormal import LogNormalRegressor

__all__ = [
    "__version__",
    "HeteroscedasticPrediction",
    "LogNormalPrediction",
    "VarianceCalibration",
    "SplitConformalRegressor",
    "gaussian_nll",
    "laplace_nll",
    "lognormal_nll",
    "student_t_nll",
    "LaplaceWrapper",
    "StudentTWrapper",
    "LogNormalRegressor",
    "TwoStageHeteroscedasticGLUM",
    "TwoStageHeteroscedasticLightGBM",
]
