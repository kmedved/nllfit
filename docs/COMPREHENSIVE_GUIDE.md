# nllfit Comprehensive Guide

This document is the canonical reference for the `nllfit` repository.
It is intended to cover:

- Every supported functionality in the package.
- Every public API (top-level and module-level).
- End-to-end usage patterns for practical workflows.

For release history, see `CHANGELOG.md`.

## Table of Contents

1. [What nllfit solves](#what-nllfit-solves)
2. [Install and dependency model](#install-and-dependency-model)
3. [Core conventions and invariants](#core-conventions-and-invariants)
4. [Top-level API surface](#top-level-api-surface)
5. [Estimator APIs](#estimator-apis)
6. [Distribution wrappers](#distribution-wrappers)
7. [Conformal prediction API](#conformal-prediction-api)
8. [Metrics API](#metrics-api)
9. [Calibration utilities API](#calibration-utilities-api)
10. [OOF utilities API](#oof-utilities-api)
11. [Splitting and time/group handling API](#splitting-and-timegroup-handling-api)
12. [Validation utilities API](#validation-utilities-api)
13. [Data container types](#data-container-types)
14. [Use-case cookbook](#use-case-cookbook)
15. [Failure modes and troubleshooting](#failure-modes-and-troubleshooting)
16. [Testing and development workflow](#testing-and-development-workflow)
17. [Migration notes](#migration-notes)
18. [Repository layout](#repository-layout)

## What nllfit solves

`nllfit` provides sklearn-style regressors that predict both:

- A point prediction `mu = E[y|x]`.
- A predictive variance `var = Var(y|x)`.

From these, the library supports:

- Gaussian negative log-likelihood evaluation.
- Distributional prediction via `predict_dist(X)`.
- Gaussian quantiles and intervals.
- Alternative likelihood wrappers (Laplace, Student-t, lognormal/log1p-normal).
- Split conformal intervals for distribution-free coverage.

Primary estimator families:

- `TwoStageHeteroscedasticLightGBM` (flexible, OOF-aware, recommended default when using LightGBM).
- `TwoStageHeteroscedasticGLUM` (parametric GLM-based alternative).

## Install and dependency model

Base install (required):

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[lgbm]   # LightGBM estimator
pip install -e .[glum]   # GLUM estimator
pip install -e .[dev]    # pytest, mypy, ruff
```

Hard dependencies:

- `numpy>=1.23`
- `scikit-learn>=1.2`

Optional runtime dependencies:

- `lightgbm` for `TwoStageHeteroscedasticLightGBM`
- `glum` for `TwoStageHeteroscedasticGLUM`
- `scipy` only for `StudentTWrapper.predict_quantiles(...)`
- `pandas` only when using pandas-specific functionality (dataframes, index/time inference, dataframe quantile output)

## Core conventions and invariants

### 1) Sample weights are frequency weights

Everywhere in `nllfit`, `sample_weight` is interpreted as frequency weights.

- Training forwards weights as-is to underlying models.
- Metrics use weighted averages (`np.average(..., weights=w)`).
- Weighted behavior is expected to match duplicated-row behavior for integer weights.

Validation requirements for weights:

- 1D
- same length as `y`
- finite
- nonnegative
- strictly positive total sum

### 2) Shape conventions

Public vector outputs are 1D numpy arrays of shape `(n,)`.

- `predict(X)` -> `(n,)`
- `predict_dist(X).mu` -> `(n,)`
- `predict_dist(X).var` -> `(n,)`
- `predict_dist(X).log_var` -> `(n,)`

Quantiles are `(n, k)` when `output="array"`.

### 3) Leakage control in flexible models

For flexible models (notably LightGBM), `nllfit` is designed to avoid variance leakage:

- Stage-2 variance targets can be built from out-of-fold (OOF) residuals.
- OOF calibration is available and recommended.
- In-sample train calibration exists but is warned against for flexible models.

### 4) Group/time awareness

When groups are provided, group-leakage-safe splitting is used.
When time is provided/inferred, time-aware splitting avoids temporal shuffling.

- Groups are not cast to float and may be strings/object dtype.
- Time can come from `time_col` or pandas time index.

### 5) Scoring convention

`score(X, y)` returns **negative NLL** (higher is better), consistent with sklearn "higher-is-better" scoring semantics.

## Top-level API surface

The package exports the following in `nllfit.__init__`:

```python
from nllfit import (
    __version__,
    HeteroscedasticPrediction,
    LogNormalPrediction,
    VarianceCalibration,
    SplitConformalRegressor,
    gaussian_nll,
    laplace_nll,
    lognormal_nll,
    student_t_nll,
    LaplaceWrapper,
    StudentTWrapper,
    LogNormalRegressor,
    TwoStageHeteroscedasticGLUM,
    TwoStageHeteroscedasticLightGBM,
)
```

Additional APIs are available via submodules (documented below), such as:

- `nllfit.metrics.log1p_normal_nll`
- `nllfit.metrics.crps_gaussian`
- `nllfit.oof.oof_squared_residuals`
- `nllfit.splitting.calibration_split`
- `nllfit.validation.validate_sample_weight`

## Estimator APIs

### Base class: `nllfit.estimators.base.HeteroscedasticRegressor`

Abstract base with sklearn-style behavior.

Methods:

```python
predict_dist(self, X) -> HeteroscedasticPrediction  # abstract
predict(self, X) -> np.ndarray
nll(self, X, y, *, sample_weight=None, include_const=False) -> float
score(self, X, y, sample_weight=None) -> float
predict_quantiles(self, X, quantiles, *, output="array", column_prefix="q")
predict_interval(self, X, *, alpha=0.1) -> tuple[np.ndarray, np.ndarray]
```

Notes:

- Quantile and interval methods assume Gaussian predictive distribution from `(mu, var)`.
- `output` supports `"array" | "dict" | "dataframe"`.

### `TwoStageHeteroscedasticLightGBM`

Location: `nllfit.estimators.lightgbm.TwoStageHeteroscedasticLightGBM`

Constructor:

```python
TwoStageHeteroscedasticLightGBM(
    *,
    mean_params=None,
    var_params=None,
    n_iterations=2,
    n_oof_folds=5,
    oof_splitter=None,
    oof_random_state=42,
    variance_mode="auto",      # "auto" | "gamma" | "log"
    oof_residuals_reuse=True,
    calibration_method="oof",  # "none" | "holdout" | "oof" | "train"
    calibration_fraction=0.2,
    calibration_random_state=123,
    time_col=None,
    eps=1e-12,
    random_state=None,
    n_jobs=None,
    calibrate=None,             # deprecated
)
```

`fit` API:

```python
fit(
    X,
    y,
    *,
    sample_weight=None,
    groups=None,
    X_cal=None,
    y_cal=None,
    sample_weight_cal=None,
) -> TwoStageHeteroscedasticLightGBM
```

`predict_dist` API:

```python
predict_dist(self, X) -> HeteroscedasticPrediction
```

Inherited APIs: `predict`, `nll`, `score`, `predict_quantiles`, `predict_interval`.

Algorithm details:

- Stage 1 mean model: `LGBMRegressor` on `y`.
- Stage 2 variance model:
  - `variance_mode="gamma"`: train on squared residuals with gamma objective.
  - `variance_mode="log"`: train on `log(residual^2)` with regression objective, exponentiate at prediction.
  - `variance_mode="auto"`: try gamma, fallback to log on failure.
- Iterative reweighting (`n_iterations>1`): stage-1 can be reweighted by inverse predicted variance.
- OOF residual target generation when `n_oof_folds>1`.

Calibration methods:

- `"oof"` (recommended): uses OOF mean predictions + training variance predictions.
- `"holdout"`: split off calibration subset or use explicit `X_cal/y_cal`.
- `"train"`: in-sample calibration (warned for flexible models).
- `"none"`: no variance scaling.

Time/group handling:

- If `time_col` is provided or pandas time index exists, data is time-sorted before internal splits.
- Group-aware splitting is used for both holdout and OOF where applicable.

Fitted attributes commonly used:

- `mean_model_`
- `var_model_`
- `variance_mode_`
- `variance_model_info_` (dataclass with mode/objective/details)
- `calibration_` (`VarianceCalibration`)
- `calibration_strategy_`
- `eps_`

`variance_model_info_` is an instance of:

```python
VarianceModelInfo(mode: str, objective: str, details: str = "")
```

where `mode` is `\"gamma\"` or `\"log\"`.

Important constraints:

- `calibration_method="oof"` requires `n_oof_folds > 1`.
- `variance_mode` must be one of `auto|gamma|log`.
- Invalid `calibration_method` raises `ValueError`.

Default model parameter sets (when not provided):

- Mean: objective regression, 500 trees, learning rate 0.05, leaves 31.
- Variance: 300 trees, learning rate 0.03, leaves 31, heavier regularization.

### `TwoStageHeteroscedasticGLUM`

Location: `nllfit.estimators.glum.TwoStageHeteroscedasticGLUM`

Constructor:

```python
TwoStageHeteroscedasticGLUM(
    *,
    alpha=0.0,
    l1_ratio=0.0,
    n_iterations=1,
    calibration_method="none",  # "none" | "holdout" | "train"
    calibration_fraction=0.2,
    calibration_random_state=123,
    time_col=None,
    eps=1e-12,
    calibrate=None,              # deprecated
)
```

`fit` API:

```python
fit(
    X,
    y,
    *,
    sample_weight=None,
    groups=None,
    X_cal=None,
    y_cal=None,
    sample_weight_cal=None,
) -> TwoStageHeteroscedasticGLUM
```

`predict_dist` API:

```python
predict_dist(self, X) -> HeteroscedasticPrediction
```

Inherited APIs: `predict`, `nll`, `score`, `predict_quantiles`, `predict_interval`.

Algorithm details:

- Stage 1 mean GLM: normal family with identity link.
- Stage 2 variance GLM: gamma family with log link on squared residuals.
- Iterative alternating updates supported (`n_iterations`).

Calibration methods:

- `"none"` (default)
- `"holdout"`
- `"train"` (safe in parametric GLM context)
- `"oof"` is not supported for GLUM; if passed, it warns and falls back to `"none"`.

Fitted attributes commonly used:

- `mean_model_`
- `var_model_`
- `calibration_`
- `calibration_strategy_`
- `eps_`

Important constraints:

- Invalid `calibration_method` raises `ValueError`.
- Explicit `X_cal/y_cal` overrides calibration strategy to holdout behavior.

## Distribution wrappers

### `LaplaceWrapper`

Location: `nllfit.distributions.LaplaceWrapper`

Purpose:

- Wrap any estimator that exposes `predict_dist(X)` with `.mu` and `.var`.
- Reinterpret `(mu, var)` under Laplace distribution semantics.

API:

```python
LaplaceWrapper(base_estimator, *, eps=1e-12)
fit(X, y, *, sample_weight=None, groups=None, **kwargs)
predict(X)
predict_dist(X)
nll(X, y, *, sample_weight=None, include_const=False)
score(X, y, sample_weight=None)
predict_quantiles(X, quantiles, *, output="array", column_prefix="q")
predict_interval(X, *, alpha=0.1)
```

### `StudentTWrapper`

Location: `nllfit.distributions.StudentTWrapper`

Purpose:

- Wrap any `mu/var` estimator and evaluate/predict using Student-t family.

API:

```python
StudentTWrapper(base_estimator, df, *, eps=1e-12)
fit(X, y, *, sample_weight=None, groups=None, **kwargs)
predict(X)
predict_dist(X)
nll(X, y, *, sample_weight=None, include_const=False)
score(X, y, sample_weight=None)
predict_quantiles(X, quantiles, *, output="array", column_prefix="q")
predict_interval(X, *, alpha=0.1)
```

Important constraints:

- `df > 2` required when `var` is interpreted as variance.
- `predict_quantiles` requires `scipy` (`scipy.stats.t.ppf`).
- `nll` does not require scipy.

### `laplace_ppf`

Location: `nllfit.distributions.laplace_ppf`

API:

```python
laplace_ppf(p, mu, var, *, eps=1e-12) -> np.ndarray
```

Notes:

- `p` quantiles must be strictly inside `(0, 1)`.

## Conformal prediction API

### `SplitConformalRegressor`

Location: `nllfit.conformal.SplitConformalRegressor`

Constructor:

```python
SplitConformalRegressor(
    base_estimator,
    *,
    method="absolute",      # "absolute" | "normalized" | "cqr"
    alpha=0.1,
    calibration_fraction=0.2,
    time_col=None,
    calibration_random_state=42,
)
```

`fit` API:

```python
fit(
    X,
    y,
    *,
    sample_weight=None,
    groups=None,
    X_cal=None,
    y_cal=None,
    sample_weight_cal=None,
) -> SplitConformalRegressor
```

Prediction APIs:

```python
predict(X) -> np.ndarray
predict_interval(X) -> tuple[np.ndarray, np.ndarray]
predict_dist(X) -> delegated distribution object
```

Method semantics:

- `absolute`: score is `|y - mu|`; constant-width intervals.
- `normalized`: score is `|y - mu| / sigma`; adaptive-width intervals.
- `cqr`: conformalized quantile regression using base `predict_interval`.

Base-estimator requirements:

- always: `fit(X, y, **kwargs)` and `predict(X)`
- for `normalized`: `predict_dist(X)` with `.var`
- for `cqr`: `predict_interval(X, alpha=...)`

Internal quantile helper:

- `_conformal_quantile(scores, alpha, sample_weight=None)`
- Uses finite-sample correction and weighted-frequency semantics.

Fitted state:

- `q_` conformal correction amount
- `is_fitted_` boolean guard

## Metrics API

Location: `nllfit.metrics`

### Likelihood / scoring metrics

```python
gaussian_nll(y, mu, var, *, sample_weight=None, eps=1e-12, include_const=False) -> float
laplace_nll(y, mu, var, *, sample_weight=None, eps=1e-12, include_const=False) -> float
student_t_nll(y, mu, var, df, *, sample_weight=None, eps=1e-12, include_const=False) -> float
lognormal_nll(y, mu_log, var_log, *, sample_weight=None, eps=1e-12, include_const=False) -> float
log1p_normal_nll(y, mu_log, var_log, *, sample_weight=None, eps=1e-12, include_const=False) -> float
```

### Point / interval quality metrics

```python
rmse(y, mu, *, sample_weight=None) -> float
mae(y, mu, *, sample_weight=None) -> float
interval_coverage(y, lo, hi, *, sample_weight=None) -> float
interval_width(lo, hi, *, sample_weight=None) -> float
crps_gaussian(y, mu, var, *, sample_weight=None, eps=1e-12) -> float
pit_gaussian(y, mu, var, *, eps=1e-12) -> np.ndarray
```

Domain and behavior notes:

- `student_t_nll` requires `df > 2`.
- `lognormal_nll` requires `y > 0`.
- `log1p_normal_nll` requires `y >= 0`.
- `pit_gaussian` returns per-sample CDF values (calibration diagnostic).
- Variances are clipped by `eps` for numerical stability.

Internal helper in module:

- `_erf_approx(x)` is used to avoid mandatory scipy dependency for Gaussian CDF/CRPS formulas.

## Calibration utilities API

Location: `nllfit.calibration`

```python
fit_variance_scale(y, mu, var_raw, *, sample_weight=None, eps=1e-12, source="holdout") -> VarianceCalibration
apply_variance_calibration(var_raw, cal, *, eps=1e-12) -> np.ndarray
```

Behavior:

- Fits scalar `c` in `var_cal = c * var_raw` minimizing Gaussian NLL.
- Closed-form optimum uses mean/weighted-mean of `(y-mu)^2 / var_raw`.
- `source` is metadata (`"holdout"`, `"oof"`, `"train"`, `"none"`, etc.).

## OOF utilities API

Location: `nllfit.oof`

```python
choose_oof_splitter(X, *, n_splits, random_state, splitter=None, groups=None)
oof_squared_residuals(
    X,
    y,
    *,
    model_factory,
    n_splits=5,
    sample_weight=None,
    splitter=None,
    groups=None,
    random_state=42,
    eps=1e-12,
) -> np.ndarray
oof_mean_predictions(
    X,
    y,
    *,
    model_factory,
    n_splits=5,
    sample_weight=None,
    splitter=None,
    groups=None,
    random_state=42,
) -> np.ndarray
```

Splitter selection priority:

1. explicit `splitter` argument
2. `GroupKFold` when groups are provided
3. `TimeSeriesSplit` for pandas time-like index
4. shuffled `KFold`

Safety behavior:

- warns/clamps when `n_splits` exceeds groups or samples.
- raises when splits are invalid (e.g., fewer than 2 effective folds).

## Splitting and time/group handling API

Location: `nllfit.splitting`

### `TimeInfo` dataclass

```python
TimeInfo(kind: str, values: Optional[np.ndarray], name: Optional[str] = None)
```

- `kind` is one of `"none" | "index" | "column"`.

### `infer_time`

```python
infer_time(X, *, time_col=None) -> TimeInfo
```

Priority:

1. explicit `time_col` in pandas dataframe
2. pandas time-like index
3. no time information

### `time_sort`

```python
time_sort(X, y, w, groups, time) -> (X_sorted, y_sorted, w_sorted, g_sorted, order)
```

- Sorts by inferred/provided time where available.
- Returns `order=None` when no time data exists.

### `calibration_split`

```python
calibration_split(
    X,
    y,
    *,
    sample_weight,
    groups,
    time,
    calibration_fraction,
    random_state,
)
```

Returns:

```python
(
    X_tr, X_cal,
    y_tr, y_cal,
    w_tr, w_cal,
    g_tr, g_cal,
    strategy_name,
)
```

Strategy selection:

- groups + time: group-time-ordered split by group last timestamp
- groups only: `GroupShuffleSplit`
- time only: quantile cutoff on time values
- neither: random train/test split

## Validation utilities API

Location: `nllfit.validation`

```python
as_1d_float(name, arr) -> np.ndarray
validate_1d_same_length(y, **arrays) -> None
validate_sample_weight(y, sample_weight, *, allow_none=True, require_positive_sum=True) -> Optional[np.ndarray]
validate_groups(y, groups) -> Optional[np.ndarray]
validate_time_values(y, time_values) -> Optional[np.ndarray]
```

Highlights:

- `as_1d_float` accepts scalar and `(n,1)` and normalizes to `(n,)`.
- Validation functions raise explicit `ValueError` on mismatches.

## Data container types

Location: `nllfit.types`

### `HeteroscedasticPrediction`

```python
HeteroscedasticPrediction(mu, var, log_var)
```

### `LogNormalPrediction`

```python
LogNormalPrediction(mu, var, log_var, mu_log, var_log)
```

### `VarianceCalibration`

```python
VarianceCalibration(scale: float = 1.0, source: str = "none")
```

## Use-case cookbook

The examples below intentionally cover all supported workflows in the current codebase.

### 1) Basic LightGBM two-stage heteroscedastic regression

```python
import numpy as np
import pandas as pd
from nllfit import TwoStageHeteroscedasticLightGBM

n = 5000
rng = np.random.default_rng(0)
X = pd.DataFrame({
    "x1": rng.normal(size=n),
    "x2": rng.normal(size=n),
})
true_var = np.exp(0.5 + 1.0 * X["x1"].values)
y = (2 * X["x1"] - X["x2"]).values + rng.normal(scale=np.sqrt(true_var))

m = TwoStageHeteroscedasticLightGBM(
    n_iterations=2,
    n_oof_folds=5,
    variance_mode="auto",
    calibration_method="oof",
)
m.fit(X, y)

pred = m.predict_dist(X)
mu = pred.mu
var = pred.var
lo, hi = m.predict_interval(X, alpha=0.1)
```

### 2) GLUM-based two-stage model

```python
from nllfit import TwoStageHeteroscedasticGLUM

m = TwoStageHeteroscedasticGLUM(
    alpha=0.0,
    l1_ratio=0.0,
    n_iterations=1,
    calibration_method="train",  # valid for GLM context
)
m.fit(X, y)
```

### 3) Weighted training (frequency semantics)

```python
weights = np.ones(len(y))
weights[:100] = 3.0

m = TwoStageHeteroscedasticLightGBM(calibration_method="none")
m.fit(X, y, sample_weight=weights)
```

### 4) Group-aware splitting/calibration

```python
groups = np.array([f"g{i % 10}" for i in range(len(y))], dtype=object)

m = TwoStageHeteroscedasticLightGBM(
    n_oof_folds=3,
    calibration_method="holdout",
    calibration_fraction=0.2,
)
m.fit(X, y, groups=groups)
```

### 5) Time-aware splitting via explicit `time_col`

```python
X = X.copy()
X["event_time"] = pd.date_range("2024-01-01", periods=len(X), freq="h")

m = TwoStageHeteroscedasticLightGBM(
    calibration_method="holdout",
    time_col="event_time",
)
m.fit(X, y)
```

### 6) Time-aware splitting via datetime index

```python
X_idx = X.copy()
X_idx.index = pd.date_range("2024-01-01", periods=len(X_idx), freq="h")

m = TwoStageHeteroscedasticLightGBM(calibration_method="holdout")
m.fit(X_idx, y)
```

### 7) Explicit calibration set override

```python
from sklearn.model_selection import train_test_split

X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.2, random_state=0)

m = TwoStageHeteroscedasticLightGBM(calibration_method="none")
m.fit(X_tr, y_tr, X_cal=X_cal, y_cal=y_cal)
# calibration uses explicit holdout regardless of calibration_method
```

### 8) Control variance-stage mode

```python
# force gamma
m_gamma = TwoStageHeteroscedasticLightGBM(variance_mode="gamma")

# force log-residual regression
m_log = TwoStageHeteroscedasticLightGBM(variance_mode="log")

# auto: gamma first, fallback to log if needed
m_auto = TwoStageHeteroscedasticLightGBM(variance_mode="auto")
```

### 9) Quantiles in multiple output formats

```python
m = TwoStageHeteroscedasticLightGBM(calibration_method="none")
m.fit(X, y)

arr = m.predict_quantiles(X, [0.1, 0.5, 0.9], output="array")
as_dict = m.predict_quantiles(X, [0.1, 0.5, 0.9], output="dict")
as_df = m.predict_quantiles(X, [0.1, 0.5, 0.9], output="dataframe")  # requires pandas
```

### 10) NLL and sklearn-compatible score

```python
nll = m.nll(X, y)
score = m.score(X, y)   # equals -nll
```

### 11) Laplace likelihood wrapper

```python
from nllfit import LaplaceWrapper, TwoStageHeteroscedasticLightGBM

base = TwoStageHeteroscedasticLightGBM(calibration_method="oof")
lap = LaplaceWrapper(base)
lap.fit(X, y)
lap_nll = lap.nll(X, y)
```

### 12) Student-t likelihood wrapper

```python
from nllfit import StudentTWrapper, TwoStageHeteroscedasticLightGBM

base = TwoStageHeteroscedasticLightGBM(calibration_method="oof")
t_model = StudentTWrapper(base, df=5.0)
t_model.fit(X, y)
t_nll = t_model.nll(X, y)
```

### 13) Positive-target modeling with `LogNormalRegressor`

```python
from nllfit import LogNormalRegressor, TwoStageHeteroscedasticLightGBM

# y must be > 0 for transform="log"
y_pos = np.exp(y - y.min() + 0.1)

base = TwoStageHeteroscedasticLightGBM(calibration_method="none")
log_model = LogNormalRegressor(base, transform="log")
log_model.fit(X, y_pos)
pred = log_model.predict_dist(X)
```

### 14) Nonnegative-target modeling with `transform="log1p"`

```python
y_nonneg = np.maximum(y, 0.0)
base = TwoStageHeteroscedasticLightGBM(calibration_method="none")
log1p_model = LogNormalRegressor(base, transform="log1p")
log1p_model.fit(X, y_nonneg)
```

### 15) Distribution-free split conformal intervals (absolute)

```python
from nllfit import SplitConformalRegressor, TwoStageHeteroscedasticLightGBM

base = TwoStageHeteroscedasticLightGBM(calibration_method="none")
cr = SplitConformalRegressor(base, method="absolute", alpha=0.1)
cr.fit(X, y)
lo, hi = cr.predict_interval(X)
```

### 16) Conformal normalized intervals (variance-adaptive)

```python
cr_norm = SplitConformalRegressor(base, method="normalized", alpha=0.1)
cr_norm.fit(X, y)
lo, hi = cr_norm.predict_interval(X)
```

### 17) Conformalized quantile regression (CQR)

```python
cr_cqr = SplitConformalRegressor(base, method="cqr", alpha=0.1)
cr_cqr.fit(X, y)
lo, hi = cr_cqr.predict_interval(X)
```

### 18) Low-level metric evaluation

```python
from nllfit.metrics import (
    gaussian_nll,
    crps_gaussian,
    pit_gaussian,
    interval_coverage,
    interval_width,
)

pred = m.predict_dist(X)
nll = gaussian_nll(y, pred.mu, pred.var)
crps = crps_gaussian(y, pred.mu, pred.var)
pit = pit_gaussian(y, pred.mu, pred.var)

lo, hi = m.predict_interval(X, alpha=0.1)
cov = interval_coverage(y, lo, hi)
wid = interval_width(lo, hi)
```

### 19) Direct calibration utility usage

```python
from nllfit.calibration import fit_variance_scale, apply_variance_calibration

raw_var = pred.var
cal = fit_variance_scale(y, pred.mu, raw_var)
cal_var = apply_variance_calibration(raw_var, cal)
```

### 20) Manual OOF utility usage

```python
from sklearn.linear_model import LinearRegression
from nllfit.oof import oof_squared_residuals, oof_mean_predictions

X_np = np.asarray(X)

def model_factory():
    return LinearRegression()

mu_oof = oof_mean_predictions(X_np, y, model_factory=model_factory, n_splits=5)
res2_oof = oof_squared_residuals(X_np, y, model_factory=model_factory, n_splits=5)
```

## Failure modes and troubleshooting

### Import errors for optional backends

- If `lightgbm` is missing: LightGBM estimator raises `ImportError` during `fit`.
- If `glum` is missing: GLUM estimator raises `ImportError` during `fit`.
- If `scipy` is missing: `StudentTWrapper.predict_quantiles(...)` raises `ImportError`.

### Invalid calibration method

- LightGBM accepts: `none`, `holdout`, `oof`, `train`.
- GLUM accepts: `none`, `holdout`, `train` (`oof` falls back to `none` with warning).

### OOF setup errors

- `n_oof_folds <= 1` is incompatible with `calibration_method="oof"`.
- Too few samples/groups for chosen folds raises errors or emits clamping warnings.

### Domain errors

- `lognormal_nll` requires strictly positive targets.
- `log1p_normal_nll` requires nonnegative targets.
- Student-t variance-parameterized NLL requires `df > 2`.

### Shape and type errors

Typical causes:

- non-1D `y`/predictions not compatible with validators
- length mismatches
- invalid sample weights (negative/non-finite/zero-sum)

### Calibration caveat

`calibration_method="train"` on flexible models can shrink variance and under-cover.
Prefer `"oof"` or `"holdout"` for LightGBM.

## Testing and development workflow

Install dev dependencies:

```bash
pip install -e .[dev]
```

Run tests:

```bash
python -m pytest -q
```

Recommended quality checks:

```bash
ruff check .
ruff format .
mypy nllfit
```

Current tests cover:

- metrics and weighted semantics
- OOF and splitting logic
- conformal quantile correctness
- LightGBM/GLUM smoke behavior
- wrappers (Laplace/Student-t/lognormal)

## Migration notes

From `hetero_nll` to `nllfit`:

- package name changed: `hetero_nll` -> `nllfit`
- update all imports accordingly

Calibration API migration:

- `calibrate: bool` is deprecated in both main estimators.
- use `calibration_method` instead.
- deprecation warnings indicate planned removal in v0.3.0.

## Repository layout

```text
nllfit/
  __init__.py
  metrics.py
  validation.py
  splitting.py
  oof.py
  calibration.py
  conformal.py
  distributions.py
  lognormal.py
  types.py
  estimators/
    base.py
    glum.py
    lightgbm.py
examples/
  basic_usage.py
tests/
  test_*.py
README.md
CHANGELOG.md
CONTRIBUTING.md
```
