# nllfit

`nllfit` is a sklearn-style library for two-stage heteroscedastic regression and uncertainty evaluation.

It provides:

- Point predictions via `predict(X)`.
- Distributional predictions via `predict_dist(X)` with `.mu`, `.var`, `.log_var`.
- Proper-likelihood scoring via `nll(...)` and `score(...)` (`score = -NLL`).
- Quantiles/intervals, conformal intervals, and calibration utilities.

## Full Documentation

- Comprehensive guide: [`docs/COMPREHENSIVE_GUIDE.md`](docs/COMPREHENSIVE_GUIDE.md)
- Example script: [`examples/basic_usage.py`](examples/basic_usage.py)
- Release history: [`CHANGELOG.md`](CHANGELOG.md)
- Contributing guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[lgbm]  # LightGBM estimator
pip install -e .[glum]  # GLUM estimator
pip install -e .[dev]   # pytest, ruff, mypy
```

## Quickstart

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nllfit import TwoStageHeteroscedasticLightGBM

rng = np.random.default_rng(0)
n = 5000
X = pd.DataFrame({
    "x1": rng.normal(size=n),
    "x2": rng.normal(size=n),
})

true_var = np.exp(0.5 + 1.0 * X["x1"].values)
y = (2 * X["x1"] - X["x2"]).values + rng.normal(scale=np.sqrt(true_var))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

m = TwoStageHeteroscedasticLightGBM(
    n_iterations=2,
    n_oof_folds=5,
    variance_mode="auto",
    calibration_method="oof",
)
m.fit(X_train, y_train)

pred = m.predict_dist(X_test)
print("NLL:", m.nll(X_test, y_test))

lo, hi = m.predict_interval(X_test, alpha=0.1)
q = m.predict_quantiles(X_test, [0.1, 0.5, 0.9], output="dict")
```

## Main APIs

Top-level exports:

- Estimators:
  - `TwoStageHeteroscedasticLightGBM`
  - `TwoStageHeteroscedasticGLUM`
  - `LogNormalRegressor`
- Wrappers:
  - `LaplaceWrapper`
  - `StudentTWrapper`
- Conformal:
  - `SplitConformalRegressor`
- Metrics:
  - `gaussian_nll`
  - `laplace_nll`
  - `lognormal_nll`
  - `student_t_nll`
- Types:
  - `HeteroscedasticPrediction`
  - `LogNormalPrediction`
  - `VarianceCalibration`

For module-level APIs (`metrics`, `validation`, `oof`, `splitting`, `calibration`, etc.), see the [comprehensive guide](docs/COMPREHENSIVE_GUIDE.md).

## License

MIT
