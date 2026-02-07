# nllfit

A small utility package for **two-stage heteroscedastic regression** producing
Gaussian predictive distributions \((\mu(x), \sigma^2(x))\) and evaluating
**Gaussian NLL**.

Key features:

- Two-stage mean/variance modeling for:
  - `glum` (parametric GLMs)
  - `LightGBM` (flexible models)
- Out-of-fold (OOF) residuals for variance modeling to reduce variance leakage.
- Optional scalar variance calibration (closed form).
- Gaussian predictive intervals and quantiles.

## Install (editable / local)

This repo is intended as a starting point. In a real project you would publish to PyPI.

```bash
pip install -e .
```

Optional dependencies:

```bash
pip install -e .[lgbm]
pip install -e .[glum]
pip install -e .[dev]
```

## Quickstart

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nllfit import TwoStageHeteroscedasticLightGBM

n = 5000
X = pd.DataFrame({
    "x1": np.random.randn(n),
    "x2": np.random.randn(n),
})
true_var = np.exp(0.5 + 1.0 * X["x1"].values)
y = (2 * X["x1"] - X["x2"]).values + np.sqrt(true_var) * np.random.randn(n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

m = TwoStageHeteroscedasticLightGBM(
    n_oof_folds=5,
    variance_mode="auto",
    calibration_method="oof",  # "oof" | "holdout" | "none"
)
m.fit(X_train, y_train)

pred = m.predict_dist(X_test)
print("NLL:", m.nll(X_test, y_test))

lo, hi = m.predict_interval(X_test, alpha=0.1)
q = m.predict_quantiles(X_test, [0.1, 0.5, 0.9], output="dict")
```

## License

MIT
