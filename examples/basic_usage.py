import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hetero_nll import TwoStageHeteroscedasticLightGBM, gaussian_nll


def main():
    np.random.seed(0)
    n = 5000
    X = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "x3": np.random.randn(n),
    })

    # Optional: time index makes time-aware splitting kick in
    X.index = pd.date_range("2020-01-01", periods=n, freq="H")

    true_mu = 2 * X["x1"] + 0.5 * X["x2"] - X["x3"]
    true_var = np.exp(0.5 + 1.0 * X["x1"].values)
    y = true_mu.values + np.sqrt(true_var) * np.random.randn(n)

    X_train, X_test, y_train, y_test, var_train, var_test = train_test_split(
        X, y, true_var, test_size=0.2, random_state=42
    )

    model = TwoStageHeteroscedasticLightGBM(
        n_oof_folds=5,
        n_iterations=2,
        variance_mode="auto",
        calibration_method="oof",
        time_col=None,
    )
    model.fit(X_train, y_train)

    pred = model.predict_dist(X_test)
    print("test NLL:", gaussian_nll(y_test, pred.mu, pred.var))
    print("oracle NLL:", gaussian_nll(y_test, pred.mu, var_test))

    lo, hi = model.predict_interval(X_test, alpha=0.1)
    q = model.predict_quantiles(X_test, [0.1, 0.5, 0.9], output="dict")
    print("interval head:", lo[:3], hi[:3])
    print("median head:", q[0.5][:3])


if __name__ == "__main__":
    main()
