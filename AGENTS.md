<<<<<<< ours
# AGENTS.md — nllfit

This repository implements two-stage heteroscedastic regression and uncertainty evaluation.
Agents should follow the rules below when making changes.

## Project goals

- Provide **reliable distributional predictions** with a sklearn-like API:
  - `predict(X)` returns point prediction (mean in original target units)
  - `predict_dist(X)` returns distributional parameters with `.mu`, `.var`, `.log_var`
  - `score(X, y)` returns **negative** NLL (higher is better)
- Make uncertainty methods **robust** to:
  - sample weights
  - group splits
  - time-aware splits
  - small datasets / edge cases

## Non-negotiable invariants

### 1) Sample weights semantics (frequency weights)
- Everywhere in this codebase, `sample_weight` is treated as **frequency weights**.
  - Metrics use a **weighted average** (e.g., `np.average(per_sample, weights=w)`).
  - Training forwards `sample_weight` to the underlying model **as-is**.
- Validation requirements:
  - 1D, length = `len(y)`
  - finite
  - nonnegative
  - sum(weights) > 0
- If a new function computes a statistic over samples (mean/quantile/calibration),
  it must support `sample_weight` unless there is a clear documented reason.

### 2) Shapes and types
- Public-facing vector outputs should be **1D numpy arrays** of shape `(n,)`.
- Use `nllfit.validation` helpers:
  - `as_1d_float`
  - `validate_1d_same_length`
  - `validate_sample_weight`
  - `validate_groups`
  - `validate_time_values`
- If adding new prediction container types, they must at minimum provide:
  - `.mu`, `.var`, `.log_var`

### 3) No in-sample leakage for flexible models
- For flexible models (e.g., tree boosting), avoid in-sample residual leakage:
  - Stage-2 targets should be based on **out-of-fold** mean predictions when possible.
  - Calibration must **not** use in-sample residuals unless explicitly requested.
- If a feature requires in-sample residuals, it must:
  - be opt-in, and
  - include a warning in the docstring and at runtime.

### 4) Time and groups handling
- If time is available (time index or a `time_col`), time-aware sorting/splitting must not shuffle time.
- When groups are provided:
  - splits must avoid group leakage (GroupKFold / GroupShuffleSplit / group-time-ordered split).
- Groups may be object dtype (e.g., strings). Never cast groups to float.

### 5) Backward compatibility
- Do not break existing imports and class names without a deprecation cycle.
- Add warnings for deprecated parameters.
- Keep defaults stable unless there is a strong correctness reason.

### 6) Dependencies
- Keep hard dependencies minimal (numpy + sklearn + optional lightgbm/glum).
- Optional functionality may use optional dependencies:
  - Import optional deps inside functions/classes and raise a clear `ImportError`.
- Do not add a new hard dependency without updating documentation and tests accordingly.

## Implementation style

- Prefer small, focused functions and modules.
- Keep docstrings explicit about:
  - distributional assumptions
  - sample weight semantics
  - calibration strategy
  - what is and isn’t covered by guarantees (especially conformal)

## Testing expectations (“definition of done”)

A change is complete only when:
1) New behavior is covered by unit tests.
2) Existing tests still pass.
3) Tests cover sample_weight handling if the change touches:
   - metrics
   - calibration
   - conformal quantiles
   - OOF generation
   - splitting
   - estimator fit/predict logic

Suggested commands:
- `python -m pytest -q`

## Common pitfalls to avoid

- Returning negative conformal scores (CQR scores must be >= 0).
- Using `np.quantile` interpolation for conformal order statistics (can break guarantees).
- Forgetting to forward `sample_weight` into internal folds / calibration splits.
- Casting `groups` to float (breaks string groups).
- Failing silently on mismatched lengths (must raise ValueError).

## PR / patch checklist

Before finishing:
- [ ] Updated/added tests
- [ ] Verified sample_weight semantics end-to-end
- [ ] Verified time/group split behavior (where relevant)
- [ ] Verified predict_dist outputs positive variance and finite log_var
- [ ] Ran `pytest`
=======
## Repository instructions

- `sample_weight` is always treated as frequency weights.
- All metrics must accept `sample_weight`.
- Never use in-sample residuals for variance modeling in flexible models unless explicitly requested.
- Run `pytest` before marking tasks complete.
>>>>>>> theirs
