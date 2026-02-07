# Changelog

## 0.2.0 - 2026-02-07

### Breaking Changes
- **Package renamed from `hetero_nll` to `nllfit`.** All imports must change:
  `from hetero_nll import ...` → `from nllfit import ...`.
- `TwoStageHeteroscedasticLightGBM`: replaced `calibrate: bool` parameter with
  `calibration_method: str` accepting `"none"`, `"holdout"`, `"oof"`, or `"train"`.
  The old `calibrate` parameter still works but emits a `FutureWarning` and will
  be removed in v0.3.0.
- `TwoStageHeteroscedasticGLUM`: replaced `calibrate: bool` parameter with
  `calibration_method: str` accepting `"none"`, `"holdout"`, or `"train"`.
  `"oof"` is not supported for GLM and falls back to `"none"` with a warning.
  The old `calibrate` parameter still works but emits a `FutureWarning`.
- Default LightGBM `calibration_method` changed from train calibration to `"oof"`.
- Default LightGBM `n_iterations` changed from 1 to 2.
- Default LightGBM `oof_residuals_reuse` changed from False to True.

### Added
- `calibration_method="oof"` for LightGBM: calibrates variance scale using
  out-of-fold mean predictions. No held-out data required, no in-sample bias.
- `oof_mean_predictions()` in `nllfit.oof`: computes OOF mean predictions
  (reuses same OOF splitter infrastructure).
- `calibration_method="train"` on LightGBM emits a `UserWarning` about
  systematic variance shrinkage with flexible models.
- Comprehensive tests for all calibration methods on both estimators.

### Fixed
- Train calibration (calibrating on in-sample residuals) was producing scale << 1
  with flexible mean models, causing severe variance shrinkage and undercoverage.
  This is now deprecated in favor of OOF calibration.

## 0.1.0 - 2026-02-07
- Initial release: two-stage heteroscedastic regression for GLUM and LightGBM.
- OOF residuals for LightGBM variance stage.
- Scalar variance calibration (closed form).
- Gaussian NLL metric + predictive quantiles/intervals.
- Group/time-aware calibration splitting (timestamp-based).
