# Changelog

## 0.2.0 - 2026-02-07

### Breaking Changes
- `TwoStageHeteroscedasticLightGBM`: replaced `calibrate: bool` parameter with
  `calibration_method: str` accepting `"none"`, `"holdout"`, `"oof"`, or `"train"`.
  The old `calibrate` parameter still works but emits a `FutureWarning` and will
  be removed in v0.3.0.
- Default changed from `calibrate=True, calibration_fraction=0.0` (which did train
  calibration) to `calibration_method="oof"` (OOF calibration, no data loss).
- Default `n_iterations` changed from 1 to 2.
- Default `oof_residuals_reuse` changed from False to True.

### Added
- `calibration_method="oof"`: calibrates variance scale using out-of-fold mean
  predictions. No held-out data required, no in-sample bias. Recommended default.
- `oof_mean_predictions()` in `hetero_nll.oof`: computes OOF mean predictions
  (reuses same OOF splitter infrastructure).
- `calibration_method="train"` emits a `UserWarning` about systematic variance
  shrinkage with flexible models.
- New tests: `test_calibration_method_oof`, `test_calibration_method_holdout`,
  `test_calibration_method_train_warns`, `test_calibration_method_none`,
  `test_calibration_method_invalid_raises`, `test_deprecated_calibrate_param_warns`,
  `test_two_iter_oof_cal_improves_nll`, `test_explicit_cal_data_overrides`.

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
