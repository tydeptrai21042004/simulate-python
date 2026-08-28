# Changelog

## 2026-08-28 — End-to-end guarded LTH/INT8 tracking export

- Added `src/uwb_tracking/esp32/evaluation.py` to evaluate folded FP32 and integer-reference INT8 students through the full uncertainty-aware Particle Filter.
- Candidate selection now checks clean/robust ToF, NLL, outlier BCE and held-out tracking before and after quantization; strict official runs stop if either FP32 or INT8 quality guards fail.
- Upgraded raw quantization to per-output-channel symmetric INT8 weights with Q31 multiplier + right-shift requantization; version-2 binary blobs include weights, biases and fixed-point requantization metadata.
- Added an explicit learned outlier/confidence head to U-FusePF and propagated it to the adaptive PF; the `no_uncertainty` ablation disables both scale and outlier confidence.
- Made research checkpoint selection MAE-first with uncertainty NLL as a near-tie breaker.
- Reduced PF transient memory by streaming link residual/likelihood accumulation instead of materializing `[particles, links]` expected/residual matrices.
- Added official-data SHA-256 provenance and source-vs-geometry ToF auditing.
- Added `scripts/run_esp32_study.py`, `RUN_ESP32_STUDY.sh` and `.bat` for full case/seed LTH -> INT8 -> PF studies, including support for external absolute output directories.
- Added `configs/full_official.yaml` for the three-case/five-seed research benchmark with automatic official-data setup.
- Expanded verification to **64 passing tests**. The regenerated synthetic smoke selects an 827-parameter structured LTH ticket, emits a 1,069-byte v2 raw model blob and reports 4,461 bytes of core static deployment data; these are mechanism-only figures.

## 2026-08-27 — Official auto-data + accuracy-guarded structured LTH export

- Added `scripts/fetch_original_data.py` and `uwb_tracking.official_data` to automatically download the official GitHub repository and the author-linked `Dyn_CIR_VAR.mat`, then build `data/uwb_original_standard.mat`.
- Corrected official MATLAB conversion: 4x3 XYZ anchors are reduced to XY exactly as the official PF does; `Dyn_var_CIRxx` / `Bg_var_CIRxx` are recognized; complex CIR uses magnitude instead of a lossy float cast.
- Reworked ESP32 structured Lottery Ticket selection to prune both Conv channels and FC hidden neurons with dependency-aware hierarchical scoring, then rewind survivors to their original initialization.
- Added progressive smallest-first LTH candidate search with explicit relative + absolute ToF-MAE guards plus NLL/outlier-BCE quality guards. `best_student.pt` is now always a structured rewound LTH ticket; the random compact model is a control only and can never replace the deployment checkpoint.
- Added representative clean/corrupted calibration sampling before INT8 export and changed checkpoint selection to MAE-first to prevent negative NLL from hiding worse ToF accuracy.
- Added `configs/esp32s3_official.yaml`, `RUN_ESP32_OFFICIAL.sh`, and `RUN_ESP32_OFFICIAL.bat` for one-command official-data training/export.
- Expanded the suite to **61 passing tests**, including official-schema conversion of complex CIR/XYZ anchors and hidden-neuron LTH pruning.


## 1.0.0

- Replaced MATLAB pipeline with installable Python/PyTorch package.
- Added paper CIR-CNN and variance-CNN reproduction.
- Added U-FusePF learned uncertainty fusion.
- Replaced heuristic confidence with heteroscedastic Student-t scale.
- Added uncertainty-aware robust Particle Filter.
- Removed simulator bias caused by always-independent CIR/variance false peaks.
- Added time-level validation separation.
- Added 3 cases, multi-seed evaluation, five scenarios, ablation and robustness sweep.
- Added confidence calibration and corruption-detection metrics.
- Added mean/std/95% CI and paired Wilcoxon reporting.
- Added original MATLAB data converter.
- Added automated tests and example quick-run outputs.

## 2026-08-27 — ESP32 finalization and expanded validation

- Preserved the original Python research package/layout and added the ESP32 deployment path alongside it.
- Added `src/uwb_tracking/esp32/` for the tiny early-fusion student, structured Lottery-Ticket channel selection/rewinding, training, BN folding, INT8 calibration/export, and preprocessing/geometry export.
- Added raw INT8 `.bin`, C-header, golden-vector, manifest, ONNX and optional ESP-DL export paths.
- Added SHA-256 integrity metadata for raw weight blobs.
- Added stricter validation for ESP32 model shapes, calibration data, streaming input, training arrays and target lengths.
- Optimized the adaptive Student-t PF for embedded use by reusing particle-to-anchor distances, precomputing Student-t constants, removing SciPy from the deployment PF path, and allowing float32 operation and per-link outlier probability.
- Added extended robustness scenarios: `burst_nlos`, `burst_dropout`, and `mixed` without changing the original default experiment scenarios.
- Added `configs/robustness_extended.yaml` and `configs/esp32s3_smoke.yaml`.
- Expanded the automated suite to **58 passing tests**, including data edge cases, all corruption scenarios, PF numerical stability/determinism, streaming preprocessing, structured-ticket failure cases, checkpoint round-trip, deterministic binary export, quantizer saturation, header/LUT content and a short train-to-export integration path.
- Verified `scripts/train_esp32_pipeline.py` end to end with `configs/esp32s3_smoke.yaml`; the smoke pipeline exported a 951-parameter compact model to a 1,044-byte raw INT8 weight blob with approximately 0.051 ns float-vs-INT8 mean-delay difference on its golden vectors. This is a pipeline validation result, not a final accuracy claim.
