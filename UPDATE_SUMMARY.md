# 2026-08-28 update — guarded LTH + per-channel fixed-point INT8 + end-to-end tracking evidence

The Python project is now aligned around its intended boundary: **simulation/reproduction/training/evaluation and compact weight/runtime export**.

## Main changes

- Official-data acquisition/conversion remains automatic and now records SHA-256 provenance plus a source-vs-geometry ToF audit.
- U-FusePF has an explicit learned outlier/confidence head supervised by simulated corruption labels; full U-FusePF passes this probability to the adaptive PF.
- `no_uncertainty` disables both learned scale and learned outlier confidence.
- Research checkpoint selection is MAE-first so uncertainty NLL cannot mask worse ToF accuracy.
- Structured LTH ranking is dependency-aware and BN-weighted; Conv channels and FC hidden neurons are physically pruned and survivors are rewound to initialization.
- LTH candidate acceptance now checks clean/robust FP32 quality **and held-out PF tracking**.
- Every candidate is quantized before final acceptance; the INT8 model must also pass clean/robust/uncertainty/tracking degradation limits.
- Weights use per-output-channel INT8 quantization and fixed-point Q31 multiplier + right-shift requantization.
- The version-2 raw blob includes weights, biases and requantization metadata.
- End-to-end test-split evaluation writes FP32-LTH, INT8-LTH and same-architecture random-control PF results.
- `scripts/run_esp32_study.py` repeats the entire pipeline across cases/seeds and aggregates final evidence; it supports output directories both inside and outside the repository.

## Verified mechanism-only smoke

- structured rewound LTH selected;
- Conv `[6,8,8]`, hidden `8`;
- 827 parameters;
- 1,069-byte v2 raw model blob;
- 4,461 bytes core static deployment data;
- FP32 guard: pass;
- INT8 guard: pass;
- 64 automated tests pass.

The smoke dataset/training duration are deliberately synthetic/short. Use `configs/esp32s3_official.yaml` or `RUN_ESP32_STUDY.sh` for final experimental claims.
