# Final implementation report — Python simulation/training to LTH ESP32 weight export

The original Python research repository structure is preserved. Python remains the simulation, training, validation, compression and export environment; the generated binary/header files are the contract for later C/C++ firmware.

## Final pipeline

```text
MAT/UWB data
 -> if needed: auto-download official GitHub repo + Dyn_CIR_VAR.mat
 -> paper-faithful official-data conversion
 -> preprocessing / robustness augmentation
 -> optional CIR+variance+U-Fuse teacher
 -> wider ESP32 discovery supernet
 -> dependency-aware structured Lottery-Ticket ranking
 -> rewind retained Conv channels + FC neurons to initialization
 -> progressive smallest-first compact-ticket retraining
 -> explicit validation ToF-MAE accuracy guard
 -> same-architecture random compact control (evidence only)
 -> selected LTH ticket copied to best_student.pt
 -> representative clean/corrupted calibration
 -> BatchNorm folding
 -> calibrated symmetric INT8 export
 -> raw .bin + generated C header + manifests + golden vectors
 -> optional ONNX opset 18 / ESP-DL export
```

## Lottery Ticket policy

The exporter no longer chooses between LTH and a random compact model. Deployment is **always** a structured rewound Lottery Ticket candidate. The random model remains a required scientific control so the report can still show whether the LTH initialization itself helped.

The search physically removes Conv channels and FC hidden neurons. Candidate architectures are tried from the smallest parameter count upward. A ticket is exportable only when its validation ToF MAE stays inside both the configured relative and absolute degradation limits relative to the uncompressed supernet. With `require_accuracy_guard: true`, failure to preserve accuracy stops deployment export instead of silently exporting a weak model.

## Official data automation and conversion

`RUN_ESP32_OFFICIAL.sh` / `configs/esp32s3_official.yaml` can automatically obtain the public source files. The converter now matches the official MATLAB schema:

- `AnchorPos.mat` is 4x3 XYZ; the official particle filter uses XY, so the standardized dataset stores the first two coordinates.
- Variance arrays use the actual `Dyn_var_CIRxx` / `Bg_var_CIRxx` names.
- CIR arrays are complex; conversion uses `abs(CIR)`, matching the official MATLAB `mat2gray(abs(...))` path rather than discarding the imaginary component.
- The six-link geometry is standardized and validated before training.

## Embedded artifacts

The exporter generates:

- `ufuse_weights_int8.bin`: compact raw LTH weight/bias blob;
- `ufuse_weights_int8.h`: embeddable header with weights, biases, scales, multipliers and decode LUTs;
- `ufuse_weights_manifest.json`: tensor shapes, offsets, quantization metadata and SHA-256;
- `golden_vectors.npz`: FP32 and integer-reference outputs for later firmware parity testing;
- `uwb_background_u8.bin`: pre-normalized static CIR/variance backgrounds;
- `uwb_runtime_constants.h`: geometry, link pairs, delay constants and background constants;
- optional ONNX and `.espdl` files.

## Verification

- `python -m compileall -q src scripts tests`: **PASS**
- `pytest -q`: **61/61 PASS**
- `train_esp32_pipeline.py` with `configs/esp32s3_smoke.yaml`: **PASS end to end**
- regenerated smoke export: structured rewound LTH ticket, **827 parameters**, **904-byte** raw INT8 weight/bias blob, about **0.0925 ns** decoded mean-delay difference between folded FP32 and the custom INT8 reference on golden samples.

The smoke configuration is intentionally short and uses a loose accuracy guard; it verifies the mechanism only. Final model-size/accuracy claims should come from `configs/esp32s3_official.yaml` on the original experimental data.
