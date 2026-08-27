# Final implementation report — Python training to ESP32 export

The original Python research repository structure is preserved. The embedded deployment path is added alongside the existing paper-reproduction and U-FusePF code rather than replacing it.

## Final pipeline

```text
MAT/UWB data
 -> original preprocessing / optional robustness augmentation
 -> optional full CIR+variance+U-Fuse teacher
 -> compact ESP32 supernet
 -> structured Lottery-Ticket channel selection
 -> rewind selected initialization
 -> compact ticket retraining
 -> same-size fresh compact control
 -> validation-based winner selection
 -> BatchNorm folding
 -> calibrated symmetric INT8 export
 -> raw .bin + generated C header + manifests + golden vectors
 -> optional ONNX opset 18 / ESP-DL export
```

The ESP32 deployment student is deliberately restricted to Conv1D/ReLU, global-average pooling and two fully connected layers. The final three raw outputs represent ToF, uncertainty and link-outlier quality; C/C++ can decode them using generated LUTs rather than runtime sigmoid/softplus calls.

## Embedded artifacts

The exporter generates:

- `ufuse_weights_int8.bin`: smallest raw weight/bias blob for a fixed C/C++ graph;
- `ufuse_weights_int8.h`: directly embeddable header with weights, biases, scales, multipliers and decode LUTs;
- `ufuse_weights_manifest.json`: tensor shapes, offsets, quantization metadata and SHA-256;
- `golden_vectors.npz`: Python FP32 and integer-reference outputs for firmware parity testing;
- `uwb_background_u8.bin`: pre-normalized static CIR/variance backgrounds;
- `uwb_runtime_constants.h`: geometry, link pairs, delay constants and embedded backgrounds;
- optional ONNX and `.espdl` files.

## Robustness and PF changes

The original default scenarios are unchanged. An optional extended configuration adds `burst_nlos`, `burst_dropout` and `mixed` corruption. The deployment PF now reuses particle-to-anchor distances, precomputes Student-t constants, can run in float32, and can consume the learned per-link outlier probability.

## Verification

- `python -m compileall -q src scripts tests`: PASS
- `pytest -q`: **58/58 PASS**
- `train_esp32_pipeline.py` with `configs/esp32s3_smoke.yaml`: PASS end to end
- short smoke export: 951 parameters, 1,044-byte raw INT8 weights, ~0.051 ns decoded mean-delay difference between folded FP32 and the custom integer reference on golden samples.

The repository also retains `artifacts/esp32_demo/`, a verified reference export from the fuller default compact architecture.
