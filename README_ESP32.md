# UWB U-FusePF — Simulation, Structured-LTH Training, and INT8 Weight Export

This repository's deployment scope is intentionally the **Python side**: reproduce/simulate the UWB tracking pipeline, train and validate the estimator, compress it with a structured Lottery Ticket Hypothesis (LTH) search, validate the quantized model through the Particle Filter, and export compact model/runtime files. Firmware is a later consumer of these artifacts and is not required for the Python experiment to be complete.

## Deployment student

The student processes one UWB link at a time with six normalized channels:

1. dynamic CIR;
2. background CIR;
3. CIR difference;
4. dynamic variance;
5. background variance;
6. variance difference.

The physical graph is selected automatically. A common candidate is:

```text
[6 x input_length]
 -> Conv1D -> ReLU
 -> Conv1D -> ReLU
 -> Conv1D -> ReLU
 -> GlobalAveragePool
 -> FC -> ReLU
 -> FC(3)
```

The three raw outputs decode to excess-delay mean, Student-t scale, and outlier probability. Decode lookup tables are exported so the later runtime does not need sigmoid/softplus for these outputs.

## Structured Lottery Ticket policy

Training starts from a wider discovery supernet. The compact search:

```text
train supernet
 -> rank Conv1 using BN-weighted channel saliency
 -> rank Conv2 only through retained Conv1 inputs
 -> rank Conv3 only through retained Conv2 inputs
 -> rank FC hidden neurons from incoming + outgoing importance
 -> physically build candidate graph
 -> rewind retained parameters to the original initialization
 -> retrain candidate
 -> validate FP32 clean + corrupted data + held-out PF tracking
 -> quantize candidate
 -> validate INT8 clean + corrupted data + held-out PF tracking
 -> accept the smallest candidate satisfying all configured guards
```

`best_student.pt` is therefore always a **structured rewound LTH ticket**. A fresh random network of the same selected architecture is still trained as a scientific control, but normal export rejects it.

The official configuration protects quality with limits on clean/robust ToF MAE, uncertainty NLL, outlier BCE, FP32 tracking RMSE, quantization-induced ToF degradation, and quantization-induced tracking degradation. With `require_accuracy_guard: true`, the pipeline stops rather than exporting a lighter but unacceptable model.

## Official data: automatic acquisition and faithful conversion

Run:

```bash
python scripts/fetch_original_data.py
```

or allow the official configurations to fetch data automatically. The converter handles the original repository schema correctly:

- `AnchorPos.mat` XYZ anchors are reduced to XY for the 2D tracking model;
- the actual `Dyn_var_CIRxx` / `Bg_var_CIRxx` variables are recognized;
- complex CIR uses magnitude `abs(CIR)` rather than discarding the imaginary component;
- geometry and source ToF are audited;
- a SHA-256 provenance manifest records source files and the standardized output.

The standardized dataset is written to `data/uwb_original_standard.mat` by default.

## One-run official LTH -> INT8 export

Linux/macOS:

```bash
pip install -r requirements-esp32-train.txt
pip install -e .
./RUN_ESP32_OFFICIAL.sh
```

Windows:

```bat
pip install -r requirements-esp32-train.txt
pip install -e .
RUN_ESP32_OFFICIAL.bat
```

Equivalent direct command:

```bash
python scripts/train_esp32_pipeline.py --config configs/esp32s3_official.yaml --auto-data
```

The full path is:

```text
official/synthetic MAT data
 -> preprocessing + corruption augmentation
 -> optional teacher targets
 -> discovery supernet
 -> progressive structured rewound LTH search
 -> FP32 clean/robust/tracking guards
 -> same-architecture random control
 -> representative clean/corrupted calibration
 -> per-output-channel INT8 quantization
 -> Q31 multiplier + right-shift requantization
 -> INT8 clean/robust/tracking guards
 -> selected LTH checkpoint
 -> binary/header/runtime constants/golden vectors
 -> end-to-end test-split FP32 vs INT8 PF evaluation
```

## Multi-case / multi-seed deployment study

A single case/seed is useful for development, but the final lightweight claim should be aggregated across cases and seeds:

```bash
./RUN_ESP32_STUDY.sh
```

or:

```bash
python scripts/run_esp32_study.py \
  --config configs/esp32s3_official.yaml \
  --cases 1 2 3 \
  --seeds 11 22 33
```

The launcher writes per-run resolved configurations/logs plus `study_runs.csv` and `study_summary.json`, including parameter/blob size, guard status, FP32-vs-INT8 changes, tracking metrics, and the LTH-vs-random-control comparison. External absolute output directories are supported as well.

## Export format

The raw export is versioned as `uwb-esp32-int8-v2-per-channel-fixedpoint`. Each layer contributes:

- INT8 weights with **per-output-channel symmetric scales**;
- INT32 biases;
- INT32 Q31 requantization multipliers;
- UINT8 right shifts.

Generated files include:

```text
best_student.pt
export/ufuse_weights_int8.bin
export/ufuse_weights_int8.h
export/ufuse_weights_manifest.json
export/golden_vectors.npz
export/uwb_background_u8.bin
export/uwb_runtime_constants.h
export/uwb_runtime_manifest.json
export/export_report.json
pipeline_report.json
deployment_tracking_results.json
deployment_tracking_results.csv
```

`ufuse_weights_int8.bin` is the compact raw model-data blob. `core_static_deployment_bytes` in `export_report.json` additionally counts the output-decode LUTs and background runtime blob; activation/PF working memory is not included in that static-data figure.

## End-to-end deployment validation

Candidate selection no longer stops at neural-network validation. The exact BN-folded FP32 student and the Python integer-reference INT8 student are both decoded into ToF mean/scale/outlier probability and passed through the uncertainty-aware Particle Filter. This provides directly comparable tracking RMSE/P90 before the model is accepted.

The official end-to-end evaluation uses the test split and the configured LoS, NLoS-1, NLoS-2, outlier, and dropout scenarios. `deployment_tracking_results.csv` therefore connects the exported model to the actual tracking objective rather than reporting only quantization tensor error.

## Research benchmark on the official data

For the full research comparison rather than the deployment student:

```bash
uwb-track full --config configs/full_official.yaml
```

This uses all three trajectory cases, five seeds, the five primary robustness scenarios, and a controlled 200-particle comparison.

For repository-parity reproduction:

```bash
uwb-track reproduce-paper --config configs/paper_reproduction_original.yaml
```

Both official configurations can auto-fetch/convert the source data if the standardized file is absent.

## Optional ONNX / ESP-DL outputs

The raw INT8 exporter is independent of ONNX/ESP-DL. Optional formats remain available:

```bash
python scripts/export_esp32.py \
  --checkpoint results/esp32s3_official/checkpoints/best_student.pt \
  --target esp32s3 \
  --onnx
```

and, after installing `requirements-espdl-export.txt`, use `--espdl` for an ESP-DL artifact.

## Golden-vector verification

`golden_vectors.npz` contains normalized inputs plus outputs from the BN-folded FP32 graph and the integer-reference path. It is the parity contract for any later implementation and detects tensor-order, padding, stride, scaling, rounding, or decode mistakes.

## Current mechanism-only smoke verification

The bundled smoke output is **synthetic and short**, so it is not a tracking-accuracy claim. It verifies the complete mechanism:

- selected deployment role: structured rewound LTH ticket;
- selected smoke graph: Conv `[6, 8, 8]`, hidden `8`;
- parameters: **827**;
- raw v2 model blob: **1,069 bytes** (weights + biases + fixed-point requantization metadata);
- core static deployment data: **4,461 bytes** including decode LUTs and background blob;
- both FP32 and INT8 quality guards: pass in the smoke configuration;
- end-to-end FP32/INT8 PF evaluation files are produced.

Final model-size and accuracy claims must come from the official-data configuration/study, not these smoke metrics.

## Verification

```bash
python -m compileall -q src scripts tests
PYTHONPATH=src pytest -q
```

Current result: **64 tests passed**.
