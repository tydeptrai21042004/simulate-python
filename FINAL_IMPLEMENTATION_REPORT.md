# Final implementation report — UWB simulation to guarded structured-LTH INT8 export

## Scope

Python is the research and deployment-preparation environment: official-data reproduction, simulation, model training, confidence/uncertainty evaluation, Particle Filter tracking, Lottery-Ticket compression, integer-reference validation, and export of compact weight/runtime files. A target firmware implementation is deliberately outside this repository's required scope.

## Final pipeline

```text
official/synthetic UWB data
 -> automatic source acquisition when requested
 -> MATLAB-faithful conversion + provenance/geometry audit
 -> CIR/variance preprocessing and corruption simulation
 -> paper CIR/variance baselines + proposed U-FusePF research model
 -> explicit learned outlier/confidence output + predictive scale
 -> wider deployment supernet
 -> BN-weighted, dependency-aware structured LTH ranking
 -> rewind retained Conv channels + FC neurons to initialization
 -> smallest-first candidate retraining
 -> FP32 clean + robust + PF-tracking guard
 -> same-architecture random control (evidence only)
 -> representative calibration
 -> per-output-channel INT8 + Q31/right-shift requantization
 -> INT8 clean + robust + PF-tracking guard
 -> selected LTH ticket only
 -> raw binary/header/manifests/runtime constants/golden vectors
 -> held-out end-to-end FP32-vs-INT8 deployment evaluation
 -> optional multi-case/multi-seed study aggregation
```

## Improvements relative to the earlier implementation

### Official-data reproducibility

The converter now handles the official schema instead of assuming synthetic conventions: XYZ anchors are mapped to XY for 2D tracking, the real variance variable names are recognized, and complex CIR is converted by magnitude. Conversion also records a source-vs-geometry ToF audit and a SHA-256 provenance manifest for the input `.mat` files and standardized output.

### Proposed confidence-aware method

The research U-Fuse model now exposes an explicit learned outlier/confidence logit in addition to fused ToF and predictive scale. Corruption labels from the simulator supervise this head. Full U-FusePF passes the learned outlier probability into the adaptive Student-t PF; the `no_uncertainty` ablation disables both learned scale confidence and outlier confidence so the ablation has a clean interpretation.

Research-model checkpoint selection is MAE-first, with NLL used only as a near-tie breaker, so uncertainty calibration cannot hide a materially worse ToF estimate.

### Structured Lottery Ticket export

Deployment no longer chooses a random compact network when it happens to validate better. `best_student.pt` is always a **structured rewound LTH ticket**. Channel importance combines trained weight magnitude with BatchNorm scale and respects inter-layer dependencies. FC hidden neurons are physically pruned as well. Survivors are copied from the initial supernet state and retrained.

Candidates are tried smallest-first, but size alone cannot win. Acceptance checks clean/robust ToF quality, uncertainty NLL, outlier BCE and held-out PF tracking in FP32, then repeats clean/robust and tracking guards after INT8 quantization. Strict official runs abort export when no candidate satisfies the configured quality budget.

### Quantization/export

Weights use per-output-channel symmetric INT8 quantization. Requantization metadata is stored as integer Q31 multipliers plus right shifts rather than one floating multiplier per layer. The version-2 raw blob contains weights, biases and the fixed-point requantization metadata; its manifest records offsets, shapes, scales and SHA-256.

The exporter also writes output-decode LUTs, background/geometry runtime constants and golden vectors. `core_static_deployment_bytes` reports the model blob + decode LUTs + background blob separately from activation/PF working memory.

### End-to-end evidence

The selected student is no longer judged only by tensor/ToF agreement. Both the folded FP32 graph and Python INT8 reference are decoded and run through the Particle Filter on held-out sequences. The official configuration evaluates LoS, NLoS-1, NLoS-2, outlier and dropout. A separate study launcher repeats the entire LTH→INT8→PF chain across trajectory cases and random seeds and aggregates model size, guards and tracking performance.

## Current verification

- `python -m compileall -q src scripts tests`: **PASS**
- `PYTHONPATH=src pytest -q`: **64 passed**
- `configs/esp32s3_smoke.yaml`: complete train → structured LTH → FP32 guard → per-channel INT8 → INT8 guard → PF evaluation → export: **PASS**
- multi-case study launcher dry-run with a repository-relative output and an external absolute output: **PASS**

The regenerated synthetic smoke selects a structured rewound `[6,8,8]`, hidden-8 ticket with **827 parameters**. Its v2 model blob is **1,069 bytes** and its reported core static deployment data is **4,461 bytes**. These are integration figures only; the short synthetic smoke must not be presented as official DWM1000 tracking accuracy.

## Recommended final experiment

Use `./RUN_ESP32_STUDY.sh` for the lightweight deployment study and `uwb-track full --config configs/full_official.yaml` for the full research-method comparison. Final claims should report results over all requested cases/seeds, show LTH-vs-same-architecture-random control, report FP32-vs-INT8 tracking deltas, and retain the strict official quality guards.
