# Lightweight deployment: structured Lottery Ticket to fixed-point INT8 export

## Purpose

The lightweight path is designed for **simulation/training/export**. It produces a physically compact neural network and a deterministic integer-reference contract; it does not require this repository to contain target firmware.

## Current deployment model

`src/uwb_tracking/esp32/model.py` defines the early-fusion student. Input channels are dynamic/background/difference for CIR and variance. The physical architecture is discovered from a wider supernet rather than fixed to the older research-lite network in `models/lite.py`.

The official candidate ladder is currently:

| Conv widths | FC hidden | Approx. trainable parameters |
|---|---:|---:|
| 6 / 8 / 8 | 10 | 851 |
| 8 / 10 / 10 | 12 | 1,263 |
| 8 / 12 / 12 | 16 | 1,571 |
| 10 / 16 / 20 | 18 | 2,707 |
| 12 / 18 / 24 | 20 | 3,551 |

The exact parameter count can differ when another hidden width is configured; the pipeline records the actual selected graph.

## Structured-LTH search

The procedure is intentionally **structured** so parameter reduction also reduces the physical graph:

```text
train wider supernet
 -> score Conv1 output channels with |W| and |BN gamma|
 -> retain Conv1
 -> score Conv2 through retained Conv1 inputs
 -> retain Conv2
 -> score Conv3 through retained Conv2 inputs
 -> retain Conv3
 -> rank FC hidden neurons from incoming + outgoing importance
 -> build compact graph
 -> rewind retained weights/BN/FC parameters to initialization
 -> retrain
```

The random compact network of the same architecture is a control only. Normal export checks the checkpoint metadata and refuses a non-LTH checkpoint unless an explicit debug override is used.

## Quality-preserving acceptance

The smallest graph is not automatically accepted. The official guard checks:

- clean FP32 ToF MAE relative/absolute loss vs the discovery supernet;
- robust FP32 ToF MAE on deterministic corruption;
- robust NLL and outlier BCE;
- held-out contiguous-sequence FP32 tracking RMSE;
- clean and robust INT8 ToF degradation vs candidate FP32;
- INT8 NLL/outlier-BCE degradation;
- INT8 tracking RMSE degradation vs candidate FP32.

This makes the optimization objective effectively:

```text
minimize physical model size
subject to localization + uncertainty + tracking quality constraints
before and after INT8 quantization.
```

## Integer export

The custom exporter folds BatchNorm and quantizes weights per output channel. For each layer the v2 raw blob stores:

```text
INT8 weights
INT32 biases
INT32 Q31 requantization multipliers
UINT8 right shifts
```

The Python integer reference performs INT8×INT8 accumulation in integer accumulators and fixed-point requantization. Output mean/scale/outlier nonlinearities are represented by LUTs in the generated header/runtime constants.

## What model-size number to report

Do not mix these quantities:

- **parameters** — trainable scalar count of the selected network;
- **raw model blob** — weights + biases + fixed-point requantization metadata;
- **core static deployment data** — raw model blob + decode LUTs + static background profiles;
- **working RAM** — activations and Particle Filter state/temporaries, which is separate.

The synthetic smoke currently demonstrates 827 parameters, a 1,069-byte v2 model blob, and 4,461 bytes of core static data. Those are mechanism measurements, not official accuracy results.

## Particle Filter efficiency

The PF keeps the same bistatic geometry and Student-t likelihood concept but avoids allocating full `[particles, links]` expected/residual matrices. Particle-to-anchor distances are reused and links are accumulated one at a time. This reduces transient memory without changing the asymptotic `O(T × L × Np)` order.

For fair scientific comparison, the official research benchmark uses 200 particles. The LTH selection guard can use fewer particles for development speed, while the final `deployment_evaluation` returns to the configured deployment count.

## End-to-end validation

`src/uwb_tracking/esp32/evaluation.py` evaluates both the folded FP32 graph and integer-reference graph through the same PF. Therefore quantization is accepted based on the application objective rather than only layer-output similarity.

Run a single official case/seed with:

```bash
./RUN_ESP32_OFFICIAL.sh
```

Run the final multi-case/multi-seed deployment study with:

```bash
./RUN_ESP32_STUDY.sh
```

The latter aggregates sizes, guards, ToF metrics, tracking metrics, FP32-vs-INT8 deltas and LTH-vs-random-control outcomes.

## Legacy lightweight model

`models/lite.py` and the older deployment-lottery scripts remain available for historical/research comparison. They are **not** the canonical final export path. The canonical export is `scripts/train_esp32_pipeline.py` plus `src/uwb_tracking/esp32/`.
