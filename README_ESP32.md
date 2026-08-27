# UWB U-FusePF — Python Training and ESP32 Export Pipeline

This directory contains the complete **PC/Python side** of an ESP32 deployment workflow.
Python is used for training, Lottery-Ticket model selection, optional teacher distillation,
BatchNorm folding, calibration and export. The ESP32 only needs the exported model data and
C/C++ inference/PF code.

## Final embedded network

The ESP32 student consumes the same six normalized features already used by the research model:

1. CIR dynamic
2. CIR background
3. CIR difference
4. variance dynamic
5. variance background
6. variance difference

The deployment architecture is selected automatically by a progressive structured Lottery-Ticket search.
Candidate graphs range from a very small `6/8/8 + FC10` model to a conservative
`12/18/24 + FC20` model. The search always tries the smallest physical graph first and stops
when validation ToF MAE stays inside the configured accuracy budget.

A common middle candidate is:

```text
[6 x 176]
  -> Conv1D 6->8, k=7, stride=2 -> ReLU
  -> Conv1D 8->12, k=5, stride=2 -> ReLU
  -> Conv1D 12->12, k=3, stride=2 -> ReLU
  -> GlobalAveragePool
  -> FC 12->16 -> ReLU
  -> FC 16->3
```

The three raw outputs are decoded as ToF mean, Student-t scale and outlier probability. For the
custom C/C++ path, the exporter generates lookup tables so the MCU does not need sigmoid or
softplus at runtime. The `8/12/12 + FC16` candidate has **1,571 parameters** and historically
exports to about **1.7 KB** of raw INT8 weights/biases; more aggressive candidates can be smaller.

## Why the Lottery-Ticket step is structured

The training supernet is wider (`12,18,24` with a wider FC hidden layer). After training, the code
ranks channels **hierarchically**: conv2 is scored only through retained conv1 inputs, conv3 only
through retained conv2 inputs, and FC hidden neurons are ranked from both their incoming and
outgoing magnitudes. The selected channels/neurons are then copied from the **initial** supernet
state and retrained. This preserves the rewind principle of the Lottery Ticket Hypothesis while
physically deleting channels and neurons instead of exporting sparse zero masks.

Export is now **LTH-only**. A fresh random network with the exact selected architecture is still
trained as a scientific control, but it can never replace `best_student.pt`. To protect accuracy,
the search exports only a structured rewound ticket that meets the configured MAE/NLL/outlier-quality guard
(`max_relative_mae_loss` and `max_absolute_mae_loss_ns`). If no ticket satisfies a strict guard,
no misleading "best" deployment export is produced.

## Optional rich teacher

`results/quick/checkpoints/case1_seed11` is used by the default config as a smoke-test teacher.
The teacher reproduces the current research deployment:

```text
paper CIR CNN + paper variance CNN + U-Fuse reliability network
```

The ESP32 student is trained against ground truth and may additionally distill the teacher's fused
ToF and uncertainty. The teacher is **never exported**.

For final experiments, train a converged teacher first:

```bash
python scripts/train_teacher.py \
  --config configs/full.yaml \
  --case 1 \
  --seed 11 \
  --output results/teacher/case1_seed11
```

Then point `teacher_checkpoint_dir` in `configs/esp32s3.yaml` to that directory.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-esp32-train.txt
pip install -e .
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-esp32-train.txt
pip install -e .
```

## One-command training + raw INT8 export

Synthetic/demo path:

```bash
python scripts/train_esp32_pipeline.py --config configs/esp32s3.yaml
```

Official-data path (automatic download + conversion on first run):

```bash
./RUN_ESP32_OFFICIAL.sh
# or
python scripts/train_esp32_pipeline.py --config configs/esp32s3_official.yaml --auto-data
```

The official path downloads `CLongLi/UWB-Radar-Pedestrian-Tracking` from GitHub and obtains
`Dyn_CIR_VAR.mat` from the author-provided Google Drive link referenced by that repository. The
converter handles the official `4x3` anchors, `Dyn_var_CIRxx` / `Bg_var_CIRxx` variables, and uses
`abs(complex CIR)` as the MATLAB scripts do.

The training/export sequence is:

```text
load or auto-fetch official data
 -> build clean + corrupted training set
 -> optional rich-teacher targets
 -> train wider discovery supernet
 -> try smallest structured rewound LTH ticket
 -> enlarge only if the validation-MAE guard is not met
 -> train same-architecture random control for evidence only
 -> copy the selected LTH checkpoint to best_student.pt
 -> balanced clean/corrupted INT8 calibration
 -> fold BatchNorm
 -> export C header + raw binary + golden vectors
```

Main outputs:

```text
results/esp32s3/
├── pipeline_report.json
├── checkpoints/
│   ├── supernet.pt
│   ├── lth_ticket_00_c..._h....pt   # one or more progressive candidates
│   ├── random_compact_control.pt   # evidence only, never deployed
│   └── best_student.pt             # always the selected structured LTH ticket
└── export/
    ├── ufuse_weights_int8.h
    ├── ufuse_weights_int8.bin
    ├── ufuse_weights_manifest.json
    ├── golden_vectors.npz
    └── export_report.json
```

`ufuse_weights_int8.bin` is the smallest raw model-data artifact. It assumes the fixed network
architecture is compiled into the C/C++ firmware. `ufuse_weights_int8.h` is larger as a source
file but is easiest to embed directly into firmware.

A verified reference export is also retained under `artifacts/esp32_demo/` so the generated binary,
header, runtime constants, manifests and golden vectors can be inspected without retraining.

## ONNX export

Install ONNX support:

```bash
pip install onnx onnxscript
```

Then:

```bash
python scripts/export_esp32.py \
  --checkpoint results/esp32s3/checkpoints/best_student.pt \
  --target esp32s3 \
  --onnx
```

The exporter uses ONNX opset 18.

## ESP-DL `.espdl` export

Install the Espressif quantization stack in a dedicated environment:

```bash
pip install -r requirements-espdl-export.txt
```

Then:

```bash
python scripts/export_esp32.py \
  --checkpoint results/esp32s3/checkpoints/best_student.pt \
  --target esp32s3 \
  --espdl
```

This calls the official `espdl_quantize_onnx` API, enables TQT optimization, and requests
board-test values in the `.espdl` export.

For an original ESP32 target, pass `--target esp32`; the exporter automatically maps the ESP-PPQ
quantization target to `c`, as required by current ESP-DL guidance. For ESP32-S3, use
`--target esp32s3`.

## Raw INT8 path versus ESP-DL

### Smallest / most controllable

Use:

```text
ufuse_weights_int8.bin
```

or the generated C header, and implement the fixed five-layer integer graph directly. This avoids
model-parser metadata and is the smallest model-data representation.

### Easiest optimized deployment on ESP32-S3

Use:

```text
ufuse_s8_esp32s3.espdl
```

with ESP-DL/ESP-NN.

The custom raw quantizer is intentionally independent of ESP-DL. Its scales are not expected to
be bit-identical to ESP-PPQ's power-of-two quantization. Use the accompanying golden vectors to
validate whichever firmware path you choose.

## Golden-vector verification

`golden_vectors.npz` contains:

- `input`: normalized six-channel input samples
- `float_raw`: outputs of the BN-folded FP32 network
- `int8_raw`: outputs of the Python integer reference implementation

Before trusting the firmware, run exactly these samples on the ESP32 and compare all three output
values. This catches tensor-order, padding, stride, quantization and rounding mistakes.

## Current included smoke verification

The regenerated two-epoch smoke run is only a pipeline test, not a scientific accuracy result. It
verified that the new exporter selects a **structured rewound LTH ticket** and cannot substitute
the random control. In that smoke run the selected graph had **827 parameters** and the raw INT8
weight/bias blob was **904 bytes**. Final claims must use the official data and the stricter
`configs/esp32s3_official.yaml` quality guard.

## Tests

```bash
pytest -q
```

The project currently has **61 passing tests**. They cover the original research pipeline plus:

- data/preprocessing edge cases and normalization bounds;
- native and resampled streaming preprocessing;
- all original corruption scenarios plus `burst_nlos`, `burst_dropout`, and `mixed`;
- PF float32/float64 stability, determinism, tiny scales, and learned outlier probabilities;
- ESP32 architecture validation, hidden-neuron pruning and structured-ticket failure cases;
- BatchNorm folding equivalence;
- INT8 calibration validation, saturation behavior, deterministic binary export and SHA-256 integrity;
- generated LUT/header contents;
- checkpoint save/load round-trip;
- official MATLAB-schema conversion including complex CIR and 4x3 anchors;
- LTH checkpoint-policy enforcement and a short train -> checkpoint -> raw INT8 export integration test.

Run the fast embedded smoke pipeline with:

```bash
python scripts/train_esp32_pipeline.py --config configs/esp32s3_smoke.yaml --no-teacher
```

For additional robustness cases without changing the paper/default scenario list, use
`configs/robustness_extended.yaml`.
