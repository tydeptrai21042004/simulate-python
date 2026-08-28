# ESP32 deployment edition

For the complete Python training → structured Lottery Ticket → INT8/ESP-DL export workflow, see **[README_ESP32.md](README_ESP32.md)**.

---

# UWB Passive Human Tracking — Python Research Pipeline

This repository is a complete Python/PyTorch replacement for the previous MATLAB-only simulation. It contains two connected research tracks:

1. **Paper reproduction:** CIR-based and variance-based residual CNNs followed by a Student-t Particle Filter, using the three consecutive trajectory partitions described by Li et al.
2. **Proposed method — U-FusePF:** learned CIR–variance reliability fusion with heteroscedastic Student-t ToF regression and direct uncertainty propagation into the Particle Filter.

> The bundled `data/uwb_demo_input.mat` is a deterministic synthetic dataset for testing the complete pipeline. It is **not** the experimental DWM1000 dataset and must not be used to claim reproduction of the paper's reported centimetre-level results. A converter for the official MATLAB files is included.

## What was corrected

### Paper reproduction

- PyTorch implementation of the published input format: dynamic and background profile as a `500 × 2` tensor.
- Exact official graph: Conv `10×1`/8 → same pool `10×1`; Conv `4×2`/16 → same pool `4×2`; residual stages 32/64/128 using `4×1` kernels and `1×1` projection shortcuts.
- MATLAB image-mean subtraction, global max pooling, FC-10 → FC-1, no output sigmoid, and one-based delay-index regression.
- Separate CIR-CNN and variance-CNN baselines.
- Three consecutive trajectory train/test cases.
- Official t-location-scale error fit and a separate faithful 200-particle repository-PF port (fixed diffusion and resampling every update).
- Explicit repository-parity mode and corrected/leak-safe scientific mode.
- Adapter for `Bg_CIR_VAR.mat`, `Dyn_CIR_VAR.mat` and `AnchorPos.mat` from the official repository.

### Scientific novelty

The proposed method is no longer a handcrafted peak-score fusion. `U-FusePF` contains:

- the paper-faithful CIR-CNN and variance-CNN as two ToF experts;
- a compact local-reliability network using dynamic, background and absolute-difference profiles;
- local uncertainty for CIR and variance;
- validation-prior reliability fusion, where global validation accuracy is combined with sample-specific reliability without using test labels;
- a fused ToF mean and heteroscedastic Student-t scale, augmented by cross-expert disagreement;
- direct use of the predicted scale in the per-link Particle Filter likelihood;
- an explicit learned outlier-probability head supervised by corruption labels, propagated into the broad-component mixture of the adaptive Particle Filter.

The core hypothesis is testable: **a model that estimates both ToF and link-specific uncertainty should reduce the tail of tracking error under NLoS, outliers and link dropout.**

### Experimental rigor

- 3 trajectory partitions.
- Default 5 random seeds.
- LoS, one-link NLoS, two-link NLoS, random outlier and dropout scenarios. An optional extended robustness config adds burst-NLoS, burst-dropout and mixed corruption without changing the default protocol.
- Time-level train/validation separation for the scientific protocol, plus a separate official 85/15 sample-split mode for parity auditing.
- Mean, standard deviation and 95% confidence interval.
- Paired Wilcoxon tests and proposed-method win rate.
- Ablation modes: CIR-only, variance-only, fixed fusion, no uncertainty and full method.
- NLoS severity and cross-modal false-peak-correlation sweep.
- ToF MAE/RMSE/P90, tracking RMSE/MAE/median/P90, runtime, parameter count, ECE, Brier score, corruption AUROC and AUPRC.
- Incremental CSV saving so long experiments retain completed runs.
- Deterministic seeds and unit tests.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e .[dev]
```

Python 3.10+ is supported. A CUDA GPU is recommended for the full 3-case × 5-seed experiment, but the quick run works on CPU.

## Commands

### 1. Verify the code

```bash
pytest
```

The current simulation/LTH/export edition passes **64 tests**.

### 2. Quick end-to-end run

```bash
uwb-track quick-run --config configs/quick.yaml
```

The quick configuration uses one case, one seed, a shorter input and fewer epochs. It verifies the workflow; it is not a final scientific experiment.

### 3. Reproduce the official baselines

```bash
uwb-track reproduce-paper --config configs/paper_reproduction.yaml
```

With the bundled synthetic file, this validates the architecture and experiment code only. `configs/paper_reproduction_original.yaml` enables the official 50-epoch/Adam/10%-batch/sample-split/indexing protocol after the original data are converted.

### 4. Full proposed-method benchmark

Synthetic/config-development benchmark:

```bash
uwb-track full --config configs/full.yaml
```

Final official-data benchmark (3 cases × 5 seeds, automatic data setup):

```bash
uwb-track full --config configs/full_official.yaml
```

### 5. Ablation

```bash
uwb-track ablation --config configs/ablation.yaml
```

### 6. Robustness sweep

```bash
uwb-track robustness-sweep --config configs/robustness_sweep.yaml
```

## Use the original experimental data

The project can now fetch the public sources automatically. The official GitHub repository provides
`AnchorPos.mat` and `Bg_CIR_VAR.mat`; its README links the separately hosted `Dyn_CIR_VAR.mat`.
Run:

```bash
python scripts/fetch_original_data.py
```

or directly start the official LTH/INT8 pipeline:

```bash
./RUN_ESP32_OFFICIAL.sh
```

The converter explicitly matches the official MATLAB schema: `AnchorPos` is reduced from XYZ to
XY because the official PF uses only XY, `Dyn_var_CIRxx` / `Bg_var_CIRxx` are recognized, and
complex CIR is converted with magnitude rather than a lossy float cast. Manual conversion remains
available through `scripts/convert_original_matlab_data.py`.

## Main outputs

Each experiment directory contains:

- `results_raw.csv`: one row per case, seed, scenario and method;
- `results_summary.csv`: mean, standard deviation and 95% CI;
- `statistical_tests.csv`: paired comparisons against repository-PF and controlled equal-PF CNN baselines;
- `checkpoints/`: model weights and model reports;
- `figures/`: trajectories, CDFs and summary chart;
- `resolved_config.json`: exact configuration used.

## Repository structure

```text
src/uwb_tracking/
├── data.py                 # MAT loading, geometry, splits, preprocessing
├── simulation.py           # original + optional extended corruption scenarios
├── training.py             # research-model training/validation/prediction
├── experiments.py          # benchmark, statistics, ablation, sweeps
├── deployment.py           # frame-at-a-time low-RAM preprocessing/inference
├── metrics.py              # tracking, ToF and uncertainty metrics
├── plotting.py             # trajectory, CDF and summary plots
├── models/
│   ├── paper_cnn.py        # paper-faithful residual CNN baseline
│   ├── proposed.py         # U-FusePF uncertainty fusion network
│   ├── lite.py             # lightweight two-branch model
│   └── lottery.py          # structured/unstructured LTH utilities
├── esp32/
│   ├── model.py            # ~1-2k parameter deployment student
│   ├── training.py         # compact training/distillation
│   ├── teacher.py          # offline ensemble-teacher targets
│   ├── exporter.py         # BN fold + per-channel fixed-point INT8/ONNX/ESP-DL export
│   ├── evaluation.py       # FP32/INT8 ToF + end-to-end PF deployment metrics
│   └── preprocess_export.py# background/geometry runtime constants
└── tracking/
    └── particle_filter.py  # optimized uncertainty-aware Student-t PF
```

## Interpretation and research claims

A valid final conclusion requires the official experimental data or new measurements from four DWM1000 nodes. Synthetic results may be used to debug algorithms, test hypotheses and conduct controlled stress tests, but they cannot establish real-world tracking accuracy.

## Primary references

1. C. Li et al., “Multi-Static UWB Radar-based Passive Human Tracking Using COTS Devices,” *IEEE Antennas and Wireless Propagation Letters*, 2022, DOI: 10.1109/LAWP.2022.3141869.
2. Official code: `https://github.com/CLongLi/UWB-Radar-Pedestrian-Tracking`.
3. A. Ledergerber and R. D’Andrea, “A multi-static radar network with ultra-wideband radio-equipped devices,” *Sensors*, 2020.
4. F. Gustafsson et al., “Particle filters for positioning, navigation, and tracking,” *IEEE Transactions on Signal Processing*, 2002.

## Lightweight / Lottery-Ticket Deployment Path

For CPU/edge deployment, see [`docs/LIGHTWEIGHT_DEPLOYMENT.md`](docs/LIGHTWEIGHT_DEPLOYMENT.md).
The canonical lightweight path is now `scripts/train_esp32_pipeline.py` plus `src/uwb_tracking/esp32/`. It performs progressive structured-LTH architecture search with initialization rewinding, clean/robust/Particle-Filter quality guards, per-output-channel fixed-point INT8 export, and an end-to-end FP32-vs-INT8 tracking check. The older `models/lite.py` / `train_structured_ticket.py` path is retained only for historical research comparison.

Single official case/seed:

```bash
./RUN_ESP32_OFFICIAL.sh
```

Final multi-case/multi-seed lightweight study:

```bash
./RUN_ESP32_STUDY.sh
```
