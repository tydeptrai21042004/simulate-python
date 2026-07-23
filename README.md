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
- a broad outlier component to prevent one unreliable link from collapsing the particle weights.

The core hypothesis is testable: **a model that estimates both ToF and link-specific uncertainty should reduce the tail of tracking error under NLoS, outliers and link dropout.**

### Experimental rigor

- 3 trajectory partitions.
- Default 5 random seeds.
- LoS, one-link NLoS, two-link NLoS, random outlier and dropout scenarios.
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

```bash
uwb-track full --config configs/full.yaml
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

Obtain the files referenced by the official repository:

- `Bg_CIR_VAR.mat`
- `Dyn_CIR_VAR.mat`
- `AnchorPos.mat`

Then run:

```bash
python scripts/convert_original_matlab_data.py \
  --background path/to/Bg_CIR_VAR.mat \
  --dynamic path/to/Dyn_CIR_VAR.mat \
  --anchors path/to/AnchorPos.mat \
  --output data/uwb_original_standard.mat
```

Change `data_path` in the YAML configuration to `data/uwb_original_standard.mat`.

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
├── simulation.py           # unbiased NLoS/outlier/dropout corruption
├── training.py             # losses, training, validation, prediction
├── experiments.py          # benchmark, statistics, ablation, sweeps
├── metrics.py              # tracking, ToF and uncertainty metrics
├── plotting.py             # trajectory, CDF and summary plots
├── models/
│   ├── paper_cnn.py        # paper-faithful residual CNN baseline
│   └── proposed.py         # U-FusePF uncertainty fusion network
└── tracking/
    └── particle_filter.py  # equal and uncertainty-aware Student-t PF
```

## Interpretation and research claims

A valid final conclusion requires the official experimental data or new measurements from four DWM1000 nodes. Synthetic results may be used to debug algorithms, test hypotheses and conduct controlled stress tests, but they cannot establish real-world tracking accuracy.

## Primary references

1. C. Li et al., “Multi-Static UWB Radar-based Passive Human Tracking Using COTS Devices,” *IEEE Antennas and Wireless Propagation Letters*, 2022, DOI: 10.1109/LAWP.2022.3141869.
2. Official code: `https://github.com/CLongLi/UWB-Radar-Pedestrian-Tracking`.
3. A. Ledergerber and R. D’Andrea, “A multi-static radar network with ultra-wideband radio-equipped devices,” *Sensors*, 2020.
4. F. Gustafsson et al., “Particle filters for positioning, navigation, and tracking,” *IEEE Transactions on Signal Processing*, 2002.
