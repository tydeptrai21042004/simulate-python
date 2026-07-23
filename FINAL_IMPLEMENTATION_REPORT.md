# Final Implementation Report

## Delivered scope

The MATLAB-only prototype was replaced by an installable Python/PyTorch research repository containing:

- an official-graph CIR-CNN port;
- an official-graph variance-CNN port;
- a vectorized port of the official 200-particle t-location-scale Particle Filter;
- repository-parity and leak-safe/corrected protocols;
- the proposed U-FusePF method;
- synthetic stress tests without structurally guaranteed fusion success;
- multi-case/multi-seed benchmarking, ablation, robustness sweeps and paired statistics;
- unit tests, YAML configs, checkpoints, CSV outputs and plots;
- a revised five-page Vietnamese proposal.

## Reassessment

| Area | Before | Current implementation assessment | Important boundary |
|---|---:|---:|---|
| Reproduction of original method | ~3/10 | **8.5/10 for architecture and repository protocol** | Numerical parity remains unverified until `Dyn_CIR_VAR.mat` is supplied and the original data are run |
| Scientific novelty | ~4.5/10 | **7.5/10** | Novelty must be supported by ablation and real/original-data results, not quick synthetic numbers |
| Experimental rigor | ~4.5/10 | **8/10 at protocol/code level** | The long 3-case × 5-seed experiments still need to be executed for final evidence |

## Why the reproduction score improved

The baseline is no longer a generic residual approximation. It now matches the official MATLAB graph's channel counts, two pooling stages, `4×1` residual kernels, projection shortcuts, zero-centering, FC layout and one-based delay-index regression. The repository training choices and PF behavior are explicit configuration modes rather than undocumented approximations.

## Why the novelty score improved

The proposed contribution is a testable mechanism rather than handcrafted peak fusion:

1. two official CNN experts provide independent CIR and variance ToF estimates;
2. a learned local reliability network predicts modality-specific reliability and aleatoric scale;
3. validation error supplies a global reliability prior without test-label access;
4. expert disagreement contributes an epistemic uncertainty proxy;
5. the final per-link uncertainty is propagated into a robust Student-t mixture Particle Filter.

## Why the rigor score improved

The repository separates code-parity from fair scientific comparison, controls timestamp leakage, evaluates five corruption conditions, supports correlated false peaks, saves every run incrementally, reports tail metrics and uncertainty calibration, and uses paired tests over cases and seeds.

## Verified state

- Python compilation: passed.
- Automated tests: 10/10 passed.
- Quick end-to-end experiment: passed and produced 16 result rows, checkpoints, CSV files and plots.
- Proposal rendering: five pages, visually checked without clipping or broken tables.

## Remaining work before a final thesis/paper claim

1. Obtain and convert the official dynamic MATLAB data.
2. Run repository-parity and compare intermediate tensors, delay indices, Student-t fit and PF trajectories against MATLAB.
3. Run the complete five-seed protocol on original or newly measured data.
4. Report all failure cases and avoid selecting only favorable seeds/scenarios.
5. Treat the quick synthetic output solely as a software smoke test.
