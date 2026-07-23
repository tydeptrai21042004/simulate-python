# Experimental Protocol

## Primary comparison

- Official CIR-CNN + official-repository Particle Filter.
- Official variance-CNN + official-repository Particle Filter.
- Both official CNNs + the same stabilized equal-scale Particle Filter used by the controlled comparison.
- U-Fuse ToF fusion + stabilized equal-scale Particle Filter.
- Full U-FusePF with predicted per-link scale.

The repository-PF rows answer the reproduction question. Re-running each official CNN with the stabilized equal-scale PF prevents PF implementation from confounding the estimator comparison. `U-Fuse + equal PF` versus `U-FusePF` isolates the contribution of uncertainty-aware tracking.

## Splits and leakage control

The trajectory is divided into three consecutive held-out thirds. Each case trains on two regions and tests on the remaining continuous region.

Two validation modes are explicit:

- `sample`: official MATLAB-style random 85/15 split of concatenated link samples;
- `time`: timestamp-level split used for the main scientific comparison, so another link or augmented copy of the same timestamp cannot leak into validation.

## Randomness

The full protocol uses five seeds by default. Model initialization, augmentation, corruption and particle resampling are all seeded. Every result row records case, seed, scenario and exact resolved configuration.

## Robustness conditions

- LoS.
- NLoS-1: one blocked link per timestamp.
- NLoS-2: two blocked links.
- Outlier: random false-path corruption.
- Dropout: observations collapse toward the background.

False peaks can be correlated or independent between CIR and variance. The robustness sweep changes both corruption severity and cross-modal false-peak correlation, preventing the simulator from structurally guaranteeing that fusion succeeds.

## Ablation

- `cir_only`;
- `var_only`;
- `fixed_fusion`;
- `no_uncertainty`;
- `full`.

The equal-PF deployment of the same fused ToF predictions separates estimator gains from adaptive-PF gains.

## Metrics

ToF: MAE, RMSE, median absolute error and P90 in ns.

Tracking: RMSE, MAE, median and P90 in cm, plus error CDF.

Efficiency: parameter count, training time, inference ms/link, PF ms/update and total ms/update.

Uncertainty: ECE, Brier score, corruption AUROC and AUPRC.

## Statistical analysis

Results are paired by `(case, seed)`. For each scenario the project writes:

- mean, standard deviation and 95% confidence interval;
- mean and median paired difference;
- proposed-method win rate;
- two-sided Wilcoxon signed-rank test when at least five pairs are available.

## Claim boundary

Synthetic results establish software correctness and controlled behavior only. A strong real-world conclusion requires the official experimental dataset or newly collected four-node DWM1000 data with credible ground truth.
