# Original-Paper / Official-Repository Reproduction

## Reproduced implementation details

The Python baseline is a direct port of `CIR_CNN_CIRVar_Tst.m` rather than a generic residual CNN:

- input `500 × 2 × 1`: independently min–max-normalized dynamic and background profiles;
- image-mean subtraction corresponding to MATLAB `imageInputLayer` zero-centering;
- Conv `10×1`, 8 channels → BN → ReLU → same max-pool `10×1`, stride `5×1`;
- Conv `4×2`, 16 channels → BN → ReLU → same max-pool `4×2`, stride `2×2`;
- three residual stages using `4×1` convolutions and `1×1` projection shortcuts, with 32, 64 and 128 channels;
- global max pooling → FC-10 → FC-1, with no sigmoid and no activation between the two fully connected layers;
- one-based delay-index regression;
- separate CIR and variance networks.

The official repository Particle Filter is also available as a separate baseline path:

- 200 particles;
- initialization in `x∈[-5,3]`, `y∈[-5,1]`;
- position diffusion `Δt × 5 × N(0,1)`;
- one fitted t-location-scale error model shared by all links;
- multinomial resampling at every timestamp.

## Two protocol modes

### Repository protocol

`configs/paper_reproduction_original.yaml` preserves the official implementation choices:

- 50 epochs;
- Adam, learning rate `1e-3`;
- mini-batch `floor(num_train/10)`;
- random 85/15 sample-level validation split;
- last-epoch deployment;
- the repository's one-bin indexing formula;
- repository Particle Filter.

This mode is needed for code-to-code parity.

### Leak-safe scientific protocol

The full proposed-method configs use:

- timestamp-level validation separation;
- corrected delay-index conversion;
- best-validation checkpoint and early stopping;
- multiple seeds and paired statistics.

This mode is used for fair scientific comparison, while the repository protocol is retained to audit reproduction.

## Data dependency and honest status

The official GitHub repository does not bundle `Dyn_CIR_VAR.mat`. The bundled synthetic file can verify architecture, training, tracking, logging and plots, but it cannot verify the paper's experimental RMSE. Use `scripts/convert_original_matlab_data.py` when the three official MATLAB files are available.

The converter standardizes the profiles and common tracking timeline. If exact per-link native timestamps are required for a forensic parity study, retain the original files and compare intermediate arrays against MATLAB before interpreting final RMSE.

## Reproduction acceptance checks

On original data, audit in this order:

1. normalized input tensors and image means;
2. one-based labels and index offset;
3. three consecutive held-out trajectory regions;
4. validation residual t-location-scale parameters;
5. six resampled ToF streams;
6. PF initialization, diffusion and resampling;
7. ToF and tracking error trends across all three cases.
