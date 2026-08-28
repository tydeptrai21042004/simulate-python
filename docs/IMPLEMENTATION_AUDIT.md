# Implementation Audit and Readiness

## Current state

| Area | Earlier limitation | Current implementation |
|---|---|---|
| Original method | Handcrafted/partial baselines | Direct official CIR/variance CNN graph plus faithful repository PF path |
| Official data | Manual/inexact conversion | Auto-acquisition, official variable names, XYZ→XY, complex-CIR magnitude, ToF audit and SHA-256 provenance |
| Target | Peak-delay heuristic | One-based delay-index regression with repository/corrected indexing modes |
| Novelty | Weighted handcrafted fusion | Learned CIR/variance reliability, heteroscedastic scale, expert disagreement and explicit learned outlier probability |
| Tracking | Equal/heuristic confidence | Predicted scale and learned outlier probability enter the adaptive Student-t mixture PF |
| Leakage | Weak split control | Timestamp validation for main protocol; official sample split retained only for parity |
| Simulator bias | Independent false peaks favored fusion | Correlated/independent false-peak sweep and multiple corruption scenarios |
| Rigor | One run / limited metrics | 3 cases, multiple seeds, ablations, robust scenarios, CI and paired tests |
| Lightweight export | Parameter-count-only compression | Structured rewound LTH with FP32 + INT8 + tracking guards and same-architecture random control |
| Quantization | Layerwise float requant multiplier | Per-output-channel INT8 + integer Q31 multiplier/right shift, manifests and golden vectors |
| Deployment evidence | ToF/tensor comparison only | Held-out end-to-end FP32 and INT8 Particle-Filter tracking evaluation |
| Reproducibility | Repeated scripts | Installable package, YAML, resolved configs, provenance, checkpoints, incremental CSV, 64 tests |

## Readiness assessment

- **Architecture/protocol reproduction:** strong; official graph/PF logic and official-schema conversion are implemented and tested.
- **Numerical paper reproduction:** the pipeline can now acquire/standardize the public source data automatically, but final reproduction claims still require actually running the long official configurations and comparing the resulting metrics with the paper.
- **Scientific novelty:** falsifiable via CIR-only, variance-only, fixed fusion, no-uncertainty and full U-FusePF comparisons, including the learned outlier-confidence pathway.
- **Lightweight claim:** technically well controlled because the deployment checkpoint must be a rewound structured-LTH ticket and must survive both FP32 and quantized tracking guards. Final quantitative claims still require the multi-case/multi-seed official study.
- **Experimental rigor:** the full protocols and aggregators are implemented; the remaining work is experimental execution/reporting rather than missing core pipeline code.

## Claim boundary

`results/quick` and `results/esp32s3_smoke` are synthetic/short integration runs. They demonstrate software and export mechanics only. They must not be quoted as real DWM1000 tracking accuracy or as proof that LTH is superior across the official cases/seeds.
