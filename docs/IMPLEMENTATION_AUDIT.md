# Implementation Audit and Readiness

## What was fixed

| Area | Previous state | Python project state |
|---|---|---|
| Original method | Handcrafted peak baselines | Direct official CNN graph plus official PF port |
| Target | Peak delay heuristic | One-based delay-index regression with repository/corrected indexing modes |
| Training parity | No paper model | Adam, 50 epochs, 10% batch and sample validation mode available |
| Novelty | Weighted handcrafted fusion | Learned local reliability, validation prior, heteroscedastic scale and expert disagreement |
| Tracking | Heuristic confidence weight | Predictive scale enters a robust Student-t mixture likelihood |
| Leakage | Not controlled | Timestamp validation for main protocol; official sample split retained only for parity |
| Simulator bias | Independent false peaks favored fusion | Correlated/independent false-peak sweep |
| Rigor | One run, limited metrics | 3 cases, multiple seeds, ablation, severity sweep, CI and paired tests |
| Reproducibility | Repeated MATLAB scripts | Installable package, YAML, checkpoints, incremental CSV and tests |

## Current readiness assessment

- **Architecture/protocol reproduction:** strong. The official graph and PF logic are implemented and unit-tested.
- **Numerical paper reproduction:** pending the missing official dynamic MATLAB file; synthetic numbers are not substitutes.
- **Scientific novelty:** materially improved and falsifiable through estimator/PF ablations.
- **Experimental rigor:** full protocol is implemented; final evidence still requires actually running the long configurations and reporting all failures.

## Quick-run boundary

`results/quick` is a one-case, one-seed smoke test. It proves the pipeline executes end-to-end. It must not be quoted as final experimental evidence.
