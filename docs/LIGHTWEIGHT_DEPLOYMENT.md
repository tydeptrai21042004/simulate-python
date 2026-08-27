# Lightweight U-FusePF deployment

This branch adds a deployment-first path without changing the Student-t measurement model or the Particle Filter geometry.

## Recommended stack

- Native 176 delay bins first; test 128 only after retraining and multi-seed validation.
- `LiteUncertaintyFusionNet`: dense Conv1D widths 8/12/16, 5,167 trainable parameters.
- Batched six-link inference with `torch.inference_mode()`.
- Streaming preprocessing with pre-normalized static backgrounds.
- Particle Filter with anchor-distance reuse, precomputed Student-t constants, and optional FP32 arithmetic.
- Start with 128 particles and validate 96/128/180/256 before fixing the deployment value.

## Why dense Conv1D instead of depthwise Conv1D?

Depthwise separable convolution has fewer MACs, but weak/general-purpose CPUs do not always provide optimized grouped-convolution kernels. The included `scripts/benchmark_deployment.py` measures the actual target rather than assuming fewer MACs means lower latency.

## Structured lottery ticket

Run:

```bash
python scripts/train_structured_ticket.py \
  --config configs/deployment_lottery.yaml \
  --case 1 --seed 11
```

The script performs three experiments:

1. Train a wider 12/18/24 supernet.
2. Rank convolution channels by trained magnitude, select an 8/12/16 subnetwork, rewind those surviving weights to their original initialization, and retrain the physically compact ticket.
3. Train an 8/12/16 random-initialization control.

The third experiment is essential. A single ticket result is not evidence for the Lottery Ticket Hypothesis; the rewound ticket should match or beat the same compact architecture from a fresh initialization over repeated seeds.

## Canonical unstructured IMP

`uwb_tracking.models.apply_global_lottery_pruning()` and `rewind_pruned_model_()` are included for iterative magnitude-pruning experiments. Unstructured zeros mainly reduce stored nonzero weights unless the inference runtime has a sparse kernel. For generic CPU deployment, prefer the structured physically compact ticket.

## Streaming frame path

```python
from uwb_tracking.deployment import StreamingPreprocessor, infer_frame

prep = StreamingPreprocessor.from_data(data, input_length=176)
features = prep.prepare_frame(cir_frame, var_frame)  # [6, 6, 176]
mu_ns, sigma_ns = infer_frame(model, features, delay_max_ns=data.delay_grid_ns[-1])
```

The background profiles are normalized once; only one frame is processed at a time.

## Scientific validation before claiming improvement

Report at least three seeds and all held-out cases. Compare:

- ToF MAE/RMSE/P90.
- tracking RMSE/MAE/P90.
- uncertainty ECE/Brier and corruption AUROC/AUPRC.
- parameters and nonzero parameters.
- measured latency on the target hardware, not only MACs.
- peak RAM / package size.

For pruning, also compare a random compact model with exactly the same architecture and training budget.

## Optional learned link-quality head

The lite model also predicts `outlier_probability` with only 25 extra parameters. During robust training, the head receives the simulator's per-link corruption mask as an auxiliary target. `run_particle_filter` accepts an optional `predicted_outlier_probability` array so a calibrated quality score can increase the broad Student-t contamination prior on suspicious links.

Treat this as an optional calibrated feature: a class-balanced quality head can have a nonzero baseline even on clean links. Validate or calibrate the probability on held-out training/validation timestamps before allowing it to change the PF mixture prior. The default PF behavior is unchanged when this argument is omitted.

## Local smoke measurements in this repository

On the development CPU with one PyTorch thread and six links at 176 bins:

- research `UncertaintyFusionNet`: 32,950 parameters, about 1.16 ms per six-link frame;
- dense lite model: 5,167 parameters, about 0.65 ms per six-link frame.

The optimized PF preserves the same bistatic geometry and Student-t formula, but computes each particle-to-anchor distance once per update. In a local 180-particle benchmark it reduced median update time from roughly 0.38 ms to roughly 0.21 ms. Absolute latency is hardware-dependent; rerun the benchmark on the actual target.

A 12-epoch one-seed structured-ticket smoke run produced 5,167 parameters and validation MAE about 0.585 ns versus about 0.737 ns for the same compact architecture from a fresh random initialization. This is only a smoke result, not a scientific LTH conclusion; use multiple seeds/cases before making that claim.
