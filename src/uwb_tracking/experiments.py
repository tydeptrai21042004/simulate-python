from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from scipy.stats import wilcoxon

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import UWBData, get_case_split, load_uwb_mat, prepare_inputs
from .metrics import tof_metrics, tracking_metrics, uncertainty_metrics
from .plotting import plot_cdf, plot_summary, plot_trajectory
from .simulation import corrupt_observations
from .tracking import run_particle_filter, run_repository_particle_filter
from .training import TrainedModel, predict_delays, train_models_for_case
from .utils import ensure_dir, mean_std_ci, save_json


def _reshape_samples(values: np.ndarray, n_time: int, n_links: int) -> np.ndarray:
    return np.asarray(values).reshape(n_time, n_links)


def evaluate_scenario(
    data: UWBData,
    test_indices: np.ndarray,
    models: dict[str, TrainedModel],
    cfg: ExperimentConfig,
    case_id: int,
    seed: int,
    scenario: str,
    make_plots: bool,
    output_dir: Path,
    severity: float = 1.0,
    correlation_probability: float | None = None,
) -> tuple[list[dict[str, float | int | str]], dict[str, np.ndarray]]:
    obs = corrupt_observations(
        data,
        test_indices,
        scenario,
        seed=seed + 1000 * case_id + 17,
        correlated_false_peak_probability=(
            cfg.correlated_false_peak_probability
            if correlation_probability is None
            else correlation_probability
        ),
        severity=severity,
    )
    prepared = prepare_inputs(data, obs, cfg.model.input_length)
    n, l = len(test_indices), data.num_links
    true_delay = prepared.target_delay_ns.reshape(n, l)
    rows: list[dict[str, float | int | str]] = []
    trajectories: dict[str, np.ndarray] = {}
    tracking_errors: dict[str, np.ndarray] = {}

    # Predict all modality experts and the local reliability network once.
    cir_delay_flat, cir_scale_flat, _, cir_inference = predict_delays(models["paper_cir"], prepared, cfg)
    var_delay_flat, var_scale_flat, _, var_inference = predict_delays(models["paper_var"], prepared, cfg)

    method_specs: list[dict[str, object]] = [
        {
            "method": "Official CIR-CNN + repository PF",
            "pf_kind": "repository",
            "trained": models["paper_cir"],
            "adaptive": False,
            "pred_delay_flat": cir_delay_flat,
            "pred_scale_flat": cir_scale_flat,
            "inference_ms": cir_inference,
            "parameters": models["paper_cir"].parameter_count,
            "training_seconds": models["paper_cir"].training_seconds,
        },
        {
            "method": "Official Variance-CNN + repository PF",
            "pf_kind": "repository",
            "trained": models["paper_var"],
            "adaptive": False,
            "pred_delay_flat": var_delay_flat,
            "pred_scale_flat": var_scale_flat,
            "inference_ms": var_inference,
            "parameters": models["paper_var"].parameter_count,
            "training_seconds": models["paper_var"].training_seconds,
        },
        {
            "method": "Official CIR-CNN + stabilized equal PF",
            "pf_kind": "stabilized",
            "trained": models["paper_cir"],
            "adaptive": False,
            "pred_delay_flat": cir_delay_flat,
            "pred_scale_flat": cir_scale_flat,
            "inference_ms": cir_inference,
            "parameters": models["paper_cir"].parameter_count,
            "training_seconds": models["paper_cir"].training_seconds,
        },
        {
            "method": "Official Variance-CNN + stabilized equal PF",
            "pf_kind": "stabilized",
            "trained": models["paper_var"],
            "adaptive": False,
            "pred_delay_flat": var_delay_flat,
            "pred_scale_flat": var_scale_flat,
            "inference_ms": var_inference,
            "parameters": models["paper_var"].parameter_count,
            "training_seconds": models["paper_var"].training_seconds,
        },
    ]

    if "proposed" in models:
        _, local_scale_flat, extras, local_inference = predict_delays(models["proposed"], prepared, cfg)
        local_cir = extras.get("gate_cir", np.full_like(cir_delay_flat, 0.5))
        local_var = extras.get("gate_var", np.full_like(var_delay_flat, 0.5))

        # Global validation reliability is a prior; sample-specific predicted
        # reliability is the likelihood term. Their product yields the final
        # modality gate without using any test labels.
        prior_cir = 1.0 / (models["paper_cir"].validation_mae_ns ** 2 + 1e-4)
        prior_var = 1.0 / (models["paper_var"].validation_mae_ns ** 2 + 1e-4)
        weight_cir = local_cir * prior_cir
        weight_var = local_var * prior_var
        denominator = np.maximum(weight_cir + weight_var, 1e-8)
        weight_cir = weight_cir / denominator
        weight_var = weight_var / denominator
        fused_delay_flat = weight_cir * cir_delay_flat + weight_var * var_delay_flat

        # Total predictive uncertainty combines the learned aleatoric scale
        # and cross-expert disagreement (an epistemic proxy).
        disagreement = np.abs(cir_delay_flat - var_delay_flat)
        fused_scale_flat = np.sqrt(local_scale_flat ** 2 + (0.25 * disagreement) ** 2)
        deployed_parameters = (
            models["paper_cir"].parameter_count
            + models["paper_var"].parameter_count
            + models["proposed"].parameter_count
        )
        deployed_training = (
            models["paper_cir"].training_seconds
            + models["paper_var"].training_seconds
            + models["proposed"].training_seconds
        )
        deployed_inference = cir_inference + var_inference + local_inference
        common = {
            "trained": models["proposed"],
            "pred_delay_flat": fused_delay_flat,
            "pred_scale_flat": fused_scale_flat,
            "inference_ms": deployed_inference,
            "parameters": deployed_parameters,
            "training_seconds": deployed_training,
        }
        method_specs.extend([
            {"method": "U-Fuse + equal PF", "adaptive": False, "pf_kind": "stabilized", **common},
            {"method": "U-FusePF proposed", "adaptive": True, "pf_kind": "stabilized", **common},
        ])

    for spec in method_specs:
        method = str(spec["method"])
        trained = spec["trained"]
        assert isinstance(trained, TrainedModel)
        adaptive = bool(spec["adaptive"])
        pred_delay_flat = np.asarray(spec["pred_delay_flat"])
        pred_scale_flat = np.asarray(spec["pred_scale_flat"])
        inference_ms = float(spec["inference_ms"])
        pred_delay = _reshape_samples(pred_delay_flat, n, l)
        pred_scale = _reshape_samples(pred_scale_flat, n, l)
        pred_total = pred_delay + data.tof_los_ns[None, :]
        pf_kind = str(spec.get("pf_kind", "stabilized"))
        if pf_kind == "repository":
            trajectory, pf_diag = run_repository_particle_filter(
                pred_total,
                data,
                obs.time_s,
                seed=seed + 500 + case_id,
                error_scale_ns=trained.global_scale_ns,
                error_location_ns=trained.global_location_ns,
                error_nu=trained.global_nu,
                num_particles=200,
                velocity_noise_mps=5.0,
                c_m_per_ns=cfg.c_m_per_ns,
            )
        else:
            trajectory, pf_diag = run_particle_filter(
                pred_total,
                pred_scale,
                data,
                obs.time_s,
                cfg.particle_filter,
                seed=seed + (700 if adaptive else 500) + case_id,
                adaptive=adaptive,
                global_scale_ns=trained.global_scale_ns,
                c_m_per_ns=cfg.c_m_per_ns,
            )
        tm = tof_metrics(pred_delay, true_delay)
        trm = tracking_metrics(trajectory, obs.true_xy)
        um = uncertainty_metrics(
            pred_delay,
            true_delay,
            pred_scale,
            obs.corruption_mask,
        )
        row: dict[str, float | int | str] = {
            "case": case_id,
            "seed": seed,
            "scenario": scenario,
            "method": method,
            "severity": severity,
            "false_peak_correlation": (
                cfg.correlated_false_peak_probability
                if correlation_probability is None
                else correlation_probability
            ),
            "parameters": int(spec["parameters"]),
            "validation_mae_ns": trained.validation_mae_ns,
            "training_seconds": float(spec["training_seconds"]),
            "inference_ms_per_link": inference_ms,
            "pf_ms_per_update": pf_diag.runtime_ms_per_update,
            "total_ms_per_update": inference_ms * l + pf_diag.runtime_ms_per_update,
            "resample_count": pf_diag.resample_count,
            **tm,
            **trm,
            **um,
        }
        rows.append(row)
        trajectories[method] = trajectory
        tracking_errors[method] = 100.0 * np.linalg.norm(trajectory - obs.true_xy, axis=1)

    if make_plots:
        ensure_dir(output_dir)
        plot_trajectory(obs.true_xy, trajectories, output_dir / f"trajectory_case{case_id}_{scenario}_seed{seed}.png")
        plot_cdf(tracking_errors, output_dir / f"cdf_case{case_id}_{scenario}_seed{seed}.png")
    return rows, trajectories


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "tof_mae_ns",
        "tof_p90_ae_ns",
        "tracking_rmse_cm",
        "tracking_mae_cm",
        "tracking_p90_cm",
        "total_ms_per_update",
        "confidence_ece",
        "corruption_auroc",
    ]
    summary_rows: list[dict[str, float | str]] = []
    for (method, scenario), group in results.groupby(["method", "scenario"]):
        row: dict[str, float | str] = {"method": method, "scenario": scenario, "n_runs": float(len(group))}
        for metric in metric_columns:
            values = [float(v) for v in group[metric].dropna().tolist()]
            if not values:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_ci95"] = float("nan")
                continue
            mean, std, ci = mean_std_ci(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def paired_statistical_tests(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    proposed_name = "U-FusePF proposed"
    if proposed_name not in set(results["method"]):
        return pd.DataFrame(rows)
    for scenario, scenario_df in results.groupby("scenario"):
        pivot = scenario_df.pivot_table(
            index=["case", "seed"], columns="method", values="tracking_rmse_cm", aggfunc="mean"
        )
        for baseline in (
            "Official CIR-CNN + repository PF",
            "Official Variance-CNN + repository PF",
            "Official CIR-CNN + stabilized equal PF",
            "Official Variance-CNN + stabilized equal PF",
        ):
            if baseline not in pivot or proposed_name not in pivot:
                continue
            paired = pivot[[proposed_name, baseline]].dropna()
            if paired.empty:
                continue
            diff = paired[proposed_name] - paired[baseline]
            if len(paired) >= 5 and not np.allclose(diff, 0):
                stat, p_value = wilcoxon(paired[proposed_name], paired[baseline], alternative="two-sided")
            else:
                stat, p_value = float("nan"), float("nan")
            rows.append({
                "scenario": scenario,
                "baseline": baseline,
                "n_pairs": len(paired),
                "mean_difference_cm": float(diff.mean()),
                "median_difference_cm": float(diff.median()),
                "proposed_win_rate": float(np.mean(diff < 0)),
                "wilcoxon_statistic": float(stat),
                "p_value": float(p_value),
            })
    return pd.DataFrame(rows)


def run_benchmark(
    cfg: ExperimentConfig,
    proposed_ablation: str = "full",
    make_plots: bool = True,
    include_proposed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_uwb_mat(cfg.data_path)
    output = ensure_dir(cfg.output_dir)
    save_json(output / "resolved_config.json", cfg.to_dict())
    all_rows: list[dict[str, float | int | str]] = []

    for case_id in cfg.cases:
        train_idx, test_idx = get_case_split(data.num_time, case_id)
        for seed in cfg.seeds:
            print(f"[train] case={case_id} seed={seed} ablation={proposed_ablation}", flush=True)
            checkpoint_dir = output / "checkpoints" / f"case{case_id}_seed{seed}"
            models = train_models_for_case(
                data,
                train_idx,
                cfg,
                seed,
                checkpoint_dir,
                proposed_ablation=proposed_ablation,
                include_proposed=include_proposed,
            )
            model_report = {
                key: {
                    "name": value.name,
                    "parameters": value.parameter_count,
                    "validation_mae_ns": value.validation_mae_ns,
                    "global_scale_ns": value.global_scale_ns,
                    "global_location_ns": value.global_location_ns,
                    "global_nu": value.global_nu,
                    "training_seconds": value.training_seconds,
                }
                for key, value in models.items()
            }
            save_json(checkpoint_dir / "model_report.json", model_report)
            for scenario in cfg.scenarios:
                print(f"[eval] case={case_id} seed={seed} scenario={scenario}", flush=True)
                rows, _ = evaluate_scenario(
                    data,
                    test_idx,
                    models,
                    cfg,
                    case_id,
                    seed,
                    scenario,
                    make_plots=make_plots and case_id == cfg.cases[0] and seed == cfg.seeds[0],
                    output_dir=output / "figures",
                )
                for row in rows:
                    row["ablation"] = proposed_ablation
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(output / "results_raw.csv", index=False)

    results = pd.DataFrame(all_rows)
    summary = aggregate_results(results)
    results.to_csv(output / "results_raw.csv", index=False)
    summary.to_csv(output / "results_summary.csv", index=False)
    paired_statistical_tests(results).to_csv(output / "statistical_tests.csv", index=False)
    if not results.empty:
        plot_summary(results, output / "summary_rmse.png")
    return results, summary


def run_robustness_sweep(
    cfg: ExperimentConfig,
    severities: Iterable[float] = (0.5, 1.0, 1.5),
    correlations: Iterable[float] = (0.0, 0.5, 1.0),
) -> pd.DataFrame:
    data = load_uwb_mat(cfg.data_path)
    output = ensure_dir(cfg.output_dir)
    rows: list[dict[str, float | int | str]] = []
    for case_id in cfg.cases:
        train_idx, test_idx = get_case_split(data.num_time, case_id)
        for seed in cfg.seeds:
            models = train_models_for_case(
                data, train_idx, cfg, seed, output / "checkpoints" / f"case{case_id}_seed{seed}"
            )
            for severity in severities:
                for corr in correlations:
                    result_rows, _ = evaluate_scenario(
                        data, test_idx, models, cfg, case_id, seed, "nlos1", False, output / "figures",
                        severity=float(severity), correlation_probability=float(corr)
                    )
                    rows.extend(result_rows)
                    pd.DataFrame(rows).to_csv(output / "robustness_sweep_raw.csv", index=False)
    result = pd.DataFrame(rows)
    if not result.empty:
        summary = result.groupby(
            ["method", "severity", "false_peak_correlation"], as_index=False
        ).agg(
            tracking_rmse_cm_mean=("tracking_rmse_cm", "mean"),
            tracking_rmse_cm_std=("tracking_rmse_cm", "std"),
            tof_mae_ns_mean=("tof_mae_ns", "mean"),
            confidence_auroc_mean=("corruption_auroc", "mean"),
        )
        summary.to_csv(output / "robustness_sweep_summary.csv", index=False)
    return result


def run_ablation_suite(
    cfg: ExperimentConfig,
    modes: Iterable[str] = ("cir_only", "var_only", "fixed_fusion", "no_uncertainty", "full"),
) -> pd.DataFrame:
    original_output = cfg.output_dir
    frames: list[pd.DataFrame] = []
    for mode in modes:
        cfg.output_dir = str(Path(original_output) / mode)
        results, _ = run_benchmark(cfg, proposed_ablation=mode, make_plots=False)
        proposed = results[results["method"].str.contains("U-Fuse")].copy()
        proposed["ablation"] = mode
        frames.append(proposed)
    cfg.output_dir = original_output
    combined = pd.concat(frames, ignore_index=True)
    ensure_dir(original_output)
    combined.to_csv(Path(original_output) / "ablation_raw.csv", index=False)
    aggregate_results(combined).to_csv(Path(original_output) / "ablation_summary.csv", index=False)
    return combined
