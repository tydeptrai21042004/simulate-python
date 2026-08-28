#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import json
from dataclasses import asdict
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwb_tracking.data import get_case_split, load_uwb_mat, prepare_inputs, subset_observations
from uwb_tracking.simulation import augment_training_observations, corrupt_observations
from uwb_tracking.official_data import ensure_official_standard_dataset
from uwb_tracking.esp32.exporter import ESP32ExportNet, calibrate_and_quantize, export_checkpoint
from uwb_tracking.esp32.preprocess_export import export_preprocess_and_geometry
from uwb_tracking.esp32.evaluation import (
    deployment_point_metrics,
    deployment_tracking_metrics,
    particle_filter_config_from_mapping,
    predict_fp32_raw,
    predict_int8_raw,
)
from uwb_tracking.esp32.model import ESP32Architecture, ESP32StudentNet, build_rewound_structured_ticket
from uwb_tracking.esp32.teacher import load_ensemble_teacher, teacher_targets
from uwb_tracking.esp32.training import (
    ESP32TrainingConfig,
    save_student_checkpoint,
    train_student,
)


def _split_train_val(indices: np.ndarray, seed: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Hold out one contiguous validation segment whenever possible.

    Per-link ToF validation works with arbitrary timestamp samples, but an
    end-to-end Particle Filter guard needs a coherent temporal sequence. The
    split therefore finds consecutive runs inside the case training indices and
    removes a deterministic contiguous block.
    """

    idx = np.sort(np.asarray(indices, dtype=np.int64))
    if idx.size < 2:
        raise ValueError("training split must contain at least two timestamps")
    if not (0.0 < float(fraction) < 0.5):
        raise ValueError("validation_fraction must be in (0, 0.5)")
    n_val = max(1, int(round(len(idx) * fraction)))
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    runs = [run for run in np.split(idx, breaks) if run.size >= n_val]
    rng = np.random.default_rng(seed)
    if runs:
        # Prefer the longest run; seeded start location avoids always selecting
        # the same trajectory edge while keeping exact reproducibility.
        max_len = max(run.size for run in runs)
        candidates = [run for run in runs if run.size == max_len]
        run = candidates[int(rng.integers(0, len(candidates)))]
        start = int(rng.integers(0, run.size - n_val + 1))
        val = run[start : start + n_val]
    else:
        # Extremely fragmented custom data: fall back to timestamp-level split.
        val = np.sort(rng.choice(idx, size=n_val, replace=False))
    train = np.setdiff1d(idx, val, assume_unique=True)
    if train.size < 1:
        raise ValueError("validation split consumed all training timestamps")
    return train, val


def _arch(raw: dict) -> ESP32Architecture:
    return ESP32Architecture(tuple(int(v) for v in raw["channels"]), int(raw["hidden"]))


def _train_cfg(raw: dict, seed: int) -> ESP32TrainingConfig:
    return ESP32TrainingConfig(
        epochs_supernet=int(raw.get("epochs_supernet", 30)),
        epochs_ticket=int(raw.get("epochs_ticket", 40)),
        epochs_control=int(raw.get("epochs_control", 40)),
        batch_size=int(raw.get("batch_size", 64)),
        learning_rate=float(raw.get("learning_rate", 2e-3)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        patience=int(raw.get("patience", 10)),
        student_nu=float(raw.get("student_nu", 4.0)),
        min_scale_fraction=float(raw.get("min_scale_fraction", 0.004)),
        distill_mean_weight=float(raw.get("distill_mean_weight", 1.0)),
        distill_scale_weight=float(raw.get("distill_scale_weight", 0.25)),
        corruption_weight=float(raw.get("corruption_weight", 0.15)),
        seed=seed,
        device=str(raw.get("device", "cpu")),
    )


def _parameter_count(arch: ESP32Architecture, min_scale_fraction: float) -> int:
    return sum(p.numel() for p in ESP32StudentNet(arch, min_scale_fraction).parameters())


def _ticket_candidates(cfg: dict, super_arch: ESP32Architecture, min_scale_fraction: float) -> list[ESP32Architecture]:
    raw_candidates = cfg.get("ticket_candidates")
    if raw_candidates is None:
        raw_candidates = [cfg["ticket"]]
    candidates = [_arch(raw) for raw in raw_candidates]
    unique: dict[tuple[tuple[int, int, int], int], ESP32Architecture] = {}
    for arch in candidates:
        if any(small > large for small, large in zip(arch.channels, super_arch.channels)):
            raise ValueError(f"ticket candidate {arch} is wider than supernet {super_arch}")
        if arch.hidden > super_arch.hidden:
            raise ValueError(f"ticket candidate hidden={arch.hidden} exceeds supernet hidden={super_arch.hidden}")
        if arch.channels == super_arch.channels and arch.hidden == super_arch.hidden:
            raise ValueError("ticket candidates must remove at least one physical channel/neuron")
        unique[(arch.channels, arch.hidden)] = arch
    # The progressive search must try the physically smallest model first.
    return sorted(unique.values(), key=lambda a: (_parameter_count(a, min_scale_fraction), a.channels, a.hidden))


def _quality_guard_limits(
    super_clean: dict[str, float],
    super_robust: dict[str, float],
    super_tracking: dict[str, float | int | str],
    raw: dict,
) -> dict[str, float]:
    """Build FP32 + INT8 quality limits from the uncompressed supernet."""

    rel = float(raw.get("max_relative_mae_loss", 0.08))
    absolute = float(raw.get("max_absolute_mae_loss_ns", 0.03))
    robust_rel = float(raw.get("max_robust_relative_mae_loss", rel))
    robust_abs = float(raw.get("max_robust_absolute_mae_loss_ns", max(absolute, 0.05)))
    max_nll_increase = float(raw.get("max_nll_increase", 0.25))
    max_bce_increase = float(raw.get("max_outlier_bce_increase", 0.10))
    max_int8_mae = float(raw.get("max_int8_mae_increase_ns", 0.03))
    max_int8_robust_mae = float(raw.get("max_int8_robust_mae_increase_ns", max_int8_mae))
    max_int8_nll = float(raw.get("max_int8_nll_increase", 0.25))
    max_int8_bce = float(raw.get("max_int8_outlier_bce_increase", 0.10))
    max_tracking = float(raw.get("max_tracking_rmse_increase_cm", 8.0))
    max_int8_tracking = float(raw.get("max_int8_tracking_rmse_increase_cm", 8.0))
    values = (
        rel, absolute, robust_rel, robust_abs, max_nll_increase, max_bce_increase,
        max_int8_mae, max_int8_robust_mae, max_int8_nll, max_int8_bce,
        max_tracking, max_int8_tracking,
    )
    if min(values) < 0:
        raise ValueError("LTH/INT8 quality tolerances must be >= 0")

    clean_ref = float(super_clean["tof_mae_ns"])
    robust_ref = float(super_robust["tof_mae_ns"])
    return {
        "mae_limit_ns": min(clean_ref * (1.0 + rel), clean_ref + absolute),
        "robust_mae_limit_ns": min(
            robust_ref * (1.0 + robust_rel), robust_ref + robust_abs
        ),
        "robust_nll_limit": float(super_robust["nll"]) + max_nll_increase,
        "robust_outlier_bce_limit": float(super_robust["outlier_bce"]) + max_bce_increase,
        "max_relative_mae_loss": rel,
        "max_absolute_mae_loss_ns": absolute,
        "max_robust_relative_mae_loss": robust_rel,
        "max_robust_absolute_mae_loss_ns": robust_abs,
        "max_nll_increase": max_nll_increase,
        "max_outlier_bce_increase": max_bce_increase,
        "max_int8_mae_increase_ns": max_int8_mae,
        "max_int8_robust_mae_increase_ns": max_int8_robust_mae,
        "max_int8_nll_increase": max_int8_nll,
        "max_int8_outlier_bce_increase": max_int8_bce,
        "tracking_rmse_limit_cm": float(super_tracking["tracking_rmse_cm"]) + max_tracking,
        "max_tracking_rmse_increase_cm": max_tracking,
        "max_int8_tracking_rmse_increase_cm": max_int8_tracking,
    }


def _balanced_calibration_subset(
    x: np.ndarray,
    corruption: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Choose deterministic clean/corrupted calibration samples when available."""

    count = min(int(count), len(x))
    if count < 1:
        raise ValueError("calibration_samples must be >= 1")
    rng = np.random.default_rng(seed)
    corruption = np.asarray(corruption).reshape(-1)
    clean_idx = np.flatnonzero(corruption < 0.5)
    bad_idx = np.flatnonzero(corruption >= 0.5)
    desired_bad = min(len(bad_idx), count // 2)
    desired_clean = min(len(clean_idx), count - desired_bad)
    selected: list[int] = []
    if desired_bad:
        selected.extend(rng.choice(bad_idx, size=desired_bad, replace=False).tolist())
    if desired_clean:
        selected.extend(rng.choice(clean_idx, size=desired_clean, replace=False).tolist())
    remaining = count - len(selected)
    if remaining:
        pool = np.setdiff1d(np.arange(len(x)), np.asarray(selected, dtype=np.int64), assume_unique=False)
        selected.extend(rng.choice(pool, size=remaining, replace=False).tolist())
    rng.shuffle(selected)
    return x[np.asarray(selected, dtype=np.int64)]


def _resolve_data(cfg: dict, auto_data_cli: bool) -> tuple[Path, dict[str, object]]:
    data_path = ROOT / str(cfg["data_path"])
    official = cfg.get("official_data", {}) or {}
    auto = bool(official.get("auto_download", False) or auto_data_cli)
    if data_path.exists():
        return data_path, {"auto_download": auto, "downloaded": False, "path": str(data_path)}
    if not auto:
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. For official data either run "
            "python scripts/fetch_original_data.py or enable official_data.auto_download."
        )
    source_dir = ROOT / str(
        official.get("source_dir", "data/original_uwb/UWB-Radar-Pedestrian-Tracking")
    )
    output = ensure_official_standard_dataset(
        data_path,
        source_dir,
        force_download=bool(official.get("force_download", False)),
        force_convert=bool(official.get("force_convert", False)),
    )
    return output, {
        "auto_download": True,
        "downloaded": True,
        "path": str(output),
        "source_dir": str(source_dir),
    }


def _evaluate_deployment_scenarios(
    *,
    cfg: dict,
    data,
    test_idx: np.ndarray,
    input_length: int,
    seed: int,
    selected_model: ESP32StudentNet,
    control_model: ESP32StudentNet,
    calibration_x: np.ndarray,
    delay_max_ns: float,
    training_cfg: ESP32TrainingConfig,
    output_dir: Path,
) -> dict[str, object]:
    """Held-out end-to-end check of FP32 ticket, INT8 ticket and random control."""

    raw_cfg = cfg.get("deployment_evaluation", {}) or {}
    if not bool(raw_cfg.get("enabled", True)):
        return {"enabled": False, "rows": []}
    scenarios = [str(v) for v in raw_cfg.get("scenarios", ["los", "nlos1", "nlos2", "outlier", "dropout"])]
    pf_cfg = particle_filter_config_from_mapping(raw_cfg.get("particle_filter", {}))
    corr_prob = float(raw_cfg.get("correlated_false_peak_probability", 0.5))
    severity = float(raw_cfg.get("severity", 1.0))

    selected_bundle = calibrate_and_quantize(
        ESP32ExportNet(selected_model).eval(),
        calibration_x,
        delay_max_ns=delay_max_ns,
        min_scale_fraction=training_cfg.min_scale_fraction,
    )
    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(scenarios):
        if scenario == "los":
            obs = subset_observations(data, test_idx)
        else:
            obs = corrupt_observations(
                data, test_idx, scenario, seed + 12000 + scenario_index,
                correlated_false_peak_probability=corr_prob, severity=severity,
            )
        prepared = prepare_inputs(data, obs, input_length)
        raw_by_variant = {
            "lth_fp32": predict_fp32_raw(selected_model, prepared.fusion),
            "lth_int8_reference": predict_int8_raw(selected_bundle, prepared.fusion),
            "random_compact_fp32": predict_fp32_raw(control_model, prepared.fusion),
        }
        pf_seed = seed + 13000 + scenario_index
        for variant, raw_pred in raw_by_variant.items():
            metrics = deployment_tracking_metrics(
                raw_pred,
                prepared,
                data,
                obs.true_xy,
                obs.time_s,
                pf_cfg,
                seed=pf_seed,
                delay_max_ns=delay_max_ns,
                min_scale_fraction=training_cfg.min_scale_fraction,
                student_nu=training_cfg.student_nu,
            )
            rows.append({
                "case": int(cfg.get("case", 1)),
                "seed": seed,
                "scenario": scenario,
                "variant": variant,
                "num_particles": int(pf_cfg.num_particles),
                **metrics,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "deployment_tracking_results.json"
    csv_path = output_dir / "deployment_tracking_results.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _find(scenario: str, variant: str) -> dict[str, object] | None:
        return next((r for r in rows if r["scenario"] == scenario and r["variant"] == variant), None)

    los_fp32 = _find("los", "lth_fp32")
    los_int8 = _find("los", "lth_int8_reference")
    quant_delta = None
    if los_fp32 is not None and los_int8 is not None:
        quant_delta = {
            "tof_mae_delta_ns": float(los_int8["tof_mae_ns"]) - float(los_fp32["tof_mae_ns"]),
            "tracking_rmse_delta_cm": float(los_int8["tracking_rmse_cm"]) - float(los_fp32["tracking_rmse_cm"]),
            "tracking_p90_delta_cm": float(los_int8["tracking_p90_cm"]) - float(los_fp32["tracking_p90_cm"]),
        }
    return {
        "enabled": True,
        "scenarios": scenarios,
        "particle_filter": asdict(pf_cfg),
        "results_json": str(json_path.relative_to(ROOT)),
        "results_csv": str(csv_path.relative_to(ROOT)),
        "quantization_los_delta": quant_delta,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a progressive structured-LTH ESP32 student and export the selected ticket."
    )
    parser.add_argument("--config", default="configs/esp32s3.yaml")
    parser.add_argument("--no-teacher", action="store_true", help="Disable teacher distillation.")
    parser.add_argument("--auto-data", action="store_true", help="Auto-fetch official data if data_path is missing.")
    parser.add_argument("--onnx", action="store_true", help="Also export ONNX opset 18.")
    parser.add_argument("--espdl", action="store_true", help="Also quantize/export .espdl with ESP-PPQ.")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 11))
    case = int(cfg.get("case", 1))
    input_length = int(cfg.get("input_length", 176))
    output_dir = ROOT / str(cfg.get("output_dir", "results/esp32s3"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_path, data_info = _resolve_data(cfg, args.auto_data)
    data = load_uwb_mat(data_path)
    train_idx, test_idx = get_case_split(data.num_time, case)
    train_core, val_idx = _split_train_val(
        train_idx, seed, float(cfg.get("validation_fraction", 0.15))
    )

    clean_obs = subset_observations(data, train_core)
    val_obs = subset_observations(data, val_idx)
    aug_obs = augment_training_observations(
        data,
        train_core,
        seed + 7000,
        probability=float(cfg.get("augmentation_probability", 0.30)),
    )
    robust_val_cfg = cfg.get("robust_validation", {}) or {}
    robust_val_obs = corrupt_observations(
        data,
        val_idx,
        str(robust_val_cfg.get("scenario", "mixed")),
        seed + 8000,
        correlated_false_peak_probability=float(
            robust_val_cfg.get("correlated_false_peak_probability", 0.5)
        ),
        severity=float(robust_val_cfg.get("severity", 1.0)),
    )
    clean = prepare_inputs(data, clean_obs, input_length)
    val = prepare_inputs(data, val_obs, input_length)
    robust_val = prepare_inputs(data, robust_val_obs, input_length)
    aug = prepare_inputs(data, aug_obs, input_length)

    train_x = np.concatenate([clean.fusion, clean.fusion, aug.fusion], axis=0).astype(np.float32)
    train_y = np.concatenate([clean.target_fraction, clean.target_fraction, aug.target_fraction], axis=0).astype(np.float32)
    train_corruption = np.concatenate([clean.corruption, clean.corruption, aug.corruption], axis=0).astype(np.float32)
    val_x = val.fusion.astype(np.float32)
    val_y = val.target_fraction.astype(np.float32)
    val_corruption = val.corruption.astype(np.float32)
    delay_max_ns = float(data.delay_grid_ns[-1])
    export_cfg = cfg.get("export", {}) or {}
    calibration_x = _balanced_calibration_subset(
        train_x,
        train_corruption,
        int(export_cfg.get("calibration_samples", 256)),
        seed + 9000,
    )

    teacher_mean = None
    teacher_scale = None
    teacher_info: dict[str, object] = {"enabled": False}
    use_teacher = bool(cfg.get("use_teacher_distillation", True)) and not args.no_teacher
    teacher_dir = ROOT / str(cfg.get("teacher_checkpoint_dir", ""))
    if use_teacher and teacher_dir.exists():
        teacher = load_ensemble_teacher(teacher_dir)
        tm_clean, ts_clean = teacher_targets(teacher, data, clean_obs, device="cpu")
        tm_aug, ts_aug = teacher_targets(teacher, data, aug_obs, device="cpu")
        teacher_mean = np.concatenate([tm_clean, tm_clean, tm_aug]).astype(np.float32)
        teacher_scale = np.concatenate([ts_clean, ts_clean, ts_aug]).astype(np.float32)
        teacher_info = {
            "enabled": True,
            "checkpoint_dir": str(teacher_dir.relative_to(ROOT) if teacher_dir.is_relative_to(ROOT) else teacher_dir),
            "teacher_input_length": teacher.input_length,
        }
    elif use_teacher:
        teacher_info = {"enabled": False, "reason": f"teacher directory not found: {teacher_dir}"}

    train_cfg = _train_cfg(cfg.get("training", {}), seed)
    super_arch = _arch(cfg["supernet"])
    lth_cfg = cfg.get("lth_search", {}) or {}
    candidates = _ticket_candidates(cfg, super_arch, train_cfg.min_scale_fraction)

    # 1) Wider supernet discovers channel/neuron identities.
    torch.manual_seed(seed + 100)
    supernet = ESP32StudentNet(super_arch, train_cfg.min_scale_fraction)
    initial_supernet = copy.deepcopy(supernet.state_dict())
    super_result = train_student(
        supernet,
        train_x,
        train_y,
        train_corruption,
        val_x,
        val_y,
        val_corruption,
        delay_max_ns,
        train_cfg,
        epochs=train_cfg.epochs_supernet,
        seed=seed + 100,
        teacher_mean=teacher_mean,
        teacher_scale=teacher_scale,
    )
    save_student_checkpoint(
        checkpoint_dir / "supernet.pt",
        super_result,
        input_length,
        delay_max_ns,
        train_cfg,
        {"role": "channel-neuron-discovery-supernet", "teacher": teacher_info},
    )

    # 2) Progressive structured Lottery Ticket search. Every deployable candidate
    # is selected from the trained supernet and rewound to its initial values.
    # Search starts from the smallest graph and stops at the first ticket that
    # stays inside the configured MAE/NLL/outlier-quality budget relative to the uncompressed supernet.
    super_clean_raw = predict_fp32_raw(super_result.model, val.fusion)
    super_clean_metrics = deployment_point_metrics(
        super_clean_raw,
        val,
        delay_max_ns,
        train_cfg.min_scale_fraction,
        train_cfg.student_nu,
    )
    super_robust_metrics = deployment_point_metrics(
        predict_fp32_raw(super_result.model, robust_val.fusion),
        robust_val,
        delay_max_ns,
        train_cfg.min_scale_fraction,
        train_cfg.student_nu,
    )
    tracking_guard_pf_raw = dict(((cfg.get("deployment_evaluation", {}) or {}).get("particle_filter", {}) or {}))
    tracking_guard_pf_raw["num_particles"] = int(lth_cfg.get("tracking_guard_particles", 128))
    tracking_guard_pf = particle_filter_config_from_mapping(tracking_guard_pf_raw)
    tracking_guard_seed = seed + 9100
    super_tracking_metrics = deployment_tracking_metrics(
        super_clean_raw, val, data, val_obs.true_xy, val_obs.time_s, tracking_guard_pf,
        seed=tracking_guard_seed, delay_max_ns=delay_max_ns,
        min_scale_fraction=train_cfg.min_scale_fraction, student_nu=train_cfg.student_nu,
    )
    quality_limits = _quality_guard_limits(
        super_clean_metrics, super_robust_metrics, super_tracking_metrics, lth_cfg
    )
    search_records: list[dict[str, object]] = []
    selected_ticket = None
    selected_result = None
    selected_selection = None
    selected_path: Path | None = None
    best_ticket = None

    for idx, ticket_arch in enumerate(candidates):
        ticket, selection = build_rewound_structured_ticket(
            super_result.model, initial_supernet, ticket_arch
        )
        ticket_result = train_student(
            ticket,
            train_x,
            train_y,
            train_corruption,
            val_x,
            val_y,
            val_corruption,
            delay_max_ns,
            train_cfg,
            epochs=train_cfg.epochs_ticket,
            seed=seed + 101 + idx,
            teacher_mean=teacher_mean,
            teacher_scale=teacher_scale,
        )
        selection_dict = {
            "c1": selection.c1.tolist(),
            "c2": selection.c2.tolist(),
            "c3": selection.c3.tolist(),
            "hidden": selection.hidden.tolist(),
        }
        candidate_path = checkpoint_dir / (
            f"lth_ticket_{idx:02d}_c{'-'.join(map(str, ticket_arch.channels))}_h{ticket_arch.hidden}.pt"
        )
        fp32_clean_raw = predict_fp32_raw(ticket_result.model, val.fusion)
        fp32_clean_metrics = deployment_point_metrics(
            fp32_clean_raw,
            val, delay_max_ns, train_cfg.min_scale_fraction, train_cfg.student_nu,
        )
        fp32_robust_metrics = deployment_point_metrics(
            predict_fp32_raw(ticket_result.model, robust_val.fusion),
            robust_val, delay_max_ns, train_cfg.min_scale_fraction, train_cfg.student_nu,
        )
        candidate_bundle = calibrate_and_quantize(
            ESP32ExportNet(ticket_result.model).eval(),
            calibration_x,
            delay_max_ns=delay_max_ns,
            min_scale_fraction=train_cfg.min_scale_fraction,
        )
        int8_clean_raw = predict_int8_raw(candidate_bundle, val.fusion)
        int8_clean_metrics = deployment_point_metrics(
            int8_clean_raw,
            val, delay_max_ns, train_cfg.min_scale_fraction, train_cfg.student_nu,
        )
        int8_robust_metrics = deployment_point_metrics(
            predict_int8_raw(candidate_bundle, robust_val.fusion),
            robust_val, delay_max_ns, train_cfg.min_scale_fraction, train_cfg.student_nu,
        )

        fp32_tracking_metrics = deployment_tracking_metrics(
            fp32_clean_raw, val, data, val_obs.true_xy, val_obs.time_s, tracking_guard_pf,
            seed=tracking_guard_seed, delay_max_ns=delay_max_ns,
            min_scale_fraction=train_cfg.min_scale_fraction, student_nu=train_cfg.student_nu,
        )
        int8_tracking_metrics = deployment_tracking_metrics(
            int8_clean_raw, val, data, val_obs.true_xy, val_obs.time_s, tracking_guard_pf,
            seed=tracking_guard_seed, delay_max_ns=delay_max_ns,
            min_scale_fraction=train_cfg.min_scale_fraction, student_nu=train_cfg.student_nu,
        )

        fp32_guard_met = (
            float(fp32_clean_metrics["tof_mae_ns"]) <= quality_limits["mae_limit_ns"]
            and float(fp32_robust_metrics["tof_mae_ns"]) <= quality_limits["robust_mae_limit_ns"]
            and float(fp32_robust_metrics["nll"]) <= quality_limits["robust_nll_limit"]
            and float(fp32_robust_metrics["outlier_bce"]) <= quality_limits["robust_outlier_bce_limit"]
            and float(fp32_tracking_metrics["tracking_rmse_cm"]) <= quality_limits["tracking_rmse_limit_cm"]
        )
        int8_guard_met = (
            float(int8_clean_metrics["tof_mae_ns"])
            <= float(fp32_clean_metrics["tof_mae_ns"]) + quality_limits["max_int8_mae_increase_ns"]
            and float(int8_robust_metrics["tof_mae_ns"])
            <= float(fp32_robust_metrics["tof_mae_ns"]) + quality_limits["max_int8_robust_mae_increase_ns"]
            and float(int8_robust_metrics["nll"])
            <= float(fp32_robust_metrics["nll"]) + quality_limits["max_int8_nll_increase"]
            and float(int8_robust_metrics["outlier_bce"])
            <= float(fp32_robust_metrics["outlier_bce"]) + quality_limits["max_int8_outlier_bce_increase"]
            and float(int8_tracking_metrics["tracking_rmse_cm"])
            <= float(fp32_tracking_metrics["tracking_rmse_cm"]) + quality_limits["max_int8_tracking_rmse_increase_cm"]
        )
        meets_guard = fp32_guard_met and int8_guard_met
        save_student_checkpoint(
            candidate_path,
            ticket_result,
            input_length,
            delay_max_ns,
            train_cfg,
            {
                "role": "structured-rewound-lth-ticket",
                "candidate_index": idx,
                "selected_channels": selection_dict,
                "accuracy_guard_met": meets_guard,
                "fp32_guard_met": fp32_guard_met,
                "int8_guard_met": int8_guard_met,
                "fp32_clean_metrics": fp32_clean_metrics,
                "fp32_robust_metrics": fp32_robust_metrics,
                "int8_clean_metrics": int8_clean_metrics,
                "int8_robust_metrics": int8_robust_metrics,
                "fp32_tracking_guard_metrics": fp32_tracking_metrics,
                "int8_tracking_guard_metrics": int8_tracking_metrics,
                "teacher": teacher_info,
            },
        )
        record = {
            "candidate_index": idx,
            "arch": asdict(ticket_arch),
            "parameters": sum(p.numel() for p in ticket_result.model.parameters()),
            "fp32_clean": fp32_clean_metrics,
            "fp32_robust": fp32_robust_metrics,
            "int8_clean": int8_clean_metrics,
            "int8_robust": int8_robust_metrics,
            "fp32_tracking_guard": fp32_tracking_metrics,
            "int8_tracking_guard": int8_tracking_metrics,
            "fp32_guard_met": fp32_guard_met,
            "int8_guard_met": int8_guard_met,
            "accuracy_guard_met": meets_guard,
            "checkpoint": str(candidate_path.relative_to(ROOT)),
            "selected_channels": selection_dict,
        }
        search_records.append(record)
        if best_ticket is None or ticket_result.validation_mae_ns < best_ticket[0].validation_mae_ns:
            best_ticket = (ticket_result, ticket_arch, selection, candidate_path)
        if meets_guard:
            selected_result = ticket_result
            selected_ticket = ticket_arch
            selected_selection = selection
            selected_path = candidate_path
            break

    require_guard = bool(lth_cfg.get("require_accuracy_guard", True))
    if selected_result is None:
        assert best_ticket is not None
        if require_guard:
            partial_report = {
                "status": "lth_accuracy_guard_failed",
                "reference_supernet_mae_ns": super_result.validation_mae_ns,
                "quality_limits": quality_limits,
                "ticket_search": search_records,
                "message": (
                    "No structured LTH candidate preserved the requested accuracy. "
                    "No deployment export was produced; enlarge the last candidate or relax the explicit guard."
                ),
            }
            (output_dir / "pipeline_report.json").write_text(json.dumps(partial_report, indent=2), encoding="utf-8")
            raise RuntimeError(partial_report["message"])
        selected_result, selected_ticket, selected_selection, selected_path = best_ticket

    assert selected_result is not None
    assert selected_ticket is not None
    assert selected_selection is not None
    assert selected_path is not None

    # 3) Scientific control only: same exact selected architecture with fresh
    # initialization. It is reported, but never allowed to replace the LTH model
    # in best_student.pt. This guarantees export uses Lottery Ticket Hypothesis.
    torch.manual_seed(seed + 500)
    control = ESP32StudentNet(selected_ticket, train_cfg.min_scale_fraction)
    control_result = train_student(
        control,
        train_x,
        train_y,
        train_corruption,
        val_x,
        val_y,
        val_corruption,
        delay_max_ns,
        train_cfg,
        epochs=train_cfg.epochs_control,
        seed=seed + 500,
        teacher_mean=teacher_mean,
        teacher_scale=teacher_scale,
    )
    save_student_checkpoint(
        checkpoint_dir / "random_compact_control.pt",
        control_result,
        input_length,
        delay_max_ns,
        train_cfg,
        {"role": "random-compact-control-only", "teacher": teacher_info},
    )

    best_path = checkpoint_dir / "best_student.pt"
    shutil.copy2(selected_path, best_path)

    # 4) Export the selected LTH ticket using the exact representative calibration
    # set already used by the candidate-level INT8 quality guard.
    export_report = export_checkpoint(
        best_path,
        calibration_x,
        output_dir / "export",
        target=str(export_cfg.get("target", "esp32s3")),
        export_onnx_file=bool(args.onnx or export_cfg.get("onnx", False)),
        export_espdl_file=bool(args.espdl or export_cfg.get("espdl", False)),
    )
    export_report["runtime_constants"] = export_preprocess_and_geometry(
        data, input_length, output_dir / "export"
    )
    export_report["core_static_deployment_bytes"] = int(
        export_report["weight_blob_bytes"]
        + export_report["decode_lut_bytes"]
        + export_report["runtime_constants"]["background_blob_bytes"]
    )
    (output_dir / "export" / "export_report.json").write_text(
        json.dumps(export_report, indent=2), encoding="utf-8"
    )

    # 5) The deployment artifact is judged on the actual held-out tracking task,
    # not only on flattened ToF samples. This also quantifies INT8 degradation.
    deployment_evaluation = _evaluate_deployment_scenarios(
        cfg=cfg,
        data=data,
        test_idx=test_idx,
        input_length=input_length,
        seed=seed,
        selected_model=selected_result.model,
        control_model=control_result.model,
        calibration_x=calibration_x,
        delay_max_ns=delay_max_ns,
        training_cfg=train_cfg,
        output_dir=output_dir,
    )

    selected_record = next(
        record for record in search_records
        if record["checkpoint"] == str(selected_path.relative_to(ROOT))
    )
    selected_selection_dict = {
        "c1": selected_selection.c1.tolist(),
        "c2": selected_selection.c2.tolist(),
        "c3": selected_selection.c3.tolist(),
        "hidden": selected_selection.hidden.tolist(),
    }
    report = {
        "status": "ok",
        "case": case,
        "seed": seed,
        "input_length": input_length,
        "data": data_info,
        "teacher": teacher_info,
        "lth_policy": {
            "export_is_always_lth": True,
            "progressive_smallest_first": True,
            "bn_weighted_dependency_aware_channel_ranking": True,
            "fp32_and_int8_guard_required": True,
            "reference": "uncompressed_supernet_clean_and_corrupted_validation",
            **quality_limits,
            "accuracy_guard_met": bool(selected_record["accuracy_guard_met"]),
        },
        "supernet": {
            "arch": asdict(super_arch),
            "parameters": sum(p.numel() for p in super_result.model.parameters()),
            "training_validation_mae_ns": super_result.validation_mae_ns,
            "clean_validation": super_clean_metrics,
            "robust_validation": super_robust_metrics,
            "tracking_guard_validation": super_tracking_metrics,
        },
        "ticket_search": search_records,
        "selected_for_deployment": {
            "type": "structured_rewound_lth_ticket",
            "arch": asdict(selected_ticket),
            "parameters": sum(p.numel() for p in selected_result.model.parameters()),
            "training_validation_mae_ns": selected_result.validation_mae_ns,
            "fp32_clean": selected_record["fp32_clean"],
            "fp32_robust": selected_record["fp32_robust"],
            "int8_clean": selected_record["int8_clean"],
            "int8_robust": selected_record["int8_robust"],
            "fp32_tracking_guard": selected_record["fp32_tracking_guard"],
            "int8_tracking_guard": selected_record["int8_tracking_guard"],
            "fp32_guard_met": selected_record["fp32_guard_met"],
            "int8_guard_met": selected_record["int8_guard_met"],
            "selected_channels": selected_selection_dict,
            "checkpoint": str(best_path.relative_to(ROOT)),
        },
        "random_compact_control": {
            "role": "scientific_control_only_not_exported",
            "arch": asdict(selected_ticket),
            "parameters": sum(p.numel() for p in control_result.model.parameters()),
            "mae_ns": control_result.validation_mae_ns,
            "nll": control_result.validation_nll,
            "outlier_bce": control_result.validation_outlier_bce,
            "lth_beats_control_on_mae": selected_result.validation_mae_ns <= control_result.validation_mae_ns,
        },
        "deployment_evaluation": deployment_evaluation,
        "export": export_report,
    }
    (output_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
