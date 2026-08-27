#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from uwb_tracking.simulation import augment_training_observations
from uwb_tracking.esp32.exporter import export_checkpoint
from uwb_tracking.esp32.preprocess_export import export_preprocess_and_geometry
from uwb_tracking.esp32.model import ESP32Architecture, ESP32StudentNet, build_rewound_structured_ticket
from uwb_tracking.esp32.teacher import load_ensemble_teacher, teacher_targets
from uwb_tracking.esp32.training import (
    ESP32TrainingConfig,
    save_student_checkpoint,
    train_student,
)


def _split_train_val(indices: np.ndarray, seed: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(indices)
    n_val = max(1, int(round(len(indices) * fraction)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, select, and export the ESP32 U-Fuse student.")
    parser.add_argument("--config", default="configs/esp32s3.yaml")
    parser.add_argument("--no-teacher", action="store_true", help="Disable teacher distillation.")
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

    data = load_uwb_mat(ROOT / str(cfg["data_path"]))
    train_idx, _ = get_case_split(data.num_time, case)
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
    clean = prepare_inputs(data, clean_obs, input_length)
    val = prepare_inputs(data, val_obs, input_length)
    aug = prepare_inputs(data, aug_obs, input_length)

    train_x = np.concatenate([clean.fusion, clean.fusion, aug.fusion], axis=0).astype(np.float32)
    train_y = np.concatenate([clean.target_fraction, clean.target_fraction, aug.target_fraction], axis=0).astype(np.float32)
    train_corruption = np.concatenate([clean.corruption, clean.corruption, aug.corruption], axis=0).astype(np.float32)
    val_x = val.fusion.astype(np.float32)
    val_y = val.target_fraction.astype(np.float32)
    val_corruption = val.corruption.astype(np.float32)
    delay_max_ns = float(data.delay_grid_ns[-1])

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
    ticket_arch = _arch(cfg["ticket"])

    # 1) Train a wider supernet only to discover robust channel identities.
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
        {"role": "channel-discovery-supernet", "teacher": teacher_info},
    )

    # 2) Structured Lottery Ticket: select trained channels, rewind surviving
    # weights to initialization, then retrain only the physically compact net.
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
        seed=seed + 101,
        teacher_mean=teacher_mean,
        teacher_scale=teacher_scale,
    )
    selection_dict = {
        "c1": selection.c1.tolist(),
        "c2": selection.c2.tolist(),
        "c3": selection.c3.tolist(),
    }
    save_student_checkpoint(
        checkpoint_dir / "structured_ticket.pt",
        ticket_result,
        input_length,
        delay_max_ns,
        train_cfg,
        {"role": "structured-rewound-ticket", "selected_channels": selection_dict, "teacher": teacher_info},
    )

    # 3) Required control: same tiny architecture, fresh initialization.
    torch.manual_seed(seed + 102)
    control = ESP32StudentNet(ticket_arch, train_cfg.min_scale_fraction)
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
        seed=seed + 102,
        teacher_mean=teacher_mean,
        teacher_scale=teacher_scale,
    )
    save_student_checkpoint(
        checkpoint_dir / "random_compact_control.pt",
        control_result,
        input_length,
        delay_max_ns,
        train_cfg,
        {"role": "random-compact-control", "teacher": teacher_info},
    )

    # Deploy whichever physical compact network validates better. We still
    # report both, so LTH is never claimed beneficial without evidence.
    def score(result):
        return result.validation_mae_ns + 0.25 * result.validation_nll + 0.10 * result.validation_outlier_bce

    if score(ticket_result) <= score(control_result):
        winner_name = "structured_ticket"
        winner_result = ticket_result
        winner_src = checkpoint_dir / "structured_ticket.pt"
    else:
        winner_name = "random_compact_control"
        winner_result = control_result
        winner_src = checkpoint_dir / "random_compact_control.pt"
    best_path = checkpoint_dir / "best_student.pt"
    shutil.copy2(winner_src, best_path)

    # 4) Export raw INT8 immediately. ONNX/.espdl are optional because ESP-PPQ
    # is a separate Espressif-side dependency.
    export_cfg = cfg.get("export", {})
    calib_count = min(int(export_cfg.get("calibration_samples", 256)), len(train_x))
    calibration_x = train_x[:calib_count]
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

    report = {
        "case": case,
        "seed": seed,
        "input_length": input_length,
        "teacher": teacher_info,
        "supernet": {
            "arch": asdict(super_arch),
            "parameters": sum(p.numel() for p in super_result.model.parameters()),
            "mae_ns": super_result.validation_mae_ns,
            "nll": super_result.validation_nll,
        },
        "structured_ticket": {
            "arch": asdict(ticket_arch),
            "parameters": sum(p.numel() for p in ticket_result.model.parameters()),
            "mae_ns": ticket_result.validation_mae_ns,
            "nll": ticket_result.validation_nll,
            "outlier_bce": ticket_result.validation_outlier_bce,
            "selected_channels": selection_dict,
        },
        "random_compact_control": {
            "parameters": sum(p.numel() for p in control_result.model.parameters()),
            "mae_ns": control_result.validation_mae_ns,
            "nll": control_result.validation_nll,
            "outlier_bce": control_result.validation_outlier_bce,
        },
        "selected_for_deployment": winner_name,
        "selected_validation_mae_ns": winner_result.validation_mae_ns,
        "export": export_report,
    }
    (output_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
