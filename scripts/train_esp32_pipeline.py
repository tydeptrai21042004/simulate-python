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
from uwb_tracking.official_data import ensure_official_standard_dataset
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


def _quality_guard_limits(super_result, raw: dict) -> dict[str, float]:
    rel = float(raw.get("max_relative_mae_loss", 0.08))
    absolute = float(raw.get("max_absolute_mae_loss_ns", 0.03))
    max_nll_increase = float(raw.get("max_nll_increase", 0.25))
    max_bce_increase = float(raw.get("max_outlier_bce_increase", 0.10))
    if min(rel, absolute, max_nll_increase, max_bce_increase) < 0:
        raise ValueError("LTH quality tolerances must be >= 0")
    relative_limit = super_result.validation_mae_ns * (1.0 + rel)
    absolute_limit = super_result.validation_mae_ns + absolute
    return {
        "mae_limit_ns": min(relative_limit, absolute_limit),
        "max_relative_mae_loss": rel,
        "max_absolute_mae_loss_ns": absolute,
        "nll_limit": super_result.validation_nll + max_nll_increase,
        "max_nll_increase": max_nll_increase,
        "outlier_bce_limit": super_result.validation_outlier_bce + max_bce_increase,
        "max_outlier_bce_increase": max_bce_increase,
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
    quality_limits = _quality_guard_limits(super_result, lth_cfg)
    mae_limit = quality_limits["mae_limit_ns"]
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
        meets_guard = (
            ticket_result.validation_mae_ns <= quality_limits["mae_limit_ns"]
            and ticket_result.validation_nll <= quality_limits["nll_limit"]
            and ticket_result.validation_outlier_bce <= quality_limits["outlier_bce_limit"]
        )
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
                "teacher": teacher_info,
            },
        )
        record = {
            "candidate_index": idx,
            "arch": asdict(ticket_arch),
            "parameters": sum(p.numel() for p in ticket_result.model.parameters()),
            "mae_ns": ticket_result.validation_mae_ns,
            "nll": ticket_result.validation_nll,
            "outlier_bce": ticket_result.validation_outlier_bce,
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

    # 4) Representative calibration and export of the selected LTH ticket only.
    export_cfg = cfg.get("export", {})
    calibration_x = _balanced_calibration_subset(
        train_x,
        train_corruption,
        int(export_cfg.get("calibration_samples", 256)),
        seed + 9000,
    )
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
            "reference": "uncompressed_supernet_validation_mae",
            **quality_limits,
            "accuracy_guard_met": (
                selected_result.validation_mae_ns <= quality_limits["mae_limit_ns"]
                and selected_result.validation_nll <= quality_limits["nll_limit"]
                and selected_result.validation_outlier_bce <= quality_limits["outlier_bce_limit"]
            ),
        },
        "supernet": {
            "arch": asdict(super_arch),
            "parameters": sum(p.numel() for p in super_result.model.parameters()),
            "mae_ns": super_result.validation_mae_ns,
            "nll": super_result.validation_nll,
            "outlier_bce": super_result.validation_outlier_bce,
        },
        "ticket_search": search_records,
        "selected_for_deployment": {
            "type": "structured_rewound_lth_ticket",
            "arch": asdict(selected_ticket),
            "parameters": sum(p.numel() for p in selected_result.model.parameters()),
            "mae_ns": selected_result.validation_mae_ns,
            "nll": selected_result.validation_nll,
            "outlier_bce": selected_result.validation_outlier_bce,
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
        "export": export_report,
    }
    (output_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
