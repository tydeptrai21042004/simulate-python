from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t_distribution
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig
from .data import PreparedInputs, UWBData, prepare_inputs, subset_observations
from .models import PaperResidualCNN, UncertaintyFusionNet
from .simulation import augment_training_observations
from .utils import count_parameters, ensure_dir, resolve_device, set_seed


@dataclass
class TrainedModel:
    name: str
    model: nn.Module
    input_kind: str
    delay_max_ns: float
    delay_start_ns: float
    delay_step_ns: float
    input_length: int
    global_scale_ns: float
    global_location_ns: float
    global_nu: float
    scale_multiplier: float
    parameter_count: int
    validation_mae_ns: float
    validation_nll: float
    training_seconds: float


def student_t_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    nu: float = 4.0,
) -> torch.Tensor:
    z2 = ((target - mean) / scale) ** 2
    constant = (
        torch.lgamma(torch.tensor(nu / 2.0, device=target.device, dtype=target.dtype))
        - torch.lgamma(torch.tensor((nu + 1.0) / 2.0, device=target.device, dtype=target.dtype))
        + 0.5 * torch.log(torch.tensor(nu * np.pi, device=target.device, dtype=target.dtype))
    )
    return constant + torch.log(scale) + 0.5 * (nu + 1.0) * torch.log1p(z2 / nu)


def proposed_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, nu: float) -> torch.Tensor:
    main = student_t_nll(target, outputs["mean_fraction"], outputs["scale_fraction"], nu).mean()
    cir = student_t_nll(target, outputs["cir_mean_fraction"], outputs["cir_scale_fraction"], nu).mean()
    var = student_t_nll(target, outputs["var_mean_fraction"], outputs["var_scale_fraction"], nu).mean()
    location = torch.nn.functional.smooth_l1_loss(outputs["mean_fraction"], target, beta=0.02)
    aux_location = 0.5 * (
        torch.nn.functional.smooth_l1_loss(outputs["cir_mean_fraction"], target, beta=0.03)
        + torch.nn.functional.smooth_l1_loss(outputs["var_mean_fraction"], target, beta=0.03)
    )
    reliability = torch.exp(-outputs["cir_scale_fraction"].detach() - outputs["var_scale_fraction"].detach())
    consistency = (reliability * torch.abs(outputs["cir_mean_fraction"] - outputs["var_mean_fraction"])).mean()
    scale_regularizer = outputs["scale_fraction"].mean()
    return (
        0.5 * main
        + 4.0 * location
        + 0.08 * (cir + var)
        + 0.50 * aux_location
        + 0.02 * consistency
        + 0.01 * scale_regularizer
    )


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=max(1, min(int(batch_size), len(dataset))),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _index_to_delay(
    index: np.ndarray,
    delay_start_ns: float,
    delay_step_ns: float,
    indexing_mode: str,
) -> np.ndarray:
    # MATLAB labels are one-based. The repository's deployment formula uses
    # start + step*index (an extra one-bin shift); corrected uses index-1.
    offset = 0.0 if indexing_mode == "repository" else 1.0
    return delay_start_ns + delay_step_ns * (np.asarray(index) - offset)


def _evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    input_kind: str,
    delay_max_ns: float,
    delay_start_ns: float,
    delay_step_ns: float,
    paper_indexing_mode: str,
    device: torch.device,
    nu: float,
    batch_size: int,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    means: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    loader = _make_loader(x, y, batch_size, False, 0)
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            if input_kind == "fusion":
                mean_native = out["mean_fraction"]
                scale_native = out["scale_fraction"]
                loss = student_t_nll(yb, mean_native, scale_native, nu).mean()
            else:
                mean_native = out["mean_index"]
                scale_native = torch.full_like(mean_native, 1.0)
                loss = torch.mean(torch.abs(mean_native - yb)) * delay_step_ns
            losses.append(float(loss.cpu()))
            means.append(mean_native.cpu().numpy())
            scales.append(scale_native.cpu().numpy())

    mean_native_np = np.concatenate(means)
    scale_native_np = np.concatenate(scales)
    if input_kind == "fusion":
        pred_delay = mean_native_np * delay_max_ns
        scale_ns = scale_native_np * delay_max_ns
        true_delay = y * delay_max_ns
    else:
        pred_delay = _index_to_delay(
            mean_native_np, delay_start_ns, delay_step_ns, paper_indexing_mode
        )
        # Validation residuals in the official code are step*(Y-Y_pred), so
        # use the physically equivalent corrected target location here.
        true_delay = delay_start_ns + delay_step_ns * (y - 1.0)
        scale_ns = np.full_like(pred_delay, delay_step_ns, dtype=np.float64)
    val_mae = float(np.mean(np.abs(pred_delay - true_delay)))
    return val_mae, float(np.mean(losses)), pred_delay, scale_ns, true_delay


def _fit_student_t(residual: np.ndarray) -> tuple[float, float, float]:
    residual = np.asarray(residual, dtype=np.float64)
    robust_scale = max(0.15, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
    try:
        nu, loc, scale = student_t_distribution.fit(residual)
        if not np.isfinite(nu + loc + scale) or scale <= 0:
            raise ValueError("invalid fit")
        return float(np.clip(nu, 2.1, 100.0)), float(loc), float(max(scale, 0.05))
    except Exception:
        return 4.0, float(np.median(residual)), robust_scale


def train_one(
    name: str,
    model: nn.Module,
    input_kind: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    cfg: ExperimentConfig,
    seed: int,
    checkpoint_path: Path | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
) -> TrainedModel:
    set_seed(seed)
    device = resolve_device(cfg.device)
    model = model.to(device)
    lr = cfg.model.learning_rate if learning_rate is None else learning_rate
    if cfg.model.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif cfg.model.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.model.weight_decay)
    else:
        raise ValueError("model.optimizer must be 'adam' or 'adamw'")
    actual_batch = cfg.model.batch_size if batch_size is None else batch_size
    train_loader = _make_loader(train_x, train_y, actual_batch, True, seed)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0
    started = time.perf_counter()

    for _epoch in range(cfg.model.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            if input_kind == "fusion":
                loss = proposed_loss(out, yb, cfg.model.student_nu)
            else:
                # MATLAB regressionLayer optimizes mean squared error.
                loss = torch.nn.functional.mse_loss(out["mean_index"], yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        val_mae, val_nll, _, _, _ = _evaluate(
            model,
            val_x,
            val_y,
            input_kind,
            float(cfg._delay_max_ns),
            float(cfg._delay_start_ns),
            float(cfg._delay_step_ns),
            cfg.model.paper_indexing_mode,
            device,
            cfg.model.student_nu,
            actual_batch,
        )
        score = val_nll if input_kind == "fusion" else val_mae
        if score < best_val - 1e-5:
            best_val = score
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if cfg.model.early_stopping and bad_epochs >= cfg.model.patience:
                break

    if cfg.model.restore_best:
        model.load_state_dict(best_state)
    val_mae, val_nll, pred, pred_scale, true_delay = _evaluate(
        model,
        val_x,
        val_y,
        input_kind,
        float(cfg._delay_max_ns),
        float(cfg._delay_start_ns),
        float(cfg._delay_step_ns),
        cfg.model.paper_indexing_mode,
        device,
        cfg.model.student_nu,
        actual_batch,
    )
    residual = true_delay - pred
    fitted_nu, fitted_location, fitted_scale = _fit_student_t(residual)
    if input_kind == "fusion":
        ratio = np.median(np.abs(residual) / np.maximum(pred_scale, 1e-3))
        scale_multiplier = float(np.clip(ratio, 0.5, 4.0))
        global_scale = max(0.15, float(np.median(pred_scale) * scale_multiplier))
    else:
        scale_multiplier = 1.0
        global_scale = fitted_scale
    seconds = time.perf_counter() - started
    trained = TrainedModel(
        name=name,
        model=model.cpu(),
        input_kind=input_kind,
        delay_max_ns=float(cfg._delay_max_ns),
        delay_start_ns=float(cfg._delay_start_ns),
        delay_step_ns=float(cfg._delay_step_ns),
        input_length=int(cfg.model.input_length),
        global_scale_ns=global_scale,
        global_location_ns=fitted_location,
        global_nu=fitted_nu,
        scale_multiplier=scale_multiplier,
        parameter_count=count_parameters(model),
        validation_mae_ns=val_mae,
        validation_nll=val_nll,
        training_seconds=seconds,
    )
    if checkpoint_path is not None:
        ensure_dir(checkpoint_path.parent)
        torch.save(
            {
                "name": trained.name,
                "input_kind": trained.input_kind,
                "delay_max_ns": trained.delay_max_ns,
                "delay_start_ns": trained.delay_start_ns,
                "delay_step_ns": trained.delay_step_ns,
                "input_length": trained.input_length,
                "global_scale_ns": trained.global_scale_ns,
                "global_location_ns": trained.global_location_ns,
                "global_nu": trained.global_nu,
                "scale_multiplier": trained.scale_multiplier,
                "parameter_count": trained.parameter_count,
                "validation_mae_ns": trained.validation_mae_ns,
                "state_dict": trained.model.state_dict(),
            },
            checkpoint_path,
        )
    return trained


def _time_level_train_val_split(indices: np.ndarray, seed: int, val_fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(indices)
    val_count = max(1, int(round(len(indices) * val_fraction)))
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def _paper_batch_size(num_samples: int, cfg: ExperimentConfig) -> int:
    if cfg.model.paper_batch_fraction is not None:
        return max(1, int(np.floor(num_samples * float(cfg.model.paper_batch_fraction))))
    return cfg.model.batch_size


def train_models_for_case(
    data: UWBData,
    train_indices: np.ndarray,
    cfg: ExperimentConfig,
    seed: int,
    checkpoint_dir: str | Path,
    proposed_ablation: str = "full",
    include_proposed: bool = True,
) -> dict[str, TrainedModel]:
    cfg._delay_max_ns = float(data.delay_grid_ns[-1])  # type: ignore[attr-defined]
    cfg._delay_start_ns = float(data.delay_grid_ns[0])  # type: ignore[attr-defined]
    grid = np.linspace(cfg._delay_start_ns, cfg._delay_max_ns, cfg.model.input_length)
    cfg._delay_step_ns = float(grid[1] - grid[0]) if grid.size > 1 else 1.0  # type: ignore[attr-defined]

    train_core, val_idx = _time_level_train_val_split(
        train_indices, seed, val_fraction=cfg.model.paper_validation_fraction
    )
    clean_train = prepare_inputs(data, subset_observations(data, train_core), cfg.model.input_length)
    val = prepare_inputs(data, subset_observations(data, val_idx), cfg.model.input_length)

    # The official MATLAB scripts randomly split the concatenated link samples
    # 85/15. The main scientific protocol instead separates timestamps, which
    # prevents the same instant from appearing through another link in both sets.
    if cfg.model.paper_validation_mode == "sample":
        paper_all = prepare_inputs(data, subset_observations(data, train_indices), cfg.model.input_length)
        rng = np.random.default_rng(seed + 404)
        permutation = rng.permutation(paper_all.target_index.size)
        val_count = max(1, int(round(permutation.size * cfg.model.paper_validation_fraction)))
        paper_val_idx = permutation[-val_count:]
        paper_train_idx = permutation[:-val_count]
        paper_cir_train = paper_all.paper_cir[paper_train_idx]
        paper_var_train = paper_all.paper_var[paper_train_idx]
        paper_target_train = paper_all.target_index[paper_train_idx]
        paper_cir_val = paper_all.paper_cir[paper_val_idx]
        paper_var_val = paper_all.paper_var[paper_val_idx]
        paper_target_val = paper_all.target_index[paper_val_idx]
    elif cfg.model.paper_validation_mode == "time":
        paper_cir_train = clean_train.paper_cir
        paper_var_train = clean_train.paper_var
        paper_target_train = clean_train.target_index
        paper_cir_val = val.paper_cir
        paper_var_val = val.paper_var
        paper_target_val = val.target_index
    else:
        raise ValueError("paper_validation_mode must be 'time' or 'sample'")

    augmented_obs = augment_training_observations(
        data,
        train_core,
        seed + 7000,
        probability=cfg.model.augmentation_probability,
    )
    augmented = prepare_inputs(data, augmented_obs, cfg.model.input_length)
    fusion_x = np.concatenate([clean_train.fusion, clean_train.fusion, augmented.fusion], axis=0)
    fusion_y = np.concatenate(
        [clean_train.target_fraction, clean_train.target_fraction, augmented.target_fraction], axis=0
    )
    out = Path(checkpoint_dir)

    cir_mean = torch.from_numpy(paper_cir_train.mean(axis=0, keepdims=True))
    var_mean = torch.from_numpy(paper_var_train.mean(axis=0, keepdims=True))
    paper_batch = _paper_batch_size(paper_cir_train.shape[0], cfg)

    models: dict[str, TrainedModel] = {}
    set_seed(seed + 101)
    paper_cir_model = PaperResidualCNN(cfg.model.input_length, input_mean=cir_mean)
    models["paper_cir"] = train_one(
        "Official CIR-CNN",
        paper_cir_model,
        "paper_cir",
        paper_cir_train,
        paper_target_train,
        paper_cir_val,
        paper_target_val,
        cfg,
        seed + 101,
        out / "paper_cir.pt",
        batch_size=paper_batch,
    )
    set_seed(seed + 202)
    paper_var_model = PaperResidualCNN(cfg.model.input_length, input_mean=var_mean)
    models["paper_var"] = train_one(
        "Official Variance-CNN",
        paper_var_model,
        "paper_var",
        paper_var_train,
        paper_target_train,
        paper_var_val,
        paper_target_val,
        cfg,
        seed + 202,
        out / "paper_var.pt",
        batch_size=paper_batch,
    )
    if include_proposed:
        set_seed(seed + 303)
        proposed_model = UncertaintyFusionNet(cfg.model.min_scale_fraction, ablation=proposed_ablation)
        models["proposed"] = train_one(
            f"U-FusePF ({proposed_ablation})",
            proposed_model,
            "fusion",
            fusion_x,
            fusion_y,
            val.fusion,
            val.target_fraction,
            cfg,
            seed + 303,
            out / f"proposed_{proposed_ablation}.pt",
            learning_rate=2.0 * cfg.model.learning_rate,
            batch_size=cfg.model.batch_size,
        )
    return models


def predict_delays(
    trained: TrainedModel,
    prepared: PreparedInputs,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], float]:
    device = resolve_device(cfg.device)
    model = trained.model.to(device).eval()
    x = {
        "paper_cir": prepared.paper_cir,
        "paper_var": prepared.paper_var,
        "fusion": prepared.fusion,
    }[trained.input_kind]
    dummy = np.zeros(x.shape[0], dtype=np.float32)
    loader = _make_loader(x, dummy, cfg.model.batch_size, False, 0)
    means: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    extras: dict[str, list[np.ndarray]] = {}
    started = time.perf_counter()
    with torch.no_grad():
        for xb, _ in loader:
            out = model(xb.to(device))
            if trained.input_kind == "fusion":
                means.append(out["mean_fraction"].cpu().numpy())
                scales.append(out["scale_fraction"].cpu().numpy())
            else:
                means.append(out["mean_index"].cpu().numpy())
                scales.append(np.full(xb.shape[0], trained.global_scale_ns, dtype=np.float32))
            for key in ("gate_cir", "gate_var", "cir_scale_fraction", "var_scale_fraction"):
                if key in out:
                    extras.setdefault(key, []).append(out[key].cpu().numpy())
    elapsed_ms_per_sample = 1000.0 * (time.perf_counter() - started) / max(1, x.shape[0])
    mean_native = np.concatenate(means)
    if trained.input_kind == "fusion":
        mean_ns = mean_native * trained.delay_max_ns
        scale_ns = np.maximum(
            np.concatenate(scales) * trained.delay_max_ns * trained.scale_multiplier, 0.05
        )
    else:
        mean_ns = _index_to_delay(
            mean_native,
            trained.delay_start_ns,
            trained.delay_step_ns,
            cfg.model.paper_indexing_mode,
        )
        scale_ns = np.maximum(np.concatenate(scales), 0.05)
    extra_np = {k: np.concatenate(v) for k, v in extras.items()}
    trained.model = model.cpu()
    return mean_ns, scale_ns, extra_np, elapsed_ms_per_sample
