from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import ESP32Architecture, ESP32StudentNet, build_rewound_structured_ticket


@dataclass
class ESP32TrainingConfig:
    epochs_supernet: int = 30
    epochs_ticket: int = 40
    epochs_control: int = 40
    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 10
    student_nu: float = 4.0
    min_scale_fraction: float = 0.004
    distill_mean_weight: float = 1.0
    distill_scale_weight: float = 0.25
    corruption_weight: float = 0.15
    seed: int = 11
    device: str = "cpu"


@dataclass
class TrainResult:
    model: ESP32StudentNet
    best_loss: float
    validation_mae_ns: float
    validation_nll: float
    validation_outlier_bce: float
    epochs_ran: int
    training_seconds: float


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def student_t_nll(target: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, nu: float) -> torch.Tensor:
    nu_t = torch.as_tensor(float(nu), device=target.device, dtype=target.dtype)
    constant = (
        torch.lgamma(nu_t / 2.0)
        - torch.lgamma((nu_t + 1.0) / 2.0)
        + 0.5 * torch.log(nu_t * math.pi)
    )
    z2 = ((target - mean) / scale) ** 2
    return constant + torch.log(scale) + 0.5 * (nu_t + 1.0) * torch.log1p(z2 / nu_t)


def _loss(
    out: dict[str, torch.Tensor],
    y: torch.Tensor,
    corruption: torch.Tensor,
    cfg: ESP32TrainingConfig,
    teacher_mean: torch.Tensor | None,
    teacher_scale: torch.Tensor | None,
) -> torch.Tensor:
    nll = student_t_nll(y, out["mean_fraction"], out["scale_fraction"], cfg.student_nu).mean()
    location = torch.nn.functional.smooth_l1_loss(out["mean_fraction"], y, beta=0.02)
    scale_reg = out["scale_fraction"].mean()

    positives = corruption.sum()
    negatives = corruption.numel() - positives
    pos_weight = (negatives / torch.clamp(positives, min=1.0)).clamp(1.0, 8.0)
    quality = torch.nn.functional.binary_cross_entropy_with_logits(
        out["outlier_logit"], corruption, pos_weight=pos_weight
    )

    loss = 0.55 * nll + 4.0 * location + 0.01 * scale_reg + cfg.corruption_weight * quality
    if teacher_mean is not None:
        loss = loss + cfg.distill_mean_weight * torch.nn.functional.smooth_l1_loss(
            out["mean_fraction"], teacher_mean, beta=0.02
        )
    if teacher_scale is not None:
        # Log-domain matching is stable across small/large predicted scales.
        loss = loss + cfg.distill_scale_weight * torch.nn.functional.smooth_l1_loss(
            torch.log(out["scale_fraction"] + 1e-6),
            torch.log(teacher_scale + 1e-6),
            beta=0.10,
        )
    return loss


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    corruption: np.ndarray,
    teacher_mean: np.ndarray | None,
    teacher_scale: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    tensors: list[torch.Tensor] = [
        torch.from_numpy(x.astype(np.float32, copy=False)),
        torch.from_numpy(y.astype(np.float32, copy=False)),
        torch.from_numpy(corruption.astype(np.float32, copy=False)),
    ]
    if teacher_mean is not None and teacher_scale is not None:
        tensors.extend(
            [
                torch.from_numpy(teacher_mean.astype(np.float32, copy=False)),
                torch.from_numpy(teacher_scale.astype(np.float32, copy=False)),
            ]
        )
    ds = TensorDataset(*tensors)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle, generator=gen, num_workers=0)


@torch.inference_mode()
def evaluate_student(
    model: ESP32StudentNet,
    x: np.ndarray,
    y: np.ndarray,
    corruption: np.ndarray,
    delay_max_ns: float,
    cfg: ESP32TrainingConfig,
) -> tuple[float, float, float]:
    device = torch.device(cfg.device)
    model = model.to(device).eval()
    loader = _loader(x, y, corruption, None, None, cfg.batch_size, False, 0)
    abs_errors: list[np.ndarray] = []
    nlls: list[float] = []
    bces: list[float] = []
    for batch in loader:
        xb, yb, cb = (t.to(device) for t in batch[:3])
        out = model(xb)
        abs_errors.append((torch.abs(out["mean_fraction"] - yb) * delay_max_ns).cpu().numpy())
        nlls.append(float(student_t_nll(yb, out["mean_fraction"], out["scale_fraction"], cfg.student_nu).mean().cpu()))
        bces.append(float(torch.nn.functional.binary_cross_entropy_with_logits(out["outlier_logit"], cb).cpu()))
    model.cpu()
    return float(np.mean(np.concatenate(abs_errors))), float(np.mean(nlls)), float(np.mean(bces))


def train_student(
    model: ESP32StudentNet,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_corruption: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_corruption: np.ndarray,
    delay_max_ns: float,
    cfg: ESP32TrainingConfig,
    epochs: int,
    seed: int,
    teacher_mean: np.ndarray | None = None,
    teacher_scale: np.ndarray | None = None,
) -> TrainResult:
    train_x = np.asarray(train_x, dtype=np.float32)
    val_x = np.asarray(val_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32).reshape(-1)
    val_y = np.asarray(val_y, dtype=np.float32).reshape(-1)
    train_corruption = np.asarray(train_corruption, dtype=np.float32).reshape(-1)
    val_corruption = np.asarray(val_corruption, dtype=np.float32).reshape(-1)
    if train_x.ndim != 3 or train_x.shape[1] != 6 or val_x.ndim != 3 or val_x.shape[1] != 6:
        raise ValueError("train_x and val_x must have shape [samples, 6, input_length]")
    if train_x.shape[0] < 1 or val_x.shape[0] < 1:
        raise ValueError("training and validation sets must be non-empty")
    if not (train_x.shape[0] == train_y.size == train_corruption.size):
        raise ValueError("training arrays have inconsistent sample counts")
    if not (val_x.shape[0] == val_y.size == val_corruption.size):
        raise ValueError("validation arrays have inconsistent sample counts")
    if teacher_mean is not None or teacher_scale is not None:
        if teacher_mean is None or teacher_scale is None:
            raise ValueError("teacher_mean and teacher_scale must be provided together")
        teacher_mean = np.asarray(teacher_mean, dtype=np.float32).reshape(-1)
        teacher_scale = np.asarray(teacher_scale, dtype=np.float32).reshape(-1)
        if teacher_mean.size != train_x.shape[0] or teacher_scale.size != train_x.shape[0]:
            raise ValueError("teacher targets must match the training sample count")
    if float(delay_max_ns) <= 0:
        raise ValueError("delay_max_ns must be > 0")
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1")
    set_seed(seed)
    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loader = _loader(
        train_x,
        train_y,
        train_corruption,
        teacher_mean,
        teacher_scale,
        cfg.batch_size,
        True,
        seed,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    bad_epochs = 0
    started = time.perf_counter()
    epochs_ran = 0

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            xb, yb, cb = (batch[i].to(device) for i in range(3))
            tm = batch[3].to(device) if len(batch) > 3 else None
            ts = batch[4].to(device) if len(batch) > 4 else None
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = _loss(out, yb, cb, cfg, tm, ts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_mae, val_nll, val_bce = evaluate_student(
            model, val_x, val_y, val_corruption, delay_max_ns, cfg
        )
        # Checkpoint selection is MAE-first because deployment quality is dominated by ToF accuracy.
        # NLL/BCE remain reported diagnostics and training losses, but negative NLL values must not
        # accidentally make a worse-ToF checkpoint look better.
        score = val_mae
        epochs_ran = epoch + 1
        if score < best_loss - 1e-5:
            best_loss = score
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    model.load_state_dict(best_state)
    val_mae, val_nll, val_bce = evaluate_student(
        model, val_x, val_y, val_corruption, delay_max_ns, cfg
    )
    return TrainResult(
        model=model.cpu(),
        best_loss=best_loss,
        validation_mae_ns=val_mae,
        validation_nll=val_nll,
        validation_outlier_bce=val_bce,
        epochs_ran=epochs_ran,
        training_seconds=time.perf_counter() - started,
    )


def save_student_checkpoint(
    path: str | Path,
    result: TrainResult,
    input_length: int,
    delay_max_ns: float,
    training_cfg: ESP32TrainingConfig,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "uwb-esp32-student-v1",
        "arch": {
            "channels": list(result.model.arch.channels),
            "hidden": result.model.arch.hidden,
        },
        "min_scale_fraction": result.model.min_scale_fraction,
        "input_length": int(input_length),
        "delay_max_ns": float(delay_max_ns),
        "validation_mae_ns": result.validation_mae_ns,
        "validation_nll": result.validation_nll,
        "validation_outlier_bce": result.validation_outlier_bce,
        "training": asdict(training_cfg),
        "state_dict": result.model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)



def is_structured_lth_checkpoint(meta: dict) -> bool:
    """Return True only for checkpoints produced by structured rewound ticket training."""

    extra = meta.get("extra", {}) or {}
    role = str(extra.get("role", "")).lower()
    return "structured" in role and ("ticket" in role or "lth" in role)

def load_student_checkpoint(path: str | Path) -> tuple[ESP32StudentNet, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    arch_raw = payload["arch"]
    arch = ESP32Architecture(tuple(int(v) for v in arch_raw["channels"]), int(arch_raw["hidden"]))
    model = ESP32StudentNet(arch, float(payload.get("min_scale_fraction", 0.004)))
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload
