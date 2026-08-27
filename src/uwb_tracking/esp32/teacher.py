from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data import ObservationSet, UWBData, prepare_inputs
from ..models import PaperResidualCNN, UncertaintyFusionNet


@dataclass
class EnsembleTeacher:
    paper_cir: PaperResidualCNN
    paper_var: PaperResidualCNN
    proposed: UncertaintyFusionNet
    input_length: int
    delay_start_ns: float
    delay_step_ns: float
    delay_max_ns: float
    cir_validation_mae_ns: float
    var_validation_mae_ns: float
    proposed_scale_multiplier: float


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_ensemble_teacher(checkpoint_dir: str | Path) -> EnsembleTeacher:
    directory = Path(checkpoint_dir)
    cir_ckpt = _load_checkpoint(directory / "paper_cir.pt")
    var_ckpt = _load_checkpoint(directory / "paper_var.pt")
    proposed_path = directory / "proposed_full.pt"
    if not proposed_path.exists():
        candidates = sorted(directory.glob("proposed_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No proposed checkpoint found in {directory}")
        proposed_path = candidates[0]
    prop_ckpt = _load_checkpoint(proposed_path)

    lengths = {int(cir_ckpt["input_length"]), int(var_ckpt["input_length"]), int(prop_ckpt["input_length"])}
    if len(lengths) != 1:
        raise ValueError(f"Teacher checkpoints use different input lengths: {sorted(lengths)}")
    input_length = lengths.pop()

    cir = PaperResidualCNN(input_length)
    cir.load_state_dict(cir_ckpt["state_dict"])
    var = PaperResidualCNN(input_length)
    var.load_state_dict(var_ckpt["state_dict"])
    proposed = UncertaintyFusionNet()
    proposed.load_state_dict(prop_ckpt["state_dict"])

    return EnsembleTeacher(
        paper_cir=cir.eval(),
        paper_var=var.eval(),
        proposed=proposed.eval(),
        input_length=input_length,
        delay_start_ns=float(cir_ckpt["delay_start_ns"]),
        delay_step_ns=float(cir_ckpt["delay_step_ns"]),
        delay_max_ns=float(prop_ckpt["delay_max_ns"]),
        cir_validation_mae_ns=float(cir_ckpt["validation_mae_ns"]),
        var_validation_mae_ns=float(var_ckpt["validation_mae_ns"]),
        proposed_scale_multiplier=float(prop_ckpt.get("scale_multiplier", 1.0)),
    )


def _run_model(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> list[dict[str, np.ndarray]]:
    outputs: list[dict[str, np.ndarray]] = []
    model = model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            out = model(xb)
            outputs.append({k: v.detach().cpu().numpy() for k, v in out.items()})
    model.cpu()
    return outputs


def _concat(chunks: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.concatenate([chunk[key] for chunk in chunks], axis=0)


def teacher_targets(
    teacher: EnsembleTeacher,
    data: UWBData,
    observations: ObservationSet,
    batch_size: int = 128,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return fused teacher mean/scale fractions for distillation.

    This reproduces the deployment fusion used by the research pipeline:
    paper CIR + paper variance experts, gated by the local U-Fuse reliability
    and validation-error priors, plus expert disagreement in the uncertainty.
    """

    prepared = prepare_inputs(data, observations, teacher.input_length)
    dev = torch.device(device)
    cir_chunks = _run_model(teacher.paper_cir, prepared.paper_cir, batch_size, dev)
    var_chunks = _run_model(teacher.paper_var, prepared.paper_var, batch_size, dev)
    prop_chunks = _run_model(teacher.proposed, prepared.fusion, batch_size, dev)

    cir_index = _concat(cir_chunks, "mean_index")
    var_index = _concat(var_chunks, "mean_index")
    cir_delay = teacher.delay_start_ns + teacher.delay_step_ns * (cir_index - 1.0)
    var_delay = teacher.delay_start_ns + teacher.delay_step_ns * (var_index - 1.0)

    gate_cir = _concat(prop_chunks, "gate_cir")
    gate_var = _concat(prop_chunks, "gate_var")
    prior_cir = 1.0 / (teacher.cir_validation_mae_ns**2 + 1e-4)
    prior_var = 1.0 / (teacher.var_validation_mae_ns**2 + 1e-4)
    wc = gate_cir * prior_cir
    wv = gate_var * prior_var
    denom = np.maximum(wc + wv, 1e-8)
    wc = wc / denom
    wv = wv / denom
    fused_delay = wc * cir_delay + wv * var_delay

    local_scale = (
        _concat(prop_chunks, "scale_fraction")
        * teacher.delay_max_ns
        * teacher.proposed_scale_multiplier
    )
    disagreement = np.abs(cir_delay - var_delay)
    fused_scale = np.sqrt(local_scale**2 + (0.25 * disagreement) ** 2)

    mean_fraction = np.clip(fused_delay / teacher.delay_max_ns, 0.0, 1.0).astype(np.float32)
    scale_fraction = np.maximum(fused_scale / teacher.delay_max_ns, 1e-5).astype(np.float32)
    return mean_fraction, scale_fraction
