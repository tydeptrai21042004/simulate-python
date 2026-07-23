from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    def convert(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        return x

    Path(path).write_text(json.dumps(convert(obj), indent=2), encoding="utf-8")


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(x.size) if x.size > 1 else 0.0
    return mean, std, ci95
