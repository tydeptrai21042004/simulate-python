from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    input_length: int = 500
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    num_workers: int = 0
    student_nu: float = 4.0
    min_scale_fraction: float = 0.004
    augmentation_probability: float = 0.25
    # Official MATLAB repository protocol controls. A null fraction uses
    # batch_size; 0.10 reproduces floor(numTrain/10).
    paper_batch_fraction: float | None = None
    optimizer: str = "adamw"
    early_stopping: bool = True
    restore_best: bool = True
    paper_indexing_mode: str = "corrected"  # corrected or repository
    paper_validation_mode: str = "time"  # time (leak-safe) or sample (official repository)
    paper_validation_fraction: float = 0.15


@dataclass
class ParticleFilterConfig:
    num_particles: int = 400
    position_noise_m: float = 0.08
    velocity_noise_mps: float = 0.20
    resample_fraction: float = 0.60
    outlier_prior: float = 0.05
    broad_scale_ns: float = 7.0
    student_nu: float = 4.0
    # Keep float64 as the scientific default; deployment config can select float32.
    numeric_dtype: str = "float64"
    bounds_xy: list[list[float]] = field(
        default_factory=lambda: [[-4.5, 3.2], [-4.4, 0.8]]
    )


@dataclass
class ExperimentConfig:
    data_path: str = "data/uwb_demo_input.mat"
    output_dir: str = "results"
    cases: list[int] = field(default_factory=lambda: [1, 2, 3])
    seeds: list[int] = field(default_factory=lambda: [11, 22, 33, 44, 55])
    scenarios: list[str] = field(
        default_factory=lambda: ["los", "nlos1", "nlos2", "outlier", "dropout"]
    )
    c_m_per_ns: float = 0.299792458
    dt_s: float = 0.20
    device: str = "auto"
    correlated_false_peak_probability: float = 0.5
    # Optional public-source automation. When enabled, CLI commands can fetch
    # and convert the official MATLAB repository before starting experiments.
    official_data: dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    particle_filter: ParticleFilterConfig = field(default_factory=ParticleFilterConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path is None:
        return cfg
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    model_raw = raw.pop("model", {})
    pf_raw = raw.pop("particle_filter", {})
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    for key, value in model_raw.items():
        if hasattr(cfg.model, key):
            setattr(cfg.model, key, value)
    for key, value in pf_raw.items():
        if hasattr(cfg.particle_filter, key):
            setattr(cfg.particle_filter, key, value)
    return cfg
