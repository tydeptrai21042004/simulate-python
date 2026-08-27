#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwb_tracking.config import load_config
from uwb_tracking.data import (
    get_case_split,
    load_uwb_mat,
    prepare_inputs,
    subset_observations,
)
from uwb_tracking.models import (
    LiteArchitecture,
    LiteUncertaintyFusionNet,
    build_rewound_structured_ticket,
)
from uwb_tracking.simulation import augment_training_observations
from uwb_tracking.training import _time_level_train_val_split, train_one
from uwb_tracking.utils import ensure_dir, set_seed


def parse_widths(value: str) -> tuple[int, int, int]:
    parts = tuple(int(v.strip()) for v in value.split(","))
    if len(parts) != 3 or min(parts) <= 0:
        raise argparse.ArgumentTypeError("widths must be three positive integers, e.g. 12,18,24")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a structured lottery-ticket-inspired U-Fuse deployment model."
    )
    parser.add_argument("--config", default="configs/deployment_lottery.yaml")
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--super-widths", type=parse_widths, default=(12, 18, 24))
    parser.add_argument("--ticket-widths", type=parse_widths, default=(8, 12, 16))
    parser.add_argument("--output", default="results/deployment_lottery/ticket")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data = load_uwb_mat(ROOT / cfg.data_path)
    train_indices, _ = get_case_split(data.num_time, args.case)
    cfg._delay_max_ns = float(data.delay_grid_ns[-1])  # type: ignore[attr-defined]
    cfg._delay_start_ns = float(data.delay_grid_ns[0])  # type: ignore[attr-defined]
    grid = np.linspace(cfg._delay_start_ns, cfg._delay_max_ns, cfg.model.input_length)
    cfg._delay_step_ns = float(grid[1] - grid[0]) if grid.size > 1 else 1.0  # type: ignore[attr-defined]

    train_core, val_idx = _time_level_train_val_split(
        train_indices, args.seed, val_fraction=cfg.model.paper_validation_fraction
    )
    clean = prepare_inputs(data, subset_observations(data, train_core), cfg.model.input_length)
    val = prepare_inputs(data, subset_observations(data, val_idx), cfg.model.input_length)
    augmented = prepare_inputs(
        data,
        augment_training_observations(
            data,
            train_core,
            args.seed + 7000,
            probability=cfg.model.augmentation_probability,
        ),
        cfg.model.input_length,
    )
    train_x = np.concatenate([clean.fusion, clean.fusion, augmented.fusion], axis=0)
    train_y = np.concatenate(
        [clean.target_fraction, clean.target_fraction, augmented.target_fraction], axis=0
    )
    train_corruption = np.concatenate(
        [clean.corruption, clean.corruption, augmented.corruption], axis=0
    ).astype(np.float32)

    out = ROOT / args.output
    ensure_dir(out)
    aux_hidden = 12
    fusion_hidden = 24
    super_arch = LiteArchitecture(args.super_widths, aux_hidden, fusion_hidden)
    ticket_arch = LiteArchitecture(args.ticket_widths, aux_hidden, fusion_hidden)

    # Dense supernet used only to discover channel identities.
    set_seed(args.seed + 300)
    supernet = LiteUncertaintyFusionNet(cfg.model.min_scale_fraction, arch=super_arch)
    initial_state = copy.deepcopy(supernet.state_dict())
    trained_super = train_one(
        "Lite supernet",
        supernet,
        "fusion",
        train_x,
        train_y,
        val.fusion,
        val.target_fraction,
        cfg,
        args.seed + 300,
        out / "supernet.pt",
        learning_rate=2.0 * cfg.model.learning_rate,
        train_corruption=train_corruption,
    )

    # Select channels from the trained model, then rewind those channels to the
    # *initial* supernet weights before retraining the compact ticket.
    ticket, selection = build_rewound_structured_ticket(
        trained_super.model, initial_state, ticket_arch
    )
    trained_ticket = train_one(
        "Structured rewound ticket",
        ticket,
        "fusion",
        train_x,
        train_y,
        val.fusion,
        val.target_fraction,
        cfg,
        args.seed + 301,
        out / "structured_ticket.pt",
        learning_rate=2.0 * cfg.model.learning_rate,
        train_corruption=train_corruption,
    )

    # Same physical architecture with a fresh random initialization. This is
    # the control needed to determine whether the ticket/rewind itself helps.
    set_seed(args.seed + 302)
    random_compact = LiteUncertaintyFusionNet(
        cfg.model.min_scale_fraction, arch=ticket_arch
    )
    trained_random = train_one(
        "Random compact control",
        random_compact,
        "fusion",
        train_x,
        train_y,
        val.fusion,
        val.target_fraction,
        cfg,
        args.seed + 302,
        out / "random_compact.pt",
        learning_rate=2.0 * cfg.model.learning_rate,
        train_corruption=train_corruption,
    )

    report = {
        "case": args.case,
        "seed": args.seed,
        "input_length": cfg.model.input_length,
        "supernet": {
            "channels": list(super_arch.channels),
            "parameters": trained_super.parameter_count,
            "validation_mae_ns": trained_super.validation_mae_ns,
            "validation_nll": trained_super.validation_nll,
        },
        "structured_rewound_ticket": {
            "channels": list(ticket_arch.channels),
            "parameters": trained_ticket.parameter_count,
            "validation_mae_ns": trained_ticket.validation_mae_ns,
            "validation_nll": trained_ticket.validation_nll,
        },
        "random_compact_control": {
            "channels": list(ticket_arch.channels),
            "parameters": trained_random.parameter_count,
            "validation_mae_ns": trained_random.validation_mae_ns,
            "validation_nll": trained_random.validation_nll,
        },
        "selected_channels": {
            "cir_c1": selection.cir_c1.tolist(),
            "cir_c2": selection.cir_c2.tolist(),
            "var_c1": selection.var_c1.tolist(),
            "var_c2": selection.var_c2.tolist(),
            "shared_c3": selection.shared_c3.tolist(),
        },
        "interpretation": (
            "The structured ticket is useful only if it matches/beats the random compact "
            "control over repeated seeds. The compact architecture itself is deployment-safe "
            "because channels are physically removed rather than zero-masked."
        ),
    }
    (out / "ticket_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
