from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .experiments import run_ablation_suite, run_benchmark, run_robustness_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UWB passive tracking research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    quick = sub.add_parser("quick-run", help="Small end-to-end smoke experiment")
    quick.add_argument("--config", default="configs/quick.yaml")

    full = sub.add_parser("full", help="Three cases, multiple seeds, all scenarios")
    full.add_argument("--config", default="configs/full.yaml")

    reproduce = sub.add_parser("reproduce-paper", help="Run paper CIR/variance CNN baselines and PF")
    reproduce.add_argument("--config", default="configs/full.yaml")

    ablation = sub.add_parser("ablation", help="Run proposed-method ablations")
    ablation.add_argument("--config", default="configs/quick.yaml")

    sweep = sub.add_parser("robustness-sweep", help="Sweep NLoS severity and cross-modal false-peak correlation")
    sweep.add_argument("--config", default="configs/quick.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    root = Path(__file__).resolve().parents[2]
    if not Path(cfg.data_path).is_absolute():
        cfg.data_path = str(root / cfg.data_path)
    if not Path(cfg.output_dir).is_absolute():
        cfg.output_dir = str(root / cfg.output_dir)

    if args.command in {"quick-run", "full", "reproduce-paper"}:
        results, summary = run_benchmark(
            cfg, proposed_ablation="full", make_plots=True, include_proposed=(args.command != "reproduce-paper")
        )
        print("\nCompleted. Raw rows:", len(results))
        print(summary[["method", "scenario", "tracking_rmse_cm_mean", "tracking_rmse_cm_std"]].to_string(index=False))
    elif args.command == "ablation":
        combined = run_ablation_suite(cfg)
        print(f"Completed ablation suite with {len(combined)} result rows.")
    elif args.command == "robustness-sweep":
        result = run_robustness_sweep(cfg)
        print(f"Completed robustness sweep with {len(result)} result rows.")


if __name__ == "__main__":
    main()
