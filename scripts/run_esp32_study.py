#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _display_path(path: Path) -> str:
    """Return a stable repo-relative path when possible, else an absolute path."""

    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _find_tracking(report: dict, scenario: str, variant: str) -> dict | None:
    rows = ((report.get("deployment_evaluation") or {}).get("rows") or [])
    return next(
        (row for row in rows if row.get("scenario") == scenario and row.get("variant") == variant),
        None,
    )


def _run_row(report: dict, report_path: Path) -> dict[str, object]:
    selected = report["selected_for_deployment"]
    export = report["export"]
    los_int8 = _find_tracking(report, "los", "lth_int8_reference")
    los_fp32 = _find_tracking(report, "los", "lth_fp32")
    los_control = _find_tracking(report, "los", "random_compact_fp32")
    robust_rows = [
        row
        for row in ((report.get("deployment_evaluation") or {}).get("rows") or [])
        if row.get("variant") == "lth_int8_reference" and row.get("scenario") != "los"
    ]
    row: dict[str, object] = {
        "case": int(report["case"]),
        "seed": int(report["seed"]),
        "parameters": int(selected["parameters"]),
        "weight_blob_bytes": int(export["weight_blob_bytes"]),
        "core_static_deployment_bytes": int(export.get("core_static_deployment_bytes", 0)),
        "fp32_clean_tof_mae_ns": float(selected["fp32_clean"]["tof_mae_ns"]),
        "int8_clean_tof_mae_ns": float(selected["int8_clean"]["tof_mae_ns"]),
        "fp32_robust_tof_mae_ns": float(selected["fp32_robust"]["tof_mae_ns"]),
        "int8_robust_tof_mae_ns": float(selected["int8_robust"]["tof_mae_ns"]),
        "fp32_guard_met": bool(selected["fp32_guard_met"]),
        "int8_guard_met": bool(selected["int8_guard_met"]),
        "report": _display_path(report_path),
    }
    if los_int8:
        row.update(
            {
                "los_int8_tracking_rmse_cm": float(los_int8["tracking_rmse_cm"]),
                "los_int8_tracking_p90_cm": float(los_int8["tracking_p90_cm"]),
                "los_int8_tof_mae_ns": float(los_int8["tof_mae_ns"]),
            }
        )
    if los_fp32:
        row["los_quant_tracking_rmse_delta_cm"] = (
            float(los_int8["tracking_rmse_cm"]) - float(los_fp32["tracking_rmse_cm"])
            if los_int8
            else float("nan")
        )
    if los_control and los_int8:
        row["lth_int8_beats_random_control_los_tracking"] = (
            float(los_int8["tracking_rmse_cm"]) <= float(los_control["tracking_rmse_cm"])
        )
    if robust_rows:
        row["robust_int8_tracking_rmse_mean_cm"] = float(
            np.mean([float(r["tracking_rmse_cm"]) for r in robust_rows])
        )
        row["robust_int8_tracking_p90_mean_cm"] = float(
            np.mean([float(r["tracking_p90_cm"]) for r in robust_rows])
        )
    return row


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = [
        "parameters",
        "weight_blob_bytes",
        "core_static_deployment_bytes",
        "fp32_clean_tof_mae_ns",
        "int8_clean_tof_mae_ns",
        "fp32_robust_tof_mae_ns",
        "int8_robust_tof_mae_ns",
        "los_int8_tracking_rmse_cm",
        "los_int8_tracking_p90_cm",
        "los_quant_tracking_rmse_delta_cm",
        "robust_int8_tracking_rmse_mean_cm",
        "robust_int8_tracking_p90_mean_cm",
    ]
    out: dict[str, object] = {"completed_runs": len(rows)}
    for key in numeric_keys:
        values = np.asarray([float(r[key]) for r in rows if key in r], dtype=np.float64)
        if values.size:
            out[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    wins = [bool(r["lth_int8_beats_random_control_los_tracking"]) for r in rows if "lth_int8_beats_random_control_los_tracking" in r]
    if wins:
        out["lth_int8_tracking_win_rate_vs_random_control"] = float(np.mean(wins))
    out["all_fp32_guards_met"] = all(bool(r["fp32_guard_met"]) for r in rows)
    out["all_int8_guards_met"] = all(bool(r["int8_guard_met"]) for r in rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete structured-LTH -> INT8 -> PF evaluation across cases/seeds."
    )
    parser.add_argument("--config", default="configs/esp32s3_official.yaml")
    parser.add_argument("--cases", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--output-root", default="results/esp32s3_official_study")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_path = (ROOT / args.config).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    output_root = (ROOT / args.output_root).resolve()
    config_dir = output_root / "resolved_configs"
    logs_dir = output_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for case in args.cases:
        if case not in (1, 2, 3):
            raise ValueError("cases must be drawn from 1, 2, 3")
        for seed in args.seeds:
            run_dir = output_root / f"case_{case}" / f"seed_{seed}"
            report_path = run_dir / "pipeline_report.json"
            if report_path.exists() and not args.force:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("status") == "ok":
                    rows.append(_run_row(report, report_path))
                    print(f"[skip] case={case} seed={seed}: completed")
                    continue

            cfg = json.loads(json.dumps(base))
            cfg["case"] = int(case)
            cfg["seed"] = int(seed)
            cfg["output_dir"] = _display_path(run_dir)
            resolved = config_dir / f"case_{case}_seed_{seed}.yaml"
            resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            command = [
                sys.executable,
                str(ROOT / "scripts" / "train_esp32_pipeline.py"),
                "--config",
                str(resolved),
            ]
            print("[run]", " ".join(command))
            if args.dry_run:
                continue
            log_path = logs_dir / f"case_{case}_seed_{seed}.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if completed.returncode != 0:
                failure = {"case": case, "seed": seed, "returncode": completed.returncode, "log": _display_path(log_path)}
                failures.append(failure)
                print(f"[fail] case={case} seed={seed}; see {log_path}")
                if not args.continue_on_error:
                    raise RuntimeError(json.dumps(failure))
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rows.append(_run_row(report, report_path))
            print(f"[ok] case={case} seed={seed}")

    if args.dry_run:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "study_runs.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "base_config": _display_path(base_path),
        "cases": args.cases,
        "seeds": args.seeds,
        "requested_runs": len(args.cases) * len(args.seeds),
        "failures": failures,
        "aggregate": _summary(rows),
    }
    (output_root / "study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
