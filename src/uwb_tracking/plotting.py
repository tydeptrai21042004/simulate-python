from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_trajectory(true_xy: np.ndarray, estimates: dict[str, np.ndarray], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot(true_xy[:, 0], true_xy[:, 1], linewidth=2.0, label="Ground truth")
    for name, xy in estimates.items():
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.2, label=name)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cdf(errors: dict[str, np.ndarray], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for name, values in errors.items():
        x = np.sort(np.asarray(values))
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, linewidth=1.6, label=name)
    ax.set_xlabel("Tracking error (cm)")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(results: pd.DataFrame, path: str | Path) -> None:
    summary = (
        results.groupby(["method", "scenario"], as_index=False)["tracking_rmse_cm"]
        .mean()
        .pivot(index="scenario", columns="method", values="tracking_rmse_cm")
    )
    ax = summary.plot(kind="bar", figsize=(10, 5.5))
    ax.set_ylabel("Mean tracking RMSE (cm)")
    ax.set_xlabel("Scenario")
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
