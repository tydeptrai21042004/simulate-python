from pathlib import Path

import numpy as np

from uwb_tracking.config import ParticleFilterConfig
from uwb_tracking.data import load_uwb_mat
from uwb_tracking.metrics import tracking_metrics
from uwb_tracking.tracking import run_particle_filter, run_repository_particle_filter

ROOT = Path(__file__).resolve().parents[1]


def test_pf_tracks_near_oracle_tof():
    data = load_uwb_mat(ROOT / "data/uwb_demo_input.mat")
    idx = np.arange(40, 100)
    scales = np.full((idx.size, data.num_links), 0.15)
    cfg = ParticleFilterConfig(num_particles=500)
    est, diagnostics = run_particle_filter(
        data.tof_total_ns[idx], scales, data, data.time_s[idx], cfg, seed=9,
        adaptive=True, global_scale_ns=0.15
    )
    metrics = tracking_metrics(est, data.trajectory_xy[idx])
    assert metrics["tracking_rmse_cm"] < 45.0
    assert diagnostics.runtime_ms_per_update > 0


def test_repository_pf_port_runs_and_is_deterministic():
    data = load_uwb_mat(ROOT / "data/uwb_demo_input.mat")
    idx = np.arange(40, 80)
    first, diagnostics = run_repository_particle_filter(
        data.tof_total_ns[idx], data, data.time_s[idx], seed=21,
        error_scale_ns=0.20, error_location_ns=0.0, error_nu=4.0,
    )
    second, _ = run_repository_particle_filter(
        data.tof_total_ns[idx], data, data.time_s[idx], seed=21,
        error_scale_ns=0.20, error_location_ns=0.0, error_nu=4.0,
    )
    assert first.shape == (idx.size, 2)
    assert np.allclose(first, second)
    assert diagnostics.resample_count == idx.size - 1
