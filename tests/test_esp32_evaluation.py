from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from uwb_tracking.data import load_uwb_mat, prepare_inputs, subset_observations
from uwb_tracking.esp32.evaluation import (
    deployment_point_metrics,
    deployment_tracking_metrics,
    particle_filter_config_from_mapping,
    predict_fp32_raw,
    predict_int8_raw,
)
from uwb_tracking.esp32.exporter import ESP32ExportNet, calibrate_and_quantize
from uwb_tracking.esp32.model import ESP32Architecture, ESP32StudentNet


ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    obs = subset_observations(data, np.arange(8))
    prepared = prepare_inputs(data, obs, input_length=96)
    torch.manual_seed(12)
    model = ESP32StudentNet(ESP32Architecture((4, 6, 6), 6)).eval()
    bundle = calibrate_and_quantize(
        ESP32ExportNet(model).eval(),
        prepared.fusion,
        delay_max_ns=float(data.delay_grid_ns[-1]),
        min_scale_fraction=model.min_scale_fraction,
    )
    return data, obs, prepared, model, bundle


def test_point_metrics_cover_fp32_and_integer_reference():
    data, _, prepared, model, bundle = _fixture()
    fp32 = predict_fp32_raw(model, prepared.fusion, batch_size=13)
    int8 = predict_int8_raw(bundle, prepared.fusion, batch_size=11)
    assert fp32.shape == int8.shape == (prepared.fusion.shape[0], 3)
    m_fp = deployment_point_metrics(
        fp32,
        prepared,
        float(data.delay_grid_ns[-1]),
        model.min_scale_fraction,
    )
    m_q = deployment_point_metrics(
        int8,
        prepared,
        float(data.delay_grid_ns[-1]),
        model.min_scale_fraction,
    )
    assert np.isfinite(m_fp["tof_mae_ns"])
    assert np.isfinite(m_q["tof_mae_ns"])
    assert "outlier_probability_brier" in m_q


def test_complete_deployment_tracking_metric_path_runs():
    data, obs, prepared, model, bundle = _fixture()
    raw = predict_int8_raw(bundle, prepared.fusion)
    pf_cfg = particle_filter_config_from_mapping(
        {
            "num_particles": 32,
            "numeric_dtype": "float32",
            "bounds_xy": [[-4.5, 3.2], [-4.4, 0.8]],
        }
    )
    metrics = deployment_tracking_metrics(
        raw,
        prepared,
        data,
        obs.true_xy,
        obs.time_s,
        pf_cfg,
        seed=7,
        delay_max_ns=float(data.delay_grid_ns[-1]),
        min_scale_fraction=model.min_scale_fraction,
    )
    assert np.isfinite(metrics["tracking_rmse_cm"])
    assert np.isfinite(metrics["pf_runtime_ms_per_update"])
    assert metrics["pf_resample_count"] >= 0


def test_particle_filter_mapping_rejects_too_few_particles():
    try:
        particle_filter_config_from_mapping({"num_particles": 1})
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
