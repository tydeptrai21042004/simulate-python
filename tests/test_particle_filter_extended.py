from pathlib import Path

import numpy as np
import pytest

from uwb_tracking.config import ParticleFilterConfig
from uwb_tracking.data import load_uwb_mat
from uwb_tracking.tracking.particle_filter import _student_t_logpdf, run_particle_filter

ROOT = Path(__file__).resolve().parents[1]


def _problem(n=30):
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    idx = np.arange(20, 20 + n)
    scales = np.full((n, data.num_links), 0.20, dtype=np.float32)
    return data, idx, scales


def test_student_t_logpdf_is_finite_for_tiny_scale():
    residual = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    out = _student_t_logpdf(residual, 0.0, 4.0)
    assert np.all(np.isfinite(out))
    assert out[1] > out[0]
    assert out[1] > out[2]


def test_pf_float32_is_finite_and_close_to_float64():
    data, idx, scales = _problem(35)
    cfg64 = ParticleFilterConfig(num_particles=220, numeric_dtype="float64")
    cfg32 = ParticleFilterConfig(num_particles=220, numeric_dtype="float32")
    e64, _ = run_particle_filter(data.tof_total_ns[idx], scales, data, data.time_s[idx], cfg64, 17, True, 0.2)
    e32, _ = run_particle_filter(data.tof_total_ns[idx], scales, data, data.time_s[idx], cfg32, 17, True, 0.2)
    assert np.all(np.isfinite(e32))
    # RNG arithmetic differs slightly by dtype, so compare trajectory-level proximity.
    assert float(np.mean(np.linalg.norm(e64 - e32, axis=1))) < 0.75


def test_pf_outlier_probability_shape_validation():
    data, idx, scales = _problem(10)
    cfg = ParticleFilterConfig(num_particles=80)
    bad = np.zeros((10, data.num_links - 1), dtype=np.float32)
    with pytest.raises(ValueError, match="must match"):
        run_particle_filter(
            data.tof_total_ns[idx], scales, data, data.time_s[idx], cfg, 3, True, 0.2,
            predicted_outlier_probability=bad,
        )


def test_pf_accepts_quality_probabilities_and_remains_finite():
    data, idx, scales = _problem(20)
    cfg = ParticleFilterConfig(num_particles=120, numeric_dtype="float32")
    quality = np.zeros_like(scales)
    quality[5:10, 0] = 1.0
    est, diag = run_particle_filter(
        data.tof_total_ns[idx], scales, data, data.time_s[idx], cfg, 4, True, 0.2,
        predicted_outlier_probability=quality,
    )
    assert est.shape == (20, 2)
    assert np.all(np.isfinite(est))
    assert np.all(np.isfinite(diag.effective_sample_size))


def test_pf_is_deterministic_for_same_seed():
    data, idx, scales = _problem(20)
    cfg = ParticleFilterConfig(num_particles=100, numeric_dtype="float32")
    a, _ = run_particle_filter(data.tof_total_ns[idx], scales, data, data.time_s[idx], cfg, 99, True, 0.2)
    b, _ = run_particle_filter(data.tof_total_ns[idx], scales, data, data.time_s[idx], cfg, 99, True, 0.2)
    assert np.allclose(a, b)
