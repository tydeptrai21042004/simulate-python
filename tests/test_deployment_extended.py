from pathlib import Path

import numpy as np
import pytest
import torch

from uwb_tracking.data import load_uwb_mat, subset_observations
from uwb_tracking.deployment import StreamingPreprocessor, infer_frame
from uwb_tracking.esp32.model import ESP32Architecture, ESP32StudentNet

ROOT = Path(__file__).resolve().parents[1]


def test_streaming_resampled_shape_and_range():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    obs = subset_observations(data, np.array([5]))
    stream = StreamingPreprocessor.from_data(data, input_length=128)
    frame = stream.prepare_frame(obs.cir_dynamic[0], obs.var_dynamic[0])
    assert frame.shape == (data.num_links, 6, 128)
    assert np.all(np.isfinite(frame))
    assert 0.0 <= float(frame.min()) <= float(frame.max()) <= 1.0 + 1e-6


def test_streaming_rejects_invalid_length():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    with pytest.raises(ValueError, match="input_length"):
        StreamingPreprocessor.from_data(data, input_length=0)


def test_streaming_rejects_bad_frame_shape():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    stream = StreamingPreprocessor.from_data(data)
    with pytest.raises(ValueError, match="shape"):
        stream.prepare_frame(np.zeros((1, 10)), np.zeros((1, 10)))


def test_streaming_rejects_nan_frame():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    obs = subset_observations(data, np.array([0]))
    stream = StreamingPreprocessor.from_data(data)
    cir = obs.cir_dynamic[0].copy()
    cir[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        stream.prepare_frame(cir, obs.var_dynamic[0])


def test_infer_frame_returns_physical_arrays():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    obs = subset_observations(data, np.array([0]))
    stream = StreamingPreprocessor.from_data(data)
    x = stream.prepare_frame(obs.cir_dynamic[0], obs.var_dynamic[0])
    model = ESP32StudentNet(ESP32Architecture((4, 6, 8), 8)).eval()
    mean, scale = infer_frame(model, x, delay_max_ns=float(data.delay_grid_ns[-1]))
    assert mean.shape == (data.num_links,)
    assert scale.shape == (data.num_links,)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(scale))
    assert np.all(scale >= 0.05)
