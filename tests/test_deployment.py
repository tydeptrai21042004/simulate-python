from pathlib import Path

import numpy as np

from uwb_tracking.data import load_uwb_mat, prepare_inputs, subset_observations
from uwb_tracking.deployment import StreamingPreprocessor

ROOT = Path(__file__).resolve().parents[1]


def test_streaming_native_preprocessing_matches_batch_preprocessing():
    data = load_uwb_mat(ROOT / "data/uwb_demo_input.mat")
    idx = np.array([10])
    obs = subset_observations(data, idx)
    batch = prepare_inputs(data, obs, input_length=data.delay_grid_ns.size)
    stream = StreamingPreprocessor.from_data(data, input_length=data.delay_grid_ns.size)
    frame = stream.prepare_frame(obs.cir_dynamic[0], obs.var_dynamic[0])
    assert frame.shape == (data.num_links, 6, data.delay_grid_ns.size)
    assert np.allclose(frame, batch.fusion, atol=1e-6)
