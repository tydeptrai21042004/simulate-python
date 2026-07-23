from pathlib import Path

import numpy as np

from uwb_tracking.data import load_uwb_mat
from uwb_tracking.simulation import corrupt_observations

ROOT = Path(__file__).resolve().parents[1]


def test_corruption_is_deterministic_and_nontrivial():
    data = load_uwb_mat(ROOT / "data/uwb_demo_input.mat")
    idx = np.arange(20)
    a = corrupt_observations(data, idx, "nlos1", 123)
    b = corrupt_observations(data, idx, "nlos1", 123)
    assert np.allclose(a.cir_dynamic, b.cir_dynamic)
    assert np.any(a.corruption_mask)
    assert not np.allclose(a.cir_dynamic, data.cir_dynamic[idx])
