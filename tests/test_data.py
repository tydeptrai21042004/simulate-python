from pathlib import Path

import numpy as np

from uwb_tracking.data import get_case_split, load_uwb_mat, prepare_inputs, subset_observations

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_and_geometry():
    data = load_uwb_mat(ROOT / "data/uwb_demo_input.mat")
    assert data.num_time == 240
    assert data.num_links == 6
    data.validate()


def test_three_splits_are_disjoint_and_cover():
    for case in (1, 2, 3):
        train, test = get_case_split(240, case)
        assert not set(train).intersection(set(test))
        assert len(set(train).union(set(test))) == 240


def test_prepared_shapes():
    data = load_uwb_mat(ROOT / "data/uwb_demo_input.mat")
    obs = subset_observations(data, np.arange(5))
    prepared = prepare_inputs(data, obs, input_length=128)
    assert prepared.paper_cir.shape == (30, 1, 128, 2)
    assert prepared.paper_var.shape == (30, 1, 128, 2)
    assert prepared.fusion.shape == (30, 6, 128)
    assert prepared.target_index.shape == (30,)
    assert prepared.target_index.min() >= 1
    assert prepared.target_index.max() <= 128
