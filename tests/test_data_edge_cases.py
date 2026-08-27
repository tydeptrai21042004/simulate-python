from pathlib import Path

import numpy as np
import pytest

from uwb_tracking.data import (
    _minmax_rows,
    get_case_split,
    load_uwb_mat,
    prepare_inputs,
    subset_observations,
)

ROOT = Path(__file__).resolve().parents[1]


def test_minmax_constant_rows_are_zero_and_finite():
    x = np.full((3, 8), 7.5, dtype=np.float32)
    y = _minmax_rows(x)
    assert np.all(np.isfinite(y))
    assert np.all(y == 0.0)


def test_prepare_inputs_is_finite_and_normalized():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    obs = subset_observations(data, np.arange(3))
    prepared = prepare_inputs(data, obs, input_length=176)
    assert np.all(np.isfinite(prepared.fusion))
    assert float(prepared.fusion.min()) >= 0.0
    assert float(prepared.fusion.max()) <= 1.0 + 1e-6
    assert np.all((prepared.target_fraction >= 0) & (prepared.target_fraction <= 1))


def test_prepare_inputs_supports_single_bin_for_edge_validation():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    obs = subset_observations(data, np.arange(2))
    prepared = prepare_inputs(data, obs, input_length=1)
    assert prepared.fusion.shape == (2 * data.num_links, 6, 1)
    assert np.all(prepared.target_index == 1)


def test_prepare_inputs_rejects_zero_length():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    obs = subset_observations(data, np.arange(2))
    with pytest.raises(ValueError, match="input_length"):
        prepare_inputs(data, obs, input_length=0)


def test_prepare_inputs_rejects_empty_observation_set():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    obs = subset_observations(data, np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="at least one"):
        prepare_inputs(data, obs, input_length=176)


def test_invalid_case_id_rejected():
    with pytest.raises(ValueError, match="case_id"):
        get_case_split(240, 4)
