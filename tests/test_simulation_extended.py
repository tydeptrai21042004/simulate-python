from pathlib import Path

import numpy as np
import pytest

from uwb_tracking.data import load_uwb_mat
from uwb_tracking.simulation import SUPPORTED_SCENARIOS, augment_training_observations, corrupt_observations

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scenario", SUPPORTED_SCENARIOS)
def test_all_supported_scenarios_are_deterministic(scenario: str):
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    idx = np.arange(30)
    a = corrupt_observations(data, idx, scenario, seed=31)
    b = corrupt_observations(data, idx, scenario, seed=31)
    assert np.array_equal(a.corruption_mask, b.corruption_mask)
    assert np.allclose(a.cir_dynamic, b.cir_dynamic)
    assert np.allclose(a.var_dynamic, b.var_dynamic)
    if scenario == "los":
        assert not np.any(a.corruption_mask)
    else:
        assert np.any(a.corruption_mask)


def test_burst_dropout_has_contiguous_corruption():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    obs = corrupt_observations(data, np.arange(60), "burst_dropout", seed=9)
    time_any = np.any(obs.corruption_mask, axis=1)
    indices = np.flatnonzero(time_any)
    assert indices.size > 0
    assert np.all(np.diff(indices) == 1)
    assert np.max(obs.corruption_mask.sum(axis=1)) >= 2


def test_higher_severity_changes_corrupted_profiles_more():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    idx = np.arange(30)
    low = corrupt_observations(data, idx, "nlos1", seed=7, severity=0.5)
    high = corrupt_observations(data, idx, "nlos1", seed=7, severity=1.5)
    original = data.cir_dynamic[idx]
    low_delta = float(np.mean(np.abs(low.cir_dynamic - original)))
    high_delta = float(np.mean(np.abs(high.cir_dynamic - original)))
    assert high_delta > low_delta


def test_invalid_corruption_arguments_are_rejected():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    idx = np.arange(5)
    with pytest.raises(ValueError, match="Unknown scenario"):
        corrupt_observations(data, idx, "not-a-scenario", seed=1)
    with pytest.raises(ValueError, match="severity"):
        corrupt_observations(data, idx, "nlos1", seed=1, severity=0)
    with pytest.raises(ValueError, match="probability"):
        augment_training_observations(data, idx, seed=1, probability=1.2)


def test_augmentation_probability_extremes():
    data = load_uwb_mat(ROOT / "data" / "uwb_demo_input.mat")
    idx = np.arange(8)
    none = augment_training_observations(data, idx, seed=2, probability=0.0)
    all_ = augment_training_observations(data, idx, seed=2, probability=1.0)
    assert not np.any(none.corruption_mask)
    assert np.all(all_.corruption_mask)
