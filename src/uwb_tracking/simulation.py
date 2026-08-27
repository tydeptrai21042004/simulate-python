from __future__ import annotations

import numpy as np

from .data import ObservationSet, UWBData, subset_observations


SUPPORTED_SCENARIOS = (
    "los",
    "nlos1",
    "nlos2",
    "outlier",
    "dropout",
    "burst_nlos",
    "burst_dropout",
    "mixed",
)


def _scenario_mask(n: int, links: int, scenario: str, rng: np.random.Generator) -> np.ndarray:
    if n < 0 or links <= 0:
        raise ValueError("n must be >= 0 and links must be > 0")
    mask = np.zeros((n, links), dtype=bool)
    if scenario == "los":
        return mask
    if scenario == "nlos1":
        # One obstructed link at each timestamp, changed in contiguous blocks.
        block = max(4, n // max(2 * links, 1))
        for t in range(n):
            mask[t, (t // block) % links] = True
        return mask
    if scenario == "nlos2":
        block = max(4, n // max(2 * links, 1))
        for t in range(n):
            first = (t // block) % links
            mask[t, first] = True
            mask[t, (first + 2) % links] = True
        return mask
    if scenario == "outlier":
        return rng.random((n, links)) < 0.12
    if scenario == "dropout":
        return rng.random((n, links)) < 0.18
    if scenario == "burst_nlos":
        if n == 0:
            return mask
        burst = max(2, int(round(0.35 * n)))
        start_max = max(n - burst, 0)
        start = int(rng.integers(0, start_max + 1)) if start_max else 0
        link = int(rng.integers(0, links))
        mask[start : start + burst, link] = True
        return mask
    if scenario == "burst_dropout":
        if n == 0:
            return mask
        burst = max(2, int(round(0.25 * n)))
        start_max = max(n - burst, 0)
        start = int(rng.integers(0, start_max + 1)) if start_max else 0
        first = int(rng.integers(0, links))
        second = (first + max(1, links // 2)) % links
        mask[start : start + burst, first] = True
        mask[start : start + burst, second] = True
        return mask
    if scenario == "mixed":
        # A moderately sparse mask. Each selected sample is later assigned
        # independently to dropout-like or false-peak/NLoS corruption.
        return rng.random((n, links)) < 0.16
    raise ValueError(f"Unknown scenario: {scenario}. Supported: {', '.join(SUPPORTED_SCENARIOS)}")


def corrupt_observations(
    data: UWBData,
    indices: np.ndarray,
    scenario: str,
    seed: int,
    correlated_false_peak_probability: float = 0.5,
    severity: float = 1.0,
) -> ObservationSet:
    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Supported: {', '.join(SUPPORTED_SCENARIOS)}")
    if not (0.0 <= correlated_false_peak_probability <= 1.0):
        raise ValueError("correlated_false_peak_probability must be in [0, 1]")
    if severity <= 0:
        raise ValueError("severity must be > 0")

    obs = subset_observations(data, indices)
    rng = np.random.default_rng(seed)
    mask = _scenario_mask(len(indices), data.num_links, scenario, rng)
    obs.corruption_mask = mask
    if scenario == "los":
        return obs

    grid = data.delay_grid_ns
    bins = grid.size
    for t in range(mask.shape[0]):
        for link in range(mask.shape[1]):
            if not mask[t, link]:
                continue
            bg_c = data.cir_background[link].astype(np.float64)
            bg_v = data.var_background[link].astype(np.float64)
            row_c = obs.cir_dynamic[t, link].astype(np.float64)
            row_v = obs.var_dynamic[t, link].astype(np.float64)

            is_dropout = scenario in {"dropout", "burst_dropout"}
            if scenario == "mixed":
                is_dropout = bool(rng.random() < 0.35)

            if is_dropout:
                obs.cir_dynamic[t, link] = np.maximum(
                    bg_c + 0.015 * severity * rng.normal(size=bins), 0
                )
                obs.var_dynamic[t, link] = np.maximum(
                    bg_v + 0.010 * severity * rng.normal(size=bins), 0
                )
                continue

            attenuation_c = np.clip(0.12 / max(severity, 0.25), 0.03, 0.40)
            attenuation_v = np.clip(0.25 / max(severity, 0.25), 0.06, 0.55)
            new_c = bg_c + attenuation_c * (row_c - bg_c) + 0.040 * severity * rng.normal(size=bins)
            new_v = bg_v + attenuation_v * (row_v - bg_v) + 0.030 * severity * rng.normal(size=bins)

            center_c = rng.uniform(4.0, 27.0)
            if rng.random() < correlated_false_peak_probability:
                center_v = center_c + rng.normal(0.0, 0.7)
            else:
                center_v = rng.uniform(4.0, 27.0)
            false_c = np.exp(-0.5 * ((grid - center_c) / 0.8) ** 2)
            false_v = np.exp(-0.5 * ((grid - center_v) / 1.0) ** 2)
            new_c += 0.65 * severity * false_c
            new_v += 0.32 * severity * false_v
            obs.cir_dynamic[t, link] = np.maximum(new_c, 0)
            obs.var_dynamic[t, link] = np.maximum(new_v, 0)
    return obs


def augment_training_observations(
    data: UWBData,
    indices: np.ndarray,
    seed: int,
    probability: float = 0.25,
) -> ObservationSet:
    """Randomized corruption family used only for proposed-model robustness training.

    False peaks are sometimes shared and sometimes independent across CIR/variance,
    preventing the simulator from structurally favoring agreement-based fusion.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError("probability must be in [0, 1]")
    obs = subset_observations(data, indices)
    rng = np.random.default_rng(seed)
    mask = rng.random((len(indices), data.num_links)) < probability
    obs.corruption_mask = mask
    grid = data.delay_grid_ns
    bins = grid.size
    for t, link in zip(*np.where(mask)):
        severity = rng.uniform(0.6, 1.4)
        bg_c = data.cir_background[link].astype(np.float64)
        bg_v = data.var_background[link].astype(np.float64)
        row_c = obs.cir_dynamic[t, link].astype(np.float64)
        row_v = obs.var_dynamic[t, link].astype(np.float64)
        # Include background-only/dropout examples so the quality head learns
        # that a missing dynamic return is itself an unreliable measurement.
        if rng.random() < 0.20:
            obs.cir_dynamic[t, link] = np.maximum(
                bg_c + rng.uniform(0.008, 0.020) * rng.normal(size=bins), 0
            )
            obs.var_dynamic[t, link] = np.maximum(
                bg_v + rng.uniform(0.006, 0.015) * rng.normal(size=bins), 0
            )
            continue
        new_c = bg_c + rng.uniform(0.05, 0.35) * (row_c - bg_c)
        new_v = bg_v + rng.uniform(0.10, 0.55) * (row_v - bg_v)
        new_c += rng.uniform(0.02, 0.06) * rng.normal(size=bins)
        new_v += rng.uniform(0.015, 0.045) * rng.normal(size=bins)
        center_c = rng.uniform(3.0, 29.0)
        center_v = center_c + rng.normal(0, 0.8) if rng.random() < 0.5 else rng.uniform(3.0, 29.0)
        new_c += rng.uniform(0.25, 0.80) * severity * np.exp(
            -0.5 * ((grid - center_c) / rng.uniform(0.6, 1.2)) ** 2
        )
        new_v += rng.uniform(0.15, 0.50) * severity * np.exp(
            -0.5 * ((grid - center_v) / rng.uniform(0.8, 1.5)) ** 2
        )
        obs.cir_dynamic[t, link] = np.maximum(new_c, 0)
        obs.var_dynamic[t, link] = np.maximum(new_v, 0)
    return obs
