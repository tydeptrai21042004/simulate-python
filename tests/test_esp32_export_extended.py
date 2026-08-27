from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from uwb_tracking.esp32.exporter import (
    ESP32ExportNet,
    calibrate_and_quantize,
    export_c_header,
    export_raw_binary,
    raw_int8_inference,
)
from uwb_tracking.esp32.model import ESP32Architecture, ESP32StudentNet


def _fixture_bundle(seed: int = 1):
    torch.manual_seed(seed)
    model = ESP32StudentNet(ESP32Architecture((6, 8, 10), 12)).eval()
    rng = np.random.default_rng(seed)
    x = rng.random((16, 6, 176), dtype=np.float32)
    folded = ESP32ExportNet(model).eval()
    bundle = calibrate_and_quantize(folded, x, delay_max_ns=35.0, min_scale_fraction=0.004)
    return model, folded, bundle, x


def test_architecture_validation_rejects_invalid_widths():
    with pytest.raises(ValueError):
        ESP32Architecture((4, 8), 8)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ESP32Architecture((4, 0, 8), 8)
    with pytest.raises(ValueError):
        ESP32Architecture((4, 8, 8), 0)


def test_student_rejects_wrong_input_channels():
    model = ESP32StudentNet(ESP32Architecture((4, 6, 8), 8)).eval()
    with pytest.raises(ValueError, match=r"\[batch, 6, length\]"):
        model(torch.rand(2, 5, 176))


def test_calibration_rejects_empty_nonfinite_and_wrong_shape():
    model = ESP32ExportNet(ESP32StudentNet(ESP32Architecture((4, 6, 8), 8)).eval())
    with pytest.raises(ValueError):
        calibrate_and_quantize(model, np.empty((0, 6, 176), np.float32), 35.0, 0.004)
    with pytest.raises(ValueError):
        calibrate_and_quantize(model, np.zeros((2, 5, 176), np.float32), 35.0, 0.004)
    bad = np.zeros((2, 6, 176), np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        calibrate_and_quantize(model, bad, 35.0, 0.004)


def test_raw_int8_reference_saturates_extreme_input_without_nan():
    _, _, bundle, x = _fixture_bundle()
    extreme = x[:2].copy()
    extreme[0] *= 1e6
    extreme[1] *= -1e6
    y = raw_int8_inference(bundle, extreme)
    assert y.shape == (2, 3)
    assert np.all(np.isfinite(y))


def test_raw_int8_rejects_nan_and_wrong_shape():
    _, _, bundle, _ = _fixture_bundle()
    with pytest.raises(ValueError):
        raw_int8_inference(bundle, np.zeros((1, 5, 176), np.float32))
    x = np.zeros((1, 6, 176), np.float32)
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        raw_int8_inference(bundle, x)


def test_binary_export_is_deterministic_and_hash_matches(tmp_path: Path):
    _, _, bundle, _ = _fixture_bundle(3)
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    ma = export_raw_binary(bundle, a)
    mb = export_raw_binary(bundle, b)
    assert a.read_bytes() == b.read_bytes()
    digest = hashlib.sha256(a.read_bytes()).hexdigest()
    assert ma["sha256"] == digest == mb["sha256"]
    assert ma["total_bytes"] == a.stat().st_size


def test_header_contains_luts_and_all_layers(tmp_path: Path):
    _, _, bundle, _ = _fixture_bundle(4)
    path = tmp_path / "weights.h"
    export_c_header(bundle, path)
    text = path.read_text(encoding="utf-8")
    for name in ("conv1_weight", "conv2_weight", "conv3_weight", "fc1_weight", "fc2_weight"):
        assert name in text
    assert "mean_fraction_q15_lut" in text
    assert "scale_ns_q8_lut" in text
    assert "outlier_probability_q8_lut" in text


def test_quantized_reference_is_reasonably_close_to_folded_float():
    _, folded, bundle, x = _fixture_bundle(8)
    with torch.inference_mode():
        fp = folded(torch.from_numpy(x[:8])).numpy()
    q = raw_int8_inference(bundle, x[:8])
    # Raw logits need only be close enough that decoded physical outputs remain stable.
    assert float(np.mean(np.abs(fp - q))) < 0.15
