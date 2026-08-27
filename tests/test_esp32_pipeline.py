from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from uwb_tracking.esp32.exporter import (
    ESP32ExportNet,
    calibrate_and_quantize,
    export_c_header,
    export_raw_binary,
    raw_int8_inference,
)
from uwb_tracking.esp32.model import ESP32Architecture, ESP32StudentNet, build_rewound_structured_ticket


def test_esp32_student_is_tiny_and_shapes_are_correct():
    model = ESP32StudentNet(ESP32Architecture((8, 12, 12), 16))
    x = torch.rand(4, 6, 176)
    out = model(x)
    assert out["mean_fraction"].shape == (4,)
    assert out["scale_fraction"].shape == (4,)
    assert out["outlier_probability"].shape == (4,)
    assert torch.all(out["scale_fraction"] > 0)
    assert sum(p.numel() for p in model.parameters()) < 2500


def test_structured_ticket_is_physically_smaller_and_rewound():
    torch.manual_seed(7)
    supernet = ESP32StudentNet(ESP32Architecture((12, 18, 24), 16))
    initial = copy.deepcopy(supernet.state_dict())
    with torch.no_grad():
        supernet.conv1.weight.add_(0.1 * torch.randn_like(supernet.conv1.weight))
    ticket, selection = build_rewound_structured_ticket(
        supernet, initial, ESP32Architecture((8, 12, 12), 16)
    )
    assert ticket.conv1.out_channels == 8
    assert ticket.conv2.out_channels == 12
    assert ticket.conv3.out_channels == 12
    assert selection.c1.numel() == 8
    expected = initial["conv1.weight"][selection.c1]
    assert torch.allclose(ticket.conv1.weight, expected)


def test_bn_fold_and_raw_int8_export(tmp_path: Path):
    torch.manual_seed(3)
    model = ESP32StudentNet(ESP32Architecture((6, 8, 12), 16)).eval()
    x = np.random.default_rng(4).random((12, 6, 176), dtype=np.float32)
    folded = ESP32ExportNet(model).eval()
    with torch.inference_mode():
        raw_original = model.forward_raw(torch.from_numpy(x)).numpy()
        raw_folded = folded(torch.from_numpy(x)).numpy()
    assert np.max(np.abs(raw_original - raw_folded)) < 1e-4

    bundle = calibrate_and_quantize(folded, x, delay_max_ns=35.0, min_scale_fraction=0.004)
    raw_int8 = raw_int8_inference(bundle, x[:4])
    assert raw_int8.shape == (4, 3)
    assert np.all(np.isfinite(raw_int8))

    header = tmp_path / "weights.h"
    blob = tmp_path / "weights.bin"
    export_c_header(bundle, header)
    manifest = export_raw_binary(bundle, blob)
    assert header.stat().st_size > 1000
    assert blob.stat().st_size == manifest["total_bytes"]
    assert blob.stat().st_size < 16_000


def test_runtime_constants_export(tmp_path: Path):
    from uwb_tracking.data import load_uwb_mat
    from uwb_tracking.esp32.preprocess_export import export_preprocess_and_geometry

    data = load_uwb_mat(Path(__file__).resolve().parents[1] / "data" / "uwb_demo_input.mat")
    report = export_preprocess_and_geometry(data, 176, tmp_path)
    assert Path(report["runtime_constants_header"]).exists()
    assert Path(report["background_blob"]).exists()
    assert report["background_blob_bytes"] == data.num_links * 176 * 2
