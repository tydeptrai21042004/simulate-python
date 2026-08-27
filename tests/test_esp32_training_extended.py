from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from uwb_tracking.esp32.exporter import export_checkpoint
from uwb_tracking.esp32.model import ESP32Architecture, ESP32StudentNet, build_rewound_structured_ticket
from uwb_tracking.esp32.training import (
    ESP32TrainingConfig,
    load_student_checkpoint,
    is_structured_lth_checkpoint,
    save_student_checkpoint,
    train_student,
)


def _tiny_arrays(seed=1):
    rng = np.random.default_rng(seed)
    x = rng.random((24, 6, 32), dtype=np.float32)
    y = rng.random(24, dtype=np.float32)
    c = (rng.random(24) < 0.25).astype(np.float32)
    return x[:18], y[:18], c[:18], x[18:], y[18:], c[18:]


def test_ticket_supports_hidden_pruning_and_rejects_wider_targets():
    supernet = ESP32StudentNet(ESP32Architecture((6, 8, 10), 12))
    initial = {k: v.clone() for k, v in supernet.state_dict().items()}
    ticket, selection = build_rewound_structured_ticket(
        supernet, initial, ESP32Architecture((4, 6, 8), 8)
    )
    assert ticket.fc1.out_features == 8
    assert selection.hidden.numel() == 8
    assert torch.allclose(ticket.fc1.bias, initial["fc1.bias"][selection.hidden])
    with pytest.raises(ValueError, match="must not exceed"):
        build_rewound_structured_ticket(supernet, initial, ESP32Architecture((7, 8, 10), 12))
    with pytest.raises(ValueError, match="hidden width"):
        build_rewound_structured_ticket(supernet, initial, ESP32Architecture((6, 8, 10), 13))


def test_train_student_rejects_inconsistent_sample_counts():
    tr_x, tr_y, tr_c, va_x, va_y, va_c = _tiny_arrays()
    model = ESP32StudentNet(ESP32Architecture((4, 6, 8), 8))
    cfg = ESP32TrainingConfig(epochs_supernet=1, device="cpu")
    with pytest.raises(ValueError, match="inconsistent"):
        train_student(model, tr_x, tr_y[:-1], tr_c, va_x, va_y, va_c, 35.0, cfg, 1, 1)


def test_train_student_rejects_partial_teacher_targets():
    tr_x, tr_y, tr_c, va_x, va_y, va_c = _tiny_arrays()
    model = ESP32StudentNet(ESP32Architecture((4, 6, 8), 8))
    cfg = ESP32TrainingConfig(device="cpu")
    with pytest.raises(ValueError, match="provided together"):
        train_student(
            model, tr_x, tr_y, tr_c, va_x, va_y, va_c, 35.0, cfg, 1, 1,
            teacher_mean=np.zeros(len(tr_x), dtype=np.float32), teacher_scale=None,
        )


def test_short_train_checkpoint_roundtrip_and_export(tmp_path: Path):
    torch.manual_seed(2)
    tr_x, tr_y, tr_c, va_x, va_y, va_c = _tiny_arrays(2)
    model = ESP32StudentNet(ESP32Architecture((4, 6, 8), 8))
    cfg = ESP32TrainingConfig(batch_size=8, learning_rate=1e-3, patience=1, device="cpu")
    result = train_student(model, tr_x, tr_y, tr_c, va_x, va_y, va_c, 35.0, cfg, epochs=1, seed=2)
    ckpt = tmp_path / "student.pt"
    save_student_checkpoint(ckpt, result, input_length=32, delay_max_ns=35.0, training_cfg=cfg)
    loaded, meta = load_student_checkpoint(ckpt)
    assert meta["format"] == "uwb-esp32-student-v1"
    with torch.inference_mode():
        a = result.model.forward_raw(torch.from_numpy(va_x)).numpy()
        b = loaded.forward_raw(torch.from_numpy(va_x)).numpy()
    assert np.allclose(a, b)

    report = export_checkpoint(ckpt, tr_x[:8], tmp_path / "export", export_onnx_file=False)
    assert report["parameters"] == sum(p.numel() for p in loaded.parameters())
    assert report["weight_blob_bytes"] < 10_000
    assert (tmp_path / "export" / "golden_vectors.npz").exists()


def test_training_with_teacher_targets_runs():
    tr_x, tr_y, tr_c, va_x, va_y, va_c = _tiny_arrays(4)
    model = ESP32StudentNet(ESP32Architecture((4, 6, 8), 8))
    cfg = ESP32TrainingConfig(batch_size=8, patience=1, device="cpu")
    result = train_student(
        model, tr_x, tr_y, tr_c, va_x, va_y, va_c, 35.0, cfg, epochs=1, seed=4,
        teacher_mean=np.clip(tr_y * 0.95 + 0.02, 0, 1).astype(np.float32),
        teacher_scale=np.full_like(tr_y, 0.03, dtype=np.float32),
    )
    assert np.isfinite(result.validation_mae_ns)
    assert result.epochs_ran == 1


def test_lth_checkpoint_metadata_policy():
    assert is_structured_lth_checkpoint({"extra": {"role": "structured-rewound-lth-ticket"}})
    assert is_structured_lth_checkpoint({"extra": {"role": "structured-rewound-ticket"}})
    assert not is_structured_lth_checkpoint({"extra": {"role": "random-compact-control-only"}})
    assert not is_structured_lth_checkpoint({})
