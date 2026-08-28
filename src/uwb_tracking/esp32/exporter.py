from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .model import ESP32StudentNet
from .training import load_student_checkpoint


class ESP32ExportNet(nn.Module):
    """BN-folded export graph: Conv/ReLU x3 -> GAP -> FC/ReLU -> FC."""

    def __init__(self, source: ESP32StudentNet) -> None:
        super().__init__()
        c1, c2, c3 = source.arch.channels
        self.conv1 = nn.Conv1d(6, c1, 7, stride=2, padding=3, bias=True)
        self.conv2 = nn.Conv1d(c1, c2, 5, stride=2, padding=2, bias=True)
        self.conv3 = nn.Conv1d(c2, c3, 3, stride=2, padding=1, bias=True)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(c3, source.arch.hidden, bias=True)
        self.fc2 = nn.Linear(source.arch.hidden, 3, bias=True)
        self._copy_folded(source)

    @staticmethod
    def _fold_conv_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> tuple[torch.Tensor, torch.Tensor]:
        if conv.bias is None:
            bias = torch.zeros(conv.out_channels, dtype=conv.weight.dtype, device=conv.weight.device)
        else:
            bias = conv.bias.detach()
        weight = conv.weight.detach()
        inv = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
        folded_w = weight * inv[:, None, None]
        folded_b = bn.bias.detach() + (bias - bn.running_mean.detach()) * inv
        return folded_w, folded_b

    def _copy_folded(self, source: ESP32StudentNet) -> None:
        source = source.eval()
        with torch.no_grad():
            for dst, conv, bn in (
                (self.conv1, source.conv1, source.bn1),
                (self.conv2, source.conv2, source.bn2),
                (self.conv3, source.conv3, source.bn3),
            ):
                w, b = self._fold_conv_bn(conv, bn)
                dst.weight.copy_(w)
                dst.bias.copy_(b)
            self.fc1.weight.copy_(source.fc1.weight)
            self.fc1.bias.copy_(source.fc1.bias)
            self.fc2.weight.copy_(source.fc2.weight)
            self.fc2.bias.copy_(source.fc2.bias)

    def forward_intermediates(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        a1 = F.relu(self.conv1(x), inplace=False)
        a2 = F.relu(self.conv2(a1), inplace=False)
        a3 = F.relu(self.conv3(a2), inplace=False)
        pooled = self.avg(a3).squeeze(-1)
        a4 = F.relu(self.fc1(pooled), inplace=False)
        raw = self.fc2(a4)
        return raw, [a1, a2, a3, pooled, a4, raw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_intermediates(x)[0]


@dataclass
class QuantizedLayer:
    name: str
    weight: np.ndarray
    bias: np.ndarray
    # Per-output-channel symmetric INT8 weight scales improve accuracy at
    # negligible model-size cost. Firmware does not need these floating scales
    # because requantization is exported as integer multiplier + right shift.
    weight_scale: np.ndarray
    input_scale: float
    output_scale: float
    multiplier: np.ndarray
    requant_multiplier_q31: np.ndarray
    requant_shift: np.ndarray
    stride: int = 1
    padding: int = 0


@dataclass
class RawINT8Bundle:
    input_scale: float
    layers: list[QuantizedLayer]
    delay_max_ns: float
    min_scale_fraction: float
    activation_scales: dict[str, float]


def _safe_scale(max_abs: float) -> float:
    return max(float(max_abs) / 127.0, 1e-8)


def _quantize_weight(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric INT8 quantization with one scale per output channel."""

    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim < 2:
        raise ValueError("weight tensor must have an output-channel axis")
    reduce_axes = tuple(range(1, weight.ndim))
    max_abs = np.max(np.abs(weight), axis=reduce_axes)
    scale = np.maximum(max_abs / 127.0, 1e-8).astype(np.float64)
    reshape = (scale.size,) + (1,) * (weight.ndim - 1)
    q = np.clip(np.rint(weight / scale.reshape(reshape)), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def _fixedpoint_multiplier(multiplier: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate positive real multipliers as Q31 integer + right shift.

    For each m > 0, frexp gives m = mantissa * 2**exp with mantissa in
    [0.5, 1). We quantize mantissa to signed Q31 and use shift=31-exp, so
    round(acc*m) can be reproduced using only int64 multiply, rounding and a
    right shift.
    """

    multiplier = np.asarray(multiplier, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(multiplier)) or np.any(multiplier <= 0):
        raise ValueError("requantization multipliers must be finite and > 0")
    q = np.empty(multiplier.size, dtype=np.int64)
    shift = np.empty(multiplier.size, dtype=np.int32)
    for i, value in enumerate(multiplier):
        mantissa, exponent = np.frexp(float(value))
        q31 = int(round(mantissa * (1 << 31)))
        if q31 == (1 << 31):
            q31 //= 2
            exponent += 1
        right_shift = 31 - exponent
        if not (1 <= right_shift <= 62):
            raise ValueError(f"unsupported requantization shift {right_shift} for multiplier {value}")
        q[i] = q31
        shift[i] = right_shift
    return q.astype(np.int32), shift.astype(np.uint8)


def _quantize_layer(
    name: str,
    weight: np.ndarray,
    bias: np.ndarray,
    input_scale: float,
    output_scale: float,
    stride: int = 1,
    padding: int = 0,
) -> QuantizedLayer:
    q_weight, weight_scale = _quantize_weight(weight)
    bias_scale = float(input_scale) * weight_scale.astype(np.float64)
    bias_f = np.asarray(bias, dtype=np.float64).reshape(-1)
    if bias_f.size != weight_scale.size:
        raise ValueError(f"{name}: bias/output-channel mismatch")
    q_bias64 = np.rint(bias_f / bias_scale)
    info = np.iinfo(np.int32)
    if np.any(q_bias64 < info.min) or np.any(q_bias64 > info.max):
        raise OverflowError(f"{name}: quantized bias exceeds int32 range")
    q_bias = q_bias64.astype(np.int32)
    multiplier = bias_scale / float(output_scale)
    requant_q31, requant_shift = _fixedpoint_multiplier(multiplier)
    return QuantizedLayer(
        name=name,
        weight=q_weight,
        bias=q_bias,
        weight_scale=weight_scale,
        input_scale=float(input_scale),
        output_scale=float(output_scale),
        multiplier=multiplier.astype(np.float32),
        requant_multiplier_q31=requant_q31,
        requant_shift=requant_shift,
        stride=stride,
        padding=padding,
    )


@torch.inference_mode()
def calibrate_and_quantize(
    model: ESP32ExportNet,
    calibration_x: np.ndarray,
    delay_max_ns: float,
    min_scale_fraction: float,
    batch_size: int = 128,
) -> RawINT8Bundle:
    calibration_x = np.asarray(calibration_x, dtype=np.float32)
    if calibration_x.ndim != 3 or calibration_x.shape[1] != 6:
        raise ValueError("calibration_x must have shape [samples, 6, input_length]")
    if calibration_x.shape[0] < 1 or calibration_x.shape[2] < 1:
        raise ValueError("calibration_x must contain at least one non-empty sample")
    if not np.all(np.isfinite(calibration_x)):
        raise ValueError("calibration_x must contain only finite values")
    if float(delay_max_ns) <= 0:
        raise ValueError("delay_max_ns must be > 0")
    if float(min_scale_fraction) <= 0:
        raise ValueError("min_scale_fraction must be > 0")
    model = model.cpu().eval()
    maxima = np.zeros(6, dtype=np.float64)
    input_max = float(np.max(np.abs(calibration_x)))
    for start in range(0, len(calibration_x), batch_size):
        xb = torch.from_numpy(calibration_x[start : start + batch_size].astype(np.float32, copy=False))
        _, acts = model.forward_intermediates(xb)
        for i, act in enumerate(acts):
            maxima[i] = max(maxima[i], float(act.detach().abs().max().cpu()))

    input_scale = _safe_scale(input_max)
    a1_scale, a2_scale, a3_scale, pool_scale, a4_scale, raw_scale = [_safe_scale(v) for v in maxima]
    # GAP preserves the activation unit. Use the Conv3 scale exactly so an
    # integer average can be used in C without an extra requantization stage.
    pool_scale = a3_scale

    layers = [
        _quantize_layer(
            "conv1",
            model.conv1.weight.detach().numpy(),
            model.conv1.bias.detach().numpy(),
            input_scale,
            a1_scale,
            stride=model.conv1.stride[0],
            padding=model.conv1.padding[0],
        ),
        _quantize_layer(
            "conv2",
            model.conv2.weight.detach().numpy(),
            model.conv2.bias.detach().numpy(),
            a1_scale,
            a2_scale,
            stride=model.conv2.stride[0],
            padding=model.conv2.padding[0],
        ),
        _quantize_layer(
            "conv3",
            model.conv3.weight.detach().numpy(),
            model.conv3.bias.detach().numpy(),
            a2_scale,
            a3_scale,
            stride=model.conv3.stride[0],
            padding=model.conv3.padding[0],
        ),
        _quantize_layer(
            "fc1",
            model.fc1.weight.detach().numpy(),
            model.fc1.bias.detach().numpy(),
            pool_scale,
            a4_scale,
        ),
        _quantize_layer(
            "fc2",
            model.fc2.weight.detach().numpy(),
            model.fc2.bias.detach().numpy(),
            a4_scale,
            raw_scale,
        ),
    ]
    return RawINT8Bundle(
        input_scale=input_scale,
        layers=layers,
        delay_max_ns=float(delay_max_ns),
        min_scale_fraction=float(min_scale_fraction),
        activation_scales={
            "input": input_scale,
            "conv1": a1_scale,
            "conv2": a2_scale,
            "conv3": a3_scale,
            "pool": pool_scale,
            "fc1": a4_scale,
            "raw": raw_scale,
        },
    )


def _requantize_int64(acc: np.ndarray, layer: QuantizedLayer) -> np.ndarray:
    """Per-output-channel fixed-point requantization used by the C reference path."""

    acc = np.asarray(acc, dtype=np.int64)
    qmul = layer.requant_multiplier_q31.astype(np.int64)
    shift = layer.requant_shift.astype(np.int64)
    if acc.shape[-1] != qmul.size:
        raise ValueError(f"{layer.name}: accumulator/output-channel mismatch")
    product = acc * qmul
    magnitude = np.abs(product)
    rounding = np.left_shift(np.ones_like(shift, dtype=np.int64), shift - 1)
    rounded = np.right_shift(magnitude + rounding, shift)
    return np.where(product < 0, -rounded, rounded)


def _conv1d_int8(x: np.ndarray, layer: QuantizedLayer, relu: bool) -> np.ndarray:
    # x: [B, Cin, L], weight: [Cout, Cin, K]
    bsz, cin, length = x.shape
    cout, wcin, kernel = layer.weight.shape
    if cin != wcin:
        raise ValueError(f"{layer.name}: input channels mismatch")
    out_len = (length + 2 * layer.padding - kernel) // layer.stride + 1
    padded = np.pad(x.astype(np.int32), ((0, 0), (0, 0), (layer.padding, layer.padding)))
    out = np.empty((bsz, cout, out_len), dtype=np.int8)
    w = layer.weight.astype(np.int32)
    for t in range(out_len):
        patch = padded[:, :, t * layer.stride : t * layer.stride + kernel]
        acc = np.tensordot(patch, w, axes=([1, 2], [1, 2])).astype(np.int64)
        acc += layer.bias[None, :].astype(np.int64)
        q = _requantize_int64(acc, layer)
        if relu:
            q = np.maximum(q, 0)
        out[:, :, t] = np.clip(q, -128, 127).astype(np.int8)
    return out


def _linear_int8(x: np.ndarray, layer: QuantizedLayer, relu: bool) -> np.ndarray:
    acc = x.astype(np.int32) @ layer.weight.astype(np.int32).T
    acc = acc.astype(np.int64) + layer.bias[None, :].astype(np.int64)
    q = _requantize_int64(acc, layer)
    if relu:
        q = np.maximum(q, 0)
    return np.clip(q, -128, 127).astype(np.int8)


def raw_int8_inference(bundle: RawINT8Bundle, x_float: np.ndarray) -> np.ndarray:
    """Reference for the simple C/C++ integer inference path."""

    x_float = np.asarray(x_float, dtype=np.float32)
    if x_float.ndim != 3 or x_float.shape[1] != 6 or x_float.shape[0] < 1:
        raise ValueError("x_float must have shape [samples, 6, input_length]")
    if not np.all(np.isfinite(x_float)):
        raise ValueError("x_float must contain only finite values")
    q = np.clip(np.rint(x_float / bundle.input_scale), -128, 127).astype(np.int8)
    q = _conv1d_int8(q, bundle.layers[0], relu=True)
    q = _conv1d_int8(q, bundle.layers[1], relu=True)
    q = _conv1d_int8(q, bundle.layers[2], relu=True)
    # Rounded integer global-average pool; scale remains conv3 output scale.
    q = np.clip(np.rint(q.astype(np.float32).mean(axis=2)), -128, 127).astype(np.int8)
    q = _linear_int8(q, bundle.layers[3], relu=True)
    q = _linear_int8(q, bundle.layers[4], relu=False)
    return q.astype(np.float32) * bundle.layers[-1].output_scale


def _c_array(name: str, array: np.ndarray, ctype: str, values_per_line: int = 16) -> str:
    flat = array.reshape(-1)
    lines = []
    for start in range(0, flat.size, values_per_line):
        values = ", ".join(str(int(v)) for v in flat[start : start + values_per_line])
        lines.append("    " + values)
    body = ",\n".join(lines)
    return f"static const {ctype} {name}[{flat.size}] = {{\n{body}\n}};\n"


def _decode_luts(bundle: RawINT8Bundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_scale = bundle.layers[-1].output_scale
    q = np.arange(-128, 128, dtype=np.float64)
    raw = q * raw_scale
    sigmoid = 1.0 / (1.0 + np.exp(-raw))
    softplus = np.logaddexp(0.0, raw)
    mean_q15 = np.clip(np.rint(sigmoid * 32767.0), 0, 32767).astype(np.uint16)
    scale_ns_q8 = np.clip(
        np.rint((softplus + bundle.min_scale_fraction) * bundle.delay_max_ns * 256.0),
        0,
        65535,
    ).astype(np.uint16)
    outlier_q8 = np.clip(np.rint(sigmoid * 255.0), 0, 255).astype(np.uint8)
    return mean_q15, scale_ns_q8, outlier_q8


def export_c_header(bundle: RawINT8Bundle, path: str | Path, namespace: str = "uwb_esp32") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_lut, scale_lut, outlier_lut = _decode_luts(bundle)
    pieces = [
        "#pragma once\n#include <stdint.h>\n#include <stddef.h>\n\n",
        f"namespace {namespace} {{\n",
        f"static constexpr float INPUT_SCALE = {bundle.input_scale:.12g}f;\n",
        f"static constexpr float DELAY_MAX_NS = {bundle.delay_max_ns:.12g}f;\n",
        f"static constexpr float MIN_SCALE_FRACTION = {bundle.min_scale_fraction:.12g}f;\n\n",
    ]
    for layer in bundle.layers:
        upper = layer.name.upper()
        pieces.extend(
            [
                f"// {layer.name}: weight shape {list(layer.weight.shape)}; per-output-channel quantization\n",
                f"static constexpr float {upper}_INPUT_SCALE = {layer.input_scale:.12g}f;\n",
                f"static constexpr float {upper}_OUTPUT_SCALE = {layer.output_scale:.12g}f;\n",
                f"static constexpr int {upper}_STRIDE = {layer.stride};\n",
                f"static constexpr int {upper}_PADDING = {layer.padding};\n",
                _c_array(f"{layer.name}_requant_multiplier_q31", layer.requant_multiplier_q31, "int32_t", values_per_line=8),
                _c_array(f"{layer.name}_requant_shift", layer.requant_shift, "uint8_t", values_per_line=16),
                _c_array(f"{layer.name}_weight", layer.weight, "int8_t"),
                _c_array(f"{layer.name}_bias", layer.bias, "int32_t", values_per_line=8),
                "\n",
            ]
        )
    pieces.extend(
        [
            "// LUT index = int8_raw + 128. Avoids sigmoid/softplus on MCU.\n",
            _c_array("mean_fraction_q15_lut", mean_lut, "uint16_t", values_per_line=12),
            _c_array("scale_ns_q8_lut", scale_lut, "uint16_t", values_per_line=12),
            _c_array("outlier_probability_q8_lut", outlier_lut, "uint8_t", values_per_line=16),
            f"}} // namespace {namespace}\n",
        ]
    )
    path.write_text("".join(pieces), encoding="utf-8")


def export_raw_binary(bundle: RawINT8Bundle, path: str | Path) -> dict:
    """Export a headerless weight blob plus a JSON manifest with offsets.

    The C-header path is easiest for firmware. The raw blob is the smallest
    standalone weight file if the application has the architecture compiled in.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "uwb-esp32-int8-v2-per-channel-fixedpoint",
        "input_scale": bundle.input_scale,
        "delay_max_ns": bundle.delay_max_ns,
        "min_scale_fraction": bundle.min_scale_fraction,
        "layers": [],
    }
    offset = 0
    with path.open("wb") as f:
        for layer in bundle.layers:
            layer_info: dict[str, object] = {
                "name": layer.name,
                "weight_shape": list(layer.weight.shape),
                "quantization": "symmetric_int8_per_output_channel",
                "weight_scale": layer.weight_scale.astype(float).tolist(),
                "input_scale": layer.input_scale,
                "output_scale": layer.output_scale,
                "multiplier": layer.multiplier.astype(float).tolist(),
                "requant_multiplier_q31": layer.requant_multiplier_q31.astype(int).tolist(),
                "requant_shift": layer.requant_shift.astype(int).tolist(),
                "stride": layer.stride,
                "padding": layer.padding,
                "weight_offset": offset,
                "weight_bytes": int(layer.weight.nbytes),
            }
            raw_w = layer.weight.tobytes(order="C")
            f.write(raw_w)
            offset += len(raw_w)
            layer_info["bias_offset"] = offset
            layer_info["bias_bytes"] = int(layer.bias.nbytes)
            raw_b = layer.bias.astype("<i4", copy=False).tobytes(order="C")
            f.write(raw_b)
            offset += len(raw_b)
            layer_info["requant_multiplier_q31_offset"] = offset
            raw_qmul = layer.requant_multiplier_q31.astype("<i4", copy=False).tobytes(order="C")
            layer_info["requant_multiplier_q31_bytes"] = len(raw_qmul)
            f.write(raw_qmul)
            offset += len(raw_qmul)
            layer_info["requant_shift_offset"] = offset
            raw_shift = layer.requant_shift.astype(np.uint8, copy=False).tobytes(order="C")
            layer_info["requant_shift_bytes"] = len(raw_shift)
            f.write(raw_shift)
            offset += len(raw_shift)
            cast_layers = manifest["layers"]
            assert isinstance(cast_layers, list)
            cast_layers.append(layer_info)
    manifest["total_bytes"] = offset
    manifest["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def export_onnx(model: ESP32ExportNet, input_length: int, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 6, input_length, dtype=torch.float32)
    try:
        torch.onnx.export(
            model.eval(),
            dummy,
            path,
            input_names=["uwb_features"],
            output_names=["raw_outputs"],
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    except Exception as exc:
        raise RuntimeError(
            "ONNX export failed. Install the training export extras: pip install onnx onnxscript"
        ) from exc


def export_espdl_from_onnx(
    onnx_path: str | Path,
    espdl_path: str | Path,
    calibration_x: np.ndarray,
    target: str = "esp32s3",
    calib_steps: int = 32,
    tqt: bool = True,
) -> None:
    """Use the official ESP-PPQ API when the optional package is installed."""

    try:
        from esp_ppq import QuantizationSettingFactory
        from esp_ppq.api import espdl_quantize_onnx
    except ImportError as exc:
        raise RuntimeError(
            "ESP-PPQ is not installed. Run `pip install esp-ppq` in the export environment."
        ) from exc

    quant_setting = QuantizationSettingFactory.espdl_setting()
    if tqt:
        quant_setting.tqt_optimization = True
        setting = quant_setting.tqt_optimization_setting
        setting.steps = 300
        setting.lr = 1e-5
        setting.collecting_device = "cpu"
        setting.block_size = 2

    if len(calibration_x) == 0:
        raise ValueError("calibration set is empty")
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.from_numpy(calibration_x.astype(np.float32, copy=False)))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    def collate_fn(batch):
        return batch[0].to("cpu")

    esp_target = "c" if target == "esp32" else target
    espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=dataloader,
        calib_steps=min(int(calib_steps), len(dataset)),
        input_shape=[1] + list(calibration_x.shape[1:]),
        target=esp_target,
        num_of_bits=8,
        collate_fn=collate_fn,
        setting=quant_setting,
        device="cpu",
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=0,
        inputs=None,
    )


def export_checkpoint(
    checkpoint: str | Path,
    calibration_x: np.ndarray,
    output_dir: str | Path,
    target: str = "esp32s3",
    export_onnx_file: bool = True,
    export_espdl_file: bool = False,
) -> dict:
    model, meta = load_student_checkpoint(checkpoint)
    export_model = ESP32ExportNet(model).eval()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = calibrate_and_quantize(
        export_model,
        calibration_x,
        delay_max_ns=float(meta["delay_max_ns"]),
        min_scale_fraction=float(meta.get("min_scale_fraction", 0.004)),
    )

    header_path = output / "ufuse_weights_int8.h"
    bin_path = output / "ufuse_weights_int8.bin"
    manifest_path = output / "ufuse_weights_manifest.json"
    export_c_header(bundle, header_path)
    manifest = export_raw_binary(bundle, bin_path)
    manifest.update(
        {
            "input_length": int(meta["input_length"]),
            "arch": meta["arch"],
            "activation_scales": bundle.activation_scales,
            "target": target,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Golden vectors: both float folded and integer reference outputs.
    golden_x = calibration_x[: min(8, len(calibration_x))].astype(np.float32, copy=False)
    with torch.inference_mode():
        float_raw = export_model(torch.from_numpy(golden_x)).numpy().astype(np.float32)
    int8_raw = raw_int8_inference(bundle, golden_x).astype(np.float32)
    np.savez_compressed(output / "golden_vectors.npz", input=golden_x, float_raw=float_raw, int8_raw=int8_raw)

    def _sigmoid(v: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-v))

    float_mean_ns = _sigmoid(float_raw[:, 0]) * bundle.delay_max_ns
    int8_mean_ns = _sigmoid(int8_raw[:, 0]) * bundle.delay_max_ns
    float_scale_ns = (np.logaddexp(0.0, float_raw[:, 1]) + bundle.min_scale_fraction) * bundle.delay_max_ns
    int8_scale_ns = (np.logaddexp(0.0, int8_raw[:, 1]) + bundle.min_scale_fraction) * bundle.delay_max_ns
    float_outlier = _sigmoid(float_raw[:, 2])
    int8_outlier = _sigmoid(int8_raw[:, 2])

    mean_lut, scale_lut, outlier_lut = _decode_luts(bundle)
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    report = {
        "checkpoint": str(checkpoint),
        "parameters": parameter_count,
        "quantization": "symmetric_int8_per_output_channel",
        "requantization": "q31_multiplier_plus_right_shift",
        "weight_blob_bytes": int(bin_path.stat().st_size),
        "fp32_parameter_bytes": int(parameter_count * 4),
        "weight_storage_reduction_vs_fp32": float(1.0 - bin_path.stat().st_size / max(parameter_count * 4, 1)),
        "decode_lut_bytes": int(mean_lut.nbytes + scale_lut.nbytes + outlier_lut.nbytes),
        "weight_blob_sha256": str(manifest["sha256"]),
        "header_bytes": int(header_path.stat().st_size),
        "float_vs_raw_int8_raw_mae": float(np.mean(np.abs(float_raw - int8_raw))),
        "float_vs_raw_int8_raw_max_abs": float(np.max(np.abs(float_raw - int8_raw))),
        "float_vs_raw_int8_mean_delay_mae_ns": float(np.mean(np.abs(float_mean_ns - int8_mean_ns))),
        "float_vs_raw_int8_scale_mae_ns": float(np.mean(np.abs(float_scale_ns - int8_scale_ns))),
        "float_vs_raw_int8_outlier_mae": float(np.mean(np.abs(float_outlier - int8_outlier))),
        "onnx": None,
        "espdl": None,
    }

    if export_onnx_file:
        onnx_path = output / "ufuse_esp32.onnx"
        export_onnx(export_model, int(meta["input_length"]), onnx_path)
        report["onnx"] = str(onnx_path)
    if export_espdl_file:
        if report["onnx"] is None:
            onnx_path = output / "ufuse_esp32.onnx"
            export_onnx(export_model, int(meta["input_length"]), onnx_path)
        else:
            onnx_path = Path(str(report["onnx"]))
        espdl_path = output / f"ufuse_s8_{target}.espdl"
        export_espdl_from_onnx(onnx_path, espdl_path, calibration_x, target=target)
        report["espdl"] = str(espdl_path)

    (output / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
