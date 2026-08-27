from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data import UWBData


def _minmax_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lo = np.min(x, axis=-1, keepdims=True)
    hi = np.max(x, axis=-1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, eps)


def _resample_rows(x: np.ndarray, old_grid: np.ndarray, length: int) -> np.ndarray:
    if x.shape[-1] == length:
        return x.astype(np.float32, copy=True)
    new_grid = np.linspace(float(old_grid[0]), float(old_grid[-1]), length)
    out = np.empty((*x.shape[:-1], length), dtype=np.float32)
    flat_in = x.reshape(-1, x.shape[-1])
    flat_out = out.reshape(-1, length)
    for i, row in enumerate(flat_in):
        flat_out[i] = np.interp(new_grid, old_grid, row).astype(np.float32)
    return out


def _u8_rows(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def _c_array(name: str, array: np.ndarray, ctype: str, per_line: int = 16) -> str:
    flat = array.reshape(-1)
    rows = []
    for start in range(0, len(flat), per_line):
        chunk = flat[start : start + per_line]
        if np.issubdtype(array.dtype, np.floating):
            values = ", ".join(f"{float(v):.9g}f" for v in chunk)
        else:
            values = ", ".join(str(int(v)) for v in chunk)
        rows.append("    " + values)
    return f"static const {ctype} {name}[{len(flat)}] = {{\n" + ",\n".join(rows) + "\n};\n"


def export_preprocess_and_geometry(
    data: UWBData,
    input_length: int,
    output_dir: str | Path,
) -> dict:
    """Export all non-learned constants needed by the ESP32 runtime."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cir_bg = _resample_rows(data.cir_background, data.delay_grid_ns, input_length)
    var_bg = _resample_rows(data.var_background, data.delay_grid_ns, input_length)
    cir_bg_n = _minmax_rows(np.abs(cir_bg))
    var_bg_n = _minmax_rows(np.maximum(var_bg, 0.0))
    cir_u8 = _u8_rows(cir_bg_n)
    var_u8 = _u8_rows(var_bg_n)

    # Smallest raw background blob: CIR followed by variance, both row-major.
    background_bin = output / "uwb_background_u8.bin"
    with background_bin.open("wb") as f:
        f.write(cir_u8.tobytes(order="C"))
        f.write(var_u8.tobytes(order="C"))

    header = output / "uwb_runtime_constants.h"
    text = [
        "#pragma once\n#include <stdint.h>\n\nnamespace uwb_esp32 {\n",
        f"static constexpr int NUM_LINKS = {data.num_links};\n",
        f"static constexpr int NUM_ANCHORS = {data.anchors.shape[0]};\n",
        f"static constexpr int INPUT_LENGTH = {input_length};\n",
        f"static constexpr float DELAY_START_NS = {float(data.delay_grid_ns[0]):.9g}f;\n",
        f"static constexpr float DELAY_MAX_NS = {float(data.delay_grid_ns[-1]):.9g}f;\n",
        "static constexpr float C_M_PER_NS = 0.299792458f;\n\n",
        _c_array("cir_background_u8", cir_u8, "uint8_t"),
        "\n",
        _c_array("var_background_u8", var_u8, "uint8_t"),
        "\n",
        _c_array("anchors_xy", data.anchors.astype(np.float32), "float", per_line=8),
        "\n",
        _c_array("link_pairs", data.link_pairs.astype(np.uint8), "uint8_t", per_line=12),
        "\n",
        _c_array("tof_los_ns", data.tof_los_ns.astype(np.float32), "float", per_line=8),
        "\n} // namespace uwb_esp32\n",
    ]
    header.write_text("".join(text), encoding="utf-8")

    manifest = {
        "format": "uwb-esp32-runtime-constants-v1",
        "input_length": input_length,
        "num_links": data.num_links,
        "num_anchors": int(data.anchors.shape[0]),
        "background_binary": {
            "path": background_bin.name,
            "bytes": background_bin.stat().st_size,
            "cir_offset": 0,
            "cir_bytes": int(cir_u8.nbytes),
            "variance_offset": int(cir_u8.nbytes),
            "variance_bytes": int(var_u8.nbytes),
            "shape": [data.num_links, input_length],
        },
        "preprocessing": {
            "dynamic_cir": "abs -> per-link minmax -> uint8[0,255]",
            "background_cir": "precomputed uint8 in export",
            "cir_difference": "abs(dynamic_u8-background_u8) -> per-link minmax to uint8",
            "dynamic_variance": "max(x,0) -> per-link minmax -> uint8[0,255]",
            "background_variance": "precomputed uint8 in export",
            "variance_difference": "abs(dynamic_u8-background_u8) -> per-link minmax to uint8",
            "network_channel_order": [
                "cir_dynamic",
                "cir_background",
                "cir_difference",
                "var_dynamic",
                "var_background",
                "var_difference",
            ],
        },
    }
    (output / "uwb_runtime_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "runtime_constants_header": str(header),
        "background_blob": str(background_bin),
        "background_blob_bytes": int(background_bin.stat().st_size),
    }
