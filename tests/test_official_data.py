from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from uwb_tracking.data import geometry_tof, load_uwb_mat
from uwb_tracking.official_data import (
    LINKS,
    PAIRS,
    convert_official_matlab_data,
    download_official_repository,
)


def _write_official_style_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    anchors3 = np.array(
        [
            [-2.0, 0.0, 1.1],
            [2.0, 0.0, 1.2],
            [2.0, -3.0, 1.2],
            [-2.0, -3.0, 1.1],
        ],
        dtype=np.float64,
    )
    anchors_xy = anchors3[:, :2]
    trajectory = np.array(
        [[-0.8, -0.5], [-0.4, -0.8], [0.0, -1.0], [0.4, -1.2], [0.8, -1.5]],
        dtype=np.float64,
    )
    time = np.arange(len(trajectory), dtype=np.float64) * 0.1
    delay = np.linspace(0.0, 3.0, 4)
    total = geometry_tof(trajectory, anchors_xy, PAIRS)

    bg: dict[str, object] = {}
    dyn: dict[str, object] = {"re_SampTime": delay}
    for link_idx, link in enumerate(LINKS):
        bg[f"ToF_TRx{link}"] = 10.0 + link_idx
        bg[f"Bg_re_CIR{link}"] = np.array([3 + 4j, 1 + 1j, 2 - 2j, 4 + 0j])
        bg[f"Bg_var_CIR{link}"] = np.arange(4, dtype=np.float64) + link_idx
        dyn[f"Dyn_re_tUWB{link}"] = time
        dyn[f"Dyn_re_CIR{link}"] = np.tile(
            np.array([3 + 4j, 1 + 1j, 2 - 2j, 4 + 0j]), (len(time), 1)
        )
        dyn[f"Dyn_var_CIR{link}"] = np.tile(np.arange(4, dtype=np.float64), (len(time), 1))
        dyn[f"Dyn_re_MU{link}"] = np.c_[trajectory, np.zeros(len(time))]
        dyn[f"Dyn_real_ToF{link}"] = total[:, link_idx]

    anchors_path = tmp_path / "AnchorPos.mat"
    bg_path = tmp_path / "Bg_CIR_VAR.mat"
    dyn_path = tmp_path / "Dyn_CIR_VAR.mat"
    savemat(anchors_path, {"AnchorPos": anchors3})
    savemat(bg_path, bg)
    savemat(dyn_path, dyn)
    return bg_path, dyn_path, anchors_path


def test_official_converter_handles_actual_schema_complex_cir_and_xyz_anchors(tmp_path: Path):
    bg, dyn, anchors = _write_official_style_files(tmp_path)
    out = convert_official_matlab_data(bg, dyn, anchors, tmp_path / "standard.mat")
    data = load_uwb_mat(out)
    assert data.anchors.shape == (4, 2)
    assert data.cir_background.shape == (6, 4)
    assert data.var_background.shape == (6, 4)
    assert data.cir_dynamic.shape[1:] == (6, 4)
    # abs(3+4j)=5; a float cast of the complex value would incorrectly produce 3.
    assert np.isclose(data.cir_background[0, 0], 5.0)
    assert np.isclose(data.cir_dynamic[0, 0, 0], 5.0)
    data.validate()


def test_official_repo_download_is_skipped_when_required_files_exist(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("AnchorPos.mat", "Bg_CIR_VAR.mat", "README.md"):
        (repo / name).write_bytes(b"present")
    assert download_official_repository(repo) == repo
