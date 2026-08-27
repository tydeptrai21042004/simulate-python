from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from .data import geometry_tof

OFFICIAL_GITHUB_REPO = "https://github.com/CLongLi/UWB-Radar-Pedestrian-Tracking"
OFFICIAL_GITHUB_ZIP = (
    "https://github.com/CLongLi/UWB-Radar-Pedestrian-Tracking/"
    "archive/refs/heads/main.zip"
)
OFFICIAL_DYNAMIC_GDRIVE_ID = "1jo-PErF5nnqWJ8UUdzZv_OpWcDMesgxB"
LINKS = ("01", "02", "04", "12", "14", "24")
PAIRS = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)


def _download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "uwb-passive-tracking/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def _safe_extract_zip(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        if not members:
            raise RuntimeError(f"Downloaded archive is empty: {archive}")
        dest_resolved = destination.resolve()
        for member in members:
            candidate = (destination / member.filename).resolve()
            if dest_resolved not in candidate.parents and candidate != dest_resolved:
                raise RuntimeError(f"Unsafe path in downloaded zip: {member.filename}")
        zf.extractall(destination)
    top = members[0].filename.split("/", 1)[0]
    return destination / top


def download_official_repository(destination: str | Path, force: bool = False) -> Path:
    """Download the official GitHub repository without requiring git."""

    destination = Path(destination)
    required = ("AnchorPos.mat", "Bg_CIR_VAR.mat", "README.md")
    if not force and all((destination / name).exists() for name in required):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uwb_repo_") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "official_repo.zip"
        _download_url(OFFICIAL_GITHUB_ZIP, archive)
        extracted = _safe_extract_zip(archive, tmp_path / "extract")
        if destination.exists() and force:
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(extracted, destination, dirs_exist_ok=True)

    missing = [name for name in required if not (destination / name).exists()]
    if missing:
        raise RuntimeError(f"Official GitHub download is incomplete; missing {missing}")
    return destination


def download_dynamic_mat(destination: str | Path, force: bool = False) -> Path:
    """Download Dyn_CIR_VAR.mat from the Google Drive link in the official README."""

    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 1024 and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        raise RuntimeError(
            "Automatic Dyn_CIR_VAR.mat download requires gdown. "
            "Install project requirements or run: pip install gdown>=5.2"
        ) from exc

    result = gdown.download(
        id=OFFICIAL_DYNAMIC_GDRIVE_ID,
        output=str(destination),
        quiet=False,
    )
    if result is None or not destination.exists() or destination.stat().st_size <= 1024:
        raise RuntimeError(
            "Could not download Dyn_CIR_VAR.mat from the author-provided Google Drive link. "
            "The link may require manual access; see the official repository README."
        )
    return destination


def _merged(*paths: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        out.update({k: v for k, v in loadmat(path, squeeze_me=True).items() if not k.startswith("__")})
    return out


def _find_key(raw: dict[str, object], candidates: list[str]) -> str:
    lower = {k.lower(): k for k in raw}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(f"Missing variable. Tried {candidates}. Available examples: {list(raw)[:30]}")


def _get(raw: dict[str, object], candidates: list[str]) -> np.ndarray:
    return np.asarray(raw[_find_key(raw, candidates)])


def _anchors_xy(value: np.ndarray) -> np.ndarray:
    anchors = np.asarray(value, dtype=np.float64)
    if anchors.ndim != 2:
        raise ValueError(f"AnchorPos must be 2-D, got {anchors.shape}")
    # Official file is 4x3 (x,y,z); the published PF explicitly uses x/y only.
    if anchors.shape[0] == 4 and anchors.shape[1] >= 2:
        return anchors[:, :2]
    if anchors.shape[1] == 4 and anchors.shape[0] >= 2:
        return anchors[:2, :].T
    raise ValueError(f"Expected four anchors with at least x/y coordinates, got {anchors.shape}")


def _time_major(values: np.ndarray, n_time: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        if arr.size != n_time:
            raise ValueError(f"{name}: expected {n_time} samples, got {arr.shape}")
        return arr.reshape(n_time, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D time/profile array, got {arr.shape}")
    if arr.shape[0] == n_time:
        return arr
    if arr.shape[1] == n_time:
        return arr.T
    raise ValueError(f"{name}: neither axis matches time length {n_time}; got {arr.shape}")


def _interp_columns(time_old: np.ndarray, values: np.ndarray, time_new: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    result = np.empty((time_new.size, values.shape[1]), dtype=np.float64)
    for col in range(values.shape[1]):
        result[:, col] = np.interp(time_new, time_old, values[:, col])
    return result


def convert_official_matlab_data(
    background: str | Path,
    dynamic: str | Path,
    anchors_file: str | Path,
    output: str | Path,
) -> Path:
    """Convert the official MATLAB variables to the project's standard MAT schema.

    Important paper-faithful corrections made here:
    - official AnchorPos is 4x3, but the MATLAB PF uses only x/y;
    - official CIR arrays are complex and are converted with abs(CIR), matching
      the MATLAB `mat2gray(abs(...))` input path instead of discarding phase;
    - official variance variables are named `Dyn_var_CIRxx` / `Bg_var_CIRxx`.
    """

    background = Path(background)
    dynamic = Path(dynamic)
    anchors_file = Path(anchors_file)
    output = Path(output)
    raw = _merged(background, dynamic, anchors_file)

    anchors = _anchors_xy(_get(raw, ["AnchorPos", "anchors"]))
    delay_grid = _get(raw, ["re_SampTime", "delay_grid_ns"]).reshape(-1).astype(np.float64)
    if delay_grid.size < 1:
        raise ValueError("re_SampTime/delay grid is empty")

    times: list[np.ndarray] = []
    cirs: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    mus: list[np.ndarray] = []
    source_tofs: list[np.ndarray] = []
    cir_bg: list[np.ndarray] = []
    var_bg: list[np.ndarray] = []
    los: list[float] = []

    for link in LINKS:
        time = _get(raw, [f"Dyn_re_tUWB{link}"]).reshape(-1).astype(np.float64)
        if time.size < 2:
            raise ValueError(f"Dyn_re_tUWB{link} must contain at least two timestamps")

        cir_raw = _time_major(_get(raw, [f"Dyn_re_CIR{link}"]), time.size, f"Dyn_re_CIR{link}")
        # This abs() is required for parity with the official MATLAB scripts.
        cir = np.abs(cir_raw).astype(np.float64)

        var = _time_major(
            _get(
                raw,
                [
                    f"Dyn_var_CIR{link}",
                    f"Dyn_re_VAR{link}",
                    f"Dyn_re_CIRVar{link}",
                    f"Dyn_re_Var{link}",
                ],
            ),
            time.size,
            f"Dyn_var_CIR{link}",
        ).astype(np.float64)
        mu = _time_major(_get(raw, [f"Dyn_re_MU{link}"]), time.size, f"Dyn_re_MU{link}").astype(np.float64)
        if mu.shape[1] < 2:
            raise ValueError(f"Dyn_re_MU{link} must include x/y coordinates")
        tof = _get(raw, [f"Dyn_real_ToF{link}"]).reshape(-1).astype(np.float64)
        if tof.size != time.size:
            raise ValueError(f"Dyn_real_ToF{link}: expected {time.size} values, got {tof.size}")

        bg_cir_raw = _get(raw, [f"Bg_re_CIR{link}"]).reshape(-1)
        bg_var_raw = _get(
            raw,
            [f"Bg_var_CIR{link}", f"Bg_re_VAR{link}", f"Bg_re_CIRVar{link}", f"Bg_re_Var{link}"],
        ).reshape(-1)
        if bg_cir_raw.size != delay_grid.size or bg_var_raw.size != delay_grid.size:
            raise ValueError(
                f"Background profile length for link {link} does not match delay grid "
                f"({bg_cir_raw.size}, {bg_var_raw.size}) vs {delay_grid.size}"
            )
        if cir.shape[1] != delay_grid.size or var.shape[1] != delay_grid.size:
            raise ValueError(
                f"Dynamic profile length for link {link} does not match delay grid "
                f"({cir.shape[1]}, {var.shape[1]}) vs {delay_grid.size}"
            )

        times.append(time)
        cirs.append(cir)
        variances.append(np.maximum(var, 0.0))
        mus.append(mu[:, :2])
        source_tofs.append(tof)
        cir_bg.append(np.abs(bg_cir_raw).astype(np.float32))
        var_bg.append(np.maximum(np.real(bg_var_raw), 0.0).astype(np.float32))
        los.append(float(_get(raw, [f"ToF_TRx{link}"]).reshape(-1)[0]))

    start = max(float(t.min()) for t in times)
    end = min(float(t.max()) for t in times)
    if not end > start:
        raise ValueError("The six official link timelines do not overlap")
    diffs = np.concatenate([np.diff(t) for t in times])
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        raise ValueError("Could not infer a positive common sampling interval")
    dt = float(np.median(diffs))
    common_time = np.arange(start, end + 0.5 * dt, dt, dtype=np.float64)

    t_count, l_count, b_count = common_time.size, len(LINKS), delay_grid.size
    cir_dynamic = np.empty((t_count, l_count, b_count), dtype=np.float32)
    var_dynamic = np.empty_like(cir_dynamic)
    tof_source_interp = np.empty((t_count, l_count), dtype=np.float64)
    xy_per_link: list[np.ndarray] = []

    for link_idx in range(l_count):
        cir_dynamic[:, link_idx] = _interp_columns(times[link_idx], cirs[link_idx], common_time).astype(np.float32)
        var_dynamic[:, link_idx] = _interp_columns(times[link_idx], variances[link_idx], common_time).astype(np.float32)
        tof_source_interp[:, link_idx] = np.interp(common_time, times[link_idx], source_tofs[link_idx])
        xy_per_link.append(_interp_columns(times[link_idx], mus[link_idx], common_time))

    trajectory_xy = np.mean(np.stack(xy_per_link, axis=0), axis=0)
    # Use the same 2-D ellipse geometry as ParticleFilter4Nodes.m. Keeping a
    # geometry-consistent target also prevents tiny interpolation inconsistencies
    # between per-link MU timelines from failing downstream validation.
    tof_total = geometry_tof(trajectory_xy, anchors, PAIRS)

    output.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output,
        {
            "anchors": anchors,
            "link_pairs": PAIRS + 1,
            "time_s": common_time,
            "delay_grid_ns": delay_grid,
            "trajectory_xy": trajectory_xy,
            "tof_total_ns": tof_total,
            "tof_source_interpolated_ns": tof_source_interp,
            "tof_los_ns": np.asarray(los, dtype=np.float64),
            "cir_background": np.stack(cir_bg).astype(np.float32),
            "var_background": np.stack(var_bg).astype(np.float32),
            "cir_dynamic": cir_dynamic,
            "var_dynamic": var_dynamic,
            "description": (
                "Converted from CLongLi/UWB-Radar-Pedestrian-Tracking. "
                "Complex CIR is magnitude-converted and 4x3 anchors are reduced to XY "
                "to match the official MATLAB particle filter."
            ),
        },
        do_compression=True,
    )
    return output


def ensure_official_standard_dataset(
    output: str | Path,
    source_dir: str | Path,
    force_download: bool = False,
    force_convert: bool = False,
) -> Path:
    """Auto-download official sources and create the standard MAT dataset."""

    output = Path(output)
    if output.exists() and not force_convert:
        return output
    source_dir = download_official_repository(source_dir, force=force_download)
    dynamic = download_dynamic_mat(source_dir / "Dyn_CIR_VAR.mat", force=force_download)
    return convert_official_matlab_data(
        source_dir / "Bg_CIR_VAR.mat",
        dynamic,
        source_dir / "AnchorPos.mat",
        output,
    )
