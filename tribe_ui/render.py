from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from tqdm import tqdm

from tribev2.plotting import PlotBrain
from tribev2.plotting.utils import get_cmap, get_scalar_mappable


VIEW_VECTORS = {
    "left": ([-1, 0, 0], [0, 0, 1]),
    "right": ([1, 0, 0], [0, 0, 1]),
}


def _vertex_colors(
    plotter: PlotBrain,
    values: np.ndarray,
    hemi: str,
    sm,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = plotter._mesh[hemi]
    stat_map = plotter.get_stat_map(values)[hemi]
    rgba = sm.to_rgba(stat_map)

    bg_map = mesh["bg_map"]
    bg_norm = (bg_map - bg_map.min()) / (bg_map.max() - bg_map.min() + 1e-8)
    sulcal = 0.18 + 0.34 * bg_norm
    bg_rgb = np.column_stack([sulcal * 0.72, sulcal * 0.80, sulcal * 0.92])

    colors = rgba[:, :3] * 0.84 + bg_rgb * 0.16
    return mesh["coords"], mesh["faces"], np.clip(colors, 0, 1)


def _render_surface(
    plotter: PlotBrain,
    values: np.ndarray,
    sm,
    *,
    hemi: str,
    view: str,
    window_size: tuple[int, int],
) -> np.ndarray:
    vertices, faces, colors = _vertex_colors(plotter, values, hemi, sm)
    pv_faces = np.column_stack([np.full(len(faces), 3), faces])
    surf = pv.PolyData(vertices, pv_faces)
    surf.point_data["colors"] = colors

    pl = pv.Plotter(window_size=window_size, off_screen=True)
    pl.set_background("#090d12")
    pl.add_mesh(
        surf,
        scalars="colors",
        rgb=True,
        smooth_shading=True,
        ambient=0.22,
        diffuse=0.82,
        specular=0.18,
        specular_power=18,
    )
    pl.add_light(pv.Light(position=(0, -180, 160), intensity=0.55))

    vec, up = VIEW_VECTORS[view]
    pl.view_vector(vec, viewup=up)
    pl.camera.zoom(1.34)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def _compose_frame(
    left: np.ndarray,
    right: np.ndarray,
    sm,
    out_file: Path,
    *,
    timestamp: float,
    values: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#090d12")
    fig.patch.set_alpha(1)

    for ax_pos, image, label in (
        ([0.04, 0.14, 0.42, 0.72], left, "Left lateral"),
        ([0.50, 0.14, 0.42, 0.72], right, "Right lateral"),
    ):
        ax = fig.add_axes(ax_pos, facecolor="#101822")
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.04,
            0.08,
            label,
            color="#d9e2ec",
            fontsize=10,
            transform=ax.transAxes,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "#0b1118cc",
                "edgecolor": "#263544",
            },
        )

    cax = fig.add_axes([0.955, 0.19, 0.012, 0.62])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(colors="#aeb8c2", labelsize=8, length=0)
    cbar.outline.set_edgecolor("#2c3a46")
    cbar.set_label("Predicted response", color="#cdd6df", fontsize=9)

    abs_values = np.abs(values)
    p95 = float(np.percentile(abs_values, 95))
    fig.text(
        0.035,
        0.045,
        (
            f"t={timestamp:0.1f}s  |  cortical vertices: {values.size:,}"
            f"  |  frame abs(response) p95={p95:0.3f}"
        ),
        color="#b9c4ce",
        fontsize=10,
    )
    fig.text(
        0.04,
        0.91,
        "TRIBE v2 prediction",
        color="#f2f6f8",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.875,
        "Stable left/right lateral views. Color scale is symmetric and fixed across the run.",
        color="#93a1ad",
        fontsize=9,
    )
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def render_left_right_brain_video(
    preds_path: Path,
    out_path: Path,
    segments_path: Path | None = None,
    fps: int = 6,
    cmap: str = "coolwarm",
) -> Path:
    """Render an MP4 brain-response review with stable cortical views.

    The prediction values are not smoothed or rescaled per-frame. Colors use one
    symmetric scale from the run-wide 99th percentile, while the side panels keep
    stable lateral views for comparison.
    """

    preds = np.load(preds_path)
    segments = None
    if segments_path and segments_path.exists():
        with segments_path.open("rb") as f:
            segments = pickle.load(f)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / "tmp_left_right"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    lim = float(np.percentile(np.abs(preds), 99))
    if lim == 0:
        lim = 1.0

    plotter = PlotBrain(mesh="fsaverage5")
    cmap_obj = get_cmap(cmap)
    sm = get_scalar_mappable(
        preds,
        cmap_obj,
        vmin=-lim,
        vmax=lim,
        symmetric_cbar=True,
    )
    for i, values in enumerate(tqdm(preds, desc="Rendering stable brain views")):
        timestamp = float(i)
        if segments and i < len(segments):
            timestamp = float(getattr(segments[i], "start", i))
        left = _render_surface(
            plotter,
            values,
            sm,
            hemi="left",
            view="left",
            window_size=(760, 640),
        )
        right = _render_surface(
            plotter,
            values,
            sm,
            hemi="right",
            view="right",
            window_size=(760, 640),
        )
        _compose_frame(
            left,
            right,
            sm,
            tmp_dir / f"frame_{i:05d}.png",
            timestamp=timestamp,
            values=values,
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "1",
        "-i",
        str(tmp_dir / "frame_%05d.png"),
        "-vf",
        f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path
