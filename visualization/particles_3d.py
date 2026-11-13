"""3D particle visualisation (Matplotlib, no pyplot state).

This module is deliberately decoupled from the physics code: it consumes plain
NumPy arrays only and never imports anything from ``src/``. Anything that can
produce snapshots of shape ``(F, N, 3)`` -- a live simulation, a saved ``.npz``,
another code's output -- can be rendered with it.

Public API
----------
* :func:`speed`, :func:`local_density`, :func:`color_values`: scalar fields used
  for colouring particles.
* :func:`render_animation`: write a ``.gif`` or ``.mp4`` of the evolution.
* :func:`render_montage`: write a still image comparing several snapshots.
* :func:`ffmpeg_available`: whether ``.mp4`` output is possible.

Colour limits and axis limits are fixed across all frames so colours and
distances mean the same thing throughout the animation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3DType
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ColorBy = Literal["speed", "density"]

_BG = "black"
_FG = "0.85"


# ---------------------------------------------------------------------- #
# Scalar fields used for colouring
# ---------------------------------------------------------------------- #
def speed(velocities: FloatArray) -> FloatArray:
    """Velocity magnitude ``|v|`` along the last axis."""
    return np.sqrt(np.einsum("...k,...k->...", velocities, velocities))


def local_density(positions: FloatArray, k: int = 16) -> FloatArray:
    """``log10`` of a k-nearest-neighbour density estimate for one snapshot.

    ``rho_i ~ k / ((4/3) pi r_k^3)`` where ``r_k`` is the distance from particle
    ``i`` to its ``k``-th nearest neighbour (unit particle mass; only relative
    values matter for colouring). Fully vectorised, ``O(N^2)`` per snapshot.

    Args:
        positions: ``(N, 3)`` positions.
        k: Number of neighbours (must satisfy ``1 <= k < N``).
    """
    n = positions.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < N={n}; got {k}")
    xt = np.ascontiguousarray(positions.T)
    dx = xt[:, :, None] - xt[:, None, :]
    r2 = np.einsum("kij,kij->ij", dx, dx)
    r2[np.arange(n), np.arange(n)] = np.inf  # exclude self
    rk2 = np.partition(r2, k - 1, axis=1)[:, k - 1]
    rk = np.sqrt(np.maximum(rk2, 1e-30))
    return np.log10(k / (4.0 / 3.0 * np.pi * rk**3))


def color_values(
    positions: FloatArray,
    velocities: FloatArray | None,
    color_by: ColorBy,
    density_neighbors: int = 16,
) -> FloatArray:
    """Per-particle colour scalar for every frame, shape ``(F, N)``."""
    if color_by == "speed":
        if velocities is None:
            raise ValueError("velocities are required for color_by='speed'")
        return speed(velocities)
    if color_by == "density":
        return np.stack([local_density(frame, density_neighbors) for frame in positions])
    raise ValueError(f"unknown color_by: {color_by!r}")


def ffmpeg_available() -> bool:
    """Whether Matplotlib can find an ``ffmpeg`` binary for ``.mp4`` output."""
    return bool(animation.writers.is_available("ffmpeg"))


# ---------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnimationConfig:
    """Rendering options.

    Attributes:
        color_by: ``"speed"`` (|v|) or ``"density"`` (kNN local density).
        cmap: Matplotlib colormap name.
        point_size: Scatter marker area in points^2.
        alpha: Marker opacity.
        fps: Frames per second of the output movie.
        dpi: Output resolution.
        figsize: Figure size in inches (keep ``figsize * dpi`` even for mp4).
        elev, azim: Initial camera angles in degrees.
        rotate_deg: Total camera azimuth rotation over the animation.
        limit: Half-width of the cubic view volume, centred on the origin. If
            ``None`` it is set from the 99.5th percentile of ``|x|`` over all
            frames so a few ejected particles do not shrink the view.
        density_neighbors: ``k`` for the density estimate.
        time_scale: Times are displayed as ``t / time_scale``.
        time_label: Label shown next to the time readout.
        color_label: Colorbar label (defaults depend on ``color_by``).
    """

    color_by: ColorBy = "speed"
    cmap: str = "plasma"
    point_size: float = 6.0
    alpha: float = 0.85
    fps: int = 24
    dpi: int = 80
    figsize: tuple[float, float] = (7.0, 6.5)
    elev: float = 20.0
    azim: float = 35.0
    rotate_deg: float = 90.0
    limit: float | None = None
    density_neighbors: int = 16
    time_scale: float = 1.0
    time_label: str = "t"
    color_label: str | None = None


@dataclass(frozen=True)
class _SceneData:
    colors: FloatArray  # (F, N)
    vmin: float
    vmax: float
    limit: float
    color_label: str


def _prepare(
    positions: FloatArray, velocities: FloatArray | None, times: FloatArray, cfg: AnimationConfig
) -> _SceneData:
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(f"positions must have shape (F, N, 3); got {positions.shape}")
    if times.shape != (positions.shape[0],):
        raise ValueError(f"times must have shape ({positions.shape[0]},); got {times.shape}")
    if velocities is not None and velocities.shape != positions.shape:
        raise ValueError("velocities must have the same shape as positions")

    colors = color_values(positions, velocities, cfg.color_by, cfg.density_neighbors)
    vmin, vmax = (float(v) for v in np.percentile(colors, (1.0, 99.0)))
    if vmax <= vmin:
        vmax = vmin + 1.0
    limit = cfg.limit
    if limit is None:
        limit = 1.05 * float(np.percentile(np.abs(positions), 99.5))
    default_label = "|v|" if cfg.color_by == "speed" else "log10 local density (kNN)"
    return _SceneData(colors, vmin, vmax, limit, cfg.color_label or default_label)


# ---------------------------------------------------------------------- #
# Scene construction helpers
# ---------------------------------------------------------------------- #
def _style_axes(ax: Axes3DType, limit: float, cfg: AnimationConfig) -> None:
    ax.set_facecolor(_BG)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.04, 0.04, 0.07, 1.0))
    ax.tick_params(colors="0.6", labelsize=7)
    ax.set_xlabel("x", color=_FG)
    ax.set_ylabel("y", color=_FG)
    ax.set_zlabel("z", color=_FG)
    ax.view_init(elev=cfg.elev, azim=cfg.azim)


def _add_scatter(
    ax: Axes3DType, pos: FloatArray, colors: FloatArray, scene: _SceneData, cfg: AnimationConfig
) -> PathCollection:
    return ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=colors, cmap=cfg.cmap, vmin=scene.vmin, vmax=scene.vmax,
        s=cfg.point_size, alpha=cfg.alpha, depthshade=False, linewidths=0,
    )


def _add_colorbar(fig: Figure, mappable: PathCollection, axes: Axes3DType | list[Axes3DType],
                  label: str) -> None:
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label(label, color=_FG)
    cbar.ax.tick_params(colors=_FG)
    cbar.outline.set_edgecolor("0.4")  # type: ignore[union-attr]


def _time_text(t: float, cfg: AnimationConfig) -> str:
    return f"{cfg.time_label} = {t / cfg.time_scale:.2f}"


# ---------------------------------------------------------------------- #
# Public renderers
# ---------------------------------------------------------------------- #
def render_animation(
    positions: FloatArray,
    velocities: FloatArray | None,
    times: FloatArray,
    output_path: str | Path,
    config: AnimationConfig | None = None,
) -> Path:
    """Render the evolution to a ``.gif`` or ``.mp4`` file.

    Args:
        positions: ``(F, N, 3)`` snapshots.
        velocities: ``(F, N, 3)`` velocities (may be ``None`` if colouring by density).
        times: ``(F,)`` snapshot times.
        output_path: Destination; the suffix (``.gif`` or ``.mp4``) selects the writer.
        config: Rendering options.

    Returns:
        The path written.
    """
    cfg = config or AnimationConfig()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene = _prepare(positions, velocities, times, cfg)
    n_frames = positions.shape[0]

    suffix = out.suffix.lower()
    writer: animation.AbstractMovieWriter
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=cfg.fps)
    elif suffix == ".mp4":
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg not found; cannot write .mp4 (use .gif instead)")
        writer = animation.FFMpegWriter(
            fps=cfg.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]
        )
    else:
        raise ValueError(f"unsupported output format {suffix!r}; use .gif or .mp4")

    fig = Figure(figsize=cfg.figsize, facecolor=_BG)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    _style_axes(ax, scene.limit, cfg)
    scatter = _add_scatter(ax, positions[0], scene.colors[0], scene, cfg)
    _add_colorbar(fig, scatter, ax, scene.color_label)
    label = ax.text2D(0.04, 0.94, _time_text(times[0], cfg), transform=ax.transAxes,
                      color="white", fontsize=12)

    def update(frame: int) -> tuple[PathCollection]:
        pos = positions[frame]
        scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])  # type: ignore[attr-defined]
        scatter.set_array(scene.colors[frame])
        label.set_text(_time_text(times[frame], cfg))
        frac = frame / max(n_frames - 1, 1)
        ax.view_init(elev=cfg.elev, azim=cfg.azim + cfg.rotate_deg * frac)
        return (scatter,)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(str(out), writer=writer, dpi=cfg.dpi, savefig_kwargs={"facecolor": _BG})
    return out


def render_montage(
    positions: FloatArray,
    velocities: FloatArray | None,
    times: FloatArray,
    frames: Sequence[int],
    output_path: str | Path,
    config: AnimationConfig | None = None,
) -> Path:
    """Render a still image with one 3D panel per selected frame index."""
    cfg = config or AnimationConfig()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene = _prepare(positions, velocities, times, cfg)

    fig = Figure(figsize=(5.0 * len(frames), 5.6), facecolor=_BG)
    FigureCanvasAgg(fig)
    axes: list[Axes3DType] = []
    last: PathCollection | None = None
    for i, frame in enumerate(frames):
        ax = fig.add_subplot(1, len(frames), i + 1, projection="3d")
        _style_axes(ax, scene.limit, cfg)
        last = _add_scatter(ax, positions[frame], scene.colors[frame], scene, cfg)
        ax.set_title(_time_text(times[frame], cfg), color="white", fontsize=12)
        axes.append(ax)
    assert last is not None, "frames must be non-empty"
    _add_colorbar(fig, last, axes, scene.color_label)
    fig.savefig(out, dpi=max(cfg.dpi, 100), facecolor=_BG)
    return out
