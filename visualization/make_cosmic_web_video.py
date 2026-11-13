"""Render the Cosmic Web Formation videos from the saved simulation snapshots.

Run from the project root::

    python -m visualization.make_cosmic_web_video --mode evolution
    python -m visualization.make_cosmic_web_video --mode cinematic
    python -m visualization.make_cosmic_web_video --mode both          # default

This script only *consumes* ``results/snapshots.npz``; it never runs or
modifies the simulation.

Two outputs, with deliberately different scientific status
----------------------------------------------------------
``results/cosmic_web_evolution.mp4``  (scientific)
    Only the stored snapshots are shown, one after another (hard cuts, no
    interpolation, no smoothing). One fixed global colour scale.

``results/cosmic_web_cinematic.mp4``  (visualization only)
    Frames *between* snapshots are a cross-fade of the two neighbouring
    snapshots, paced in scale factor ``a = 1/(1+z)`` with smoothstep easing.
    They are NOT simulated states and no physics was integrated for them.
    Mild Gaussian smoothing, a restrained unsharp mask and a slow zoom are
    applied for presentation. The redshift/age readout on those frames is
    simply the position on the eased timeline.

Both modes plot ``log10(1 + delta)`` of a central slice of the stored mesh
density, with one fixed colour range computed from the whole dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.ndimage import gaussian_filter, zoom  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # make `src.*` importable however we are launched
    sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ constants
W, H, DPI = 1920, 1080, 100          # 16:9 frame
MAP_PX = 1000                        # square map, centred; text lives in the side bars
BG, FG, DIM, FAINT = "#05070b", "#e9ecf3", "#8a93a6", "#3d4555"
MONO = "DejaVu Sans Mono"
FLOOR = 1e-2                         # floor on (1+delta) before log10 (empty mesh cells)
SCALE_BAR = 20.0                     # Mpc/h
GYR_PER_MPC_PER_KMS = 3.0856775814913673e19 / 3.15576e16  # 1 Mpc/(km/s) in Gyr
Z_NAMES = ("redshifts", "redshift", "z", "zs", "snapshot_redshifts", "z_snapshots")
DENSITY_HINTS = ("dens", "rho", "delta", "field", "slice", "snap")
META_ALIASES = {
    "box_size": ("box_size", "boxsize", "box", "box_length", "L"),
    "H0": ("H0",),
    "Omega_m": ("Omega_m", "omega_m", "Om", "omega_matter"),
    "Omega_Lambda": ("Omega_Lambda", "omega_lambda", "OmegaL", "Omega_L", "omega_l"),
}


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------- loading
@dataclass
class Snapshots:
    z: np.ndarray            # (n,) descending redshift
    fields: list             # n arrays, each (M, M, M) or (M, M)
    meta: dict               # scalar metadata found in the file (may be empty)

    @property
    def a(self) -> np.ndarray:
        return 1.0 / (1.0 + self.z)


def _is_number_array(a: np.ndarray) -> bool:
    return a.dtype.kind in "fiu"


def _square(shape: tuple) -> bool:
    return len(shape) >= 2 and len(set(shape)) == 1


def load_snapshots(path: Path, density_key: str | None, redshift_key: str | None) -> Snapshots:
    """
    Load simulation snapshots from results/snapshots.npz.

    Supported layouts:

    Layout A:
        redshifts + one stacked density array
        shape (n, M, M) or (n, M, M, M)

    Layout B:
        one density array per snapshot with the redshift encoded
        in the key name, e.g. density_z49, density_z0.5

    Layout C:
        redshifts + slice_0, slice_1, ..., slice_N
        where each slice is already a 2-D (M, M) visualization field.

    The actual Cosmic Web Formation Simulator snapshots.npz uses Layout C.
    """

    if not path.is_file():
        raise SystemExit(
            f"error: snapshot file not found: {path}\n"
            "Run the simulation first so that it writes results/snapshots.npz "
            "(this script never runs the simulation itself)."
        )

    with np.load(path, allow_pickle=False) as npz:
        arrs = {k: npz[k] for k in npz.files}

    print(f"Inspecting {path}")
    for k, a in arrs.items():
        print(f"  {k:<24s} shape={a.shape}  dtype={a.dtype}")

    # --------------------------------------------------------------- metadata
    meta = {}
    lower = {k.lower(): k for k in arrs}

    for canon, aliases in META_ALIASES.items():
        for name in aliases:
            k = lower.get(name.lower())
            if k is not None and arrs[k].size == 1 and _is_number_array(arrs[k]):
                meta[canon] = float(arrs[k].reshape(-1)[0])
                break

    z, fields, how = None, None, ""

    # ------------------------------------------------------------- redshifts
    z_key = redshift_key or next(
        (
            k
            for k, a in arrs.items()
            if k.lower() in Z_NAMES
            and a.ndim == 1
            and _is_number_array(a)
        ),
        None,
    )

    if z_key is not None:
        if z_key not in arrs:
            raise SystemExit(
                f"error: --redshift-key {z_key!r} not in file "
                f"(keys: {list(arrs)})"
            )

        zz = arrs[z_key].astype(np.float64).ravel()

        # ---------------------------------------------------------
        # Layout A: stacked density array
        # ---------------------------------------------------------
        cands = [
            k
            for k, a in arrs.items()
            if k != z_key
            and _is_number_array(a)
            and a.ndim in (3, 4)
            and a.shape[0] == zz.size
            and _square(a.shape[1:])
        ]

        if density_key:
            if density_key not in arrs:
                raise SystemExit(
                    f"error: --density-key {density_key!r} not in file "
                    f"(keys: {list(arrs)})"
                )
            cands = [density_key]

        if len(cands) > 1:
            ranked = sorted(
                cands,
                key=lambda k: next(
                    (
                        i
                        for i, h in enumerate(DENSITY_HINTS)
                        if h in k.lower()
                    ),
                    99,
                ),
            )

            if (
                next(
                    (
                        i
                        for i, h in enumerate(DENSITY_HINTS)
                        if h in ranked[0].lower()
                    ),
                    99,
                )
                == 99
            ):
                raise SystemExit(
                    f"error: several candidate density arrays {cands}; "
                    "choose with --density-key"
                )

            cands = ranked[:1]

        if cands:
            stack = arrs[cands[0]]
            z = zz
            fields = [stack[i] for i in range(zz.size)]
            how = (
                f"'{cands[0]}' stacked, "
                f"redshifts from '{z_key}'"
            )

        # ---------------------------------------------------------
        # Layout C: redshifts + slice_0, slice_1, ...
        #
        # This is the format used by the actual project snapshots.
        # ---------------------------------------------------------
        if z is None:
            slice_pattern = re.compile(r"^slice_(\d+)$", re.I)

            found_slices = []

            for k, a in arrs.items():
                m = slice_pattern.match(k)

                if (
                    m
                    and _is_number_array(a)
                    and a.ndim == 2
                    and _square(a.shape)
                ):
                    found_slices.append((int(m.group(1)), k))

            found_slices.sort(key=lambda x: x[0])

            if found_slices:
                indices = [idx for idx, _ in found_slices]

                expected_indices = list(range(zz.size))

                if indices != expected_indices:
                    raise SystemExit(
                        "error: slice arrays do not form a complete "
                        f"slice_0 ... slice_{zz.size - 1} sequence.\n"
                        f"Found slice indices: {indices}\n"
                        f"Expected: {expected_indices}"
                    )

                fields = [arrs[k] for _, k in found_slices]
                z = zz

                how = (
                    f"'{z_key}' + slice_0 ... "
                    f"slice_{zz.size - 1} (stored 2-D slices)"
                )

    # ------------------------------------------------------------- Layout B
    # One array per snapshot with redshift encoded in key name.
    #
    # Examples:
    #     density_z49
    #     density_z20
    #     density_z0.5
    #
    # This is only used if Layout A/C did not find anything.
    # -------------------------------------------------------------
    if z is None:
        pat = re.compile(
            r"z[_=]?(\d+(?:[.p]\d+)?)",
            re.I
        )

        found = []

        for k, a in arrs.items():
            m = pat.search(k)

            if (
                m
                and _is_number_array(a)
                and a.ndim in (2, 3)
                and _square(a.shape)
            ):
                found.append(
                    (
                        float(m.group(1).replace("p", ".")),
                        k,
                    )
                )

        if len(found) >= 2:
            z = np.array([f[0] for f in found])
            fields = [arrs[f[1]] for f in found]

            how = (
                f"one array per snapshot "
                f"({found[0][1]}, ...), "
                "redshift parsed from key names"
            )

    # --------------------------------------------------------------- failure
    if z is None:
        raise SystemExit(
            "error: could not find snapshot density data in the NPZ.\n"
            "\n"
            "Supported layouts are:\n"
            "  1. A 1-D redshift array plus a stacked density array "
            "of shape (n,M,M) or (n,M,M,M).\n"
            "  2. One square 2-D/3-D array per snapshot with "
            "'z<value>' in its key.\n"
            "  3. A 1-D redshift array plus slice_0, slice_1, ... "
            "where each slice is a 2-D square field.\n"
            "\n"
            "Use --redshift-key / --density-key to point at the "
            "appropriate arrays."
        )

    # -------------------------------------------------------------- validation
    if z.size < 2:
        raise SystemExit(
            "error: need at least 2 snapshots to make a video."
        )

    if not np.all(np.isfinite(z)) or z.min() < -1e-9:
        raise SystemExit(
            f"error: invalid redshifts in file: {z}"
        )

    z = np.clip(z, 0.0, None)

    # Sort from high redshift -> present day.
    order = np.argsort(-z)

    z = z[order]
    fields = [fields[i] for i in order]

    if len(np.unique(np.round(z, 9))) != z.size:
        raise SystemExit(
            f"error: duplicate redshifts in file: {z}"
        )

    if len({f.shape for f in fields}) != 1:
        raise SystemExit(
            "error: snapshot fields have inconsistent shapes."
        )

    for zi, f in zip(z, fields):
        if not np.all(np.isfinite(f)):
            raise SystemExit(
                f"error: non-finite values in the density field at z={zi:g}"
            )

    print(
        f"Using {how}: {z.size} snapshots, "
        f"z = {z[0]:g} ... {z[-1]:g}, "
        f"field shape {fields[0].shape}"
    )

    return Snapshots(
        z=z,
        fields=fields,
        meta=meta,
    )
# --------------------------------------------------------------- field prep
def normalise_overdensity(f: np.ndarray) -> np.ndarray:
    """Return 1 + delta whether the file stores rho, 1+delta, or delta."""
    f = np.asarray(f, dtype=np.float64)
    if f.min() < 0.0:                       # delta (mean zero, bounded below by -1)
        if f.min() < -1.0 - 1e-6:
            raise SystemExit("error: field has values below -1; it is not a density contrast.")
        return f + 1.0
    mean = f.mean()
    if not mean > 0.0:
        raise SystemExit("error: density field has non-positive mean.")
    return f / mean                         # rho or 1+delta -> 1+delta


def prepare_slices(snaps: Snapshots, slab: int) -> list:
    """Central slice (z-axis, same [:, :, M//2] convention as the project) as image arrays."""
    onepd = [normalise_overdensity(f) for f in snaps.fields]
    if onepd[0].ndim == 2:
        print("Note: stored fields are already 2-D slices; --slab-cells is ignored.")
        return [s.T.copy() for s in onepd]

    m = onepd[0].shape[2]
    start = m // 2 - slab // 2
    if slab < 1 or start < 0 or start + slab > m:
        raise SystemExit(f"error: --slab-cells {slab} does not fit a mesh of {m} planes.")

    planes = onepd[0].mean(axis=(0, 1))     # earliest snapshot is most lattice-like
    even, odd = planes[::2].mean(), planes[1::2].mean()
    lattice = min(even, odd) == 0.0 or abs(np.log(even / odd)) > 0.35
    if slab < 2 and lattice:
        print(
            "WARNING: mesh planes alternate strongly in mean density at the earliest snapshot "
            f"(even/odd = {even:.3g}/{odd:.3g}). This is the particle lattice sampled on a finer mesh, "
            "so a single plane can look empty. Consider --slab-cells 2 (one particle spacing)."
        )
    return [s[:, :, start:start + slab].mean(axis=2).T.copy() for s in onepd]


def log_field(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(x, FLOOR, None))


def process_cinematic(slices, sigma: float, upsample: int, sharp_sigma: float, amount: float) -> list:
    """Visualization-only look: mild blur -> cubic upsample -> log -> unsharp mask (periodic)."""
    out = []
    for s in slices:
        x = gaussian_filter(s, sigma, mode="wrap") if sigma > 0 else s
        fac = upsample / x.shape[0]
        big = log_field(zoom(x, fac, order=3, mode="grid-wrap", grid_mode=True))
        blur = gaussian_filter(big, sharp_sigma * fac, mode="wrap")
        out.append((big + amount * (big - blur)).astype(np.float32))
    return out


# ------------------------------------------------------------------ cosmology
def build_cosmology(meta: dict, args):
    try:
        from src.cosmology import Cosmology
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"error: cannot import src.cosmology.Cosmology ({exc}). Run from the project root.")
    params, source = {}, {}
    for name, cli, default in (
        ("H0", args.H0, 70.0), ("Omega_m", args.omega_m, 0.3), ("Omega_Lambda", args.omega_lambda, 0.7),
    ):
        if cli is not None:
            params[name], source[name] = cli, "command line"
        elif name in meta:
            params[name], source[name] = meta[name], "snapshots.npz"
        else:
            params[name], source[name] = default, "default (matches experiments/exp_005_cosmic_web.py)"
    cosmo = Cosmology(**params)
    print("Cosmology for the age readout: " + ", ".join(f"{k}={v:g} [{source[k]}]" for k, v in params.items()))
    print(f"  (H0 is taken as km/s/Mpc; age(a=1) = {cosmo.age(1.0) * GYR_PER_MPC_PER_KMS:.2f} Gyr)")
    return cosmo


def age_gyr(cosmo, a: float) -> float:
    return float(cosmo.age(a)) * GYR_PER_MPC_PER_KMS


def fmt_age(g: float) -> str:
    return f"{g:.3f} Gyr" if g < 1.0 else f"{g:.2f} Gyr"


# -------------------------------------------------------------------- ffmpeg
def find_ffmpeg() -> str:
    exe = shutil.which(matplotlib.rcParams.get("animation.ffmpeg_path", "ffmpeg")) or shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg  # optional fallback

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
    if exe is None:
        raise SystemExit(
            "error: FFmpeg was not found on PATH.\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS:         brew install ffmpeg\n"
            "  conda:         conda install -c conda-forge ffmpeg\n"
            "  pip fallback:  pip install imageio-ffmpeg"
        )
    return exe


class VideoWriter:
    """Streams raw RGBA frames into FFmpeg (H.264, yuv420p, BT.709). Nothing is buffered in RAM."""

    def __init__(self, ffmpeg: str, path: Path, fps: int, crf: int, preset: str = "slow"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.tmp, self.frames = path, path.with_name(path.name + ".partial"), 0
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-movflags", "+faststart", "-f", "mp4", str(self.tmp),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: bytes, repeat: int = 1) -> None:
        try:
            for _ in range(repeat):
                self.proc.stdin.write(frame)
        except BrokenPipeError:
            raise SystemExit("error: FFmpeg exited early:\n" + self.proc.stderr.read().decode(errors="replace"))
        self.frames += repeat

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            self.tmp.unlink(missing_ok=True)
            raise SystemExit(f"error: FFmpeg failed (is libx264 available?):\n{err}")
        os.replace(self.tmp, self.path)

    def abort(self) -> None:
        try:
            self.proc.kill()
        finally:
            self.tmp.unlink(missing_ok=True)


def verify_video(path: Path, expected_frames: int, fps: int) -> None:
    size = path.stat().st_size if path.exists() else 0
    if size == 0:
        raise SystemExit(f"error: {path} is missing or empty")
    line = f"  {path}  ({size / 1e6:.1f} MB, {expected_frames} frames, {expected_frames / fps:.1f} s @ {fps} fps)"
    probe = shutil.which("ffprobe")
    if probe:
        res = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames:format=duration",
             "-of", "json", str(path)], capture_output=True, text=True)
        if res.returncode == 0:
            info = json.loads(res.stdout)
            s, d = info["streams"][0], float(info["format"]["duration"])
            line = (f"  {path}  {s['width']}x{s['height']} {s['codec_name']}/{s['pix_fmt']}, "
                    f"{s['nb_read_frames']} frames, {d:.1f} s, {size / 1e6:.1f} MB")
    print("Verified:" + line)


# --------------------------------------------------------------------- scene
class Scene:
    """Matplotlib figure: a square map in the centre, text in the left/right side bars."""

    def __init__(self, init_img, box, lo, hi, cmap, interpolation, colorbar):
        self.box = box
        self.fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
        fx, fy = MAP_PX / W, MAP_PX / H
        self.ax = self.fig.add_axes([(1 - fx) / 2, (1 - fy) / 2, fx, fy], facecolor="black")
        self.im = self.ax.imshow(init_img, origin="lower", extent=(0, box, 0, box), cmap=cmap,
                                 vmin=lo, vmax=hi, interpolation=interpolation)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_color("#2a3040")
        halo = [pe.withStroke(linewidth=5, foreground="black", alpha=0.55)]
        (self.bar,) = self.ax.plot([], [], color=FG, lw=3.5, solid_capstyle="butt", path_effects=halo)
        self.bar_txt = self.ax.text(0, 0, f"{SCALE_BAR:g} Mpc/h", color=FG, fontsize=15, ha="center",
                                    va="bottom", path_effects=halo)
        self.cax = self.cb_label = None
        if colorbar:
            self.cax = self.fig.add_axes([0.778, 0.32, 0.008, 0.46])
            cb = self.fig.colorbar(self.im, cax=self.cax)
            cb.outline.set_edgecolor("#2a3040")
            self.cax.tick_params(colors=DIM, labelsize=12, length=3)
            self.cb_label = self.fig.text(0.778, 0.815, r"$\log_{10}(1+\delta)$", color=FG, fontsize=15, ha="left")
        self.fade = Rectangle((0, 0), 1, 1, transform=self.fig.transFigure, facecolor="black",
                              alpha=0.0, zorder=100)
        self.fig.add_artist(self.fade)
        self.set_view(1.0)

    def text(self, x, y, s, size=14, color=FG, mono=False, weight="normal", **kw):
        return self.fig.text(x, y, s, fontsize=size, color=color, weight=weight,
                             family=MONO if mono else "DejaVu Sans", **kw)

    def set_view(self, zoom_factor: float) -> None:
        """Centred zoom; the scale bar stays in a fixed screen corner and scales with the data."""
        half = self.box / 2.0 / zoom_factor
        c = self.box / 2.0
        self.ax.set_xlim(c - half, c + half)
        self.ax.set_ylim(c - half, c + half)
        x0, y0 = c - half + 0.06 * 2 * half, c - half + 0.06 * 2 * half
        self.bar.set_data([x0, x0 + SCALE_BAR], [y0, y0])
        self.bar_txt.set_position((x0 + SCALE_BAR / 2, y0 + 0.012 * 2 * half))

    def frame(self) -> bytes:
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        if buf.shape[:2] != (H, W):
            raise RuntimeError(f"unexpected canvas size {buf.shape[:2]}, expected {(H, W)}")
        return buf.tobytes()

    def close(self) -> None:
        plt.close(self.fig)


class Progress:
    def __init__(self, label, total):
        self.label, self.total, self.next = label, total, 10

    def update(self, done):
        pct = 100 * done / self.total
        if pct >= self.next:
            print(f"  {self.label}: {pct:3.0f}%", flush=True)
            self.next += 10


# ----------------------------------------------------------------- evolution
def render_evolution(snaps, slices, box, mesh, slab, out, ffmpeg, fps, crf) -> int:
    n = len(slices)
    imgs = [log_field(s).astype(np.float32) for s in slices]
    lo, hi = np.percentile(np.concatenate([i.ravel() for i in imgs]), [0.5, 99.8])
    print(f"Evolution colour range (fixed, global): log10(1+delta) in [{lo:.2f}, {hi:.2f}]")

    sc = Scene(imgs[0], box, lo, hi, "inferno", "nearest", colorbar=True)
    sc.set_view(1.0)
    l, r = 0.03, 0.775
    title_main = sc.text(l, 0.925, "Cosmic Web Formation", 18, weight="bold")
    title_sub = sc.text(l, 0.895, "Particle-Mesh N-body Simulation", 13, DIM)
    lab_z = sc.text(l, 0.640, "REDSHIFT", 12, DIM)
    val_z = sc.text(l, 0.585, "", 40, mono=True)
    val_a = sc.text(l, 0.535, "", 20, mono=True)
    caption = sc.text(l, 0.480, "Present-day matter distribution", 15, FG)
    counter = sc.text(l, 0.430, "", 13, DIM)
    rows = [sc.text(l + 0.012, 0.335 - 0.0265 * i, f"z = {z:g}", 12, FAINT, mono=True) for i, z in enumerate(snaps.z)]
    note = sc.text(r, 0.255, f"central slice of {mesh}$^3$ mesh\n{slab} plane(s) averaged, {box:g} Mpc/h box\n"
                             "fixed colour scale", 11, DIM, linespacing=1.5, va="top")
    side = [title_main, title_sub, lab_z, val_z, val_a, caption, counter, note, *rows]
    t_main = sc.text(0.5, 0.52, "Cosmic Web Formation", 46, weight="bold", ha="center")
    t_sub = sc.text(0.5, 0.45, "Particle-Mesh N-body Simulation", 22, DIM, ha="center")

    def show_title(on: bool, alpha: float = 1.0):
        for art in (sc.ax, sc.cax, *side):
            art.set_visible(not on)
        sc.cb_label.set_visible(not on)
        for t in (t_main, t_sub):
            t.set_visible(on)
            t.set_alpha(alpha)

    def show_snapshot(i: int, final: bool):
        show_title(False)
        sc.im.set_data(imgs[i])
        val_z.set_text(f"z = {snaps.z[i]:.2f}")
        val_a.set_text(f"a = {snaps.a[i]:.4f}")
        counter.set_text(f"snapshot {i + 1} / {n}")
        caption.set_visible(final)
        for j, row in enumerate(rows):
            row.set_color(FG if j == i else (DIM if j < i else FAINT))
            row.set_text(("\u25b6 " if j == i else "   ") + f"z = {snaps.z[j]:g}")

    sec = float(np.clip(30.0 / n, 2.0, 5.0))
    title_f, hold_f, final_f = int(3 * fps), int(round(sec * fps)), int(3 * fps)
    total = title_f + n * hold_f + final_f
    w = VideoWriter(ffmpeg, out, fps, crf)
    prog = Progress("evolution", total)
    try:
        done = 0
        fade_f = int(0.5 * fps)
        for k in range(title_f):                       # title card: text fades in/out, no data shown
            alpha = min(1.0, k / fade_f, (title_f - 1 - k) / fade_f)
            show_title(True, max(alpha, 0.0))
            w.write(sc.frame())
        done += title_f
        for i in range(n):                              # each stored snapshot, hard cut, held as one frame
            show_snapshot(i, final=False)
            w.write(sc.frame(), hold_f)
            done += hold_f
            prog.update(done)
        show_snapshot(n - 1, final=True)               # closing hold on z = 0
        w.write(sc.frame(), final_f)
        w.close()
    except BaseException:
        w.abort()
        raise
    finally:
        sc.close()
    return total


# ----------------------------------------------------------------- cinematic
def cinematic_timeline(n_snap: int, fps: int, total_s: float, hold_start: float, hold_end: float) -> Iterator[tuple]:
    """Yield (segment index, eased weight). Weight is the position between snapshot i and i+1 in a."""
    n_seg = n_snap - 1
    seg_f = max(2, round((total_s - hold_start - hold_end) * fps / n_seg))
    for _ in range(round(hold_start * fps)):
        yield 0, 0.0
    for i in range(n_seg):
        for j in range(seg_f):
            yield i, smoothstep(j / seg_f)
    for _ in range(round(hold_end * fps)):
        yield n_seg - 1, 1.0


def render_cinematic(snaps, slices, box, mesh, cosmo, args, out, ffmpeg) -> int:
    fps = args.fps
    proc = process_cinematic(slices, args.smooth, 1024, 1.5, 0.5)
    lo, hi = np.percentile(np.concatenate([p.ravel() for p in proc]), [0.5, 99.8])
    print(f"Cinematic colour range (fixed, global): log10(1+delta) in [{lo:.2f}, {hi:.2f}]")

    sc = Scene(proc[0], box, lo, hi, "magma", "bilinear", colorbar=False)
    l, r = 0.03, 0.795
    sc.text(l, 0.925, "Cosmic Web Formation", 18, weight="bold")
    sc.text(l, 0.895, "Particle-Mesh N-body Simulation", 13, DIM)
    sc.text(l, 0.150, f"\u039bCDM  \u00b7  \u03a9m = {cosmo.Omega_m:g}, \u03a9\u039b = {cosmo.Omega_Lambda:g}\n"
                      f"{mesh}$^3$ mesh  \u00b7  {box:g} Mpc/h box", 11, DIM, linespacing=1.6)
    sc.text(l, 0.075, "Frames between simulation snapshots\nare interpolated for visualization only.",
            10.5, FAINT, linespacing=1.5)
    sc.text(r, 0.725, "REDSHIFT", 11, DIM)
    val_z = sc.text(r, 0.680, "", 30, mono=True)
    sc.text(r, 0.590, "SCALE FACTOR", 11, DIM)
    val_a = sc.text(r, 0.545, "", 30, mono=True)
    sc.text(r, 0.455, "UNIVERSE AGE", 11, DIM)
    val_t = sc.text(r, 0.410, "", 24, mono=True)

    a_s = snaps.a
    frames = list(cinematic_timeline(len(slices), fps, args.duration, 2.0, 4.0))
    total = len(frames)
    fade_f = int(1.5 * fps)
    w = VideoWriter(ffmpeg, out, fps, args.crf)
    prog = Progress("cinematic", total)
    try:
        for k, (i, wt) in enumerate(frames):
            sc.im.set_data((1.0 - wt) * proc[i] + wt * proc[i + 1])
            a = a_s[i] + wt * (a_s[i + 1] - a_s[i])              # pacing in scale factor
            val_z.set_text(f"z = {1.0 / a - 1.0:.2f}")
            val_a.set_text(f"a = {a:.4f}")
            val_t.set_text(fmt_age(age_gyr(cosmo, a)))
            sc.set_view(1.0 + args.zoom * smoothstep(k / max(total - 1, 1)))
            sc.fade.set_alpha(float(np.clip(1.0 - min(k, total - 1 - k) / fade_f, 0.0, 1.0)))
            w.write(sc.frame())
            prog.update(k + 1)
        w.close()
    except BaseException:
        w.abort()
        raise
    finally:
        sc.close()
    return total


# ----------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("evolution", "cinematic", "both"), default="both")
    p.add_argument("--input", type=Path, default=ROOT / "results" / "snapshots.npz")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better/larger)")
    p.add_argument("--box-size", type=float, default=None, help="Mpc/h; default: from the NPZ, else 100")
    p.add_argument("--slab-cells", type=int, default=2,
                   help="mesh planes averaged around the centre (1 = exactly the single central plane)")
    p.add_argument("--evolution-smooth", type=float, default=0.0,
                   help="optional Gaussian sigma (mesh cells) for the evolution video; 0 = raw stored mesh")
    p.add_argument("--duration", type=float, default=40.0, help="cinematic length in seconds")
    p.add_argument("--smooth", type=float, default=1.0, help="cinematic Gaussian sigma, in mesh cells")
    p.add_argument("--zoom", type=float, default=0.12, help="cinematic total zoom-in (0.12 = 12%%)")
    p.add_argument("--density-key", default=None)
    p.add_argument("--redshift-key", default=None)
    p.add_argument("--H0", type=float, default=None, help="km/s/Mpc, overrides the NPZ/default")
    p.add_argument("--omega-m", type=float, default=None)
    p.add_argument("--omega-lambda", type=float, default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    ffmpeg = find_ffmpeg()
    snaps = load_snapshots(args.input, args.density_key, args.redshift_key)
    box = args.box_size or snaps.meta.get("box_size") or 100.0
    slices = prepare_slices(snaps, args.slab_cells)
    mesh = snaps.fields[0].shape[0]
    print(f"Box size {box:g} Mpc/h, slab = {args.slab_cells} plane(s), redshifts: "
          + ", ".join(f"{z:g}" for z in snaps.z))

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    if args.mode in ("evolution", "both"):
        path = out_dir / "cosmic_web_evolution.mp4"
        print(f"\n[1] Scientific evolution -> {path}")
        evo = slices if args.evolution_smooth <= 0 else [
            gaussian_filter(s, args.evolution_smooth, mode="wrap") for s in slices]
        n = render_evolution(snaps, evo, box, mesh, args.slab_cells, path, ffmpeg, args.fps, args.crf)
        done.append((path, n))
    if args.mode in ("cinematic", "both"):
        path = out_dir / "cosmic_web_cinematic.mp4"
        print(f"\n[2] Cinematic (interpolated for visualization only) -> {path}")
        cosmo = build_cosmology(snaps.meta, args)
        n = render_cinematic(snaps, slices, box, mesh, cosmo, args, path, ffmpeg)
        done.append((path, n))
    print()
    for path, n in done:
        verify_video(path, n, args.fps)


if __name__ == "__main__":
    main()
