"""Experiment 002: cold collapse of a 1,000-particle cluster.

A roughly uniform sphere of equal-mass particles is released with only small
random velocities (initial virial ratio ``2K/|U| << 1``). With almost no
support against gravity it falls in on roughly a free-fall time, reaches a
compact maximum-density state, rebounds, and relaxes towards virial
equilibrium (``2K/|U| -> 1``) through violent relaxation.

Structure of this script:

* **Physics** (``make_cold_sphere``, ``run_cluster``): builds the initial
  state and integrates it, recording snapshots and diagnostics into plain
  NumPy arrays. It knows nothing about plotting.
* **Persistence** (``save_snapshots``): the physics output is written to an
  ``.npz`` file.
* **Rendering** (``render_outputs``): consumes those arrays through
  ``visualization.particles_3d``, which never imports the physics code.

Units: ``G = M_total = R = 1``. The free-fall time of a uniform sphere is
``t_ff = (pi/2) sqrt(R^3 / (2 G M))`` (about 1.11 in these units).

Run from anywhere::

    python experiments/exp_002_cluster.py                # simulate + render
    python experiments/exp_002_cluster.py --render-only  # re-render saved snapshots
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gravity import DirectGravity  # noqa: E402
from src.integrator import LeapfrogKDK  # noqa: E402
from src.particles import FloatArray, ParticleSet  # noqa: E402

OUTPUT_DIR = ROOT / "results" / "exp_002_cluster"


@dataclass(frozen=True)
class ClusterConfig:
    n_particles: int = 1000
    radius: float = 1.0
    total_mass: float = 1.0
    G: float = 1.0
    softening: float = 0.05
    initial_virial_ratio: float = 0.05  # 2K/|U| at t = 0 ("cold")
    dt: float = 0.004
    t_end_in_tff: float = 5.0
    n_frames: int = 120
    view_limit: float = 2.5  # half-width of the rendered cube, in units of the initial radius
    seed: int = 42

    @property
    def free_fall_time(self) -> float:
        return float(np.pi / 2.0 * np.sqrt(self.radius**3 / (2.0 * self.G * self.total_mass)))


@dataclass(frozen=True)
class ClusterRun:
    times: FloatArray  # (F,)
    positions: FloatArray  # (F, N, 3)
    velocities: FloatArray  # (F, N, 3)
    energy: FloatArray  # (F,)
    virial_ratio: FloatArray  # (F,)  2K/|U|
    half_mass_radius: FloatArray  # (F,)
    free_fall_time: float
    wall_seconds: float
    n_steps: int


# ---------------------------------------------------------------------- #
# Physics
# ---------------------------------------------------------------------- #
def make_cold_sphere(cfg: ClusterConfig, gravity: DirectGravity) -> ParticleSet:
    """Uniform sphere, equal masses, small random velocities, COM at rest at the origin."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_particles

    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = cfg.radius * rng.random(n) ** (1.0 / 3.0)  # uniform in volume
    positions = directions * radii[:, None]
    positions -= positions.mean(axis=0)

    masses = np.full(n, cfg.total_mass / n)

    velocities = rng.normal(size=(n, 3))
    velocities -= velocities.mean(axis=0)
    particles = ParticleSet(positions, velocities, masses)

    # Rescale the random velocities so that 2K/|U| equals the requested value.
    potential = gravity.potential_energy(particles)
    scale = np.sqrt(cfg.initial_virial_ratio * abs(potential) / (2.0 * particles.kinetic_energy))
    particles.velocities *= scale
    return particles


def half_mass_radius(particles: ParticleSet) -> float:
    """Radius (about the centre of mass) enclosing half of the total mass."""
    r = np.linalg.norm(particles.positions - particles.center_of_mass, axis=1)
    order = np.argsort(r)
    cumulative = np.cumsum(particles.masses[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    return float(r[order][index])


def run_cluster(cfg: ClusterConfig) -> ClusterRun:
    gravity = DirectGravity(softening=cfg.softening, G=cfg.G)
    integrator = LeapfrogKDK(gravity)
    particles = make_cold_sphere(cfg, gravity)

    n_steps = int(np.ceil(cfg.t_end_in_tff * cfg.free_fall_time / cfg.dt))
    every = max(1, n_steps // cfg.n_frames)
    n_snap = n_steps // every + 1

    times = np.empty(n_snap)
    positions = np.empty((n_snap, particles.n, 3))
    velocities = np.empty((n_snap, particles.n, 3))
    energy = np.empty(n_snap)
    virial = np.empty(n_snap)
    r_half = np.empty(n_snap)

    def record(slot: int, step: int, p: ParticleSet) -> None:
        potential = gravity.potential_energy(p)
        kinetic = p.kinetic_energy
        times[slot] = step * cfg.dt
        positions[slot] = p.positions
        velocities[slot] = p.velocities
        energy[slot] = kinetic + potential
        virial[slot] = 2.0 * kinetic / abs(potential)
        r_half[slot] = half_mass_radius(p)

    def callback(step: int, p: ParticleSet) -> None:
        if step % every == 0:
            record(step // every, step, p)

    start = time.perf_counter()
    record(0, 0, particles)
    integrator.run(particles, cfg.dt, n_steps, callback=callback)
    wall = time.perf_counter() - start

    return ClusterRun(times, positions, velocities, energy, virial, r_half,
                      cfg.free_fall_time, wall, n_steps)


def save_snapshots(run: ClusterRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=run.times,
        positions=run.positions,
        velocities=run.velocities,
        energy=run.energy,
        virial_ratio=run.virial_ratio,
        half_mass_radius=run.half_mass_radius,
        free_fall_time=run.free_fall_time,
    )


def load_snapshots(path: Path) -> ClusterRun:
    with np.load(path) as d:
        return ClusterRun(
            times=d["times"], positions=d["positions"], velocities=d["velocities"],
            energy=d["energy"], virial_ratio=d["virial_ratio"],
            half_mass_radius=d["half_mass_radius"],
            free_fall_time=float(d["free_fall_time"]), wall_seconds=float("nan"),
            n_steps=-1,
        )


# ---------------------------------------------------------------------- #
# Post-processing: rendering and diagnostics (no physics is run here)
# ---------------------------------------------------------------------- #
def plot_diagnostics(run: ClusterRun, cfg: ClusterConfig, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    t = run.times / run.free_fall_time
    rel_err = np.abs((run.energy - run.energy[0]) / run.energy[0])

    fig = Figure(figsize=(8.0, 8.0), dpi=120)
    FigureCanvasAgg(fig)
    ax0, ax1, ax2 = fig.subplots(3, 1, sharex=True)

    ax0.plot(t, run.half_mass_radius / cfg.radius)
    ax0.set_ylabel(r"half-mass radius / $R_0$")
    ax0.set_title("Cold collapse of a 1,000-particle cluster")

    ax1.plot(t, run.virial_ratio)
    ax1.axhline(1.0, color="gray", ls="--", lw=1.0)
    ax1.set_ylabel(r"virial ratio $2K/|U|$")

    ax2.semilogy(t, np.maximum(rel_err, 1e-12))
    ax2.set_ylabel(r"$|\Delta E / E_0|$")
    ax2.set_xlabel(r"time [$t_\mathrm{ff}$]")

    for ax in (ax0, ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def render_outputs(run: ClusterRun, cfg: ClusterConfig, formats: list[str],
                   color_by: str) -> list[Path]:
    """Write movie(s), a still montage and a diagnostics plot."""
    from visualization.particles_3d import (
        AnimationConfig, render_animation, render_montage,
    )

    anim_cfg = AnimationConfig(
        color_by=color_by,  # type: ignore[arg-type]
        limit=cfg.view_limit * cfg.radius,
        time_scale=run.free_fall_time,
        time_label=r"t / t_ff",
    )
    written: list[Path] = []
    for fmt in formats:
        out = OUTPUT_DIR / f"cluster_collapse.{fmt}"
        print(f"Rendering {out.name} ...", flush=True)
        written.append(render_animation(run.positions, run.velocities, run.times, out, anim_cfg))

    i_min = int(np.argmin(run.half_mass_radius))
    n = run.times.size
    frames = sorted({0, i_min, (i_min + n - 1) // 2, n - 1})
    written.append(render_montage(run.positions, run.velocities, run.times, frames,
                                  OUTPUT_DIR / "cluster_montage.png", anim_cfg))

    plot_diagnostics(run, cfg, OUTPUT_DIR / "diagnostics.png")
    written.append(OUTPUT_DIR / "diagnostics.png")
    return written


def summarize(run: ClusterRun) -> None:
    t = run.times / run.free_fall_time
    i_min = int(np.argmin(run.half_mass_radius))
    tail = run.virial_ratio[run.virial_ratio.size * 3 // 4:]
    rel_err = np.abs((run.energy - run.energy[0]) / run.energy[0])

    print(f"Free-fall time t_ff     : {run.free_fall_time:.4f}")
    if run.n_steps > 0:
        print(f"Steps / wall time       : {run.n_steps} / {run.wall_seconds:.1f} s "
              f"({1e3 * run.wall_seconds / run.n_steps:.1f} ms per step)")
    print(f"Snapshots               : {run.times.size}")
    print(f"Initial virial ratio    : {run.virial_ratio[0]:.3f}")
    print(f"Maximum compression     : t = {t[i_min]:.2f} t_ff, "
          f"r_half = {run.half_mass_radius[i_min]:.3f} (from {run.half_mass_radius[0]:.3f})")
    print(f"Late-time virial ratio  : {tail.mean():.3f} +/- {tail.std():.3f} (last 25% of run)")
    print(f"Final half-mass radius  : {run.half_mass_radius[-1]:.3f}")
    print(f"Max |dE/E0|             : {rel_err.max():.3e}")


def main() -> None:
    d = ClusterConfig()
    parser = argparse.ArgumentParser(description="Cold-collapse N-body cluster experiment")
    parser.add_argument("--n", type=int, default=d.n_particles, help="number of particles")
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--softening", type=float, default=d.softening)
    parser.add_argument("--dt", type=float, default=d.dt)
    parser.add_argument("--virial-ratio", type=float, default=d.initial_virial_ratio,
                        help="initial 2K/|U| (small = cold)")
    parser.add_argument("--t-end", type=float, default=d.t_end_in_tff, help="run length in t_ff")
    parser.add_argument("--frames", type=int, default=d.n_frames, help="approximate snapshot count")
    parser.add_argument("--view-limit", type=float, default=d.view_limit,
                        help="half-width of the rendered cube in units of the initial radius")
    parser.add_argument("--color-by", choices=("speed", "density"), default="speed")
    parser.add_argument("--formats", nargs="+", choices=("gif", "mp4"), default=["gif", "mp4"])
    parser.add_argument("--no-render", action="store_true", help="simulate and save only")
    parser.add_argument("--render-only", action="store_true",
                        help="skip the simulation; re-render saved snapshots")
    args = parser.parse_args()

    cfg = ClusterConfig(
        n_particles=args.n, seed=args.seed, softening=args.softening, dt=args.dt,
        initial_virial_ratio=args.virial_ratio, t_end_in_tff=args.t_end, n_frames=args.frames,
        view_limit=args.view_limit,
    )
    snapshot_path = OUTPUT_DIR / "snapshots.npz"

    if args.render_only:
        run = load_snapshots(snapshot_path)
    else:
        run = run_cluster(cfg)
        save_snapshots(run, snapshot_path)
    summarize(run)

    if not args.no_render:
        for path in render_outputs(run, cfg, args.formats, args.color_by):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
