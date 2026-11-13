"""Experiment 001: two-body circular orbit.

Two massive particles are placed on a circular orbit about their common
centre of mass. For the softened force law the exact circular-orbit angular
frequency is::

    omega^2 = G (m1 + m2) / (d^2 + eps^2)^(3/2)

where ``d`` is the separation. The system is integrated with the KDK leapfrog
for many orbits; a symplectic integrator should keep the orbit closed, with
bounded (non-growing) energy error and constant separation.

Run from anywhere::

    python experiments/exp_001_twobody.py [--orbits 20] [--show]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gravity import DirectGravity  # noqa: E402
from src.integrator import LeapfrogKDK  # noqa: E402
from src.particles import FloatArray, ParticleSet  # noqa: E402

OUTPUT_DIR = ROOT / "results" / "exp_001_twobody"


@dataclass(frozen=True)
class TwoBodyConfig:
    G: float = 1.0
    m1: float = 1.0
    m2: float = 1.0
    separation: float = 1.0
    softening: float = 1.0e-3
    n_orbits: int = 20
    steps_per_orbit: int = 200


@dataclass(frozen=True)
class TwoBodyResult:
    times: FloatArray  # (T,)
    positions: FloatArray  # (T, 2, 3)
    energy: FloatArray  # (T,)
    angular_momentum_z: FloatArray  # (T,)
    separation: FloatArray  # (T,)
    period: float
    steps_per_orbit: int


def circular_angular_frequency(cfg: TwoBodyConfig) -> float:
    """Angular frequency of the exact circular orbit under softened gravity."""
    m_tot = cfg.m1 + cfg.m2
    return float(np.sqrt(cfg.G * m_tot / (cfg.separation**2 + cfg.softening**2) ** 1.5))


def make_initial_conditions(cfg: TwoBodyConfig) -> ParticleSet:
    """Two particles on a circular orbit in the x-y plane, centre of mass at rest at the origin."""
    m_tot = cfg.m1 + cfg.m2
    omega = circular_angular_frequency(cfg)
    r1 = cfg.separation * cfg.m2 / m_tot  # distance of body 1 from the COM
    r2 = cfg.separation * cfg.m1 / m_tot

    positions = np.array([[-r1, 0.0, 0.0], [r2, 0.0, 0.0]])
    velocities = np.array([[0.0, -omega * r1, 0.0], [0.0, omega * r2, 0.0]])
    return ParticleSet(positions, velocities, np.array([cfg.m1, cfg.m2]))


def run_experiment(cfg: TwoBodyConfig) -> TwoBodyResult:
    particles = make_initial_conditions(cfg)
    gravity = DirectGravity(softening=cfg.softening, G=cfg.G)
    integrator = LeapfrogKDK(gravity)

    period = 2.0 * np.pi / circular_angular_frequency(cfg)
    dt = period / cfg.steps_per_orbit
    n_steps = cfg.n_orbits * cfg.steps_per_orbit

    positions = np.empty((n_steps + 1, particles.n, particles.dim))
    energy = np.empty(n_steps + 1)
    lz = np.empty(n_steps + 1)

    def record(k: int, p: ParticleSet) -> None:
        positions[k] = p.positions
        energy[k] = gravity.total_energy(p)
        lz[k] = p.angular_momentum()[2]

    positions[0] = particles.positions
    energy[0] = gravity.total_energy(particles)
    lz[0] = particles.angular_momentum()[2]

    integrator.run(particles, dt, n_steps, callback=record)

    times = dt * np.arange(n_steps + 1)
    separation = np.linalg.norm(positions[:, 0, :] - positions[:, 1, :], axis=1)
    return TwoBodyResult(times, positions, energy, lz, separation, period, cfg.steps_per_orbit)


def save_results(result: TwoBodyResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        times=result.times,
        positions=result.positions,
        energy=result.energy,
        angular_momentum_z=result.angular_momentum_z,
        separation=result.separation,
        period=result.period,
    )


def plot_results(result: TwoBodyResult, cfg: TwoBodyConfig, path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    pos = result.positions
    spo = result.steps_per_orbit
    t_orbits = result.times / result.period
    e0 = result.energy[0]
    rel_err = np.abs((result.energy - e0) / e0)
    sep_ratio = result.separation / cfg.separation

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # Trajectories: first vs last orbit overlaid to expose any decay.
    ax = axes[0]
    colors = ("tab:blue", "tab:orange")
    for i, color in enumerate(colors):
        ax.plot(pos[:, i, 0], pos[:, i, 1], color=color, lw=0.6, alpha=0.35)
        ax.plot(pos[: spo + 1, i, 0], pos[: spo + 1, i, 1], color=color, lw=2.0,
                label=f"Body {i + 1}: orbit 1")
        ax.plot(pos[-spo - 1 :, i, 0], pos[-spo - 1 :, i, 1], color="k", lw=1.0, ls="--",
                label="Last orbit" if i == 0 else None)
        ax.plot(*pos[0, i, :2], "o", color=color, ms=8)
    ax.plot(0.0, 0.0, "+", color="gray", ms=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Trajectories over {cfg.n_orbits} orbits")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(t_orbits, sep_ratio, lw=1.2)
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("time [orbital periods]")
    ax.set_ylabel(r"separation / $d_0$")
    ax.set_title("Orbital separation")

    ax = axes[2]
    ax.semilogy(t_orbits, np.maximum(rel_err, 1e-18), lw=1.2)
    ax.set_xlabel("time [orbital periods]")
    ax.set_ylabel(r"$|\Delta E / E_0|$")
    ax.set_title("Relative energy error (bounded, no secular drift)")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-body circular orbit experiment")
    parser.add_argument("--orbits", type=int, default=TwoBodyConfig.n_orbits)
    parser.add_argument("--steps-per-orbit", type=int, default=TwoBodyConfig.steps_per_orbit)
    parser.add_argument("--softening", type=float, default=TwoBodyConfig.softening)
    parser.add_argument("--show", action="store_true", help="open an interactive plot window")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    cfg = TwoBodyConfig(
        n_orbits=args.orbits,
        steps_per_orbit=args.steps_per_orbit,
        softening=args.softening,
    )
    result = run_experiment(cfg)

    save_results(result, OUTPUT_DIR / "trajectory.npz")
    plot_results(result, cfg, OUTPUT_DIR / "orbit.png", show=args.show)

    rel_err = np.abs((result.energy - result.energy[0]) / result.energy[0])
    lz_err = np.abs((result.angular_momentum_z - result.angular_momentum_z[0])
                    / result.angular_momentum_z[0])
    print(f"Period T                : {result.period:.6f}")
    print(f"Steps                   : {result.times.size - 1} ({cfg.steps_per_orbit}/orbit)")
    print(f"Separation min / max    : {result.separation.min():.6f} / {result.separation.max():.6f}"
          f"  (initial {cfg.separation:.6f})")
    print(f"Max |dE/E0|             : {rel_err.max():.3e}")
    print(f"Max |dLz/Lz0|           : {lz_err.max():.3e}")
    print(f"Outputs written to      : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
