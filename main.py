#!/usr/bin/env python3
"""Cosmic Web Formation Simulator: master pipeline.

Runs the full simulation start to finish:

1. Build a flat Lambda-CDM :class:`~src.cosmology.Cosmology`.
2. Generate Zel'dovich initial conditions (Gaussian random field, BBKS-shaped
   spectrum) for N = 32^3 particles in a periodic 100 Mpc/h box at z = 49.
3. Evolve with the Particle-Mesh (PM) solver and the comoving KDK leapfrog
   down to z = 0, pausing at each requested redshift to measure a mid-plane
   density slice and the matter power spectrum P(k).
4. Save the raw snapshot data and the final figures to ``results/``.

Usage::

    python main.py                       # default 32^3 particles, 64^3 mesh
    python main.py --seed 7 --output-dir results/run_seed7
    python main.py --help

Units: comoving Mpc/h, km/s, Msun/h, with H0 = 100 (so time is in Mpc/(km/s)).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

# Make ``src`` and ``analysis`` importable no matter where the script is run from.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.power_spectrum import compute_power_spectrum  # noqa: E402
from src.cosmology import Cosmology  # noqa: E402
from src.gravity import PMGravity  # noqa: E402
from src.initial_conditions import (  # noqa: E402
    power_spectrum_bbks,
    zeldovich_initial_conditions_arrays,
)
from src.integrator import LeapfrogKDK  # noqa: E402
from src.mesh import cic_stencil, deposit_cic  # noqa: E402
from src.particles import ParticleSet  # noqa: E402
from src.poisson import overdensity  # noqa: E402


# ---------------------------------------------------------------------- #
# Configuration and data containers
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class PipelineConfig:
    """All tunable parameters of a run."""

    # Particles, mesh and box
    n_particles_per_side: int = 32  # N = 32^3 particles
    grid_size: int = 64  # PM mesh: 64^3 nodes
    box_size: float = 100.0  # Mpc/h

    # Cosmology (flat: Omega_Lambda = 1 - Omega_m)
    H0: float = 100.0  # km/s/(Mpc/h): exactly 100 in h-units
    omega_m: float = 0.3

    # Initial conditions
    z_init: float = 49.0  # a_init = 0.02
    spectral_index: float = -2.0  # steep tilt: large scales dominate
    sigma8: float = 0.2  # pre-clamp amplitude of the BBKS-shaped spectrum
    max_delta_init: float = 0.05  # cap on |delta| at z_init (near-uniform start)
    seed: int = 42

    # Evolution: redshifts (descending, first == z_init) where data are recorded
    snapshot_redshifts: tuple[float, ...] = (49.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.0)
    steps_per_efold: int = 80  # leapfrog steps per unit of ln(a)
    min_steps_per_segment: int = 15

    # Output
    montage_redshifts: tuple[float, ...] = (49.0, 10.0, 3.0, 1.0, 0.0)
    slice_blur_sigma: float = 1.0  # Gaussian blur (cells) applied to plotted slices
    output_dir: Path = ROOT / "results"

    @property
    def a_init(self) -> float:
        return 1.0 / (1.0 + self.z_init)

    @property
    def n_particles(self) -> int:
        return self.n_particles_per_side**3


@dataclass(frozen=True)
class Snapshot:
    """Measurements taken at one redshift."""

    z: float
    a: float
    density_slice: np.ndarray  # mid-plane slice of 1 + delta, shape (M, M)
    k: np.ndarray  # power-spectrum bin centres [h/Mpc]
    power: np.ndarray  # shot-noise-subtracted P(k) [(Mpc/h)^3]
    max_density: float  # max(1 + delta) on the mesh


# ---------------------------------------------------------------------- #
# Pipeline stages
# ---------------------------------------------------------------------- #
def build_cosmology(cfg: PipelineConfig) -> Cosmology:
    """Flat Lambda-CDM background."""
    return Cosmology(H0=cfg.H0, Omega_m=cfg.omega_m, Omega_Lambda=1.0 - cfg.omega_m)


def build_initial_conditions(cfg: PipelineConfig, cosmo: Cosmology) -> ParticleSet:
    """Zel'dovich initial conditions on a uniform lattice at ``z_init``."""

    def power_spectrum(k: np.ndarray) -> np.ndarray:
        return power_spectrum_bbks(k, cosmo, ns=cfg.spectral_index, sigma8=cfg.sigma8)

    x, v, masses = zeldovich_initial_conditions_arrays(
        cfg.n_particles_per_side,
        cfg.box_size,
        cosmo,
        cfg.a_init,
        power_spectrum,
        seed=cfg.seed,
        max_delta_init=cfg.max_delta_init,
    )
    return ParticleSet(x, v, masses)


def measure_snapshot(
    particles: ParticleSet, cfg: PipelineConfig, mean_density: float, z: float, a: float
) -> Snapshot:
    """Deposit particles on the mesh (CIC) and measure the density slice and P(k)."""
    stencil = cic_stencil(particles.positions, cfg.grid_size, cfg.box_size)
    density = deposit_cic(stencil, particles.masses)
    delta = overdensity(density, mean_density)  # delta = (rho - rho_bar) / rho_bar

    # CIC conserves mass, so the mean of (1 + delta) must be exactly 1.
    if abs((1.0 + delta).mean() - 1.0) > 1.0e-10:
        raise RuntimeError(f"mean(1 + delta) = {(1.0 + delta).mean()} at z = {z:g}")

    k, power = compute_power_spectrum(delta, cfg.box_size, n_particles=cfg.n_particles)
    mid = cfg.grid_size // 2
    return Snapshot(
        z=z,
        a=a,
        density_slice=(1.0 + delta[:, :, mid]).copy(),
        k=k,
        power=power,
        max_density=float((1.0 + delta).max()),
    )


def run_simulation(cfg: PipelineConfig) -> list[Snapshot]:
    """Generate initial conditions, integrate to z = 0 and record snapshots."""
    redshifts = list(cfg.snapshot_redshifts)
    if redshifts != sorted(redshifts, reverse=True) or redshifts[0] != cfg.z_init:
        raise ValueError("snapshot_redshifts must be descending and start at z_init")

    cosmo = build_cosmology(cfg)
    particles = build_initial_conditions(cfg, cosmo)
    mean_density = particles.total_mass / cfg.box_size**3

    # Comoving PM gravity: nabla^2 Phi = (3/2) Omega_m H0^2 delta / a.
    gravity = PMGravity.from_cosmology(cfg.grid_size, cfg.box_size, cosmo)
    integrator = LeapfrogKDK(gravity, cosmology=cosmo, a_initial=cfg.a_init)

    print(
        f"N = {cfg.n_particles_per_side}^3 = {cfg.n_particles} particles | "
        f"PM mesh {cfg.grid_size}^3 | box {cfg.box_size:g} Mpc/h | "
        f"z = {cfg.z_init:g} -> 0"
    )
    print(f"{'z':>6} {'a':>8} {'steps':>6} {'max(1+delta)':>13}")

    snapshots: list[Snapshot] = []
    start = time.perf_counter()
    for z in redshifts:
        a_target = 1.0 / (1.0 + z)

        # Advance to the next snapshot. Step count scales with ln(a) so the
        # time resolution is comparable across segments.
        n_steps = 0
        if a_target > integrator.a * (1.0 + 1.0e-12):
            n_steps = max(
                cfg.min_steps_per_segment,
                math.ceil(cfg.steps_per_efold * math.log(a_target / integrator.a)),
            )
            dt = cosmo.time_between(integrator.a, a_target) / n_steps
            integrator.run(particles, dt, n_steps)

        snap = measure_snapshot(particles, cfg, mean_density, z, integrator.a)
        snapshots.append(snap)
        print(f"{z:>6.1f} {snap.a:>8.4f} {n_steps:>6d} {snap.max_density:>13.2f}")

    print(f"Simulation finished in {time.perf_counter() - start:.1f} s")
    return snapshots


# ---------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------- #
def save_snapshots(snapshots: list[Snapshot], path: Path) -> None:
    """Store slices and spectra of every snapshot in a single ``.npz`` file."""
    arrays: dict[str, np.ndarray] = {"redshifts": np.array([s.z for s in snapshots])}
    for i, s in enumerate(snapshots):
        arrays[f"slice_{i}"] = s.density_slice
        arrays[f"k_{i}"] = s.k
        arrays[f"power_{i}"] = s.power
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def plot_density_montage(snapshots: list[Snapshot], cfg: PipelineConfig, path: Path) -> None:
    """1 x N panel figure of the mid-plane density slice, log colour scale."""
    import matplotlib.pyplot as plt

    # Use the snapshot closest to each requested redshift.
    chosen = [min(snapshots, key=lambda s: abs(s.z - z)) for z in cfg.montage_redshifts]

    fig, axes = plt.subplots(
        1, len(chosen), figsize=(4.0 * len(chosen), 4.5), constrained_layout=True
    )
    norm = LogNorm(vmin=0.1, vmax=100.0)
    image = None
    for ax, snap in zip(np.atleast_1d(axes), chosen):
        # Light blur removes single-particle pixel artifacts from the sparse sampling.
        field = gaussian_filter(snap.density_slice, sigma=cfg.slice_blur_sigma)
        image = ax.imshow(
            field.T,
            origin="lower",
            extent=(0.0, cfg.box_size, 0.0, cfg.box_size),
            norm=norm,
            cmap="inferno",
        )
        ax.set_title(f"z = {snap.z:g}")
        ax.set_xlabel("Mpc/h")
    np.atleast_1d(axes)[0].set_ylabel("Mpc/h")
    fig.colorbar(image, ax=axes, shrink=0.85, label=r"$1+\delta$")
    fig.suptitle("Cosmic web formation: Zel'dovich ICs $\\to$ PM $N$-body")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_power_spectra(snapshots: list[Snapshot], path: Path) -> None:
    """Log-log P(k) for every snapshot that has signal above the shot noise."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    with_signal = [s for s in snapshots if s.k.size > 0]
    colors = plt.cm.plasma(np.linspace(0.0, 0.85, max(len(with_signal), 1)))
    for color, snap in zip(colors, with_signal):
        ax.loglog(snap.k, snap.power, marker="o", ms=3, lw=1.5, color=color, label=f"z = {snap.z:g}")

    ax.set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]")
    ax.set_ylabel(r"$P(k)$  [$(\mathrm{Mpc}/h)^3$]")
    ax.set_title("Matter power spectrum evolution")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def parse_args() -> PipelineConfig:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(
        description="Cosmic Web Formation Simulator: full PM simulation from z=49 to z=0."
    )
    parser.add_argument("--n-particles", type=int, default=defaults.n_particles_per_side,
                        help="particles per side (N = n^3)")
    parser.add_argument("--grid", type=int, default=defaults.grid_size, help="PM mesh nodes per side")
    parser.add_argument("--box-size", type=float, default=defaults.box_size, help="box side [Mpc/h]")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="random seed for the ICs")
    parser.add_argument("--steps-per-efold", type=int, default=defaults.steps_per_efold,
                        help="leapfrog steps per unit ln(a)")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir,
                        help="directory for figures and snapshot data")
    args = parser.parse_args()
    return PipelineConfig(
        n_particles_per_side=args.n_particles,
        grid_size=args.grid,
        box_size=args.box_size,
        seed=args.seed,
        steps_per_efold=args.steps_per_efold,
        output_dir=args.output_dir,
    )


def main() -> None:
    matplotlib.use("Agg")  # headless: figures are written to disk, never shown
    cfg = parse_args()

    snapshots = run_simulation(cfg)

    out = cfg.output_dir
    save_snapshots(snapshots, out / "snapshots.npz")
    plot_density_montage(snapshots, cfg, out / "cosmic_web_evolution.png")
    plot_power_spectra(snapshots, out / "power_spectrum_evolution.png")
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
