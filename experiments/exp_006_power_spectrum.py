"""Experiment 006: matter power spectrum evolution."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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

N_GRID_PARTICLES = 32          # N = 32**3 particles
PM_GRID_SIZE = 64              # 64**3 PM mesh
BOX_SIZE = 100.0               # Mpc/h
A_INIT = 0.02                  # z = 49
G = 1.0
STEPS_PER_SEGMENT = 60
SEED = 42

OUTPUT_PATH = ROOT / "results" / "power_spectrum_evolution.png"


def main() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cosmo = Cosmology(H0=100.0, Omega_m=0.3, Omega_Lambda=0.7)

    def power_spectrum(k: np.ndarray) -> np.ndarray:
        return power_spectrum_bbks(k, cosmo, ns=-2.0, sigma8=0.2)

    x0, v0, masses = zeldovich_initial_conditions_arrays(
        N_GRID_PARTICLES, BOX_SIZE, cosmo, A_INIT, power_spectrum, seed=SEED, max_delta_init=0.05
    )
    particles = ParticleSet(x0, v0, masses)
    mean_density = masses.sum() / BOX_SIZE**3

    pm = PMGravity.from_cosmology(PM_GRID_SIZE, BOX_SIZE, cosmo)
    integrator = LeapfrogKDK(pm, cosmology=cosmo, a_initial=A_INIT)

    redshifts = [10.0, 3.0, 1.0, 0.0]
    snapshot_a = [1.0 / (1.0 + z) for z in redshifts]

    results: list[tuple[np.ndarray, np.ndarray]] = []
    for a_target in snapshot_a:
        dt = cosmo.time_between(integrator.a, a_target) / STEPS_PER_SEGMENT
        integrator.run(particles, dt, STEPS_PER_SEGMENT)

        stencil = cic_stencil(particles.positions, PM_GRID_SIZE, BOX_SIZE)
        density = deposit_cic(stencil, particles.masses)
        delta = overdensity(density, mean_density)

        k, p_k = compute_power_spectrum(delta, BOX_SIZE, n_particles=32768)
        results.append((k, p_k))

    fig, ax = plt.subplots(figsize=(7, 6))
    for z, (k, p_k) in zip(redshifts, results):
        ax.loglog(k, p_k, marker="o", ms=3, lw=1.5, label=f"z = {z:g}")

    ax.set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]")
    ax.set_ylabel(r"$P(k)$  [$(\mathrm{Mpc}/h)^3$]")
    ax.set_title("Matter power spectrum evolution")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)


if __name__ == "__main__":
    main()
