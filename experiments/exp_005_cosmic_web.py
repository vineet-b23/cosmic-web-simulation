# --- experiments/exp_005_cosmic_web.py ---

"""Experiment 005: cosmic web formation with Zel'dovich ICs + PM gravity."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

OUTPUT_PATH = ROOT / "results" / "cosmic_web_evolution.png"


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

    pm = PMGravity(PM_GRID_SIZE, BOX_SIZE, G)
    integrator = LeapfrogKDK(pm, cosmology=cosmo, a_initial=A_INIT)

    redshifts = [49.0, 10.0, 3.0, 1.0, 0.0]
    snapshot_a = [1.0 / (1.0 + z) for z in redshifts]
    snapshot_a[0] = A_INIT  # z = 49 <-> a = 0.02 exactly, matching a_init

    one_plus_delta_slices: list[np.ndarray] = []
    for i, a_target in enumerate(snapshot_a):
        if i > 0:
            dt = cosmo.time_between(integrator.a, a_target) / STEPS_PER_SEGMENT
            integrator.run(particles, dt, STEPS_PER_SEGMENT)

        stencil = cic_stencil(particles.positions, PM_GRID_SIZE, BOX_SIZE)
        density = deposit_cic(stencil, particles.masses)
        delta = overdensity(density, mean_density)
        assert abs((1.0 + delta).mean() - 1.0) < 1.0e-10

        mid = PM_GRID_SIZE // 2
        one_plus_delta_slices.append(gaussian_filter(1.0 + delta[:, :, mid], sigma=1.0))

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), constrained_layout=True)
    norm = LogNorm(vmin=0.1, vmax=100.0)
    im = None
    for ax, z, field in zip(axes, redshifts, one_plus_delta_slices):
        im = ax.imshow(
            field.T,
            origin="lower",
            extent=(0.0, BOX_SIZE, 0.0, BOX_SIZE),
            norm=norm,
            cmap="inferno",
        )
        ax.set_title(f"z = {z:g}")
        ax.set_xlabel("Mpc/h")
    axes[0].set_ylabel("Mpc/h")
    fig.colorbar(im, ax=axes, shrink=0.85, label=r"$1+\delta$")
    fig.suptitle("Cosmic web formation: Zel'dovich ICs $\\to$ PM $N$-body")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)


if __name__ == "__main__":
    main()