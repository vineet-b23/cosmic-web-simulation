"""Experiment 003: isolated cosmological expansion (no gravity).

Two massless particles, so the gravitational acceleration is exactly zero,
start at rest in the Hubble flow (zero peculiar velocity) at a fixed comoving
separation ``d``. With ``g = 0`` and ``v = 0`` the comoving positions must
never change, so the physical separation is purely the expansion::

    r(a) = a * d

The run integrates from ``a = 0.1`` to ``a = 1.0`` with the comoving KDK
leapfrog and compares against that analytic result. A control run with the
cosmology switched off (same integrator, ``cosmology=None``) shows that the
separation then stays constant, demonstrating the toggle.

What this checks: the integrator adds no spurious motion when the peculiar
velocity is zero, and the scale-factor integration (RK4 on ``da/dt = a H(a)``)
reproduces the exact flat Lambda-CDM solution. Hubble drag and the ``1/a``
drift are exercised separately by giving particles non-zero velocities.

Units: ``H0 = 1`` (time in units of the Hubble time), comoving length arbitrary.

Run from anywhere::

    python experiments/exp_003_expansion.py [--steps 200] [--omega-m 0.3]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cosmology import Cosmology  # noqa: E402
from src.gravity import DirectGravity  # noqa: E402
from src.integrator import LeapfrogKDK  # noqa: E402
from src.particles import FloatArray, ParticleSet  # noqa: E402

OUTPUT_DIR = ROOT / "results" / "exp_003_expansion"


@dataclass(frozen=True)
class ExpansionConfig:
    H0: float = 1.0
    Omega_m: float = 0.3
    a_start: float = 0.1
    a_end: float = 1.0
    comoving_separation: float = 1.0
    n_steps: int = 200

    @property
    def cosmology(self) -> Cosmology:
        return Cosmology(H0=self.H0, Omega_m=self.Omega_m, Omega_Lambda=1.0 - self.Omega_m)


@dataclass(frozen=True)
class ExpansionRun:
    """Time series recorded at every step (index 0 is the initial state)."""

    times: FloatArray  # (T,) time since a_start
    scale_factor: FloatArray  # (T,) integrator's a (all 1 for a static run)
    separation: FloatArray  # (T,) physical separation
    comoving_separation: FloatArray  # (T,) |x1 - x0| in integrator coordinates
    max_speed: FloatArray  # (T,) largest peculiar-velocity magnitude


def make_massless_pair(cfg: ExpansionConfig) -> ParticleSet:
    """Two massless particles at rest, separated by ``cfg.comoving_separation`` along x."""
    half = 0.5 * cfg.comoving_separation
    positions = np.array([[-half, 0.0, 0.0], [half, 0.0, 0.0]])
    return ParticleSet(positions, np.zeros((2, 3)), np.zeros(2))


def run_pair(cfg: ExpansionConfig, expanding: bool) -> ExpansionRun:
    """Integrate the pair from ``a_start`` to ``a_end`` in cosmic time.

    Args:
        expanding: ``True`` turns the cosmology on (comoving integration);
            ``False`` runs the identical setup as a static Newtonian box.
    """
    cosmo = cfg.cosmology
    particles = make_massless_pair(cfg)
    gravity = DirectGravity(softening=1.0e-3)  # exactly zero force: both masses are 0

    integrator: LeapfrogKDK
    if expanding:
        def comoving_force(p: ParticleSet, a: float) -> FloatArray:
            del a  # direct Newtonian gravity has no a-dependence; forces vanish here anyway
            return gravity(p)

        integrator = LeapfrogKDK(comoving_force, cosmology=cosmo, a_initial=cfg.a_start)
    else:
        integrator = LeapfrogKDK(gravity)

    dt = cosmo.time_between(cfg.a_start, cfg.a_end) / cfg.n_steps
    n = cfg.n_steps + 1
    times = np.empty(n)
    a_hist = np.empty(n)
    sep = np.empty(n)
    sep_comoving = np.empty(n)
    vmax = np.empty(n)

    def record(k: int, p: ParticleSet) -> None:
        a = integrator.a
        d = float(np.linalg.norm(p.positions[1] - p.positions[0]))
        times[k] = integrator.time
        a_hist[k] = a
        sep_comoving[k] = d
        sep[k] = a * d if expanding else d  # positions are already physical when static
        vmax[k] = float(np.linalg.norm(p.velocities, axis=1).max())

    record(0, particles)
    integrator.run(particles, dt, cfg.n_steps, callback=record)
    return ExpansionRun(times, a_hist, sep, sep_comoving, vmax)


def analytic_scale_factor(cfg: ExpansionConfig, times: FloatArray) -> FloatArray:
    """Exact ``a(t)`` from the closed-form flat Lambda-CDM solution."""
    cosmo = cfg.cosmology
    return cosmo.scale_factor(cosmo.age(cfg.a_start) + times)


# ---------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------- #
def plot_results(
    cfg: ExpansionConfig,
    expanding: ExpansionRun,
    static: ExpansionRun,
    a_exact: FloatArray,
    path: Path,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    a = expanding.scale_factor
    d = cfg.comoving_separation
    r_exact = a_exact * d
    rel_err = np.abs(expanding.separation / r_exact - 1.0)

    fig = Figure(figsize=(8.0, 7.5), dpi=130)
    FigureCanvasAgg(fig)
    ax0, ax1 = fig.subplots(2, 1, sharex=True, height_ratios=(3, 1.4))

    ax0.plot(a_exact, r_exact, color="0.35", lw=2.4, label=r"analytic  $r = a\,d$")
    ax0.plot(a[::8], expanding.separation[::8], "o", color="tab:blue", ms=5,
             label="comoving leapfrog (cosmology on)")
    ax0.plot(a, static.separation, "--", color="tab:red", lw=1.6,
             label="control: cosmology off (static box)")
    ax0.set_ylabel(r"physical separation $r$  [comoving units]")
    ax0.set_title(
        rf"Two massless particles, $\Omega_m={cfg.Omega_m:g}$, "
        rf"$\Omega_\Lambda={1.0 - cfg.Omega_m:g}$, $d_\mathrm{{comoving}}={d:g}$"
    )
    ax0.legend(loc="lower right")
    ax0.grid(alpha=0.3)

    ax1.semilogy(a, np.maximum(rel_err, 1e-17), color="tab:blue")
    ax1.set_ylabel(r"$|r_\mathrm{num}/r_\mathrm{exact} - 1|$")
    ax1.set_xlabel("scale factor $a$")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def main() -> None:
    d = ExpansionConfig()
    parser = argparse.ArgumentParser(description="Isolated expansion test (massless particles)")
    parser.add_argument("--steps", type=int, default=d.n_steps)
    parser.add_argument("--omega-m", type=float, default=d.Omega_m,
                        help="matter density; Omega_Lambda = 1 - Omega_m (flat)")
    parser.add_argument("--a-start", type=float, default=d.a_start)
    parser.add_argument("--a-end", type=float, default=d.a_end)
    parser.add_argument("--separation", type=float, default=d.comoving_separation,
                        help="initial comoving separation")
    args = parser.parse_args()

    cfg = ExpansionConfig(
        Omega_m=args.omega_m, a_start=args.a_start, a_end=args.a_end,
        comoving_separation=args.separation, n_steps=args.steps,
    )

    expanding = run_pair(cfg, expanding=True)
    static = run_pair(cfg, expanding=False)
    a_exact = analytic_scale_factor(cfg, expanding.times)
    r_exact = a_exact * cfg.comoving_separation

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_DIR / "expansion.npz",
        times=expanding.times,
        scale_factor=expanding.scale_factor,
        scale_factor_exact=a_exact,
        separation=expanding.separation,
        separation_exact=r_exact,
        separation_static=static.separation,
    )
    plot_results(cfg, expanding, static, a_exact, OUTPUT_DIR / "expansion.png")

    rel_err = np.abs(expanding.separation / r_exact - 1.0)
    print(f"Cosmic time span          : {expanding.times[-1]:.6f} / H0")
    print(f"Final scale factor        : {expanding.scale_factor[-1]:.10f}  (target {cfg.a_end:g})")
    print(f"Max |a_num - a_exact|     : {np.abs(expanding.scale_factor - a_exact).max():.3e}")
    print(f"Final separation r(a_end) : {expanding.separation[-1]:.10f}  "
          f"(analytic {r_exact[-1]:.10f})")
    print(f"Max |r/r_exact - 1|       : {rel_err.max():.3e}")
    print(f"Max comoving drift        : {np.abs(expanding.comoving_separation - cfg.comoving_separation).max():.3e}")
    print(f"Max peculiar speed        : {expanding.max_speed.max():.3e}")
    print(f"Control (cosmology off)   : r stays {static.separation.min():.6f} to "
          f"{static.separation.max():.6f}")
    print(f"Outputs written to        : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
