"""Experiment 004: Particle-Mesh vs direct summation.

Part A - speed. Time per timestep for direct ``O(N^2)`` summation and for PM
on a 64^3 mesh, for N = 1,000 / 5,000 / 10,000 / 50,000 particles placed
uniformly at random in a periodic unit box. A leapfrog step reuses the force
from the end of the previous step, so its cost is one force evaluation plus
``O(N)`` array updates; the force evaluation is what is timed.

Part B - accuracy. For N = 1,000 the PM acceleration is compared particle by
particle with softened direct summation, using the relative error
``|a_PM - a_ref| / |a_ref|``. PM is periodic, so the direct reference must be
periodic too. Three references are computed to show why that matters:

* **Ewald** (``EwaldGravity``): exact periodic direct sum. This is the
  reference the reported PM error is measured against.
* **Minimum image** (``DirectGravity(box_size=L)``): each pair uses its
  nearest image. Cheap, but it truncates the periodic sum, so it carries an
  error of its own.
* **Isolated** (``DirectGravity()``): ignores periodicity entirely. It solves a
  different physical problem and is printed only as a cautionary number.

The direct softening is ``softening_cells * h`` with ``h = L / M`` the mesh
spacing. PM cannot resolve forces below a few cells, so the error depends on
this choice; a small sensitivity table is printed.

Outputs (``results/exp_004_pm_vs_direct/``): ``time_vs_n.png``,
``force_error_hist.png`` and ``benchmark.npz``.

Run from anywhere::

    python experiments/exp_004_pm_vs_direct.py [--skip-direct-above 10000]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gravity import DirectGravity, EwaldGravity, PMGravity  # noqa: E402
from src.mesh import cic_stencil, deposit_cic, interpolate_cic  # noqa: E402
from src.particles import FloatArray, ParticleSet  # noqa: E402
from src.poisson import PoissonSolver, overdensity  # noqa: E402

OUTPUT_DIR = ROOT / "results" / "exp_004_pm_vs_direct"


@dataclass(frozen=True)
class BenchmarkConfig:
    box_size: float = 1.0
    grid_size: int = 64
    particle_counts: tuple[int, ...] = (1000, 5000, 10000, 50000)
    error_n: int = 1000
    softening_cells: float = 0.5  # direct softening in units of the mesh spacing
    softening_sweep: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    G: float = 1.0
    seed: int = 12345

    @property
    def cell_size(self) -> float:
        return self.box_size / self.grid_size


@dataclass(frozen=True)
class TimingRow:
    n: int
    direct_seconds: float  # NaN if skipped
    pm_seconds: float
    pm_stage_seconds: tuple[float, float, float]  # (CIC deposit, FFT solve, CIC interpolate)


# ---------------------------------------------------------------------- #
# Setup and timing helpers
# ---------------------------------------------------------------------- #
def make_random_particles(n: int, box_size: float, seed: int) -> ParticleSet:
    """``n`` equal-mass particles, uniform in the periodic box, at rest (total mass 1)."""
    rng = np.random.default_rng(seed)
    return ParticleSet(rng.random((n, 3)) * box_size, np.zeros((n, 3)), np.full(n, 1.0 / n))


def best_time(fn: Callable[[], object], repeats: int, warmup: bool) -> float:
    """Minimum wall time of ``repeats`` calls (the minimum is the least noisy estimate)."""
    if warmup:
        fn()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def direct_effort(n: int) -> tuple[int, bool]:
    """(repeats, warm-up?) for the direct solver; fewer repeats where one call is slow."""
    if n <= 1000:
        return 5, True
    if n <= 5000:
        return 3, True
    if n <= 10000:
        return 2, False
    return 1, False


def time_pm_stages(
    particles: ParticleSet, cfg: BenchmarkConfig, solver: PoissonSolver
) -> tuple[float, float, float]:
    """Best-of-3 time of the three PM stages for one particle set."""
    m, box = cfg.grid_size, cfg.box_size
    mean_density = particles.total_mass / box**3
    stencil = cic_stencil(particles.positions, m, box)
    density = deposit_cic(stencil, particles.masses)
    delta = overdensity(density, mean_density)
    field = solver.acceleration(delta, mean_density)

    def deposit() -> object:
        return deposit_cic(cic_stencil(particles.positions, m, box), particles.masses)

    return (
        best_time(deposit, 3, warmup=True),
        best_time(lambda: solver.acceleration(delta, mean_density), 3, warmup=True),
        best_time(lambda: interpolate_cic(stencil, field), 3, warmup=True),
    )


def run_timing(cfg: BenchmarkConfig, skip_direct_above: int | None) -> list[TimingRow]:
    pm = PMGravity(cfg.grid_size, cfg.box_size, cfg.G)
    solver = PoissonSolver(cfg.grid_size, cfg.box_size, cfg.G)
    # Plain (isolated) direct sum: the cheapest O(N^2) baseline, so PM's speed-up is not inflated.
    direct = DirectGravity(softening=cfg.softening_cells * cfg.cell_size, G=cfg.G)
    rows: list[TimingRow] = []
    for n in cfg.particle_counts:
        particles = make_random_particles(n, cfg.box_size, cfg.seed)
        t_pm = best_time(lambda: pm(particles), repeats=5, warmup=True)
        stages = time_pm_stages(particles, cfg, solver)

        if skip_direct_above is not None and n > skip_direct_above:
            t_direct = float("nan")
        else:
            repeats, warmup = direct_effort(n)
            print(f"  N = {n:>6}: timing direct summation ({repeats} repeat(s)) ...", flush=True)
            t_direct = best_time(lambda: direct(particles), repeats, warmup)
        rows.append(TimingRow(n, t_direct, t_pm, stages))
    return rows


# ---------------------------------------------------------------------- #
# Force accuracy
# ---------------------------------------------------------------------- #
def relative_error(a: FloatArray, reference: FloatArray) -> FloatArray:
    """Per-particle ``|a - ref| / |ref|``, shape ``(N,)``."""
    return np.linalg.norm(a - reference, axis=1) / np.linalg.norm(reference, axis=1)  # type: ignore[no-any-return]


@dataclass(frozen=True)
class AccuracyResult:
    err_ewald: FloatArray  # PM vs exact periodic direct sum
    err_min_image: FloatArray  # PM vs minimum-image direct sum
    err_isolated: FloatArray  # PM vs isolated direct sum (not comparable)
    err_reference_min_image: FloatArray  # minimum-image direct vs Ewald (reference quality)
    rms_force: float
    sweep: dict[float, tuple[float, float]]  # softening (cells) -> (median, p95) of PM vs Ewald


def run_accuracy(cfg: BenchmarkConfig) -> AccuracyResult:
    particles = make_random_particles(cfg.error_n, cfg.box_size, cfg.seed + 1)
    h, box = cfg.cell_size, cfg.box_size
    eps = cfg.softening_cells * h

    a_pm = PMGravity(cfg.grid_size, box, cfg.G)(particles)
    a_ewald = EwaldGravity(box, eps, cfg.G)(particles)
    a_min_image = DirectGravity(eps, cfg.G, box_size=box)(particles)
    a_isolated = DirectGravity(eps, cfg.G)(particles)

    sweep: dict[float, tuple[float, float]] = {}
    for cells in cfg.softening_sweep:
        ref = a_ewald if cells == cfg.softening_cells else EwaldGravity(box, cells * h, cfg.G)(particles)
        err = relative_error(a_pm, ref)
        sweep[cells] = (float(np.median(err)), float(np.percentile(err, 95)))

    return AccuracyResult(
        err_ewald=relative_error(a_pm, a_ewald),
        err_min_image=relative_error(a_pm, a_min_image),
        err_isolated=relative_error(a_pm, a_isolated),
        err_reference_min_image=relative_error(a_min_image, a_ewald),
        rms_force=float(np.sqrt(np.mean(np.einsum("ij,ij->i", a_ewald, a_ewald)))),
        sweep=sweep,
    )


# ---------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------- #
def percentiles(err: FloatArray) -> str:
    p50, p68, p95, p99 = np.percentile(err, [50, 68, 95, 99])
    return f"median {100 * p50:6.1f}%   68% {100 * p68:6.1f}%   95% {100 * p95:6.1f}%   99% {100 * p99:6.1f}%"


def plot_timing(rows: list[TimingRow], cfg: BenchmarkConfig, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n = np.array([r.n for r in rows], dtype=float)
    t_direct = np.array([r.direct_seconds for r in rows])
    t_pm = np.array([r.pm_seconds for r in rows])
    have = np.isfinite(t_direct)

    fig = Figure(figsize=(7.5, 5.6), dpi=130)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.loglog(n[have], t_direct[have], "o-", color="tab:red", lw=2, ms=7, label="Direct summation")
    ax.loglog(n, t_pm, "s-", color="tab:blue", lw=2, ms=7, label=f"PM ({cfg.grid_size}$^3$ mesh)")
    if have.any():
        first = int(np.argmax(have))
        guide = t_direct[first] * (n / n[first]) ** 2
        ax.loglog(n, guide, "--", color="0.55", lw=1.2, label=r"$\propto N^2$ guide")
    ax.set_xlabel("particle count $N$")
    ax.set_ylabel("time per timestep [s]  (one force evaluation)")
    ax.set_title("Force-evaluation cost: direct summation vs Particle-Mesh")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def plot_error_histogram(res: AccuracyResult, cfg: BenchmarkConfig, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    bins = np.logspace(-3, 1, 60)
    fig = Figure(figsize=(8.0, 5.4), dpi=130)
    FigureCanvasAgg(fig)
    ax = fig.subplots()

    series = (
        (res.err_ewald, "tab:blue", "PM vs exact periodic direct (Ewald)"),
        (res.err_min_image, "tab:orange", "PM vs minimum-image direct"),
    )
    for err, color, label in series:
        ax.hist(np.clip(err, bins[0], bins[-1]), bins=bins, histtype="stepfilled",
                color=color, alpha=0.35, label=label)
        ax.hist(np.clip(err, bins[0], bins[-1]), bins=bins, histtype="step", color=color, lw=1.6)
        ax.axvline(np.median(err), color=color, ls="--", lw=1.4)

    ax.set_xscale("log")
    ax.set_xlabel(r"relative force error  $|a_\mathrm{PM} - a_\mathrm{direct}| / |a_\mathrm{direct}|$")
    ax.set_ylabel("number of particles")
    ax.set_title(
        f"PM force error, N = {cfg.error_n}, {cfg.grid_size}$^3$ mesh, "
        f"softening = {cfg.softening_cells:g} cell (dashed: medians)"
    )
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def main() -> None:
    d = BenchmarkConfig()
    parser = argparse.ArgumentParser(description="PM vs direct summation: speed and accuracy")
    parser.add_argument("--grid", type=int, default=d.grid_size, help="PM mesh nodes per side")
    parser.add_argument("--softening-cells", type=float, default=d.softening_cells,
                        help="direct-sum softening in units of the mesh spacing")
    parser.add_argument("--skip-direct-above", type=int, default=None,
                        help="do not time direct summation for N above this (it is slow)")
    parser.add_argument("--seed", type=int, default=d.seed)
    args = parser.parse_args()

    cfg = BenchmarkConfig(grid_size=args.grid, softening_cells=args.softening_cells, seed=args.seed)
    if cfg.softening_cells not in cfg.softening_sweep:
        cfg = replace(cfg, softening_sweep=tuple(sorted(cfg.softening_sweep + (cfg.softening_cells,))))

    print("Part A: time per timestep")
    rows = run_timing(cfg, args.skip_direct_above)
    print(f"\n{'N':>7} {'direct [s]':>12} {'PM [s]':>10} {'speed-up':>9}   "
          f"PM stages [ms]: deposit / FFT solve / interpolate")
    for r in rows:
        speedup = r.direct_seconds / r.pm_seconds
        s = r.pm_stage_seconds
        print(f"{r.n:>7} {r.direct_seconds:>12.4f} {r.pm_seconds:>10.4f} {speedup:>8.1f}x   "
              f"{1e3 * s[0]:6.1f} / {1e3 * s[1]:6.1f} / {1e3 * s[2]:6.1f}")

    print(f"\nPart B: force error at N = {cfg.error_n} "
          f"(softening {cfg.softening_cells:g} cell = {cfg.softening_cells * cfg.cell_size:.5f})")
    res = run_accuracy(cfg)
    print(f"  PM vs exact periodic direct (Ewald)  : {percentiles(res.err_ewald)}")
    print(f"  PM vs minimum-image direct           : {percentiles(res.err_min_image)}")
    print(f"  (reference check) min-image vs Ewald : {percentiles(res.err_reference_min_image)}")
    print(f"  PM vs isolated direct (not comparable): {percentiles(res.err_isolated)}")
    print("  Softening sensitivity, PM vs Ewald (median / 95th percentile):")
    for cells, (med, p95) in sorted(res.sweep.items()):
        print(f"    eps = {cells:4.2f} cell : {100 * med:6.1f}% / {100 * p95:6.1f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_DIR / "benchmark.npz",
        n=np.array([r.n for r in rows]),
        direct_seconds=np.array([r.direct_seconds for r in rows]),
        pm_seconds=np.array([r.pm_seconds for r in rows]),
        pm_stage_seconds=np.array([r.pm_stage_seconds for r in rows]),
        err_ewald=res.err_ewald,
        err_min_image=res.err_min_image,
        err_isolated=res.err_isolated,
    )
    plot_timing(rows, cfg, OUTPUT_DIR / "time_vs_n.png")
    plot_error_histogram(res, cfg, OUTPUT_DIR / "force_error_hist.png")
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
