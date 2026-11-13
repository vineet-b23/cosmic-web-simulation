"""Newtonian gravity solvers: direct summation and Particle-Mesh (PM).

Both solvers are callables ``solver(particles) -> accelerations`` of shape
``(N, D)`` and are interchangeable in the integrator; :func:`make_gravity`
builds either from a method name.

Direct summation (:class:`DirectGravity`)
-----------------------------------------
Acceleration of particle ``i``::

    a_i = -G * sum_{j != i} m_j (x_i - x_j) / (|x_i - x_j|^2 + eps^2)^(3/2)

The matching softened potential is ``phi(r) = -G m / sqrt(r^2 + eps^2)``, so
the force is exactly its gradient and total energy is conserved by a
symplectic integrator. Every pairwise quantity is computed with NumPy
broadcasting; there is no Python loop over particles. The full ``(N, N, D)``
separation tensor is built in one shot whenever it fits in
``MEMORY_BUDGET_BYTES`` (about 128 MB, i.e. N up to roughly 2,300 in 3-D).
For larger N the target particles are split into row blocks to bound memory;
the loop then runs over blocks of rows, never over individual particles.
Cost is ``O(N^2)``.

Optionally, ``box_size`` applies the minimum-image convention so each pair
uses its nearest periodic image. This makes the direct sum a reasonable
reference for periodic (PM) forces when the mass is not spread over more than
about half the box.

Periodic reference (:class:`EwaldGravity`)
------------------------------------------
Exact fully periodic direct summation by Ewald's method, for measuring the
accuracy of PM. The minimum-image option above truncates the periodic sum
and is only approximate, and an isolated (open) sum is a different physical
problem, so neither is a clean reference for a periodic PM solver.

Particle-Mesh (:class:`PMGravity`)
----------------------------------
CIC mass assignment (:mod:`src.mesh`), an FFT Poisson solve on a periodic
mesh (:mod:`src.poisson`) and CIC interpolation of the resulting acceleration
back to the particles. Cost is ``O(N + M^3 log M)``. The mesh smooths the
force below a few cell sizes, so PM is not a softened-direct-sum drop-in at
small separations; it is accurate for separations of a few cells and larger.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from src.mesh import cic_stencil, deposit_cic, interpolate_cic
from src.particles import FloatArray, ParticleSet
from src.poisson import PoissonSolver, overdensity

MEMORY_BUDGET_BYTES: int = 128 * 1024 * 1024

GravityMethod = Literal["direct", "pm"]


class GravitySolver(Protocol):
    """Anything that maps a particle set to accelerations of shape ``(N, D)``."""

    def __call__(self, particles: ParticleSet) -> FloatArray: ...


# ---------------------------------------------------------------------- #
# Direct summation
# ---------------------------------------------------------------------- #
def _validate(softening: float, block_size: int | None, box_size: float | None = None) -> None:
    if not np.isfinite(softening) or softening < 0.0:
        raise ValueError(f"softening must be finite and >= 0; got {softening}")
    if block_size is not None and block_size < 1:
        raise ValueError(f"block_size must be >= 1 (or None for auto); got {block_size}")
    if box_size is not None and (not np.isfinite(box_size) or box_size <= 0.0):
        raise ValueError(f"box_size must be finite and > 0 (or None); got {box_size}")


def _resolve_block_size(n: int, dim: int, block_size: int | None) -> int:
    """Number of target rows per broadcast block (``n`` means single shot)."""
    if block_size is not None:
        return min(block_size, max(n, 1))
    rows = MEMORY_BUDGET_BYTES // (max(n, 1) * max(dim, 1) * 8)
    return min(max(rows, 1), max(n, 1))


def _separations(
    positions: FloatArray, start: int, stop: int, box_size: float | None
) -> FloatArray:
    """``dx[i, j] = x_i - x_j`` for targets ``start:stop``, minimum-imaged if periodic."""
    dx = positions[start:stop, None, :] - positions[None, :, :]  # (B, N, D)
    if box_size is not None:
        dx -= box_size * np.round(dx / box_size)
    return dx


def _acceleration_block(
    positions: FloatArray,
    masses: FloatArray,
    start: int,
    stop: int,
    eps2: float,
    G: float,
    box_size: float | None,
) -> FloatArray:
    """Accelerations of targets ``start:stop`` from all sources (no Python loops)."""
    rows = np.arange(stop - start)
    cols = rows + start

    dx = _separations(positions, start, stop, box_size)
    r2 = np.einsum("ijk,ijk->ij", dx, dx)
    r2 += eps2

    # Remove self-interaction (and avoid 0**-1.5 when eps == 0).
    r2[rows, cols] = 1.0
    w = r2**-1.5
    w[rows, cols] = 0.0
    w *= masses[None, :]

    return -G * np.einsum("ij,ijk->ik", w, dx)


def _potential_block(
    positions: FloatArray,
    masses: FloatArray,
    start: int,
    stop: int,
    eps2: float,
    box_size: float | None,
) -> float:
    """Sum over targets ``start:stop`` of ``m_i sum_{j != i} m_j / sqrt(r_ij^2 + eps^2)``."""
    rows = np.arange(stop - start)
    cols = rows + start

    dx = _separations(positions, start, stop, box_size)
    r2 = np.einsum("ijk,ijk->ij", dx, dx)
    r2 += eps2
    r2[rows, cols] = 1.0
    inv_r = r2**-0.5
    inv_r[rows, cols] = 0.0

    return float(masses[start:stop] @ (inv_r @ masses))


def compute_accelerations(
    positions: FloatArray,
    masses: FloatArray,
    softening: float,
    G: float = 1.0,
    block_size: int | None = None,
    box_size: float | None = None,
) -> FloatArray:
    """Return softened Newtonian accelerations, shape ``(N, D)``.

    Args:
        positions: ``(N, D)`` particle positions.
        masses: ``(N,)`` particle masses.
        softening: Plummer softening length ``eps`` (>= 0).
        G: Gravitational constant in the chosen unit system.
        block_size: Target rows per broadcast block. ``None`` (default) picks
            the largest block that fits the memory budget, which is a single
            fully-broadcast evaluation for N up to ~2,300 in 3-D.
        box_size: If given, use the minimum-image convention in a periodic box
            of this side. ``None`` (default) is an isolated (open) system.

    Note:
        With ``softening == 0`` two distinct particles at exactly the same
        position still produce a singularity; use ``softening > 0`` for
        systems that may have close encounters.
    """
    _validate(softening, block_size, box_size)
    n, dim = positions.shape
    eps2 = softening * softening
    bs = _resolve_block_size(n, dim, block_size)

    if bs >= n:
        return _acceleration_block(positions, masses, 0, n, eps2, G, box_size)

    acc = np.empty_like(positions)
    for start in range(0, n, bs):  # loop over row blocks, not particles
        stop = min(start + bs, n)
        acc[start:stop] = _acceleration_block(positions, masses, start, stop, eps2, G, box_size)
    return acc


def compute_potential_energy(
    positions: FloatArray,
    masses: FloatArray,
    softening: float,
    G: float = 1.0,
    block_size: int | None = None,
    box_size: float | None = None,
) -> float:
    """Return the softened potential energy ``-G sum_{i<j} m_i m_j / sqrt(r_ij^2 + eps^2)``."""
    _validate(softening, block_size, box_size)
    n, dim = positions.shape
    eps2 = softening * softening
    bs = _resolve_block_size(n, dim, block_size)

    if bs >= n:
        total = _potential_block(positions, masses, 0, n, eps2, box_size)
    else:
        total = 0.0
        for start in range(0, n, bs):
            total += _potential_block(positions, masses, start, min(start + bs, n), eps2, box_size)

    # Each pair was counted twice: (i, j) and (j, i).
    return -0.5 * G * total


class DirectGravity:
    """Configurable direct-summation gravity solver, ``O(N^2)``.

    Instances are callable with a :class:`ParticleSet` and return the
    acceleration array, so they plug directly into the integrator.
    """

    __slots__ = ("G", "softening", "block_size", "box_size")

    def __init__(
        self,
        softening: float,
        G: float = 1.0,
        block_size: int | None = None,
        box_size: float | None = None,
    ) -> None:
        _validate(softening, block_size, box_size)
        self.G: float = G
        self.softening: float = softening
        self.block_size: int | None = block_size
        self.box_size: float | None = box_size

    def __call__(self, particles: ParticleSet) -> FloatArray:
        return compute_accelerations(
            particles.positions, particles.masses, self.softening, self.G,
            self.block_size, self.box_size,
        )

    def potential_energy(self, particles: ParticleSet) -> float:
        return compute_potential_energy(
            particles.positions, particles.masses, self.softening, self.G,
            self.block_size, self.box_size,
        )

    def total_energy(self, particles: ParticleSet) -> float:
        return particles.kinetic_energy + self.potential_energy(particles)


# ---------------------------------------------------------------------- #
# Periodic direct summation (Ewald) -- accuracy reference for PM
# ---------------------------------------------------------------------- #
# The 27 image cells (-1, 0, 1)^3 as float offsets, shape (27, 3).
_IMAGE_SHIFTS: FloatArray = np.indices((3, 3, 3)).reshape(3, 27).T.astype(np.float64) - 1.0


def _ewald_kvectors(box_size: float, alpha: float, k_cut_sq: int) -> tuple[FloatArray, FloatArray]:
    """Reciprocal lattice vectors ``k = 2 pi n / L`` (``0 < |n|^2 <= k_cut_sq``) and
    their weights ``(4 pi / L^3) exp(-k^2 / 4 alpha^2) / k^2``."""
    n_max = int(np.floor(np.sqrt(k_cut_sq)))
    r = np.arange(-n_max, n_max + 1)
    nx, ny, nz = np.meshgrid(r, r, r, indexing="ij")
    keep = (nx**2 + ny**2 + nz**2 > 0) & (nx**2 + ny**2 + nz**2 <= k_cut_sq)
    kvec = (2.0 * np.pi / box_size) * np.stack((nx[keep], ny[keep], nz[keep]), axis=1).astype(np.float64)
    k2 = np.einsum("ij,ij->i", kvec, kvec)
    coef = (4.0 * np.pi / box_size**3) * np.exp(-k2 / (4.0 * alpha**2)) / k2
    return kvec, coef


def _ewald_real_block(
    positions: FloatArray,
    masses: FloatArray,
    start: int,
    stop: int,
    eps2: float,
    box_size: float,
    alpha: float,
) -> FloatArray:
    """Real-space Ewald sum for targets ``start:stop`` (before the ``-G`` factor).

    The bare pair kernel ``1 / (r^2 + eps^2)^(3/2)`` has the Gaussian-smoothed
    long-range part ``[erf(a r) - 2 a r exp(-a^2 r^2) / sqrt(pi)] / r^3``
    removed; that part is smooth and is summed in reciprocal space instead.
    """
    try:
        from scipy.special import erf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("EwaldGravity needs scipy (scipy.special.erf)") from exc

    rows = np.arange(stop - start)
    cols = rows + start
    dx0 = _separations(positions, start, stop, box_size)  # nearest images, (B, N, 3)
    acc = np.zeros((stop - start, 3))
    two_over_sqrt_pi = 2.0 / np.sqrt(np.pi)

    for shift in _IMAGE_SHIFTS:  # loop over the 27 image cells
        d = dx0 + box_size * shift
        r2 = np.einsum("ijk,ijk->ij", d, d)
        is_home_cell = not shift.any()
        if is_home_cell:
            r2[rows, cols] = 1.0  # self pair: avoid 0/0, weight zeroed below
        r = np.sqrt(r2)
        ar = alpha * r
        long_range = (erf(ar) - two_over_sqrt_pi * ar * np.exp(-ar * ar)) / (r2 * r)
        w = (r2 + eps2) ** -1.5 - long_range
        if is_home_cell:
            w[rows, cols] = 0.0
        w *= masses[None, :]
        acc += np.einsum("ij,ijk->ik", w, d)
    return acc


def _ewald_reciprocal(
    positions: FloatArray,
    masses: FloatArray,
    kvec: FloatArray,
    coef: FloatArray,
    block_rows: int = 4096,
) -> FloatArray:
    """Reciprocal-space Ewald sum (before the ``-G`` factor) via structure factors."""
    n = positions.shape[0]
    cos_sum = np.zeros(kvec.shape[0])
    sin_sum = np.zeros(kvec.shape[0])
    for start in range(0, n, block_rows):  # loop over row blocks
        phase = positions[start : start + block_rows] @ kvec.T
        m = masses[start : start + block_rows]
        cos_sum += m @ np.cos(phase)
        sin_sum += m @ np.sin(phase)

    acc = np.empty_like(positions)
    for start in range(0, n, block_rows):
        phase = positions[start : start + block_rows] @ kvec.T
        # sum_j m_j sin(k . (x_i - x_j)) = sin(k x_i) C - cos(k x_i) S
        term = coef[None, :] * (np.sin(phase) * cos_sum - np.cos(phase) * sin_sum)
        acc[start : start + block_rows] = term @ kvec
    return acc


def compute_accelerations_ewald(
    positions: FloatArray,
    masses: FloatArray,
    softening: float,
    box_size: float,
    G: float = 1.0,
    alpha: float | None = None,
    k_cut_sq: int = 20,
    block_size: int | None = None,
) -> FloatArray:
    """Softened gravity in a fully periodic box via Ewald summation, shape ``(N, 3)``.

    Each particle interacts with all periodic images of every other particle
    (and of itself), with the uniform neutralising background implied by
    dropping the ``k = 0`` term, which is the same physics a PM solver
    computes. The Plummer softening acts on the short-range part only; the
    smooth long-range part is independent of ``eps`` (valid for
    ``eps << 1 / alpha``). Cost is ``O(27 N^2 + N K)`` with ``K`` reciprocal
    vectors, so this is a reference solver, not a production one.

    Args:
        alpha: Ewald splitting parameter (default ``3 / L``). The result is
            independent of it to the accuracy set by ``k_cut_sq``.
        k_cut_sq: Include reciprocal vectors with ``0 < |n|^2 <= k_cut_sq``.
    """
    _validate(softening, block_size, box_size)
    n, dim = positions.shape
    if dim != 3:
        raise ValueError(f"Ewald summation supports 3-D only; got dim={dim}")
    alpha = 3.0 / box_size if alpha is None else alpha
    eps2 = softening * softening

    bs = _resolve_block_size(n, dim, block_size)
    real = np.empty_like(positions)
    for start in range(0, n, bs):  # loop over row blocks, not particles
        stop = min(start + bs, n)
        real[start:stop] = _ewald_real_block(positions, masses, start, stop, eps2, box_size, alpha)

    kvec, coef = _ewald_kvectors(box_size, alpha, k_cut_sq)
    return -G * (real + _ewald_reciprocal(positions, masses, kvec, coef))


class EwaldGravity:
    """Periodic direct-summation gravity (Ewald), the exact reference for PM.

    Same calling convention as :class:`PMGravity`: ``solver(particles)`` gives
    the ordinary periodic acceleration and ``solver(particles, a)`` the
    peculiar acceleration for comoving coordinates (a factor ``1 / a^2``).
    Requires ``scipy``.
    """

    __slots__ = ("box_size", "softening", "G", "alpha", "k_cut_sq", "block_size")

    def __init__(
        self,
        box_size: float,
        softening: float = 0.0,
        G: float = 1.0,
        alpha: float | None = None,
        k_cut_sq: int = 20,
        block_size: int | None = None,
    ) -> None:
        _validate(softening, block_size, box_size)
        self.box_size: float = box_size
        self.softening: float = softening
        self.G: float = G
        self.alpha: float | None = alpha
        self.k_cut_sq: int = k_cut_sq
        self.block_size: int | None = block_size

    def __call__(self, particles: ParticleSet, a: float = 1.0) -> FloatArray:
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError(f"scale factor a must be finite and > 0; got {a}")
        acc = compute_accelerations_ewald(
            particles.positions, particles.masses, self.softening, self.box_size,
            self.G, self.alpha, self.k_cut_sq, self.block_size,
        )
        return acc / (a * a)


# ---------------------------------------------------------------------- #
# Particle-Mesh
# ---------------------------------------------------------------------- #
# --- src/gravity.py (PMGravity class) ---

class PMGravity:
    __slots__ = ("grid_size", "box_size", "G", "omega_m0", "H0", "_solver")

    def __init__(
        self,
        grid_size: int,
        box_size: float,
        G: float = 1.0,
        omega_m0: float | None = None,
        H0: float | None = None,
    ) -> None:
        self._solver: PoissonSolver = PoissonSolver(grid_size, box_size, G, omega_m0, H0)
        self.grid_size: int = grid_size
        self.box_size: float = box_size
        self.G: float = G
        self.omega_m0: float | None = omega_m0
        self.H0: float | None = H0

    def __call__(self, particles: ParticleSet, a: float = 1.0) -> FloatArray:
        if particles.dim != 3:
            raise ValueError(f"PMGravity supports 3-D only; got dim={particles.dim}")
        if particles.total_mass <= 0.0:
            return np.zeros_like(particles.positions)

        stencil = cic_stencil(particles.positions, self.grid_size, self.box_size)
        density = deposit_cic(stencil, particles.masses)
        mean_density = particles.total_mass / self.box_size**3
        delta = overdensity(density, mean_density)
        grad_phi = self._solver.acceleration(delta, mean_density, a)
        return interpolate_cic(stencil, grad_phi) / a

    @classmethod
    def from_cosmology(cls, grid_size: int, box_size: float, cosmology: "Cosmology") -> "PMGravity":
        return cls(grid_size, box_size, omega_m0=cosmology.Omega_m, H0=cosmology.H0)


def make_gravity(
    method: GravityMethod,
    *,
    G: float = 1.0,
    softening: float = 0.0,
    box_size: float | None = None,
    grid_size: int | None = None,
) -> DirectGravity | PMGravity:
    """Build a gravity solver by name.

    Args:
        method: ``"direct"`` or ``"pm"``.
        G: Gravitational constant.
        softening: Plummer softening (direct only).
        box_size: Periodic box side. Required for ``"pm"``; optional for
            ``"direct"`` (enables the minimum-image convention).
        grid_size: Mesh nodes per side (``"pm"`` only, required).
    """
    if method == "direct":
        return DirectGravity(softening=softening, G=G, box_size=box_size)
    if method == "pm":
        if grid_size is None or box_size is None:
            raise ValueError("method='pm' requires grid_size and box_size")
        return PMGravity(grid_size=grid_size, box_size=box_size, G=G)
    raise ValueError(f"unknown gravity method {method!r}; use 'direct' or 'pm'")
