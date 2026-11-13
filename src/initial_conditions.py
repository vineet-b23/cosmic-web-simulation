"""Gaussian Random Field initial conditions and the Zel'dovich approximation.

Pipeline
--------
1. **Power spectrum** (:class:`PowerSpectrum`): shape ``P(k) propto k^ns T(k)^2``
   with a choice of transfer function, normalised so the field's RMS
   overdensity in an 8 Mpc/h top-hat sphere equals ``sigma8``.
2. **Gaussian random field** (:func:`generate_overdensity_k`): a real white
   noise field is drawn on the mesh and filtered in Fourier space by
   ``sqrt(P(k))``. Because the filter is real and a function of ``|k|`` only,
   it preserves the exact Hermitian symmetry of the real-valued white-noise
   field's FFT, so the resulting ``delta(k)`` is *exactly* the transform of a
   real field with no ad-hoc fix-ups at the self-conjugate (k=0, Nyquist)
   planes.
3. **Zel'dovich displacement** (:func:`displacement_field`): the Lagrangian
   displacement ``Psi(k) = -i k delta(k) / k^2`` (k = 0 set to zero), inverse
   transformed to real space.
4. **Particle initial conditions** (:func:`zeldovich_initial_conditions`):
   particles start on a uniform grid ``q`` (so ``Psi`` needs no interpolation)
   and are displaced and given peculiar velocities using the linear growth
   factor ``D(a)`` from :mod:`src.cosmology`::

       x = q + D(a_init) Psi(q)
       v = a_init H(a_init) dD/da(a_init) Psi(q)

Conventions
-----------
Positions are in comoving Mpc/h, box side ``box_size`` in Mpc/h, masses in
Msun/h and velocities in km/s. With these units ``H0 = 100`` exactly (by the
definition of little ``h``) and Newton's constant is the fixed number
:data:`G_COSMO`. Any :class:`~src.cosmology.Cosmology` passed in should use
``H0 = 100`` for the particle masses and velocities here to be consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from src.cosmology import Cosmology
from src.particles import FloatArray, ParticleSet

ComplexArray = NDArray[np.complex128]
TransferKind = Literal["power_law", "bbks"]

#: Newton's constant in Mpc/h, km/s, Msun/h units (the h's cancel, so this is
#: also G in plain Mpc, km/s, Msun units): G = 6.674e-11 m^3 kg^-1 s^-2.
G_COSMO: float = 4.30091e-9


# ---------------------------------------------------------------------- #
# Power spectrum
# ---------------------------------------------------------------------- #
def bbks_transfer_function(k: FloatArray, cosmology: Cosmology) -> FloatArray:
    """BBKS (Bardeen, Bond, Kaiser & Szalay 1986) cold dark matter transfer function.

    Args:
        k: Wavenumber in h/Mpc (may be an array; ``k <= 0`` gives ``T = 1``).
        cosmology: Used for the shape parameter ``Gamma = Omega_m h``.
    """
    h = cosmology.H0 / 100.0
    gamma = cosmology.Omega_m * h
    k = np.asarray(k, dtype=np.float64)
    q = np.divide(k, gamma, out=np.zeros_like(k), where=k > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.log1p(2.34 * q) / (2.34 * q)
        t = np.where(q > 0.0, t, 1.0)  # ln(1+x)/x -> 1 as x -> 0
    poly = 1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
    return t * poly**-0.25  # type: ignore[no-any-return]


def _power_shape(
    k: FloatArray, cosmology: Cosmology, ns: float, transfer: TransferKind, power_law_index: float
) -> FloatArray:
    """Unnormalised ``P(k)``: ``k^ns T(k)^2`` (BBKS) or ``k^power_law_index``."""
    k = np.asarray(k, dtype=np.float64)
    if transfer == "power_law":
        return np.where(k > 0.0, k**power_law_index, 0.0)
    if transfer == "bbks":
        t = bbks_transfer_function(k, cosmology)
        return np.where(k > 0.0, k**ns * t * t, 0.0)
    raise ValueError(f"unknown transfer {transfer!r}; use 'power_law' or 'bbks'")


def _top_hat_window(x: FloatArray) -> FloatArray:
    """Fourier transform of a top-hat sphere, ``W(x) = 3(sin x - x cos x)/x^3``."""
    x = np.asarray(x, dtype=np.float64)
    small = np.abs(x) < 1.0e-3
    safe = np.where(small, 1.0, x)  # avoid 0/0; the small-x branch is used there anyway
    w = 3.0 * (np.sin(safe) - safe * np.cos(safe)) / safe**3
    return np.where(small, 1.0 - x * x / 10.0, w)  # type: ignore[no-any-return]


def sigma_squared(shape_fn: Callable[[float], float], radius: float) -> float:
    """Top-hat variance ``sigma^2(R) = int_0^inf P(k) W(kR)^2 k^2 dk / (2 pi^2)``."""
    def integrand(k: float) -> float:
        return shape_fn(k) * _top_hat_window(k * radius) ** 2 * k * k

    value, _ = quad(integrand, 0.0, np.inf, limit=200)
    return value / (2.0 * np.pi**2)


class PowerSpectrum:
    """A CDM power spectrum, normalised to a target ``sigma8``.

    Args:
        cosmology: Sets the BBKS shape parameter ``Gamma = Omega_m h``.
        ns: Primordial spectral index (BBKS transfer only).
        sigma8: RMS overdensity in an 8 Mpc/h top-hat sphere today (``a = 1``,
            i.e. in terms of the present-day linear power spectrum).
        transfer: ``"bbks"`` for a realistic turnover, or ``"power_law"`` for
            the simple ``P(k) propto k^power_law_index`` approximation.
        power_law_index: Spectral index when ``transfer="power_law"``.
    """

    __slots__ = ("cosmology", "ns", "sigma8", "transfer", "power_law_index", "amplitude")

    def __init__(
        self,
        cosmology: Cosmology,
        ns: float = 1.0,
        sigma8: float = 0.8,
        transfer: TransferKind = "bbks",
        power_law_index: float = -1.0,
    ) -> None:
        self.cosmology: Cosmology = cosmology
        self.ns: float = ns
        self.sigma8: float = sigma8
        self.transfer: TransferKind = transfer
        self.power_law_index: float = power_law_index

        def shape(k: float) -> float:
            return float(_power_shape(np.array(k), cosmology, ns, transfer, power_law_index))

        sigma2_unit = sigma_squared(shape, radius=8.0)
        self.amplitude: float = sigma8 * sigma8 / sigma2_unit

    def __call__(self, k: FloatArray) -> FloatArray:
        """Normalised power spectrum ``P(k)``, in (Mpc/h)^3, for ``k`` in h/Mpc."""
        return self.amplitude * _power_shape(k, self.cosmology, self.ns, self.transfer, self.power_law_index)


def power_spectrum_bbks(
    k: FloatArray, cosmology: Cosmology, ns: float = -2.0, sigma8: float = 0.2
) -> FloatArray:
    return _cached_bbks_amplitude(cosmology, ns, sigma8) * _power_shape(k, cosmology, ns, "bbks", 0.0)


@lru_cache(maxsize=32)
def _cached_bbks_amplitude(cosmology: Cosmology, ns: float, sigma8: float) -> float:
    def shape(k: float) -> float:
        return float(_power_shape(np.array(k), cosmology, ns, "bbks", 0.0))

    return sigma8 * sigma8 / sigma_squared(shape, radius=8.0)



# ---------------------------------------------------------------------- #
# Gaussian random field and Zel'dovich displacement
# ---------------------------------------------------------------------- #
def _wavevectors(n_grid: int, box_size: float) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """``(kx, ky, kz, |k|)`` on the ``rfftn`` half-grid, broadcastable to ``(M, M, M//2+1)``."""
    k1d = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=box_size / n_grid)
    k1d_half = 2.0 * np.pi * np.fft.rfftfreq(n_grid, d=box_size / n_grid)
    kx = k1d[:, None, None]
    ky = k1d[None, :, None]
    kz = k1d_half[None, None, :]
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    return kx, ky, kz, kmag


def generate_overdensity_k(
    n_grid: int,
    box_size: float,
    power_spectrum: Callable[[FloatArray], FloatArray],
    seed: int | None = None,
) -> ComplexArray:
    """Draw ``delta(k)`` on the ``rfftn`` half-grid for a periodic box.

    A real white-noise field is generated on the mesh and its (real) FFT is
    scaled by ``sqrt(P(k) M^3 / V)``. Since the scaling is real and isotropic,
    it cannot break the Hermitian symmetry the real-valued white noise's FFT
    already has, so ``irfftn`` of the result is exactly real -- no special
    handling of the ``k = 0`` or Nyquist planes is needed.

    Returns:
        Complex array, shape ``(M, M, M//2+1)``, matching :func:`numpy.fft.rfftn`.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n_grid, n_grid, n_grid))
    white_k = np.fft.rfftn(white)  # Var(white_k[k]) = n_grid**3 exactly (white is real)

    _, _, _, kmag = _wavevectors(n_grid, box_size)
    volume = box_size**3
    amplitude = np.zeros_like(kmag)
    nonzero = kmag > 0.0
    amplitude[nonzero] = np.sqrt(power_spectrum(kmag[nonzero]) * n_grid**3 / volume)
    return amplitude * white_k  # type: ignore[no-any-return]


def displacement_field(delta_k: ComplexArray, n_grid: int, box_size: float) -> FloatArray:
    """Zel'dovich displacement ``Psi(x) = IFFT{-i k delta(k) / k^2}``, shape ``(3, M, M, M)``.

    The ``k = 0`` mode (net displacement / mean motion) is set to zero. For
    even ``n_grid`` the Nyquist frequency of each axis is also zeroed in the
    ``-i k`` factor: ``numpy.fft.fftfreq`` stores that bin as the single value
    ``-Nyquist`` rather than its true +/-Nyquist double value, so a linear-in-k
    (gradient-type) operator built from it would not satisfy the reflection
    symmetry a real field requires (the same fix used for the gradient in
    :class:`src.poisson.PoissonSolver`).
    """
    kx, ky, kz, kmag = _wavevectors(n_grid, box_size)
    inv_k2 = np.zeros_like(kmag)
    nonzero = kmag > 0.0
    inv_k2[nonzero] = 1.0 / (kmag[nonzero] ** 2)

    def zero_nyquist(k1d: FloatArray) -> FloatArray:
        # The Nyquist bin sits at index n_grid // 2 on *both* the full-length
        # fftfreq axis and the (n_grid // 2 + 1)-length rfftfreq axis (there
        # it is the last entry) -- never at len(k1d) // 2 of the array itself,
        # which differs between the two and silently zeros the wrong bin.
        k1d = k1d.copy()
        if n_grid % 2 == 0:
            k1d[n_grid // 2] = 0.0
        return k1d

    k1d = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=box_size / n_grid)
    k1d_half = 2.0 * np.pi * np.fft.rfftfreq(n_grid, d=box_size / n_grid)
    kx_grad = zero_nyquist(k1d)[:, None, None]
    ky_grad = zero_nyquist(k1d)[None, :, None]
    kz_grad = zero_nyquist(k1d_half)[None, None, :]

    shape = (n_grid, n_grid, n_grid)
    psi = np.empty((3,) + shape)
    for axis, k_comp in enumerate((kx_grad, ky_grad, kz_grad)):
        psi_hat = -1j * k_comp * inv_k2 * delta_k
        psi[axis] = np.fft.irfftn(psi_hat, s=shape)
    return psi


@dataclass(frozen=True)
class ZeldovichResult:
    """Output of :func:`zeldovich_initial_conditions`."""

    particles: ParticleSet
    overdensity: FloatArray  # (M, M, M) real-space delta at a_init (linear field, undisplaced grid)
    displacement: FloatArray  # (3, M, M, M) comoving Zel'dovich displacement Psi
    lagrangian_positions: FloatArray  # (N, 3) the undisplaced grid q
    growth_factor: float  # D(a_init)
    growth_rate: float  # dD/da at a_init


def zeldovich_initial_conditions(
    n_grid: int,
    box_size: float,
    cosmology: Cosmology,
    a_init: float,
    power_spectrum: PowerSpectrum,
    seed: int | None = None,
) -> ZeldovichResult:
    """Build Zel'dovich particle initial conditions on an ``n_grid**3`` uniform lattice.

    Args:
        n_grid: Particles per side (``N = n_grid**3`` total).
        box_size: Periodic box side, comoving Mpc/h.
        cosmology: Should use ``H0 = 100`` (see module docstring); its
            ``Omega_m`` sets both the BBKS shape (via ``power_spectrum``) and
            the mean particle mass through the critical density.
        a_init: Initial scale factor (e.g. 0.02 for z ~ 49).
        power_spectrum: Already normalised (see :class:`PowerSpectrum`).
        seed: Random seed for the Gaussian field.
    """
    cell = box_size / n_grid
    idx = np.indices((n_grid, n_grid, n_grid), dtype=np.float64)  # (3, M, M, M)
    q = ((idx + 0.5) * cell).reshape(3, -1).T  # cell-centred Lagrangian grid, (N, 3)

    delta_k = generate_overdensity_k(n_grid, box_size, power_spectrum, seed)
    delta_x = np.fft.irfftn(delta_k, s=(n_grid, n_grid, n_grid))
    psi = displacement_field(delta_k, n_grid, box_size)
    psi_at_q = psi.reshape(3, -1).T  # q is exactly the grid, so no interpolation is needed

    growth = cosmology.growth_factor(a_init)
    growth_rate = cosmology.growth_factor_derivative(a_init)
    hubble = cosmology.hubble(a_init)

    positions = np.mod(q + growth * psi_at_q, box_size)
    # src.integrator.LeapfrogKDK integrates dx/dt = v / a with v the *physical*
    # peculiar velocity v = a * dx_comoving/dt (see its module docstring).
    # With x_comoving = q + D(a) Psi(q), dx_comoving/dt = Psi dD/dt
    # = Psi (dD/da)(da/dt) = Psi (dD/da)(a H(a)), so the velocity handed to
    # the integrator needs a SECOND factor of a: v = a^2 H(a) (dD/da) Psi.
    # (A single factor of a, as in v = a H(a) dD/da Psi, is the correct
    # *comoving* rate dx/dt itself, not the a*dx/dt convention this
    # integrator expects; using it here would make the initial velocities
    # too large by a factor of 1/a_init.)
    velocities = a_init**2 * hubble * growth_rate * psi_at_q

    rho_crit0 = 3.0 * cosmology.H0**2 / (8.0 * np.pi * G_COSMO)
    total_mass = cosmology.Omega_m * rho_crit0 * box_size**3
    masses = np.full(n_grid**3, total_mass / n_grid**3)

    particles = ParticleSet(positions, velocities, masses)
    return ZeldovichResult(particles, delta_x, psi, q, growth, growth_rate)


def zeldovich_initial_conditions_arrays(
    n_grid: int,
    box_size: float,
    cosmology: Cosmology,
    a_init: float,
    power_spectrum: Callable[[FloatArray], FloatArray],
    seed: int | None = None,
    max_delta_init: float = 0.05,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    cell = box_size / n_grid
    q = ((np.indices((n_grid, n_grid, n_grid), dtype=np.float64) + 0.5) * cell).reshape(3, -1).T

    delta_k = generate_overdensity_k(n_grid, box_size, power_spectrum, seed)
    delta_x = np.fft.irfftn(delta_k, s=(n_grid, n_grid, n_grid))
    psi = displacement_field(delta_k, n_grid, box_size).reshape(3, -1).T

    growth = cosmology.growth_factor(a_init)
    max_abs = float(np.max(np.abs(growth * delta_x)))
    if max_abs > 0.0:
        scale = max_delta_init / max_abs
        delta_k = delta_k * scale
        psi = psi * scale

    growth_rate_f = cosmology.growth_rate_f(a_init)
    hubble = cosmology.hubble(a_init)

    x = np.mod(q + growth * psi, box_size)
    v = a_init * hubble * growth_rate_f * growth * psi

    rho_crit0 = 3.0 * cosmology.H0**2 / (8.0 * np.pi * G_COSMO)
    masses = np.full(n_grid**3, cosmology.Omega_m * rho_crit0 * box_size**3 / n_grid**3)

    return x, v, masses