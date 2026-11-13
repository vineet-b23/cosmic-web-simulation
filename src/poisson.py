"""FFT-based Poisson solver on a periodic cubic mesh.

Pipeline (all real-to-complex transforms via ``numpy.fft.rfftn`` / ``irfftn``)::

    delta(x) = (rho(x) - rho_bar) / rho_bar
    delta(k) = F{delta}
    Phi(k)   = -(4 pi G rho_bar / a) delta(k) / k^2          (k = 0 mode set to 0)
    g(k)     = -i k Phi(k)
    g(x)     = F^-1{g(k)}

Conventions
-----------
* ``rho_bar`` is the mean mass density per unit *comoving* volume, so it stays
  constant as the universe expands, and ``a`` is the scale factor. With
  ``a = 1`` this is ordinary periodic Newtonian gravity.
* :meth:`PoissonSolver.acceleration` returns exactly ``g = -grad_x Phi`` as in
  the equations above, a gradient with respect to *comoving* coordinates. It is
  not yet the peculiar acceleration used by the comoving integrator, which
  needs one further ``1/a`` (peculiar acceleration ``= -grad_x Phi / a``);
  :class:`src.gravity.PMGravity` applies it.
* The ``k = 0`` mode is the mean of ``delta``, which is zero by construction.
  It is set to zero explicitly (a uniform background exerts no force in a
  periodic box) rather than dividing by zero.
* For even ``M`` the Nyquist mode is dropped from the gradient. Its derivative
  is not representable for a real field and would otherwise inject noise.
* ``rfftn`` stores only half the spectrum (last axis has ``M // 2 + 1``
  entries), halving memory and time compared with a complex FFT.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.particles import FloatArray

ComplexArray = NDArray[np.complex128]


def overdensity(density: FloatArray, mean_density: float | None = None) -> FloatArray:
    """Overdensity ``delta = (rho - rho_bar) / rho_bar``.

    Args:
        density: Mass density on the mesh.
        mean_density: ``rho_bar``. Defaults to the mesh mean, which makes
            ``delta`` sum to zero exactly.
    """
    rho_bar = float(density.mean()) if mean_density is None else mean_density
    if not np.isfinite(rho_bar) or rho_bar <= 0.0:
        raise ValueError(f"mean density must be finite and > 0; got {rho_bar}")
    return density / rho_bar - 1.0  # type: ignore[no-any-return]


class PoissonSolver:
    """Spectral solver for a periodic ``M x M x M`` mesh of side ``L``.

    Wavenumbers and the ``1/k^2`` factor are computed once at construction,
    so each solve costs one forward and (for the acceleration) three inverse
    real FFTs: ``O(M^3 log M)``.

    Args:
        grid_size: Nodes per side ``M``.
        box_size: Box side ``L``.
        G: Gravitational constant (flat, non-cosmological mode).
        omega_m0: If given together with ``H0``, the Poisson source uses the
            exact cosmological amplitude ``(3/2) Omega_m0 H0^2`` (Peebles 1980)
            instead of ``4 pi G rho_bar``, so the result no longer depends on
            ``G`` or the empirical mean density matching the cosmology by
            construction. Use this for the comoving PM solver; leave both
            ``None`` for flat, non-expanding Newtonian gravity (``a = 1``).
        H0: Hubble constant, same units as ``Cosmology.H0`` (e.g. 100 for
            comoving Mpc/h, km/s units). Required together with ``omega_m0``.
    """

    __slots__ = ("grid_size", "box_size", "G", "omega_m0", "H0", "_k_grad", "_inv_k2")

    def __init__(
        self,
        grid_size: int,
        box_size: float,
        G: float = 1.0,
        omega_m0: float | None = None,
        H0: float | None = None,
    ) -> None:
        if grid_size < 2:
            raise ValueError(f"grid_size must be >= 2; got {grid_size}")
        if not np.isfinite(box_size) or box_size <= 0.0:
            raise ValueError(f"box_size must be finite and > 0; got {box_size}")
        if (omega_m0 is None) != (H0 is None):
            raise ValueError("omega_m0 and H0 must be given together (or both omitted)")
        self.grid_size: int = grid_size
        self.box_size: float = box_size
        self.G: float = G
        self.omega_m0: float | None = omega_m0
        self.H0: float | None = H0

        m = grid_size
        spacing = box_size / m
        k_full = 2.0 * np.pi * np.fft.fftfreq(m, d=spacing)  # (M,)   for the two full axes
        k_half = 2.0 * np.pi * np.fft.rfftfreq(m, d=spacing)  # (M//2+1,) for the last axis

        # Broadcastable views: kx varies along axis 0, ky along 1, kz (half) along 2.
        kx = k_full[:, None, None]
        ky = k_full[None, :, None]
        kz = k_half[None, None, :]

        k2 = kx * kx + ky * ky + kz * kz  # (M, M, M//2+1)
        inv_k2 = np.zeros_like(k2)
        np.divide(1.0, k2, out=inv_k2, where=k2 > 0.0)  # k = 0 mode stays 0
        self._inv_k2: FloatArray = inv_k2

        # Wavenumbers used for the gradient: zero the Nyquist mode (even M).
        kg_full = k_full.copy()
        kg_half = k_half.copy()
        if m % 2 == 0:
            kg_full[m // 2] = 0.0
            kg_half[-1] = 0.0
        self._k_grad: tuple[FloatArray, FloatArray, FloatArray] = (
            kg_full[:, None, None],
            kg_full[None, :, None],
            kg_half[None, None, :],
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _check(self, delta: FloatArray, mean_density: float, a: float) -> None:
        m = self.grid_size
        if delta.shape != (m, m, m):
            raise ValueError(f"delta must have shape ({m}, {m}, {m}); got {delta.shape}")
        if not np.isfinite(mean_density) or mean_density <= 0.0:
            raise ValueError(f"mean_density must be finite and > 0; got {mean_density}")
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError(f"scale factor a must be finite and > 0; got {a}")

    def _potential_k(self, delta: FloatArray, mean_density: float, a: float) -> ComplexArray:
        """``Phi(k) = -amplitude(a) delta(k) / k^2`` on the half spectrum.

        ``amplitude(a) = (3/2) Omega_m0 H0^2 / a`` in cosmological mode
        (``nabla^2 Phi = (3/2) Omega_m0 H0^2 delta / a``), else the flat
        ``4 pi G rho_bar / a``.
        """
        delta_k = np.fft.rfftn(delta)
        if self.omega_m0 is not None:
            prefactor = -1.5 * self.omega_m0 * self.H0 * self.H0 / a
        else:
            prefactor = -4.0 * np.pi * self.G * mean_density / a
        return prefactor * self._inv_k2 * delta_k  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def potential(self, delta: FloatArray, mean_density: float, a: float = 1.0) -> FloatArray:
        """Gravitational potential ``Phi(x)`` on the mesh, shape ``(M, M, M)``."""
        self._check(delta, mean_density, a)
        m = self.grid_size
        return np.fft.irfftn(self._potential_k(delta, mean_density, a), s=(m, m, m))

    def acceleration(self, delta: FloatArray, mean_density: float, a: float = 1.0) -> FloatArray:
        """Spectral gradient ``g = -grad_x Phi`` on the mesh, shape ``(3, M, M, M)``.

        Computed as ``g(k) = -i k Phi(k)`` followed by an inverse real FFT per
        component (see the module docstring for the ``1/a`` convention).
        """
        self._check(delta, mean_density, a)
        m = self.grid_size
        phi_k = self._potential_k(delta, mean_density, a)

        g = np.empty((3, m, m, m))
        for axis, k in enumerate(self._k_grad):  # fixed loop over the 3 components
            g[axis] = np.fft.irfftn(-1j * k * phi_k, s=(m, m, m))
        return g