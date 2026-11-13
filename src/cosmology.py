"""Flat Lambda-CDM background cosmology.

The expansion history follows the Friedmann equation for a flat universe
containing matter and a cosmological constant::

    H(a) = H0 * sqrt(Omega_m * a^-3 + Omega_Lambda),      Omega_m + Omega_Lambda = 1

Units are whatever the caller chooses for ``H0`` (an inverse time). Using
``H0 = 1`` makes time come out in units of the Hubble time ``1/H0``.

Two independent routes from time to scale factor are provided so each can
validate the other:

* **Closed form** (``age`` / ``scale_factor``): exact for flat Lambda-CDM.
* **Numerical** (``advance_scale_factor``): fourth-order Runge-Kutta on
  ``da/dt = a H(a)``. This is what the integrator uses each step, and it
  keeps working if the model is later extended (e.g. radiation, curvature)
  in a way that has no closed form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

FloatArray = NDArray[np.float64]

_FLATNESS_TOL: float = 1.0e-6
_LAMBDA_ZERO: float = 1.0e-14


@dataclass(frozen=True)
class Cosmology:
    """Flat Lambda-CDM expansion history.

    Attributes:
        H0: Hubble constant today (inverse time, in the caller's time unit).
        Omega_m: Matter density parameter today (> 0).
        Omega_Lambda: Dark-energy density parameter today (>= 0).

    Raises:
        ValueError: If a parameter is out of range or the universe is not flat
            (``Omega_m + Omega_Lambda`` must equal 1; curvature is not modelled).
    """

    H0: float
    Omega_m: float
    Omega_Lambda: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.H0) or self.H0 <= 0.0:
            raise ValueError(f"H0 must be finite and > 0; got {self.H0}")
        if not np.isfinite(self.Omega_m) or self.Omega_m <= 0.0:
            raise ValueError(f"Omega_m must be finite and > 0; got {self.Omega_m}")
        if not np.isfinite(self.Omega_Lambda) or self.Omega_Lambda < 0.0:
            raise ValueError(f"Omega_Lambda must be finite and >= 0; got {self.Omega_Lambda}")
        if abs(self.Omega_m + self.Omega_Lambda - 1.0) > _FLATNESS_TOL:
            raise ValueError(
                "only flat cosmologies are supported: Omega_m + Omega_Lambda must be 1 "
                f"(got {self.Omega_m + self.Omega_Lambda})"
            )

    # ------------------------------------------------------------------ #
    # Expansion rate
    # ------------------------------------------------------------------ #
    @overload
    def E(self, a: float) -> float: ...
    @overload
    def E(self, a: FloatArray) -> FloatArray: ...
    def E(self, a: float | FloatArray) -> float | FloatArray:
        """Dimensionless expansion rate ``E(a) = H(a) / H0``."""
        return np.sqrt(self.Omega_m * np.power(a, -3.0) + self.Omega_Lambda)  # type: ignore[no-any-return]

    @overload
    def hubble(self, a: float) -> float: ...
    @overload
    def hubble(self, a: FloatArray) -> FloatArray: ...
    def hubble(self, a: float | FloatArray) -> float | FloatArray:
        """Hubble parameter ``H(a) = H0 sqrt(Omega_m a^-3 + Omega_Lambda)``."""
        return self.H0 * self.E(a)  # type: ignore[no-any-return, call-overload]

    def adot(self, a: float) -> float:
        """Expansion rate of the scale factor, ``da/dt = a H(a)``.

        Written as ``H0 sqrt(Omega_m / a + Omega_Lambda a^2)``, which is
        algebraically identical and avoids a needless multiply by ``a``.
        """
        return float(self.H0 * np.sqrt(self.Omega_m / a + self.Omega_Lambda * a * a))

    # ------------------------------------------------------------------ #
    # Closed-form time <-> scale factor
    # ------------------------------------------------------------------ #
    @overload
    def age(self, a: float) -> float: ...
    @overload
    def age(self, a: FloatArray) -> FloatArray: ...
    def age(self, a: float | FloatArray) -> float | FloatArray:
        """Cosmic time ``t(a)`` since ``a = 0`` (exact, flat Lambda-CDM)."""
        a_arr = np.asarray(a, dtype=np.float64)
        if np.any(a_arr < 0.0):
            raise ValueError("scale factor must be >= 0")
        if self.Omega_Lambda < _LAMBDA_ZERO:  # Einstein-de Sitter
            t = 2.0 / (3.0 * self.H0 * np.sqrt(self.Omega_m)) * a_arr**1.5
        else:
            sqrt_l = np.sqrt(self.Omega_Lambda)
            x = np.sqrt(self.Omega_Lambda / self.Omega_m) * a_arr**1.5
            t = 2.0 / (3.0 * self.H0 * sqrt_l) * np.arcsinh(x)
        return t if isinstance(a, np.ndarray) else float(t)

    def time_between(self, a_start: float, a_end: float) -> float:
        """Elapsed cosmic time as the universe expands from ``a_start`` to ``a_end``."""
        return float(self.age(a_end) - self.age(a_start))

    @overload
    def scale_factor(self, t: float) -> float: ...
    @overload
    def scale_factor(self, t: FloatArray) -> FloatArray: ...
    def scale_factor(self, t: float | FloatArray) -> float | FloatArray:
        """Inverse of :meth:`age`: scale factor at cosmic time ``t`` (exact)."""
        t_arr = np.asarray(t, dtype=np.float64)
        if np.any(t_arr < 0.0):
            raise ValueError("cosmic time must be >= 0")
        if self.Omega_Lambda < _LAMBDA_ZERO:
            a = (1.5 * np.sqrt(self.Omega_m) * self.H0 * t_arr) ** (2.0 / 3.0)
        else:
            arg = 1.5 * np.sqrt(self.Omega_Lambda) * self.H0 * t_arr
            a = (self.Omega_m / self.Omega_Lambda) ** (1.0 / 3.0) * np.sinh(arg) ** (2.0 / 3.0)
        return a if isinstance(t, np.ndarray) else float(a)

    # ------------------------------------------------------------------ #
    # Numerical integration of da/dt = a H(a)
    # ------------------------------------------------------------------ #
    def advance_scale_factor(self, a: float, dt: float, n_substeps: int = 1) -> float:
        """Integrate ``da/dt = a H(a)`` forward by ``dt`` with classical RK4.

        Args:
            a: Scale factor at the start (> 0).
            dt: Time interval (may be negative to run backwards).
            n_substeps: RK4 sub-steps to split ``dt`` into.

        Returns:
            The scale factor after ``dt``.
        """
        if a <= 0.0:
            raise ValueError(f"scale factor must be > 0; got {a}")
        if n_substeps < 1:
            raise ValueError(f"n_substeps must be >= 1; got {n_substeps}")

        h = dt / n_substeps
        for _ in range(n_substeps):
            k1 = self.adot(a)
            k2 = self.adot(a + 0.5 * h * k1)
            k3 = self.adot(a + 0.5 * h * k2)
            k4 = self.adot(a + h * k3)
            a += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return a

    # ------------------------------------------------------------------ #
    # Linear growth factor
    # ------------------------------------------------------------------ #
    def _unnormalized_growth(self, a: float) -> float:
        """``E(a) * int_0^a da' / (a' E(a'))^3`` (Heath 1977), before normalising to D(1)=1."""
        integral, _ = quad(lambda x: 1.0 / (x * self.E(x)) ** 3, 0.0, a)
        return float(self.E(a) * integral)

    def growth_factor(self, a: float) -> float:
        """Linear growth factor ``D(a)``, normalised so ``D(1) = 1``.

        Solves the standard flat Lambda-CDM result (Heath 1977) by
        quadrature rather than the growth ODE directly, which is exact for
        this background and avoids an extra numerical integration.
        """
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError(f"scale factor must be finite and > 0; got {a}")
        return self._unnormalized_growth(a) / self._unnormalized_growth(1.0)

    def growth_factor_derivative(self, a: float) -> float:
        """``dD/da`` at scale factor ``a``, from the analytic derivative of :meth:`growth_factor`.

        Writing the unnormalised growth as ``d(a) = E(a) I(a)`` with
        ``I(a) = int_0^a da'/(a' E(a'))^3``, the derivative is
        ``d'(a) = E'(a) I(a) + 1/(a^3 E(a)^2)`` (using ``I'(a) = 1/(a E(a))^3``
        and ``dE/da = -1.5 Omega_m a^-4 / E`` from the Friedmann equation).
        """
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError(f"scale factor must be finite and > 0; got {a}")
        e_a = self.E(a)
        e_prime = -1.5 * self.Omega_m * a**-4 / e_a
        integral, _ = quad(lambda x: 1.0 / (x * self.E(x)) ** 3, 0.0, a)
        d_prime = e_prime * integral + 1.0 / (a**3 * e_a**2)
        return d_prime / self._unnormalized_growth(1.0)

    def omega_m_of_a(self, a: float) -> float:
        """``Omega_m(a) = Omega_m a^-3 / E(a)^2``."""
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError(f"scale factor must be finite and > 0; got {a}")
        return float(self.Omega_m * a**-3 / self.E(a) ** 2)

    def growth_rate_f(self, a: float) -> float:
        """Logarithmic growth rate ``f(a) = dlnD/dlna``, approximated as ``Omega_m(a)^0.55``.

        This fit (Peebles 1980; Lahav et al. 1991) is accurate to a few times
        ``1e-3`` for flat Lambda-CDM and much cheaper than differentiating
        :meth:`growth_factor`. For the exact value use
        ``a * growth_factor_derivative(a) / growth_factor(a)``.
        """
        return self.omega_m_of_a(a) ** 0.55
