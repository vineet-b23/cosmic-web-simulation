"""
Add these two methods to the existing `Cosmology` class in src/cosmology.py.
They assume the class already exposes:
    self.Omega_m, self.Omega_L, self.H0
    self.H(a) -> H(a) = H0 * sqrt(Omega_m * a**-3 + Omega_L)

Insert the imports at the top of cosmology.py and paste the two methods
into the class body.
"""

import numpy as np
from scipy.integrate import quad


# --- add to top of src/cosmology.py -----------------------------------
# import numpy as np
# from scipy.integrate import quad


class _CosmologyGrowthMixin:
    """Mixin illustrating the two methods; merge these into `Cosmology`."""

    def _E(self, a: float) -> float:
        """Dimensionless Hubble parameter E(a) = H(a)/H0."""
        return np.sqrt(self.Omega_m * a ** -3 + self.Omega_L)

    def growth_factor(self, a: float) -> float:
        """
        Linear growth factor D(a) for flat LCDM, normalized to D(a=1)=1.

        Uses the standard growing-mode solution of the linear perturbation
        equation, written as the quadrature (e.g. Peebles 1980; Heath 1977):

            D(a) ~ E(a) * \\int_0^a da' / (a' E(a'))^3

        The integrand ~ a'^{-3/2} as a' -> 0, which is integrable, so the
        lower limit a'=0 is safe for scipy.integrate.quad.
        """
        def integrand(ap):
            return 1.0 / (ap * self._E(ap)) ** 3

        def unnormalized_D(aa):
            integral, _ = quad(integrand, 0.0, aa, limit=200)
            return 2.5 * self.Omega_m * self._E(aa) * integral

        return unnormalized_D(a) / unnormalized_D(1.0)

    def growth_rate_f(self, a: float) -> float:
        """
        Logarithmic growth rate f(a) = d ln D / d ln a ~= Omega_m(a)^0.55.

        Omega_m(a) = Omega_m * a^-3 / E(a)^2
        """
        Omega_m_a = self.Omega_m * a ** -3 / self._E(a) ** 2
        return Omega_m_a ** 0.55
