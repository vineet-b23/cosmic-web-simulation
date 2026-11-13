"""Particle state container for the collisionless N-body simulator.

All state is held in contiguous ``float64`` NumPy arrays so that force
calculations and integrators can operate on whole arrays at once.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class ParticleSet:
    """State vectors of ``N`` particles in ``D`` spatial dimensions.

    Attributes:
        positions: Array of shape ``(N, D)``.
        velocities: Array of shape ``(N, D)``.
        masses: Array of shape ``(N,)``.

    The constructor copies its inputs, so the caller's arrays are never
    aliased. Integrators are expected to update ``positions`` and
    ``velocities`` in place.
    """

    __slots__ = ("positions", "velocities", "masses")

    positions: FloatArray
    velocities: FloatArray
    masses: FloatArray

    def __init__(
        self,
        positions: ArrayLike,
        velocities: ArrayLike,
        masses: ArrayLike,
    ) -> None:
        pos = np.array(positions, dtype=np.float64, order="C")
        vel = np.array(velocities, dtype=np.float64, order="C")
        mass = np.array(masses, dtype=np.float64, order="C")

        if pos.ndim != 2:
            raise ValueError(f"positions must have shape (N, D); got {pos.shape}")
        if vel.shape != pos.shape:
            raise ValueError(
                f"velocities shape {vel.shape} must match positions shape {pos.shape}"
            )
        if mass.shape != (pos.shape[0],):
            raise ValueError(
                f"masses must have shape ({pos.shape[0]},); got {mass.shape}"
            )
        if not (np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(mass).all()):
            raise ValueError("particle state contains non-finite values")
        if (mass < 0.0).any():
            raise ValueError("masses must be non-negative")

        self.positions = pos
        self.velocities = vel
        self.masses = mass

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        """Number of particles."""
        return int(self.positions.shape[0])

    @property
    def dim(self) -> int:
        """Number of spatial dimensions."""
        return int(self.positions.shape[1])

    @property
    def total_mass(self) -> float:
        return float(self.masses.sum())

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    @property
    def center_of_mass(self) -> FloatArray:
        """Mass-weighted mean position, shape ``(D,)``."""
        return (self.masses @ self.positions) / self.total_mass

    @property
    def center_of_mass_velocity(self) -> FloatArray:
        """Mass-weighted mean velocity, shape ``(D,)``."""
        return (self.masses @ self.velocities) / self.total_mass

    @property
    def total_momentum(self) -> FloatArray:
        """Total linear momentum, shape ``(D,)``."""
        return self.masses @ self.velocities

    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy ``sum_i (1/2) m_i |v_i|^2``."""
        return float(0.5 * np.einsum("i,ij,ij->", self.masses, self.velocities, self.velocities))

    def angular_momentum(self) -> FloatArray:
        """Total angular momentum about the origin (3-D only), shape ``(3,)``."""
        if self.dim != 3:
            raise ValueError("angular momentum is only defined here for 3-D systems")
        return np.cross(self.positions, self.masses[:, None] * self.velocities).sum(axis=0)

    def copy(self) -> ParticleSet:
        """Return an independent deep copy."""
        return ParticleSet(self.positions, self.velocities, self.masses)

    def __repr__(self) -> str:
        return f"ParticleSet(n={self.n}, dim={self.dim}, total_mass={self.total_mass:g})"
