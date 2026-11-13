"""Cloud-in-Cell (CIC) mass assignment and force interpolation.

Particles live in a periodic cubic box of side ``L``. The mesh has ``M`` nodes
per side (spacing ``h = L / M``), with node ``(i, j, k)`` at ``(i h, j h, k h)``.
Each particle is a cube of side ``h``: its mass is shared between the 8
surrounding nodes with trilinear weights.

Assignment and interpolation use the *same* :class:`CICStencil` (node indices
and weights). Using one kernel for both guarantees that a particle feels no
force from itself and that pair forces are equal and opposite, so total
momentum is conserved to round-off.

Everything is vectorised: the ``(N, 8)`` stencil is built with broadcasting,
mass is deposited with a single ``bincount``, and fields are gathered with a
single fancy-indexing operation. Cost is ``O(N)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.particles import FloatArray

IntArray = NDArray[np.int64]

# Offsets (di, dj, dk) of the 8 corner nodes of a cell, shape (8, 3).
_CORNERS: IntArray = np.array(
    [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64
)


@dataclass(frozen=True)
class CICStencil:
    """Node indices and weights linking ``N`` particles to a periodic mesh.

    Attributes:
        indices: ``(N, 8)`` flat (C-order) indices of the 8 nodes per particle.
        weights: ``(N, 8)`` CIC weights; each row sums to 1.
        grid_size: Nodes per side ``M``.
        box_size: Box side ``L``.
    """

    indices: IntArray
    weights: FloatArray
    grid_size: int
    box_size: float

    @property
    def cell_volume(self) -> float:
        return (self.box_size / self.grid_size) ** 3


def cic_stencil(positions: FloatArray, grid_size: int, box_size: float) -> CICStencil:
    """Build the CIC stencil for ``(N, 3)`` particle positions.

    Positions are wrapped into ``[0, L)`` (periodic), so particles that have
    drifted outside the box are handled without any special care.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3); got {positions.shape}")
    if grid_size < 2:
        raise ValueError(f"grid_size must be >= 2; got {grid_size}")
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError(f"box_size must be finite and > 0; got {box_size}")

    m = grid_size
    s = np.mod(positions, box_size) * (m / box_size)  # position in cell units, [0, M]
    i0 = np.floor(s).astype(np.int64)  # lower node of the containing cell
    frac = s - i0  # fractional offset within the cell, [0, 1)

    # Per-axis weights for the lower/upper node: (N, 3, 2).
    w_axis = np.stack((1.0 - frac, frac), axis=-1)

    # Broadcast over the 8 corners -> (N, 8). Product of the three axis weights.
    weights = (
        w_axis[:, 0, :][:, _CORNERS[:, 0]]
        * w_axis[:, 1, :][:, _CORNERS[:, 1]]
        * w_axis[:, 2, :][:, _CORNERS[:, 2]]
    )

    # Node indices with periodic wrap (i0 == M only when s rounds up to M).
    node = (i0[:, None, :] + _CORNERS[None, :, :]) % m  # (N, 8, 3)
    flat = (node[..., 0] * m + node[..., 1]) * m + node[..., 2]
    return CICStencil(flat, weights, m, box_size)


def deposit_cic(stencil: CICStencil, masses: FloatArray) -> FloatArray:
    """CIC mass assignment. Returns the mass *density* on the mesh, ``(M, M, M)``.

    The total mass is conserved exactly: ``density.sum() * h^3 == masses.sum()``.
    """
    if masses.shape != (stencil.indices.shape[0],):
        raise ValueError(
            f"masses must have shape ({stencil.indices.shape[0]},); got {masses.shape}"
        )
    m = stencil.grid_size
    mass_grid = np.bincount(
        stencil.indices.ravel(),
        weights=(stencil.weights * masses[:, None]).ravel(),
        minlength=m**3,
    )
    return (mass_grid / stencil.cell_volume).reshape(m, m, m)


def interpolate_cic(stencil: CICStencil, field: FloatArray) -> FloatArray:
    """CIC interpolation of a mesh field back to the particle positions.

    Args:
        stencil: The stencil used for the particles.
        field: Either a scalar field ``(M, M, M)`` -> returns ``(N,)``, or a
            vector field ``(C, M, M, M)`` -> returns ``(N, C)``.
    """
    m = stencil.grid_size
    if field.shape[-3:] != (m, m, m) or field.ndim not in (3, 4):
        raise ValueError(f"field must have shape (M, M, M) or (C, M, M, M) with M={m}; got {field.shape}")

    if field.ndim == 3:
        gathered = field.reshape(-1)[stencil.indices]  # (N, 8)
        return np.einsum("ij,ij->i", gathered, stencil.weights)

    flat = field.reshape(field.shape[0], -1)
    gathered = flat[:, stencil.indices]  # (C, N, 8)
    return np.einsum("cij,ij->ic", gathered, stencil.weights)
