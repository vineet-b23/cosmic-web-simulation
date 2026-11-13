"""Isotropic matter power spectrum estimator via FFT + radial binning."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def compute_power_spectrum(
    delta: FloatArray,
    box_size: float,
    n_particles: int,
    n_bins: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Isotropic power spectrum P(k) of a 3D overdensity grid delta(x).

    Args:
        delta: Overdensity grid, shape (M, M, M).
        box_size: Periodic box side length.
        n_particles: Total particle count; used for the Poisson shot noise
            P_shot = box_size**3 / n_particles, subtracted from every bin.
        n_bins: Number of log-spaced k bins (default: grid_size // 2).

    Returns:
        (k_bins, P_k): shot-noise-subtracted power, truncated to
        k <= k_Nyquist with non-positive (noise-dominated) bins removed.
    """
    m = delta.shape[0]
    volume = box_size**3
    cell = box_size / m

    delta_k = np.fft.rfftn(delta) * cell**3
    power_grid = np.abs(delta_k) ** 2 / volume

    k1d = 2.0 * np.pi * np.fft.fftfreq(m, d=cell)
    k1d_half = 2.0 * np.pi * np.fft.rfftfreq(m, d=cell)
    kx = k1d[:, None, None]
    ky = k1d[None, :, None]
    kz = k1d_half[None, None, :]
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)

    k_fundamental = 2.0 * np.pi / box_size
    k_nyquist = np.pi * m / box_size
    n_bins = m // 2 if n_bins is None else n_bins

    edges = np.linspace(k_fundamental, k_nyquist, n_bins + 1)
    flat_k = kmag.ravel()
    flat_p = power_grid.ravel()
    bin_idx = np.digitize(flat_k, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)

    counts = np.bincount(bin_idx[valid], minlength=n_bins)
    p_sum = np.bincount(bin_idx[valid], weights=flat_p[valid], minlength=n_bins)
    k_sum = np.bincount(bin_idx[valid], weights=flat_k[valid], minlength=n_bins)

    nonempty = counts > 0
    k_bins = np.full(n_bins, np.nan)
    P_k_raw = np.full(n_bins, np.nan)
    k_bins[nonempty] = k_sum[nonempty] / counts[nonempty]
    P_k_raw[nonempty] = p_sum[nonempty] / counts[nonempty]

    # 1. Subtract Poisson shot noise (P_shot = Volume / N_particles)
    P_shot = (box_size**3) / n_particles
    P_k_true = P_k_raw - P_shot

    # 2. Truncate at Nyquist frequency and strip negative noise values
    n_grid = delta.shape[0]
    k_nyq = np.pi * n_grid / box_size
    valid_mask = (k_bins > 0) & (k_bins <= k_nyq) & (P_k_true > 0)

    return k_bins[valid_mask], P_k_true[valid_mask]
