"""Kick-drift-kick (KDK) leapfrog time integrator, static or comoving.

The cosmology is an optional component. Without one, the integrator is the
plain symplectic leapfrog used in Phases 1 and 2. With a
:class:`~src.cosmology.Cosmology` it integrates comoving positions ``x`` and
peculiar velocities ``v`` (physical velocity relative to the Hubble flow)::

    dx/dt = v / a
    dv/dt = g - H(a) v

where ``g`` is the peculiar gravitational acceleration supplied by the force
function (already including any ``1/a`` factors and mean-density
subtraction). The physical position is ``a x``.

Static step of size ``dt``::

    v <- v + (dt/2) g(x)          # kick
    x <- x + dt v                 # drift
    v <- v + (dt/2) g(x)          # kick

Comoving step (``a0 -> a_half -> a1`` over ``dt/2`` each, via RK4)::

    v <- (a0 v + g I) / a_half    # kick with Hubble drag
    x <- x + (a_half v) int dt/a^2   # drift over the whole step
    v <- (a_half v + g I) / a1    # kick with Hubble drag

Combining ``dv/dt = g - H v`` with ``H = adot/a`` gives ``d(a v)/dt = a g``,
so the drag is applied *exactly* as the factor ``a_from / a_to`` (a free
particle's peculiar velocity falls as ``1/a`` for any step size) and gravity
enters through ``I = int a dt`` (trapezoid rule). Because ``u = a v`` only
changes through gravity, the drift uses ``dx/dt = u / a^2`` with the
integral ``int dt / a^2`` evaluated by Simpson's rule; for a free particle
this makes the drift accurate to Simpson's fourth order. The step is
time-centred and second-order accurate overall. The Hubble drag makes the system non-Hamiltonian, so the
energy-conservation guarantee of the static scheme does not carry over.

The force from the end of a step is reused as the first kick of the next, so
each step costs one force evaluation.
"""

from __future__ import annotations

from typing import Callable, cast, overload

import numpy as np

from src.cosmology import Cosmology
from src.particles import FloatArray, ParticleSet

StaticAccelerationFn = Callable[[ParticleSet], FloatArray]
ComovingAccelerationFn = Callable[[ParticleSet, float], FloatArray]
AccelerationFn = StaticAccelerationFn  # backwards-compatible alias (Phases 1-2)
StepCallback = Callable[[int, ParticleSet], None]


class LeapfrogKDK:
    """Kick-drift-kick leapfrog integrator with optional cosmological expansion.

    Static mode (no cosmology)::

        LeapfrogKDK(gravity)                      # gravity(particles) -> (N, D)

    Comoving mode::

        LeapfrogKDK(force, cosmology=cosmo, a_initial=0.1)
                                                  # force(particles, a) -> (N, D)

    Args:
        acceleration_fn: Force function. In comoving mode it also receives the
            scale factor at which the force is evaluated.
        cosmology: Expansion history. ``None`` (default) switches all
            cosmological effects off.
        a_initial: Scale factor at the start of integration (comoving mode).
        hubble_drag: If ``False`` (comoving mode only), the ``-H v`` term is
            dropped while the ``dx/dt = v/a`` drift is kept. Useful for
            isolating the two effects in tests.

    Note:
        The last acceleration is cached between steps. If you modify the
        particle state externally between steps, call :meth:`reset`.
    """

    __slots__ = ("_acceleration_fn", "_cosmology", "_hubble_drag", "_a", "_time", "_acc")

    @overload
    def __init__(self, acceleration_fn: StaticAccelerationFn) -> None: ...
    @overload
    def __init__(
        self,
        acceleration_fn: ComovingAccelerationFn,
        *,
        cosmology: Cosmology,
        a_initial: float,
        hubble_drag: bool = True,
    ) -> None: ...
    def __init__(
        self,
        acceleration_fn: Callable[..., FloatArray],
        *,
        cosmology: Cosmology | None = None,
        a_initial: float = 1.0,
        hubble_drag: bool = True,
    ) -> None:
        if cosmology is None:
            if a_initial != 1.0:
                raise ValueError("a_initial requires a cosmology; omit it for a static run")
        elif not np.isfinite(a_initial) or a_initial <= 0.0:
            raise ValueError(f"a_initial must be finite and > 0; got {a_initial}")

        self._acceleration_fn: Callable[..., FloatArray] = acceleration_fn
        self._cosmology: Cosmology | None = cosmology
        self._hubble_drag: bool = hubble_drag
        self._a: float = a_initial
        self._time: float = 0.0
        self._acc: FloatArray | None = None

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def cosmology(self) -> Cosmology | None:
        return self._cosmology

    @property
    def is_comoving(self) -> bool:
        return self._cosmology is not None

    @property
    def a(self) -> float:
        """Current scale factor (always 1 in static mode)."""
        return self._a

    @property
    def time(self) -> float:
        """Time elapsed since construction or the last :meth:`reset`."""
        return self._time

    def reset(self, a: float | None = None) -> None:
        """Discard the cached acceleration; optionally set a new scale factor."""
        self._acc = None
        if a is not None:
            if self._cosmology is None:
                raise ValueError("cannot set a scale factor without a cosmology")
            if not np.isfinite(a) or a <= 0.0:
                raise ValueError(f"a must be finite and > 0; got {a}")
            self._a = a

    # ------------------------------------------------------------------ #
    # Stepping
    # ------------------------------------------------------------------ #
    def step(self, particles: ParticleSet, dt: float) -> None:
        """Advance ``particles`` in place by one timestep ``dt``."""
        if not np.isfinite(dt) or dt == 0.0:
            raise ValueError(f"dt must be finite and non-zero; got {dt}")

        if self._cosmology is None:
            self._step_static(particles, dt)
        else:
            self._step_comoving(particles, dt, self._cosmology)
        self._time += dt

    def _step_static(self, particles: ParticleSet, dt: float) -> None:
        accel = cast(StaticAccelerationFn, self._acceleration_fn)
        if self._acc is None:
            self._acc = accel(particles)

        half_dt = 0.5 * dt
        particles.velocities += half_dt * self._acc  # kick
        particles.positions += dt * particles.velocities  # drift
        self._acc = accel(particles)  # new forces
        particles.velocities += half_dt * self._acc  # kick

    # --- src/integrator.py (LeapfrogKDK comoving step — verified correct, unchanged) ---

    def _step_comoving(self, particles: ParticleSet, dt: float, cosmology: Cosmology) -> None:
        accel = cast(ComovingAccelerationFn, self._acceleration_fn)
        a0 = self._a
        a_half = cosmology.advance_scale_factor(a0, 0.5 * dt)
        a1 = cosmology.advance_scale_factor(a_half, 0.5 * dt)

        if self._acc is None:
            self._acc = accel(particles, a0)

        self._kick(particles, self._acc, a0, a_half, 0.5 * dt)

        if self._hubble_drag:
            drift = a_half * dt / 6.0 * (1.0 / a0**2 + 4.0 / a_half**2 + 1.0 / a1**2)
        else:
            drift = dt / 6.0 * (1.0 / a0 + 4.0 / a_half + 1.0 / a1)
        particles.positions += drift * particles.velocities

        self._acc = accel(particles, a1)
        self._kick(particles, self._acc, a_half, a1, 0.5 * dt)
        self._a = a1

    def _kick(
        self,
        particles: ParticleSet,
        g: FloatArray,
        a_from: float,
        a_to: float,
        dt: float,
    ) -> None:
        if self._hubble_drag:
            impulse = 0.5 * dt * (a_from + a_to)
            particles.velocities *= a_from / a_to
            particles.velocities += (impulse / a_to) * g
        else:
            particles.velocities += dt * g

    def run(
        self,
        particles: ParticleSet,
        dt: float,
        n_steps: int,
        callback: StepCallback | None = None,
    ) -> None:
        """Take ``n_steps`` steps of size ``dt``.

        Args:
            callback: Optional ``callback(step_index, particles)`` invoked after
                each step, with ``step_index`` running from 1 to ``n_steps``.
                In comoving mode the current scale factor is :attr:`a`.
        """
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0; got {n_steps}")
        for k in range(1, n_steps + 1):
            self.step(particles, dt)
            if callback is not None:
                callback(k, particles)
