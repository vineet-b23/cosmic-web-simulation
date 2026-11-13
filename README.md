# Cosmic Web Formation Simulator — Project Document

**Document type:** Consolidated project charter, scientific requirements, technical design, numerical-methods record, experiment register and validation plan  
**Project status:** Working research/educational implementation  
**Primary entry point:** `main.py`  
**Primary model:** Collisionless matter in a periodic, spatially flat matter + cosmological-constant background

---

## 1. Project charter

### 1.1 Problem statement

The large-scale matter distribution in the Universe is commonly described as a cosmic web: dense nodes and clusters connected by filaments, sheets or walls, and separated by underdense voids. In the gravitational-instability picture, this structure develops as initially small density perturbations grow and collapse anisotropically under gravity while the background Universe expands.

The purpose of this project is to implement a compact numerical model that demonstrates this process. Particles begin close to a uniform Lagrangian lattice, receive small correlated displacements and peculiar velocities, and evolve under self-gravity in an expanding periodic volume. The resulting density field and matter power spectrum are recorded at selected redshifts.

The project is intended to make the numerical pipeline inspectable: cosmological background, initial-condition generation, force calculation, time integration, density measurement and statistical analysis are separated into modules and small experiments.

### 1.2 Objectives

1. Implement a collisionless N-body model in comoving coordinates.
2. Represent the background expansion with a spatially flat matter + cosmological-constant model.
3. Generate correlated initial perturbations from a Fourier-space power spectrum and use the Zel'dovich approximation to initialise particles.
4. Implement periodic Particle-Mesh gravity using CIC assignment, an FFT Poisson solve and CIC force interpolation.
5. Provide direct and Ewald-summed periodic force solvers for small-system checks and PM accuracy comparisons.
6. Integrate static and cosmological particle systems with a Kick-Drift-Kick scheme.
7. Measure density slices and the isotropic matter power spectrum at selected epochs.
8. Provide visual outputs that distinguish actual stored simulation states from presentation-only interpolation.
9. Document numerical assumptions, known limitations and validation evidence.

### 1.3 Scope

**Implemented in the current project**

- Collisionless particle dynamics under Newtonian self-gravity.
- Flat matter + cosmological-constant background expansion.
- Comoving coordinates and periodic cubic volume.
- Gaussian random-field initial conditions with a BBKS-shaped power spectrum.
- Zel'dovich particle displacements and peculiar velocities.
- Direct softened gravity, Ewald-summed periodic reference gravity and Particle-Mesh gravity.
- CIC particle-to-mesh mass assignment and mesh-to-particle force interpolation.
- Periodic FFT Poisson solver.
- Static and comoving KDK integration.
- Density-slice and power-spectrum measurement.
- Six standalone experiments.
- 3-D particle animations and cosmic-web density visualisations.

**Not implemented in the current project snapshot**

- Hydrodynamics or gas pressure forces.
- Radiative cooling/heating or radiation transport.
- Star formation, stellar evolution or feedback.
- Black-hole dynamics or feedback.
- Magnetic fields or plasma physics.
- A completed halo finder or halo catalogue pipeline.
- A completed general-purpose density-field analysis package beyond the snapshot measurement and power-spectrum estimator.
- A completed interactive dashboard.
- A populated automated test suite.

The original project structure contains placeholder files for some future modules. Their presence should not be interpreted as implemented functionality.

### 1.4 Intended use

The code is suitable for:

- learning and demonstrating the computational steps of a cosmological PM simulation;
- inspecting how density perturbations evolve in a simple expanding background;
- comparing direct summation and mesh-based force calculations;
- studying numerical effects such as force resolution, shot noise and mesh sampling;
- producing reproducible teaching, portfolio and exploratory research figures.

It is not currently suitable for precision cosmological parameter inference, publication-grade predictions, or quantitative comparison with galaxy surveys without substantial additional calibration and validation.

---

## 2. Scientific requirements

### 2.1 Physical model

The simulation treats matter as collisionless particles interacting through gravity. The cosmological background is spatially flat and contains pressureless matter and a cosmological constant:

\[
\Omega_{m,0}+\Omega_{\Lambda,0}=1,
\]

\[
H(a)=H_0\sqrt{\Omega_{m,0}a^{-3}+\Omega_{\Lambda,0}}.
\]

Radiation and spatial curvature are not included in the current background model. The density field is represented in a periodic cubic comoving box.

### 2.2 Scientific requirements register

| ID | Requirement | Current implementation / status |
|---|---|---|
| SCI-01 | Represent a flat matter + Lambda expansion history | Implemented in `src/cosmology.py` |
| SCI-02 | Evolve collisionless particles under gravity | Implemented in `src/particles.py`, `src/gravity.py` and `src/integrator.py` |
| SCI-03 | Use a periodic cubic domain for the cosmological run | Implemented by wrapped particle positions and periodic mesh operations |
| SCI-04 | Generate a correlated initial density field | Implemented using a Gaussian random field filtered by the configured power spectrum |
| SCI-05 | Initialise particle displacements and velocities using the Zel'dovich approximation | Implemented in `src/initial_conditions.py` |
| SCI-06 | Account for cosmological expansion during integration | Implemented in the comoving KDK path |
| SCI-07 | Compute gravitational forces with a scalable mesh method | Implemented using CIC + FFT Poisson + CIC interpolation |
| SCI-08 | Provide a periodic reference force calculation | Implemented with the Ewald-summed solver |
| SCI-09 | Record density information at multiple redshifts | Implemented in `main.py`; 2-D central slices are stored |
| SCI-10 | Quantify clustering statistically | Implemented with the isotropic `P(k)` estimator |
| SCI-11 | Provide visual inspection of particle and density evolution | Implemented through plots, 3-D particle animations and cosmic-web videos |
| SCI-12 | Identify dark-matter halos and export catalogues | Not implemented |
| SCI-13 | Provide a general interactive dashboard | Not implemented |
| SCI-14 | Run a formal automated regression suite | Not implemented in the supplied project snapshot |

### 2.3 Initial-condition requirements

The initial density field should:

- be generated from a seeded Gaussian random field;
- have a Fourier-space spectrum with the configured BBKS transfer-function shape;
- use a zero-mean mode convention for the density contrast;
- generate a displacement field with the zero mode removed;
- place particles on a uniform, cell-centred Lagrangian lattice before displacement;
- use the background linear growth history to initialise peculiar velocities;
- apply the configured maximum initial density-contrast cap in the main pipeline.

The main pipeline uses `seed = 42`, `n_s = -2.0`, `sigma8 = 0.2` and `max_delta_init = 0.05`. These are project-run settings, not a claim that the run uses a best-fit observational cosmology. The explicit initial-amplitude cap means that the final initial realisation is not simply an unconstrained sigma-eight normalisation.

### 2.4 Numerical and output requirements

- Particle mass must be conserved by CIC deposition.
- The mesh density contrast must have zero mean up to floating-point tolerance.
- The zero Fourier mode of the periodic Poisson solve must not generate a force.
- Even-grid Nyquist derivative modes must be handled to preserve a real-valued gradient field.
- Snapshot redshifts must be ordered from high redshift to low redshift.
- Density slices and power-spectrum arrays must be associated with the same snapshot redshift.
- The saved snapshot file must be sufficient to render the project's 2-D density-slice outputs without rerunning the simulation.

The current main pipeline includes a runtime invariant that checks the mean of `1 + delta` after CIC deposition and raises an error if it differs from one by more than `1e-10`.

---

## 3. Default run specification

The main configuration is defined by `PipelineConfig` in `main.py`.

| Quantity | Value | Notes |
|---|---:|---|
| Particles per dimension | 32 | Uniform Lagrangian lattice |
| Total particle count | 32,768 | `32**3` equal-mass particles |
| Mesh nodes per dimension | 64 | PM grid |
| Total mesh cells/nodes | 262,144 | `64**3` |
| Box side length | 100 Mpc/h | Periodic, comoving |
| Initial redshift | 49 | `a_init = 0.02` |
| Matter density | 0.3 | `Omega_m` |
| Lambda density | 0.7 | `Omega_Lambda = 1 - Omega_m` |
| Hubble constant | 100 km/s/(Mpc/h) | Little-`h` unit convention |
| BBKS spectral index | -2.0 | Project configuration |
| `sigma8` input | 0.2 | BBKS amplitude input before the initial cap |
| Maximum initial contrast | 0.05 | `max_delta_init` |
| Random seed | 42 | Deterministic initial realisation |
| Steps per e-fold | 80 | Main pipeline default |
| Minimum steps per snapshot segment | 15 | Main pipeline default |
| Snapshot redshifts | 49, 20, 10, 5, 3, 2, 1, 0.5, 0 | Nine stored epochs |

The mean particle mass is derived from the background matter density using the critical density and the comoving box volume. All particles in the main initial-condition generator have equal mass.

### 3.1 Unit conventions

| Quantity | Unit / convention |
|---|---|
| Comoving coordinate / box length | Mpc/h |
| Particle mass | Msun/h |
| Velocity | km/s |
| Wavenumber | h/Mpc |
| Power spectrum | (Mpc/h)^3 |
| `H0` | 100 km/s/(Mpc/h) in the little-`h` convention |
| Scale factor | Dimensionless, `a = 1/(1+z)` |

The particle coordinates stored by the integrator are comoving. Physical position is \(\mathbf{r}=a\mathbf{x}\). The velocity variable is the peculiar velocity

\[
\mathbf{v}=a\frac{d\mathbf{x}}{dt}.
\]

The array-based Zel'dovich initial-condition function used by `main.py` assigns

\[
\mathbf{v}_{\rm init}=a_{\rm init}H(a_{\rm init})f(a_{\rm init})D(a_{\rm init})\,\boldsymbol{\Psi}.
\]

---

## 4. Technical architecture

### 4.1 High-level data flow

```text
Flat Lambda-CDM background
            |
            v
Gaussian random field <--- BBKS-shaped P(k)
            |
            v
Fourier-space displacement field
            |
            v
Zel'dovich particle positions + velocities
            |
            v
Periodic particle state
            |
            v
CIC mass assignment to PM mesh
            |
            v
Density contrast delta(x)
            |
            v
FFT Poisson solve ---> mesh acceleration
            |                  |
            |                  v
            |            CIC interpolation
            |                  |
            +------------------+
                       |
                       v
             Comoving KDK integration
                       |
                       v
              Redshift checkpoints
                       |
            +----------+-----------+
            |                      |
            v                      v
     Density slices             Matter P(k)
            |                      |
            +----------+-----------+
                       |
                       v
              Figures / video output
```

### 4.2 Module responsibilities

| Path | Responsibility |
|---|---|
| `main.py` | Main run configuration, cosmology construction, initial conditions, PM integration, snapshot measurement, saving and plotting |
| `src/cosmology.py` | Flat matter + Lambda expansion rate, cosmic time, scale-factor evolution and linear growth functions |
| `src/particles.py` | Particle positions, velocities and masses, with system diagnostics such as centre of mass and kinetic energy |
| `src/initial_conditions.py` | BBKS transfer function, power-spectrum amplitude, Gaussian random field, displacement field and Zel'dovich particle initialisation |
| `src/mesh.py` | Periodic CIC stencil, mass deposition and interpolation of mesh fields to particles |
| `src/poisson.py` | Density-contrast calculation and periodic FFT solution of Poisson's equation |
| `src/gravity.py` | Softened direct gravity, Ewald periodic reference gravity and PM gravity wrapper |
| `src/integrator.py` | Static Newtonian KDK and comoving KDK integration |
| `analysis/power_spectrum.py` | Isotropic FFT-based power-spectrum estimator |
| `visualization/particles_3d.py` | 3-D particle animation and montage rendering, including speed or local-density colouring |
| `visualization/make_cosmic_web_video.py` | Video rendering from saved redshift and 2-D slice arrays; snapshot-only and cinematic modes |
| `experiments/` | Standalone numerical experiments and demonstrations |

The project snapshot used for this document contains placeholder files for `src/simulation.py`, `analysis/density_field.py`, `analysis/halo_finder.py` and `visualization/dashboard.py`. They are not part of the implemented execution path described above.

### 4.3 Main execution sequence

1. Parse command-line options into `PipelineConfig`.
2. Construct a flat `Cosmology` instance with `H0 = 100`, `Omega_m = 0.3` and `Omega_Lambda = 0.7`.
3. Construct the BBKS-shaped spectrum function using `n_s = -2.0` and `sigma8 = 0.2`.
4. Generate Zel'dovich initial positions, peculiar velocities and equal particle masses on a `32^3` lattice at `a = 0.02`.
5. Construct `PMGravity.from_cosmology` for the `64^3` mesh and the `100 Mpc/h` box.
6. Advance the particle system to each requested snapshot scale factor using the comoving KDK integrator.
7. Deposit particles onto the mesh and compute `delta` at each checkpoint.
8. Assert that the mean of `1 + delta` is one to a tolerance of `1e-10`.
9. Extract the central plane `delta[:, :, M//2]`, store `1 + delta`, and calculate the power spectrum.
10. Save the snapshots and generate the density montage and power-spectrum plot.

---

## 5. Numerical methods

### 5.1 Flat cosmological background

The background model follows the flat matter + Lambda Friedmann equation:

\[
E(a)=\frac{H(a)}{H_0}=\sqrt{\Omega_{m,0}a^{-3}+\Omega_{\Lambda,0}},
\qquad \dot a=aH(a).
\]

The `Cosmology` class provides the expansion rate, cosmic age, time intervals, conversion between cosmic time and scale factor, the linear growth factor and its derivative. The closed-form age/scale-factor relation is specific to the implemented flat matter + Lambda model; the integrator advances the scale factor numerically with fourth-order Runge-Kutta substeps.

The linear growth factor is normalised such that \(D(1)=1\). The growth rate used by the main array-based initial-condition function is approximated by

\[
f(a)\simeq \Omega_m(a)^{0.55}.
\]

The code also exposes the derivative \(dD/da\) through quadrature-based evaluation.

### 5.2 BBKS spectrum and Gaussian random field

The BBKS transfer function is evaluated from the comoving wavenumber and the shape parameter based on \(\Gamma=\Omega_m h\). The configured spectral shape is

\[
P(k)=A k^{n_s}T(k)^2,
\]

where the amplitude \(A\) is obtained by normalising the top-hat variance at radius `8 Mpc/h` to the configured `sigma8` input.

A real Gaussian white-noise field is generated on the mesh. Its real-to-complex FFT is filtered by \(\sqrt{P(k)}\), with the volume/grid normalisation required by the discrete Fourier convention. Filtering by a real isotropic kernel preserves the Hermitian structure inherited from the FFT of the real field.

The main pipeline applies a final amplitude rescaling such that the largest absolute initial density contrast is no greater than `max_delta_init = 0.05`. This gives a deliberately near-uniform high-redshift starting field.

### 5.3 Zel'dovich approximation

The displacement field is calculated in Fourier space:

\[
\boldsymbol{\Psi}(\mathbf{k})=-i\frac{\mathbf{k}}{k^2}\delta(\mathbf{k}),
\]

with the zero mode set to zero. Derivative-type Nyquist modes are also suppressed for even mesh sizes to preserve the symmetry required for a real-valued field.

Particles begin on a cell-centred Lagrangian lattice \(\mathbf{q}\), then move to

\[
\mathbf{x}(a_{\rm init})=\mathbf{q}+D(a_{\rm init})\boldsymbol{\Psi}(\mathbf{q}).
\]

The peculiar velocity is set using the growth rate:

\[
\mathbf{v}(a_{\rm init})=a_{\rm init}H(a_{\rm init})f(a_{\rm init})D(a_{\rm init})\boldsymbol{\Psi}(\mathbf{q}).
\]

The initial lattice and displacement field have matching mesh sampling, so the implementation reads the displacement at the corresponding lattice locations without an additional interpolation step.

### 5.4 CIC mass assignment and interpolation

For a mesh spacing \(h=L/M\), each particle contributes to the eight surrounding mesh nodes with trilinear weights. The CIC stencil stores both the flat node indices and the weights. The same stencil is used for mass deposition and force interpolation.

The mesh density is formed by depositing the weighted particle masses and dividing by the cell volume. The implementation uses a vectorised `bincount` operation for deposition and indexed weighted gathers for interpolation.

A key invariant is mass conservation:

\[
\sum_{\rm cells}\rho_{ijk}\,h^3=\sum_{\rm particles}m_p.
\]

Using a matched assignment/interpolation kernel is also intended to reduce self-force and preserve pairwise momentum balance to floating-point accuracy in the discrete mesh coupling.

### 5.5 Density contrast

The mesh overdensity is

\[
\delta(\mathbf{x})=\frac{\rho(\mathbf{x})-\bar\rho}{\bar\rho}.
\]

The main pipeline checks the mean density contrast through

\[
\left\langle 1+\delta\right\rangle=1.
\]

This check is performed at every recorded snapshot and raises a runtime error if the difference exceeds `1e-10`.

### 5.6 Periodic FFT Poisson solver

The cosmological source equation used by the PM solver is

\[
\nabla^2\Phi=\frac{3}{2}\Omega_{m,0}H_0^2\frac{\delta}{a}.
\]

In Fourier space, for non-zero modes,

\[
\Phi(\mathbf{k})=-\frac{3}{2}\Omega_{m,0}H_0^2\frac{\delta(\mathbf{k})}{a k^2}.
\]

The solver uses `rfftn` to store the real-field half spectrum and `irfftn` to transform the potential and each acceleration component back to real space. The zero mode is set to zero because the uniform background does not create a net periodic force. For an even-sized grid, Nyquist derivative modes are zeroed before evaluating the spectral gradient.

The Poisson solver returns the comoving-coordinate gradient force. The PM wrapper applies the additional `1/a` factor required for the peculiar acceleration convention used by the comoving integrator.

The FFT solve scales approximately as \(O(M^3\log M)\).

### 5.7 Gravity solvers

**Direct summation.** The direct solver calculates softened Newtonian pair forces:

\[
\mathbf{a}_i=-G\sum_{j\ne i}m_j
\frac{\mathbf{x}_i-\mathbf{x}_j}
{\left(|\mathbf{x}_i-\mathbf{x}_j|^2+\epsilon^2\right)^{3/2}}.
\]

It is vectorised with NumPy broadcasting and processes target particles in blocks when necessary to bound memory use. Its computational scaling is \(O(N^2)\). It can use isolated boundaries or a minimum-image periodic convention; the minimum-image sum is not the full infinite periodic sum.

**Ewald periodic reference.** `EwaldGravity` splits the periodic interaction into real-space and reciprocal-space contributions. It is used by the PM comparison experiment as the periodic reference calculation. As with numerical Ewald implementations generally, the real- and reciprocal-space sums are evaluated with finite cutoffs; reported comparisons should be interpreted with the solver's chosen truncation parameters in mind.

**Particle-Mesh solver.** `PMGravity` deposits density with CIC, solves Poisson's equation on the periodic mesh, differentiates the potential spectrally, and interpolates the acceleration to particle locations. Its cost is approximately \(O(N+M^3\log M)\). The method does not resolve the particle force accurately below a few mesh cells.

### 5.8 KDK time integration

In static mode, the integrator performs the standard Kick-Drift-Kick update:

\[
\mathbf{v}_{n+1/2}=\mathbf{v}_n+\frac{\Delta t}{2}\mathbf{g}(\mathbf{x}_n),
\]

\[
\mathbf{x}_{n+1}=\mathbf{x}_n+\Delta t\,\mathbf{v}_{n+1/2},
\]

\[
\mathbf{v}_{n+1}=\mathbf{v}_{n+1/2}+\frac{\Delta t}{2}\mathbf{g}(\mathbf{x}_{n+1}).
\]

In comoving mode, it evolves

\[
\dot{\mathbf{x}}=\frac{\mathbf{v}}{a},
\qquad
\dot{\mathbf{v}}=\mathbf{g}-H(a)\mathbf{v}.
\]

Using \(H=\dot a/a\), the drag equation can be written as

\[
\frac{d(a\mathbf{v})}{dt}=a\mathbf{g}.
\]

The code applies the drag part through the exact scale-factor ratio between kick endpoints and evaluates the force impulse with a time-centred trapezoidal approximation. The drift integral is evaluated with Simpson's rule. The comoving method is time-centred and second-order overall. Since the expanding-background equations include Hubble drag, the conservation properties of the static Hamiltonian leapfrog do not apply unchanged.

The last force evaluation is cached and reused at the beginning of the next step, avoiding an additional force calculation per step.

### 5.9 Matter power spectrum

The power-spectrum estimator operates on a three-dimensional density-contrast mesh. It computes the real FFT, applies the cell-volume and box-volume normalisation, bins Fourier modes into isotropic wavenumber shells and subtracts the Poisson particle shot-noise level:

\[
P_{\rm shot}=\frac{V}{N_{\rm particles}}.
\]

Only bins satisfying both \(k\leq k_{\rm Nyquist}\) and positive shot-noise-subtracted power are returned. Empty bins and non-positive residual bins are discarded.

The estimator is a low-resolution diagnostic. The current implementation does not deconvolve the CIC assignment window, and shot-noise subtraction does not remove every source of finite-resolution or discreteness error.

---

## 6. Experiment register

The experiments are standalone scripts. Their configurations are intentionally smaller and/or different from the main cosmological run when that makes a numerical test easier to interpret.

### EXP-001 — Two-body circular orbit

**Script:** `experiments/exp_001_twobody.py`  
**Objective:** Check the static direct-gravity force and KDK integrator on a controlled orbit with a known circular configuration.

**Default setup:** two equal-mass particles (`m1 = m2 = 1`), unit separation, gravitational constant `G = 1`, Plummer softening `1e-3`, 20 orbits and 200 steps per orbit.

**Measurements:** particle trajectories, pair separation, total energy and the z-component of angular momentum.

**Interpretation:** bounded energy error and stable orbital geometry are the relevant numerical indicators. This is a controlled integrator demonstration, not a cosmological expansion test.

**Outputs:** `results/exp_001_twobody/orbit.png` and `trajectory.npz`.

### EXP-002 — Cold cluster collapse

**Script:** `experiments/exp_002_cluster.py`  
**Objective:** Explore nonlinear collapse of a self-gravitating particle sphere and inspect its energetic and structural evolution.

**Default setup:** 1,000 equal-mass particles, initial sphere radius 1, total mass 1, Plummer softening 0.05, initial virial ratio `2K/|U| = 0.05`, timestep 0.004 and an integration duration of five free-fall times.

**Measurements:** total energy, virial ratio, half-mass radius and particle positions/velocities at recorded frames. The script reports the time and radius at maximum compression, a late-time virial-ratio summary and maximum relative energy deviation.

**Outputs:** snapshot data, diagnostic figure, montage and GIF/MP4 animation under `results/exp_002_cluster/`.

**Interpretation:** the virial ratio and half-mass radius describe collapse and subsequent evolution. The reported late-time ratio should be read as a diagnostic of this finite, softened, finite-duration calculation rather than proof of a general equilibrium solution.

### EXP-003 — Expansion-only control

**Script:** `experiments/exp_003_expansion.py`  
**Objective:** Isolate the coordinate and background-expansion behaviour of the comoving integrator without gravitational forces.

**Default setup:** two zero-mass particles at rest with comoving separation 1, starting at `a = 0.1` and advancing to `a = 1` in 200 steps. The background uses `H0 = 1`, `Omega_m = 0.3` and `Omega_Lambda = 0.7` for dimensionless time.

**Comparison:** an expanding run is compared with a static run. In the expanding case the comoving separation should remain constant while the physical separation follows `r = a d`. The measured scale factor is compared against the analytic flat matter + Lambda scale-factor relation.

**Outputs:** `results/exp_003_expansion/expansion.npz` and `expansion.png`.

### EXP-004 — PM versus direct gravity

**Script:** `experiments/exp_004_pm_vs_direct.py`  
**Objective:** Measure the PM method's computational scaling and characterise its force error relative to periodic and non-periodic direct-sum references.

**Default setup:** unit box, `64^3` PM mesh, timing particle counts of 1,000, 5,000, 10,000 and 50,000, and a separate 1,000-particle force-accuracy sample. The direct reference softening is initially half a mesh cell and is also swept across 0.25, 0.5, 1 and 2 cells.

**Performance measurements:** total force-evaluation timings for PM and direct summation, plus timings for CIC deposition, FFT solve and CIC interpolation.

**Accuracy measurements:** per-particle relative force errors against Ewald-periodic, minimum-image direct and isolated direct calculations. The isolated calculation is intentionally labelled non-comparable as a periodic reference. The minimum-image result is also compared with the Ewald result to show the difference between the two periodic treatments.

**Outputs:** `time_vs_n.png`, `force_error_hist.png` and `benchmark.npz` under `results/exp_004_pm_vs_direct/`.

**Interpretation:** the benchmark illustrates the PM scaling advantage and its loss of small-scale force detail. A broad force-error tail is an expected limitation of a mesh method at insufficient force resolution; quote actual measured percentiles from the benchmark output rather than using a generic accuracy number.

### EXP-005 — Cosmic-web density evolution figure

**Script:** `experiments/exp_005_cosmic_web.py`  
**Objective:** Produce a standalone density-slice demonstration from Zel'dovich initial conditions and PM evolution.

**Default setup:** 32 particles per side, 64 mesh nodes per side, 100 Mpc/h box, initial scale factor 0.02, seed 42, and 60 integration steps per snapshot segment. It plots snapshots at `z = 49, 10, 3, 1, 0`.

**Output:** `results/cosmic_web_evolution.png`.

**Important distinction:** this standalone experiment has its own five-snapshot setup and should not be confused with the nine-snapshot `main.py` pipeline. It also constructs `PMGravity` through the non-cosmological `G`-based constructor, whereas `main.py` uses `PMGravity.from_cosmology`. As currently written, EXP-005 is therefore a separate legacy visual demonstration, not a numerically identical rerun or validation of the main pipeline. If it is intended to reproduce the main cosmological force normalisation, update it to use `PMGravity.from_cosmology(PM_GRID_SIZE, BOX_SIZE, cosmo)` and rerun the experiment before presenting its output as a main-pipeline comparison.

### EXP-006 — Matter power-spectrum evolution figure

**Script:** `experiments/exp_006_power_spectrum.py`  
**Objective:** Generate a standalone time sequence of isotropic matter power spectra.

**Default setup:** 32 particles per side, 64 mesh nodes per side, 100 Mpc/h box, seed 42 and 60 steps per snapshot interval. The plotted redshifts are `z = 10, 3, 1, 0`.

**Measurements:** shot-noise-subtracted `P(k)` at each selected redshift, using `N = 32^3` in the shot-noise estimate.

**Output:** `results/power_spectrum_evolution.png`.

---

## 7. Validation strategy and current evidence

### 7.1 Validation approach

The code uses small, interpretable experiments to check components independently before relying on the full cosmological run.

| Area | Check | Evidence / implementation |
|---|---|---|
| Particle and mesh coupling | Total mass is conserved during CIC deposit | `deposit_cic` uses weights that sum to one; `main.py` checks the mean of `1 + delta` |
| Density contrast | Mean density contrast is zero to numerical tolerance | Runtime assertion in `measure_snapshot` |
| Static integration | Two-body orbit, energy and angular momentum histories | EXP-001 |
| Collapse dynamics | Energy, virial ratio and half-mass-radius diagnostics | EXP-002 |
| Cosmological coordinates | Constant comoving separation and physical `r = a d` in a force-free control | EXP-003 |
| PM computational cost | Timing comparison and PM-stage timings | EXP-004 |
| PM force accuracy | Relative force-error distribution against Ewald periodic reference | EXP-004 |
| Power-spectrum pipeline | Snapshot-wise shell-binned spectra with shot-noise and Nyquist filtering | EXP-006 and main pipeline |
| End-to-end evolution | Density slices and `P(k)` at nine redshifts | `main.py` |

### 7.2 Invariants enforced by code

- CIC weights are constructed as products of per-axis linear weights; each particle's eight weights sum to one within floating-point precision.
- `deposit_cic` deposits mass density by dividing the deposited cell mass by the mesh-cell volume.
- The Poisson solver explicitly sets the zero-mode inverse wavenumber to zero.
- The Poisson solver removes derivative Nyquist modes for even mesh sizes.
- The main snapshot measurement checks `abs(mean(1 + delta) - 1) <= 1e-10`.
- The snapshot writer stores one redshift value for each indexed slice and power-spectrum pair.

### 7.3 Current validation boundary

The project snapshot supplied for documentation contains `tests/.gitkeep` but no populated automated test modules. The experiment scripts provide manual and numerical diagnostics, but there is not yet a formal CI-backed regression suite with tolerances and expected reference values.

The PM-versus-direct experiment is particularly important: a speed advantage does not imply force equivalence at all scales. The PM force is mesh-smoothed and the current implementation exhibits a broad error distribution relative to the Ewald periodic reference. The existing README notes a heavy error tail; exact median and percentile values should be taken from the actual `exp_004` run output or `benchmark.npz` for any report.

A stronger validation release should add:

1. automated tests for CIC mass conservation and interpolation consistency;
2. convergence checks for the Poisson solve and force field as mesh resolution changes;
3. timestep convergence for static and comoving KDK trajectories;
4. growth-factor and linear-regime comparisons against analytic expectations;
5. power-spectrum convergence tests varying particle count, mesh size and box size;
6. comparison against an established cosmological simulation or published reference on matched initial conditions.

---

## 8. Snapshot and result formats

### 8.1 Main `snapshots.npz`

The main pipeline writes a compressed NumPy archive. Its current schema is:

| Key | Shape | Contents |
|---|---|---|
| `redshifts` | `(n_snapshots,)` | Snapshot redshifts in descending order |
| `slice_0` ... `slice_(n-1)` | `(M, M)` each | Stored 2-D central-plane fields of `1 + delta` |
| `k_0` ... `k_(n-1)` | Variable length | Wavenumber-bin centres for the corresponding snapshot |
| `power_0` ... `power_(n-1)` | Variable length | Positive shot-noise-subtracted power values corresponding to `k_i` |

The `k_i` and `power_i` arrays can be empty or have different lengths across snapshots because the estimator discards empty bins and bins with non-positive shot-noise-subtracted power.

The cosmic-web video loader supports this `redshifts + slice_i` format as well as stacked density arrays and redshift-keyed arrays. The stored slices are already 2-D; the video script should not try to reconstruct a 3-D density cube from them.

### 8.2 Figure outputs

- `cosmic_web_evolution.png`: a five-panel display montage at selected redshifts. The plotting routine applies a light Gaussian blur to the displayed fields to reduce single-particle pixel artefacts.
- `power_spectrum_evolution.png`: log-log power-spectrum curves. Snapshots with no positive bins after shot-noise subtraction are omitted from the plot.

### 8.3 Video outputs

The video generator reads saved snapshots and never executes the simulation.

**Scientific evolution video**

- displays actual stored snapshots with hard cuts;
- uses a fixed global colour range across the stored snapshots;
- does not interpolate between snapshots;
- does not apply image smoothing by default (an explicit optional smoothing parameter exists);
- reports the snapshot redshift and scale factor.

**Cinematic video**

- smooths and upsamples the stored slices;
- applies a restrained unsharp mask;
- cross-fades between adjacent snapshots using smoothstep easing;
- paces transitions in scale factor rather than uniform redshift;
- applies a slow zoom and presentation overlays.

The frames between snapshots in the cinematic video are visual interpolation, not new numerical states. They must not be used for quantitative measurements or described as additional simulation output.

---

## 9. Known limitations and interpretation

### 9.1 Resolution and particle sampling

The default run has 32,768 particles and 262,144 mesh nodes, an average of one particle per eight mesh cells. This is a sparse particle-to-mesh sampling regime. Discreteness, shot noise and the initial lattice can affect small-scale density structure. A single stored mesh plane can show parity or lattice-sampling artefacts, particularly at early times.

The current density outputs are therefore best interpreted at large scales. The effective resolution is constrained by both particle spacing and mesh spacing, not just by the nominal `64^3` mesh size.

### 9.2 Initial-spectrum calibration

The run uses `n_s = -2.0`, `sigma8 = 0.2` and a maximum initial density contrast of 0.05. The configured spectral slope is a deliberate project setting, not a claim to use a precision observational primordial spectrum. The explicit maximum-contrast rescaling changes the final amplitude of the generated realisation. Quantitative claims about the observed amplitude of clustering are not supported by this setup.

### 9.3 Force accuracy

PM force accuracy is limited below a few mesh cells. CIC assignment/interpolation and spectral differentiation define a mesh force law that is not identical to a softened pairwise Newtonian force at small separations. The PM-versus-Ewald comparison should be used to describe the measured error distribution; no universal force-accuracy percentage should be inferred from one configuration.

### 9.4 Power-spectrum accuracy

The estimator subtracts Poisson shot noise and truncates at the mesh Nyquist frequency. It does not currently correct the CIC mass-assignment window. Finite particle number, finite box size, mesh assignment and mode counting all affect the recovered spectrum. The power spectrum is a diagnostic for this low-resolution demonstration, not a precision cosmological estimator.

### 9.5 Model completeness

The code is gravity-only and does not model baryonic processes. It cannot predict gas morphology, star formation, galaxies, emitted light or observable survey catalogues. It is also not a substitute for mature, extensively validated cosmological N-body codes.

### 9.6 Validation coverage

The experiments provide targeted checks, but the absence of an automated test suite and systematic convergence studies limits the strength of any numerical-accuracy claim. Machine-specific runtime measurements should not be presented as universal performance guarantees.

---

## 10. Reproducibility and repository practice

### 10.1 Reproducing the main run

Use the documented default configuration and seed:

```bash
python main.py
```

To create a separate realisation without overwriting the default result directory:

```bash
python main.py --seed 7 --output-dir results/run_seed7
```

To change resolution and integration density:

```bash
python main.py --n-particles 48 --grid 96 --steps-per-efold 120 \
  --output-dir results/run_48_96
```

Record the command line, seed, particle count, mesh size, box size, cosmological parameters and software versions alongside any reported result.

### 10.2 Version control recommendations

Track source code, documentation, small figures and lightweight examples. Treat `snapshots.npz`, full MP4/GIF outputs, temporary video files, Python caches and virtual environments as generated artefacts unless there is a specific reason to publish them. Large raw data can be published as a separately versioned release asset or archival dataset with a clear description and checksum.

Before public release:

- populate the repository's `LICENSE` file with the licence actually selected by the project owner;
- fill in `CITATION.cff` if formal citation metadata is desired;
- verify that the committed README commands match the current CLI;
- include a small representative figure in `docs/figures/` or the README;
- confirm that no generated data, credentials or machine-specific paths are committed.

---

## 11. Future work

Potential future improvements are grouped by scientific value rather than presentation polish.

### 11.1 Numerical validation

- Build a `pytest` suite for the mesh, Poisson solver, growth functions, force solvers and integrator.
- Add timestep and mesh-resolution convergence studies.
- Quantify the PM force error as a function of separation, softening and mesh size.
- Validate linear growth and the early-time power spectrum against analytic expectations.
- Compare matched initial conditions with a well-established cosmological N-body solver.

### 11.2 Analysis

- Add a documented density-field analysis API.
- Implement and validate a halo finder and halo catalogue format.
- Add two-point correlation functions and additional power-spectrum diagnostics.
- Add CIC window correction and a documented treatment of shot noise and aliasing.

### 11.3 Performance and scale

- Profile the full pipeline before optimisation.
- Explore compiled or GPU-accelerated kernels only after establishing numerical regression tests.
- Increase particle and mesh resolution and quantify convergence rather than simply increasing defaults.

### 11.4 Visualisation and usability

- Add an interactive dashboard after the underlying analysis API is stable.
- Preserve explicit labelling of simulated snapshots versus interpolated presentation frames.
- Add lightweight, reproducible examples for the README and project portfolio.

---

## 12. Release-level summary

The current implementation is a compact cosmological PM N-body demonstration with a meaningful numerical-methods core: seeded Gaussian initial conditions, Zel'dovich displacements, flat matter + Lambda expansion, periodic CIC/FFT PM gravity, comoving KDK integration and snapshot-wise density and power-spectrum measurement. Direct and Ewald-summed solvers and the six experiment scripts provide diagnostic reference points.

Its scientific value is strongest as an inspectable computational experiment and a platform for further numerical validation. Its low resolution, configured initial spectrum, PM force limitations, incomplete analysis modules and lack of a formal automated test suite must remain explicit in any academic or portfolio presentation.
