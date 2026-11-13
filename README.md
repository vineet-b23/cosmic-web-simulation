# Cosmic Web Formation Simulator

A compact, NumPy-based cosmological N-body simulator for exploring how gravitational instability turns a nearly uniform early matter distribution into a large-scale cosmic web of **sheets, filaments, dense nodes and underdense voids**.

The code evolves collisionless matter in a periodic, expanding, spatially flat \(\Lambda\)CDM box. Its main simulation path uses a **Particle-Mesh (PM) gravity solver**, a Fourier-space Poisson solver and a comoving Kick-Drift-Kick (KDK) integrator. Separate experiments compare numerical components against simpler or higher-fidelity reference calculations.

> **Scope:** this is a compact educational and research-oriented numerical experiment, not a production cosmological simulation. It is gravity-only. Baryonic hydrodynamics, gas cooling, star formation, stellar or black-hole feedback, and radiation transport are not modelled.

## Features

- **Particle-Mesh gravity.** Periodic Cloud-in-Cell (CIC) mass assignment and force interpolation, with an FFT Poisson solver using `numpy.fft.rfftn` / `irfftn`. In cosmological mode, the potential satisfies
  \[
  \nabla^2\Phi = \frac{3}{2}\,\Omega_{m,0} H_0^2\,\frac{\delta}{a}.
  \]
  An Ewald-summed periodic direct solver is provided as a force-accuracy reference.
- **Comoving KDK integration.** A time-centred Kick-Drift-Kick scheme evolves comoving positions and peculiar velocities. Hubble drag is handled analytically through the scale-factor ratio. Without a cosmology object, the integrator runs as a standard Newtonian leapfrog.
- **Zel'dovich initial conditions.** A Gaussian random field is generated from a BBKS-shaped power spectrum, converted to a displacement field in Fourier space, and mapped onto particles using the linear growth factor \(D(a)\) and growth rate \(f(a)\).
- **Power-spectrum analysis.** An isotropic matter power-spectrum estimator uses FFTs and radial shell binning, subtracts Poisson shot noise, and discards non-positive noise-dominated bins and modes above the mesh Nyquist frequency.
- **Reference gravity solvers.** A softened direct-summation solver provides an \(O(N^2)\) baseline; an Ewald-summed periodic solver provides a periodic reference for the PM force comparison.
- **Visualisation.** Three-dimensional particle animations (`.gif` / `.mp4`), density-slice figures, a power-spectrum evolution plot, and two cosmic-web video styles.

## Model and default configuration

The main pipeline uses a spatially flat matter-plus-cosmological-constant background:

\[
H(a) = H_0\sqrt{\Omega_{m,0}a^{-3}+\Omega_{\Lambda,0}},
\qquad \Omega_{m,0}+\Omega_{\Lambda,0}=1.
\]

The defaults in `main.py` are:

| Parameter | Default | Meaning |
|---|---:|---|
| Particles per side | `32` | A cubic lattice of \(32^3 = 32,768\) equal-mass particles |
| PM mesh | `64` | A periodic \(64^3\) mesh |
| Box size | `100 Mpc/h` | Comoving side length of the cubic volume |
| \(H_0\) | `100 km/s/(Mpc/h)` | Hubble constant in the project's little-\(h\) unit convention |
| \(\Omega_{m,0}\) | `0.3` | Present-day matter density parameter |
| \(\Omega_{\Lambda,0}\) | `0.7` | Present-day cosmological-constant density parameter |
| Initial redshift | `49` | Initial scale factor \(a_{\rm init}=0.02\) |
| Spectral index | `-2.0` | Configured slope used by this demonstration run |
| `sigma8` input | `0.2` | Amplitude input to the BBKS-shaped spectrum |
| Initial contrast cap | `0.05` | Rescales the initial realization so \(\max |\delta_{\rm init}|\leq 0.05\) |
| Random seed | `42` | Reproducible Gaussian-field realization |
| Steps per e-fold | `80` | Main integration step-density setting |

The main pipeline records snapshots at:

`z = 49, 20, 10, 5, 3, 2, 1, 0.5, 0`.

The `sigma8` parameter is used to set the amplitude of the input BBKS-shaped spectrum, but the main pipeline also applies the explicit `max_delta_init` rescaling. The resulting initial realization is therefore intended as a smooth, near-uniform starting point, not as a calibrated prediction of observed clustering.

## Numerical model

### Initial conditions

A real white-noise field is transformed to Fourier space and filtered by the square root of the target spectrum. The configured spectral shape is

\[
P(k) \propto k^{n_s}T_{\rm BBKS}(k)^2,
\]

where \(T_{\rm BBKS}\) is the BBKS cold-dark-matter transfer function. The displacement field is obtained from

\[
\boldsymbol{\Psi}(\mathbf{k}) = -\,i\,\frac{\mathbf{k}}{k^2}\,\delta(\mathbf{k}),
\]

with the zero mode removed. Particles are displaced from a cell-centred uniform Lagrangian lattice and assigned Zel'dovich peculiar velocities using the growth history.

### Particle-Mesh gravity

For each force evaluation, the PM solver:

1. Wraps particle positions into the periodic box.
2. Builds the eight-node CIC stencil for every particle.
3. Deposits particle mass onto the mesh and forms the density contrast \(\delta=(\rho-\bar\rho)/\bar\rho\).
4. Solves the periodic Poisson equation in Fourier space.
5. Computes the spectral gradient of the potential, with the zero mode and even-grid Nyquist derivative modes handled explicitly.
6. Interpolates the mesh acceleration back to particle positions using the same CIC stencil.

The PM method has approximate cost \(O(N+M^3\log M)\), where \(N\) is the particle count and \(M^3\) is the mesh size. Its mesh representation smooths the force on scales smaller than a few mesh cells.

### Comoving integration

The cosmological integrator evolves comoving position \(\mathbf{x}\) and peculiar velocity \(\mathbf{v}=a\,d\mathbf{x}/dt\):

\[
\frac{d\mathbf{x}}{dt}=\frac{\mathbf{v}}{a},
\qquad
\frac{d\mathbf{v}}{dt}=\mathbf{g}-H(a)\mathbf{v}.
\]

The Hubble-drag part of the velocity update is applied using the scale-factor ratio. The drift integral is evaluated with Simpson's rule, while the force impulse uses a time-centred quadrature. The cosmological step is second-order time-centred; because expansion introduces Hubble drag, the static-mode energy-conservation guarantee does not carry over unchanged.

## Installation

Python 3.10 or newer is required.

```bash
git clone <your-repository-url>
cd "Cosmic Web Formation Simulator"

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat

pip install -r requirements.txt
```

### Dependencies

| Package | Used for |
|---|---|
| `numpy` | Particle arrays, mesh operations, FFTs and numerical kernels |
| `scipy` | Quadrature, special functions for Ewald summation, and image processing |
| `matplotlib` | Scientific figures and particle animations |

An FFmpeg executable on `PATH` is needed for MP4 rendering. The cosmic-web video generator can also use the optional `imageio-ffmpeg` package as an FFmpeg fallback:

```bash
pip install imageio-ffmpeg
```

This fallback is optional and is not part of the core `requirements.txt` dependency set.

## Run the main simulation

Run from the repository root:

```bash
python main.py
```

Useful options:

```bash
python main.py --seed 7 --output-dir results/run_seed7
python main.py --n-particles 48 --grid 96 --steps-per-efold 120
python main.py --help
```

`--n-particles` is the number of particles **per side**; for example, `--n-particles 48` creates \(48^3\) particles. `--grid` is the number of mesh nodes per side.

The main pipeline builds the flat cosmology, generates Zel'dovich initial conditions at `z = 49`, evolves the particles to `z = 0`, and records density and power-spectrum measurements at the nine configured redshifts.

### Main outputs (`results/`)

| File | Contents |
|---|---|
| `cosmic_web_evolution.png` | Five-panel montage of the central density slice at `z = 49, 10, 3, 1, 0`; the figure applies a light Gaussian blur for display |
| `power_spectrum_evolution.png` | Shot-noise-subtracted \(P(k)\) curves for snapshots with positive retained power bins |
| `snapshots.npz` | Stored redshifts plus the per-snapshot arrays `slice_i`, `k_i`, and `power_i` |

The saved `slice_i` arrays are 2-D central-plane maps of \(1+\delta\). The stored slices are not the blurred montage images; the blur is applied when drawing the PNG figure.

## Experiments

The scripts in `experiments/` are standalone demonstrations and numerical checks. Run them from the repository root:

| Script | Purpose | Main diagnostic |
|---|---|---|
| `exp_001_twobody.py` | Two-body circular orbit under softened Newtonian gravity | Separation, total energy and angular momentum over 20 orbits |
| `exp_002_cluster.py` | Cold collapse of a 1,000-particle self-gravitating sphere | Energy, virial ratio, half-mass radius, montage and animation |
| `exp_003_expansion.py` | Massless particle pair in an expanding background, compared with a static run | Physical and comoving separation against the analytic scale-factor history |
| `exp_004_pm_vs_direct.py` | PM performance and force-accuracy comparison | Timing versus particle count; force errors against an Ewald periodic reference and other direct-sum baselines |
| `exp_005_cosmic_web.py` | Standalone cosmic-web density-slice demonstration | Five redshift panels (`z = 49, 10, 3, 1, 0`) |
| `exp_006_power_spectrum.py` | Standalone matter power-spectrum evolution | \(P(k)\) curves at `z = 10, 3, 1, 0` |

Example:

```bash
python experiments/exp_001_twobody.py
python experiments/exp_004_pm_vs_direct.py --skip-direct-above 10000
```

The standalone experiments have their own configurations and are not all identical to the `main.py` pipeline. See [the project document](docs/project-document.md) for their settings and interpretation.

## Generate the cosmic-web videos

The video generator consumes the saved `results/snapshots.npz`; it does **not** run or modify the N-body simulation. Run it from the repository root:

```bash
# Snapshot-only scientific evolution video
python -m visualization.make_cosmic_web_video --mode evolution

# Interpolated presentation video
python -m visualization.make_cosmic_web_video --mode cinematic

# Generate both
python -m visualization.make_cosmic_web_video --mode both
```

Outputs:

| File | Interpretation |
|---|---|
| `results/cosmic_web_evolution.mp4` | Stored snapshots only, hard cuts, fixed global colour range, no interpolation and no image smoothing by default |
| `results/cosmic_web_cinematic.mp4` | Presentation version with smoothing, upsampling, restrained sharpening, scale-factor-paced cross-fades and a gradual zoom |

The cinematic video's in-between frames are **visualisation-only**. They are not additional integrated N-body states. The current main pipeline stores 2-D slices, so the video script uses those stored slices directly. If supplied with full 3-D density snapshots instead, it can extract a central slice or slab.

## Project layout

```text
Cosmic Web Formation Simulator/
├── main.py                         # Main cosmological simulation pipeline
├── requirements.txt                # Core Python dependencies
├── src/
│   ├── cosmology.py                 # Flat matter + Lambda expansion and growth
│   ├── particles.py                 # Particle state container and diagnostics
│   ├── gravity.py                   # Direct, Ewald and Particle-Mesh gravity
│   ├── mesh.py                      # CIC mass assignment and interpolation
│   ├── poisson.py                   # Periodic FFT Poisson solver
│   ├── integrator.py                # Static and comoving KDK integration
│   └── initial_conditions.py        # Gaussian field, BBKS spectrum, Zel'dovich ICs
├── analysis/
│   └── power_spectrum.py            # Isotropic matter P(k) estimator
├── visualization/
│   ├── particles_3d.py               # 3-D particle animations and montages
│   └── make_cosmic_web_video.py      # Snapshot-only and cinematic cosmic-web videos
├── experiments/                    # Six standalone experiments
├── docs/
│   └── project-document.md          # Consolidated scientific and technical record
└── results/                         # Generated local results (not source code)
```

## Units and conventions

- Comoving positions and box lengths: `Mpc/h`.
- Particle masses: `Msun/h`.
- Velocities: `km/s`.
- Wavenumbers: `h/Mpc`.
- Power spectrum: `(Mpc/h)^3`.
- The code uses `H0 = 100` in the little-`h` convention, i.e. `km/s/(Mpc/h)`.
- The integrator's velocity is peculiar velocity, `v = a * dx/dt`; physical position is `r = a * x`.
- The Zel'dovich velocity used by the array-based main pipeline is `v = a H f D Psi`.

## Limitations and interpretation

- **Resolution.** The default run has 32,768 particles on a 64³ mesh: an average of one particle per eight mesh cells. Small-scale modes are affected by discreteness and mesh resolution; the trustworthy part of the measured spectrum is restricted to large scales.
- **Initial spectrum.** The main run uses the configured `n_s = -2.0`, `sigma8 = 0.2` input and caps the initial density contrast at 0.05. This is a demonstration setup, not a best-fit cosmological parameter set or a precision realisation calibrated to observed clustering.
- **PM force accuracy.** PM forces are smoothed below a few mesh cells. The force-comparison experiment shows a broad error distribution relative to the Ewald periodic reference; results should not be interpreted as accurate small-scale forces.
- **Power-spectrum systematics.** Poisson shot noise is subtracted and bins above the Nyquist frequency or with non-positive residual power are removed. The current estimator does not apply a CIC mass-assignment window deconvolution, so the spectrum should be treated as a low-resolution diagnostic rather than a precision estimator.
- **Finite volume and periodicity.** The cubic periodic box excludes modes larger than the box and represents a repeating volume.
- **Physics omitted.** No gas dynamics, baryonic pressure, cooling, star formation, feedback, radiation transport, magnetic fields, or galaxy-formation model is included.
- **Validation status.** The repository contains experiment scripts and runtime invariants, but the uploaded project snapshot does not include a populated automated `pytest` suite. The experiments are diagnostics, not a substitute for a full convergence study or comparison with a production cosmology code.

## Further documentation

The consolidated [Project Document](docs/project-document.md) records the project charter, requirements, architecture, numerical methods, experiment plans, validation approach, output formats, and limitations in one place.
