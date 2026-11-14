# Cosmic Web Formation Simulator

A compact, NumPy-based cosmological N-body simulator for exploring how gravitational instability transforms a nearly uniform early matter distribution into a large-scale cosmic web of **sheets, filaments, dense nodes, and underdense voids**.

The code evolves collisionless matter in a periodic, expanding, spatially flat ΛCDM box. Its main simulation path uses a **Particle-Mesh (PM) gravity solver**, a Fourier-space Poisson solver, and a comoving Kick-Drift-Kick (KDK) integrator. Separate experiments compare numerical components against simpler or higher-fidelity reference calculations.

> **Scope:** This is a compact educational and research-oriented numerical experiment, not a production cosmological simulation. It is gravity-only. Baryonic hydrodynamics, gas cooling, star formation, stellar or black-hole feedback, and radiation transport are not modelled.

---

## Visual Preview

<!-- Add cluster_collapse.mp4 here -->

**Cluster collapse — numerical experiment using the simulator's self-gravitating particle system.**

---

# Features

### 1. Particle-Mesh Gravity

Periodic Cloud-in-Cell (CIC) mass assignment and force interpolation, with an FFT Poisson solver using:

```python
numpy.fft.rfftn
numpy.fft.irfftn
```

In cosmological mode, the potential satisfies:

\[
\nabla^2 \Phi =
\frac{3}{2}\Omega_{m,0}H_0^2
\frac{\delta}{a}
\]

An Ewald-summed periodic direct solver is provided as a force-accuracy reference.

---

### 2. Comoving KDK Integration

A time-centred Kick-Drift-Kick scheme evolves comoving positions and peculiar velocities.

Hubble drag is handled analytically through the scale-factor ratio.

Without a cosmology object, the integrator runs as a standard Newtonian leapfrog.

---

### 3. Zel'dovich Initial Conditions

A Gaussian random field is generated from a BBKS-shaped power spectrum, converted to a displacement field in Fourier space, and mapped onto particles using the linear growth factor \(D(a)\) and growth rate \(f(a)\).

---

### 4. Power-Spectrum Analysis

An isotropic matter power-spectrum estimator:

- Uses FFTs and radial shell binning
- Subtracts Poisson shot noise
- Discards non-positive noise-dominated bins
- Removes modes above the mesh Nyquist frequency

---

### 5. Reference Gravity Solvers

- Softened direct-summation gravity as an \(O(N^2)\) baseline
- Ewald-summed periodic gravity as a periodic reference for PM comparisons

---

### 6. Visualisation

Produces:

- 3D particle animations (`.gif`, `.mp4`)
- Density-slice figures
- Power-spectrum evolution plots
- Two cosmic-web video styles

---

# Model and Default Configuration

The main pipeline uses a spatially flat matter-plus-cosmological-constant background:

\[
H(a)=H_0 \sqrt{\Omega_{m,0}a^{-3}+\Omega_{\Lambda,0}}
\]

with

\[
\Omega_{m,0}+\Omega_{\Lambda,0}=1
\]

Default parameters defined in `main.py`:

| Parameter | Default | Meaning |
|------------|----------|----------|
| Particles per side | `32` | Cubic lattice of \(32^3 = 32{,}768\) particles |
| PM mesh | `64` | Periodic \(64^3\) mesh |
| Box size | `100 Mpc/h` | Comoving simulation volume |
| \(H_0\) | `100 km/s/(Mpc/h)` | Hubble constant |
| \(\Omega_{m,0}\) | `0.3` | Present-day matter density |
| \(\Omega_{\Lambda,0}\) | `0.7` | Present-day cosmological-constant density |
| Initial redshift | `49` | \(a_{init}=0.02\) |
| Spectral index | `-2.0` | Demonstration spectral slope |
| `sigma8` input | `0.2` | BBKS spectrum amplitude |
| Initial contrast cap | `0.05` | Ensures \(\max(\delta_{init}) \le 0.05\) |
| Random seed | `42` | Reproducible realization |
| Steps per e-fold | `80` | Integration step density |

Snapshots are recorded at:

```text
z = 49, 20, 10, 5, 3, 2, 1, 0.5, 0
```

The `sigma8` parameter sets the amplitude of the BBKS-shaped input spectrum, but the pipeline additionally applies a `max_delta_init` rescaling. The resulting realization is intended as a smooth, near-uniform demonstration setup rather than a calibrated prediction of observed clustering.

---

# Numerical Model

## Initial Conditions

A real white-noise field is transformed to Fourier space and filtered using:

\[
P(k)\propto k^{n_s}T_{\rm BBKS}(k)^2
\]

where \(T_{\rm BBKS}\) is the BBKS CDM transfer function.

The displacement field is

\[
\Psi(\mathbf{k})
=
-i\frac{\mathbf{k}}{k^2}\delta(\mathbf{k})
\]

with the zero mode removed.

Particles are displaced from a cell-centred Lagrangian lattice and assigned Zel'dovich peculiar velocities using the cosmological growth history.

---

## Particle-Mesh Gravity

For each force evaluation the PM solver:

1. Wraps particles into the periodic box.
2. Builds the eight-node CIC stencil.
3. Deposits mass onto the mesh.
4. Computes density contrast:

\[
\delta=\frac{\rho-\bar{\rho}}{\bar{\rho}}
\]

5. Solves the periodic Poisson equation.
6. Computes spectral gradients.
7. Interpolates accelerations back onto particles.

Computational cost:

\[
O(N + M^3\log M)
\]

where:

- \(N\) = particle count
- \(M^3\) = mesh cells

The mesh representation smooths forces below a few cell widths.

---

## Comoving Integration

The cosmological integrator evolves:

\[
\frac{d\mathbf{x}}{dt}
=
\frac{\mathbf{v}}{a}
\]

\[
\frac{d\mathbf{v}}{dt}
=
\mathbf{g}
-
H(a)\mathbf{v}
\]

Hubble drag is applied analytically through scale-factor ratios.

- Drift integral → Simpson's rule
- Force impulse → time-centred quadrature

The scheme is second-order accurate in time.

---

# Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/vineet-b23/cosmic-web-simulation
cd "Cosmic Web Formation Simulator"

python -m venv .venv
```

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dependencies

| Package | Purpose |
|----------|----------|
| `numpy` | Particle arrays, FFTs, numerical kernels |
| `scipy` | Quadrature, Ewald summation, image processing |
| `matplotlib` | Figures and animations |

For MP4 rendering, FFmpeg must be available on `PATH`.

Optional fallback:

```bash
pip install imageio-ffmpeg
```

---

# Run the Main Simulation

From the repository root:

```bash
python main.py
```

Examples:

```bash
python main.py --seed 7 --output-dir results/run_seed7

python main.py --n-particles 48 --grid 96 --steps-per-efold 120

python main.py --help
```

Notes:

- `--n-particles` specifies particles per side
- `--n-particles 48` creates \(48^3\) particles
- `--grid` specifies mesh cells per side

The pipeline:

1. Builds a flat ΛCDM cosmology
2. Generates Zel'dovich initial conditions at `z = 49`
3. Evolves to `z = 0`
4. Records density and power-spectrum measurements

---

# Main Outputs (`results/`)

| File | Contents |
|--------|----------|
| `cosmic_web_evolution.png` | Five-panel density montage at z = 49, 10, 3, 1, 0 |
| `power_spectrum_evolution.png` | Power-spectrum evolution |
| `snapshots.npz` | Stored density slices and power-spectrum measurements |

The stored `slice_i` arrays contain central-plane maps of:

\[
1+\delta
\]

The Gaussian blur is only applied when generating the PNG montage.

---

# Experiments

Run from the repository root.

| Script | Purpose | Main Diagnostic |
|----------|----------|----------|
| `exp_001_twobody.py` | Two-body orbit | Energy and angular momentum conservation |
| `exp_002_cluster.py` | Cluster collapse | Virial ratio and half-mass radius evolution |
| `exp_003_expansion.py` | Expansion test | Physical vs. comoving separation |
| `exp_004_pm_vs_direct.py` | PM validation | Force accuracy and timing |
| `exp_005_cosmic_web.py` | Cosmic-web demonstration | Density evolution |
| `exp_006_power_spectrum.py` | Power-spectrum evolution | \(P(k)\) evolution |

Examples:

```bash
python experiments/exp_001_twobody.py

python experiments/exp_004_pm_vs_direct.py --skip-direct-above 10000
```

> The standalone experiments have independent configurations and are not guaranteed to match `main.py` settings exactly.

---

# Generate Cosmic-Web Videos

The video generator consumes:

```text
results/snapshots.npz
```

It **does not rerun the simulation**.

### Snapshot-only scientific video

```bash
python -m visualization.make_cosmic_web_video --mode evolution
```

### Cinematic presentation video

```bash
python -m visualization.make_cosmic_web_video --mode cinematic
```

### Generate both

```bash
python -m visualization.make_cosmic_web_video --mode both
```

---

## Video Output (sample)
```text
results/exp_002_cluster/clustercollapse.mp4
```


# Project Layout

```text
Cosmic Web Formation Simulator/
│
├── main.py
├── requirements.txt
│
├── src/
│   ├── cosmology.py
│   ├── particles.py
│   ├── gravity.py
│   ├── mesh.py
│   ├── poisson.py
│   ├── integrator.py
│   └── initial_conditions.py
│
├── analysis/
│   └── power_spectrum.py
│
├── visualization/
│   ├── particles_3d.py
│   └── make_cosmic_web_video.py
│
├── experiments/
│
├── docs/
│   └── project-document.md
│
└── results/
```

---

# Units and Conventions

| Quantity | Units |
|-----------|--------|
| Comoving positions | `Mpc/h` |
| Particle masses | `Msun/h` |
| Peculiar velocities | `km/s` |
| Wavenumbers | `h/Mpc` |
| Power spectrum | `(Mpc/h)^3` |

Additional conventions:

- `H0 = 100 km/s/(Mpc/h)`
- Physical position:

\[
r=a\,x
\]

- Peculiar velocity:

\[
v=a\frac{dx}{dt}
\]

- Zel'dovich velocity:

\[
v=aHfD\Psi
\]

---

# Limitations and Interpretation

### Resolution

The default run uses:

- 32,768 particles
- \(64^3\) mesh

Small-scale results are affected by discreteness and mesh resolution.

### Initial Spectrum

The demonstration uses:

```text
n_s = -2.0
sigma8 = 0.2
max_delta_init = 0.05
```

and is not calibrated to observational constraints.

### PM Force Accuracy

PM forces are smoothed below a few mesh cells.

### Power Spectrum Systematics

Current estimator:

- Subtracts Poisson shot noise
- Removes Nyquist-scale modes
- Does not perform CIC window deconvolution

### Finite Volume

The simulation volume is periodic and excludes modes larger than the box.

### Missing Physics

Not included:

- Gas dynamics
- Cooling
- Star formation
- Stellar feedback
- Black-hole feedback
- Radiation transport
- Magnetic fields
- Galaxy formation models

### Validation Status

The repository contains:

- Diagnostic experiments
- Runtime consistency checks

but does **not currently include a populated automated `pytest` suite**.

---

# Further Documentation

The consolidated project document records:

- Project charter
- Requirements
- Architecture
- Numerical methods
- Validation approach
- Experiment plans
- Output formats
- Known limitations

See:

```text
docs/project-document.pdf
```

for the complete scientific and technical record.

---


# Citation

If you use this project in academic work, please cite the repository and associated project documentation.
