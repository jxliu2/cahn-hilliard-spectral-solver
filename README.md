# cahn-hilliard-spectral-solver

Spectral IMEX Cahn-Hilliard solver for ternary polymer-solvent systems with Flory-Huggins thermodynamics. Code supporting:

> Liu, J. et al. "Evolution of Polymer Colloid Structure During Precipitation and Phase Separation." *JACS Au* 2021.
> https://pubs.acs.org/doi/full/10.1021/jacsau.1c00110

## Overview

The simulations model phase separation in a ternary A/B/solvent mixture during solvent-exchange-induced precipitation. The competition between polymer-polymer phase separation and vitrification determines the final colloid morphology (homogeneous, bicontinuous, Janus, or patchy).

Three pipelines are available:

| Pipeline | What it does |
|---|---|
| `simulate` | Time-evolves the composition fields using a Fourier-spectral IMEX scheme |
| `phase_diagram` | Computes the equilibrium ternary phase diagram from the Flory-Huggins free energy |
| `analyze` | Loads simulation checkpoints and phase diagram data to produce figures |

## Installation

Requires Python 3.13+. Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Running

Scripts live in `scripts/`. Parameters are configured via [Hydra](https://hydra.cc) — any key in the config files can be overridden on the command line.

### Phase diagram

```bash
uv run python scripts/phase_diagram.py
```

Computes the FH free energy surface, spinodal regions (Hessian eigenvalue analysis), and convex hull binodal for the default parameters (N_A=150, N_B=15, chi_AB=0.15). Output goes to `outputs/phase_diagram/`.

```bash
# different chain lengths or interaction parameters
uv run python scripts/phase_diagram.py N_A=25 N_B=25 chi_AB=0.1

# scan over N and chi_AB to map the phase space
uv run python scripts/phase_diagram.py run_N_scan=true
```

### Simulation

```bash
uv run python scripts/simulate.py
```

Runs the Cahn-Hilliard simulation with default parameters (128×128 grid, chi_AB=7, chi_AS=3.5, Gaussian nucleus initialization). Prints diagnostics to stdout and saves checkpoints + snapshot figures to `outputs/simulation/`. The simulation runs indefinitely until interrupted; set `t_end` (in units of t0) to stop automatically.

```bash
# run until t = 5000 t0, saving more frequently
uv run python scripts/simulate.py t_end=5000 save_interval_t0=500 plot_interval_t0=500

# different composition or chi parameters
uv run python scripts/simulate.py phi_A00=0.05 phi_B00=0.05 chi_AB=5.0 N_A=50

# ramp chi_AS from 0.5 to 1.15 over 1e6 t0 (mimics solvent exchange)
uv run python scripts/simulate.py chi_AS=0.5 chi_AS_final=1.15 output_dir=outputs/ramp_run

# enable Langevin noise
uv run python scripts/simulate.py langevin_noise_ampl=0.01
```

Simulations can be continued from the last checkpoint automatically — just re-run the same command pointing to the same `output_dir`.

### Analysis / figures

```bash
uv run python scripts/analyze.py
```

Loads the most recent phase diagram and simulation checkpoints and produces annotated figures (ternary diagram with binodal/tie lines, morphology evolution grid). Output goes to `outputs/figures/`.

```bash
uv run python scripts/analyze.py simulation_dir=outputs/myrun output_dir=outputs/myfigures
```

## Configuration

Default parameters live in `configs/`. The main ones:

**`configs/simulation.yaml`**
- `Nx`, `Ny`, `L` — grid resolution and box size
- `N_A`, `N_B`, `N_S` — degrees of polymerization
- `chi_AB`, `chi_AS`, `chi_BS` — Flory-Huggins interaction parameters
- `phi_A00`, `phi_B00` — background composition
- `use_nucleus`, `nucl_phi`, `nucl_R`, `offset` — Gaussian nucleus initialization
- `mobility_averaging` — `geometric` (default), `arithmetic`, or `harmonic`

**`configs/phase_diagram.yaml`**
- `N_A`, `N_B`, `chi_AB` — system parameters
- `npts` — grid resolution for the composition simplex

## Output

Simulation checkpoints are saved as `.npz` files and can be loaded with `numpy.load`. Snapshot figures are saved as `.png`.
