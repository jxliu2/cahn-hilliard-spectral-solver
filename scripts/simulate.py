"""Entry point for running the Cahn-Hilliard simulation.

Usage:
    uv run python scripts/simulate.py
    uv run python scripts/simulate.py chi_AB=5.0 N_A=50 output_dir=outputs/run2

All parameters correspond to keys in conf/simulation.yaml.
Hydra will write logs and config overrides to outputs/simulate/<date>/<time>/.
"""

import hydra
from omegaconf import DictConfig

from cahn_hilliard_spectral_solver.simulation import run_simulation


@hydra.main(version_base=None, config_path="../configs", config_name="simulation")
def main(cfg: DictConfig) -> None:
    run_simulation(cfg)


if __name__ == "__main__":
    main()
