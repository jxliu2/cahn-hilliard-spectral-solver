"""Entry point for loading simulation/phase diagram data and making figures.

Usage:
    uv run python scripts/analyze.py
    uv run python scripts/analyze.py simulation_dir=outputs/simulation \
        output_dir=outputs/figures

All parameters correspond to keys in conf/analysis.yaml.
"""

import hydra
from omegaconf import DictConfig

from cahn_hilliard_spectral_solver.analysis import run_analysis


@hydra.main(version_base=None, config_path="../configs", config_name="analysis")
def main(cfg: DictConfig) -> None:
    run_analysis(cfg)


if __name__ == "__main__":
    main()
