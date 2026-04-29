"""Entry point for computing equilibrium ternary phase diagrams.

Usage:
    uv run python scripts/phase_diagram.py
    uv run python scripts/phase_diagram.py N_A=25 N_B=25 chi_AB=0.15
    uv run python scripts/phase_diagram.py run_N_scan=true

All parameters correspond to keys in conf/phase_diagram.yaml.
"""

import hydra
from omegaconf import DictConfig

from cahn_hilliard_spectral_solver.phase_diagram import run_phase_diagram


@hydra.main(version_base=None, config_path="../configs", config_name="phase_diagram")
def main(cfg: DictConfig) -> None:
    run_phase_diagram(cfg)


if __name__ == "__main__":
    main()
