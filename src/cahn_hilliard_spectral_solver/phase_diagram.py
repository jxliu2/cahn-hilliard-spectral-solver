"""Ternary equilibrium phase diagram via Flory-Huggins free energy.

Port of FH_ph_diag_v27.m. Computes the free energy surface, spinodal stability
regions (from Hessian eigenvalues), and the phase coexistence envelope
(binodal via convex hull). Saves results and figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from omegaconf import DictConfig
from scipy.spatial import ConvexHull

from .free_energy import fh_free_energy, stability_label
from .utils import save_checkpoint, ternary_to_cartesian


def compute_phase_diagram(
    N_A, N_B, N_S, chi_AS, chi_BS, chi_AB, npts=100, epsilon=1e-10
):
    """Build the ternary composition grid and compute free energy and stability.

    Returns a dict with:
      phi_A_list, phi_B_list, phi_S_list  -- composition arrays
      x, y                                -- ternary triangle 2D coordinates
      f                                   -- FH free energy at each point
      unstable_list                       -- 0=stable, 0.5=indefinite, 1=unstable
    """
    # generate phi_list: enumerate all (phi_A, phi_B) pairs on a grid
    # subject to phi_A + phi_B < 1 (inside the ternary simplex)
    phis_A = np.linspace(epsilon, 1.0 - epsilon, npts)
    phis_B = np.linspace(epsilon, 1.0 - epsilon, npts)

    phi_A_list = []
    phi_B_list = []
    phi_S_list = []

    for phi_A in phis_A:
        for phi_B in phis_B:
            phi_S = 1.0 - phi_A - phi_B
            if phi_A + phi_B < 1.0:
                phi_A_list.append(phi_A)
                phi_B_list.append(phi_B)
                phi_S_list.append(phi_S)

    phi_A_arr = np.array(phi_A_list)
    phi_B_arr = np.array(phi_B_list)
    phi_S_arr = np.array(phi_S_list)

    # conversion to ternary triangle x,y coordinates
    # https://mathworld.wolfram.com/TernaryDiagram.html
    x, y = ternary_to_cartesian(phi_A_arr, phi_B_arr)

    # calculate free energies
    f = fh_free_energy(
        phi_A_arr, phi_B_arr, phi_S_arr, N_A, N_B, N_S, chi_AB, chi_AS, chi_BS
    )

    # calculate spinodals from analytical Hessian (eigenvalue analysis)
    unstable = stability_label(
        phi_A_arr, phi_B_arr, phi_S_arr, N_A, N_B, N_S, chi_AB, chi_AS, chi_BS
    )

    # determinant of Hessian (for reference / plotting)
    # df_det = H_AA*H_BB - H_AB^2
    from .free_energy import fh_hessian_elements

    H_AA, H_BB, H_AB = fh_hessian_elements(
        phi_A_arr, phi_B_arr, phi_S_arr, N_A, N_B, N_S, chi_AB, chi_AS, chi_BS
    )
    df_det = H_AA * H_BB - H_AB**2

    return dict(
        phi_A=phi_A_arr,
        phi_B=phi_B_arr,
        phi_S=phi_S_arr,
        x=x,
        y=y,
        f=f,
        unstable_list=unstable,
        df_det=df_det,
        H_AA=H_AA,
        H_BB=H_BB,
        H_AB=H_AB,
    )


def run_phase_diagram(cfg: DictConfig) -> None:
    """Main driver for phase diagram calculation. Port of FH_ph_diag_v27.m."""

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N_A = cfg.N_A
    N_B = cfg.N_B
    N_S = cfg.N_S

    chi_PS = cfg.chi_AS
    chi_AS = chi_PS
    chi_BS = chi_PS
    chi_AB = cfg.chi_AB

    phi_A0 = cfg.phi_A0
    phi_B0 = cfg.phi_B0

    npts_zoomout = cfg.npts
    npts_zoomin = cfg.npts_fine
    epsilon = cfg.epsilon

    lat = 0.2  # lateral width for zoomed-in view
    zi1, zi2 = 0.5 - lat, 0.5 + lat
    zi3, zi4 = 0.55, 0.87

    print("\n=== Phase diagram ===")
    print(f"N_A={N_A}, N_B={N_B}, N_S={N_S}")
    print(f"chi_AS={chi_AS}, chi_BS={chi_BS}, chi_AB={chi_AB}")
    print(f"phi_A0={phi_A0}, phi_B0={phi_B0}")
    print(f"Grid: {npts_zoomout} pts (coarse), {npts_zoomin} (fine)\n")

    # coarse grid (full triangle)
    result = compute_phase_diagram(
        N_A, N_B, N_S, chi_AS, chi_BS, chi_AB, npts=npts_zoomout, epsilon=epsilon
    )

    x = result["x"]
    y = result["y"]
    f = result["f"]
    unstable_list = result["unstable_list"]
    phi_A_arr = result["phi_A"]
    phi_B_arr = result["phi_B"]

    # initial composition marker
    x0, y0 = ternary_to_cartesian(np.array([phi_A0]), np.array([phi_B0]))
    x0, y0 = float(x0[0]), float(y0[0])

    # convex hull of free energy surface (gives binodal / phase coexistence lines)
    # k = convhull(x, y, f): lower convex hull = common tangent plane construction
    pts = np.column_stack([x, y, f])
    try:
        hull = ConvexHull(pts)
        k_triangles = hull.simplices
    except Exception as e:
        print(f"ConvexHull failed: {e}")
        k_triangles = None

    # save result data
    data_path = out_dir / f"phase_diagram_N_A{N_A}_N_B{N_B}_chiAB{chi_AB:.3f}.npz"
    save_checkpoint(
        str(data_path),
        dict(
            phi_A=phi_A_arr,
            phi_B=phi_B_arr,
            phi_S=result["phi_S"],
            x=x,
            y=y,
            f=f,
            unstable_list=unstable_list,
            df_det=result["df_det"],
            N_A=np.array(N_A),
            N_B=np.array(N_B),
            N_S=np.array(N_S),
            chi_AS=np.array(chi_AS),
            chi_BS=np.array(chi_BS),
            chi_AB=np.array(chi_AB),
            phi_A0=np.array(phi_A0),
            phi_B0=np.array(phi_B0),
        ),
    )
    print(f"Saved: {data_path}")

    from .figures import plot_ternary_phase_diagram

    fig_path = out_dir / f"phase_diagram_N_A{N_A}_N_B{N_B}_chiAB{chi_AB:.3f}.png"
    plot_ternary_phase_diagram(
        x,
        y,
        f,
        unstable_list,
        x0,
        y0,
        phi_A0,
        phi_B0,
        N_A,
        N_B,
        N_S,
        chi_AS,
        chi_BS,
        chi_AB,
        k_triangles,
        zi1,
        zi2,
        zi3,
        zi4,
        str(fig_path),
    )

    # optional parameter scans
    if cfg.get("run_N_scan", False):
        _scan_N_chi(cfg, out_dir)


def _scan_N_chi(cfg, out_dir):
    """Scan over N_A,B and chi_AB to map delta_phi phase behavior.

    Computes the composition gap delta_phi = phi_precip - phi_ucst that
    characterizes whether phase separation occurs before vitrification.
    Port of the commented sweep loops in FH_ph_diag_v27.m.
    """
    N_scan = list(cfg.get("N_scan", [15, 25, 50]))
    chi_AB_scan = list(cfg.get("chi_AB_scan", [0.05, 0.1, 0.15, 0.2, 0.3]))
    N_S = cfg.N_S
    chi_AS = cfg.chi_AS
    chi_BS = cfg.chi_AS
    epsilon = cfg.epsilon
    npts = cfg.npts

    print(f"\nRunning N/chi_AB scan: N={N_scan}, chi_AB={chi_AB_scan}")

    delta_phi_matrix = np.zeros((len(N_scan), len(chi_AB_scan)))
    phi_ucst_matrix = np.zeros_like(delta_phi_matrix)
    phi_precip_matrix = np.zeros_like(delta_phi_matrix)

    for i, N_AB in enumerate(N_scan):
        for j, chi_AB in enumerate(chi_AB_scan):
            result = compute_phase_diagram(
                N_AB, N_AB, N_S, chi_AS, chi_BS, chi_AB, npts=npts, epsilon=epsilon
            )
            unstable = result["unstable_list"]
            phi_A = result["phi_A"]

            # find UCST: max instability on phi_A ≈ phi_B symmetry axis
            sym_mask = np.abs(phi_A - result["phi_B"]) < (1.0 / npts * 2)
            if np.any(sym_mask & (unstable > 0)):
                phi_ucst = phi_A[sym_mask & (unstable > 0)].max()
            else:
                phi_ucst = 0.0

            # find precipitation composition: largest phi_A+phi_B at spinodal boundary
            if np.any(unstable > 0):
                phi_tot_unstable = (phi_A + result["phi_B"])[unstable > 0]
                phi_precip = phi_tot_unstable.max()
            else:
                phi_precip = 0.0

            delta_phi = phi_precip - phi_ucst
            delta_phi_matrix[i, j] = delta_phi
            phi_ucst_matrix[i, j] = phi_ucst
            phi_precip_matrix[i, j] = phi_precip

    scan_path = out_dir / "N_chi_scan.npz"
    save_checkpoint(
        str(scan_path),
        dict(
            N_scan=np.array(N_scan),
            chi_AB_scan=np.array(chi_AB_scan),
            delta_phi=delta_phi_matrix,
            phi_ucst=phi_ucst_matrix,
            phi_precip=phi_precip_matrix,
        ),
    )
    print(f"Saved scan: {scan_path}")

    from .figures import plot_phase_space_map

    fig_path = out_dir / "N_chi_phase_map.png"
    plot_phase_space_map(delta_phi_matrix, N_scan, chi_AB_scan, str(fig_path))
