"""Morphology analysis and post-processing tools.

Port of the analysis algorithms in FH_figure_prep_v6.m. Handles:
  - triangle-based morphology classification on ternary phase diagrams
  - binodal curve extraction and smoothing
  - tie line generation
  - phase space characterization
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from scipy.spatial import Delaunay

from .utils import load_checkpoint, smooth


def classify_morphology_triangles(
    x, y, unstable_list, cutoff1=0.04, cutoff2=0.04, cutoff3=0.1
):
    """Classify Delaunay triangles by their morphology type.

    Port of the triangle-coloring loop in FH_figure_prep_v6.m.
    Each triangle is classified based on the spread (edge length) of its
    three vertices in composition space:
      1-phase: all edges < cutoff1            (homogeneous, blue)
      3-phase: any edge > cutoff3             (three coexisting phases, red)
      2-phase: intermediate                   (two coexisting phases, green)

    Additionally, region below vitrification line (phi_A+phi_B > phi_vit)
    is given a slightly different shade.

    Returns list of (triangle_xy, color_rgba, phase_count) tuples.
    """
    xy = np.column_stack([x, y])
    tri = Delaunay(xy)

    patches = []
    for simplex in tri.simplices:
        # triangle vertices in x,y
        x1, x2, x3 = x[simplex[0]], x[simplex[1]], x[simplex[2]]
        y1, y2, y3 = y[simplex[0]], y[simplex[1]], y[simplex[2]]

        # edge lengths (distances between vertices)
        d12 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        d13 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        d23 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        dmax = max(d12, d13, d23)
        dmin = min(d12, d13, d23)

        # mean instability at the three vertices
        stab_mean = np.mean(unstable_list[simplex])

        # classification
        if dmax > cutoff3:
            phase = 3  # 3-phase: large spread between vertices
        elif dmin < cutoff1 and stab_mean > 0:
            phase = 2  # 2-phase: some instability present
        elif dmax < cutoff1:
            phase = 1  # 1-phase: tight, homogeneous triangle
        else:
            phase = 2  # default intermediate to 2-phase

        tri_xy = np.array([[x1, y1], [x2, y2], [x3, y3]])
        patches.append((tri_xy, phase, stab_mean))

    return patches


def extract_binodal(x, y, unstable_list, smooth_window=11):
    """Extract and smooth the binodal boundary from the stability classification.

    Finds the boundary between stable (unstable_list=0) and spinodal
    (unstable_list=0.5) regions. Returns smoothed (x_bino, y_bino) curve.
    """
    # points at the spinodal boundary
    bino_mask = unstable_list == 0.5
    if not np.any(bino_mask):
        return np.array([]), np.array([])

    xb = x[bino_mask]
    yb = y[bino_mask]

    # sort by angle around centroid for a clean curve
    cx, cy = xb.mean(), yb.mean()
    angles = np.arctan2(yb - cy, xb - cx)
    idx = np.argsort(angles)
    xb_sorted = xb[idx]
    yb_sorted = yb[idx]

    # close the loop
    xb_sorted = np.append(xb_sorted, xb_sorted[0])
    yb_sorted = np.append(yb_sorted, yb_sorted[0])

    # smooth with Savitzky-Golay
    if len(xb_sorted) > smooth_window:
        xb_sorted = smooth(xb_sorted, smooth_window)
        yb_sorted = smooth(yb_sorted, smooth_window)

    return xb_sorted, yb_sorted


def make_tie_lines(x_bino, y_bino, n_lines=10):
    """Generate tie lines by connecting mirrored points on the binodal curve.

    Assumes the binodal is symmetric about x = 0.5 (which holds for the
    phi_A = phi_B symmetric system). Samples n_lines evenly spaced points
    along the lower part of the curve and connects them to their mirror.
    """
    if len(x_bino) == 0:
        return []

    y_min = y_bino.min()
    y_max = y_bino.max()
    y_samples = np.linspace(y_min, y_max, n_lines + 2)[1:-1]

    tie_lines = []
    for y_val in y_samples:
        # find points on curve near this y value
        idx = np.where(np.abs(y_bino - y_val) < (y_max - y_min) / (2 * n_lines))[0]
        if len(idx) < 2:
            continue
        x_pts = x_bino[idx]
        x_left = x_pts.min()
        x_right = x_pts.max()
        if x_right - x_left > 0.01:  # only draw if there's a real gap
            tie_lines.append(([x_left, x_right], [y_val, y_val]))
    return tie_lines


def ternary_color_map(phi_A, phi_B, phi_S, phi_vit=0.66):
    """Color ternary field for display. R=phi_A, G=phi_B, B=phi_S.

    Regions above vitrification (phi_A+phi_B > phi_vit) are slightly
    darkened to indicate kinetic arrest.
    """
    phi_A = np.clip(phi_A, 0, 1)
    phi_B = np.clip(phi_B, 0, 1)
    phi_S = np.clip(phi_S, 0, 1)

    rgb = np.stack([phi_A, phi_B, phi_S], axis=-1)

    # slightly reduce brightness beyond vitrification line
    vit_mask = (phi_A + phi_B) > phi_vit
    rgb[vit_mask] *= 0.85

    return rgb


def run_analysis(cfg: DictConfig) -> None:
    """Load simulation outputs and phase diagram data, then produce figures.

    Port of FH_figure_prep_v6.m.
    """
    sim_dir = Path(cfg.simulation_dir)
    pd_dir = Path(cfg.phase_diagram_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_vit = cfg.phi_vit

    # load phase diagram data
    pd_files = sorted(pd_dir.glob("phase_diagram_*.npz"))
    if not pd_files:
        print(f"No phase diagram files found in {pd_dir}")
        pd_data = None
    else:
        pd_data = load_checkpoint(str(pd_files[-1]))
        print(f"Loaded phase diagram: {pd_files[-1]}")

    # load simulation checkpoints
    ckpt_files = sorted(sim_dir.glob("checkpoint_*.npz"))
    if not ckpt_files:
        print(f"No simulation checkpoints found in {sim_dir}")
        sim_snapshots = []
    else:
        # load a subset of checkpoints for morphology evolution grid
        n_show = min(4, len(ckpt_files))
        indices = np.round(np.linspace(0, len(ckpt_files) - 1, n_show)).astype(int)
        sim_snapshots = []
        for idx in indices:
            ckpt = load_checkpoint(str(ckpt_files[idx]))
            sim_snapshots.append(ckpt)
        print(f"Loaded {len(sim_snapshots)} simulation snapshots")

    from .figures import (
        plot_morphology_grid,
        plot_phase_space_map,
        plot_ternary_phase_diagram,
    )

    # ternary phase diagram figure
    if pd_data is not None:
        x = pd_data["x"]
        y = pd_data["y"]
        f = pd_data["f"]
        unstable_list = pd_data["unstable_list"]
        phi_A0 = float(pd_data.get("phi_A0", 0.015))
        phi_B0 = float(pd_data.get("phi_B0", 0.015))
        N_A = float(pd_data.get("N_A", 150))
        N_B = float(pd_data.get("N_B", 15))
        N_S = float(pd_data.get("N_S", 1))
        chi_AS = float(pd_data.get("chi_AS", 0.0))
        chi_BS = float(pd_data.get("chi_BS", 0.0))
        chi_AB = float(pd_data.get("chi_AB", 0.15))

        from .utils import ternary_to_cartesian as t2c

        x0, y0 = t2c(np.array([phi_A0]), np.array([phi_B0]))
        x0, y0 = float(x0[0]), float(y0[0])

        lat = 0.2
        zi1, zi2 = 0.5 - lat, 0.5 + lat
        zi3, zi4 = 0.55, 0.87

        fig_path = out_dir / "ternary_phase_diagram.png"
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
            None,
            zi1,
            zi2,
            zi3,
            zi4,
            str(fig_path),
        )

        # also add vitrification line and tie lines
        xb, yb = extract_binodal(x, y, unstable_list)
        tie_lines = make_tie_lines(xb, yb, n_lines=8)

        fig_path2 = out_dir / "ternary_phase_diagram_annotated.png"
        from .figures import plot_ternary_annotated

        plot_ternary_annotated(
            x,
            y,
            f,
            unstable_list,
            x0,
            y0,
            xb,
            yb,
            tie_lines,
            phi_vit,
            N_A,
            N_B,
            chi_AS,
            chi_AB,
            str(fig_path2),
        )

    # morphology evolution grid
    if sim_snapshots:
        fig_path = out_dir / "morphology_evolution.png"
        plot_morphology_grid(sim_snapshots, phi_vit, str(fig_path))

    # N/chi phase space map
    scan_files = sorted(pd_dir.glob("N_chi_scan.npz"))
    if scan_files:
        scan = load_checkpoint(str(scan_files[-1]))
        fig_path = out_dir / "N_chi_phase_map.png"
        plot_phase_space_map(
            scan["delta_phi"],
            list(scan["N_scan"]),
            list(scan["chi_AB_scan"]),
            str(fig_path),
        )

    print(f"Figures saved to {out_dir}")
