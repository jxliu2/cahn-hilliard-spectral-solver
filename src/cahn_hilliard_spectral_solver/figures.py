"""Figure generation for Cahn-Hilliard simulation and phase diagram results.

Port of FH_figure_prep_v6.m and the plotting blocks in FH_CH_v40_3.m and
FH_ph_diag_v27.m. All figures are saved to disk (no interactive display).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def plot_snapshot(
    phi_A,
    phi_B,
    phi_S,
    t,
    t0,
    t_list,
    max_phi_A,
    min_phi_A,
    max_phi_B,
    min_phi_B,
    max_phi_S,
    min_phi_S,
    amt_A,
    amt_B,
    amt_S,
    chi_AS_list,
    dt_t0_list,
    xgrid,
    ygrid,
    Dmob_AA,
    cfg,
    save_path,
):
    """Diagnostic snapshot figure from the running simulation.

    Mirrors the 15-subplot figure in FH_CH_v40_3.m but simplified to the
    most informative panels:
      Row 1: phi_A, phi_B, phi_S, phi_A-phi_B
      Row 2: max/min phi_A(t), max/min phi_B(t), max/min phi_S(t), chi_AS(t)/dt(t)
      Row 3: RGB composite, Dmob_AA, dilute phi_A, dilute phi_B
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    L = cfg.L
    phi_A00 = cfg.phi_A00
    phi_B00 = cfg.phi_B00

    tit_str = (
        f"N_A={cfg.N_A}, N_B={cfg.N_B}, chi_AS={cfg.chi_AS:.2f}, "
        f"chi_AB={cfg.chi_AB:.2f}, phi_A0={phi_A00:.3f}, phi_B0={phi_B00:.3f}, "
        f"grid={cfg.Nx}"
    )
    fig.suptitle(tit_str, fontsize=9)

    def _surf(ax, data, title, cmap="viridis", clim=None):
        im = ax.pcolormesh(xgrid, ygrid, data, shading="auto", cmap=cmap)
        if clim is not None:
            im.set_clim(*clim)
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlim([0.25 * L, 0.75 * L])
        ax.set_ylim([0.25 * L, 0.75 * L])

    # row 1: composition fields
    _surf(axes[0], phi_A, r"$\phi_A$ tern")
    _surf(axes[1], phi_B, r"$\phi_B$ tern")
    _surf(axes[2], phi_S, r"$\phi_S$ tern")
    _surf(axes[3], phi_A - phi_B, r"$\phi_A - \phi_B$ ternary", cmap="RdBu_r")

    # row 2: time evolution diagnostics
    if len(t_list) > 1:
        t_arr = np.array(t_list)

        ax = axes[4]
        ax.plot(t_arr, max_phi_A, "b-", linewidth=2, label="max")
        ax.plot(t_arr, min_phi_A, "r-", linewidth=2, label="min")
        ax2 = ax.twinx()
        ax2.plot(t_arr, amt_A, "g--", linewidth=1)
        ax.set_title(r"Max/min/amt $\phi_A$ tern", fontsize=8)
        ax.set_xlabel("t/t0")
        ax.legend(fontsize=7)

        ax = axes[5]
        ax.plot(t_arr, max_phi_B, "b-", linewidth=2, label="max")
        ax.plot(t_arr, min_phi_B, "r-", linewidth=2, label="min")
        ax2 = ax.twinx()
        ax2.plot(t_arr, amt_B, "g--", linewidth=1)
        ax.set_title(r"Max/min/amt $\phi_B$ tern", fontsize=8)
        ax.set_xlabel("t/t0")

        ax = axes[6]
        ax.plot(t_arr, max_phi_S, "b-", linewidth=2, label="max")
        ax.plot(t_arr, min_phi_S, "r-", linewidth=2, label="min")
        ax2 = ax.twinx()
        ax2.plot(t_arr, amt_S, "g--", linewidth=1)
        ax.set_title(r"Max/min/amt $\phi_S$ tern", fontsize=8)
        ax.set_xlabel("t/t0")

        ax = axes[7]
        ax.plot(t_arr, chi_AS_list, "b-", linewidth=2, label=r"$\chi_{AS}$")
        ax.set_xlabel("t/t0")
        ax.set_title(rf"$\chi_{{AS}}$(t) = {chi_AS_list[-1]:.3f} and dt/t0", fontsize=8)
        ax2 = ax.twinx()
        if len(dt_t0_list) == len(t_arr):
            ax2.semilogy(t_arr, dt_t0_list, "r-")
    else:
        for ax in axes[4:8]:
            ax.text(
                0.5,
                0.5,
                "no data yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )

    # row 3: RGB composite and Dmob
    ax = axes[8]
    phi_tern_rgb = np.stack([phi_A, phi_B, phi_S], axis=-1)
    phi_tern_rgb = np.flip(phi_tern_rgb, axis=0)  # flipud to match MATLAB imshow
    phi_tern_rgb = np.clip(phi_tern_rgb, 0, 1)
    ax.imshow(phi_tern_rgb, origin="lower", extent=[0, L, 0, L], aspect="equal")
    ax.set_title("R=A, G=B, B=S", fontsize=8)

    _surf(axes[9], Dmob_AA, r"$D_{mob,AA}$ tern")

    # dilute field view (clamped colorscale to see small concentrations)
    _surf(axes[10], phi_A, r"Dilute $\phi_A$ tern", clim=(0, 1.1 * phi_A00))
    _surf(axes[11], phi_B, r"Dilute $\phi_B$ tern", clim=(0, 1.1 * phi_B00))

    # annotate current time
    fig.text(0.01, 0.01, f"t/t0 = {t / t0:.2f}", fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ternary_phase_diagram(
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
    save_path,
):
    """Ternary phase diagram with free energy surface and spinodal/binodal overlay.

    Port of the visualization block in FH_ph_diag_v27.m.

    Left panel: full ternary triangle (zoomed out)
    Right panel: zoomed into the coexistence region
    """
    tit_str = (
        f"N_A={N_A:.0f}, N_B={N_B:.0f}, N_S={N_S:.0f}, "
        rf"$\chi_{{AS}}$={chi_AS:.3f}, $\chi_{{BS}}$={chi_BS:.3f}, "
        rf"$\chi_{{AB}}$={chi_AB:.3f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(tit_str, fontsize=11)

    for ax_idx, ax in enumerate(axes):
        # trisurf equivalent: use tricontourf or tripcolor for 2D projection
        triang = mtri.Triangulation(x, y)

        # free energy surface as color-coded scatter (view from above = 2D projection)
        sc = ax.tripcolor(triang, f, shading="gouraud", cmap="viridis", alpha=0.7)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="f")

        # plot spinodal (unstable) and indefinite (spinodal boundary) regions
        # red = unstable (negative definite Hessian)
        # green = indefinite (spinodal boundary)
        mask_unstable = unstable_list == 1
        mask_indef = unstable_list == 0.5
        ax.plot(
            x[mask_unstable],
            y[mask_unstable],
            ".r",
            markersize=3,
            label="unstable",
            zorder=3,
        )
        ax.plot(
            x[mask_indef], y[mask_indef], ".g", markersize=3, label="spinodal", zorder=3
        )

        # initial composition marker
        ax.plot(x0, y0, ".r", markersize=15, zorder=5, label="IC")
        composition_string = f"$\\phi_{{A0}}={phi_A0:.3f}$\n$\\phi_{{B0}}={phi_B0:.3f}$"
        ax.annotate(
            composition_string,
            (x0, y0),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
        )

        # ternary triangle outline
        ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], "k-", linewidth=1.5)

        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=8)

        if ax_idx == 0:
            # full triangle view
            ax.set_xlim([0, 1])
            ax.set_ylim([0, np.sqrt(3) / 2])
            ax.set_title("Phase coexistence and spinodal regions", fontsize=9)
            # corner labels
            ax.text(-0.05, -0.04, "Polymer B", fontsize=10)
            ax.text(0.88, -0.04, "Polymer A", fontsize=10)
            ax.text(0.45, np.sqrt(3) / 2 + 0.02, "Solvent", fontsize=10)
        else:
            # zoomed-in view
            ax.set_xlim([zi1, zi2])
            ax.set_ylim([zi3, zi4])
            tit = (
                f"Phase coexistence, "
                rf"$\phi_{{A0}}={phi_A0:.3f}$, $\phi_{{B0}}={phi_B0:.3f}$"
            )
            ax.set_title(tit, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_ternary_annotated(
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
    save_path,
):
    """Publication-style ternary phase diagram with binodal, tie lines, and vit line.

    Adds:
      - binodal curve (smoothed)
      - tie lines (light gray)
      - vitrification line at phi_A + phi_B = phi_vit (y = sqrt(3)/2*(1-phi_vit))
      - starting composition marker
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    triang = mtri.Triangulation(x, y)

    # free energy surface
    ax.tripcolor(triang, f, shading="gouraud", cmap="viridis", alpha=0.55)

    # spinodal regions
    mask_unstable = unstable_list == 1
    mask_indef = unstable_list == 0.5
    ax.plot(
        x[mask_unstable],
        y[mask_unstable],
        ".r",
        markersize=2,
        alpha=0.6,
        label="unstable",
        zorder=3,
    )
    ax.plot(
        x[mask_indef],
        y[mask_indef],
        ".g",
        markersize=2,
        alpha=0.6,
        label="spinodal boundary",
        zorder=3,
    )

    # binodal curve
    if len(xb) > 0:
        ax.plot(xb, yb, "k-", linewidth=1.5, label="binodal", zorder=4)

    # tie lines (light gray)
    tie_color = [0.85, 0.85, 0.85]
    for xl, yl in tie_lines:
        ax.plot(xl, yl, "-", color=tie_color, linewidth=0.8, zorder=2)

    # vitrification line: phi_A + phi_B = phi_vit
    # in ternary coords: y = sqrt(3)/2 * (1 - phi_vit) for phi_A = phi_B line
    # horizontal line at y = sqrt(3)/2 * (1 - phi_vit)
    # y = sqrt(3)/2 - (phi_A+phi_B)*sqrt(3)/2 => phi_A+phi_B = 1 - 2y/sqrt(3)
    y_vit = np.sqrt(3) / 2 * (1.0 - phi_vit)
    ax.axhline(
        y=y_vit,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=rf"$\phi_{{A+B}}={phi_vit}$ (vit)",
        zorder=2,
    )

    # initial composition
    ax.plot(x0, y0, "co", markersize=8, zorder=6, label="starting composition")

    # triangle outline
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], "k-", linewidth=1.5)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.04, np.sqrt(3) / 2 + 0.06])
    ax.set_aspect("equal")
    ax.set_axis_off()

    # corner labels
    ax.text(-0.07, -0.04, "Polymer B", fontsize=12)
    ax.text(0.90, -0.04, "Polymer A", fontsize=12)
    ax.text(0.44, np.sqrt(3) / 2 + 0.03, "Solvent", fontsize=12)

    tit = (
        rf"$N_{{A,B}}$={N_A:.0f}/{N_B:.0f}, "
        rf"$\chi_{{AS,BS}}$={chi_AS:.3f}, $\chi_{{AB}}$={chi_AB:.3f}"
    )
    ax.set_title(tit, fontsize=11)
    ax.legend(
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_morphology_grid(snapshots, phi_vit, save_path, times=None):
    """Grid of morphology snapshots showing temporal evolution.

    Port of the 4×5 morphology evolution subplot in FH_figure_prep_v6.m.
    Each column is a time snapshot; rows show phi_A, phi_B, phi_S, and RGB.
    """
    n_snaps = len(snapshots)
    if n_snaps == 0:
        return

    fig, axes = plt.subplots(4, n_snaps, figsize=(4 * n_snaps, 14))
    if n_snaps == 1:
        axes = axes[:, np.newaxis]

    row_labels = [r"$\phi_A$", r"$\phi_B$", r"$\phi_S$", "RGB (A,B,S)"]

    for col, ckpt in enumerate(snapshots):
        phi_A = ckpt["phi_A"]
        phi_B = ckpt["phi_B"]
        phi_S = ckpt["phi_S"]
        t_val = float(ckpt.get("t", 0))

        fields = [phi_A, phi_B, phi_S]
        for row in range(3):
            ax = axes[row, col]
            im = ax.imshow(
                fields[row],
                origin="lower",
                cmap="viridis",
                vmin=0,
                vmax=fields[row].max(),
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11)
            if row == 0:
                t_label = f"t={t_val:.1f}" if times is None else f"t={times[col]:.1f}"
                ax.set_title(t_label, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # RGB composite row
        ax = axes[3, col]
        rgb = np.stack([phi_A, phi_B, phi_S], axis=-1)
        rgb = np.flip(rgb, axis=0)  # flipud like MATLAB's imshow
        rgb = np.clip(rgb, 0, 1)
        ax.imshow(rgb, origin="lower", aspect="equal")
        if col == 0:
            ax.set_ylabel(row_labels[3], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_phase_space_map(delta_phi_matrix, N_scan, chi_AB_scan, save_path):
    """LMW vs HMW phase diagram: morphology as function of N and chi_AB.

    Port of the phase space map in FH_figure_prep_v6.m.
    Colors: blue = homogeneous, yellow/green = bicontinuous, purple = Janus.
    Regions defined by delta_phi threshold.
    """
    N_arr = np.array(N_scan)
    chi_arr = np.array(chi_AB_scan)

    fig, ax = plt.subplots(figsize=(8, 6))

    # imshow with N on y-axis, chi_AB on x-axis
    im = ax.imshow(
        delta_phi_matrix,
        origin="lower",
        aspect="auto",
        extent=[chi_arr[0], chi_arr[-1], N_arr[0], N_arr[-1]],
        cmap="RdYlBu_r",
    )
    fig.colorbar(im, ax=ax, label=r"$\Delta\phi = \phi_{precip} - \phi_{UCST}$")

    # contour at delta_phi = 0 marks the phase boundary
    if delta_phi_matrix.max() > 0 and delta_phi_matrix.min() < 0:
        ax.contour(
            chi_arr, N_arr, delta_phi_matrix, levels=[0], colors=["k"], linewidths=2
        )

    ax.set_xlabel(r"$\chi_{AB}$", fontsize=13)
    ax.set_ylabel(r"$N_{A,B}$", fontsize=13)
    ax.set_title(r"$\Delta\phi$ phase space map", fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
