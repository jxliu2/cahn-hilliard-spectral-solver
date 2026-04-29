"""Flory-Huggins free energy: bulk thermodynamics, chemical potentials, Hessian.

Port of the thermodynamic functions in FH_CH_v40_3.m and FH_ph_diag_v27.m.
"""

import numpy as np

from .utils import log_taylor


def fh_free_energy(phi_A, phi_B, phi_S, N_A, N_B, N_S, chi_AB, chi_AS, chi_BS):
    """Flory-Huggins free energy density for a ternary A-B-S mixture.

    f = phi_A*ln(phi_A)/N_A + phi_B*ln(phi_B)/N_B + phi_S*ln(phi_S)/N_S
      + chi_AB*phi_A*phi_B + chi_AS*phi_A*phi_S + chi_BS*phi_B*phi_S
    """
    phi_A = np.asarray(phi_A, dtype=float)
    phi_B = np.asarray(phi_B, dtype=float)
    phi_S = np.asarray(phi_S, dtype=float)

    # entropic terms (FH lattice model)
    f = (
        phi_A * np.log(np.maximum(phi_A, 1e-300)) / N_A
        + phi_B * np.log(np.maximum(phi_B, 1e-300)) / N_B
        + phi_S * np.log(np.maximum(phi_S, 1e-300)) / N_S
    )
    # enthalpic (chi) interaction terms
    f += chi_AB * phi_A * phi_B + chi_AS * phi_A * phi_S + chi_BS * phi_B * phi_S
    return f


def chemical_potentials(
    phi_A, phi_B, phi_S, N_A, N_B, N_S, chi_AB, chi_AS, chi_BS, beta=1e-8
):
    """Homogeneous chemical potentials df/dphi_i (no gradient energy terms).

    Uses the Saylor (2007) correction term (beta/phi^2) to regularize near
    phi = 0. This is the active implementation in FH_CH_v40_3.m.

    Returns (df_dphi_A, df_dphi_B, df_dphi_S).
    """
    phi_A = np.asarray(phi_A, dtype=float)
    phi_B = np.asarray(phi_B, dtype=float)
    phi_S = np.asarray(phi_S, dtype=float)

    # standard log derivatives of the FH free energy
    # df/dphi_A = log(phi_A)/N_A + 1/N_A + chi_AS*phi_S + chi_AB*phi_B
    df_dphi_A = (
        np.log(np.maximum(phi_A, 1e-300)) / N_A
        + 1.0 / N_A
        + chi_AS * phi_S
        + chi_AB * phi_B
    )
    df_dphi_B = (
        np.log(np.maximum(phi_B, 1e-300)) / N_B
        + 1.0 / N_B
        + chi_BS * phi_S
        + chi_AB * phi_A
    )
    df_dphi_S = (
        np.log(np.maximum(phi_S, 1e-300)) / N_S
        + 1.0 / N_S
        + chi_AS * phi_A
        + chi_BS * phi_B
    )

    # Saylor 2007 correction: adds repulsive term near phi = 0 so that
    # each component is pushed away from unphysical concentrations
    # saylor_A = -beta/phi_A^2 + beta/phi_B^2 + beta/phi_S^2
    saylor_A = -beta / phi_A**2 + beta / phi_B**2 + beta / phi_S**2
    saylor_B = -beta / phi_B**2 + beta / phi_A**2 + beta / phi_S**2
    saylor_S = -beta / phi_S**2 + beta / phi_A**2 + beta / phi_B**2

    df_dphi_A = df_dphi_A + saylor_A
    df_dphi_B = df_dphi_B + saylor_B
    df_dphi_S = df_dphi_S + saylor_S

    return df_dphi_A, df_dphi_B, df_dphi_S


def chemical_potentials_log(phi_A, phi_B, phi_S, N_A, N_B, N_S, chi_AB, chi_AS, chi_BS):
    """Standard log chemical potentials, no Saylor correction.

    Alternative to chemical_potentials(); kept for reference.
    """
    phi_A = np.asarray(phi_A, dtype=float)
    phi_B = np.asarray(phi_B, dtype=float)
    phi_S = np.asarray(phi_S, dtype=float)

    df_dphi_A = (
        np.log(np.maximum(phi_A, 1e-300)) / N_A
        + 1.0 / N_A
        + chi_AS * phi_S
        + chi_AB * phi_B
    )
    df_dphi_B = (
        np.log(np.maximum(phi_B, 1e-300)) / N_B
        + 1.0 / N_B
        + chi_BS * phi_S
        + chi_AB * phi_A
    )
    df_dphi_S = (
        np.log(np.maximum(phi_S, 1e-300)) / N_S
        + 1.0 / N_S
        + chi_AS * phi_A
        + chi_BS * phi_B
    )
    return df_dphi_A, df_dphi_B, df_dphi_S


def chemical_potentials_taylor(
    phi_A,
    phi_B,
    phi_S,
    N_A,
    N_B,
    N_S,
    chi_AB,
    chi_AS,
    chi_BS,
    expansion_point=0.2,
    order=2,
):
    """Chemical potentials using Taylor-expanded log() for small phi values.

    Uses log_taylor() to avoid divergence near phi = 0.
    This is an alternative implementation; Saylor is the active one in the paper.
    """
    phi_A = np.asarray(phi_A, dtype=float)
    phi_B = np.asarray(phi_B, dtype=float)
    phi_S = np.asarray(phi_S, dtype=float)

    log_A = log_taylor(phi_A, expansion_point, order)
    log_B = log_taylor(phi_B, expansion_point, order)
    log_S = log_taylor(phi_S, expansion_point, order)

    df_dphi_A = log_A / N_A + 1.0 / N_A + chi_AS * phi_S + chi_AB * phi_B
    df_dphi_B = log_B / N_B + 1.0 / N_B + chi_BS * phi_S + chi_AB * phi_A
    df_dphi_S = log_S / N_S + 1.0 / N_S + chi_AS * phi_A + chi_BS * phi_B
    return df_dphi_A, df_dphi_B, df_dphi_S


def fh_hessian_elements(phi_A, phi_B, phi_S, N_A, N_B, N_S, chi_AB, chi_AS, chi_BS):
    """Second derivatives of the FH free energy (2x2 Hessian elements).

    H_AA = d^2f/dphi_A^2 = 1/(phi_A*N_A) + 1/(phi_S*N_S) - 2*chi_AS
    H_BB = d^2f/dphi_B^2 = 1/(phi_B*N_B) + 1/(phi_S*N_S) - 2*chi_BS
    H_AB = d^2f/dphi_A dphi_B = 1/(phi_S*N_S) + chi_AB - chi_AS - chi_BS

    The 2x2 Hessian [[H_AA, H_AB],[H_AB, H_BB]] determines stability:
      positive definite -> stable (both eigenvalues > 0)
      one negative eigenvalue -> spinodal (indefinite)
      negative definite -> unstable
    """
    phi_A = np.asarray(phi_A, dtype=float)
    phi_B = np.asarray(phi_B, dtype=float)
    phi_S = np.asarray(phi_S, dtype=float)

    H_AA = 1.0 / (phi_A * N_A) + 1.0 / (phi_S * N_S) - 2.0 * chi_AS
    H_BB = 1.0 / (phi_B * N_B) + 1.0 / (phi_S * N_S) - 2.0 * chi_BS
    H_AB = 1.0 / (phi_S * N_S) + chi_AB - chi_AS - chi_BS
    # H_BA == H_AB (symmetric)
    return H_AA, H_BB, H_AB


def stability_label(phi_A, phi_B, phi_S, N_A, N_B, N_S, chi_AB, chi_AS, chi_BS):
    """Classify each composition point by Hessian eigenvalue analysis.

    Returns an array with values:
      0   -> stable (both eigenvalues positive, positive definite)
      0.5 -> spinodal boundary (one negative eigenvalue, indefinite)
      1   -> unstable (both eigenvalues negative, negative definite)
    """
    phi_A = np.asarray(phi_A, dtype=float)
    phi_B = np.asarray(phi_B, dtype=float)
    phi_S = np.asarray(phi_S, dtype=float)

    H_AA, H_BB, H_AB = fh_hessian_elements(
        phi_A, phi_B, phi_S, N_A, N_B, N_S, chi_AB, chi_AS, chi_BS
    )

    labels = np.zeros_like(phi_A)
    for i in range(len(phi_A.flat)):
        H = np.array([[H_AA.flat[i], H_AB.flat[i]], [H_AB.flat[i], H_BB.flat[i]]])
        eigvals = np.linalg.eigvalsh(H)
        n_pos = np.sum(eigvals > 0)
        if n_pos == 2:
            labels.flat[i] = 0  # stable: positive definite
        elif n_pos == 1:
            labels.flat[i] = 0.5  # indefinite (spinodal region boundary)
        else:
            labels.flat[i] = 1  # unstable: negative definite
    return labels
