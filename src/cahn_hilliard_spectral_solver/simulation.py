"""Ternary Cahn-Hilliard spectral IMEX solver with Flory-Huggins free energy.

Port of FH_CH_v40_3.m. Solves coupled evolution equations for three components
(A, B, S) using an implicit-explicit Fourier spectral method with adaptive
time-stepping.

The IMEX scheme follows Mao (2019): the biharmonic stabilization term is treated
implicitly (giving a diagonal solve in k-space), while mobility and bulk free
energy terms are explicit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from omegaconf import DictConfig

from .free_energy import chemical_potentials
from .utils import load_checkpoint, save_checkpoint


@dataclass
class SimulationState:
    """All mutable state for a running simulation.

    Holds field arrays and scalar diagnostics so we can checkpoint/restart
    cleanly without a bunch of globals.
    """

    # composition fields (Nx x Ny)
    phi_A: np.ndarray
    phi_B: np.ndarray
    phi_S: np.ndarray

    # time
    t: float = 0.0
    dt: float = 1e-4
    iter: int = 1

    # ramping parameters
    chi_AS: float = 3.5
    chi_BS: float = 3.5

    # diagnostic lists (appended every maxmin_interval)
    t_list: list = field(default_factory=list)
    max_phi_A: list = field(default_factory=list)
    min_phi_A: list = field(default_factory=list)
    max_phi_B: list = field(default_factory=list)
    min_phi_B: list = field(default_factory=list)
    max_phi_S: list = field(default_factory=list)
    min_phi_S: list = field(default_factory=list)
    dt_t0_list: list = field(default_factory=list)
    amt_list_A: list = field(default_factory=list)
    amt_list_B: list = field(default_factory=list)
    amt_list_S: list = field(default_factory=list)
    chi_AS_list: list = field(default_factory=list)
    chi_BS_list: list = field(default_factory=list)
    chi_AB_list: list = field(default_factory=list)
    mean_noise_A: list = field(default_factory=list)
    mean_noise_B: list = field(default_factory=list)
    mean_noise_S: list = field(default_factory=list)


def _build_spectral_operators(Nx, Ny, L):
    """Build FFT wavenumber grids and spectral derivative operators.

    Reference: https://www.mathworks.com/matlabcentral/answers/41712
    Uses ifftshift to match MATLAB's convention of shifting zero-frequency
    to the expected position for forward/inverse FFT pairs.
    """
    dx = L / Nx
    dy = L / Ny

    nyq_kx = 1.0 / (2.0 * dx)
    nyq_ky = 1.0 / (2.0 * dy)
    dkx = 1.0 / (Nx * dx)
    dky = 1.0 / (Ny * dy)

    kx = np.arange(-nyq_kx, nyq_kx, dkx)  # shape (Nx,)
    ky = np.arange(-nyq_ky, nyq_ky, dky)  # shape (Ny,)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)

    KX, KY = np.meshgrid(kx, ky, indexing="ij")  # (Nx, Ny)

    laplacian_op = (2j * np.pi * KX) ** 2 + (2j * np.pi * KY) ** 2
    biharmonic_op = laplacian_op**2
    gradX_op = 2j * np.pi * KX
    gradY_op = 2j * np.pi * KY

    x = np.linspace(0, L, Nx, endpoint=False)
    y = np.linspace(0, L, Ny, endpoint=False)
    xgrid, ygrid = np.meshgrid(x, y, indexing="ij")  # (Nx, Ny)

    return KX, KY, laplacian_op, biharmonic_op, gradX_op, gradY_op, xgrid, ygrid


def _init_fields(cfg, rng, xgrid, ygrid):
    """Initialize composition fields phi_A, phi_B from config.

    Background uniform concentration + small random noise, then optionally
    a Gaussian nucleus (the active initialization in FH_CH_v40_3.m).
    """
    Nx, Ny = cfg.Nx, cfg.Ny
    L = cfg.L
    phi_A00 = cfg.phi_A00
    phi_B00 = cfg.phi_B00
    dphi_P = cfg.dphi_P

    # background field with small amplitude noise
    phi_A0_start = phi_A00 + rng.random((Nx, Ny)) * 2 * dphi_P - dphi_P
    phi_B0_start = phi_B00 + rng.random((Nx, Ny)) * 2 * dphi_P - dphi_P

    phi_A0 = phi_A0_start.copy()
    phi_B0 = phi_B0_start.copy()

    if cfg.use_nucleus:
        # GAUSSIAN put in initial nucleus of polymer-rich material
        nucl_phi = cfg.nucl_phi
        nucl_phi_A = nucl_phi - phi_A00
        nucl_phi_B = nucl_phi - phi_B00
        offset = cfg.offset  # offset between xcenters of A and B nuclei
        xcenterA = L / 2.0 + offset / 2.0
        ycenterA = L / 2.0
        xcenterB = L / 2.0 - offset / 2.0
        ycenterB = L / 2.0
        nucl_R = cfg.nucl_R  # nucleus radius, should be ~2x lambda
        sig_x = nucl_R
        sig_y = nucl_R
        z_A0 = nucl_phi_A * np.exp(
            -((xgrid - xcenterA) ** 2) / (2 * sig_x**2)
            - (ygrid - ycenterA) ** 2 / (2 * sig_y**2)
        )
        z_B0 = nucl_phi_B * np.exp(
            -((xgrid - xcenterB) ** 2) / (2 * sig_x**2)
            - (ygrid - ycenterB) ** 2 / (2 * sig_y**2)
        )
        phi_A0 = phi_A0_start + z_A0
        phi_B0 = phi_B0_start + z_B0

    phi_S0 = 1.0 - phi_A0 - phi_B0
    return phi_A0, phi_B0, phi_S0


def _mobility_tensor(phi_A, phi_B, phi_S, D_A, D_B, D_S, vit, averaging="geometric"):
    """Compute the composition-dependent effective diffusivity and mobility matrix.

    Three averaging schemes from FH_CH_v40_3.m:
      geometric (default): D_eff = D_A^phi_A * D_S^phi_S * D_B^phi_B
      arithmetic:          D_eff = D_A*phi_A + D_S*phi_S + D_B*phi_B
      harmonic:            D_eff = 1/(phi_A/D_A + phi_S/D_S + phi_B/D_B)

    The off-diagonal mobility terms follow Mao's ternary formalism:
      Dmob_ij = D_eff * phi_i * (-phi_j)  for i != j
      Dmob_ii = D_eff * phi_i * (1 - phi_i)
    """
    if averaging == "geometric":
        D_array = D_A**phi_A * D_S**phi_S * D_B**phi_B
    elif averaging == "arithmetic":
        D_array = D_A * phi_A + D_S * phi_S + D_B * phi_B
    elif averaging == "harmonic":
        D_array = 1.0 / (phi_A / D_A + phi_S / D_S + phi_B / D_B)
    else:
        raise ValueError(f"Unknown mobility averaging: {averaging}")

    D_array = vit * D_array  # apply vitrification factor (ones if not vitrified)

    # diagonal mobility terms: M_ii = D_eff * phi_i * (1 - phi_i)
    Dmob_AA = D_array * phi_A * (1.0 - phi_A)
    Dmob_BB = D_array * phi_B * (1.0 - phi_B)
    Dmob_SS = D_array * phi_S * (1.0 - phi_S)

    # off-diagonal: M_ij = D_eff * phi_i * (-phi_j)
    Dmob_AS = D_array * phi_A * (-phi_S)
    Dmob_SA = D_array * phi_S * (-phi_A)
    Dmob_AB = D_array * phi_A * (-phi_B)
    Dmob_BA = D_array * phi_B * (-phi_A)
    Dmob_BS = D_array * phi_B * (-phi_S)
    Dmob_SB = D_array * phi_S * (-phi_B)

    return (
        D_array,
        Dmob_AA,
        Dmob_BB,
        Dmob_SS,
        Dmob_AS,
        Dmob_SA,
        Dmob_AB,
        Dmob_BA,
        Dmob_BS,
        Dmob_SB,
    )


def _timestep(
    phi_A,
    phi_B,
    phi_S,
    D_A,
    D_B,
    D_S,
    vit,
    chi_AB,
    chi_AS,
    chi_BS,
    N_A,
    N_B,
    N_S,
    kappa_AS,
    kappa_BS,
    kappa_AB,
    A_stab,
    mean_lambda,
    laplacian_op,
    biharmonic_op,
    gradX_op,
    gradY_op,
    dt,
    langevin_noise_ampl,
    rng,
    averaging,
):
    """Execute one IMEX time step for the ternary Cahn-Hilliard equations.

    The IMEX scheme:
      - biharmonic stabilizer A*Dmean*lambda^2*del4(phi) is treated implicitly
        (diagonal in k-space, so it's just a pointwise division)
      - all other terms (bulk free energy, gradient energy, mobility cross-coupling)
        are treated explicitly

    Returns updated (phi_A_new, phi_B_new, phi_S_new, Dmob_AA, D_array).
    """
    (
        D_array,
        Dmob_AA,
        Dmob_BB,
        Dmob_SS,
        Dmob_AS,
        Dmob_SA,
        Dmob_AB,
        Dmob_BA,
        Dmob_BS,
        Dmob_SB,
    ) = _mobility_tensor(phi_A, phi_B, phi_S, D_A, D_B, D_S, vit, averaging)

    # evaluate df/dphi_i (homogeneous, Saylor fix active)
    df_dphi_A, df_dphi_B, df_dphi_S = chemical_potentials(
        phi_A, phi_B, phi_S, N_A, N_B, N_S, chi_AB, chi_AS, chi_BS
    )

    # setup reciprocal space for ternary system
    phi_A_k = np.fft.fft2(phi_A)
    phi_B_k = np.fft.fft2(phi_B)
    phi_S_k = np.fft.fft2(phi_S)
    df_dphi_A_k = np.fft.fft2(df_dphi_A)
    df_dphi_B_k = np.fft.fft2(df_dphi_B)
    df_dphi_S_k = np.fft.fft2(df_dphi_S)

    # TERNARY reciprocal space IMEX with FW Euler, stabilized -A(kappa)del4(c)
    # interfacial energy (gradient energy) terms in k-space
    # kappa factors = lambda^2 * chi, penalize sharp composition gradients
    interfacial_A_k = (
        kappa_AS * laplacian_op * phi_S_k / 2.0
        + kappa_AB * laplacian_op * phi_B_k / 2.0
    )
    interfacial_S_k = (
        kappa_AS * laplacian_op * phi_A_k / 2.0
        + kappa_BS * laplacian_op * phi_B_k / 2.0
    )
    interfacial_B_k = (
        kappa_BS * laplacian_op * phi_S_k / 2.0
        + kappa_AB * laplacian_op * phi_A_k / 2.0
    )

    # total chemical potential in k-space (bulk + interfacial)
    muA_k = df_dphi_A_k + interfacial_A_k
    muS_k = df_dphi_S_k + interfacial_S_k
    muB_k = df_dphi_B_k + interfacial_B_k

    # gradient of chemical potential (back to real space for pointwise mobility mult)
    grad_muA_x = np.real(np.fft.ifft2(gradX_op * muA_k))
    grad_muA_y = np.real(np.fft.ifft2(gradY_op * muA_k))
    grad_muS_x = np.real(np.fft.ifft2(gradX_op * muS_k))
    grad_muS_y = np.real(np.fft.ifft2(gradY_op * muS_k))
    grad_muB_x = np.real(np.fft.ifft2(gradX_op * muB_k))
    grad_muB_y = np.real(np.fft.ifft2(gradY_op * muB_k))

    # mobility-weighted fluxes (real space multiplication, then back to k-space)
    # gamma is everything inside the final gradient operator
    def _to_k(arr_x, arr_y):
        return np.fft.fft2(arr_x), np.fft.fft2(arr_y)

    AA_x_k, AA_y_k = _to_k(Dmob_AA * grad_muA_x, Dmob_AA * grad_muA_y)
    AS_x_k, AS_y_k = _to_k(Dmob_AS * grad_muS_x, Dmob_AS * grad_muS_y)
    AB_x_k, AB_y_k = _to_k(Dmob_AB * grad_muB_x, Dmob_AB * grad_muB_y)

    SS_x_k, SS_y_k = _to_k(Dmob_SS * grad_muS_x, Dmob_SS * grad_muS_y)
    SA_x_k, SA_y_k = _to_k(Dmob_SA * grad_muA_x, Dmob_SA * grad_muA_y)
    SB_x_k, SB_y_k = _to_k(Dmob_SB * grad_muB_x, Dmob_SB * grad_muB_y)

    BB_x_k, BB_y_k = _to_k(Dmob_BB * grad_muB_x, Dmob_BB * grad_muB_y)
    BA_x_k, BA_y_k = _to_k(Dmob_BA * grad_muA_x, Dmob_BA * grad_muA_y)
    BS_x_k, BS_y_k = _to_k(Dmob_BS * grad_muS_x, Dmob_BS * grad_muS_y)

    # aggregate flux gamma = sum of all mobility cross-coupling terms
    gamma_A_x_k = AA_x_k + AS_x_k + AB_x_k
    gamma_A_y_k = AA_y_k + AS_y_k + AB_y_k
    gamma_S_x_k = SS_x_k + SA_x_k + SB_x_k
    gamma_S_y_k = SS_y_k + SA_y_k + SB_y_k
    gamma_B_x_k = BB_x_k + BA_x_k + BS_x_k
    gamma_B_y_k = BB_y_k + BA_y_k + BS_y_k

    # divergence of flux in k-space (grad . gamma)
    grad_gamma_A_k = gradX_op * gamma_A_x_k + gradY_op * gamma_A_y_k
    grad_gamma_S_k = gradX_op * gamma_S_x_k + gradY_op * gamma_S_y_k
    grad_gamma_B_k = gradX_op * gamma_B_x_k + gradY_op * gamma_B_y_k

    Dmean = D_array.mean()

    # generate ternary noise term
    if langevin_noise_ampl == 0:
        noise_A_k = 0.0
        noise_B_k = 0.0
        noise_S_k = 0.0
        noise_A = np.zeros_like(phi_A)
        noise_B = np.zeros_like(phi_B)
        noise_S = np.zeros_like(phi_S)
    else:
        # mobility(composition)-dependent noise amplitude: 4*phi*(1-phi)*xi
        noise_A = (
            4
            * phi_A
            * (1.0 - phi_A)
            * rng.standard_normal(phi_A.shape)
            * langevin_noise_ampl
        )
        noise_B = (
            4
            * phi_B
            * (1.0 - phi_B)
            * rng.standard_normal(phi_B.shape)
            * langevin_noise_ampl
        )
        noise_S = (
            4
            * phi_S
            * (1.0 - phi_S)
            * rng.standard_normal(phi_S.shape)
            * langevin_noise_ampl
        )
        noise_A_k = np.fft.fft2(noise_A)
        noise_B_k = np.fft.fft2(noise_B)
        noise_S_k = np.fft.fft2(noise_S)

    # IMEX implicit solve
    # N_i = noise + A*Dmean*lambda^2*del4(phi) + grad.gamma
    # rhs_i = (phi_i + dt*N_i) / (1 + dt*A*Dmean*lambda^2*del4)
    # the denominator is diagonal in k-space -> just divide elementwise
    stab = A_stab * Dmean * mean_lambda**2 * biharmonic_op
    denom = 1.0 + dt * stab

    N_A_k = noise_A_k + stab * phi_A_k + grad_gamma_A_k
    N_S_k = noise_S_k + stab * phi_S_k + grad_gamma_S_k
    N_B_k = noise_B_k + stab * phi_B_k + grad_gamma_B_k

    rhs_A = np.real(np.fft.ifft2((phi_A_k + dt * N_A_k) / denom))
    rhs_S = np.real(np.fft.ifft2((phi_S_k + dt * N_S_k) / denom))
    rhs_B = np.real(np.fft.ifft2((phi_B_k + dt * N_B_k) / denom))

    return rhs_A, rhs_B, rhs_S, Dmob_AA, D_array, noise_A, noise_B, noise_S


def run_simulation(cfg: DictConfig) -> None:
    """Main simulation driver. Port of the while-loop in FH_CH_v40_3.m."""

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    # setup grid and spectral operators
    Nx, Ny, L = cfg.Nx, cfg.Ny, cfg.L
    KX, KY, laplacian_op, biharmonic_op, gradX_op, gradY_op, xgrid, ygrid = (
        _build_spectral_operators(Nx, Ny, L)
    )

    # polymer / solvent parameters
    N_A, N_B, N_S = cfg.N_A, cfg.N_B, cfg.N_S

    chi_AB = cfg.chi_AB
    chi_AS = cfg.chi_AS  # starting chi_AS (= chi_PS)
    chi_BS = cfg.chi_BS  # starting chi_BS
    chi_AS_final = cfg.chi_AS_final

    # lambda here is equivalent to Rg; following Mao's definition NOT Kessler's lambda
    lambda_AS = cfg.lambda_AS
    lambda_BS = cfg.lambda_BS
    lambda_AB = cfg.lambda_AB
    mean_lambda = np.mean([lambda_AS, lambda_BS, lambda_AB])

    # kappa_iS = lambda^2 * chi: relates to interfacial tension
    kappa_AS = lambda_AS**2 * chi_AS
    kappa_BS = lambda_BS**2 * chi_BS
    kappa_AB = lambda_AB**2 * chi_AB

    # A = 0.5*mean([chi_AS, chi_BS, chi_AB]) stabilizer coefficient
    A_stab = 0.5 * np.mean([chi_AS, chi_BS, chi_AB])

    # time parameters: t0 = min(lambda)^2 / max(D), reference diffusion timescale
    D_A, D_B, D_S = cfg.D_A, cfg.D_B, cfg.D_S
    t0 = min(lambda_AS, lambda_BS, lambda_AB) ** 2 / max(D_A, D_B, D_S)
    dt0 = t0 * cfg.dt0_factor

    chi_PS_slope = (chi_AS_final - chi_AS) / (t0 * cfg.t0_ramp_number)
    # (chi_BS tracks chi_AS identically unless separately specified)

    plot_interval = t0 * cfg.plot_interval_t0
    maxmin_interval = plot_interval / 100.0
    tictoc_interval = plot_interval / 4.0
    save_interval = t0 * cfg.save_interval_t0

    langevin_noise_ampl = cfg.langevin_noise_ampl
    averaging = cfg.mobility_averaging

    # check for continuation run (load last checkpoint if present)
    checkpoints = sorted(out_dir.glob("checkpoint_*.npz"))
    continuation_run = len(checkpoints) > 0

    if continuation_run:
        print(f"Continuation run: loading {checkpoints[-1]}")
        ckpt = load_checkpoint(checkpoints[-1])
        phi_A = ckpt["phi_A"]
        phi_B = ckpt["phi_B"]
        phi_S = ckpt["phi_S"]
        t = float(ckpt["t"])
        dt = float(ckpt["dt"])
        iter_start = int(ckpt["iter"])
        chi_AS = float(ckpt.get("chi_AS", chi_AS))
        chi_BS = float(ckpt.get("chi_BS", chi_BS))
        # restore diagnostic lists
        t_list = list(ckpt.get("t_list", []))
        max_phi_A_tern = list(ckpt.get("max_phi_A", []))
        min_phi_A_tern = list(ckpt.get("min_phi_A", []))
        max_phi_B_tern = list(ckpt.get("max_phi_B", []))
        min_phi_B_tern = list(ckpt.get("min_phi_B", []))
        max_phi_S_tern = list(ckpt.get("max_phi_S", []))
        min_phi_S_tern = list(ckpt.get("min_phi_S", []))
        dt_t0_list = list(ckpt.get("dt_t0_list", []))
        amt_list_A = list(ckpt.get("amt_list_A", []))
        amt_list_B = list(ckpt.get("amt_list_B", []))
        amt_list_S = list(ckpt.get("amt_list_S", []))
        chi_AS_list = list(ckpt.get("chi_AS_list", []))
        chi_BS_list = list(ckpt.get("chi_BS_list", []))
        chi_AB_list = list(ckpt.get("chi_AB_list", []))
        mean_noise_A = list(ckpt.get("mean_noise_A", []))
        mean_noise_B = list(ckpt.get("mean_noise_B", []))
        mean_noise_S = list(ckpt.get("mean_noise_S", []))
    else:
        phi_A, phi_B, phi_S = _init_fields(cfg, rng, xgrid, ygrid)
        t = 0.0
        dt = dt0
        iter_start = 1
        t_list = []
        max_phi_A_tern = []
        min_phi_A_tern = []
        max_phi_B_tern = []
        min_phi_B_tern = []
        max_phi_S_tern = []
        min_phi_S_tern = []
        dt_t0_list = []
        amt_list_A = []
        amt_list_B = []
        amt_list_S = []
        chi_AS_list = []
        chi_BS_list = []
        chi_AB_list = []
        mean_noise_A = []
        mean_noise_B = []
        mean_noise_S = []

    # initial amounts of each component (used to compute composition redistribution)
    starting_amt = np.array([phi_A.mean(), phi_B.mean(), phi_S.mean()])

    # vitrification factor (ones = no vitrification; see commented block in original)
    vit = np.ones((Nx, Ny))

    tstart = time.time()
    iter_ = iter_start
    plot_interval_counter = max(1, int(t / plot_interval))
    maxmin_interval_counter = max(1, int(t / maxmin_interval))
    tictoc_interval_counter = max(1, int(t / tictoc_interval))
    save_interval_counter = max(1, int(t / save_interval))
    frame = len(checkpoints) + 1

    print("\n=== Cahn-Hilliard simulation ===")
    print(f"N_A={N_A}, N_B={N_B}, N_S={N_S}")
    print(f"chi_AB={chi_AB}, chi_AS={chi_AS:.3f} -> {chi_AS_final:.3f}")
    print(f"phi_A00={cfg.phi_A00:.4f}, phi_B00={cfg.phi_B00:.4f}")
    print(f"Grid: {Nx}x{Ny}, L={L}, t0={t0:.4e}, dt0={dt0:.4e}")
    print(f"Output: {out_dir}\n")

    run = True

    while run:
        if t > tictoc_interval * tictoc_interval_counter:
            tictoc_interval_counter += 1
            comp_redist_A = starting_amt[0] / phi_A.mean() if phi_A.mean() > 0 else 1.0
            comp_redist_B = starting_amt[1] / phi_B.mean() if phi_B.mean() > 0 else 1.0
            comp_redist_S = starting_amt[2] / phi_S.mean() if phi_S.mean() > 0 else 1.0
            print(
                f"\nt/t0: {t / t0:.2f}    dt/t0 = {dt / t0:.4e}    "
                f"wall: {time.time() - tstart:.1f}s"
            )
            mA, mB, mS = phi_A.mean(), phi_B.mean(), phi_S.mean()
            print(f"  Amounts A, B, S:  {mA:.4f}  {mB:.4f}  {mS:.4f}")
            print(
                f"  Min A, B, S:      "
                f"{phi_A.min():.4f}  {phi_B.min():.4f}  {phi_S.min():.4f}"
            )
            print(
                f"  Max A, B, S:      "
                f"{phi_A.max():.4f}  {phi_B.max():.4f}  {phi_S.max():.4f}"
            )
            print(
                f"  Composition redist:  "
                f"{comp_redist_A:.4f}  {comp_redist_B:.4f}  {comp_redist_S:.4f}"
            )
            print(f"  chi_AS={chi_AS:.4f}, chi_BS={chi_BS:.4f}, chi_AB={chi_AB:.4f}")

        # ramp chi_PS (solvent-polymer interaction increases over time)
        if chi_AS < chi_AS_final:
            chi_AS = chi_AS + dt * chi_PS_slope
            chi_BS = chi_AS  # chi_BS tracks chi_AS
            kappa_AS = lambda_AS**2 * chi_AS
            kappa_BS = lambda_BS**2 * chi_BS
            mean_lambda = np.mean([lambda_AS, lambda_BS, lambda_AB])
            A_stab = 0.5 * np.mean([chi_AS, chi_BS, chi_AB])

        rhs_A, rhs_B, rhs_S, Dmob_AA, D_array, noise_A, noise_B, noise_S = _timestep(
            phi_A,
            phi_B,
            phi_S,
            D_A,
            D_B,
            D_S,
            vit,
            chi_AB,
            chi_AS,
            chi_BS,
            N_A,
            N_B,
            N_S,
            kappa_AS,
            kappa_BS,
            kappa_AB,
            A_stab,
            mean_lambda,
            laplacian_op,
            biharmonic_op,
            gradX_op,
            gradY_op,
            dt,
            langevin_noise_ampl,
            rng,
            averaging,
        )

        phi_A2 = rhs_A
        phi_B2 = rhs_B
        phi_S2 = rhs_S

        # check simulation crash conditions
        if (
            np.any(np.isnan(phi_A2))
            or np.any(np.isnan(phi_B2))
            or np.any(np.isnan(phi_S2))
        ):
            print("phi NaN crash — stopping simulation")
            break

        # adaptive time step if any phis go below 0 or above 1
        dt_old = dt
        if np.any(phi_A2 < 0) or np.any(phi_B2 < 0) or np.any(phi_S2 < 0):
            print(
                f"  # gonegs at t/t0={t / t0:.2f}, "
                f"reducing dt: {dt:.4e} -> {dt * 0.1:.4e}"
            )
            dt = dt * 0.1
            continue  # retry current step with smaller dt, don't advance time

        elif np.any(phi_A2 > 1) or np.any(phi_B2 > 1) or np.any(phi_S2 > 1):
            print(
                f"  # goouts at t/t0={t / t0:.2f}, "
                f"reducing dt: {dt:.4e} -> {dt * 0.1:.4e}"
            )
            dt = dt * 0.1
            continue

        elif (
            dt < 0.5 * t0
            and np.all(phi_A2 > 0)
            and np.all(phi_B2 > 0)
            and np.all(phi_S2 > 0)
            and np.all(phi_A2 < 1)
            and np.all(phi_B2 < 1)
            and np.all(phi_S2 < 1)
        ):
            dt = dt * 1.2  # gradually increase dt when stable

        # enforce phi's to sum to 1
        # phi gets redistributed; this is still necessary with adaptive timestep
        phi_tot = phi_A2 + phi_B2 + phi_S2
        phi_A2 = phi_A2 / phi_tot
        phi_B2 = phi_B2 / phi_tot
        phi_S2 = phi_S2 / phi_tot

        phi_A = phi_A2
        phi_B = phi_B2
        phi_S = phi_S2

        # save running diagnostic parameters (maxmin)
        if t > maxmin_interval * maxmin_interval_counter:
            maxmin_interval_counter += 1
            t_list.append(t / t0)
            max_phi_A_tern.append(phi_A.max())
            min_phi_A_tern.append(phi_A.min())
            max_phi_B_tern.append(phi_B.max())
            min_phi_B_tern.append(phi_B.min())
            max_phi_S_tern.append(phi_S.max())
            min_phi_S_tern.append(phi_S.min())
            dt_t0_list.append(dt / t0)
            amt_list_A.append(phi_A.mean())
            amt_list_B.append(phi_B.mean())
            amt_list_S.append(phi_S.mean())
            chi_AS_list.append(chi_AS)
            chi_BS_list.append(chi_BS)
            chi_AB_list.append(chi_AB)
            mean_noise_A.append(float(np.mean(noise_A)))
            mean_noise_B.append(float(np.mean(noise_B)))
            mean_noise_S.append(float(np.mean(noise_S)))

        if t > save_interval * save_interval_counter:
            save_interval_counter += 1
            ckpt_path = out_dir / f"checkpoint_{frame:05d}.npz"
            save_checkpoint(
                str(ckpt_path),
                dict(
                    phi_A=phi_A,
                    phi_B=phi_B,
                    phi_S=phi_S,
                    t=t,
                    dt=dt,
                    iter=iter_,
                    chi_AS=chi_AS,
                    chi_BS=chi_BS,
                    t_list=np.array(t_list),
                    max_phi_A=np.array(max_phi_A_tern),
                    min_phi_A=np.array(min_phi_A_tern),
                    max_phi_B=np.array(max_phi_B_tern),
                    min_phi_B=np.array(min_phi_B_tern),
                    max_phi_S=np.array(max_phi_S_tern),
                    min_phi_S=np.array(min_phi_S_tern),
                    dt_t0_list=np.array(dt_t0_list),
                    amt_list_A=np.array(amt_list_A),
                    amt_list_B=np.array(amt_list_B),
                    amt_list_S=np.array(amt_list_S),
                    chi_AS_list=np.array(chi_AS_list),
                    chi_BS_list=np.array(chi_BS_list),
                    chi_AB_list=np.array(chi_AB_list),
                    mean_noise_A=np.array(mean_noise_A),
                    mean_noise_B=np.array(mean_noise_B),
                    mean_noise_S=np.array(mean_noise_S),
                ),
            )
            frame += 1

        if t > plot_interval * plot_interval_counter:
            plot_interval_counter += 1
            from .figures import plot_snapshot

            fig_path = out_dir / f"snapshot_{frame:05d}.png"
            plot_snapshot(
                phi_A,
                phi_B,
                phi_S,
                t,
                t0,
                t_list,
                max_phi_A_tern,
                min_phi_A_tern,
                max_phi_B_tern,
                min_phi_B_tern,
                max_phi_S_tern,
                min_phi_S_tern,
                amt_list_A,
                amt_list_B,
                amt_list_S,
                chi_AS_list,
                dt_t0_list,
                xgrid,
                ygrid,
                Dmob_AA,
                cfg,
                str(fig_path),
            )

        t = t + dt_old
        iter_ += 1

        t_end = getattr(cfg, "t_end", None)
        if t_end is not None and t >= t_end * t0:
            print(f"Reached t_end={t_end} t0 at t/t0={t / t0:.2f}")
            break
