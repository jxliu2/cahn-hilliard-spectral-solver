"""Utility functions: coordinate transforms, IO helpers, and numerics."""

import numpy as np
from scipy.signal import savgol_filter

# ternary triangle coordinate system
# (following Wolfram MathWorld TernaryDiagram convention)
# vertex layout:
#       S (0.5, sqrt(3)/2)
#      / \
#     /   \
#   B(0,0) - A(1,0)


def ternary_to_cartesian(phi_A, phi_B, phi_S=None):
    """Convert ternary compositions to 2D Cartesian coordinates.

    x = 1/2 - phi_A*cos(pi/3) + phi_B/2
    y = sqrt(3)/2 - phi_A*sin(pi/3) - phi_B*cot(pi/6)/2
    """
    phi_A = np.asarray(phi_A, dtype=float)
    phi_B = np.asarray(phi_B, dtype=float)
    x = 0.5 - phi_A * np.cos(np.pi / 3) + phi_B / 2
    y = np.sqrt(3) / 2 - phi_A * np.sin(np.pi / 3) - phi_B / np.tan(np.pi / 6) / 2
    return x, y


def cartesian_to_ternary(x, y):
    """Inverse of ternary_to_cartesian. Returns (phi_A, phi_B, phi_S)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # solve the linear system from ternary_to_cartesian
    # x = 1/2 - phi_A*cos(pi/3) + phi_B/2
    # y = sqrt(3)/2 - phi_A*sin(pi/3) - phi_B*cot(pi/6)/2
    # cot(pi/6) = cos(pi/6)/sin(pi/6) = sqrt(3)
    # => x = 1/2 - phi_A/2 + phi_B/2
    # => y = sqrt(3)/2 - phi_A*sqrt(3)/2 - phi_B*sqrt(3)/2
    # from y: phi_A + phi_B = 1 - 2*y/sqrt(3)  (note: 1 - phi_S)
    # from x: phi_B - phi_A = 2*x - 1
    phi_A = (1 - 2 * y / np.sqrt(3)) / 2 - (2 * x - 1) / 2
    phi_B = (1 - 2 * y / np.sqrt(3)) / 2 + (2 * x - 1) / 2
    phi_S = 1 - phi_A - phi_B
    return phi_A, phi_B, phi_S


def log_taylor(phi, expansion_point=0.2, order=2):
    """Taylor expansion of log(phi) around expansion_point for small phi.

    Uses Taylor series when phi < expansion_point to avoid log singularity,
    standard log otherwise. Port of local function log_taylor() in FH_CH_v40_3.m.

    expansion_point = 2e-1, order = 2 are the defaults from the MATLAB code.
    """
    phi = np.asarray(phi, dtype=float)
    result = np.log(np.maximum(phi, 1e-300))  # fallback for the standard branch
    # Taylor expansion: log(phi) ≈ log(x0) + (phi - x0)/x0 - (phi - x0)^2/(2*x0^2) + ...
    x0 = expansion_point
    mask = phi < expansion_point
    if np.any(mask):
        dp = phi[mask] - x0
        approx = np.log(x0) + dp / x0
        if order >= 2:
            approx -= dp**2 / (2 * x0**2)
        if order >= 3:
            approx += dp**3 / (3 * x0**3)
        result[mask] = approx
    return result


def ternary_rgb(phi_A, phi_B, phi_S):
    """Map ternary compositions to an RGB image array.

    R channel = phi_A, G channel = phi_B, B channel = phi_S.
    Clipped to [0, 1] per channel.
    """
    phi_A = np.clip(phi_A, 0, 1)
    phi_B = np.clip(phi_B, 0, 1)
    phi_S = np.clip(phi_S, 0, 1)
    rgb = np.stack([phi_A, phi_B, phi_S], axis=-1)
    return rgb


def save_checkpoint(path, state_dict):
    """Save simulation state to a compressed .npz file."""
    np.savez_compressed(path, **state_dict)


def load_checkpoint(path):
    """Load simulation state from a .npz file."""
    data = np.load(path, allow_pickle=True)
    return dict(data)


def smooth(y, window=11, polyorder=3):
    """Savitzky-Golay filter wrapper."""
    if len(y) < window:
        return y
    return savgol_filter(y, window_length=window, polyorder=polyorder)
