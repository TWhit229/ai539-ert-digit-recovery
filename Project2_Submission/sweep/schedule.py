"""Sigma schedules for PnP-DM.

Two options:
  - geomspace (the current default): equal spacing in log(sigma)
  - poly_rho (EDM/DAPS++ style): polynomial in sigma^(1/rho).
    Positive rho spends most iters at HIGH noise.
    Negative rho spends most iters at LOW noise.
    DAPS++ found negative rho better for nonlinear inverse problems.
"""
import numpy as np


def geomspace_schedule(sigma_max=1.0, sigma_min=0.05, n_iter=80):
    return np.geomspace(sigma_max, sigma_min, n_iter)


def poly_rho_schedule(sigma_max=1.0, sigma_min=0.05, n_iter=80, rho=-7.0):
    """sigma_k = (sigma_max^(1/rho) + (k/K)(sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

    rho = +7  → EDM standard: more iters at HIGH noise.
    rho = -7  → DAPS++ recommendation: more iters at LOW noise (good for
                nonlinear inverse problems).
    """
    k = np.arange(n_iter)
    inv = 1.0 / rho
    a = sigma_max ** inv
    b = sigma_min ** inv
    return (a + (k / max(n_iter - 1, 1)) * (b - a)) ** rho


def get_schedule(name, n_iter=80, sigma_max=1.0, sigma_min=0.05):
    if name == "geomspace":
        return geomspace_schedule(sigma_max, sigma_min, n_iter)
    if name == "neg_rho":
        return poly_rho_schedule(sigma_max, sigma_min, n_iter, rho=-7.0)
    if name == "pos_rho":
        return poly_rho_schedule(sigma_max, sigma_min, n_iter, rho=+7.0)
    raise ValueError(f"unknown schedule: {name}")
