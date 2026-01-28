#!/usr/bin/env python
"""Quick verification with smaller data."""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from core.sde_simulator import SDESimulator
from core.potentials import HarmonicPotential, GaussianInteraction
from core.nn_models import compute_pairwise_distances, compute_pairwise_diff


def harmonic_V(x, k=1.0):
    return 0.5 * k * (x ** 2).sum(dim=-1)

def harmonic_grad_V(x, k=1.0):
    return k * x

def harmonic_laplacian_V(x, k=1.0):
    d = x.shape[-1]
    return torch.full((x.shape[0],), k * d, device=x.device)

def gaussian_Phi(r, A=1.0, sigma=1.0):
    return A * torch.exp(-r ** 2 / (2 * sigma ** 2))

def gaussian_grad_Phi(r, A=1.0, sigma=1.0):
    return -A * r / (sigma ** 2) * torch.exp(-r ** 2 / (2 * sigma ** 2))

def gaussian_laplacian_Phi_1d(r, A=1.0, sigma=1.0):
    exp_term = torch.exp(-r ** 2 / (2 * sigma ** 2))
    return A / (sigma ** 2) * (r ** 2 / (sigma ** 2) - 1) * exp_term


def compute_terms(X, k, A, sigma_phi):
    N, d = X.shape

    # Drift norm squared
    grad_V = harmonic_grad_V(X, k)
    diff = compute_pairwise_diff(X)
    distances = torch.norm(diff, dim=-1)
    distances_safe = distances.clone()
    distances_safe[distances_safe < 1e-10] = 1e-10

    dPhi_dr = gaussian_grad_Phi(distances, A, sigma_phi)
    unit_diff = diff / distances_safe.unsqueeze(-1)
    grad_Phi_pairs = dPhi_dr.unsqueeze(-1) * unit_diff

    mask = torch.eye(N, device=X.device, dtype=torch.bool)
    grad_Phi_pairs[mask] = 0

    grad_Phi_mean = grad_Phi_pairs.sum(dim=1) / N
    force = grad_V + grad_Phi_mean
    drift_norm_sq = (force ** 2).sum(dim=-1).mean()

    # Laplacian mean
    laplacian_V = harmonic_laplacian_V(X, k)
    laplacian_Phi = gaussian_laplacian_Phi_1d(distances, A, sigma_phi)
    laplacian_Phi[mask] = 0
    laplacian_Phi_mean = laplacian_Phi.sum(dim=1) / N
    laplacian_mean = (laplacian_V + laplacian_Phi_mean).mean()

    # Energy
    V_vals = harmonic_V(X, k)
    V_mean = V_vals.mean()
    Phi_vals = gaussian_Phi(distances, A, sigma_phi)
    Phi_vals[mask] = 0
    Phi_mean = Phi_vals.sum() / (N * N)
    energy = V_mean + Phi_mean

    return drift_norm_sq.item(), laplacian_mean.item(), energy.item()


def test_with_dt(dt_snap, M=50, seed=42):
    N, d = 10, 1
    T = 1.0
    sigma_noise = 0.1
    k, A, sigma_phi = 1.0, 1.0, 1.0

    L = max(int(T / dt_snap), 2)

    V = HarmonicPotential(k=k)
    Phi = GaussianInteraction(A=A, sigma=sigma_phi)
    simulator = SDESimulator(V=V, Phi=Phi, sigma=sigma_noise, dt=0.001)
    data, t_snapshots = simulator.simulate(N=N, d=d, T=T, L=L, M=M, seed=seed)

    data_torch = torch.tensor(data, dtype=torch.float32)
    t_torch = torch.tensor(t_snapshots, dtype=torch.float32)

    sigma_sq_half = sigma_noise ** 2 / 2

    residuals = []
    for m in range(data_torch.shape[0]):
        for ell in range(data_torch.shape[1] - 1):
            X_curr = data_torch[m, ell]
            X_next = data_torch[m, ell + 1]
            actual_dt = (t_torch[ell + 1] - t_torch[ell]).item()

            diss, lap, E_curr = compute_terms(X_curr, k, A, sigma_phi)
            _, _, E_next = compute_terms(X_next, k, A, sigma_phi)

            J_diss = diss * actual_dt
            J_lap = lap * actual_dt
            dE = E_next - E_curr

            # Formula: J_diss - (σ²/2) J_lap + dE
            residual = J_diss - sigma_sq_half * J_lap + dE
            residuals.append(residual)

    residuals = np.array(residuals)
    return residuals.mean(), residuals.std()


def main():
    print("Testing formula: R = J_diss - (σ²/2) J_lap + dE")
    print("-" * 60)
    print(f"{'dt_snap':>10} | {'Mean R':>15} | {'Std R':>15}")
    print("-" * 60)

    for dt in [0.2, 0.1, 0.05, 0.02]:
        mean_r, std_r = test_with_dt(dt)
        print(f"{dt:>10.2f} | {mean_r:>15.6e} | {std_r:>15.6e}")

    print("-" * 60)
    print("\nIf formula is correct: Mean R should → 0 as dt → 0")


if __name__ == '__main__':
    main()
