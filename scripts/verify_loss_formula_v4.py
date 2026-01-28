#!/usr/bin/env python
"""MVP-1.2d: Test if the issue is discrete-time approximation.

Test with different dt values to see if residual → 0 as dt → 0.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from core.sde_simulator import SDESimulator
from core.potentials import HarmonicPotential, GaussianInteraction
from core.nn_models import compute_pairwise_distances, compute_pairwise_diff


def harmonic_V(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    return 0.5 * k * (x ** 2).sum(dim=-1)

def harmonic_grad_V(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    return k * x

def harmonic_laplacian_V(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    d = x.shape[-1]
    return torch.full((x.shape[0],), k * d, device=x.device)

def gaussian_Phi(r: torch.Tensor, A: float = 1.0, sigma: float = 1.0) -> torch.Tensor:
    return A * torch.exp(-r ** 2 / (2 * sigma ** 2))

def gaussian_grad_Phi(r: torch.Tensor, A: float = 1.0, sigma: float = 1.0) -> torch.Tensor:
    return -A * r / (sigma ** 2) * torch.exp(-r ** 2 / (2 * sigma ** 2))

def gaussian_laplacian_Phi_1d(r: torch.Tensor, A: float = 1.0, sigma: float = 1.0) -> torch.Tensor:
    exp_term = torch.exp(-r ** 2 / (2 * sigma ** 2))
    return A / (sigma ** 2) * (r ** 2 / (sigma ** 2) - 1) * exp_term


def compute_drift_norm_sq(X: torch.Tensor, k: float, A: float, sigma_phi: float) -> torch.Tensor:
    N, d = X.shape

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

    return drift_norm_sq


def compute_laplacian_mean(X: torch.Tensor, k: float, A: float, sigma_phi: float) -> torch.Tensor:
    N, d = X.shape

    laplacian_V = harmonic_laplacian_V(X, k)

    distances = compute_pairwise_distances(X)
    laplacian_Phi = gaussian_laplacian_Phi_1d(distances, A, sigma_phi)

    mask = torch.eye(N, device=X.device, dtype=torch.bool)
    laplacian_Phi[mask] = 0

    laplacian_Phi_mean = laplacian_Phi.sum(dim=1) / N

    return (laplacian_V + laplacian_Phi_mean).mean()


def compute_energy(X: torch.Tensor, k: float, A: float, sigma_phi: float) -> torch.Tensor:
    N = X.shape[0]

    V_vals = harmonic_V(X, k)
    V_mean = V_vals.mean()

    distances = compute_pairwise_distances(X)
    Phi_vals = gaussian_Phi(distances, A, sigma_phi)

    mask = torch.eye(N, device=X.device, dtype=torch.bool)
    Phi_vals[mask] = 0

    Phi_mean = Phi_vals.sum() / (N * N)

    return V_mean + Phi_mean


def test_with_dt(dt_sim: float, dt_snap: float, seed: int = 42):
    """Test residual with specific dt values.

    dt_sim: SDE simulation step size
    dt_snap: Time between snapshots
    """
    N, d = 10, 1
    T = 2.0
    M = 200
    sigma_noise = 0.1
    k = 1.0
    A = 1.0
    sigma_phi = 1.0

    # Number of snapshots
    L = int(T / dt_snap)

    V = HarmonicPotential(k=k)
    Phi = GaussianInteraction(A=A, sigma=sigma_phi)
    simulator = SDESimulator(V=V, Phi=Phi, sigma=sigma_noise, dt=dt_sim)
    data, t_snapshots = simulator.simulate(N=N, d=d, T=T, L=L, M=M, seed=seed)

    data_torch = torch.tensor(data, dtype=torch.float32)
    t_torch = torch.tensor(t_snapshots, dtype=torch.float32)

    M_actual, L_actual, _, _ = data_torch.shape

    sigma_sq_half = sigma_noise ** 2 / 2

    # Test: J_diss - (σ²/2) J_lap + dE
    residuals = []
    for m in range(M_actual):
        for ell in range(L_actual - 1):
            X_curr = data_torch[m, ell]
            X_next = data_torch[m, ell + 1]
            actual_dt = (t_torch[ell + 1] - t_torch[ell]).item()

            J_diss = compute_drift_norm_sq(X_curr, k, A, sigma_phi).item() * actual_dt
            J_lap = compute_laplacian_mean(X_curr, k, A, sigma_phi).item() * actual_dt
            E_curr = compute_energy(X_curr, k, A, sigma_phi).item()
            E_next = compute_energy(X_next, k, A, sigma_phi).item()
            dE = E_next - E_curr

            residual = J_diss - sigma_sq_half * J_lap + dE
            residuals.append(residual)

    residuals = np.array(residuals)
    return {
        'dt_snap': dt_snap,
        'mean_residual': residuals.mean(),
        'std_residual': residuals.std(),
        'mean_abs_residual': np.abs(residuals).mean(),
    }


def main():
    print("="*80)
    print("MVP-1.2d: Test Discrete-Time Approximation Error")
    print("="*80)

    print("\nFormula: R = J_diss - (σ²/2) J_lap + dE")
    print("Testing if R → 0 as dt → 0")

    dt_sim = 0.001  # Fixed small simulation step

    print(f"\nSDE simulation dt = {dt_sim}")
    print("-"*80)
    print(f"{'dt_snap':>10} | {'Mean Residual':>15} | {'Std Residual':>15} | {'Mean |R|':>15}")
    print("-"*80)

    for dt_snap in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
        result = test_with_dt(dt_sim=dt_sim, dt_snap=dt_snap)
        print(f"{result['dt_snap']:>10.3f} | {result['mean_residual']:>15.6e} | {result['std_residual']:>15.6e} | {result['mean_abs_residual']:>15.6e}")

    print("-"*80)

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    print("""
If the formula is correct:
- Mean residual should → 0 as dt_snap → 0 (martingale averages out)
- Std residual should → 0 as dt_snap → 0 (less noise per step)

If mean residual stays constant or increases with smaller dt:
- The formula itself has issues (wrong coefficients or signs)
    """)


if __name__ == '__main__':
    main()
