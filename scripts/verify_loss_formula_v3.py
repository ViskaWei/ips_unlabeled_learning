#!/usr/bin/env python
"""MVP-1.2c: Verify the CORRECT weak-form formula.

From Ito's lemma for SDE: dX = b dt + σ dW
d⟨f, μ⟩ = ⟨∇f · b, μ⟩ dt + (σ²/2) ⟨Δf, μ⟩ dt + martingale

For f = V + Φ*μ (self-test function), b = -∇(V + Φ*μ):
∇f · b = -|∇(V + Φ*μ)|²

So:
dE = -J_diss dt + (σ²/2) J_lap dt + martingale

Rearranging:
J_diss dt - (σ²/2) J_lap dt + dE = martingale ≈ 0 in expectation

The CORRECT formula is:
R = J_diss - (σ²/2) J_lap + dE = 0 (in expectation)

Note the MINUS sign before the Laplacian term!
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


def test_correct_formula(
    data: np.ndarray,
    t_snapshots: np.ndarray,
    sigma_noise: float,
    k: float = 1.0,
    A: float = 1.0,
    sigma_phi: float = 1.0,
):
    """Test the CORRECT formula: J_diss - (σ²/2) J_lap + dE = 0"""
    data_torch = torch.tensor(data, dtype=torch.float32)
    t_torch = torch.tensor(t_snapshots, dtype=torch.float32)

    M, L, N, d = data_torch.shape

    all_residuals_correct = []
    all_residuals_paper = []

    sigma_sq_half = sigma_noise ** 2 / 2

    for m in range(M):
        for ell in range(L - 1):
            X_curr = data_torch[m, ell]
            X_next = data_torch[m, ell + 1]
            dt = (t_torch[ell + 1] - t_torch[ell]).item()

            J_diss = compute_drift_norm_sq(X_curr, k, A, sigma_phi).item() * dt
            J_lap = compute_laplacian_mean(X_curr, k, A, sigma_phi).item() * dt
            E_curr = compute_energy(X_curr, k, A, sigma_phi).item()
            E_next = compute_energy(X_next, k, A, sigma_phi).item()
            dE = E_next - E_curr

            # CORRECT formula: J_diss - (σ²/2) J_lap + dE = 0
            residual_correct = J_diss - sigma_sq_half * J_lap + dE

            # Paper formula: J_diss + σ J_lap - 2 dE = 0
            residual_paper = J_diss + sigma_noise * J_lap - 2 * dE

            all_residuals_correct.append(residual_correct)
            all_residuals_paper.append(residual_paper)

    all_residuals_correct = np.array(all_residuals_correct)
    all_residuals_paper = np.array(all_residuals_paper)

    print("="*80)
    print("TESTING CORRECT FORMULA vs PAPER FORMULA")
    print("="*80)

    print("\nCORRECT formula (from Ito's lemma):")
    print("  R = J_diss - (σ²/2) J_lap + dE")
    print(f"  Mean residual: {all_residuals_correct.mean():.6e}")
    print(f"  Std residual:  {all_residuals_correct.std():.6e}")
    print(f"  |Mean|/Std:    {abs(all_residuals_correct.mean())/all_residuals_correct.std():.4f}")

    print("\nPAPER formula:")
    print("  R = J_diss + σ J_lap - 2 dE")
    print(f"  Mean residual: {all_residuals_paper.mean():.6e}")
    print(f"  Std residual:  {all_residuals_paper.std():.6e}")
    print(f"  |Mean|/Std:    {abs(all_residuals_paper.mean())/all_residuals_paper.std():.4f}")

    # Statistical test: is mean significantly different from 0?
    from scipy import stats

    t_stat_correct, p_val_correct = stats.ttest_1samp(all_residuals_correct, 0)
    t_stat_paper, p_val_paper = stats.ttest_1samp(all_residuals_paper, 0)

    print("\nStatistical test (H0: mean = 0):")
    print(f"  CORRECT: t={t_stat_correct:.4f}, p-value={p_val_correct:.6e}")
    print(f"  PAPER:   t={t_stat_paper:.4f}, p-value={p_val_paper:.6e}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if p_val_correct > 0.05:
        print("✅ CORRECT formula: Mean residual is NOT significantly different from 0")
        print("   Formula is VERIFIED!")
    else:
        print("⚠️ CORRECT formula: Mean residual is significantly different from 0")
        print("   (But may be due to discrete-time approximation error)")

    if p_val_paper > 0.05:
        print("✅ PAPER formula: Mean residual is NOT significantly different from 0")
    else:
        print("❌ PAPER formula: Mean residual IS significantly different from 0")
        print("   Paper formula may have errors!")

    # Try more variations
    print("\n" + "="*80)
    print("SYSTEMATIC SEARCH FOR CORRECT COEFFICIENTS")
    print("="*80)

    best_formula = None
    best_t_stat = float('inf')

    for diff_sign in [-1, +1]:
        for diff_coef in [sigma_noise, sigma_noise**2, sigma_noise**2/2]:
            for energy_coef in [1, 2]:
                for energy_sign in [-1, +1]:
                    residuals = np.array([
                        compute_drift_norm_sq(data_torch[m, ell], k, A, sigma_phi).item() * (t_torch[ell+1]-t_torch[ell]).item()
                        + diff_sign * diff_coef * compute_laplacian_mean(data_torch[m, ell], k, A, sigma_phi).item() * (t_torch[ell+1]-t_torch[ell]).item()
                        + energy_sign * energy_coef * (compute_energy(data_torch[m, ell+1], k, A, sigma_phi).item() - compute_energy(data_torch[m, ell], k, A, sigma_phi).item())
                        for m in range(M) for ell in range(L-1)
                    ])
                    t_stat, p_val = stats.ttest_1samp(residuals, 0)
                    if abs(t_stat) < abs(best_t_stat):
                        best_t_stat = t_stat
                        best_formula = f"J_diss {'+' if diff_sign > 0 else '-'} {diff_coef:.4f}*J_lap {'+' if energy_sign > 0 else '-'} {energy_coef}*dE"
                        best_residuals = residuals

    print(f"\nBest formula: {best_formula}")
    print(f"  t-stat: {best_t_stat:.4f}")
    print(f"  p-value: {stats.ttest_1samp(best_residuals, 0)[1]:.6e}")
    print(f"  Mean residual: {best_residuals.mean():.6e}")
    print(f"  Std residual: {best_residuals.std():.6e}")


def main():
    N, d = 10, 1
    L, M = 20, 500  # More samples
    dt = 0.01
    T = 2.0
    sigma_noise = 0.1
    seed = 42

    k = 1.0
    A = 1.0
    sigma_phi = 1.0

    print("MVP-1.2c: Verify CORRECT Weak-Form Formula")
    print(f"\nN={N}, d={d}, L={L}, M={M}, sigma={sigma_noise}")

    V = HarmonicPotential(k=k)
    Phi = GaussianInteraction(A=A, sigma=sigma_phi)
    simulator = SDESimulator(V=V, Phi=Phi, sigma=sigma_noise, dt=dt)
    data, t_snapshots = simulator.simulate(N=N, d=d, T=T, L=L, M=M, seed=seed)

    test_correct_formula(data, t_snapshots, sigma_noise, k, A, sigma_phi)


if __name__ == '__main__':
    main()
