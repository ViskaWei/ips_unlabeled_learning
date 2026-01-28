#!/usr/bin/env python
"""MVP-1.2: Verify the loss formula with true potentials.

If the loss formula is correct, using the true V and Phi should give loss ≈ 0.
This tests whether the weak-form equation is correctly implemented.

Test cases:
1. True V, True Phi -> Loss should be ~0
2. True V, Wrong Phi -> Loss should be > 0
3. Wrong V, True Phi -> Loss should be > 0
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from core.sde_simulator import SDESimulator
from core.potentials import HarmonicPotential, GaussianInteraction
from core.nn_models import compute_pairwise_distances, compute_pairwise_diff


def harmonic_V(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """True V(x) = (k/2) x^2"""
    return 0.5 * k * (x ** 2).sum(dim=-1)


def harmonic_grad_V(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """True grad V(x) = k * x"""
    return k * x


def harmonic_laplacian_V(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """True Laplacian V = k * d (trace of Hessian)"""
    d = x.shape[-1]
    return torch.full((x.shape[0],), k * d, device=x.device)


def gaussian_Phi(r: torch.Tensor, A: float = 1.0, sigma: float = 1.0) -> torch.Tensor:
    """True Phi(r) = A * exp(-r^2 / (2 sigma^2))"""
    return A * torch.exp(-r ** 2 / (2 * sigma ** 2))


def gaussian_grad_Phi(r: torch.Tensor, A: float = 1.0, sigma: float = 1.0) -> torch.Tensor:
    """True dPhi/dr = -A * r / sigma^2 * exp(...)"""
    return -A * r / (sigma ** 2) * torch.exp(-r ** 2 / (2 * sigma ** 2))


def gaussian_laplacian_Phi_1d(r: torch.Tensor, A: float = 1.0, sigma: float = 1.0) -> torch.Tensor:
    """True d^2Phi/dr^2 for 1D case.

    dPhi/dr = -A * r / sigma^2 * exp(-r^2/(2*sigma^2))
    d^2Phi/dr^2 = A/sigma^2 * (r^2/sigma^2 - 1) * exp(-r^2/(2*sigma^2))
    """
    exp_term = torch.exp(-r ** 2 / (2 * sigma ** 2))
    return A / (sigma ** 2) * (r ** 2 / (sigma ** 2) - 1) * exp_term


def compute_true_drift(X: torch.Tensor, k: float, A: float, sigma_phi: float) -> torch.Tensor:
    """Compute drift using true potentials.

    drift = -grad_V - (1/N) sum_j grad_Phi
    """
    N, d = X.shape

    # grad V
    grad_V = harmonic_grad_V(X, k)  # (N, d)

    # Interaction gradient
    diff = compute_pairwise_diff(X)  # (N, N, d)
    distances = torch.norm(diff, dim=-1)  # (N, N)

    distances_safe = distances.clone()
    distances_safe[distances_safe < 1e-10] = 1e-10

    dPhi_dr = gaussian_grad_Phi(distances, A, sigma_phi)  # (N, N)

    unit_diff = diff / distances_safe.unsqueeze(-1)  # (N, N, d)
    grad_Phi_pairs = dPhi_dr.unsqueeze(-1) * unit_diff  # (N, N, d)

    # Zero diagonal
    mask = torch.eye(N, device=X.device, dtype=torch.bool)
    grad_Phi_pairs[mask] = 0

    grad_Phi_mean = grad_Phi_pairs.sum(dim=1) / N  # (N, d)

    drift = -grad_V - grad_Phi_mean
    return drift


def compute_true_laplacian_sum(X: torch.Tensor, k: float, A: float, sigma_phi: float) -> torch.Tensor:
    """Compute Laplacian sum using true potentials."""
    N, d = X.shape

    # Laplacian V
    laplacian_V = harmonic_laplacian_V(X, k)  # (N,)

    # Laplacian Phi
    distances = compute_pairwise_distances(X)  # (N, N)
    laplacian_Phi = gaussian_laplacian_Phi_1d(distances, A, sigma_phi)  # (N, N)

    # Zero diagonal
    mask = torch.eye(N, device=X.device, dtype=torch.bool)
    laplacian_Phi[mask] = 0

    laplacian_Phi_mean = laplacian_Phi.sum(dim=1) / N  # (N,)

    return laplacian_V + laplacian_Phi_mean


def compute_true_energy(X: torch.Tensor, k: float, A: float, sigma_phi: float) -> torch.Tensor:
    """Compute energy using true potentials."""
    N = X.shape[0]

    # V term
    V_vals = harmonic_V(X, k)
    V_mean = V_vals.mean()

    # Phi term
    distances = compute_pairwise_distances(X)
    Phi_vals = gaussian_Phi(distances, A, sigma_phi)

    # Zero diagonal
    mask = torch.eye(N, device=X.device, dtype=torch.bool)
    Phi_vals[mask] = 0

    Phi_mean = Phi_vals.sum() / (N * N)

    return V_mean + Phi_mean


def compute_loss_with_true_potentials(
    data: np.ndarray,
    t_snapshots: np.ndarray,
    sigma_noise: float,
    k: float = 1.0,
    A: float = 1.0,
    sigma_phi: float = 1.0,
    use_sigma_squared: bool = True,  # True: use σ², False: use σ
) -> dict:
    """Compute the weak-form loss using true potential functions.

    If the formula is correct, loss should be ~0 (martingale term averages out).
    """
    data_torch = torch.tensor(data, dtype=torch.float32)
    t_torch = torch.tensor(t_snapshots, dtype=torch.float32)

    M, L, N, d = data_torch.shape

    total_diss = 0.0
    total_diff = 0.0
    total_energy_change = 0.0

    if use_sigma_squared:
        sigma_coef = sigma_noise ** 2
    else:
        sigma_coef = sigma_noise

    for m in range(M):
        for ell in range(L - 1):
            X_curr = data_torch[m, ell]
            X_next = data_torch[m, ell + 1]
            dt = t_torch[ell + 1] - t_torch[ell]

            # Dissipation
            drift = compute_true_drift(X_curr, k, A, sigma_phi)
            J_diss = (drift ** 2).sum() / N * dt

            # Diffusion
            laplacian_sum = compute_true_laplacian_sum(X_curr, k, A, sigma_phi)
            J_diff = sigma_coef * laplacian_sum.mean() * dt

            # Energy change
            E_curr = compute_true_energy(X_curr, k, A, sigma_phi)
            E_next = compute_true_energy(X_next, k, A, sigma_phi)
            J_energy_change = E_next - E_curr

            total_diss += J_diss.item()
            total_diff += J_diff.item()
            total_energy_change += J_energy_change.item()

    n_pairs = M * (L - 1)
    total_diss /= n_pairs
    total_diff /= n_pairs
    total_energy_change /= n_pairs

    # Weak form: J_diss + J_diff = 2 * J_energy_change
    # Different formulations to test:
    residual_v1 = total_diss + total_diff - 2 * total_energy_change  # Current code
    residual_v2 = total_diss + total_diff + 2 * total_energy_change  # Alternative

    return {
        'J_diss': total_diss,
        'J_diff': total_diff,
        'J_energy_change': total_energy_change,
        'residual_v1 (diss + diff - 2*dE)': residual_v1,
        'residual_v2 (diss + diff + 2*dE)': residual_v2,
        'loss_v1': residual_v1 ** 2,
        'loss_v2': residual_v2 ** 2,
        'sigma_coef': 'sigma^2' if use_sigma_squared else 'sigma',
    }


def main():
    print("=" * 80)
    print("MVP-1.2: Loss Formula Verification with True Potentials")
    print("=" * 80)

    # Parameters
    N, d = 10, 1
    L, M = 20, 100
    dt = 0.01
    T = 2.0
    sigma_noise = 0.1
    seed = 42

    # True potential parameters
    k = 1.0  # Harmonic V
    A = 1.0  # Gaussian Phi amplitude
    sigma_phi = 1.0  # Gaussian Phi width

    print(f"\nConfiguration:")
    print(f"  N={N}, d={d}, L={L}, M={M}")
    print(f"  sigma_noise={sigma_noise}, dt={dt}, T={T}")
    print(f"  V: Harmonic(k={k})")
    print(f"  Phi: Gaussian(A={A}, sigma={sigma_phi})")

    # Generate data
    print("\nGenerating data...")
    V = HarmonicPotential(k=k)
    Phi = GaussianInteraction(A=A, sigma=sigma_phi)
    simulator = SDESimulator(V=V, Phi=Phi, sigma=sigma_noise, dt=dt)
    data, t_snapshots = simulator.simulate(N=N, d=d, T=T, L=L, M=M, seed=seed)
    print(f"Data shape: {data.shape}")

    # Test 1: Compute loss with true potentials using sigma^2
    print("\n" + "-" * 80)
    print("TEST 1: True potentials with sigma^2 in diffusion term")
    print("-" * 80)
    result1 = compute_loss_with_true_potentials(
        data, t_snapshots, sigma_noise, k, A, sigma_phi,
        use_sigma_squared=True
    )
    for key, val in result1.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6e}")
        else:
            print(f"  {key}: {val}")

    # Test 2: Compute loss with true potentials using sigma (not squared)
    print("\n" + "-" * 80)
    print("TEST 2: True potentials with sigma (not squared) in diffusion term")
    print("-" * 80)
    result2 = compute_loss_with_true_potentials(
        data, t_snapshots, sigma_noise, k, A, sigma_phi,
        use_sigma_squared=False
    )
    for key, val in result2.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6e}")
        else:
            print(f"  {key}: {val}")

    # Test 3: Wrong Phi (zero interaction)
    print("\n" + "-" * 80)
    print("TEST 3: Wrong potentials (true V, but Phi=0)")
    print("-" * 80)
    result3 = compute_loss_with_true_potentials(
        data, t_snapshots, sigma_noise, k, 0.0, sigma_phi,  # A=0 -> Phi=0
        use_sigma_squared=True
    )
    for key, val in result3.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6e}")
        else:
            print(f"  {key}: {val}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print("\nIf the formula is correct, one of the residuals should be ~0 for TEST 1/2")
    print(f"\nTEST 1 (sigma^2): residual_v1={result1['residual_v1 (diss + diff - 2*dE)']:.6e}, residual_v2={result1['residual_v2 (diss + diff + 2*dE)']:.6e}")
    print(f"TEST 2 (sigma):   residual_v1={result2['residual_v1 (diss + diff - 2*dE)']:.6e}, residual_v2={result2['residual_v2 (diss + diff + 2*dE)']:.6e}")

    min_residual_1 = min(abs(result1['residual_v1 (diss + diff - 2*dE)']), abs(result1['residual_v2 (diss + diff + 2*dE)']))
    min_residual_2 = min(abs(result2['residual_v1 (diss + diff - 2*dE)']), abs(result2['residual_v2 (diss + diff + 2*dE)']))

    if min_residual_1 < min_residual_2:
        print(f"\n→ sigma^2 version has smaller residual: {min_residual_1:.6e}")
        if min_residual_1 < 0.01:
            print("  The formula with sigma^2 appears CORRECT (residual is small)")
        else:
            print("  BUT residual is still large - formula may have other issues")
    else:
        print(f"\n→ sigma version has smaller residual: {min_residual_2:.6e}")
        if min_residual_2 < 0.01:
            print("  The formula with sigma appears CORRECT (residual is small)")
        else:
            print("  BUT residual is still large - formula may have other issues")

    # Check magnitude of components
    print("\nComponent magnitudes (TEST 1):")
    print(f"  J_diss:          {result1['J_diss']:.6e}")
    print(f"  J_diff:          {result1['J_diff']:.6e}")
    print(f"  J_energy_change: {result1['J_energy_change']:.6e}")
    print(f"  Ratio |J_diff/J_diss|: {abs(result1['J_diff']/result1['J_diss']):.4f}")

    # Expected relation for equilibrium
    print("\nTheoretical expectation:")
    print("  At equilibrium, energy dissipation = energy input from diffusion")
    print("  Weak form: J_diss + J_diff - 2*dE = 0 (up to martingale fluctuation)")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if min_residual_1 < 0.01 or min_residual_2 < 0.01:
        print("✅ The loss formula appears to be CORRECT in expectation")
        print("   (Residual is small when using true potentials)")
    else:
        print("❌ The loss formula may have ISSUES")
        print("   (Residual is large even when using true potentials)")
        print("\n   Possible causes:")
        print("   1. Wrong coefficient (sigma vs sigma^2 vs sigma^2/2)")
        print("   2. Sign error in energy term")
        print("   3. Missing factor in diffusion term")
        print("   4. Discrete time approximation error")


if __name__ == '__main__':
    main()
