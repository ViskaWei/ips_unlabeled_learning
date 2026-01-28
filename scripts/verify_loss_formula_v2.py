#!/usr/bin/env python
"""MVP-1.2b: Detailed verification of weak-form loss derivation.

Key insight: The weak form comes from Ito's lemma applied to E[f(X)].
For SDE: dX = b dt + σ dW, we have:
d⟨f, μ⟩ = ⟨∇f · b + (σ²/2) Δf, μ⟩ dt + martingale term

The energy (with f = V + Φ*μ):
d E(t) = ⟨∇(V + Φ*μ) · b + (σ²/2) Δ(V + Φ*μ), μ⟩ dt + martingale

Where b = -∇V - ∇Φ*μ, so:
∇(V + Φ*μ) · b = -|∇V + ∇Φ*μ|² (negative, energy dissipation)

Rearranging:
|∇V + ∇Φ*μ|² dt = -(dE - (σ²/2) ⟨ΔV + ΔΦ*μ, μ⟩ dt)

The weak-form residual should be:
R = ⟨|∇V + ∇Φ*μ|², μ⟩ Δt + (σ²/2) ⟨ΔV + ΔΦ*μ, μ⟩ Δt - (E(t+Δt) - E(t))

Note: The diffusion coefficient should be σ²/2, not σ or σ²!
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
    """Compute |b|² = |∇V + ∇Φ*μ|² averaged over particles."""
    N, d = X.shape

    grad_V = harmonic_grad_V(X, k)  # (N, d)

    diff = compute_pairwise_diff(X)  # (N, N, d)
    distances = torch.norm(diff, dim=-1)
    distances_safe = distances.clone()
    distances_safe[distances_safe < 1e-10] = 1e-10

    dPhi_dr = gaussian_grad_Phi(distances, A, sigma_phi)
    unit_diff = diff / distances_safe.unsqueeze(-1)
    grad_Phi_pairs = dPhi_dr.unsqueeze(-1) * unit_diff

    mask = torch.eye(N, device=X.device, dtype=torch.bool)
    grad_Phi_pairs[mask] = 0

    grad_Phi_mean = grad_Phi_pairs.sum(dim=1) / N  # (N, d)

    # Total drift magnitude squared
    # Note: drift = -(grad_V + grad_Phi_mean), but |drift|² = |grad_V + grad_Phi_mean|²
    force = grad_V + grad_Phi_mean
    drift_norm_sq = (force ** 2).sum(dim=-1).mean()  # Average over particles

    return drift_norm_sq


def compute_laplacian_mean(X: torch.Tensor, k: float, A: float, sigma_phi: float) -> torch.Tensor:
    """Compute ⟨ΔV + ΔΦ*μ, μ⟩."""
    N, d = X.shape

    laplacian_V = harmonic_laplacian_V(X, k)  # (N,)

    distances = compute_pairwise_distances(X)
    laplacian_Phi = gaussian_laplacian_Phi_1d(distances, A, sigma_phi)

    mask = torch.eye(N, device=X.device, dtype=torch.bool)
    laplacian_Phi[mask] = 0

    laplacian_Phi_mean = laplacian_Phi.sum(dim=1) / N

    return (laplacian_V + laplacian_Phi_mean).mean()


def compute_energy(X: torch.Tensor, k: float, A: float, sigma_phi: float) -> torch.Tensor:
    """Compute E = ⟨V + Φ*μ, μ⟩."""
    N = X.shape[0]

    V_vals = harmonic_V(X, k)
    V_mean = V_vals.mean()

    distances = compute_pairwise_distances(X)
    Phi_vals = gaussian_Phi(distances, A, sigma_phi)

    mask = torch.eye(N, device=X.device, dtype=torch.bool)
    Phi_vals[mask] = 0

    Phi_mean = Phi_vals.sum() / (N * N)

    return V_mean + Phi_mean


def test_loss_formulas(
    data: np.ndarray,
    t_snapshots: np.ndarray,
    sigma_noise: float,
    k: float = 1.0,
    A: float = 1.0,
    sigma_phi: float = 1.0,
):
    """Test different coefficient formulations."""
    data_torch = torch.tensor(data, dtype=torch.float32)
    t_torch = torch.tensor(t_snapshots, dtype=torch.float32)

    M, L, N, d = data_torch.shape

    # Collect individual terms for analysis
    all_diss = []
    all_lap = []
    all_dE = []

    for m in range(M):
        for ell in range(L - 1):
            X_curr = data_torch[m, ell]
            X_next = data_torch[m, ell + 1]
            dt = (t_torch[ell + 1] - t_torch[ell]).item()

            diss = compute_drift_norm_sq(X_curr, k, A, sigma_phi).item() * dt
            lap = compute_laplacian_mean(X_curr, k, A, sigma_phi).item() * dt
            E_curr = compute_energy(X_curr, k, A, sigma_phi).item()
            E_next = compute_energy(X_next, k, A, sigma_phi).item()
            dE = E_next - E_curr

            all_diss.append(diss)
            all_lap.append(lap)
            all_dE.append(dE)

    all_diss = np.array(all_diss)
    all_lap = np.array(all_lap)
    all_dE = np.array(all_dE)

    print("="*80)
    print("Testing different loss formulations")
    print("="*80)

    print(f"\nMean components:")
    print(f"  Mean J_diss (|∇V + ∇Φ*μ|² Δt):  {all_diss.mean():.6e}")
    print(f"  Mean J_lap (Δ(V + Φ*μ) Δt):     {all_lap.mean():.6e}")
    print(f"  Mean dE:                         {all_dE.mean():.6e}")

    # Test different coefficients for diffusion term
    print("\n" + "-"*80)
    print("Testing different diffusion coefficients:")
    print("-"*80)

    sigma_coeffs = {
        'σ': sigma_noise,
        'σ²': sigma_noise ** 2,
        'σ²/2': sigma_noise ** 2 / 2,
        '1': 1.0,
    }

    # Energy balance: dE = -diss + σ_coef * lap (from Ito's lemma)
    # So: diss + σ_coef * lap - dE = 0

    for name, coef in sigma_coeffs.items():
        residuals = all_diss + coef * all_lap - all_dE
        mean_r = residuals.mean()
        std_r = residuals.std()
        print(f"\n  Coefficient {name} = {coef:.4f}:")
        print(f"    Formula: J_diss + {name} * J_lap - dE = 0")
        print(f"    Mean residual: {mean_r:.6e}")
        print(f"    Std residual:  {std_r:.6e}")
        print(f"    |Mean|/Std:    {abs(mean_r)/std_r:.4f}")

    # Also test: diss + coef * lap + dE = 0 (alternative sign)
    print("\n" + "-"*80)
    print("Testing with opposite energy sign (+dE instead of -dE):")
    print("-"*80)

    for name, coef in sigma_coeffs.items():
        residuals = all_diss + coef * all_lap + all_dE
        mean_r = residuals.mean()
        std_r = residuals.std()
        print(f"\n  Coefficient {name} = {coef:.4f}:")
        print(f"    Formula: J_diss + {name} * J_lap + dE = 0")
        print(f"    Mean residual: {mean_r:.6e}")
        print(f"    Std residual:  {std_r:.6e}")

    # Direct test: which formula gives mean residual closest to 0?
    print("\n" + "="*80)
    print("FINDING BEST FORMULA")
    print("="*80)

    best_formula = None
    best_residual = float('inf')

    for name, coef in sigma_coeffs.items():
        for sign_name, sign in [('-dE', -1), ('+dE', +1)]:
            residuals = all_diss + coef * all_lap + sign * all_dE
            mean_r = abs(residuals.mean())
            if mean_r < best_residual:
                best_residual = mean_r
                best_formula = f"J_diss + {name} * J_lap {sign_name}"

    print(f"\nBest formula: {best_formula}")
    print(f"Mean residual: {best_residual:.6e}")

    # Additional analysis: check if the issue is the factor of 2 in energy term
    print("\n" + "="*80)
    print("Testing factor of 2 in energy term (paper formula)")
    print("="*80)

    for name, coef in sigma_coeffs.items():
        # Paper formula: J_diss + coef * J_lap - 2*dE = 0
        residuals = all_diss + coef * all_lap - 2 * all_dE
        mean_r = residuals.mean()
        std_r = residuals.std()
        print(f"\n  {name}: J_diss + {name}*J_lap - 2*dE")
        print(f"    Mean: {mean_r:.6e}, Std: {std_r:.6e}")


def main():
    # Parameters
    N, d = 10, 1
    L, M = 20, 200  # More samples for better statistics
    dt = 0.01
    T = 2.0
    sigma_noise = 0.1
    seed = 42

    k = 1.0
    A = 1.0
    sigma_phi = 1.0

    print("MVP-1.2b: Detailed Loss Formula Verification")
    print(f"\nN={N}, d={d}, L={L}, M={M}, sigma={sigma_noise}")

    # Generate data
    V = HarmonicPotential(k=k)
    Phi = GaussianInteraction(A=A, sigma=sigma_phi)
    simulator = SDESimulator(V=V, Phi=Phi, sigma=sigma_noise, dt=dt)
    data, t_snapshots = simulator.simulate(N=N, d=d, T=T, L=L, M=M, seed=seed)

    test_loss_formulas(data, t_snapshots, sigma_noise, k, A, sigma_phi)


if __name__ == '__main__':
    main()
