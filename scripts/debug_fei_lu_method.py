#!/usr/bin/env python
"""Debug script for MVP-2.0: Verify error functional implementation.

Key test: When using the TRUE φ, the error functional should be at its minimum.
We verify this by:
1. Computing E(φ_true)
2. Computing E(φ_perturbed) for perturbed φ
3. Checking that E(φ_perturbed) > E(φ_true)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# =============================================================================
# Simplified test case: 1D Opinion dynamics-like system
# =============================================================================

def true_phi(r, A=1.0, sigma=1.0):
    """True interaction kernel: gradient of Gaussian potential.
    Φ(r) = A * exp(-r^2 / (2σ^2))
    φ(r) = dΦ/dr = -A * r / σ^2 * exp(-r^2 / (2σ^2))
    """
    return -A * r / (sigma**2) * np.exp(-r**2 / (2 * sigma**2))


def true_Phi(r, A=1.0, sigma=1.0):
    """True interaction potential (antiderivative of φ)."""
    return A * np.exp(-r**2 / (2 * sigma**2))


def generate_mean_field_solution(x_grid, t_grid, nu=0.1, phi_func=None, A=1.0, sigma_phi=1.0):
    """Generate mean-field solution by solving the PDE numerically.

    Solves: ∂_t u = ν ∂_xx u + ∂_x[u (K_φ * u)]

    Using forward Euler in time, central differences in space.
    """
    M = len(x_grid)
    L = len(t_grid)
    dx = x_grid[1] - x_grid[0]

    # Initial condition: Gaussian
    def u0(x):
        return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

    u = np.zeros((L, M))
    u[0] = u0(x_grid)
    u[0] /= np.sum(u[0]) * dx  # Normalize

    # Define phi if not provided
    if phi_func is None:
        phi_func = lambda r: true_phi(r, A, sigma_phi)

    # Time stepping
    for l in range(L - 1):
        dt = t_grid[l + 1] - t_grid[l]
        u_curr = u[l].copy()

        # Compute K_φ * u at each point
        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func)

        # Compute flux: F = u * (K_φ * u)
        flux = u_curr * K_phi_u

        # Compute ∂_x F using central differences
        dflux_dx = np.zeros(M)
        dflux_dx[1:-1] = (flux[2:] - flux[:-2]) / (2 * dx)
        dflux_dx[0] = (flux[1] - flux[0]) / dx
        dflux_dx[-1] = (flux[-1] - flux[-2]) / dx

        # Compute ∂_xx u using central differences
        d2u_dx2 = np.zeros(M)
        d2u_dx2[1:-1] = (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]) / dx**2

        # Forward Euler step: ∂_t u = ν ∂_xx u + ∂_x[u (K_φ * u)]
        u[l + 1] = u_curr + dt * (nu * d2u_dx2 + dflux_dx)

        # Ensure non-negative and normalize
        u[l + 1] = np.maximum(u[l + 1], 0)
        mass = np.sum(u[l + 1]) * dx
        if mass > 1e-10:
            u[l + 1] /= mass

    return u


def compute_K_phi_conv_u(x_grid, u, phi_func):
    """Compute K_φ * u where K_φ(x) = φ(|x|) * sign(x) for 1D.

    (K_φ * u)(x) = ∫ φ(|x-y|) * sign(x-y) * u(y) dy
    """
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    result = np.zeros(M)

    for m, x in enumerate(x_grid):
        # Compute convolution at point x
        for n, y in enumerate(x_grid):
            r = abs(x - y)
            if r > 1e-10:
                result[m] += phi_func(r) * np.sign(x - y) * u[n] * dx

    return result


def compute_Phi_conv_u(x_grid, u, Phi_func):
    """Compute Φ * u.

    (Φ * u)(x) = ∫ Φ(|x-y|) * u(y) dy
    """
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    result = np.zeros(M)

    for m, x in enumerate(x_grid):
        for n, y in enumerate(x_grid):
            r = abs(x - y)
            result[m] += Phi_func(r) * u[n] * dx

    return result


def compute_div_K_phi_conv_u(x_grid, u, dphi_func):
    """Compute (∇·K_φ) * u where ∇·K_φ = φ'(|x|) for 1D.

    For radial φ: div(K_φ) = φ'(r) for 1D
    """
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    result = np.zeros(M)

    for m, x in enumerate(x_grid):
        for n, y in enumerate(x_grid):
            r = abs(x - y)
            result[m] += dphi_func(r) * u[n] * dx

    return result


def compute_error_functional(u_time, t_grid, x_grid, phi_func, Phi_func, dphi_func, nu):
    """Compute the error functional E(φ) from Eq 2.3.

    E(φ) = (1/T) ∫_0^T ∫ [|K_φ*u|² u + 2 ∂_t u (Φ*u) + 2ν ∇u · (K_φ*u)] dx dt

    For 1D: ∇u · (K_φ*u) = (du/dx) * (K_φ*u)
    """
    L, M = u_time.shape
    dx = x_grid[1] - x_grid[0]
    T = t_grid[-1] - t_grid[0]

    E = 0.0

    for l in range(L - 1):
        u_curr = u_time[l]
        u_next = u_time[l + 1]
        dt = t_grid[l + 1] - t_grid[l]

        # du/dt
        du_dt = (u_next - u_curr) / dt

        # du/dx
        du_dx = np.gradient(u_curr, dx)

        # Convolutions
        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func)
        Phi_u = compute_Phi_conv_u(x_grid, u_curr, Phi_func)

        # Term 1: |K_φ*u|² u
        term1 = np.sum(K_phi_u**2 * u_curr) * dx

        # Term 2: 2 ∂_t u (Φ*u)
        term2 = 2 * np.sum(du_dt * Phi_u) * dx

        # Term 3: 2ν ∇u · (K_φ*u)
        term3 = 2 * nu * np.sum(du_dx * K_phi_u) * dx

        E += (term1 + term2 + term3) * dt

    E /= T
    return E


def true_dphi(r, A=1.0, sigma=1.0):
    """Derivative of φ (second derivative of Φ)."""
    # φ(r) = -A * r / σ^2 * exp(-r^2 / (2σ^2))
    # φ'(r) = -A / σ^2 * exp(-r^2 / (2σ^2)) + A * r^2 / σ^4 * exp(-r^2 / (2σ^2))
    #       = A * exp(-r^2 / (2σ^2)) * (r^2 / σ^4 - 1 / σ^2)
    return A * np.exp(-r**2 / (2 * sigma**2)) * (r**2 / sigma**4 - 1 / sigma**2)


def main():
    print("=" * 70)
    print("Debug: Verify Error Functional Implementation")
    print("=" * 70)

    # Parameters
    nu = 0.1
    A = 1.0
    sigma_phi = 1.0

    # Grid (need small dt for PDE stability)
    x_min, x_max = -5, 5
    M = 100
    L = 200  # More time steps for numerical stability
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]

    print(f"\nParameters:")
    print(f"  ν = {nu}")
    print(f"  Grid: M={M}, L={L}")
    print(f"  x range: [{x_min}, {x_max}]")
    print(f"  T = {T}")

    # Define phi functions first (needed for solution generation)
    def phi_true(r): return true_phi(r, A, sigma_phi)
    def Phi_true(r): return true_Phi(r, A, sigma_phi)
    def dphi_true(r): return true_dphi(r, A, sigma_phi)

    # Generate solution using TRUE phi
    print("\n[1] Generating mean-field solution with TRUE φ...")
    u_time = generate_mean_field_solution(x_grid, t_grid, nu, phi_func=phi_true)
    print(f"  u shape: {u_time.shape}")
    print(f"  u integral at t=0: {np.sum(u_time[0]) * dx:.4f} (should be 1)")
    print(f"  u integral at t=T: {np.sum(u_time[-1]) * dx:.4f} (should be 1)")

    # Compute error functional for true phi
    print("\n[2] Computing E(φ_true)...")
    E_true = compute_error_functional(u_time, t_grid, x_grid, phi_true, Phi_true, dphi_true, nu)
    print(f"  E(φ_true) = {E_true:.6e}")

    # Compute error functional for perturbed phi (should be larger)
    print("\n[3] Computing E(φ_perturbed)...")

    perturbations = [0.5, 0.8, 1.2, 1.5, 2.0]
    E_perturbed = []

    for scale in perturbations:
        def phi_pert(r): return scale * true_phi(r, A, sigma_phi)
        def Phi_pert(r): return scale * true_Phi(r, A, sigma_phi)
        def dphi_pert(r): return scale * true_dphi(r, A, sigma_phi)

        E = compute_error_functional(u_time, t_grid, x_grid, phi_pert, Phi_pert, dphi_pert, nu)
        E_perturbed.append(E)
        print(f"  E(φ × {scale}) = {E:.6e}")

    # Compute error functional for zero phi
    print("\n[4] Computing E(φ = 0)...")
    def phi_zero(r): return 0.0
    def Phi_zero(r): return 0.0
    def dphi_zero(r): return 0.0
    E_zero = compute_error_functional(u_time, t_grid, x_grid, phi_zero, Phi_zero, dphi_zero, nu)
    print(f"  E(φ = 0) = {E_zero:.6e}")

    # Check if E(φ_true) is minimum
    print("\n" + "=" * 70)
    print("Verification Results:")
    print("=" * 70)

    print(f"\nE(φ_true) = {E_true:.6e}")
    print(f"E(φ = 0)  = {E_zero:.6e}")

    is_minimum = all(E_true <= E for E in E_perturbed) and E_true <= E_zero

    if is_minimum:
        print("\n✅ E(φ_true) is the minimum among tested values")
    else:
        print("\n❌ E(φ_true) is NOT the minimum - implementation may have bugs")
        print("   Expected: E(φ_true) should be smallest")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Solution u(x,t)
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, L))
    for l in range(0, L, max(1, L//5)):
        ax.plot(x_grid, u_time[l], color=colors[l], label=f't={t_grid[l]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Mean-field solution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Error functional vs perturbation
    ax = axes[1]
    ax.plot([0] + perturbations, [E_zero] + E_perturbed, 'bo-', markersize=8)
    ax.axhline(E_true, color='r', linestyle='--', label=f'E(φ_true)={E_true:.2e}')
    ax.axvline(1.0, color='g', linestyle=':', alpha=0.5, label='True scale')
    ax.set_xlabel('Scale factor')
    ax.set_ylabel('Error functional E')
    ax.set_title('E vs perturbation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: True phi
    ax = axes[2]
    r = np.linspace(0, 4, 100)
    ax.plot(r, phi_true(r), 'b-', lw=2, label='φ(r)')
    ax.plot(r, Phi_true(r), 'r--', lw=2, label='Φ(r)')
    ax.set_xlabel('r')
    ax.set_ylabel('Value')
    ax.set_title('True interaction functions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/ips_unlabeled/img/debug_error_functional.png', dpi=150)
    plt.close()
    print("\nPlot saved to experiments/ips_unlabeled/img/debug_error_functional.png")

    return 0 if is_minimum else 1


if __name__ == '__main__':
    sys.exit(main())
