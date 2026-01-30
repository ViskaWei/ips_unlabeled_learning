#!/usr/bin/env python
"""Sanity check: Verify A matrix and b vector computation.

Key test: For true coefficients c_true, we should have:
- A c_true ≈ b (the normal equations should be satisfied)
"""

import numpy as np
from scipy import integrate
from scipy.interpolate import BSpline
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def true_phi(r, A=1.0, sigma=1.0):
    return -A * r / (sigma**2) * np.exp(-r**2 / (2 * sigma**2))


def true_Phi(r, A=1.0, sigma=1.0):
    return A * np.exp(-r**2 / (2 * sigma**2))


def compute_K_phi_conv_u(x_grid, u, phi_func):
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    result = np.zeros(M)
    for m, x in enumerate(x_grid):
        for n, y in enumerate(x_grid):
            r = abs(x - y)
            if r > 1e-10:
                result[m] += phi_func(r) * np.sign(x - y) * u[n] * dx
    return result


def compute_Phi_conv_u(x_grid, u, Phi_func):
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    result = np.zeros(M)
    for m, x in enumerate(x_grid):
        for n, y in enumerate(x_grid):
            r = abs(x - y)
            result[m] += Phi_func(r) * u[n] * dx
    return result


def generate_pde_solution(x_grid, t_grid, nu, phi_func):
    M = len(x_grid)
    L = len(t_grid)
    dx = x_grid[1] - x_grid[0]

    u = np.zeros((L, M))
    u[0] = np.exp(-x_grid**2 / 2) / np.sqrt(2 * np.pi)
    u[0] /= np.sum(u[0]) * dx

    for l in range(L - 1):
        dt = t_grid[l + 1] - t_grid[l]
        u_curr = u[l].copy()

        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func)
        flux = u_curr * K_phi_u

        dflux_dx = np.zeros(M)
        dflux_dx[1:-1] = (flux[2:] - flux[:-2]) / (2 * dx)
        dflux_dx[0] = (flux[1] - flux[0]) / dx
        dflux_dx[-1] = (flux[-1] - flux[-2]) / dx

        d2u_dx2 = np.zeros(M)
        d2u_dx2[1:-1] = (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]) / dx**2

        u[l + 1] = u_curr + dt * (nu * d2u_dx2 + dflux_dx)
        u[l + 1] = np.maximum(u[l + 1], 0)
        mass = np.sum(u[l + 1]) * dx
        if mass > 1e-10:
            u[l + 1] /= mass

    return u


def main():
    print("=" * 70)
    print("Sanity Check: Verify Error Functional Gradient")
    print("=" * 70)

    # Parameters
    nu = 0.1
    A_phi = 1.0
    sigma_phi = 1.0

    # Grid
    x_min, x_max = -4, 4
    M = 80
    L = 100
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]

    print(f"\nParameters: ν={nu}, M={M}, L={L}")

    # True phi functions
    phi_func = lambda r: true_phi(r, A_phi, sigma_phi)
    Phi_func = lambda r: true_Phi(r, A_phi, sigma_phi)

    # Generate PDE solution with TRUE phi
    print("\n[1] Generating PDE solution with true φ...")
    u_time = generate_pde_solution(x_grid, t_grid, nu, phi_func)
    print(f"  u shape: {u_time.shape}")

    # KEY TEST: For the TRUE phi, the error functional gradient should be ~0
    #
    # E(φ) = (1/T) ∫∫ [|K_φ*u|² u + 2∂_t u (Φ*u) + 2ν ∇u · (K_φ*u)] dx dt
    #
    # If u satisfies the PDE: ∂_t u = ν Δu + ∇·[u K_φ*u]
    # Then ∂E/∂φ = 0 at φ = φ_true
    #
    # Let's compute the gradient components directly

    print("\n[2] Computing gradient of E at φ_true...")

    grad_term1 = 0.0  # from |K_φ*u|² u term
    grad_term2 = 0.0  # from ∂_t u (Φ*u) term
    grad_term3 = 0.0  # from ∇u · (K_φ*u) term

    for ell in range(L - 1):
        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]

        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)

        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func)
        Phi_u = compute_Phi_conv_u(x_grid, u_curr, Phi_func)

        # The three terms in the error functional
        term1 = np.sum(K_phi_u**2 * u_curr) * dx  # |K_φ*u|² u
        term2 = np.sum(du_dt * Phi_u) * dx  # ∂_t u (Φ*u)
        term3 = np.sum(du_dx * K_phi_u) * dx  # ∇u · (K_φ*u)

        grad_term1 += term1 * dt
        grad_term2 += 2 * term2 * dt
        grad_term3 += 2 * nu * term3 * dt

    grad_term1 /= T
    grad_term2 /= T
    grad_term3 /= T

    E_true = grad_term1 + grad_term2 + grad_term3

    print(f"\n  Error functional E(φ_true):")
    print(f"    Term 1 (|K_φ*u|² u): {grad_term1:.6e}")
    print(f"    Term 2 (2∂_t u Φ*u): {grad_term2:.6e}")
    print(f"    Term 3 (2ν ∇u·K_φ*u): {grad_term3:.6e}")
    print(f"    Total E(φ_true): {E_true:.6e}")

    # For true phi, E should be close to its minimum (ideally 0 for exact PDE solution)
    # Let's also check if the three terms approximately cancel

    print("\n[3] Checking PDE residual...")

    # The PDE is: ∂_t u = ν Δu + ∇·[u K_φ*u]
    # Residual should be near 0 at each time step

    max_residual = 0.0
    for ell in range(L - 1):
        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]

        du_dt = (u_next - u_curr) / dt

        # ν Δu (Laplacian)
        d2u_dx2 = np.zeros(M)
        d2u_dx2[1:-1] = (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]) / dx**2

        # ∇·[u K_φ*u]
        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func)
        flux = u_curr * K_phi_u
        dflux_dx = np.zeros(M)
        dflux_dx[1:-1] = (flux[2:] - flux[:-2]) / (2 * dx)

        # PDE residual
        rhs = nu * d2u_dx2 + dflux_dx
        residual = np.max(np.abs(du_dt[1:-1] - rhs[1:-1]))
        max_residual = max(max_residual, residual)

    print(f"  Max PDE residual: {max_residual:.6e}")

    # Now let's see what happens with E(0) - phi = 0
    print("\n[4] Computing E(φ=0) for comparison...")

    E_zero = 0.0
    for ell in range(L - 1):
        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]

        du_dt = (u_next - u_curr) / dt

        # With phi=0: K_phi*u = 0, Phi*u = 0
        # E(0) = (1/T) ∫∫ [0 + 0 + 0] dx dt = 0
        pass

    print(f"  E(φ=0) = 0.0 (trivially)")

    print("\n[5] Analysis:")
    print(f"  E(φ_true) = {E_true:.6e}")
    print(f"  E(φ=0) = 0.0")
    print(f"  Max PDE residual = {max_residual:.6e}")

    if E_true < 0:
        print(f"\n  E(φ_true) < 0, so φ_true IS the minimizer (correct!)")
    else:
        print(f"\n  E(φ_true) > 0, which is unexpected.")
        print(f"  This could be due to PDE numerical errors.")

    # Also check: what's the "optimal" scale?
    print("\n[6] Scanning over scale factors to find minimum...")
    scales = np.linspace(0.0, 2.0, 21)
    E_values = []
    for scale in scales:
        phi_scaled = lambda r, s=scale: s * true_phi(r, A_phi, sigma_phi)
        Phi_scaled = lambda r, s=scale: s * true_Phi(r, A_phi, sigma_phi)

        E_scale = 0.0
        for ell in range(L - 1):
            u_curr = u_time[ell]
            u_next = u_time[ell + 1]
            dt = t_grid[ell + 1] - t_grid[ell]

            du_dt = (u_next - u_curr) / dt
            du_dx = np.gradient(u_curr, dx)

            K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_scaled)
            Phi_u = compute_Phi_conv_u(x_grid, u_curr, Phi_scaled)

            term1 = np.sum(K_phi_u**2 * u_curr) * dx
            term2 = 2 * np.sum(du_dt * Phi_u) * dx
            term3 = 2 * nu * np.sum(du_dx * K_phi_u) * dx

            E_scale += (term1 + term2 + term3) * dt

        E_scale /= T
        E_values.append(E_scale)

    min_idx = np.argmin(E_values)
    optimal_scale = scales[min_idx]
    min_E = E_values[min_idx]

    print(f"  Scale scan: {scales[0]:.1f} to {scales[-1]:.1f}")
    print(f"  Minimum E at scale = {optimal_scale:.2f}")
    print(f"  E(φ × {optimal_scale:.2f}) = {min_E:.6e}")

    if abs(optimal_scale - 1.0) < 0.2:
        print(f"\n✅ Optimal scale ≈ 1.0, algorithm should work!")
    else:
        print(f"\n❌ Optimal scale = {optimal_scale:.2f} ≠ 1.0")
        print(f"   This indicates the u(x,t) doesn't match the true φ")


if __name__ == '__main__':
    main()
