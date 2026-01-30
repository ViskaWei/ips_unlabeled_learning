#!/usr/bin/env python
"""Test with CORRECTED formula from Fei Lu paper.

Paper Eq 2.13:
b_i = -1/T ∫∫ [∂_t u (Φ_i * u) - ν ∇u · (K_φi * u)] dx dt

Note: It's MINUS ν∇u·(K*u), not PLUS!
"""

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def compute_K_phi_conv_u(x_grid, u, phi_func):
    """Compute K_φ * u where K_φ(x-y) = φ(|x-y|) * sign(x-y)."""
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
    """Compute Φ * u."""
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    result = np.zeros(M)
    for m, x in enumerate(x_grid):
        for n, y in enumerate(x_grid):
            r = abs(x - y)
            result[m] += Phi_func(r) * u[n] * dx
    return result


def generate_pde_solution(x_grid, t_grid, nu, phi_func):
    """Generate PDE solution: ∂_t u = ν Δu + ∇·(u K_φ*u)."""
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


def test_formula_variants(name, phi_func, Phi_func, nu=0.1):
    """Test different formula variants."""
    print(f"\n{'='*60}")
    print(f"Test: {name}, ν={nu}")
    print(f"{'='*60}")

    x_min, x_max = -4, 4
    M = 80
    L = 100
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]

    # Generate PDE
    u_time = generate_pde_solution(x_grid, t_grid, nu, phi_func)

    # Compute A
    A = 0.0
    for ell in range(L - 1):
        u_curr = u_time[ell]
        dt = t_grid[ell + 1] - t_grid[ell]
        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func)
        A_term = np.sum(K_phi_u**2 * u_curr) * dx
        A += A_term * dt / T

    # Compute b with ORIGINAL formula: b = -[∂_t u (Φ*u) + ν ∇u·(K*u)]
    b_original = 0.0
    for ell in range(L - 1):
        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]
        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)
        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func)
        Phi_u = compute_Phi_conv_u(x_grid, u_curr, Phi_func)
        b_term1 = np.sum(du_dt * Phi_u) * dx
        b_term2 = nu * np.sum(du_dx * K_phi_u) * dx
        b_original -= (b_term1 + b_term2) * dt / T  # ORIGINAL: + term2

    # Compute b with CORRECTED formula: b = -[∂_t u (Φ*u) - ν ∇u·(K*u)]
    b_corrected = 0.0
    for ell in range(L - 1):
        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]
        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)
        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func)
        Phi_u = compute_Phi_conv_u(x_grid, u_curr, Phi_func)
        b_term1 = np.sum(du_dt * Phi_u) * dx
        b_term2 = nu * np.sum(du_dx * K_phi_u) * dx
        b_corrected -= (b_term1 - b_term2) * dt / T  # CORRECTED: - term2

    c_original = b_original / A if abs(A) > 1e-15 else float('nan')
    c_corrected = b_corrected / A if abs(A) > 1e-15 else float('nan')

    print(f"  A = {A:.6e}")
    print(f"\n  ORIGINAL formula (+ ν term):")
    print(f"    b = {b_original:.6e}")
    print(f"    c_opt = {c_original:.4f} {'✅' if abs(c_original - 1.0) < 0.1 else '❌'}")
    print(f"\n  CORRECTED formula (- ν term):")
    print(f"    b = {b_corrected:.6e}")
    print(f"    c_opt = {c_corrected:.4f} {'✅' if abs(c_corrected - 1.0) < 0.1 else '❌'}")

    return c_original, c_corrected


def main():
    print("=" * 70)
    print("Testing Formula Variants for Error Functional")
    print("=" * 70)

    # Gaussian (localized)
    gaussian_phi = lambda r: -r * np.exp(-r**2 / 2)
    gaussian_Phi = lambda r: np.exp(-r**2 / 2)
    c1_o, c1_c = test_formula_variants("Gaussian φ(r) = -r exp(-r²/2)", gaussian_phi, gaussian_Phi)

    # Quadratic (paper's "cubic potential")
    quadratic_phi = lambda r: 3 * r**2
    quadratic_Phi = lambda r: r**3
    c2_o, c2_c = test_formula_variants("Quadratic φ(r) = 3r² (paper's cubic)", quadratic_phi, quadratic_Phi, nu=1.0)

    # Linear
    linear_phi = lambda r: r
    linear_Phi = lambda r: r**2 / 2
    c3_o, c3_c = test_formula_variants("Linear φ(r) = r", linear_phi, linear_Phi)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Kernel':<30} {'Original':>12} {'Corrected':>12}")
    print("-" * 55)
    print(f"{'Gaussian':<30} {c1_o:>12.4f} {c1_c:>12.4f}")
    print(f"{'Quadratic (3r²)':<30} {c2_o:>12.4f} {c2_c:>12.4f}")
    print(f"{'Linear (r)':<30} {c3_o:>12.4f} {c3_c:>12.4f}")


if __name__ == '__main__':
    main()
