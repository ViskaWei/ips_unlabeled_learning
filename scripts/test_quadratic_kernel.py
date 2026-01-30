#!/usr/bin/env python
"""Test with the ACTUAL cubic potential from Fei Lu paper.

Paper Section 4.2: Φ(x) = |x|³ => φ(r) = 3r²
This is a QUADRATIC kernel, not cubic polynomial.
"""

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def quadratic_phi(r):
    """φ(r) = 3r² (derivative of Φ(r) = r³)"""
    return 3 * r**2


def quadratic_Phi(r):
    """Φ(r) = r³"""
    return r**3


def compute_K_phi_conv_u(x_grid, u, phi_func):
    """Compute K_φ * u where K_φ(x-y) = φ(|x-y|) * sign(x-y).
    
    In 1D: K_φ(x) = φ(|x|) * sign(x) = φ(|x|) * x/|x|
    """
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
    # Initial: mixture of two Gaussians (like paper)
    u[0] = 0.5 * np.exp(-(x_grid - 1)**2 / (2 * 0.25)) / np.sqrt(2 * np.pi * 0.25)
    u[0] += 0.5 * np.exp(-(x_grid + 1)**2 / (2 * 0.25)) / np.sqrt(2 * np.pi * 0.25)
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


def test_oracle():
    """Test oracle basis with quadratic kernel."""
    print("=" * 70)
    print("Oracle Test: Quadratic Kernel φ(r) = 3r² (Paper's Cubic Potential)")
    print("=" * 70)

    nu = 1.0  # Paper uses ν=1.0 for cubic potential
    x_min, x_max = -4, 4
    M = 80
    L = 100
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]

    print(f"\nParameters: ν={nu}, M={M}, L={L}, domain=[{x_min}, {x_max}]")

    # Generate PDE solution
    print("\n[1] Generating PDE solution...")
    u_time = generate_pde_solution(x_grid, t_grid, nu, quadratic_phi)

    # Compute A and b
    print("\n[2] Computing A and b...")

    A = 0.0
    b = 0.0

    for ell in range(L - 1):
        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]

        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)

        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, quadratic_phi)
        Phi_u = compute_Phi_conv_u(x_grid, u_curr, quadratic_Phi)

        # A term: (K_φ*u)² u
        A_term = np.sum(K_phi_u**2 * u_curr) * dx
        A += A_term * dt / T

        # b term: -[∂_t u (Φ*u) + ν ∇u · (K_φ*u)]
        b_term1 = np.sum(du_dt * Phi_u) * dx
        b_term2 = nu * np.sum(du_dx * K_phi_u) * dx
        b -= (b_term1 + b_term2) * dt / T

    print(f"  A = {A:.6e}")
    print(f"  b = {b:.6e}")

    if abs(A) > 1e-15:
        c_opt = b / A
    else:
        c_opt = float('nan')

    print(f"\n  c_opt = b / A = {c_opt:.6f}")
    print(f"  Expected c_opt = 1.0")
    print(f"  Error = |c_opt - 1| = {abs(c_opt - 1.0):.6f}")

    if abs(c_opt - 1.0) < 0.1:
        print(f"\n✅ SUCCESS: c_opt ≈ 1.0")
    else:
        print(f"\n❌ FAIL: c_opt = {c_opt:.4f} ≠ 1.0")

    return c_opt


def main():
    print("\n" + "=" * 70)
    print("Testing Fei Lu Paper's ACTUAL Cubic Potential")
    print("Paper Section 4.2: Φ(x) = |x|³ => φ(r) = 3r² (QUADRATIC kernel)")
    print("=" * 70)
    
    c_opt = test_oracle()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Quadratic kernel φ(r) = 3r²: c_opt = {c_opt:.4f}")
    print(f"Status: {'✅ PASS' if abs(c_opt - 1.0) < 0.1 else '❌ FAIL'}")


if __name__ == '__main__':
    main()
