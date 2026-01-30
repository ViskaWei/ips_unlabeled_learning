#!/usr/bin/env python
"""Test with oracle basis: use true phi as basis function.

If c_opt ≈ 1.0, the A and b computation is correct.
"""

import numpy as np
from scipy import integrate
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
    print("Test with Oracle Basis (True φ shape)")
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

    # Oracle basis: just the true phi
    phi_basis = lambda r: true_phi(r, A_phi, sigma_phi)
    Phi_basis = lambda r: true_Phi(r, A_phi, sigma_phi)

    # True phi used in PDE
    phi_func = lambda r: true_phi(r, A_phi, sigma_phi)

    # Generate PDE solution
    print("\n[1] Generating PDE solution...")
    u_time = generate_pde_solution(x_grid, t_grid, nu, phi_func)

    # Compute A (scalar) and b (scalar) for single-basis case
    print("\n[2] Computing A and b with oracle basis...")

    A = 0.0
    b = 0.0

    for ell in range(L - 1):
        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]

        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)

        # Convolutions with oracle basis
        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_basis)
        Phi_u = compute_Phi_conv_u(x_grid, u_curr, Phi_basis)

        # A contribution: (K_φ*u)² u
        A_term = np.sum(K_phi_u**2 * u_curr) * dx
        A += A_term * dt / T

        # b contribution: -[∂_t u (Φ*u) + ν ∇u · (K_φ*u)]
        b_term1 = np.sum(du_dt * Phi_u) * dx
        b_term2 = nu * np.sum(du_dx * K_phi_u) * dx
        b -= (b_term1 + b_term2) * dt / T

    print(f"  A = {A:.6e}")
    print(f"  b = {b:.6e}")

    # Solve for c: A c = b
    if abs(A) > 1e-15:
        c_opt = b / A
    else:
        c_opt = 0.0

    print(f"\n  c_opt = b / A = {c_opt:.6f}")

    # Expected: c_opt ≈ 1.0
    print(f"\n  Expected c_opt = 1.0")
    print(f"  Error = |c_opt - 1| = {abs(c_opt - 1.0):.6f}")

    if abs(c_opt - 1.0) < 0.1:
        print(f"\n✅ SUCCESS: c_opt ≈ 1.0, A and b computation is correct!")
    else:
        print(f"\n❌ FAIL: c_opt = {c_opt:.4f} ≠ 1.0")
        print(f"   This suggests a bug in A or b computation.")

    # Let's also verify by computing E(c) directly
    print("\n[3] Verifying E(c) computation...")

    # E(c) = c² A - 2 c b + const
    # dE/dc = 2 c A - 2 b = 0 => c* = b/A

    E_at_0 = 0.0  # E(0) = 0 (since K_0*u = 0)
    E_at_1 = A - 2 * b  # E(1) = A - 2b
    E_at_c = c_opt**2 * A - 2 * c_opt * b  # E(c_opt)

    print(f"  E(0) = {E_at_0:.6e}")
    print(f"  E(1) = {E_at_1:.6e}")
    print(f"  E(c_opt={c_opt:.2f}) = {E_at_c:.6e}")

    # E(c_opt) should be minimum
    if E_at_c <= E_at_1 and E_at_c <= E_at_0:
        print(f"\n  E(c_opt) is the minimum ✓")
    else:
        print(f"\n  E(c_opt) is NOT the minimum ✗")


if __name__ == '__main__':
    main()
