#!/usr/bin/env python
"""Test quadratic kernel with STABLE PDE solver.

Use implicit/semi-implicit scheme or much smaller dt.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from pathlib import Path

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


def generate_pde_semi_implicit(x_grid, t_grid, nu, phi_func):
    """Semi-implicit scheme: implicit for diffusion, explicit for advection."""
    M = len(x_grid)
    L = len(t_grid)
    dx = x_grid[1] - x_grid[0]

    u = np.zeros((L, M))
    # Initial: mixture of Gaussians
    u[0] = 0.5 * np.exp(-(x_grid - 1)**2 / (2 * 0.25)) / np.sqrt(2 * np.pi * 0.25)
    u[0] += 0.5 * np.exp(-(x_grid + 1)**2 / (2 * 0.25)) / np.sqrt(2 * np.pi * 0.25)
    u[0] /= np.sum(u[0]) * dx

    # Build diffusion matrix (implicit)
    r = nu / dx**2

    for l in range(L - 1):
        dt = t_grid[l + 1] - t_grid[l]
        u_curr = u[l].copy()

        # Explicit advection term
        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func)
        flux = u_curr * K_phi_u
        
        dflux_dx = np.zeros(M)
        dflux_dx[1:-1] = (flux[2:] - flux[:-2]) / (2 * dx)
        dflux_dx[0] = (flux[1] - flux[0]) / dx
        dflux_dx[-1] = (flux[-1] - flux[-2]) / dx

        # Implicit diffusion: (I - dt*nu*D2) u^{n+1} = u^n + dt * advection
        alpha = dt * r
        diag_main = (1 + 2*alpha) * np.ones(M)
        diag_off = -alpha * np.ones(M-1)
        
        # Neumann BC
        diag_main[0] = 1 + alpha
        diag_main[-1] = 1 + alpha
        
        A = diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csr')
        
        rhs = u_curr + dt * dflux_dx
        u[l + 1] = spsolve(A, rhs)
        u[l + 1] = np.maximum(u[l + 1], 0)
        
        # Normalize
        mass = np.sum(u[l + 1]) * dx
        if mass > 1e-10:
            u[l + 1] /= mass

    return u


def test_oracle_stable():
    """Test oracle with stable PDE solver."""
    print("=" * 70)
    print("Oracle Test: Quadratic φ(r)=3r² with Semi-Implicit PDE Solver")
    print("=" * 70)

    nu = 1.0
    x_min, x_max = -10, 10
    M = 300
    L = 10000  # Much finer time stepping
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    print(f"\nParameters: ν={nu}, M={M}, L={L}")
    print(f"dx = {dx:.6f}, dt = {dt:.6f}")

    phi_func = lambda r: 3 * r**2
    Phi_func = lambda r: r**3

    print("\n[1] Generating PDE solution (semi-implicit)...")
    u_time = generate_pde_semi_implicit(x_grid, t_grid, nu, phi_func)

    # Subsample for error functional
    L_sub = 100
    stride = L // L_sub
    t_sub = t_grid[::stride]
    u_sub = u_time[::stride]

    print(f"\n[2] Computing A and b (L_sub={L_sub})...")

    A = 0.0
    b = 0.0

    for ell in range(len(t_sub) - 1):
        u_curr = u_sub[ell]
        u_next = u_sub[ell + 1]
        dt_sub = t_sub[ell + 1] - t_sub[ell]
        T_sub = t_sub[-1] - t_sub[0]

        du_dt = (u_next - u_curr) / dt_sub
        du_dx = np.gradient(u_curr, dx)

        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func)
        Phi_u = compute_Phi_conv_u(x_grid, u_curr, Phi_func)

        A_term = np.sum(K_phi_u**2 * u_curr) * dx
        A += A_term * dt_sub / T_sub

        b_term1 = np.sum(du_dt * Phi_u) * dx
        b_term2 = nu * np.sum(du_dx * K_phi_u) * dx
        b -= (b_term1 + b_term2) * dt_sub / T_sub

    c_opt = b / A if abs(A) > 1e-15 else float('nan')

    print(f"\n  A = {A:.6e}")
    print(f"  b = {b:.6e}")
    print(f"\n  c_opt = {c_opt:.4f}")
    print(f"  Expected: 1.0")
    print(f"  Error: {abs(c_opt - 1.0):.4f}")

    if abs(c_opt - 1.0) < 0.1:
        print(f"\n✅ SUCCESS!")
    else:
        print(f"\n❌ FAIL")

    return c_opt


def main():
    c = test_oracle_stable()
    print(f"\n{'='*70}")
    print(f"RESULT: c_opt = {c:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
