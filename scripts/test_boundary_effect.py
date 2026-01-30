#!/usr/bin/env python
"""Test boundary effects on different kernels.

Hypothesis: For non-localized kernels, the domain boundary affects the error functional.
"""

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


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


def test_with_domain(name, phi_func, Phi_func, domain_size, nu=0.1):
    """Test error functional with different domain sizes."""
    
    x_min, x_max = -domain_size, domain_size
    M = int(80 * domain_size / 4)  # Scale grid with domain
    L = 100
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]

    # Generate PDE
    u_time = generate_pde_solution(x_grid, t_grid, nu, phi_func)

    # Compute A and b
    A = 0.0
    b = 0.0

    for ell in range(L - 1):
        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]
        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)
        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func)
        Phi_u = compute_Phi_conv_u(x_grid, u_curr, Phi_func)
        
        A_term = np.sum(K_phi_u**2 * u_curr) * dx
        A += A_term * dt / T
        
        b_term1 = np.sum(du_dt * Phi_u) * dx
        b_term2 = nu * np.sum(du_dx * K_phi_u) * dx
        b -= (b_term1 + b_term2) * dt / T

    c_opt = b / A if abs(A) > 1e-15 else float('nan')
    return c_opt


def main():
    print("=" * 70)
    print("Testing Domain Size Effect on Different Kernels")
    print("=" * 70)
    
    domains = [4, 6, 8, 10, 15, 20]
    
    # Gaussian (localized)
    gaussian_phi = lambda r: -r * np.exp(-r**2 / 2)
    gaussian_Phi = lambda r: np.exp(-r**2 / 2)
    
    # Quadratic (paper's cubic)
    quadratic_phi = lambda r: 3 * r**2
    quadratic_Phi = lambda r: r**3
    
    # Linear
    linear_phi = lambda r: r
    linear_Phi = lambda r: r**2 / 2
    
    print(f"\n{'Domain':>10} {'Gaussian':>12} {'Linear':>12} {'Quadratic':>12}")
    print("-" * 50)
    
    for d in domains:
        c_g = test_with_domain("Gaussian", gaussian_phi, gaussian_Phi, d)
        c_l = test_with_domain("Linear", linear_phi, linear_Phi, d)
        c_q = test_with_domain("Quadratic", quadratic_phi, quadratic_Phi, d, nu=1.0)
        
        g_status = "✅" if abs(c_g - 1.0) < 0.1 else "❌"
        l_status = "✅" if abs(c_l - 1.0) < 0.1 else "❌"
        q_status = "✅" if abs(c_q - 1.0) < 0.1 else "❌"
        
        print(f"{d:>10} {c_g:>10.4f}{g_status:>2} {c_l:>10.4f}{l_status:>2} {c_q:>10.4f}{q_status:>2}")


if __name__ == '__main__':
    main()
