#!/usr/bin/env python
"""Debug PDE solver: check if it correctly implements the mean-field equation.

Mean-field equation: ∂_t u = ν Δu + ∇·(u K_φ*u)

Key check: Conservation of mass (∫u dx = 1) and non-negativity (u ≥ 0)
"""

import numpy as np
import matplotlib.pyplot as plt
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


def generate_pde_euler(x_grid, t_grid, nu, phi_func):
    """Simple Euler method."""
    M = len(x_grid)
    L = len(t_grid)
    dx = x_grid[1] - x_grid[0]

    u = np.zeros((L, M))
    # Paper initial: mixture
    u[0] = 0.5 * np.exp(-(x_grid - 1)**2 / (2 * 0.25)) / np.sqrt(2 * np.pi * 0.25)
    u[0] += 0.5 * np.exp(-(x_grid + 1)**2 / (2 * 0.25)) / np.sqrt(2 * np.pi * 0.25)
    u[0] /= np.sum(u[0]) * dx

    masses = [np.sum(u[0]) * dx]
    max_K = []
    
    for l in range(L - 1):
        dt = t_grid[l + 1] - t_grid[l]
        u_curr = u[l].copy()

        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func)
        max_K.append(np.max(np.abs(K_phi_u)))
        
        flux = u_curr * K_phi_u

        # Central difference for divergence
        dflux_dx = np.zeros(M)
        dflux_dx[1:-1] = (flux[2:] - flux[:-2]) / (2 * dx)
        dflux_dx[0] = (flux[1] - flux[0]) / dx
        dflux_dx[-1] = (flux[-1] - flux[-2]) / dx

        # Laplacian
        d2u_dx2 = np.zeros(M)
        d2u_dx2[1:-1] = (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]) / dx**2

        u[l + 1] = u_curr + dt * (nu * d2u_dx2 + dflux_dx)
        u[l + 1] = np.maximum(u[l + 1], 0)
        mass = np.sum(u[l + 1]) * dx
        if mass > 1e-10:
            u[l + 1] /= mass
        masses.append(np.sum(u[l + 1]) * dx)

    return u, masses, max_K


def main():
    print("=" * 70)
    print("Debug PDE Solver for Quadratic Kernel φ(r) = 3r²")
    print("=" * 70)

    nu = 1.0
    x_min, x_max = -10, 10  # Paper domain
    M = 300  # Paper grid
    L = 1000  # Fine time
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    print(f"\nParameters: ν={nu}, M={M}, L={L}")
    print(f"dx = {dx:.6f}, dt = {dt:.6f}")
    print(f"Domain: [{x_min}, {x_max}]")

    # Quadratic kernel
    phi_func = lambda r: 3 * r**2

    # Check CFL condition for stability
    # For explicit scheme: dt ≤ dx² / (2(C*dx + ν)) where C = max|K_φ*u|
    
    print("\n[1] Running PDE simulation...")
    u_time, masses, max_K = generate_pde_euler(x_grid, t_grid, nu, phi_func)
    
    print(f"\n[2] Conservation check:")
    print(f"  Initial mass: {masses[0]:.6f}")
    print(f"  Final mass: {masses[-1]:.6f}")
    print(f"  Mass variation: {np.max(masses) - np.min(masses):.6e}")
    
    print(f"\n[3] Velocity field K_φ*u:")
    print(f"  Max |K_φ*u|: {np.max(max_K):.4f}")
    print(f"  CFL number (dt * max|K|/dx): {dt * np.max(max_K) / dx:.4f}")
    
    # Check stability condition
    C = np.max(max_K)
    dt_stable = dx**2 / (2 * (C * dx + nu))
    print(f"\n[4] Stability:")
    print(f"  Required dt ≤ {dt_stable:.6f}")
    print(f"  Actual dt = {dt:.6f}")
    print(f"  {'✅ Stable' if dt <= dt_stable else '❌ UNSTABLE!'}")

    # Check if solution is reasonable
    print(f"\n[5] Solution check:")
    print(f"  u(t=0) max: {np.max(u_time[0]):.4f}")
    print(f"  u(t=T) max: {np.max(u_time[-1]):.4f}")
    print(f"  u(t=T) min: {np.min(u_time[-1]):.6e}")
    
    # For quadratic kernel, check the expected behavior
    # K_φ(x) = 3r² * sign(x) means particles are repelled from the origin
    print(f"\n[6] Physical interpretation:")
    print(f"  φ(r) = 3r² is repulsive (positive), particles spread out")
    
    # Compute the "drift" direction
    K_at_center = compute_K_phi_conv_u(x_grid, u_time[0], phi_func)
    print(f"  K_φ*u at x=0 (initial): {K_at_center[M//2]:.6f}")
    print(f"  K_φ*u at x=1 (initial): {K_at_center[M//2 + int(1/dx)]:.6f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax = axes[0]
    for l in [0, L//4, L//2, 3*L//4, L-1]:
        ax.plot(x_grid, u_time[l], label=f't={t_grid[l]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Solution u(x,t)')
    ax.legend()
    ax.grid(True)
    
    ax = axes[1]
    ax.plot(t_grid[1:], max_K)
    ax.set_xlabel('t')
    ax.set_ylabel('max |K_φ*u|')
    ax.set_title('Velocity field magnitude')
    ax.grid(True)
    
    ax = axes[2]
    ax.plot(t_grid, masses)
    ax.set_xlabel('t')
    ax.set_ylabel('Mass')
    ax.set_title('Mass conservation')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiments/ips_unlabeled/img/debug_pde_quadratic.png', dpi=150)
    plt.close()
    print(f"\nPlot saved to experiments/ips_unlabeled/img/debug_pde_quadratic.png")


if __name__ == '__main__':
    main()
