#!/usr/bin/env python
"""Verify error functional for cubic potential with oracle basis.

This is the key test: if c_opt ≠ 1.0, then the error functional
has a fundamental issue with non-localized potentials.
"""

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def cubic_phi(r, a=1.0, b=1.0):
    """φ(r) = a*r - b*r³ (odd function)."""
    return a * r - b * r**3


def cubic_Phi(r, a=1.0, b=1.0):
    """Φ(r) = ∫φ dr = a*r²/2 - b*r⁴/4."""
    return a * r**2 / 2 - b * r**4 / 4


def compute_K_phi_conv_u(x_grid, u, phi_func, r_max=None):
    """Compute K_φ * u where K_φ(x-y) = φ(|x-y|) * sign(x-y)."""
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    result = np.zeros(M)
    for m, x in enumerate(x_grid):
        for n, y in enumerate(x_grid):
            r = abs(x - y)
            if r_max is not None and r > r_max:
                continue
            if r > 1e-10:
                result[m] += phi_func(r) * np.sign(x - y) * u[n] * dx
    return result


def compute_Phi_conv_u(x_grid, u, Phi_func, r_max=None):
    """Compute Φ * u."""
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    result = np.zeros(M)
    for m, x in enumerate(x_grid):
        for n, y in enumerate(x_grid):
            r = abs(x - y)
            if r_max is not None and r > r_max:
                continue
            result[m] += Phi_func(r) * u[n] * dx
    return result


def generate_pde_solution(x_grid, t_grid, nu, phi_func, r_max=None):
    """Generate PDE solution: ∂_t u = ν Δu + ∇·(u K_φ*u)."""
    M = len(x_grid)
    L = len(t_grid)
    dx = x_grid[1] - x_grid[0]

    u = np.zeros((L, M))
    # Initial condition: Gaussian
    u[0] = np.exp(-x_grid**2 / 2) / np.sqrt(2 * np.pi)
    u[0] /= np.sum(u[0]) * dx

    for l in range(L - 1):
        dt = t_grid[l + 1] - t_grid[l]
        u_curr = u[l].copy()

        # Convolution K_φ * u
        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func, r_max)

        # Flux = u * K_φ*u
        flux = u_curr * K_phi_u

        # ∇·(flux)
        dflux_dx = np.zeros(M)
        dflux_dx[1:-1] = (flux[2:] - flux[:-2]) / (2 * dx)
        dflux_dx[0] = (flux[1] - flux[0]) / dx
        dflux_dx[-1] = (flux[-1] - flux[-2]) / dx

        # ν Δu
        d2u_dx2 = np.zeros(M)
        d2u_dx2[1:-1] = (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]) / dx**2

        # Update
        u[l + 1] = u_curr + dt * (nu * d2u_dx2 + dflux_dx)
        u[l + 1] = np.maximum(u[l + 1], 0)
        mass = np.sum(u[l + 1]) * dx
        if mass > 1e-10:
            u[l + 1] /= mass

    return u


def test_oracle(name, phi_func, Phi_func, r_max=None):
    """Test oracle basis for a given potential."""
    print(f"\n{'='*60}")
    print(f"Oracle Test: {name}")
    print(f"{'='*60}")

    nu = 0.1
    x_min, x_max = -6, 6
    M = 100
    L = 100
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]

    print(f"Parameters: ν={nu}, M={M}, L={L}, domain=[{x_min}, {x_max}]")
    if r_max is not None:
        print(f"r_max cutoff: {r_max}")

    # Generate PDE solution with true φ
    print("\n[1] Generating PDE solution...")
    u_time = generate_pde_solution(x_grid, t_grid, nu, phi_func, r_max)

    # Compute A and b with oracle basis
    print("\n[2] Computing A and b...")

    A = 0.0
    b = 0.0

    for ell in range(L - 1):
        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]

        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)

        # Convolutions
        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_func, r_max)
        Phi_u = compute_Phi_conv_u(x_grid, u_curr, Phi_func, r_max)

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

    # Verify E(c)
    E_0 = 0.0
    E_1 = A - 2 * b
    E_c = c_opt**2 * A - 2 * c_opt * b if not np.isnan(c_opt) else float('nan')

    print(f"\n  E(0) = {E_0:.6e}")
    print(f"  E(1) = {E_1:.6e}")
    print(f"  E(c_opt) = {E_c:.6e}")

    success = abs(c_opt - 1.0) < 0.1
    if success:
        print(f"\n✅ SUCCESS: c_opt ≈ 1.0")
    else:
        print(f"\n❌ FAIL: c_opt = {c_opt:.4f} ≠ 1.0")

    return c_opt, success


def main():
    print("=" * 70)
    print("Verify Error Functional for Different Potentials (Oracle Test)")
    print("=" * 70)

    # Test 1: Gaussian (should pass)
    gaussian_phi = lambda r: -r * np.exp(-r**2 / 2)
    gaussian_Phi = lambda r: np.exp(-r**2 / 2)
    c1, s1 = test_oracle("Gaussian φ(r) = -r exp(-r²/2)", gaussian_phi, gaussian_Phi)

    # Test 2: Cubic without cutoff
    cubic_phi_func = lambda r: r - r**3
    cubic_Phi_func = lambda r: r**2 / 2 - r**4 / 4
    c2, s2 = test_oracle("Cubic φ(r) = r - r³ (no cutoff)", cubic_phi_func, cubic_Phi_func)

    # Test 3: Cubic with cutoff (like paper)
    c3, s3 = test_oracle("Cubic φ(r) = r - r³ (r_max=2)", cubic_phi_func, cubic_Phi_func, r_max=2.0)

    # Test 4: Cubic with larger cutoff
    c4, s4 = test_oracle("Cubic φ(r) = r - r³ (r_max=4)", cubic_phi_func, cubic_Phi_func, r_max=4.0)

    # Test 5: Linear (simplest case)
    linear_phi = lambda r: r
    linear_Phi = lambda r: r**2 / 2
    c5, s5 = test_oracle("Linear φ(r) = r (r_max=2)", linear_phi, linear_Phi, r_max=2.0)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Potential':<35} {'c_opt':>10} {'Status':>10}")
    print("-" * 60)
    print(f"{'Gaussian (no cutoff)':<35} {c1:>10.4f} {'✅' if s1 else '❌':>10}")
    print(f"{'Cubic (no cutoff)':<35} {c2:>10.4f} {'✅' if s2 else '❌':>10}")
    print(f"{'Cubic (r_max=2)':<35} {c3:>10.4f} {'✅' if s3 else '❌':>10}")
    print(f"{'Cubic (r_max=4)':<35} {c4:>10.4f} {'✅' if s4 else '❌':>10}")
    print(f"{'Linear (r_max=2)':<35} {c5:>10.4f} {'✅' if s5 else '❌':>10}")


if __name__ == '__main__':
    main()
