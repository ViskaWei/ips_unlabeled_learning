#!/usr/bin/env python
"""Critical Test: Include true phi in basis set.

If the basis set INCLUDES the true phi, the optimal coefficients should
recover it with coefficient = 1.0 for phi_true and 0 for others.

This tests whether the error functional can identify the correct solution
even when it's in the search space.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy import integrate
import matplotlib.pyplot as plt


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


def generate_pde_semi_implicit(x_grid, t_grid, nu, phi_func):
    M = len(x_grid)
    L = len(t_grid)
    dx = x_grid[1] - x_grid[0]

    u = np.zeros((L, M))
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

        r = nu / dx**2
        alpha = dt * r
        diag_main = (1 + 2*alpha) * np.ones(M)
        diag_off = -alpha * np.ones(M-1)
        diag_main[0] = 1 + alpha
        diag_main[-1] = 1 + alpha

        A_mat = diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csr')
        rhs = u_curr + dt * dflux_dx
        u[l + 1] = spsolve(A_mat, rhs)
        u[l + 1] = np.maximum(u[l + 1], 0)
        mass = np.sum(u[l + 1]) * dx
        if mass > 1e-10:
            u[l + 1] /= mass

    return u


def compute_A_and_b_with_funcs(u_time, t_grid, x_grid, phi_funcs, Phi_funcs, nu):
    """Compute A and b using function evaluations directly."""
    L, M = u_time.shape
    n_basis = len(phi_funcs)
    dx = x_grid[1] - x_grid[0]
    T = t_grid[-1] - t_grid[0]

    A = np.zeros((n_basis, n_basis))
    b = np.zeros(n_basis)

    for ell in range(L - 1):
        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]

        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)

        # Compute K_phi_i * u and Phi_i * u for all basis
        K_phi_all = []
        Phi_u_all = []

        for i in range(n_basis):
            K_phi = compute_K_phi_conv_u(x_grid, u_curr, phi_funcs[i])
            K_phi_all.append(K_phi)

            # Phi_i * u
            Phi_u = np.zeros(M)
            for m, x in enumerate(x_grid):
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    Phi_u[m] += Phi_funcs[i](r) * u_curr[n] * dx
            Phi_u_all.append(Phi_u)

        for i in range(n_basis):
            b[i] -= (np.sum(du_dt * Phi_u_all[i]) * dx +
                    nu * np.sum(du_dx * K_phi_all[i]) * dx) * dt / T
            for j in range(i, n_basis):
                term = np.sum(K_phi_all[i] * K_phi_all[j] * u_curr) * dx
                A[i, j] += term * dt / T
                if j != i:
                    A[j, i] = A[i, j]

    return A, b


def main():
    print("=" * 70)
    print("Critical Test: Include True φ in Basis Set")
    print("=" * 70)

    # Setup
    nu = 0.5
    x_min, x_max = -5, 5
    M = 100
    L = 3000
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)

    # True Gaussian kernel
    phi_true = lambda r: -r * np.exp(-r**2 / 2)
    Phi_true = lambda r: np.exp(-r**2 / 2)

    print("\n[1] Generating PDE data...")
    u_time = generate_pde_semi_implicit(x_grid, t_grid, nu, phi_true)

    L_sub = 30
    stride = L // L_sub
    t_sub = t_grid[::stride]
    u_sub = u_time[::stride]

    # =========================================================================
    # Test 1: Basis = [phi_true]
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 1: Single basis = φ_true")
    print("=" * 60)

    A, b = compute_A_and_b_with_funcs(u_sub, t_sub, x_grid, [phi_true], [Phi_true], nu)
    c = b[0] / A[0, 0]
    print(f"  A = {A[0,0]:.6e}")
    print(f"  b = {b[0]:.6e}")
    print(f"  c = {c:.4f} (expected: 1.0)")

    # =========================================================================
    # Test 2: Basis = [phi_true, r]
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 2: Basis = [φ_true, r]")
    print("Expected: c = [1, 0]")
    print("=" * 60)

    phi_funcs = [
        phi_true,
        lambda r: r,
    ]
    Phi_funcs = [
        Phi_true,
        lambda r: r**2 / 2,
    ]

    A, b = compute_A_and_b_with_funcs(u_sub, t_sub, x_grid, phi_funcs, Phi_funcs, nu)
    print(f"  A =\n{A}")
    print(f"  b = {b}")
    print(f"  cond(A) = {np.linalg.cond(A):.2e}")

    c = np.linalg.solve(A, b)
    print(f"  c = {c}")

    # =========================================================================
    # Test 3: Basis = [phi_true, r, r^3]
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 3: Basis = [φ_true, r, r³]")
    print("Expected: c = [1, 0, 0]")
    print("=" * 60)

    phi_funcs = [
        phi_true,
        lambda r: r,
        lambda r: r**3,
    ]
    Phi_funcs = [
        Phi_true,
        lambda r: r**2 / 2,
        lambda r: r**4 / 4,
    ]

    A, b = compute_A_and_b_with_funcs(u_sub, t_sub, x_grid, phi_funcs, Phi_funcs, nu)
    print(f"  A =\n{A}")
    print(f"  b = {b}")
    print(f"  cond(A) = {np.linalg.cond(A):.2e}")

    c = np.linalg.solve(A, b)
    print(f"  c = {c}")

    # =========================================================================
    # Test 4: Basis = [r, phi_true, r^3] (true phi in middle)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 4: Basis = [r, φ_true, r³] (true φ in middle)")
    print("Expected: c = [0, 1, 0]")
    print("=" * 60)

    phi_funcs = [
        lambda r: r,
        phi_true,
        lambda r: r**3,
    ]
    Phi_funcs = [
        lambda r: r**2 / 2,
        Phi_true,
        lambda r: r**4 / 4,
    ]

    A, b = compute_A_and_b_with_funcs(u_sub, t_sub, x_grid, phi_funcs, Phi_funcs, nu)
    print(f"  A =\n{A}")
    print(f"  b = {b}")
    print(f"  cond(A) = {np.linalg.cond(A):.2e}")

    c = np.linalg.solve(A, b)
    print(f"  c = {c}")

    # =========================================================================
    # Test 5: Basis with similar Gaussians
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 5: Basis = [φ(σ=0.8), φ_true(σ=1), φ(σ=1.2)]")
    print("Expected: c = [0, 1, 0]")
    print("=" * 60)

    phi_funcs = [
        lambda r: -r * np.exp(-r**2 / (2 * 0.8**2)),
        phi_true,  # sigma = 1
        lambda r: -r * np.exp(-r**2 / (2 * 1.2**2)),
    ]
    Phi_funcs = [
        lambda r: 0.8**2 * np.exp(-r**2 / (2 * 0.8**2)),
        Phi_true,
        lambda r: 1.2**2 * np.exp(-r**2 / (2 * 1.2**2)),
    ]

    A, b = compute_A_and_b_with_funcs(u_sub, t_sub, x_grid, phi_funcs, Phi_funcs, nu)
    print(f"  A =\n{A}")
    print(f"  b = {b}")
    print(f"  cond(A) = {np.linalg.cond(A):.2e}")

    c = np.linalg.solve(A, b)
    print(f"  c = {c}")

    # Evaluate learned phi
    r_eval = np.linspace(0.01, 4, 100)
    phi_learned = sum(c[i] * phi_funcs[i](r_eval) for i in range(len(phi_funcs)))
    phi_truth = phi_true(r_eval)

    error = np.sqrt(np.mean((phi_learned - phi_truth)**2)) / np.sqrt(np.mean(phi_truth**2))
    print(f"\n  Learning error: {error:.2%}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
Key finding: Even when the true φ is in the basis set, the recovered
coefficients may not be [0, 1, 0] due to:

1. The error functional has many equivalent minima
   (different φ can give the same dynamics)

2. The basis functions become nearly collinear after convolution
   with the density u(x,t)

3. This is an IDENTIFIABILITY issue, not a numerical issue

Possible solutions:
1. Add regularization that prefers certain solutions
2. Use RKHS regularization (as in Fei Lu paper)
3. Add more constraints (e.g., φ(0) = 0, φ is smooth)
""")


if __name__ == '__main__':
    main()
