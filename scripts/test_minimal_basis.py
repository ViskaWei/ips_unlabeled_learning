#!/usr/bin/env python
"""Test with MINIMAL basis to avoid ill-conditioning.

Key insight: More basis functions → more collinearity → worse conditioning.
Try: use only 2-3 basis functions that can approximate phi_true.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time


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


def compute_A_and_b(u_time, t_grid, x_grid, phi_funcs, Phi_funcs, nu):
    """Compute A and b using function evaluations."""
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

        K_phi_all = []
        Phi_u_all = []

        for i in range(n_basis):
            K_phi = compute_K_phi_conv_u(x_grid, u_curr, phi_funcs[i])
            K_phi_all.append(K_phi)

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
    print("Test: Minimal Basis Functions (avoid ill-conditioning)")
    print("=" * 70)

    # Setup
    nu = 0.5
    x_min, x_max = -5, 5
    M = 80  # Smaller grid for speed
    L = 2000
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)

    phi_true = lambda r: -r * np.exp(-r**2 / 2)
    Phi_true = lambda r: np.exp(-r**2 / 2)

    print("\n[1] Generating PDE data...")
    t0 = time.time()
    u_time = generate_pde_semi_implicit(x_grid, t_grid, nu, phi_true)
    print(f"  Time: {time.time() - t0:.1f}s")

    L_sub = 20  # Fewer time points for speed
    stride = L // L_sub
    t_sub = t_grid[::stride]
    u_sub = u_time[::stride]

    results = []

    # =========================================================================
    # Test 1: Single basis = phi_true (Oracle)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 1: Oracle (single basis = phi_true)")
    print("=" * 60)

    A, b = compute_A_and_b(u_sub, t_sub, x_grid, [phi_true], [Phi_true], nu)
    c = b[0] / A[0, 0]
    print(f"  c_opt = {c:.4f} (expected: 1.0)")
    results.append(('Oracle', c, 1.0 - c, 'PASS' if abs(c - 1.0) < 0.1 else 'FAIL'))

    # =========================================================================
    # Test 2: Two basis - phi_true + linear
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 2: Two basis [phi_true, r]")
    print("Expected: c = [1, 0]")
    print("=" * 60)

    phi_funcs = [phi_true, lambda r: r]
    Phi_funcs = [Phi_true, lambda r: r**2 / 2]

    A, b = compute_A_and_b(u_sub, t_sub, x_grid, phi_funcs, Phi_funcs, nu)
    print(f"  cond(A) = {np.linalg.cond(A):.2e}")
    c_opt = np.linalg.solve(A, b)
    print(f"  c_opt = {c_opt}")

    # Evaluate
    r_eval = np.linspace(0.01, 3, 50)
    phi_learned = c_opt[0] * phi_true(r_eval) + c_opt[1] * r_eval
    phi_truth = phi_true(r_eval)
    err = np.sqrt(np.mean((phi_learned - phi_truth)**2)) / np.sqrt(np.mean(phi_truth**2))
    print(f"  Error: {err:.2%}")
    results.append(('[phi_true, r]', c_opt, err, 'PASS' if err < 0.1 else 'FAIL'))

    # =========================================================================
    # Test 3: Two basis - linear + cubic (NO phi_true)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 3: Two basis [r, r³] (no phi_true)")
    print("=" * 60)

    phi_funcs = [lambda r: r, lambda r: r**3]
    Phi_funcs = [lambda r: r**2 / 2, lambda r: r**4 / 4]

    # Check if they can fit phi_true
    r_fit = np.linspace(0.01, 3, 50)
    B = np.column_stack([r_fit, r_fit**3])
    c_fit = np.linalg.lstsq(B, phi_true(r_fit), rcond=None)[0]
    fit_err = np.sqrt(np.mean((B @ c_fit - phi_true(r_fit))**2)) / np.sqrt(np.mean(phi_true(r_fit)**2))
    print(f"  Basis fit of phi_true: {fit_err:.2%}")
    print(f"  Fit coefficients: {c_fit}")

    A, b = compute_A_and_b(u_sub, t_sub, x_grid, phi_funcs, Phi_funcs, nu)
    print(f"  cond(A) = {np.linalg.cond(A):.2e}")
    c_opt = np.linalg.solve(A, b)
    print(f"  c_opt = {c_opt}")

    phi_learned = c_opt[0] * r_eval + c_opt[1] * r_eval**3
    err = np.sqrt(np.mean((phi_learned - phi_true(r_eval))**2)) / np.sqrt(np.mean(phi_true(r_eval)**2))
    print(f"  Learning error: {err:.2%}")
    results.append(('[r, r³]', c_opt, err, 'PASS' if err < 0.3 else 'FAIL'))

    # =========================================================================
    # Test 4: Three basis - [r, r*exp(-r²/4), r³]
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 4: Three basis [r, r*exp(-r²/4), r³]")
    print("(Different sigma from true)")
    print("=" * 60)

    phi_funcs = [
        lambda r: r,
        lambda r: -r * np.exp(-r**2 / 4),  # sigma=sqrt(2), not 1
        lambda r: r**3
    ]
    Phi_funcs = [
        lambda r: r**2 / 2,
        lambda r: 2 * np.exp(-r**2 / 4),  # = sigma^2 * exp(-r^2/(2*sigma^2))
        lambda r: r**4 / 4
    ]

    # Check fit
    B = np.column_stack([r_fit, -r_fit * np.exp(-r_fit**2 / 4), r_fit**3])
    c_fit = np.linalg.lstsq(B, phi_true(r_fit), rcond=None)[0]
    fit_err = np.sqrt(np.mean((B @ c_fit - phi_true(r_fit))**2)) / np.sqrt(np.mean(phi_true(r_fit)**2))
    print(f"  Basis fit of phi_true: {fit_err:.2%}")

    A, b = compute_A_and_b(u_sub, t_sub, x_grid, phi_funcs, Phi_funcs, nu)
    print(f"  cond(A) = {np.linalg.cond(A):.2e}")
    c_opt = np.linalg.solve(A, b)
    print(f"  c_opt = {c_opt}")

    phi_learned = c_opt[0] * r_eval + c_opt[1] * (-r_eval * np.exp(-r_eval**2 / 4)) + c_opt[2] * r_eval**3
    err = np.sqrt(np.mean((phi_learned - phi_true(r_eval))**2)) / np.sqrt(np.mean(phi_true(r_eval)**2))
    print(f"  Learning error: {err:.2%}")
    results.append(('[r, r*exp(-r²/4), r³]', c_opt, err, 'PASS' if err < 0.3 else 'FAIL'))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, c, err, status in results:
        if isinstance(err, float):
            print(f"  {name:<25}: error={err:.2%}, {status}")
        else:
            print(f"  {name:<25}: c_opt={c:.4f}, {status}")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
Even with minimal basis (2-3 functions), the learned coefficients
differ from the "ideal" coefficients that would fit phi_true directly.

This confirms: The error functional has a DIFFERENT minimum than
the coefficient space minimum that fits phi_true.

The error functional minimizes the PDE residual, not the phi error.
Different phi can give similar PDE residuals.
""")


if __name__ == '__main__':
    main()
