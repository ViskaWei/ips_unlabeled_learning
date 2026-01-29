#!/usr/bin/env python
"""Test Fei Lu method with polynomial basis instead of B-spline.

Key insight: B-spline cannot represent Gaussian kernel well (50% fit error).
Let's try:
1. Polynomial basis (1, r, r^2, r^3, ...)
2. Gaussian-like basis (r * exp(-r^2/a^2) for different a)
3. Custom basis designed for the target kernel
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy import integrate
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

        A = diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csr')
        rhs = u_curr + dt * dflux_dx
        u[l + 1] = spsolve(A, rhs)
        u[l + 1] = np.maximum(u[l + 1], 0)
        mass = np.sum(u[l + 1]) * dx
        if mass > 1e-10:
            u[l + 1] /= mass

    return u


def test_basis_fit(basis_name, basis_funcs, phi_true, r_range, ax):
    """Test how well a basis can fit phi_true."""
    r_fit = np.linspace(0.01, r_range, 100)
    phi_true_vals = phi_true(r_fit)

    n_basis = len(basis_funcs)
    B = np.zeros((len(r_fit), n_basis))
    for i, bf in enumerate(basis_funcs):
        B[:, i] = bf(r_fit)

    # Fit
    c_fit, residuals, rank, s = np.linalg.lstsq(B, phi_true_vals, rcond=None)
    phi_fitted = B @ c_fit

    fit_error = np.sqrt(np.mean((phi_fitted - phi_true_vals)**2)) / np.sqrt(np.mean(phi_true_vals**2))

    # Plot
    ax.plot(r_fit, phi_true_vals, 'r-', lw=2, label='True φ')
    ax.plot(r_fit, phi_fitted, 'b--', lw=2, label=f'Fitted (err={fit_error:.1%})')
    ax.set_xlabel('r')
    ax.set_ylabel('φ(r)')
    ax.set_title(f'{basis_name}\nFit error: {fit_error:.1%}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fit_error, c_fit


def compute_A_and_b(u_time, t_grid, x_grid, basis_funcs, nu):
    """Compute normal matrix A and vector b."""
    L, M = u_time.shape
    n_basis = len(basis_funcs)
    dx = x_grid[1] - x_grid[0]
    T = t_grid[-1] - t_grid[0]

    r_max = x_grid[-1] - x_grid[0]
    r_grid = np.linspace(0, r_max, M)
    dr = r_grid[1] - r_grid[0]

    # Evaluate basis
    phi_vals = np.zeros((M, n_basis))
    for i, bf in enumerate(basis_funcs):
        phi_vals[:, i] = bf(r_grid)

    # Antiderivatives
    Phi_vals = np.zeros_like(phi_vals)
    for i in range(n_basis):
        Phi_vals[:, i] = integrate.cumulative_trapezoid(phi_vals[:, i], r_grid, initial=0)

    A = np.zeros((n_basis, n_basis))
    b = np.zeros(n_basis)

    for ell in range(L - 1):
        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]

        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)

        K_phi_all = np.zeros((M, n_basis))
        Phi_u_all = np.zeros((M, n_basis))

        for i in range(n_basis):
            for m, x in enumerate(x_grid):
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    if r > 1e-10:
                        r_idx = min(int(r / dr), M - 1)
                        K_phi_all[m, i] += phi_vals[r_idx, i] * np.sign(x - y) * u_curr[n] * dx

            for m, x in enumerate(x_grid):
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    r_idx = min(int(r / dr), M - 1)
                    Phi_u_all[m, i] += Phi_vals[r_idx, i] * u_curr[n] * dx

        for i in range(n_basis):
            b[i] -= (np.sum(du_dt * Phi_u_all[:, i]) * dx +
                    nu * np.sum(du_dx * K_phi_all[:, i]) * dx) * dt / T
            for j in range(i, n_basis):
                term = np.sum(K_phi_all[:, i] * K_phi_all[:, j] * u_curr) * dx
                A[i, j] += term * dt / T
                if j != i:
                    A[j, i] = A[i, j]

    return A, b


def main():
    print("=" * 70)
    print("Test: Different Basis Functions for Fei Lu Method")
    print("=" * 70)

    # Setup
    nu = 0.5
    x_min, x_max = -5, 5
    M = 100
    L = 3000
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]

    # True kernel (Gaussian)
    phi_true = lambda r: -r * np.exp(-r**2 / 2)
    r_range = 4.0

    print("\n[1] Generating PDE data...")
    u_time = generate_pde_semi_implicit(x_grid, t_grid, nu, phi_true)

    L_sub = 30
    stride = L // L_sub
    t_sub = t_grid[::stride]
    u_sub = u_time[::stride]

    # Define different basis sets
    basis_sets = []

    # 1. Polynomial basis: r, r^2, r^3, ...
    poly_basis = [
        lambda r, p=p: r**p for p in range(1, 6)  # Start from 1 (phi(0)=0)
    ]
    basis_sets.append(("Polynomial (r^1..r^5)", poly_basis))

    # 2. Odd polynomial: r, r^3, r^5, ...
    odd_poly = [
        lambda r, p=p: r**p for p in [1, 3, 5, 7]
    ]
    basis_sets.append(("Odd Polynomial", odd_poly))

    # 3. Gaussian-like: r * exp(-r^2/a^2) for different a
    gauss_basis = [
        lambda r, a=a: -r * np.exp(-r**2 / (2 * a**2)) for a in [0.5, 1.0, 1.5, 2.0, 3.0]
    ]
    basis_sets.append(("Gaussian-like", gauss_basis))

    # 4. Mixed: r, r*exp(-r^2/2), r^3
    mixed_basis = [
        lambda r: r,
        lambda r: -r * np.exp(-r**2 / 2),
        lambda r: r**3,
    ]
    basis_sets.append(("Mixed (r, r*exp, r^3)", mixed_basis))

    # 5. Taylor expansion of r*exp(-r^2/2): r - r^3/2 + r^5/8 - ...
    taylor_basis = [
        lambda r: r,
        lambda r: -r**3 / 2,
        lambda r: r**5 / 8,
        lambda r: -r**7 / 48,
    ]
    basis_sets.append(("Taylor expansion", taylor_basis))

    # Test basis fitting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    results = []

    for idx, (name, basis) in enumerate(basis_sets):
        print(f"\n[{idx+2}] Testing: {name}")

        # Test fit quality
        fit_error, c_fit = test_basis_fit(name, basis, phi_true, r_range, axes[idx])
        print(f"  Fit error: {fit_error:.2%}")
        print(f"  Coefficients: {c_fit}")

        # Compute A and b
        print("  Computing A and b...")
        A, b = compute_A_and_b(u_sub, t_sub, x_grid, basis, nu)
        cond_A = np.linalg.cond(A)
        print(f"  cond(A) = {cond_A:.2e}")

        # Solve
        try:
            c_opt = np.linalg.lstsq(A, b, rcond=1e-6)[0]
            print(f"  c_opt: {c_opt}")

            # Evaluate
            r_eval = np.linspace(0.01, r_range * 0.8, 100)
            phi_learned = sum(c_opt[i] * basis[i](r_eval) for i in range(len(basis)))
            phi_truth = phi_true(r_eval)

            learn_error = np.sqrt(np.mean((phi_learned - phi_truth)**2)) / np.sqrt(np.mean(phi_truth**2))
            print(f"  Learning error: {learn_error:.2%}")

            results.append({
                'name': name,
                'fit_error': fit_error,
                'cond_A': cond_A,
                'learn_error': learn_error,
                'status': 'PASS' if learn_error < 0.3 else 'FAIL'
            })
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'name': name,
                'fit_error': fit_error,
                'cond_A': cond_A,
                'learn_error': float('inf'),
                'status': 'ERROR'
            })

    # Fill remaining plot
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('experiments/ips_unlabeled/img/test_polynomial_basis.png', dpi=150)
    plt.close()
    print("\nPlot saved to experiments/ips_unlabeled/img/test_polynomial_basis.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Basis':<25} {'Fit%':>8} {'cond(A)':>12} {'Learn%':>10} {'Status':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['fit_error']:>7.1%} {r['cond_A']:>12.2e} {r['learn_error']:>9.1%} {r['status']:>8}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
