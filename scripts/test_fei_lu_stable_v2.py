#!/usr/bin/env python
"""Test Fei Lu method with stable PDE solver - Version 2.

Improvements:
1. Better B-spline support range (match actual data)
2. More basis functions
3. Test multiple configurations
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import BSpline
from scipy import integrate
import matplotlib.pyplot as plt
from pathlib import Path
import time


def compute_K_phi_conv_u(x_grid, u, phi_func):
    """Compute K_phi * u = sum_y phi(|x-y|) * sign(x-y) * u(y) dy."""
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    result = np.zeros(M)
    for m, x in enumerate(x_grid):
        for n, y in enumerate(x_grid):
            r = abs(x - y)
            if r > 1e-10:
                result[m] += phi_func(r) * np.sign(x - y) * u[n] * dx
    return result


def generate_pde_semi_implicit(x_grid, t_grid, nu, phi_func, verbose=False):
    """Semi-implicit scheme."""
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


def create_bspline_basis(n_basis, degree, r_min, r_max):
    """Create B-spline basis functions."""
    n_interior = n_basis - degree + 1
    interior_knots = np.linspace(r_min, r_max, n_interior + 2)[1:-1]
    knots = np.concatenate([
        [r_min] * degree,
        interior_knots,
        [r_max] * degree
    ])
    basis_funcs = []
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        basis_funcs.append(BSpline(knots, c, degree, extrapolate=False))
    return basis_funcs, knots


def evaluate_basis(basis_funcs, r):
    n_points = len(r)
    n_basis = len(basis_funcs)
    B = np.zeros((n_points, n_basis))
    for i, phi in enumerate(basis_funcs):
        vals = phi(r)
        vals = np.nan_to_num(vals, nan=0.0)
        B[:, i] = vals
    return B


def compute_A_and_b(u_time, t_grid, x_grid, basis_funcs, nu):
    """Compute normal matrix A and vector b."""
    L, M = u_time.shape
    n_basis = len(basis_funcs)
    dx = x_grid[1] - x_grid[0]
    T = t_grid[-1] - t_grid[0]

    r_max = x_grid[-1] - x_grid[0]
    r_grid = np.linspace(0, r_max, M)
    dr = r_grid[1] - r_grid[0]

    phi_vals = evaluate_basis(basis_funcs, r_grid)
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
        Phi_conv_u_all = np.zeros((M, n_basis))

        for i in range(n_basis):
            for m, x in enumerate(x_grid):
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    if r > 1e-10:
                        r_idx = min(int(r / dr), M - 1)
                        phi_val = phi_vals[r_idx, i]
                        K_phi_all[m, i] += phi_val * np.sign(x - y) * u_curr[n] * dx

            for m, x in enumerate(x_grid):
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    r_idx = min(int(r / dr), M - 1)
                    Phi_val = Phi_vals[r_idx, i]
                    Phi_conv_u_all[m, i] += Phi_val * u_curr[n] * dx

        for i in range(n_basis):
            K_phi_i = K_phi_all[:, i]
            Phi_i_u = Phi_conv_u_all[:, i]

            term1 = np.sum(du_dt * Phi_i_u) * dx
            term2 = nu * np.sum(du_dx * K_phi_i) * dx
            b[i] -= (term1 + term2) * dt / T

            for j in range(i, n_basis):
                K_phi_j = K_phi_all[:, j]
                term = np.sum(K_phi_i * K_phi_j * u_curr) * dx
                A[i, j] += term * dt / T
                if j != i:
                    A[j, i] = A[i, j]

    return A, b


def solve_tikhonov(A, b, lambda_range=None):
    """Solve with Tikhonov regularization."""
    n = len(b)
    if lambda_range is None:
        eigvals = np.linalg.eigvalsh(A)
        lambda_min = max(1e-12, eigvals.min() * 1e-6)
        lambda_max = eigvals.max() * 10
        lambda_range = np.logspace(np.log10(lambda_min), np.log10(lambda_max), 50)

    solutions = []
    errors = []
    norms = []

    for lam in lambda_range:
        c = np.linalg.solve(A + lam * np.eye(n), b)
        solutions.append(c)
        errors.append(np.linalg.norm(A @ c - b))
        norms.append(np.linalg.norm(c))

    errors = np.array(errors)
    norms = np.array(norms)

    log_errors = np.log(errors + 1e-16)
    log_norms = np.log(norms + 1e-16)

    curvature = np.zeros(len(lambda_range) - 2)
    for i in range(1, len(lambda_range) - 1):
        dx1 = log_errors[i] - log_errors[i-1]
        dy1 = log_norms[i] - log_norms[i-1]
        dx2 = log_errors[i+1] - log_errors[i]
        dy2 = log_norms[i+1] - log_norms[i]
        cross = dx1 * dy2 - dx2 * dy1
        denom = (dx1**2 + dy1**2) * (dx2**2 + dy2**2)
        if denom > 1e-20:
            curvature[i-1] = cross / np.sqrt(denom)

    opt_idx = np.argmax(curvature) + 1 if len(curvature) > 0 else len(lambda_range) // 2

    return solutions[opt_idx], lambda_range[opt_idx], opt_idx


def test_kernel(name, phi_func, Phi_func, nu, domain_size, r_support, n_basis, L=5000):
    """Test a specific kernel configuration."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"  nu={nu}, domain=[-{domain_size}, {domain_size}], r_support=[0, {r_support}]")
    print(f"  n_basis={n_basis}, L={L}")
    print(f"{'=' * 60}")

    x_min, x_max = -domain_size, domain_size
    M = 150
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    print(f"\n[1] Generating PDE solution (dt={dt:.6f})...")
    t0 = time.time()
    u_time = generate_pde_semi_implicit(x_grid, t_grid, nu, phi_func)
    print(f"  Time: {time.time() - t0:.1f}s")

    # Subsample
    L_sub = min(50, L // 10)
    stride = max(1, L // L_sub)
    t_sub = t_grid[::stride]
    u_sub = u_time[::stride]
    print(f"\n[2] Subsampled: L_sub={len(t_sub)}")

    # B-spline basis
    basis_funcs, _ = create_bspline_basis(n_basis, 2, 0, r_support)
    print(f"\n[3] B-spline: n_basis={n_basis}, support=[0, {r_support}]")

    # Compute A and b
    print("\n[4] Computing A and b...")
    t0 = time.time()
    A, b = compute_A_and_b(u_sub, t_sub, x_grid, basis_funcs, nu)
    cond_A = np.linalg.cond(A)
    print(f"  cond(A) = {cond_A:.2e}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Solve
    print("\n[5] Solving...")
    c_opt, lambda_opt, _ = solve_tikhonov(A, b)
    print(f"  lambda_opt = {lambda_opt:.2e}")

    # Evaluate
    r_eval = np.linspace(0.01, r_support * 0.8, 100)
    phi_learned = np.zeros_like(r_eval)
    for i, basis in enumerate(basis_funcs):
        vals = basis(r_eval)
        vals = np.nan_to_num(vals, nan=0.0)
        phi_learned += c_opt[i] * vals

    phi_truth = phi_func(r_eval)

    # Error
    l2_diff = np.sqrt(np.mean((phi_learned - phi_truth)**2))
    l2_true = np.sqrt(np.mean(phi_truth**2))
    rel_error = l2_diff / l2_true if l2_true > 1e-10 else float('inf')

    status = "PASS" if rel_error < 0.3 else "FAIL"
    print(f"\n[Result] {name}: error={rel_error:.2%}, cond(A)={cond_A:.2e}, {status}")

    return {
        'name': name,
        'rel_error': rel_error,
        'cond_A': cond_A,
        'phi_learned': phi_learned,
        'phi_truth': phi_truth,
        'r_eval': r_eval,
        'status': status
    }


def main():
    print("=" * 70)
    print("MVP-2.0 B-spline Learning Test - Multiple Configurations")
    print("=" * 70)

    results = []

    # Test 1: Gaussian with smaller domain and support
    results.append(test_kernel(
        name="Gaussian (compact)",
        phi_func=lambda r: -r * np.exp(-r**2 / 2),
        Phi_func=lambda r: np.exp(-r**2 / 2),
        nu=0.5,
        domain_size=5,  # Smaller domain
        r_support=3,    # Smaller support (Gaussian decays fast)
        n_basis=6,
        L=3000
    ))

    # Test 2: Gaussian with more basis functions
    results.append(test_kernel(
        name="Gaussian (more basis)",
        phi_func=lambda r: -r * np.exp(-r**2 / 2),
        Phi_func=lambda r: np.exp(-r**2 / 2),
        nu=0.5,
        domain_size=5,
        r_support=4,
        n_basis=10,
        L=3000
    ))

    # Test 3: Quadratic (known to work)
    results.append(test_kernel(
        name="Quadratic (phi=3r^2)",
        phi_func=lambda r: 3 * r**2,
        Phi_func=lambda r: r**3,
        nu=1.0,
        domain_size=10,
        r_support=6,
        n_basis=6,
        L=10000
    ))

    # Test 4: Linear kernel
    results.append(test_kernel(
        name="Linear (phi=r)",
        phi_func=lambda r: r,
        Phi_func=lambda r: r**2 / 2,
        nu=0.5,
        domain_size=5,
        r_support=4,
        n_basis=5,
        L=3000
    ))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Kernel':<25} {'Error':>10} {'cond(A)':>12} {'Status':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<25} {r['rel_error']:>9.2%} {r['cond_A']:>12.2e} {r['status']:>8}")

    # Plot
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for idx, r in enumerate(results):
        ax = axes[idx]
        ax.plot(r['r_eval'], r['phi_truth'], 'r-', lw=2, label='True')
        ax.plot(r['r_eval'], r['phi_learned'], 'b--', lw=2, label='Learned')
        ax.set_xlabel('r')
        ax.set_ylabel('φ(r)')
        ax.set_title(f"{r['name']}\nerror={r['rel_error']:.1%}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/ips_unlabeled/img/mvp2_0_multi_kernel_test.png', dpi=150)
    plt.close()
    print("\nPlot saved to experiments/ips_unlabeled/img/mvp2_0_multi_kernel_test.png")

    n_pass = sum(1 for r in results if r['status'] == 'PASS')
    print(f"\nPassed: {n_pass}/{len(results)}")
    return 0 if n_pass >= len(results) - 1 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
