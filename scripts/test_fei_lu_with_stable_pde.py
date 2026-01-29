#!/usr/bin/env python
"""Test Fei Lu method with STABLE PDE solver for data generation.

Key insight from debugging:
- Oracle test passes with semi-implicit solver (c_opt ≈ 1.0)
- Now test if B-spline learning also improves with stable PDE data

This script:
1. Generate PDE data using semi-implicit solver
2. Learn phi using B-spline basis
3. Compare with ground truth
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import BSpline
from scipy import integrate
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ==============================================================================
# Semi-Implicit PDE Solver (from test_quadratic_stable.py)
# ==============================================================================

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


def compute_Phi_conv_u(x_grid, u, Phi_func):
    """Compute Phi * u = sum_y Phi(|x-y|) * u(y) dy."""
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    result = np.zeros(M)
    for m, x in enumerate(x_grid):
        for n, y in enumerate(x_grid):
            r = abs(x - y)
            result[m] += Phi_func(r) * u[n] * dx
    return result


def generate_pde_semi_implicit(x_grid, t_grid, nu, phi_func, verbose=False):
    """Semi-implicit scheme: implicit for diffusion, explicit for advection."""
    M = len(x_grid)
    L = len(t_grid)
    dx = x_grid[1] - x_grid[0]

    u = np.zeros((L, M))
    # Initial: mixture of Gaussians (same as Fei Lu paper)
    u[0] = 0.5 * np.exp(-(x_grid - 1)**2 / (2 * 0.25)) / np.sqrt(2 * np.pi * 0.25)
    u[0] += 0.5 * np.exp(-(x_grid + 1)**2 / (2 * 0.25)) / np.sqrt(2 * np.pi * 0.25)
    u[0] /= np.sum(u[0]) * dx

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
        r = nu / dx**2
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

        if verbose and l % 1000 == 0:
            print(f"  t={t_grid[l]:.3f}, max_u={np.max(u[l]):.4f}")

    return u


# ==============================================================================
# B-spline Basis
# ==============================================================================

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
    """Evaluate all basis functions at given points."""
    n_points = len(r)
    n_basis = len(basis_funcs)
    B = np.zeros((n_points, n_basis))
    for i, phi in enumerate(basis_funcs):
        vals = phi(r)
        vals = np.nan_to_num(vals, nan=0.0)
        B[:, i] = vals
    return B


# ==============================================================================
# Error Functional (Fei Lu Eq 2.16-2.18)
# ==============================================================================

def compute_A_and_b(u_time, t_grid, x_grid, basis_funcs, nu, Phi_funcs=None):
    """Compute normal matrix A and vector b.

    Args:
        u_time: Density u(x,t), shape (L, M)
        t_grid: Time points
        x_grid: Spatial grid
        basis_funcs: List of B-spline basis functions for phi
        nu: Viscosity
        Phi_funcs: Optional list of antiderivative functions (Phi = integral of phi)
                   If None, compute numerically
    """
    L, M = u_time.shape
    n_basis = len(basis_funcs)
    dx = x_grid[1] - x_grid[0]
    T = t_grid[-1] - t_grid[0]

    # Precompute basis values on r grid
    r_max = x_grid[-1] - x_grid[0]
    r_grid = np.linspace(0, r_max, M)
    dr = r_grid[1] - r_grid[0]

    phi_vals = evaluate_basis(basis_funcs, r_grid)  # (M, n_basis)

    # Compute antiderivatives Phi_i numerically
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

        # Precompute K_phi_i * u for all basis functions
        K_phi_all = np.zeros((M, n_basis))
        Phi_conv_u_all = np.zeros((M, n_basis))

        for i in range(n_basis):
            # K_phi_i * u
            for m, x in enumerate(x_grid):
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    if r > 1e-10:
                        r_idx = min(int(r / dr), M - 1)
                        phi_val = phi_vals[r_idx, i]
                        K_phi_all[m, i] += phi_val * np.sign(x - y) * u_curr[n] * dx

            # Phi_i * u
            for m, x in enumerate(x_grid):
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    r_idx = min(int(r / dr), M - 1)
                    Phi_val = Phi_vals[r_idx, i]
                    Phi_conv_u_all[m, i] += Phi_val * u_curr[n] * dx

        # Compute A and b
        for i in range(n_basis):
            K_phi_i = K_phi_all[:, i]
            Phi_i_u = Phi_conv_u_all[:, i]

            # b_i (Eq 2.18)
            term1 = np.sum(du_dt * Phi_i_u) * dx
            term2 = nu * np.sum(du_dx * K_phi_i) * dx
            b[i] -= (term1 + term2) * dt / T

            for j in range(i, n_basis):
                K_phi_j = K_phi_all[:, j]

                # A_ij (Eq 2.17)
                term = np.sum(K_phi_i * K_phi_j * u_curr) * dx
                A[i, j] += term * dt / T
                if j != i:
                    A[j, i] = A[i, j]

    return A, b


def solve_tikhonov(A, b, lambda_range=None):
    """Solve with Tikhonov regularization using L-curve."""
    n = len(b)

    if lambda_range is None:
        eigvals = np.linalg.eigvalsh(A)
        lambda_min = max(1e-12, eigvals.min() * 1e-6)
        lambda_max = eigvals.max() * 10
        lambda_range = np.logspace(np.log10(lambda_min), np.log10(lambda_max), 50)

    errors = []
    norms = []
    solutions = []

    for lam in lambda_range:
        c = np.linalg.solve(A + lam * np.eye(n), b)
        solutions.append(c)
        errors.append(np.linalg.norm(A @ c - b))
        norms.append(np.linalg.norm(c))

    errors = np.array(errors)
    norms = np.array(norms)

    # L-curve curvature
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

    return solutions[opt_idx], lambda_range[opt_idx], {
        'lambda_range': lambda_range,
        'errors': errors,
        'norms': norms,
        'opt_idx': opt_idx
    }


# ==============================================================================
# Main Test
# ==============================================================================

def test_gaussian_kernel():
    """Test with Gaussian kernel (known to work)."""
    print("=" * 70)
    print("Test 1: Gaussian Kernel with Semi-Implicit PDE + B-spline Learning")
    print("=" * 70)

    # Parameters
    nu = 0.5
    x_min, x_max = -10, 10
    M = 150  # Spatial grid
    L = 5000  # Fine time steps for stability
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    print(f"\nParameters: nu={nu}, M={M}, L={L}, T={T}")
    print(f"dx={dx:.4f}, dt={dt:.6f}")

    # Gaussian kernel: phi(r) = -r * exp(-r^2/2) (attractive)
    phi_true = lambda r: -r * np.exp(-r**2 / 2)
    Phi_true = lambda r: np.exp(-r**2 / 2)  # Antiderivative

    print("\n[1] Generating PDE solution...")
    t0 = time.time()
    u_time = generate_pde_semi_implicit(x_grid, t_grid, nu, phi_true, verbose=True)
    print(f"  Time: {time.time() - t0:.1f}s")

    # Subsample for error functional
    L_sub = 50
    stride = L // L_sub
    t_sub = t_grid[::stride]
    u_sub = u_time[::stride]
    print(f"\n[2] Subsampled: L_sub={L_sub}")

    # Create B-spline basis
    n_basis = 8
    degree = 2
    r_max = x_max - x_min
    basis_funcs, _ = create_bspline_basis(n_basis, degree, 0, r_max * 0.5)
    print(f"\n[3] B-spline basis: n_basis={n_basis}, degree={degree}")

    # Compute A and b
    print("\n[4] Computing A and b...")
    t0 = time.time()
    A, b = compute_A_and_b(u_sub, t_sub, x_grid, basis_funcs, nu)
    cond_A = np.linalg.cond(A)
    print(f"  A condition number: {cond_A:.2e}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Solve
    print("\n[5] Solving with Tikhonov...")
    c_opt, lambda_opt, l_curve = solve_tikhonov(A, b)
    print(f"  lambda_opt: {lambda_opt:.2e}")
    print(f"  c_opt: {c_opt}")

    # Evaluate learned phi
    r_eval = np.linspace(0.01, r_max * 0.3, 100)
    phi_learned = np.zeros_like(r_eval)
    for i, basis in enumerate(basis_funcs):
        vals = basis(r_eval)
        vals = np.nan_to_num(vals, nan=0.0)
        phi_learned += c_opt[i] * vals

    phi_truth = phi_true(r_eval)

    # Compute error
    l2_diff = np.sqrt(np.mean((phi_learned - phi_truth)**2))
    l2_true = np.sqrt(np.mean(phi_truth**2))
    rel_error = l2_diff / l2_true

    print(f"\n{'=' * 70}")
    print(f"Results: Gaussian Kernel")
    print(f"  Relative L2 error: {rel_error:.4f} ({rel_error*100:.2f}%)")
    print(f"  A condition number: {cond_A:.2e}")
    status = "PASS" if rel_error < 0.2 else "FAIL"
    print(f"  Status: {status}")
    print(f"{'=' * 70}")

    return {
        'kernel': 'Gaussian',
        'rel_error': rel_error,
        'cond_A': cond_A,
        'phi_learned': phi_learned,
        'phi_truth': phi_truth,
        'r_eval': r_eval,
        'status': status
    }


def test_quadratic_kernel():
    """Test with Quadratic kernel (previously failed with unstable solver)."""
    print("\n" + "=" * 70)
    print("Test 2: Quadratic Kernel (phi=3r^2) with Semi-Implicit PDE + B-spline")
    print("=" * 70)

    # Parameters - use finer time stepping for stability
    nu = 1.0
    x_min, x_max = -10, 10
    M = 200  # Spatial grid
    L = 10000  # Very fine time steps
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    print(f"\nParameters: nu={nu}, M={M}, L={L}, T={T}")
    print(f"dx={dx:.4f}, dt={dt:.6f}")

    # Quadratic kernel from Fei Lu paper
    phi_true = lambda r: 3 * r**2
    Phi_true = lambda r: r**3

    print("\n[1] Generating PDE solution (semi-implicit)...")
    t0 = time.time()
    u_time = generate_pde_semi_implicit(x_grid, t_grid, nu, phi_true, verbose=True)
    print(f"  Time: {time.time() - t0:.1f}s")

    # Subsample
    L_sub = 100
    stride = L // L_sub
    t_sub = t_grid[::stride]
    u_sub = u_time[::stride]
    print(f"\n[2] Subsampled: L_sub={L_sub}")

    # Create B-spline basis
    n_basis = 6  # Fewer basis for quadratic (simpler function)
    degree = 2
    r_max = x_max - x_min
    basis_funcs, _ = create_bspline_basis(n_basis, degree, 0, r_max * 0.3)
    print(f"\n[3] B-spline basis: n_basis={n_basis}, degree={degree}")

    # Compute A and b
    print("\n[4] Computing A and b...")
    t0 = time.time()
    A, b = compute_A_and_b(u_sub, t_sub, x_grid, basis_funcs, nu)
    cond_A = np.linalg.cond(A)
    print(f"  A condition number: {cond_A:.2e}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Solve
    print("\n[5] Solving with Tikhonov...")
    c_opt, lambda_opt, l_curve = solve_tikhonov(A, b)
    print(f"  lambda_opt: {lambda_opt:.2e}")
    print(f"  c_opt: {c_opt}")

    # Evaluate learned phi
    r_eval = np.linspace(0.01, r_max * 0.2, 100)
    phi_learned = np.zeros_like(r_eval)
    for i, basis in enumerate(basis_funcs):
        vals = basis(r_eval)
        vals = np.nan_to_num(vals, nan=0.0)
        phi_learned += c_opt[i] * vals

    phi_truth = phi_true(r_eval)

    # Compute error
    l2_diff = np.sqrt(np.mean((phi_learned - phi_truth)**2))
    l2_true = np.sqrt(np.mean(phi_truth**2))
    rel_error = l2_diff / l2_true

    print(f"\n{'=' * 70}")
    print(f"Results: Quadratic Kernel")
    print(f"  Relative L2 error: {rel_error:.4f} ({rel_error*100:.2f}%)")
    print(f"  A condition number: {cond_A:.2e}")
    status = "PASS" if rel_error < 0.3 else "FAIL"
    print(f"  Status: {status}")
    print(f"{'=' * 70}")

    return {
        'kernel': 'Quadratic',
        'rel_error': rel_error,
        'cond_A': cond_A,
        'phi_learned': phi_learned,
        'phi_truth': phi_truth,
        'r_eval': r_eval,
        'status': status
    }


def main():
    print("=" * 70)
    print("MVP-2.0 Fix: Testing B-spline Learning with Stable PDE Solver")
    print("=" * 70)
    print("\nHypothesis: The high condition number was caused by unstable PDE data.")
    print("With semi-implicit solver, we expect:")
    print("  1. Lower A condition number")
    print("  2. Better phi recovery")
    print()

    results = []

    # Test 1: Gaussian (should work)
    results.append(test_gaussian_kernel())

    # Test 2: Quadratic (previously failed)
    results.append(test_quadratic_kernel())

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['kernel']:12s}: error={r['rel_error']:.2%}, cond(A)={r['cond_A']:.2e}, {r['status']}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, r in enumerate(results):
        ax = axes[idx]
        ax.plot(r['r_eval'], r['phi_truth'], 'r-', lw=2, label='True φ(r)')
        ax.plot(r['r_eval'], r['phi_learned'], 'b--', lw=2, label='Learned φ(r)')
        ax.set_xlabel('r')
        ax.set_ylabel('φ(r)')
        ax.set_title(f"{r['kernel']}: error={r['rel_error']:.2%}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/ips_unlabeled/img/mvp2_0_stable_pde_test.png', dpi=150)
    plt.close()
    print("\nPlot saved to experiments/ips_unlabeled/img/mvp2_0_stable_pde_test.png")

    # Return overall status
    all_pass = all(r['status'] == 'PASS' for r in results)
    return 0 if all_pass else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
