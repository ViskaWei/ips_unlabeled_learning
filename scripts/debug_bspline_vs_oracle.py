#!/usr/bin/env python
"""Debug: Compare Oracle vs B-spline to understand why B-spline fails.

Key question: Why does oracle test pass (c_opt ≈ 1.0) but B-spline fails?

Hypothesis:
1. B-spline basis functions become nearly linearly dependent after convolution
2. The A matrix has effective rank much smaller than n_basis
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import BSpline
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


def create_bspline_basis(n_basis, degree, r_min, r_max):
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


def analyze_problem():
    """Analyze why B-spline fails but Oracle passes."""
    print("=" * 70)
    print("Analysis: Oracle vs B-spline")
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

    # True Gaussian kernel
    phi_true = lambda r: -r * np.exp(-r**2 / 2)
    Phi_true = lambda r: np.exp(-r**2 / 2)

    print("\n[1] Generate PDE data...")
    u_time = generate_pde_semi_implicit(x_grid, t_grid, nu, phi_true)

    # Subsample
    L_sub = 30
    stride = L // L_sub
    t_sub = t_grid[::stride]
    u_sub = u_time[::stride]

    print(f"\n[2] Data: M={M}, L_sub={L_sub}")

    # =========================================================================
    # Test 1: Oracle (single basis = true phi)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 1: Oracle (phi_true as single basis)")
    print("=" * 50)

    A_oracle = 0.0
    b_oracle = 0.0
    T_total = t_sub[-1] - t_sub[0]

    for ell in range(len(t_sub) - 1):
        u_curr = u_sub[ell]
        u_next = u_sub[ell + 1]
        dt = t_sub[ell + 1] - t_sub[ell]

        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)

        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_true)

        # Phi * u
        Phi_u = np.zeros(M)
        for m, x in enumerate(x_grid):
            for n, y in enumerate(x_grid):
                r = abs(x - y)
                Phi_u[m] += Phi_true(r) * u_curr[n] * dx

        # A term
        A_oracle += np.sum(K_phi_u**2 * u_curr) * dx * dt / T_total

        # b term
        b_oracle -= (np.sum(du_dt * Phi_u) * dx + nu * np.sum(du_dx * K_phi_u) * dx) * dt / T_total

    c_oracle = b_oracle / A_oracle if abs(A_oracle) > 1e-15 else float('nan')
    print(f"  A = {A_oracle:.6e}")
    print(f"  b = {b_oracle:.6e}")
    print(f"  c_opt = {c_oracle:.4f} (expected: 1.0)")
    print(f"  Error: {abs(c_oracle - 1.0):.4f}")

    # =========================================================================
    # Test 2: B-spline basis
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 2: B-spline basis (n_basis=5)")
    print("=" * 50)

    n_basis = 5
    r_max = 4.0
    basis_funcs, _ = create_bspline_basis(n_basis, 2, 0, r_max)

    # Precompute basis on r grid
    r_grid = np.linspace(0, x_max - x_min, M)
    dr = r_grid[1] - r_grid[0]

    phi_vals = np.zeros((M, n_basis))
    for i, bf in enumerate(basis_funcs):
        vals = bf(r_grid)
        phi_vals[:, i] = np.nan_to_num(vals, nan=0.0)

    Phi_vals = np.zeros_like(phi_vals)
    for i in range(n_basis):
        Phi_vals[:, i] = integrate.cumulative_trapezoid(phi_vals[:, i], r_grid, initial=0)

    A_bspline = np.zeros((n_basis, n_basis))
    b_bspline = np.zeros(n_basis)

    # Also collect K_phi_i * u for each time step to analyze
    K_phi_all_time = []

    for ell in range(len(t_sub) - 1):
        u_curr = u_sub[ell]
        u_next = u_sub[ell + 1]
        dt = t_sub[ell + 1] - t_sub[ell]

        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)

        K_phi_i_all = np.zeros((M, n_basis))
        Phi_u_all = np.zeros((M, n_basis))

        for i in range(n_basis):
            # K_phi_i * u
            for m, x in enumerate(x_grid):
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    if r > 1e-10:
                        r_idx = min(int(r / dr), M - 1)
                        K_phi_i_all[m, i] += phi_vals[r_idx, i] * np.sign(x - y) * u_curr[n] * dx

            # Phi_i * u
            for m, x in enumerate(x_grid):
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    r_idx = min(int(r / dr), M - 1)
                    Phi_u_all[m, i] += Phi_vals[r_idx, i] * u_curr[n] * dx

        if ell == 0:
            K_phi_all_time.append(K_phi_i_all.copy())

        for i in range(n_basis):
            b_bspline[i] -= (np.sum(du_dt * Phi_u_all[:, i]) * dx +
                            nu * np.sum(du_dx * K_phi_i_all[:, i]) * dx) * dt / T_total
            for j in range(i, n_basis):
                term = np.sum(K_phi_i_all[:, i] * K_phi_i_all[:, j] * u_curr) * dx
                A_bspline[i, j] += term * dt / T_total
                if j != i:
                    A_bspline[j, i] = A_bspline[i, j]

    print(f"  A shape: {A_bspline.shape}")
    print(f"  A condition number: {np.linalg.cond(A_bspline):.2e}")
    print(f"  b: {b_bspline}")

    # SVD analysis
    U, s, Vt = np.linalg.svd(A_bspline)
    print(f"\n  Singular values of A:")
    for i, sv in enumerate(s):
        print(f"    σ_{i} = {sv:.4e}")

    print(f"\n  Ratio σ_0/σ_last = {s[0]/s[-1]:.2e}")
    effective_rank = np.sum(s > s[0] * 1e-6)
    print(f"  Effective rank (threshold 1e-6): {effective_rank}/{n_basis}")

    # Solve
    c_opt = np.linalg.lstsq(A_bspline, b_bspline, rcond=1e-6)[0]
    print(f"\n  c_opt (lstsq): {c_opt}")

    # Evaluate learned phi
    r_eval = np.linspace(0.01, r_max * 0.8, 50)
    phi_learned = np.zeros_like(r_eval)
    for i, bf in enumerate(basis_funcs):
        vals = bf(r_eval)
        vals = np.nan_to_num(vals, nan=0.0)
        phi_learned += c_opt[i] * vals

    phi_truth = phi_true(r_eval)

    l2_diff = np.sqrt(np.mean((phi_learned - phi_truth)**2))
    l2_true = np.sqrt(np.mean(phi_truth**2))
    rel_error = l2_diff / l2_true

    print(f"\n  Learned phi error: {rel_error:.2%}")

    # =========================================================================
    # Analysis: Why are K_phi_i * u nearly collinear?
    # =========================================================================
    print("\n" + "=" * 50)
    print("Analysis: K_phi_i * u vectors at t=0")
    print("=" * 50)

    K_t0 = K_phi_all_time[0]  # (M, n_basis)

    # Correlation matrix
    K_normalized = K_t0 / (np.linalg.norm(K_t0, axis=0, keepdims=True) + 1e-10)
    corr = K_normalized.T @ K_normalized / M

    print("  Correlation matrix of K_phi_i * u:")
    for i in range(n_basis):
        row = " ".join([f"{corr[i,j]:6.3f}" for j in range(n_basis)])
        print(f"    [{row}]")

    # =========================================================================
    # Test 3: Use phi_true projected onto B-spline basis
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 3: Project phi_true onto B-spline, then use as oracle")
    print("=" * 50)

    # Fit phi_true using B-splines
    r_fit = np.linspace(0.01, r_max * 0.9, 100)
    phi_true_vals = phi_true(r_fit)

    B_fit = np.zeros((len(r_fit), n_basis))
    for i, bf in enumerate(basis_funcs):
        vals = bf(r_fit)
        B_fit[:, i] = np.nan_to_num(vals, nan=0.0)

    c_fit = np.linalg.lstsq(B_fit, phi_true_vals, rcond=None)[0]
    phi_fitted = B_fit @ c_fit

    fit_error = np.sqrt(np.mean((phi_fitted - phi_true_vals)**2)) / np.sqrt(np.mean(phi_true_vals**2))
    print(f"  B-spline fit of phi_true: error = {fit_error:.2%}")
    print(f"  Fitted coefficients: {c_fit}")

    # Now use this fitted phi in oracle test
    phi_fitted_func = lambda r: np.interp(r, r_fit, phi_fitted)

    A_proj = 0.0
    b_proj = 0.0

    for ell in range(len(t_sub) - 1):
        u_curr = u_sub[ell]
        u_next = u_sub[ell + 1]
        dt = t_sub[ell + 1] - t_sub[ell]

        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)

        K_phi_u = compute_K_phi_conv_u(x_grid, u_curr, phi_fitted_func)

        # Need Phi for fitted phi
        Phi_fitted = integrate.cumulative_trapezoid(phi_fitted, r_fit, initial=0)
        Phi_fitted_func = lambda r: np.interp(r, r_fit, Phi_fitted)

        Phi_u = np.zeros(M)
        for m, x in enumerate(x_grid):
            for n, y in enumerate(x_grid):
                r = abs(x - y)
                Phi_u[m] += Phi_fitted_func(r) * u_curr[n] * dx

        A_proj += np.sum(K_phi_u**2 * u_curr) * dx * dt / T_total
        b_proj -= (np.sum(du_dt * Phi_u) * dx + nu * np.sum(du_dx * K_phi_u) * dx) * dt / T_total

    c_proj = b_proj / A_proj if abs(A_proj) > 1e-15 else float('nan')
    print(f"\n  Using B-spline-projected phi as oracle:")
    print(f"    A = {A_proj:.6e}")
    print(f"    b = {b_proj:.6e}")
    print(f"    c_opt = {c_proj:.4f} (expected: 1.0)")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: B-spline basis
    ax = axes[0, 0]
    r_plot = np.linspace(0, r_max, 100)
    for i, bf in enumerate(basis_funcs):
        vals = bf(r_plot)
        vals = np.nan_to_num(vals, nan=0.0)
        ax.plot(r_plot, vals, label=f'B_{i}')
    ax.set_xlabel('r')
    ax.set_ylabel('φ_i(r)')
    ax.set_title('B-spline Basis Functions')
    ax.legend()
    ax.grid(True)

    # Plot 2: K_phi_i * u at t=0
    ax = axes[0, 1]
    for i in range(n_basis):
        ax.plot(x_grid, K_t0[:, i], label=f'K_{i}*u')
    ax.set_xlabel('x')
    ax.set_ylabel('K_φ_i * u')
    ax.set_title('Convolved Basis Functions (t=0)')
    ax.legend()
    ax.grid(True)

    # Plot 3: Singular values
    ax = axes[1, 0]
    ax.semilogy(range(len(s)), s, 'bo-')
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular value')
    ax.set_title(f'Singular Values of A (cond={np.linalg.cond(A_bspline):.2e})')
    ax.grid(True)

    # Plot 4: Learned vs True phi
    ax = axes[1, 1]
    ax.plot(r_eval, phi_truth, 'r-', lw=2, label='True φ')
    ax.plot(r_eval, phi_learned, 'b--', lw=2, label=f'Learned (err={rel_error:.1%})')
    ax.plot(r_fit, phi_fitted, 'g:', lw=2, label=f'B-spline fit (err={fit_error:.1%})')
    ax.set_xlabel('r')
    ax.set_ylabel('φ(r)')
    ax.set_title('Comparison')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('experiments/ips_unlabeled/img/debug_bspline_vs_oracle.png', dpi=150)
    plt.close()
    print("\nPlot saved to experiments/ips_unlabeled/img/debug_bspline_vs_oracle.png")


if __name__ == '__main__':
    analyze_problem()
