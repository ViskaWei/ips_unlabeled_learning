#!/usr/bin/env python
"""Test Fei Lu's method on PDE-generated data (bypassing KDE).

This test verifies the algorithm works when given clean u(x,t) data
directly from solving the mean-field PDE.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import BSpline
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def true_phi(r, A=1.0, sigma=1.0):
    """True interaction kernel."""
    return -A * r / (sigma**2) * np.exp(-r**2 / (2 * sigma**2))


def true_Phi(r, A=1.0, sigma=1.0):
    """True interaction potential."""
    return A * np.exp(-r**2 / (2 * sigma**2))


def compute_K_phi_conv_u(x_grid, u, phi_func):
    """Compute K_φ * u where K_φ(x) = φ(|x|) * sign(x) for 1D."""
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    result = np.zeros(M)

    for m, x in enumerate(x_grid):
        for n, y in enumerate(x_grid):
            r = abs(x - y)
            if r > 1e-10:
                result[m] += phi_func(r) * np.sign(x - y) * u[n] * dx

    return result


def generate_pde_solution(x_grid, t_grid, nu, phi_func, A=1.0, sigma=1.0):
    """Generate mean-field PDE solution by forward Euler."""
    M = len(x_grid)
    L = len(t_grid)
    dx = x_grid[1] - x_grid[0]

    # Initial condition: Gaussian
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


def compute_normal_matrix_and_vector(u_time, t_grid, x_grid, basis_funcs, nu):
    """Compute normal matrix A and vector b."""
    L, M_grid = u_time.shape
    n_basis = len(basis_funcs)
    dx = x_grid[1] - x_grid[0]
    T = t_grid[-1] - t_grid[0]

    r_max = x_grid[-1] - x_grid[0]
    r_grid = np.linspace(0, r_max, M_grid)
    dr = r_grid[1] - r_grid[0]

    phi_vals = evaluate_basis(basis_funcs, r_grid)

    # Compute antiderivatives
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

        # Compute K_phi_i * u for each basis
        K_phi_all = np.zeros((M_grid, n_basis))
        Phi_conv_all = np.zeros((M_grid, n_basis))

        for i in range(n_basis):
            for m, x in enumerate(x_grid):
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    if r > 1e-10:
                        r_idx = min(int(r / dr), M_grid - 1)
                        phi_val = phi_vals[r_idx, i] if r_idx < M_grid else 0.0
                        K_phi_all[m, i] += phi_val * np.sign(x - y) * u_curr[n] * dx

                    r_idx = min(int(abs(x - y) / dr), M_grid - 1)
                    Phi_val = Phi_vals[r_idx, i] if r_idx < M_grid else 0.0
                    Phi_conv_all[m, i] += Phi_val * u_curr[n] * dx

        for i in range(n_basis):
            # b_i contribution
            term1 = np.sum(du_dt * Phi_conv_all[:, i]) * dx
            term2 = nu * np.sum(du_dx * K_phi_all[:, i]) * dx
            b[i] -= (term1 + term2) * dt / T

            for j in range(i, n_basis):
                # A_ij contribution
                term = np.sum(K_phi_all[:, i] * K_phi_all[:, j] * u_curr) * dx
                A[i, j] += term * dt / T
                if j != i:
                    A[j, i] = A[i, j]

    return A, b


def solve_with_tikhonov(A, b, lambda_range=None):
    """Solve with Tikhonov regularization."""
    n = len(b)

    if lambda_range is None:
        eigvals = np.abs(np.linalg.eigvalsh(A))
        lambda_min = max(1e-12, eigvals.min() * 1e-6)
        lambda_max = eigvals.max() * 10
        lambda_range = np.logspace(np.log10(lambda_min), np.log10(lambda_max), 50)

    best_c = None
    best_lambda = None
    min_resid = float('inf')

    for lam in lambda_range:
        try:
            c = np.linalg.solve(A + lam * np.eye(n), b)
            resid = np.linalg.norm(A @ c - b)
            reg = np.linalg.norm(c)
            # L-curve: minimize curvature (simplified)
            score = resid * reg
            if score < min_resid:
                min_resid = score
                best_c = c
                best_lambda = lam
        except np.linalg.LinAlgError:
            continue

    return best_c, best_lambda


def main():
    print("=" * 70)
    print("Test Fei Lu Method on Clean PDE Data")
    print("=" * 70)

    # Parameters
    nu = 0.1
    A_phi = 1.0
    sigma_phi = 1.0

    # Grid (use enough resolution)
    x_min, x_max = -5, 5
    M = 100
    L = 100  # Many time steps for stable PDE
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]

    print(f"\nParameters:")
    print(f"  ν = {nu}, M = {M}, L = {L}, T = {T}")
    print(f"  True phi: Gaussian with A={A_phi}, σ={sigma_phi}")

    # Generate PDE solution
    print("\n[1] Generating PDE solution with true φ...")
    phi_func = lambda r: true_phi(r, A_phi, sigma_phi)
    u_time = generate_pde_solution(x_grid, t_grid, nu, phi_func, A_phi, sigma_phi)
    print(f"  u shape: {u_time.shape}")
    print(f"  u integral check: t=0: {np.sum(u_time[0])*dx:.4f}, t=T: {np.sum(u_time[-1])*dx:.4f}")

    # Create basis
    print("\n[2] Creating B-spline basis...")
    n_basis = 8
    degree = 2
    r_min, r_max = 0, x_max - x_min
    basis_funcs, knots = create_bspline_basis(n_basis, degree, r_min, r_max)
    print(f"  n_basis = {n_basis}, degree = {degree}")

    # Compute normal equations
    print("\n[3] Computing normal matrix A and vector b...")
    A_mat, b_vec = compute_normal_matrix_and_vector(u_time, t_grid, x_grid, basis_funcs, nu)
    cond = np.linalg.cond(A_mat)
    print(f"  A condition number: {cond:.2e}")
    print(f"  b norm: {np.linalg.norm(b_vec):.4e}")

    # Solve
    print("\n[4] Solving with Tikhonov regularization...")
    c_opt, lambda_opt = solve_with_tikhonov(A_mat, b_vec)
    print(f"  λ_opt = {lambda_opt:.4e}")
    print(f"  Coefficients: {c_opt}")

    # Evaluate learned phi
    print("\n[5] Evaluating learned phi...")
    r_eval = np.linspace(0.1, r_max * 0.8, 100)
    phi_true_vals = np.array([true_phi(r, A_phi, sigma_phi) for r in r_eval])

    phi_learned_vals = np.zeros(len(r_eval))
    phi_basis_vals = evaluate_basis(basis_funcs, r_eval)
    for i, c in enumerate(c_opt):
        phi_learned_vals += c * phi_basis_vals[:, i]

    # Compute error
    error = np.linalg.norm(phi_learned_vals - phi_true_vals) / np.linalg.norm(phi_true_vals)
    print(f"\n  φ relative L2 error: {error*100:.2f}%")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: u(x,t)
    ax = axes[0]
    for l in range(0, L, max(1, L//5)):
        ax.plot(x_grid, u_time[l], label=f't={t_grid[l]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('PDE Solution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: phi comparison
    ax = axes[1]
    ax.plot(r_eval, phi_true_vals, 'b-', lw=2, label='True φ')
    ax.plot(r_eval, phi_learned_vals, 'r--', lw=2, label='Learned φ')
    ax.set_xlabel('r')
    ax.set_ylabel('φ(r)')
    ax.set_title(f'φ Comparison (Error: {error*100:.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: A matrix
    ax = axes[2]
    im = ax.imshow(np.log10(np.abs(A_mat) + 1e-15), cmap='viridis')
    ax.set_title(f'log10(|A|), cond={cond:.1e}')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('experiments/ips_unlabeled/img/test_fei_lu_pde.png', dpi=150)
    plt.close()
    print(f"\nPlot saved to experiments/ips_unlabeled/img/test_fei_lu_pde.png")

    # Verdict
    print("\n" + "=" * 70)
    if error < 0.1:
        print(f"✅ SUCCESS: Error {error*100:.2f}% < 10%")
        return 0
    else:
        print(f"❌ FAIL: Error {error*100:.2f}% > 10%")
        return 1


if __name__ == '__main__':
    sys.exit(main())
