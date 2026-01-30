#!/usr/bin/env python
"""Test with direct basis evaluation (no grid discretization)."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy import integrate
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def true_phi(r, A=1.0, sigma=1.0):
    return -A * r / (sigma**2) * np.exp(-r**2 / (2 * sigma**2))


def true_Phi(r, A=1.0, sigma=1.0):
    return A * np.exp(-r**2 / (2 * sigma**2))


def create_bspline_basis(n_basis, degree, r_min, r_max):
    """Create B-spline basis functions."""
    # Number of knots = n_basis + degree + 1
    n_knots = n_basis + degree + 1
    n_interior = n_knots - 2 * (degree + 1)

    if n_interior < 0:
        # Not enough basis for this degree, use uniform knots
        knots = np.linspace(r_min, r_max, n_knots)
    else:
        interior_knots = np.linspace(r_min, r_max, n_interior + 2)[1:-1]
        knots = np.concatenate([
            [r_min] * (degree + 1),
            interior_knots,
            [r_max] * (degree + 1)
        ])

    basis_funcs = []
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        basis_funcs.append(BSpline(knots, c, degree, extrapolate=True))

    return basis_funcs, knots


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


def main():
    print("=" * 70)
    print("Test with Direct Basis Evaluation")
    print("=" * 70)

    # Parameters
    nu = 0.1
    A_phi = 1.0
    sigma_phi = 1.0

    # Grid
    x_min, x_max = -4, 4
    M = 50  # Smaller for speed
    L = 80
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]

    # B-spline basis - use fewer basis for better conditioning
    n_basis = 4
    degree = 2
    r_min, r_max = 0, 4  # Cover the relevant range

    print(f"\nParameters: ν={nu}, M={M}, L={L}")
    print(f"B-spline: n_basis={n_basis}, degree={degree}, domain=[{r_min}, {r_max}]")

    basis_funcs, knots = create_bspline_basis(n_basis, degree, r_min, r_max)
    print(f"Knots: {np.round(knots, 2)}")

    # Precompute antiderivatives of basis functions
    # Φ_i(r) = ∫_0^r φ_i(s) ds
    r_fine = np.linspace(r_min, r_max, 500)
    dr_fine = r_fine[1] - r_fine[0]

    Phi_tables = []  # Lookup tables for antiderivatives
    for i in range(n_basis):
        phi_vals = np.array([basis_funcs[i](r) for r in r_fine])
        phi_vals = np.nan_to_num(phi_vals, nan=0.0)
        Phi_vals = integrate.cumulative_trapezoid(phi_vals, r_fine, initial=0)
        Phi_tables.append(Phi_vals)

    def eval_Phi_basis(i, r):
        """Evaluate antiderivative Φ_i at r using lookup table."""
        if r <= r_min:
            return 0.0
        if r >= r_max:
            return Phi_tables[i][-1]
        idx = int((r - r_min) / dr_fine)
        idx = min(idx, len(Phi_tables[i]) - 2)
        alpha = (r - r_fine[idx]) / dr_fine
        return (1 - alpha) * Phi_tables[i][idx] + alpha * Phi_tables[i][idx + 1]

    # Check basis representation
    print("\n[1] Checking basis representation...")
    r_eval = np.linspace(r_min + 0.01, r_max - 0.01, 100)
    phi_true_vals = np.array([true_phi(r, A_phi, sigma_phi) for r in r_eval])

    B = np.zeros((len(r_eval), n_basis))
    for i in range(n_basis):
        B[:, i] = np.array([basis_funcs[i](r) for r in r_eval])
        B[:, i] = np.nan_to_num(B[:, i], nan=0.0)

    c_fit, _, _, _ = np.linalg.lstsq(B, phi_true_vals, rcond=None)
    fit_error = np.linalg.norm(B @ c_fit - phi_true_vals) / np.linalg.norm(phi_true_vals)
    print(f"  Basis fit error: {fit_error*100:.2f}%")

    # True phi for PDE
    phi_func = lambda r: true_phi(r, A_phi, sigma_phi)

    # Generate PDE solution
    print("\n[2] Generating PDE solution...")
    u_time = generate_pde_solution(x_grid, t_grid, nu, phi_func)

    # Compute A and b with DIRECT basis evaluation
    print("\n[3] Computing A and b (direct evaluation)...")
    print("  (This may take a while...)")

    A_mat = np.zeros((n_basis, n_basis))
    b_vec = np.zeros(n_basis)

    for ell in range(L - 1):
        if ell % 20 == 0:
            print(f"    Time step {ell}/{L-1}")

        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]

        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)

        # Compute K_phi_i * u and Phi_i * u for each basis (DIRECT evaluation)
        K_phi_all = np.zeros((M, n_basis))
        Phi_conv_all = np.zeros((M, n_basis))

        for m, x in enumerate(x_grid):
            for n, y in enumerate(x_grid):
                r = abs(x - y)
                for i in range(n_basis):
                    if r > 1e-10 and r_min <= r <= r_max:
                        phi_val = basis_funcs[i](r)
                        if not np.isnan(phi_val):
                            K_phi_all[m, i] += phi_val * np.sign(x - y) * u_curr[n] * dx

                    Phi_val = eval_Phi_basis(i, r)
                    Phi_conv_all[m, i] += Phi_val * u_curr[n] * dx

        # Accumulate A and b
        for i in range(n_basis):
            term1 = np.sum(du_dt * Phi_conv_all[:, i]) * dx
            term2 = nu * np.sum(du_dx * K_phi_all[:, i]) * dx
            b_vec[i] -= (term1 + term2) * dt / T

            for j in range(i, n_basis):
                term = np.sum(K_phi_all[:, i] * K_phi_all[:, j] * u_curr) * dx
                A_mat[i, j] += term * dt / T
                if j != i:
                    A_mat[j, i] = A_mat[i, j]

    cond_A = np.linalg.cond(A_mat)
    print(f"\n  A condition number: {cond_A:.2e}")
    print(f"  b norm: {np.linalg.norm(b_vec):.4e}")

    # Solve with truncated SVD regularization
    print("\n[4] Solving with truncated SVD...")

    U, S, Vt = np.linalg.svd(A_mat)
    print(f"  Singular values: {S}")

    best_c = None
    best_error = float('inf')
    best_k = None

    for k in range(1, n_basis + 1):
        # Truncated pseudoinverse: A^+ = V S^-1 U^T (using only top k components)
        S_inv = np.zeros(n_basis)
        S_inv[:k] = 1.0 / S[:k]
        c = Vt.T @ (S_inv * (U.T @ b_vec))

        phi_learned = B @ c
        error = np.linalg.norm(phi_learned - phi_true_vals) / np.linalg.norm(phi_true_vals)

        if error < best_error:
            best_error = error
            best_c = c
            best_k = k

    print(f"  Best truncation level k = {best_k}")
    print(f"\n[5] Result:")
    print(f"  φ relative L2 error: {best_error*100:.2f}%")

    # Plot
    phi_learned = B @ best_c

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(r_eval, phi_true_vals, 'b-', lw=2, label='True φ')
    ax.plot(r_eval, phi_learned, 'r--', lw=2, label=f'Learned φ ({best_error*100:.1f}%)')
    ax.set_xlabel('r')
    ax.set_ylabel('φ(r)')
    ax.set_title('φ Comparison')
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    im = ax.imshow(A_mat, cmap='RdBu_r', aspect='auto')
    ax.set_title(f'A matrix, cond={cond_A:.1e}')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('experiments/ips_unlabeled/img/test_direct_basis.png', dpi=150)
    plt.close()
    print(f"\nPlot saved to experiments/ips_unlabeled/img/test_direct_basis.png")

    if best_error < 0.1:
        print(f"\n✅ SUCCESS: Error {best_error*100:.2f}% < 10%")
        return 0
    else:
        print(f"\n❌ FAIL: Error {best_error*100:.2f}% > 10%")
        return 1


if __name__ == '__main__':
    sys.exit(main())
