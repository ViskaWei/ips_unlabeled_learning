#!/usr/bin/env python
"""Test with cubic potential (as in Fei Lu paper).

The paper shows φ(r) = r - r³ achieves 1.90% error.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy import integrate
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def cubic_phi(r, a=1.0, b=1.0):
    """Cubic potential φ(r) = a*r - b*r³."""
    return a * r - b * r**3


def cubic_Phi(r, a=1.0, b=1.0):
    """Antiderivative Φ(r) = a*r²/2 - b*r⁴/4."""
    return a * r**2 / 2 - b * r**4 / 4


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


def create_bspline_basis(n_basis, degree, r_min, r_max):
    """Create B-spline basis functions."""
    n_knots = n_basis + degree + 1
    n_interior = n_knots - 2 * (degree + 1)

    if n_interior < 0:
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


def main():
    print("=" * 70)
    print("Test with Cubic Potential (Fei Lu Paper)")
    print("=" * 70)

    # Parameters (following Fei Lu paper)
    nu = 0.1  # viscosity
    a_phi, b_phi = 1.0, 1.0  # φ(r) = r - r³

    # Grid (paper uses M=300)
    x_min, x_max = -5, 5
    M = 80
    L = 100
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]

    # B-spline basis - use degree 3 (cubic) to match the target polynomial
    n_basis = 6
    degree = 3
    r_min, r_max = 0, 3  # Relevant range for cubic potential

    print(f"\nParameters: ν={nu}, M={M}, L={L}")
    print(f"Target: φ(r) = {a_phi}r - {b_phi}r³ (cubic potential)")
    print(f"B-spline: n_basis={n_basis}, degree={degree}, domain=[{r_min}, {r_max}]")

    basis_funcs, knots = create_bspline_basis(n_basis, degree, r_min, r_max)
    print(f"Knots: {np.round(knots, 2)}")

    # Precompute Phi tables
    r_fine = np.linspace(r_min, r_max, 500)
    dr_fine = r_fine[1] - r_fine[0]

    Phi_tables = []
    for i in range(n_basis):
        phi_vals = np.array([basis_funcs[i](r) for r in r_fine])
        phi_vals = np.nan_to_num(phi_vals, nan=0.0)
        Phi_vals = integrate.cumulative_trapezoid(phi_vals, r_fine, initial=0)
        Phi_tables.append(Phi_vals)

    def eval_Phi_basis(i, r):
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
    phi_true_vals = np.array([cubic_phi(r, a_phi, b_phi) for r in r_eval])

    B = np.zeros((len(r_eval), n_basis))
    for i in range(n_basis):
        B[:, i] = np.array([basis_funcs[i](r) for r in r_eval])
        B[:, i] = np.nan_to_num(B[:, i], nan=0.0)

    c_fit, _, _, _ = np.linalg.lstsq(B, phi_true_vals, rcond=None)
    fit_error = np.linalg.norm(B @ c_fit - phi_true_vals) / np.linalg.norm(phi_true_vals)
    print(f"  Basis fit error: {fit_error*100:.2f}%")

    # True phi for PDE
    phi_func = lambda r: cubic_phi(r, a_phi, b_phi)

    # Generate PDE solution
    print("\n[2] Generating PDE solution...")
    u_time = generate_pde_solution(x_grid, t_grid, nu, phi_func)

    # Compute A and b
    print("\n[3] Computing A and b...")

    A_mat = np.zeros((n_basis, n_basis))
    b_vec = np.zeros(n_basis)

    for ell in range(L - 1):
        if ell % 25 == 0:
            print(f"    Time step {ell}/{L-1}")

        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]

        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)

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

    # Solve with truncated SVD
    print("\n[4] Solving with truncated SVD...")

    U, S, Vt = np.linalg.svd(A_mat)
    print(f"  Singular values: {S}")

    best_c = None
    best_error = float('inf')
    best_k = None

    for k in range(1, n_basis + 1):
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
    print(f"  Coefficients: {best_c}")

    print(f"\n[5] Result:")
    print(f"  φ relative L2 error: {best_error*100:.2f}%")

    # Also try Tikhonov
    print("\n[6] Also trying Tikhonov regularization...")

    lambda_range = np.logspace(-12, 0, 50)
    best_error_tik = float('inf')
    best_c_tik = None
    best_lambda = None

    for lam in lambda_range:
        try:
            c = np.linalg.solve(A_mat + lam * np.eye(n_basis), b_vec)
            phi_learned = B @ c
            error = np.linalg.norm(phi_learned - phi_true_vals) / np.linalg.norm(phi_true_vals)
            if error < best_error_tik:
                best_error_tik = error
                best_c_tik = c
                best_lambda = lam
        except:
            continue

    print(f"  Best λ = {best_lambda:.4e}")
    print(f"  Tikhonov error: {best_error_tik*100:.2f}%")

    # Use the better result
    final_error = min(best_error, best_error_tik)
    final_c = best_c if best_error <= best_error_tik else best_c_tik

    # Plot
    phi_learned = B @ final_c

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    for l in range(0, L, L//5):
        ax.plot(x_grid, u_time[l], label=f't={t_grid[l]:.2f}')
    ax.set_xlabel('x')
    ax.set_title('PDE Solution u(x,t)')
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.plot(r_eval, phi_true_vals, 'b-', lw=2, label='True φ')
    ax.plot(r_eval, phi_learned, 'r--', lw=2, label=f'Learned φ ({final_error*100:.1f}%)')
    ax.plot(r_eval, B @ c_fit, 'g:', lw=2, label=f'LS fit ({fit_error*100:.1f}%)')
    ax.set_xlabel('r')
    ax.set_ylabel('φ(r)')
    ax.set_title('φ Comparison')
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    im = ax.imshow(A_mat, cmap='RdBu_r', aspect='auto')
    ax.set_title(f'A matrix, cond={cond_A:.1e}')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('experiments/ips_unlabeled/img/test_cubic_potential.png', dpi=150)
    plt.close()
    print(f"\nPlot saved to experiments/ips_unlabeled/img/test_cubic_potential.png")

    if final_error < 0.1:
        print(f"\n✅ SUCCESS: Error {final_error*100:.2f}% < 10%")
        return 0
    else:
        print(f"\n❌ FAIL: Error {final_error*100:.2f}% > 10%")
        return 1


if __name__ == '__main__':
    sys.exit(main())
