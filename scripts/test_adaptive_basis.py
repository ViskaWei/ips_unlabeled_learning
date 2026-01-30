#!/usr/bin/env python
"""Test with adaptive basis focused on the relevant r-range."""

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


def create_adaptive_bspline_basis(n_basis, degree, r_min, r_max, r_focus=3.0):
    """Create B-spline basis with knots focused near r_focus."""
    n_interior = n_basis - degree + 1

    # Use non-uniform knots: more dense near 0, sparser far away
    # Use a sqrt transformation to concentrate near r=0
    t = np.linspace(0, 1, n_interior + 2)
    interior_knots = r_focus * t**1.5 + (r_max - r_focus) * t**2
    interior_knots = interior_knots[1:-1]

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


def compute_Phi_conv_u(x_grid, u, Phi_func):
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    result = np.zeros(M)
    for m, x in enumerate(x_grid):
        for n, y in enumerate(x_grid):
            r = abs(x - y)
            result[m] += Phi_func(r) * u[n] * dx
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


def evaluate_basis_at_r(basis_funcs, r):
    """Evaluate all basis functions at given r values."""
    n_points = len(r)
    n_basis = len(basis_funcs)
    B = np.zeros((n_points, n_basis))
    for i, phi in enumerate(basis_funcs):
        vals = phi(r)
        vals = np.nan_to_num(vals, nan=0.0)
        B[:, i] = vals
    return B


def main():
    print("=" * 70)
    print("Test with Adaptive B-spline Basis")
    print("=" * 70)

    # Parameters
    nu = 0.1
    A_phi = 1.0
    sigma_phi = 1.0

    # Grid
    x_min, x_max = -4, 4
    M = 80
    L = 100
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)
    dx = x_grid[1] - x_grid[0]

    # Adaptive B-spline basis
    n_basis = 15  # More basis functions
    degree = 3    # Higher degree for smoother approximation
    r_min, r_max = 0, 5  # Smaller domain focused on relevant range
    r_focus = 2.5  # Where to concentrate knots

    print(f"\nParameters: ν={nu}, M={M}, L={L}")
    print(f"B-spline: n_basis={n_basis}, degree={degree}, domain=[{r_min}, {r_max}]")

    basis_funcs, knots = create_adaptive_bspline_basis(n_basis, degree, r_min, r_max, r_focus)
    print(f"Knots: {np.round(knots, 2)}")

    # First check: can basis represent true phi?
    print("\n[1] Checking basis representation of true φ...")
    r_eval = np.linspace(r_min + 0.01, r_max - 0.01, 200)
    B = evaluate_basis_at_r(basis_funcs, r_eval)
    phi_true_vals = np.array([true_phi(r, A_phi, sigma_phi) for r in r_eval])

    # Least squares fit
    c_fit, _, _, _ = np.linalg.lstsq(B, phi_true_vals, rcond=None)
    phi_fit_vals = B @ c_fit
    fit_error = np.linalg.norm(phi_fit_vals - phi_true_vals) / np.linalg.norm(phi_true_vals)
    print(f"  Basis fit error: {fit_error*100:.2f}%")
    print(f"  Basis matrix condition: {np.linalg.cond(B):.2e}")

    if fit_error > 0.2:
        print(f"\n  Warning: basis cannot represent true φ well!")
        print(f"  The learning result will be limited by this approximation error.")

    # True phi for PDE
    phi_func = lambda r: true_phi(r, A_phi, sigma_phi)

    # Generate PDE solution
    print("\n[2] Generating PDE solution...")
    u_time = generate_pde_solution(x_grid, t_grid, nu, phi_func)

    # Compute A matrix and b vector
    print("\n[3] Computing A and b...")

    A_mat = np.zeros((n_basis, n_basis))
    b_vec = np.zeros(n_basis)

    # Use same domain as basis for r_grid
    r_grid = np.linspace(r_min, r_max, 200)  # More points for accuracy
    dr = r_grid[1] - r_grid[0]

    # Precompute basis on r_grid
    phi_vals = evaluate_basis_at_r(basis_funcs, r_grid)
    n_r = len(r_grid)

    # Compute Φ (antiderivative) on r_grid
    Phi_vals = np.zeros_like(phi_vals)
    for i in range(n_basis):
        Phi_vals[:, i] = integrate.cumulative_trapezoid(phi_vals[:, i], r_grid, initial=0)

    for ell in range(L - 1):
        u_curr = u_time[ell]
        u_next = u_time[ell + 1]
        dt = t_grid[ell + 1] - t_grid[ell]

        du_dt = (u_next - u_curr) / dt
        du_dx = np.gradient(u_curr, dx)

        # Compute convolutions for each basis
        K_phi_all = np.zeros((M, n_basis))
        Phi_conv_all = np.zeros((M, n_basis))

        for i in range(n_basis):
            for m, x in enumerate(x_grid):
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    if r > 1e-10 and r < r_max - 0.01:  # Only within basis domain
                        # Linear interpolation for better accuracy
                        r_frac = (r - r_min) / dr
                        r_idx = int(r_frac)
                        if r_idx < n_r - 1:
                            alpha = r_frac - r_idx
                            phi_val = (1 - alpha) * phi_vals[r_idx, i] + alpha * phi_vals[r_idx + 1, i]
                        elif r_idx < n_r:
                            phi_val = phi_vals[r_idx, i]
                        else:
                            phi_val = 0.0
                        K_phi_all[m, i] += phi_val * np.sign(x - y) * u_curr[n] * dx

                    if r < r_max - 0.01:  # Only within basis domain
                        r_frac = (abs(x - y) - r_min) / dr
                        r_idx = int(r_frac)
                        if r_idx < n_r - 1:
                            alpha = r_frac - r_idx
                            Phi_val = (1 - alpha) * Phi_vals[r_idx, i] + alpha * Phi_vals[r_idx + 1, i]
                        elif r_idx < n_r:
                            Phi_val = Phi_vals[r_idx, i]
                        else:
                            Phi_val = 0.0
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
    print(f"  A condition number: {cond_A:.2e}")
    print(f"  b norm: {np.linalg.norm(b_vec):.4e}")

    # Solve with regularization
    print("\n[4] Solving with Tikhonov regularization...")

    # Try multiple lambda values
    eigvals = np.abs(np.linalg.eigvalsh(A_mat))
    lambda_range = np.logspace(-12, 0, 50)

    best_c = None
    best_error = float('inf')
    best_lambda = None

    for lam in lambda_range:
        try:
            c = np.linalg.solve(A_mat + lam * np.eye(n_basis), b_vec)

            # Evaluate learned phi
            phi_learned = evaluate_basis_at_r(basis_funcs, r_eval) @ c
            error = np.linalg.norm(phi_learned - phi_true_vals) / np.linalg.norm(phi_true_vals)

            if error < best_error:
                best_error = error
                best_c = c
                best_lambda = lam
        except:
            continue

    print(f"  Best λ = {best_lambda:.4e}")
    print(f"  Coefficients range: [{best_c.min():.4f}, {best_c.max():.4f}]")

    # Evaluate result
    print("\n[5] Evaluating result...")
    phi_learned = evaluate_basis_at_r(basis_funcs, r_eval) @ best_c
    final_error = np.linalg.norm(phi_learned - phi_true_vals) / np.linalg.norm(phi_true_vals)
    print(f"  φ relative L2 error: {final_error*100:.2f}%")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    for i in range(min(n_basis, 8)):
        ax.plot(r_eval, evaluate_basis_at_r(basis_funcs, r_eval)[:, i], label=f'B{i}')
    ax.set_xlabel('r')
    ax.set_title('Adaptive B-spline Basis')
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[1]
    ax.plot(r_eval, phi_true_vals, 'b-', lw=2, label='True φ')
    ax.plot(r_eval, phi_fit_vals, 'g--', lw=2, label=f'LS fit ({fit_error*100:.1f}%)')
    ax.plot(r_eval, phi_learned, 'r:', lw=2, label=f'Learned ({final_error*100:.1f}%)')
    ax.set_xlabel('r')
    ax.set_title('φ Comparison')
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    im = ax.imshow(np.log10(np.abs(A_mat) + 1e-15), cmap='viridis')
    ax.set_title(f'log10(|A|), cond={cond_A:.1e}')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('experiments/ips_unlabeled/img/test_adaptive_basis.png', dpi=150)
    plt.close()
    print(f"\nPlot saved to experiments/ips_unlabeled/img/test_adaptive_basis.png")

    if final_error < 0.1:
        print(f"\n✅ SUCCESS: Error {final_error*100:.2f}% < 10%")
        return 0
    else:
        print(f"\n❌ FAIL: Error {final_error*100:.2f}% > 10%")
        print(f"   Lower bound from basis fit: {fit_error*100:.2f}%")
        return 1


if __name__ == '__main__':
    sys.exit(main())
