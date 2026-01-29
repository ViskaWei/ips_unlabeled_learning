#!/usr/bin/env python
"""Test Fei Lu method with ORTHOGONAL basis functions.

Key insight: B-spline basis functions become collinear after convolution.
Solution: Use orthogonal polynomials that maintain orthogonality properties.

Options:
1. Legendre polynomials (orthogonal on [-1, 1])
2. Chebyshev polynomials (orthogonal with weight)
3. Hermite polynomials (orthogonal with Gaussian weight)
4. Gram-Schmidt orthogonalization of convolved basis
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy import integrate, special
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


def compute_A_and_b_with_funcs(u_time, t_grid, x_grid, phi_funcs, Phi_funcs, nu):
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


def create_legendre_basis(n_basis, r_max):
    """Create scaled Legendre polynomial basis for φ.

    φ_n(r) = P'_n(2r/r_max - 1) * r  (derivative ensures φ(0)=0)
    Actually, simpler: use r * P_n(r/r_max) which naturally has φ(0)=0
    """
    phi_funcs = []
    Phi_funcs = []

    for n in range(n_basis):
        # φ_n(r) = r * P_n(r/r_max)
        # This ensures φ(0) = 0
        def phi_n(r, n=n):
            x = np.clip(r / r_max, 0, 1)  # Scale to [0, 1]
            # Use Legendre on [0, 1] by shifting: P_n(2x-1)
            return r * special.eval_legendre(n, 2*x - 1)

        # Φ_n(r) = integral of φ_n from 0 to r
        def Phi_n(r, n=n):
            # Numerical integration
            if np.isscalar(r):
                result, _ = integrate.quad(lambda s: phi_n(s, n), 0, r)
                return result
            else:
                return np.array([integrate.quad(lambda s: phi_n(s, n), 0, ri)[0] for ri in r])

        phi_funcs.append(phi_n)
        Phi_funcs.append(Phi_n)

    return phi_funcs, Phi_funcs


def create_monomial_basis(n_basis, with_exp=False):
    """Create monomial basis: r, r^2, r^3, ... or r, r*exp(-r), r^3, ..."""
    phi_funcs = []
    Phi_funcs = []

    for k in range(1, n_basis + 1):
        if with_exp and k == 2:
            # Replace r^2 with r*exp(-r^2/2)
            phi_funcs.append(lambda r: -r * np.exp(-r**2 / 2))
            Phi_funcs.append(lambda r: np.exp(-r**2 / 2))
        else:
            phi_funcs.append(lambda r, p=k: r**p)
            Phi_funcs.append(lambda r, p=k: r**(p+1) / (p+1))

    return phi_funcs, Phi_funcs


def create_chebyshev_basis(n_basis, r_max):
    """Create Chebyshev basis: r * T_n(r/r_max)."""
    phi_funcs = []
    Phi_funcs = []

    for n in range(n_basis):
        def phi_n(r, n=n):
            x = np.clip(r / r_max, 0, 1)
            return r * special.eval_chebyt(n, 2*x - 1)

        def Phi_n(r, n=n):
            if np.isscalar(r):
                result, _ = integrate.quad(lambda s: phi_n(s, n), 0, r)
                return result
            else:
                return np.array([integrate.quad(lambda s: phi_n(s, n), 0, ri)[0] for ri in r])

        phi_funcs.append(phi_n)
        Phi_funcs.append(Phi_n)

    return phi_funcs, Phi_funcs


def gram_schmidt_on_K(K_vectors, u_weight):
    """Gram-Schmidt orthogonalization with weight u."""
    n = len(K_vectors)
    Q = []

    for i in range(n):
        v = K_vectors[i].copy()
        for j in range(i):
            # Inner product with weight u
            proj = np.sum(v * Q[j] * u_weight) / np.sum(Q[j]**2 * u_weight)
            v = v - proj * Q[j]

        norm = np.sqrt(np.sum(v**2 * u_weight))
        if norm > 1e-10:
            Q.append(v / norm)
        else:
            Q.append(np.zeros_like(v))

    return Q


def test_basis(name, phi_funcs, Phi_funcs, phi_true, u_sub, t_sub, x_grid, nu, r_max):
    """Test a basis set."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"{'=' * 60}")

    n_basis = len(phi_funcs)
    dx = x_grid[1] - x_grid[0]

    # First check: can basis fit phi_true?
    r_fit = np.linspace(0.01, r_max * 0.8, 100)
    phi_true_vals = phi_true(r_fit)

    B_fit = np.zeros((len(r_fit), n_basis))
    for i in range(n_basis):
        B_fit[:, i] = phi_funcs[i](r_fit)

    c_fit, _, rank, s = np.linalg.lstsq(B_fit, phi_true_vals, rcond=1e-6)
    phi_fitted = B_fit @ c_fit
    fit_error = np.sqrt(np.mean((phi_fitted - phi_true_vals)**2)) / np.sqrt(np.mean(phi_true_vals**2))

    print(f"  Basis fit of phi_true: {fit_error:.2%}")
    print(f"  Fit coefficients: {c_fit}")

    # Compute A and b
    print("  Computing A and b...")
    t0 = time.time()
    A, b = compute_A_and_b_with_funcs(u_sub, t_sub, x_grid, phi_funcs, Phi_funcs, nu)
    print(f"  Time: {time.time() - t0:.1f}s")

    cond_A = np.linalg.cond(A)
    print(f"  cond(A) = {cond_A:.2e}")
    print(f"  A diagonal: {np.diag(A)}")

    # SVD analysis
    U, s_vals, Vt = np.linalg.svd(A)
    print(f"  Singular values: {s_vals}")

    # Solve
    try:
        c_opt = np.linalg.lstsq(A, b, rcond=1e-6)[0]
        print(f"  c_opt: {c_opt}")

        # Evaluate learned phi
        phi_learned = sum(c_opt[i] * phi_funcs[i](r_fit) for i in range(n_basis))

        learn_error = np.sqrt(np.mean((phi_learned - phi_true_vals)**2)) / np.sqrt(np.mean(phi_true_vals**2))
        print(f"  Learning error: {learn_error:.2%}")

        return {
            'name': name,
            'fit_error': fit_error,
            'cond_A': cond_A,
            'learn_error': learn_error,
            'c_opt': c_opt,
            'phi_learned': phi_learned,
            'phi_true': phi_true_vals,
            'r_eval': r_fit,
            'status': 'PASS' if learn_error < 0.3 else 'FAIL'
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {
            'name': name,
            'fit_error': fit_error,
            'cond_A': cond_A,
            'learn_error': float('inf'),
            'status': 'ERROR'
        }


def main():
    print("=" * 70)
    print("Test: Orthogonal Basis Functions for Fei Lu Method")
    print("=" * 70)
    print("\nGoal: Find basis functions that avoid collinearity after convolution")

    # Setup
    nu = 0.5
    x_min, x_max = -5, 5
    M = 100
    L = 3000
    T = 1.0

    x_grid = np.linspace(x_min, x_max, M)
    t_grid = np.linspace(0, T, L)

    phi_true = lambda r: -r * np.exp(-r**2 / 2)
    r_max = 4.0

    print("\n[1] Generating PDE data...")
    u_time = generate_pde_semi_implicit(x_grid, t_grid, nu, phi_true)

    L_sub = 30
    stride = L // L_sub
    t_sub = t_grid[::stride]
    u_sub = u_time[::stride]

    results = []

    # Test 1: Simple monomials (r, r^2, r^3)
    phi_mono = [lambda r: r, lambda r: r**2, lambda r: r**3]
    Phi_mono = [lambda r: r**2/2, lambda r: r**3/3, lambda r: r**4/4]
    results.append(test_basis("Monomials (r, r², r³)", phi_mono, Phi_mono, phi_true, u_sub, t_sub, x_grid, nu, r_max))

    # Test 2: Monomials with Gaussian
    phi_mixed = [lambda r: r, lambda r: -r * np.exp(-r**2 / 2), lambda r: r**3]
    Phi_mixed = [lambda r: r**2/2, lambda r: np.exp(-r**2 / 2), lambda r: r**4/4]
    results.append(test_basis("Mixed (r, r*exp, r³)", phi_mixed, Phi_mixed, phi_true, u_sub, t_sub, x_grid, nu, r_max))

    # Test 3: Only 2 basis functions (minimal)
    phi_2 = [lambda r: r, lambda r: -r * np.exp(-r**2 / 2)]
    Phi_2 = [lambda r: r**2/2, lambda r: np.exp(-r**2 / 2)]
    results.append(test_basis("Minimal (r, r*exp)", phi_2, Phi_2, phi_true, u_sub, t_sub, x_grid, nu, r_max))

    # Test 4: Scaled Legendre
    # Note: This requires scipy for Legendre polynomials
    n_leg = 4
    phi_leg, Phi_leg = create_legendre_basis(n_leg, r_max)
    results.append(test_basis(f"Legendre (n={n_leg})", phi_leg, Phi_leg, phi_true, u_sub, t_sub, x_grid, nu, r_max))

    # Test 5: Different Gaussians (varying sigma)
    sigmas = [0.5, 1.0, 1.5, 2.0]
    phi_gauss = [lambda r, s=s: -r * np.exp(-r**2 / (2*s**2)) for s in sigmas]
    Phi_gauss = [lambda r, s=s: s**2 * np.exp(-r**2 / (2*s**2)) for s in sigmas]
    results.append(test_basis("Gaussians (σ=0.5,1,1.5,2)", phi_gauss, Phi_gauss, phi_true, u_sub, t_sub, x_grid, nu, r_max))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Basis':<30} {'Fit%':>8} {'cond(A)':>12} {'Learn%':>10} {'Status':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<30} {r['fit_error']:>7.1%} {r['cond_A']:>12.2e} {r['learn_error']:>9.1%} {r['status']:>8}")

    # Plot the best result
    best = min(results, key=lambda x: x['learn_error'])
    if best['learn_error'] < float('inf'):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(best['r_eval'], best['phi_true'], 'r-', lw=2, label='True φ')
        ax.plot(best['r_eval'], best['phi_learned'], 'b--', lw=2, label=f"Learned ({best['name']})")
        ax.set_xlabel('r')
        ax.set_ylabel('φ(r)')
        ax.set_title(f"Best Result: {best['name']}\nError: {best['learn_error']:.1%}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('experiments/ips_unlabeled/img/test_orthogonal_basis.png', dpi=150)
        plt.close()
        print(f"\nPlot saved to experiments/ips_unlabeled/img/test_orthogonal_basis.png")

    return 0 if any(r['status'] == 'PASS' for r in results) else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
