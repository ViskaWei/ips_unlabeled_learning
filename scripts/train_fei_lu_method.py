#!/usr/bin/env python
"""MVP-2.0: Implement Fei Lu's method for learning interaction kernels.

Reference: Lang & Lu, "Learning interaction kernels in mean-field equations
of 1st-order systems of interacting particles", SIAM J. Sci. Comput. 2022

Key differences from our weak-form loss:
1. No external potential V (only interaction kernel φ)
2. Use B-spline basis functions (not NN)
3. Use Tikhonov regularization with L-curve
4. Error functional based on likelihood (Eq 2.3)
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time

import numpy as np
from scipy import integrate
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.sde_simulator import SDESimulator
from core.potentials import GaussianInteraction


# ==============================================================================
# B-spline basis functions
# ==============================================================================

def create_bspline_basis(n_basis: int, degree: int, r_min: float, r_max: float):
    """Create B-spline basis functions.

    Args:
        n_basis: Number of basis functions
        degree: Spline degree (1=linear, 2=quadratic, 3=cubic)
        r_min, r_max: Domain bounds
    Returns:
        List of callable basis functions, knot vector
    """
    # Number of interior knots
    n_interior = n_basis - degree + 1

    # Create knot vector with repeated endpoints
    interior_knots = np.linspace(r_min, r_max, n_interior + 2)[1:-1]
    knots = np.concatenate([
        [r_min] * degree,
        interior_knots,
        [r_max] * degree
    ])

    # Create basis functions
    basis_funcs = []
    for i in range(n_basis):
        # Create coefficient vector with 1 at position i
        c = np.zeros(n_basis)
        c[i] = 1.0
        basis_funcs.append(BSpline(knots, c, degree, extrapolate=False))

    return basis_funcs, knots


def evaluate_basis(basis_funcs, r: np.ndarray) -> np.ndarray:
    """Evaluate all basis functions at given points.

    Args:
        basis_funcs: List of basis functions
        r: Points to evaluate, shape (n_points,)
    Returns:
        Basis matrix, shape (n_points, n_basis)
    """
    n_points = len(r)
    n_basis = len(basis_funcs)
    B = np.zeros((n_points, n_basis))
    for i, phi in enumerate(basis_funcs):
        vals = phi(r)
        # Handle NaN from extrapolation
        vals = np.nan_to_num(vals, nan=0.0)
        B[:, i] = vals
    return B


def evaluate_basis_derivative(basis_funcs, r: np.ndarray, order: int = 1) -> np.ndarray:
    """Evaluate derivatives of basis functions.

    Args:
        basis_funcs: List of basis functions
        r: Points to evaluate
        order: Derivative order
    Returns:
        Derivative matrix, shape (n_points, n_basis)
    """
    n_points = len(r)
    n_basis = len(basis_funcs)
    B = np.zeros((n_points, n_basis))
    for i, phi in enumerate(basis_funcs):
        dphi = phi.derivative(order)
        vals = dphi(r)
        vals = np.nan_to_num(vals, nan=0.0)
        B[:, i] = vals
    return B


# ==============================================================================
# Kernel density estimation from particles
# ==============================================================================

def estimate_density(particles: np.ndarray, x_grid: np.ndarray,
                    bandwidth: float = 0.3) -> np.ndarray:
    """Estimate density u(x,t) from particle positions using KDE.

    Args:
        particles: Particle positions, shape (N,) for 1D
        x_grid: Grid points, shape (M,)
        bandwidth: KDE bandwidth
    Returns:
        Density estimate, shape (M,)
    """
    N = len(particles)
    M = len(x_grid)
    dx = x_grid[1] - x_grid[0]

    # Simple Gaussian KDE
    u = np.zeros(M)
    for xi in particles:
        # Gaussian kernel centered at xi
        u += np.exp(-0.5 * ((x_grid - xi) / bandwidth)**2) / (bandwidth * np.sqrt(2 * np.pi))
    u /= N

    # Normalize to ensure integral = 1
    integral = np.sum(u) * dx
    if integral > 1e-10:
        u /= integral

    return u


def estimate_density_time_series(data: np.ndarray, x_grid: np.ndarray,
                                 bandwidth: float = 0.3) -> np.ndarray:
    """Estimate u(x,t) for all snapshots.

    Args:
        data: Particle data, shape (M_samples, L, N, d)
        x_grid: Spatial grid, shape (M_grid,)
        bandwidth: KDE bandwidth
    Returns:
        Density estimates, shape (L, M_grid)
    """
    M_samples, L, N, d = data.shape
    M_grid = len(x_grid)

    # Average over all samples for each time
    u_all = np.zeros((L, M_grid))
    for ell in range(L):
        # Collect all particles at time ell across all samples
        all_particles = data[:, ell, :, 0].flatten()  # (M_samples * N,)
        u_all[ell] = estimate_density(all_particles, x_grid, bandwidth)

    return u_all


# ==============================================================================
# Fei Lu's Error Functional (Eq 2.16-2.18)
# ==============================================================================

def compute_convolution(f: np.ndarray, u: np.ndarray, dx: float) -> np.ndarray:
    """Compute convolution f * u using discrete approximation.

    Args:
        f: Function values on grid, shape (M,)
        u: Density values on grid, shape (M,)
        dx: Grid spacing
    Returns:
        Convolution values, shape (M,)
    """
    # Use numpy convolution
    conv = np.convolve(f, u, mode='same') * dx
    return conv


def compute_normal_matrix_and_vector(
    u_time: np.ndarray,  # (L, M_grid)
    t_snapshots: np.ndarray,  # (L,)
    x_grid: np.ndarray,  # (M_grid,)
    basis_funcs: list,
    nu: float,
) -> tuple:
    """Compute normal matrix A and vector b for least squares.

    Reference: Equations (2.17) and (2.18) in Lang & Lu 2022.

    Args:
        u_time: Density u(x,t), shape (L, M_grid)
        t_snapshots: Time points, shape (L,)
        x_grid: Spatial grid, shape (M_grid,)
        basis_funcs: List of B-spline basis functions
        nu: Viscosity (= sigma^2 / 2)
    Returns:
        A: Normal matrix, shape (n_basis, n_basis)
        b: Normal vector, shape (n_basis,)
    """
    L, M_grid = u_time.shape
    n_basis = len(basis_funcs)
    dx = x_grid[1] - x_grid[0]
    T = t_snapshots[-1] - t_snapshots[0]

    A = np.zeros((n_basis, n_basis))
    b = np.zeros(n_basis)

    # Compute pairwise distance grid
    # For 1D: r = |x - y|, so we need to handle this carefully
    # We'll use the approach from the paper: evaluate on a radial grid

    # Get support of density
    r_max = x_grid[-1] - x_grid[0]  # Maximum possible distance
    r_grid = np.linspace(0, r_max, M_grid)
    dr = r_grid[1] - r_grid[0] if len(r_grid) > 1 else 1.0

    # Evaluate basis functions on r_grid
    # phi_i(r) and Phi_i(r) = integral_0^r phi_i(s) ds
    phi_vals = evaluate_basis(basis_funcs, r_grid)  # (M_grid, n_basis)

    # K_phi(x) = phi(|x|) * x / |x| for x != 0
    # For 1D: K_phi(x) = phi(|x|) * sign(x)

    # Compute antiderivatives Phi_i
    Phi_vals = np.zeros_like(phi_vals)
    for i in range(n_basis):
        Phi_vals[:, i] = integrate.cumulative_trapezoid(phi_vals[:, i], r_grid, initial=0)

    # Note: We no longer need dphi_vals for b vector computation

    # Time loop
    for ell in range(L - 1):
        u_curr = u_time[ell]  # (M_grid,)
        u_next = u_time[ell + 1]  # (M_grid,)
        dt = t_snapshots[ell + 1] - t_snapshots[ell]

        # Compute du/dt (finite difference)
        du_dt = (u_next - u_curr) / dt

        # Compute du/dx (central difference)
        du_dx = np.gradient(u_curr, dx)

        for i in range(n_basis):
            # Compute K_phi_i * u for current density
            # For 1D with radial phi: K_phi(x) = phi(|x|) * sign(x)
            # Convolution needs special handling

            # Simplified approach: use symmetric extension
            # K_phi_i * u (x) = sum_y phi_i(|x-y|) * sign(x-y) * u(y) * dy

            # Build K_phi_i on the full x grid
            K_phi_i = np.zeros(M_grid)
            for m, x in enumerate(x_grid):
                # Integrate over y
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    if r > 1e-10:
                        # Find phi_i(r)
                        r_idx = min(int(r / dr), M_grid - 1)
                        phi_val = phi_vals[r_idx, i] if r_idx < M_grid else 0.0
                        K_phi_i[m] += phi_val * np.sign(x - y) * u_curr[n] * dx

            # Compute Phi_i * u
            Phi_i_conv_u = np.zeros(M_grid)
            for m, x in enumerate(x_grid):
                for n, y in enumerate(x_grid):
                    r = abs(x - y)
                    r_idx = min(int(r / dr), M_grid - 1)
                    Phi_val = Phi_vals[r_idx, i] if r_idx < M_grid else 0.0
                    Phi_i_conv_u[m] += Phi_val * u_curr[n] * dx

            # b_i contribution (Eq 2.18)
            # b_i = -1/T sum_l sum_m [du_dt * Phi_i*u + nu * ∇u · (K_phi_i*u)] dx dt
            # Note: ∇u · (K_φ*u) = (du/dx) * (K_φ*u) in 1D
            term1 = np.sum(du_dt * Phi_i_conv_u) * dx
            term2 = nu * np.sum(du_dx * K_phi_i) * dx  # Fixed: was computing u*(div*u), should be ∇u·(K*u)
            b[i] -= (term1 + term2) * dt / T

            for j in range(i, n_basis):
                # Compute K_phi_j * u
                K_phi_j = np.zeros(M_grid)
                for m, x in enumerate(x_grid):
                    for n, y in enumerate(x_grid):
                        r = abs(x - y)
                        if r > 1e-10:
                            r_idx = min(int(r / dr), M_grid - 1)
                            phi_val = phi_vals[r_idx, j] if r_idx < M_grid else 0.0
                            K_phi_j[m] += phi_val * np.sign(x - y) * u_curr[n] * dx

                # A_ij contribution (Eq 2.17)
                # A_ij = 1/T sum_l sum_m [(K_phi_i * u) . (K_phi_j * u) * u] dx dt
                term = np.sum(K_phi_i * K_phi_j * u_curr) * dx
                A[i, j] += term * dt / T
                if j != i:
                    A[j, i] = A[i, j]  # Symmetry

    return A, b


def solve_with_tikhonov(A: np.ndarray, b: np.ndarray,
                        reg_matrix: np.ndarray = None,
                        lambda_range: np.ndarray = None) -> tuple:
    """Solve least squares with Tikhonov regularization using L-curve.

    Args:
        A: Normal matrix
        b: Normal vector
        reg_matrix: Regularization matrix (default: identity)
        lambda_range: Range of regularization parameters to try
    Returns:
        c: Optimal coefficients
        lambda_opt: Optimal regularization parameter
        l_curve_data: L-curve data for plotting
    """
    n = len(b)
    if reg_matrix is None:
        reg_matrix = np.eye(n)

    if lambda_range is None:
        # Use eigenvalues of A to set range
        eigvals = np.linalg.eigvalsh(A)
        lambda_min = max(1e-10, eigvals.min() * 1e-6)
        lambda_max = eigvals.max() * 10
        lambda_range = np.logspace(np.log10(lambda_min), np.log10(lambda_max), 50)

    errors = []
    norms = []
    solutions = []

    for lam in lambda_range:
        # Solve (A + lambda * B) c = b
        c = np.linalg.solve(A + lam * reg_matrix, b)
        solutions.append(c)

        # Residual norm
        residual = np.dot(A, c) - b
        errors.append(np.linalg.norm(residual))

        # Solution norm
        norms.append(np.sqrt(np.dot(c, np.dot(reg_matrix, c))))

    errors = np.array(errors)
    norms = np.array(norms)

    # L-curve: find maximum curvature
    log_errors = np.log(errors + 1e-16)
    log_norms = np.log(norms + 1e-16)

    # Compute curvature using finite differences
    curvature = np.zeros(len(lambda_range) - 2)
    for i in range(1, len(lambda_range) - 1):
        dx1 = log_errors[i] - log_errors[i-1]
        dy1 = log_norms[i] - log_norms[i-1]
        dx2 = log_errors[i+1] - log_errors[i]
        dy2 = log_norms[i+1] - log_norms[i]

        # Cross product gives signed curvature
        cross = dx1 * dy2 - dx2 * dy1
        denom = (dx1**2 + dy1**2) * (dx2**2 + dy2**2)
        if denom > 1e-20:
            curvature[i-1] = cross / np.sqrt(denom)

    # Find maximum curvature
    if len(curvature) > 0:
        opt_idx = np.argmax(curvature) + 1
    else:
        opt_idx = len(lambda_range) // 2

    lambda_opt = lambda_range[opt_idx]
    c_opt = solutions[opt_idx]

    l_curve_data = {
        'lambda_range': lambda_range,
        'errors': errors,
        'norms': norms,
        'curvature': curvature,
        'lambda_opt': lambda_opt,
        'opt_idx': opt_idx,
    }

    return c_opt, lambda_opt, l_curve_data


# ==============================================================================
# Main training function
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='MVP-2.0: Fei Lu method')
    parser.add_argument('--N', type=int, default=100, help='Number of particles')
    parser.add_argument('--L', type=int, default=50, help='Number of time snapshots')
    parser.add_argument('--M_samples', type=int, default=50, help='Number of trajectory samples')
    parser.add_argument('--M_grid', type=int, default=200, help='Spatial grid size')
    parser.add_argument('--T', type=float, default=1.0, help='Total time')
    parser.add_argument('--dt', type=float, default=0.01, help='Simulation time step')
    parser.add_argument('--nu', type=float, default=0.1, help='Viscosity (sigma^2/2)')
    parser.add_argument('--n_basis', type=int, default=15, help='Number of B-spline basis functions')
    parser.add_argument('--spline_degree', type=int, default=2, help='B-spline degree')
    parser.add_argument('--bandwidth', type=float, default=0.3, help='KDE bandwidth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default='results/mvp2_0')
    parser.add_argument('--img_dir', type=str, default='experiments/ips_unlabeled/img')
    # Phi parameters
    parser.add_argument('--Phi_A', type=float, default=1.0)
    parser.add_argument('--Phi_sigma', type=float, default=1.0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    sigma = np.sqrt(2 * args.nu)  # SDE noise from viscosity

    print("=" * 70)
    print("MVP-2.0: Fei Lu Method for Learning Interaction Kernels")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Particles N = {args.N}")
    print(f"  Time snapshots L = {args.L}")
    print(f"  Trajectory samples M = {args.M_samples}")
    print(f"  Spatial grid M_grid = {args.M_grid}")
    print(f"  Viscosity nu = {args.nu} (sigma = {sigma:.4f})")
    print(f"  B-spline: n_basis={args.n_basis}, degree={args.spline_degree}")
    print(f"\nTrue Phi: Gaussian(A={args.Phi_A}, sigma={args.Phi_sigma})")
    print(f"Note: V = 0 (no external potential, following Fei Lu's setup)")

    # =========================================================================
    # Generate data (V=0, only interaction)
    # =========================================================================
    print("\n[1] Generating data...")
    start_time = time.time()

    # Create simulator with V=0
    from core.potentials import ZeroPotential
    V = ZeroPotential()
    Phi = GaussianInteraction(A=args.Phi_A, sigma=args.Phi_sigma)

    simulator = SDESimulator(V=V, Phi=Phi, sigma=sigma, dt=args.dt)

    data, t_snapshots = simulator.simulate(
        N=args.N, d=1, T=args.T, L=args.L, M=args.M_samples, seed=args.seed
    )
    print(f"  Data shape: {data.shape}")
    print(f"  Time range: [{t_snapshots[0]:.3f}, {t_snapshots[-1]:.3f}]")
    print(f"  Time: {time.time() - start_time:.1f}s")

    # =========================================================================
    # Estimate density u(x,t) using KDE
    # =========================================================================
    print("\n[2] Estimating density u(x,t) using KDE...")
    start_time = time.time()

    # Spatial grid
    x_min = data[:, :, :, 0].min() - 1
    x_max = data[:, :, :, 0].max() + 1
    x_grid = np.linspace(x_min, x_max, args.M_grid)
    dx = x_grid[1] - x_grid[0]

    u_time = estimate_density_time_series(data, x_grid, bandwidth=args.bandwidth)
    print(f"  Density shape: {u_time.shape}")
    print(f"  x range: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Time: {time.time() - start_time:.1f}s")

    # =========================================================================
    # Create B-spline basis
    # =========================================================================
    print("\n[3] Creating B-spline basis...")
    r_max = x_max - x_min
    basis_funcs, knots = create_bspline_basis(
        args.n_basis, args.spline_degree, 0, r_max
    )
    print(f"  Number of basis functions: {len(basis_funcs)}")
    print(f"  Knot vector length: {len(knots)}")

    # =========================================================================
    # Compute normal matrix and vector
    # =========================================================================
    print("\n[4] Computing normal matrix A and vector b...")
    print("  (This may take a while due to double convolution loop...)")
    start_time = time.time()

    A, b = compute_normal_matrix_and_vector(
        u_time, t_snapshots, x_grid, basis_funcs, args.nu
    )
    print(f"  A shape: {A.shape}, condition number: {np.linalg.cond(A):.2e}")
    print(f"  b shape: {b.shape}, norm: {np.linalg.norm(b):.4e}")
    print(f"  Time: {time.time() - start_time:.1f}s")

    # =========================================================================
    # Solve with Tikhonov regularization
    # =========================================================================
    print("\n[5] Solving with Tikhonov regularization + L-curve...")
    start_time = time.time()

    c_opt, lambda_opt, l_curve_data = solve_with_tikhonov(A, b)
    print(f"  Optimal lambda: {lambda_opt:.4e}")
    print(f"  Coefficients: {c_opt}")
    print(f"  Time: {time.time() - start_time:.1f}s")

    # =========================================================================
    # Evaluate learned phi and compare to truth
    # =========================================================================
    print("\n[6] Evaluating learned phi...")

    r_eval = np.linspace(0, r_max * 0.5, 200)

    # Evaluate learned phi
    phi_learned = np.zeros_like(r_eval)
    for i, phi_i in enumerate(basis_funcs):
        vals = phi_i(r_eval)
        vals = np.nan_to_num(vals, nan=0.0)
        phi_learned += c_opt[i] * vals

    # True phi (Gaussian interaction kernel)
    # Phi(r) = A * exp(-r^2 / (2*sigma^2))
    # phi(r) = dPhi/dr = -A * r / sigma^2 * exp(-r^2 / (2*sigma^2))
    phi_true = -args.Phi_A * r_eval / (args.Phi_sigma**2) * np.exp(-r_eval**2 / (2 * args.Phi_sigma**2))

    # Compute error (L2 relative)
    # Center both (remove constant offset)
    phi_learned_centered = phi_learned - np.mean(phi_learned)
    phi_true_centered = phi_true - np.mean(phi_true)

    l2_diff = np.sqrt(np.mean((phi_learned_centered - phi_true_centered)**2))
    l2_true = np.sqrt(np.mean(phi_true_centered**2))
    rel_error = l2_diff / l2_true if l2_true > 1e-10 else float('inf')

    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  phi relative L2 error: {rel_error:.4f} ({rel_error*100:.2f}%)")
    print(f"{'='*70}")

    passed = rel_error < 0.10
    print(f"\nValidation: {'PASS' if passed else 'FAIL'} (threshold: 10%)")

    # =========================================================================
    # Plotting
    # =========================================================================
    print("\n[7] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Learned vs True phi
    ax = axes[0, 0]
    ax.plot(r_eval, phi_true, 'r-', lw=2, label='True φ(r)')
    ax.plot(r_eval, phi_learned, 'b--', lw=2, label='Learned φ(r)')
    ax.set_xlabel('r')
    ax.set_ylabel('φ(r)')
    ax.set_title(f'Interaction Kernel (rel. error: {rel_error:.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Centered comparison
    ax = axes[0, 1]
    ax.plot(r_eval, phi_true_centered, 'r-', lw=2, label='True (centered)')
    ax.plot(r_eval, phi_learned_centered, 'b--', lw=2, label='Learned (centered)')
    ax.set_xlabel('r')
    ax.set_ylabel('φ(r) - mean')
    ax.set_title('Centered Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: L-curve
    ax = axes[1, 0]
    ax.loglog(l_curve_data['errors'], l_curve_data['norms'], 'b.-')
    ax.loglog(l_curve_data['errors'][l_curve_data['opt_idx']],
              l_curve_data['norms'][l_curve_data['opt_idx']],
              'ro', markersize=10, label=f'λ={lambda_opt:.2e}')
    ax.set_xlabel('Residual norm')
    ax.set_ylabel('Solution norm')
    ax.set_title('L-curve for Regularization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Density estimate at different times
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    t_indices = np.linspace(0, args.L - 1, 5, dtype=int)
    for idx, t_idx in enumerate(t_indices):
        ax.plot(x_grid, u_time[t_idx], color=colors[idx],
                label=f't={t_snapshots[t_idx]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Estimated Density at Different Times')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.img_dir, 'mvp2_0_fei_lu_results.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Save results
    # =========================================================================
    metrics = {
        'config': vars(args),
        'phi_l2_error_rel': float(rel_error),
        'lambda_opt': float(lambda_opt),
        'A_condition_number': float(np.linalg.cond(A)),
        'validation_passed': bool(passed),
    }
    with open(os.path.join(args.results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    np.savez(os.path.join(args.results_dir, 'results.npz'),
             c_opt=c_opt,
             r_eval=r_eval,
             phi_learned=phi_learned,
             phi_true=phi_true,
             A=A,
             b=b,
             l_curve_data=l_curve_data)

    print(f"\n{'='*70}")
    print(f"MVP-2.0 COMPLETE: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*70}")

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
