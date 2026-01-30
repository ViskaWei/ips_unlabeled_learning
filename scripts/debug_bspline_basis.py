#!/usr/bin/env python
"""Debug B-spline basis: check if it can represent the true phi."""

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


def main():
    print("=" * 70)
    print("Debug B-spline Basis")
    print("=" * 70)

    # True phi parameters
    A_phi = 1.0
    sigma_phi = 1.0

    # B-spline parameters
    n_basis = 8
    degree = 2
    r_min, r_max = 0, 8  # domain

    # Create basis
    basis_funcs, knots = create_bspline_basis(n_basis, degree, r_min, r_max)
    print(f"\nB-spline: n_basis={n_basis}, degree={degree}")
    print(f"Knots: {knots}")

    # Evaluation grid
    r_eval = np.linspace(r_min + 0.01, r_max - 0.01, 200)

    # Evaluate basis functions
    B = np.zeros((len(r_eval), n_basis))
    for i, phi in enumerate(basis_funcs):
        vals = phi(r_eval)
        vals = np.nan_to_num(vals, nan=0.0)
        B[:, i] = vals

    # Evaluate true phi
    phi_true = np.array([true_phi(r, A_phi, sigma_phi) for r in r_eval])

    # Try to fit true phi with B-splines
    print("\n[1] Fitting true φ with B-splines...")

    # Least squares: minimize ||B c - phi_true||²
    c_fit, residuals, rank, s = np.linalg.lstsq(B, phi_true, rcond=None)
    phi_fit = B @ c_fit

    fit_error = np.linalg.norm(phi_fit - phi_true) / np.linalg.norm(phi_true)
    print(f"  Fit coefficients: {c_fit}")
    print(f"  Fit relative error: {fit_error*100:.2f}%")
    print(f"  Matrix condition number: {np.linalg.cond(B):.2e}")

    # Check for NaN in basis
    print("\n[2] Checking basis functions...")
    for i, phi in enumerate(basis_funcs):
        vals = phi(r_eval)
        n_nan = np.sum(np.isnan(vals))
        n_zero = np.sum(np.abs(vals) < 1e-10)
        print(f"  Basis {i}: {n_nan} NaN, {n_zero} zeros, max={np.nanmax(np.abs(vals)):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Basis functions
    ax = axes[0]
    for i in range(n_basis):
        ax.plot(r_eval, B[:, i], label=f'B{i}')
    ax.set_xlabel('r')
    ax.set_ylabel('Basis value')
    ax.set_title('B-spline Basis Functions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: True vs Fit
    ax = axes[1]
    ax.plot(r_eval, phi_true, 'b-', lw=2, label='True φ')
    ax.plot(r_eval, phi_fit, 'r--', lw=2, label=f'B-spline fit (err={fit_error*100:.1f}%)')
    ax.set_xlabel('r')
    ax.set_ylabel('φ(r)')
    ax.set_title('True φ vs B-spline Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Fit residual
    ax = axes[2]
    ax.plot(r_eval, phi_fit - phi_true, 'k-', lw=1)
    ax.set_xlabel('r')
    ax.set_ylabel('Residual')
    ax.set_title('Fit Residual')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/ips_unlabeled/img/debug_bspline.png', dpi=150)
    plt.close()
    print(f"\nPlot saved to experiments/ips_unlabeled/img/debug_bspline.png")

    # Check if the issue is with antiderivative computation
    print("\n[3] Checking antiderivative Φ computation...")

    # True Φ
    def true_Phi(r, A=1.0, sigma=1.0):
        return A * np.exp(-r**2 / (2 * sigma**2))

    Phi_true = np.array([true_Phi(r, A_phi, sigma_phi) for r in r_eval])

    # B-spline antiderivatives
    Phi_basis = np.zeros((len(r_eval), n_basis))
    dr = r_eval[1] - r_eval[0]
    for i in range(n_basis):
        # Cumulative integral
        Phi_basis[:, i] = integrate.cumulative_trapezoid(B[:, i], r_eval, initial=0)

    # Fit Φ with B-spline antiderivatives
    Phi_fit = Phi_basis @ c_fit
    Phi_error = np.linalg.norm(Phi_fit - Phi_true) / np.linalg.norm(Phi_true)
    print(f"  Φ fit relative error: {Phi_error*100:.2f}%")

    if fit_error < 0.1:
        print(f"\n✅ B-splines can represent true φ with {fit_error*100:.1f}% error")
    else:
        print(f"\n❌ B-splines cannot represent true φ well (error={fit_error*100:.1f}%)")
        print(f"   Try: more basis functions, higher degree, or different domain")


if __name__ == '__main__':
    main()
