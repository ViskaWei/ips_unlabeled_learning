"""Validation utilities for SDE simulation."""

import numpy as np
from scipy import stats
from typing import Dict, Tuple


def compute_ou_theoretical_stats(t: float, sigma: float, x0_var: float = 1.0) -> Tuple[float, float]:
    """Compute theoretical mean and variance for OU process.

    For dX = -X dt + sigma dW with X_0 ~ N(0, x0_var):
    E[X_t] = 0 (since E[X_0] = 0)
    Var[X_t] = x0_var * exp(-2t) + (sigma^2/2) * (1 - exp(-2t))

    Stationary variance = sigma^2 / 2
    """
    mean = 0.0
    var = x0_var * np.exp(-2 * t) + (sigma**2 / 2) * (1 - np.exp(-2 * t))
    return mean, var


def compute_kl_divergence_gaussian(mu1: float, var1: float, mu2: float, var2: float) -> float:
    """Compute KL divergence between two Gaussians KL(N(mu1,var1) || N(mu2,var2))."""
    if var1 <= 0 or var2 <= 0:
        return float('inf')
    return 0.5 * (np.log(var2 / var1) + var1 / var2 + (mu1 - mu2)**2 / var2 - 1)


def validate_ou_simulation(
    data: np.ndarray,
    t_snapshots: np.ndarray,
    sigma: float,
    x0_var: float = 1.0,
) -> Dict:
    """Validate OU process simulation against theoretical values.

    Args:
        data: Simulation data, shape (M, L, N, d)
        t_snapshots: Time points, shape (L,)
        sigma: Noise strength
        x0_var: Initial variance

    Returns:
        Dictionary with validation metrics
    """
    M, L, N, d = data.shape

    results = {
        'time': [],
        'empirical_mean': [],
        'empirical_var': [],
        'theoretical_mean': [],
        'theoretical_var': [],
        'mean_error': [],
        'var_error': [],
        'kl_divergence': [],
    }

    for ell, t in enumerate(t_snapshots):
        # Flatten all particles at this time: shape (M * N * d,)
        samples = data[:, ell, :, :].flatten()

        emp_mean = np.mean(samples)
        emp_var = np.var(samples)

        theo_mean, theo_var = compute_ou_theoretical_stats(t, sigma, x0_var)

        mean_error = np.abs(emp_mean - theo_mean)
        var_error = np.abs(emp_var - theo_var) / theo_var if theo_var > 0 else float('inf')

        kl = compute_kl_divergence_gaussian(emp_mean, emp_var, theo_mean, theo_var)

        results['time'].append(t)
        results['empirical_mean'].append(emp_mean)
        results['empirical_var'].append(emp_var)
        results['theoretical_mean'].append(theo_mean)
        results['theoretical_var'].append(theo_var)
        results['mean_error'].append(mean_error)
        results['var_error'].append(var_error)
        results['kl_divergence'].append(kl)

    # Summary statistics (at final time)
    results['final_kl'] = results['kl_divergence'][-1]
    results['final_mean_error'] = results['mean_error'][-1]
    results['final_var_error'] = results['var_error'][-1]
    results['stationary_var_theory'] = sigma**2 / 2
    results['stationary_var_empirical'] = results['empirical_var'][-1]

    return results


def print_validation_summary(results: Dict) -> None:
    """Print validation summary."""
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    print(f"Final time: {results['time'][-1]:.2f}")
    print(f"Theoretical stationary variance: {results['stationary_var_theory']:.6f}")
    print(f"Empirical stationary variance:   {results['stationary_var_empirical']:.6f}")
    print(f"Final KL divergence:             {results['final_kl']:.6f}")
    print(f"Final mean error:                {results['final_mean_error']:.6f}")
    print(f"Final variance error (relative): {results['final_var_error']:.4f} ({results['final_var_error']*100:.2f}%)")
    print("="*60)

    # Pass/fail
    kl_threshold = 0.05
    var_threshold = 0.1  # 10% relative error

    kl_pass = results['final_kl'] < kl_threshold
    var_pass = results['final_var_error'] < var_threshold

    print(f"\nValidation Results:")
    print(f"  KL divergence < {kl_threshold}: {'PASS' if kl_pass else 'FAIL'}")
    print(f"  Variance error < {var_threshold*100:.0f}%: {'PASS' if var_pass else 'FAIL'}")
    print(f"  Overall: {'PASS' if kl_pass and var_pass else 'FAIL'}")
    print("="*60 + "\n")

    return kl_pass and var_pass
