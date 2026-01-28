#!/usr/bin/env python
"""Generate SDE simulation data for MVP-0.0 baseline validation."""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.sde_simulator import simulate_ou_process
from core.validation import validate_ou_simulation, print_validation_summary


def parse_args():
    parser = argparse.ArgumentParser(description='Generate OU process data')
    parser.add_argument('--N', type=int, default=10, help='Number of particles')
    parser.add_argument('--d', type=int, default=1, help='Spatial dimension')
    parser.add_argument('--L', type=int, default=20, help='Number of snapshots')
    parser.add_argument('--M', type=int, default=200, help='Number of samples')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    parser.add_argument('--T', type=float, default=2.0, help='Total time')
    parser.add_argument('--sigma', type=float, default=0.1, help='Noise strength')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='data/ips_baseline.npz', help='Output file')
    parser.add_argument('--results_dir', type=str, default='results/mvp0_0', help='Results directory')
    parser.add_argument('--img_dir', type=str, default='experiments/ips_unlabeled/img', help='Image directory')
    return parser.parse_args()


def plot_distribution_evolution(data, t_snapshots, sigma, save_path):
    """Plot distribution evolution over time."""
    M, L, N, d = data.shape

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    # Select time indices to plot
    indices = [0, L//4, L//2, 3*L//4, L-1]
    if len(indices) > 6:
        indices = indices[:6]

    # Theoretical stationary distribution
    x_range = np.linspace(-0.5, 0.5, 200)
    stationary_var = sigma**2 / 2
    stationary_pdf = np.exp(-x_range**2 / (2 * stationary_var)) / np.sqrt(2 * np.pi * stationary_var)

    for ax_idx, t_idx in enumerate(indices):
        if ax_idx >= len(axes):
            break

        ax = axes[ax_idx]
        t = t_snapshots[t_idx]

        # Flatten samples at this time
        samples = data[:, t_idx, :, :].flatten()

        # Histogram
        ax.hist(samples, bins=50, density=True, alpha=0.7, color='steelblue', label='Simulation')

        # Theoretical distribution at this time
        # Var(X_t) = exp(-2t) + (sigma^2/2)(1 - exp(-2t))
        theo_var = np.exp(-2*t) + (sigma**2/2) * (1 - np.exp(-2*t))
        theo_pdf = np.exp(-x_range**2 / (2 * theo_var)) / np.sqrt(2 * np.pi * theo_var)
        ax.plot(x_range, theo_pdf, 'r-', lw=2, label=f'Theory (var={theo_var:.4f})')

        # Stationary distribution
        ax.plot(x_range, stationary_pdf, 'g--', lw=1.5, alpha=0.7, label=f'Stationary (var={stationary_var:.4f})')

        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.set_title(f't = {t:.2f}')
        ax.legend(fontsize=8)
        ax.set_xlim(-0.5, 0.5)

    # Use last subplot for empty or remove
    if len(indices) < 6:
        axes[-1].axis('off')

    plt.suptitle(f'Distribution Evolution (N={N}, M={M}, sigma={sigma})', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution evolution plot to {save_path}")


def plot_mean_var_dynamics(results, save_path):
    """Plot mean and variance dynamics vs theory."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Mean dynamics
    ax = axes[0]
    ax.plot(results['time'], results['empirical_mean'], 'b-o', markersize=4, label='Empirical')
    ax.plot(results['time'], results['theoretical_mean'], 'r--', lw=2, label='Theory (= 0)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean')
    ax.set_title('Mean Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Variance dynamics
    ax = axes[1]
    ax.plot(results['time'], results['empirical_var'], 'b-o', markersize=4, label='Empirical')
    ax.plot(results['time'], results['theoretical_var'], 'r--', lw=2, label='Theory')
    ax.axhline(y=results['stationary_var_theory'], color='g', linestyle=':', lw=2,
               label=f'Stationary = {results["stationary_var_theory"]:.4f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Variance')
    ax.set_title('Variance Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved mean/variance dynamics plot to {save_path}")


def plot_kl_divergence(results, save_path):
    """Plot KL divergence over time."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(results['time'], results['kl_divergence'], 'b-o', markersize=4)
    ax.axhline(y=0.05, color='r', linestyle='--', lw=2, label='Threshold (0.05)')
    ax.set_xlabel('Time')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence from Theoretical Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved KL divergence plot to {save_path}")


def main():
    args = parse_args()

    # Create output directories
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    print("="*60)
    print("MVP-0.0: SDE Data Generation + Baseline Validation")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  N (particles):  {args.N}")
    print(f"  d (dimension):  {args.d}")
    print(f"  L (snapshots):  {args.L}")
    print(f"  M (samples):    {args.M}")
    print(f"  dt (timestep):  {args.dt}")
    print(f"  T (total time): {args.T}")
    print(f"  sigma (noise):  {args.sigma}")
    print(f"  seed:           {args.seed}")
    print()

    # Run simulation
    print("Running SDE simulation...")
    data, t_snapshots, config = simulate_ou_process(
        N=args.N,
        d=args.d,
        L=args.L,
        M=args.M,
        dt=args.dt,
        T=args.T,
        sigma=args.sigma,
        seed=args.seed,
    )
    print(f"Generated data shape: {data.shape}")
    print(f"Time snapshots: {len(t_snapshots)} points from {t_snapshots[0]:.3f} to {t_snapshots[-1]:.3f}")

    # Save data
    np.savez(
        args.output,
        data=data,
        t_snapshots=t_snapshots,
        config=config,
    )
    print(f"\nSaved data to {args.output}")

    # Validate
    print("\nValidating against OU theory...")
    results = validate_ou_simulation(data, t_snapshots, args.sigma)
    passed = print_validation_summary(results)

    # Generate plots
    print("\nGenerating plots...")
    plot_distribution_evolution(
        data, t_snapshots, args.sigma,
        os.path.join(args.img_dir, 'distribution_evolution.png')
    )
    plot_mean_var_dynamics(
        results,
        os.path.join(args.img_dir, 'mean_var_dynamics.png')
    )
    plot_kl_divergence(
        results,
        os.path.join(args.img_dir, 'kl_divergence.png')
    )

    # Save metrics
    metrics = {
        'config': config,
        'final_kl_divergence': float(results['final_kl']),
        'final_mean_error': float(results['final_mean_error']),
        'final_var_error_relative': float(results['final_var_error']),
        'stationary_var_theory': float(results['stationary_var_theory']),
        'stationary_var_empirical': float(results['stationary_var_empirical']),
        'validation_passed': bool(passed),
        'time_series': {
            'time': [float(x) for x in results['time']],
            'empirical_mean': [float(x) for x in results['empirical_mean']],
            'empirical_var': [float(x) for x in results['empirical_var']],
            'theoretical_var': [float(x) for x in results['theoretical_var']],
            'kl_divergence': [float(x) for x in results['kl_divergence']],
        }
    }

    metrics_path = os.path.join(args.results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    print("\n" + "="*60)
    print(f"MVP-0.0 COMPLETE: {'PASS' if passed else 'FAIL'}")
    print("="*60)

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
