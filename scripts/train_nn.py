#!/usr/bin/env python
"""Train neural networks to learn potentials using trajectory-free loss.

MVP-1.0: 1D simple system verification.

Supports:
  - Command line args: python train_nn.py --N 10 --epochs 100
  - Config file: python train_nn.py --config configs/mvp1_0.yaml
  - Experiment ID: python train_nn.py --config configs/mvp1_0.yaml --exp-id EXP-20260128-001
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.nn_models import PotentialNetworks
from core.trajectory_free_loss import TrajectoryFreeLoss
from core.true_potentials import evaluate_V_error, evaluate_Phi_error
from core.sde_simulator import SDESimulator
from core.potentials import HarmonicPotential, GaussianInteraction


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Train NN with trajectory-free loss')

    # Config file support (overrides other args)
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--exp-id', type=str, default=None, help='Experiment ID (auto-generated if not provided)')

    # Data parameters
    parser.add_argument('--N', type=int, default=10, help='Number of particles')
    parser.add_argument('--d', type=int, default=1, help='Spatial dimension')
    parser.add_argument('--L', type=int, default=20, help='Number of snapshots')
    parser.add_argument('--M', type=int, default=200, help='Number of samples')
    parser.add_argument('--dt', type=float, default=0.01, help='SDE time step')
    parser.add_argument('--T', type=float, default=2.0, help='Total simulation time')
    parser.add_argument('--sigma', type=float, default=0.1, help='Noise strength')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Model parameters
    parser.add_argument('--hidden_dims', type=str, default='64,64', help='Hidden dimensions')

    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--log_interval', type=int, default=100, help='Log every N epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')

    # Output paths
    parser.add_argument('--results_dir', type=str, default='results/mvp1_0', help='Results directory')
    parser.add_argument('--img_dir', type=str, default='experiments/ips_unlabeled/img', help='Image directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')

    # True potential parameters
    parser.add_argument('--V_k', type=float, default=1.0, help='Harmonic V constant')
    parser.add_argument('--Phi_A', type=float, default=1.0, help='Gaussian Phi amplitude')
    parser.add_argument('--Phi_sigma', type=float, default=1.0, help='Gaussian Phi width')

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)

        # Flatten nested config and override args
        def flatten_config(cfg, prefix=''):
            flat = {}
            for k, v in cfg.items():
                if isinstance(v, dict):
                    flat.update(flatten_config(v, prefix=k + '_'))
                else:
                    flat[prefix + k] = v
            return flat

        flat_config = flatten_config(config)

        # Map config keys to arg names
        key_mapping = {
            'data_N': 'N', 'data_d': 'd', 'data_L': 'L', 'data_M': 'M',
            'data_dt': 'dt', 'data_T': 'T', 'data_sigma': 'sigma',
            'training_epochs': 'epochs', 'training_lr': 'lr',
            'training_seed': 'seed', 'training_patience': 'patience',
            'model_hidden_dims': 'hidden_dims',
            'output_dir': 'results_dir', 'output_img_dir': 'img_dir',
            'potential_V_k': 'V_k', 'potential_Phi_A': 'Phi_A', 'potential_Phi_sigma': 'Phi_sigma',
        }

        for config_key, value in flat_config.items():
            arg_name = key_mapping.get(config_key, config_key)
            if hasattr(args, arg_name):
                setattr(args, arg_name, value)

        # Also check direct keys
        for k, v in config.items():
            if not isinstance(v, dict) and hasattr(args, k):
                setattr(args, k, v)

    # Auto-generate exp_id if not provided
    if args.exp_id is None:
        from datetime import datetime
        args.exp_id = f"EXP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Update results_dir with exp_id
    if args.exp_id and 'EXP-' in args.exp_id:
        args.results_dir = f"results/{args.exp_id}"

    return args


def generate_data_with_interaction(
    N: int, d: int, L: int, M: int, dt: float, T: float, sigma: float, seed: int,
    V_k: float = 1.0, Phi_A: float = 1.0, Phi_sigma: float = 1.0,
):
    """Generate SDE data with interaction potential."""
    print("Generating data with interaction potential...")

    V = HarmonicPotential(k=V_k)
    Phi = GaussianInteraction(A=Phi_A, sigma=Phi_sigma)

    simulator = SDESimulator(V=V, Phi=Phi, sigma=sigma, dt=dt)
    data, t_snapshots = simulator.simulate(N=N, d=d, T=T, L=L, M=M, seed=seed)

    config = {
        'N': N, 'd': d, 'L': L, 'M': M, 'dt': dt, 'T': T, 'sigma': sigma, 'seed': seed,
        'V': f'Harmonic(k={V_k})',
        'Phi': f'Gaussian(A={Phi_A}, sigma={Phi_sigma})',
    }

    print(f"Generated data shape: {data.shape}")
    return data, t_snapshots, config


def train(
    data: np.ndarray,
    t_snapshots: np.ndarray,
    networks: PotentialNetworks,
    loss_fn: TrajectoryFreeLoss,
    optimizer: optim.Optimizer,
    epochs: int,
    log_interval: int,
    device: str,
    patience: int = 50,
    min_delta: float = 1e-6,
) -> list:
    """Train networks using trajectory-free loss with early stopping."""
    # Convert to torch
    data_torch = torch.tensor(data, dtype=torch.float32, device=device)
    t_torch = torch.tensor(t_snapshots, dtype=torch.float32, device=device)

    history = []
    networks.train()

    print("\nTraining...")
    print("-" * 70)

    start_time = time.time()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss, info = loss_fn(data_torch, t_torch, networks)

        loss.backward()
        optimizer.step()

        history.append(info)

        # Early stopping check
        current_loss = info['loss']
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in networks.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            elapsed = time.time() - start_time
            residual = info.get('residual', info['loss'])
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {info['loss']:.6f} | "
                  f"Res: {residual:.6f} | J_diss: {info['J_diss']:.6f} | "
                  f"J_E: {info['J_energy_change']:.6f} | Time: {elapsed:.1f}s", flush=True)

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print("-" * 70)
    print(f"Training complete in {time.time() - start_time:.1f}s")

    # Restore best model
    if best_state is not None:
        networks.load_state_dict(best_state)
        print(f"Restored best model (loss: {best_loss:.6f})")

    return history


def plot_training_history(history: list, save_path: str):
    """Plot training loss history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = list(range(1, len(history) + 1))
    losses = [h['loss'] for h in history]
    j_diss = [h['J_diss'] for h in history]
    j_diff = [h['J_diff'] for h in history]
    j_energy = [h['J_energy_change'] for h in history]

    # Total loss
    ax = axes[0]
    ax.semilogy(epochs, np.abs(losses), 'b-', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|Loss|')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    # Components
    ax = axes[1]
    ax.plot(epochs, j_diss, 'r-', label='J_diss', alpha=0.7)
    ax.plot(epochs, j_diff, 'g-', label='J_diff', alpha=0.7)
    ax.plot(epochs, j_energy, 'b-', label='J_energy', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot to {save_path}")


def plot_potential_comparison(
    networks: PotentialNetworks,
    V_k: float, Phi_A: float, Phi_sigma: float,
    device: str, save_path: str,
):
    """Plot learned vs true potentials."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # V comparison
    x_range = np.linspace(-2, 2, 200)
    V_result = evaluate_V_error(networks, x_range, k=V_k, device=device)

    ax = axes[0]
    ax.plot(x_range, V_result['V_true_centered'], 'r-', lw=2, label='True V')
    ax.plot(x_range, V_result['V_pred_centered'], 'b--', lw=2, label='Learned V')
    ax.set_xlabel('x')
    ax.set_ylabel('V(x) - mean')
    ax.set_title(f'Kinetic Potential V (rel. error: {V_result["l2_error_rel"]:.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phi comparison
    r_range = np.linspace(0, 3, 200)
    Phi_result = evaluate_Phi_error(networks, r_range, A=Phi_A, sigma=Phi_sigma, device=device)

    ax = axes[1]
    ax.plot(r_range, Phi_result['Phi_true_centered'], 'r-', lw=2, label='True Phi')
    ax.plot(r_range, Phi_result['Phi_pred_centered'], 'b--', lw=2, label='Learned Phi')
    ax.set_xlabel('r')
    ax.set_ylabel('Phi(r) - mean')
    ax.set_title(f'Interaction Potential Phi (rel. error: {Phi_result["l2_error_rel"]:.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved potential comparison plot to {save_path}")

    return V_result, Phi_result


def main():
    args = parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    print("=" * 70)
    print(f"Experiment: {args.exp_id}")
    print("MVP-1.0: Trajectory-Free Loss Verification")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  exp_id: {args.exp_id}")
    print(f"  N={args.N}, d={args.d}, L={args.L}, M={args.M}")
    print(f"  sigma={args.sigma}, dt={args.dt}, T={args.T}")
    print(f"  V: Harmonic(k={args.V_k})")
    print(f"  Phi: Gaussian(A={args.Phi_A}, sigma={args.Phi_sigma})")
    print(f"  Network: MLP with hidden_dims={args.hidden_dims}")
    print(f"  Training: lr={args.lr}, epochs={args.epochs}")
    print(f"  Results: {args.results_dir}")
    print()

    # Generate data
    data, t_snapshots, data_config = generate_data_with_interaction(
        N=args.N, d=args.d, L=args.L, M=args.M,
        dt=args.dt, T=args.T, sigma=args.sigma, seed=args.seed,
        V_k=args.V_k, Phi_A=args.Phi_A, Phi_sigma=args.Phi_sigma,
    )

    # Initialize networks
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    networks = PotentialNetworks(d=args.d, hidden_dims=hidden_dims).to(args.device)
    print(f"\nNetwork parameters: {sum(p.numel() for p in networks.parameters())}")

    # Loss and optimizer
    loss_fn = TrajectoryFreeLoss(sigma=args.sigma, d=args.d)
    optimizer = optim.Adam(networks.parameters(), lr=args.lr)

    # Train
    history = train(
        data, t_snapshots, networks, loss_fn, optimizer,
        epochs=args.epochs, log_interval=args.log_interval, device=args.device,
        patience=args.patience,
    )

    # Evaluate
    print("\nEvaluating...")
    x_range = np.linspace(-2, 2, 200)
    r_range = np.linspace(0, 3, 200)

    V_result = evaluate_V_error(networks, x_range, k=args.V_k, device=args.device)
    Phi_result = evaluate_Phi_error(networks, r_range, A=args.Phi_A, sigma=args.Phi_sigma, device=args.device)

    print(f"\n{'='*70}")
    print("Evaluation Results")
    print(f"{'='*70}")
    print(f"V relative L2 error:   {V_result['l2_error_rel']:.4f} ({V_result['l2_error_rel']*100:.2f}%)")
    print(f"Phi relative L2 error: {Phi_result['l2_error_rel']:.4f} ({Phi_result['l2_error_rel']*100:.2f}%)")
    print(f"{'='*70}")

    # Check pass/fail
    threshold = 0.10  # 10%
    V_pass = V_result['l2_error_rel'] < threshold
    Phi_pass = Phi_result['l2_error_rel'] < threshold
    overall_pass = V_pass and Phi_pass

    print(f"\nValidation (threshold < {threshold*100:.0f}%):")
    print(f"  V error < {threshold*100:.0f}%:   {'PASS' if V_pass else 'FAIL'}")
    print(f"  Phi error < {threshold*100:.0f}%: {'PASS' if Phi_pass else 'FAIL'}")
    print(f"  Overall:        {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'='*70}")

    # Save plots
    plot_training_history(history, os.path.join(args.img_dir, 'mvp1_0_training_history.png'))
    plot_potential_comparison(
        networks, args.V_k, args.Phi_A, args.Phi_sigma,
        args.device, os.path.join(args.img_dir, 'mvp1_0_potential_comparison.png'),
    )

    # Save metrics
    metrics = {
        'exp_id': args.exp_id,
        'config': {
            'N': args.N, 'd': args.d, 'L': args.L, 'M': args.M,
            'sigma': args.sigma, 'dt': args.dt, 'T': args.T,
            'V_k': args.V_k, 'Phi_A': args.Phi_A, 'Phi_sigma': args.Phi_sigma,
            'hidden_dims': hidden_dims, 'lr': args.lr, 'epochs': args.epochs,
        },
        'V_l2_error_rel': float(V_result['l2_error_rel']),
        'Phi_l2_error_rel': float(Phi_result['l2_error_rel']),
        'final_loss': float(history[-1]['loss']),
        'validation_passed': bool(overall_pass),
        'training_epochs_actual': len(history),
        'training_history': {
            'loss': [h['loss'] for h in history[::10]],  # subsample
            'J_diss': [h['J_diss'] for h in history[::10]],
            'J_diff': [h['J_diff'] for h in history[::10]],
            'J_energy_change': [h['J_energy_change'] for h in history[::10]],
        }
    }

    metrics_path = os.path.join(args.results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Create .done marker for agent to detect completion
    done_path = os.path.join(args.results_dir, '.done')
    with open(done_path, 'w') as f:
        f.write(f"completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"passed: {overall_pass}\n")
    print(f"Created completion marker: {done_path}")

    # Save model
    model_path = os.path.join(args.results_dir, 'model.pt')
    torch.save(networks.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    print(f"\n{'='*70}")
    print(f"MVP-1.0 COMPLETE: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'='*70}")

    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
