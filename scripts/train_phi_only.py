#!/usr/bin/env python
"""Train only Phi (interaction potential) with known V.

This addresses the identifiability problem by fixing V to the true harmonic potential.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.nn_models import SymmetricMLP, compute_pairwise_distances, compute_pairwise_diff
from core.true_potentials import evaluate_Phi_error, true_Phi_gaussian
from core.sde_simulator import SDESimulator
from core.potentials import HarmonicPotential, GaussianInteraction


class PhiOnlyLoss(nn.Module):
    """Trajectory-free loss with known V, only learning Phi."""

    def __init__(self, V_k: float = 1.0, sigma: float = 0.1, d: int = 1):
        super().__init__()
        self.V_k = V_k
        self.sigma = sigma
        self.sigma_sq = sigma ** 2
        self.d = d

    def true_grad_V(self, x: torch.Tensor) -> torch.Tensor:
        """True gradient of V(x) = 0.5 * k * x^2."""
        return self.V_k * x

    def true_laplacian_V(self, x: torch.Tensor) -> torch.Tensor:
        """True Laplacian of V = k * d."""
        return torch.full((x.shape[0],), self.V_k * self.d, device=x.device)

    def compute_drift(self, X: torch.Tensor, Phi_net: nn.Module) -> torch.Tensor:
        """Compute drift with known grad_V and learned grad_Phi."""
        N, d = X.shape

        # Known gradient of V
        grad_V = self.true_grad_V(X)  # shape (N, d)

        # Learned interaction gradient
        diff = compute_pairwise_diff(X)  # shape (N, N, d)
        distances = torch.norm(diff, dim=-1)  # shape (N, N)
        distances_safe = distances.clone()
        distances_safe[distances_safe < 1e-10] = 1e-10

        # dPhi/dr
        r_flat = distances.flatten().requires_grad_(True)
        Phi_vals = Phi_net(r_flat.unsqueeze(-1)).squeeze(-1)
        dPhi_dr = torch.autograd.grad(Phi_vals.sum(), r_flat, create_graph=True)[0]
        dPhi_dr = dPhi_dr.reshape(N, N)

        # grad_Phi pairs
        unit_diff = diff / distances_safe.unsqueeze(-1)
        grad_Phi_pairs = dPhi_dr.unsqueeze(-1) * unit_diff

        # Zero diagonal
        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        grad_Phi_pairs[mask] = 0

        # Mean field
        grad_Phi_mean = grad_Phi_pairs.sum(dim=1) / N

        return -grad_V - grad_Phi_mean

    def compute_laplacian_sum(self, X: torch.Tensor, Phi_net: nn.Module) -> torch.Tensor:
        """Compute Laplacian sum with known Laplacian_V and learned Laplacian_Phi."""
        N, d = X.shape

        # Known Laplacian of V
        laplacian_V = self.true_laplacian_V(X)  # shape (N,)

        # Learned Laplacian of Phi
        distances = compute_pairwise_distances(X)
        r_flat = distances.flatten().requires_grad_(True)

        # First derivative
        Phi_vals = Phi_net(r_flat.unsqueeze(-1)).squeeze(-1)
        grad1 = torch.autograd.grad(Phi_vals.sum(), r_flat, create_graph=True)[0]

        # Second derivative
        grad2 = torch.autograd.grad(grad1.sum(), r_flat, create_graph=True)[0]

        laplacian_Phi = grad2.reshape(N, N)

        # Zero diagonal
        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        laplacian_Phi[mask] = 0

        laplacian_Phi_mean = laplacian_Phi.sum(dim=1) / N

        return laplacian_V + laplacian_Phi_mean

    def compute_energy(self, X: torch.Tensor, Phi_net: nn.Module) -> torch.Tensor:
        """Compute energy with known V and learned Phi."""
        N = X.shape[0]

        # Known V
        V_vals = 0.5 * self.V_k * (X ** 2).sum(dim=-1)  # shape (N,)
        V_mean = V_vals.mean()

        # Learned Phi
        distances = compute_pairwise_distances(X)
        Phi_vals = Phi_net(distances.unsqueeze(-1)).squeeze(-1)

        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        Phi_vals[mask] = 0

        Phi_mean = Phi_vals.sum() / (N * N)

        return V_mean + Phi_mean

    def forward(self, data: torch.Tensor, t_snapshots: torch.Tensor, Phi_net: nn.Module):
        M, L, N, d = data.shape
        device = data.device

        total_diss = torch.tensor(0.0, device=device)
        total_diff = torch.tensor(0.0, device=device)
        total_energy_change = torch.tensor(0.0, device=device)

        for m in range(M):
            for ell in range(L - 1):
                X_curr = data[m, ell]
                X_next = data[m, ell + 1]
                dt = t_snapshots[ell + 1] - t_snapshots[ell]

                drift = self.compute_drift(X_curr, Phi_net)
                J_diss = (drift ** 2).sum() / N * dt

                laplacian_sum = self.compute_laplacian_sum(X_curr, Phi_net)
                J_diff = self.sigma_sq * laplacian_sum.mean() * dt

                E_curr = self.compute_energy(X_curr, Phi_net)
                E_next = self.compute_energy(X_next, Phi_net)
                J_energy_change = E_next - E_curr

                total_diss = total_diss + J_diss
                total_diff = total_diff + J_diff
                total_energy_change = total_energy_change + J_energy_change

        n_pairs = M * (L - 1)
        total_diss = total_diss / n_pairs
        total_diff = total_diff / n_pairs
        total_energy_change = total_energy_change / n_pairs

        residual = total_diss + total_diff - 2 * total_energy_change
        loss = residual ** 2

        info = {
            'loss': loss.item(),
            'residual': residual.item(),
            'J_diss': total_diss.item(),
            'J_diff': total_diff.item(),
            'J_energy_change': total_energy_change.item(),
        }

        return loss, info


def parse_args():
    parser = argparse.ArgumentParser(description='Train Phi only with known V')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--L', type=int, default=20)
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--T', type=float, default=2.0)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--results_dir', type=str, default='results/mvp1_0_phi_only')
    parser.add_argument('--img_dir', type=str, default='experiments/ips_unlabeled/img')
    parser.add_argument('--V_k', type=float, default=1.0)
    parser.add_argument('--Phi_A', type=float, default=1.0)
    parser.add_argument('--Phi_sigma', type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    print("=" * 70)
    print("MVP-1.0b: Learn Phi Only (V is known)")
    print("=" * 70)
    print(f"\nConfiguration: N={args.N}, L={args.L}, M={args.M}, sigma={args.sigma}")
    print(f"True V: Harmonic(k={args.V_k})")
    print(f"True Phi: Gaussian(A={args.Phi_A}, sigma={args.Phi_sigma})")

    # Generate data
    print("\nGenerating data...")
    V = HarmonicPotential(k=args.V_k)
    Phi = GaussianInteraction(A=args.Phi_A, sigma=args.Phi_sigma)
    simulator = SDESimulator(V=V, Phi=Phi, sigma=args.sigma, dt=args.dt)
    data, t_snapshots = simulator.simulate(
        N=args.N, d=args.d, T=args.T, L=args.L, M=args.M, seed=args.seed
    )
    print(f"Data shape: {data.shape}")

    # Initialize Phi network
    Phi_net = SymmetricMLP(input_dim=1, hidden_dims=[64, 64], output_dim=1)
    print(f"Phi network parameters: {sum(p.numel() for p in Phi_net.parameters())}")

    # Loss and optimizer
    loss_fn = PhiOnlyLoss(V_k=args.V_k, sigma=args.sigma, d=args.d)
    optimizer = optim.Adam(Phi_net.parameters(), lr=args.lr)

    # Convert to torch
    data_torch = torch.tensor(data, dtype=torch.float32)
    t_torch = torch.tensor(t_snapshots, dtype=torch.float32)

    # Training
    print("\nTraining...")
    print("-" * 70)
    history = []
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    start_time = time.time()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss, info = loss_fn(data_torch, t_torch, Phi_net)
        loss.backward()
        optimizer.step()
        history.append(info)

        if info['loss'] < best_loss - 1e-6:
            best_loss = info['loss']
            patience_counter = 0
            best_state = {k: v.clone() for k, v in Phi_net.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{args.epochs} | Loss: {info['loss']:.6f} | "
                  f"Res: {info['residual']:.6f} | J_diss: {info['J_diss']:.6f} | "
                  f"Time: {time.time() - start_time:.1f}s", flush=True)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("-" * 70)
    print(f"Training complete in {time.time() - start_time:.1f}s")

    if best_state:
        Phi_net.load_state_dict(best_state)

    # Evaluate
    print("\nEvaluating Phi...")
    r_range = np.linspace(0, 3, 200)

    Phi_net.eval()
    r_torch = torch.tensor(r_range, dtype=torch.float32)
    with torch.no_grad():
        Phi_pred = Phi_net(r_torch.unsqueeze(-1)).squeeze(-1).numpy()
    Phi_true = true_Phi_gaussian(r_range, A=args.Phi_A, sigma=args.Phi_sigma)

    # Center both
    Phi_pred_centered = Phi_pred - Phi_pred.mean()
    Phi_true_centered = Phi_true - Phi_true.mean()

    l2_diff = np.sqrt(np.mean((Phi_pred_centered - Phi_true_centered)**2))
    l2_true = np.sqrt(np.mean(Phi_true_centered**2))
    Phi_error = l2_diff / l2_true if l2_true > 1e-10 else float('inf')

    Phi_result = {
        'Phi_pred': Phi_pred,
        'Phi_true': Phi_true,
        'Phi_pred_centered': Phi_pred_centered,
        'Phi_true_centered': Phi_true_centered,
        'l2_error_rel': Phi_error,
    }

    print(f"\n{'='*70}")
    print(f"Phi relative L2 error: {Phi_result['l2_error_rel']:.4f} ({Phi_result['l2_error_rel']*100:.2f}%)")
    print(f"{'='*70}")

    passed = Phi_result['l2_error_rel'] < 0.10
    print(f"\nValidation: {'PASS' if passed else 'FAIL'}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_range, Phi_result['Phi_true_centered'], 'r-', lw=2, label='True Phi')
    ax.plot(r_range, Phi_result['Phi_pred_centered'], 'b--', lw=2, label='Learned Phi')
    ax.set_xlabel('r')
    ax.set_ylabel('Phi(r) - mean')
    ax.set_title(f'Interaction Potential Phi (rel. error: {Phi_result["l2_error_rel"]:.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.img_dir, 'mvp1_0b_phi_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'config': vars(args),
        'Phi_l2_error_rel': float(Phi_result['l2_error_rel']),
        'final_loss': float(history[-1]['loss']),
        'validation_passed': bool(passed),
    }
    with open(os.path.join(args.results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*70}")
    print(f"MVP-1.0b COMPLETE: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*70}")

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
