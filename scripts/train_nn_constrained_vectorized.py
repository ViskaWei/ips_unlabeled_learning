#!/usr/bin/env python
"""Train neural networks with identifiability constraints + VECTORIZED loss.

MVP-1.1: Combines:
1. Identifiability constraints (V(0)=0, Phi(r_ref)=0)
2. Vectorized loss computation (~10-20x speedup)
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

from core.nn_models import MLP, SymmetricMLP
from core.trajectory_free_loss_vectorized import (
    compute_pairwise_distances_batched,
    compute_pairwise_diff_batched,
)
from core.true_potentials import evaluate_V_error, evaluate_Phi_error
from core.sde_simulator import SDESimulator
from core.potentials import HarmonicPotential, GaussianInteraction


class ConstrainedPotentialNetworks(nn.Module):
    """Potential networks with built-in identifiability constraints.

    Key design:
    1. V is parameterized as V(x) = V_raw(x) - V_raw(0), so V(0) = 0
    2. Phi is parameterized as Phi(r) = Phi_raw(r) - Phi_raw(r_ref), so Phi(r_ref) = 0
    """

    def __init__(self, d: int = 1, hidden_dims: list = [64, 64], r_ref: float = 5.0):
        super().__init__()
        self.d = d
        self.r_ref = r_ref

        self.V_net = MLP(input_dim=d, hidden_dims=hidden_dims, output_dim=1)
        self.Phi_net = SymmetricMLP(input_dim=1, hidden_dims=hidden_dims, output_dim=1)

        self._zero_d = None
        self._r_ref_tensor = None

    def _get_zero_d(self, device):
        if self._zero_d is None or self._zero_d.device != device:
            self._zero_d = torch.zeros(1, self.d, device=device)
        return self._zero_d

    def _get_r_ref_tensor(self, device):
        if self._r_ref_tensor is None or self._r_ref_tensor.device != device:
            self._r_ref_tensor = torch.tensor([[self.r_ref]], device=device)
        return self._r_ref_tensor

    def V_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.V_net(x).squeeze(-1)

    def Phi_raw(self, r: torch.Tensor) -> torch.Tensor:
        r_input = r.unsqueeze(-1) if r.dim() == 0 or r.shape[-1] != 1 else r
        return self.Phi_net(r_input).squeeze(-1)

    def V(self, x: torch.Tensor) -> torch.Tensor:
        """V(x) with constraint V(0) = 0."""
        V_raw_x = self.V_raw(x)
        V_raw_0 = self.V_raw(self._get_zero_d(x.device)).squeeze()
        return V_raw_x - V_raw_0

    def Phi(self, r: torch.Tensor) -> torch.Tensor:
        """Phi(r) with constraint Phi(r_ref) = 0."""
        Phi_raw_r = self.Phi_raw(r)
        Phi_raw_ref = self.Phi_raw(self._get_r_ref_tensor(r.device)).squeeze()
        return Phi_raw_r - Phi_raw_ref

    def grad_V(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        V_vals = self.V(x)
        grad = torch.autograd.grad(V_vals.sum(), x, create_graph=True)[0]
        return grad

    def laplacian_V(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        V_vals = self.V(x)
        grad = torch.autograd.grad(V_vals.sum(), x, create_graph=True)[0]

        laplacian = torch.zeros(x.shape[0], device=x.device)
        for i in range(self.d):
            grad_i = grad[:, i]
            grad2 = torch.autograd.grad(grad_i.sum(), x, create_graph=True)[0][:, i]
            laplacian = laplacian + grad2

        return laplacian

    def grad_Phi(self, r: torch.Tensor) -> torch.Tensor:
        r_flat = r.flatten().requires_grad_(True)
        Phi_vals = self.Phi(r_flat)
        grad = torch.autograd.grad(Phi_vals.sum(), r_flat, create_graph=True)[0]
        return grad.reshape(r.shape)

    def laplacian_Phi_1d(self, r: torch.Tensor) -> torch.Tensor:
        r_flat = r.flatten().requires_grad_(True)
        Phi_vals = self.Phi(r_flat)
        grad1 = torch.autograd.grad(Phi_vals.sum(), r_flat, create_graph=True)[0]
        grad2 = torch.autograd.grad(grad1.sum(), r_flat, create_graph=True)[0]
        return grad2.reshape(r.shape)


class VectorizedConstrainedLoss(nn.Module):
    """Vectorized trajectory-free loss with identifiability regularization."""

    def __init__(
        self,
        sigma: float = 0.1,
        d: int = 1,
        lambda_anchor: float = 1.0,
        lambda_decay: float = 0.1,
        lambda_grad_scale: float = 0.01,
        r_decay: float = 5.0,
    ):
        super().__init__()
        self.sigma = sigma
        self.sigma_sq = sigma ** 2
        self.d = d
        self.lambda_anchor = lambda_anchor
        self.lambda_decay = lambda_decay
        self.lambda_grad_scale = lambda_grad_scale
        self.r_decay = r_decay

    def compute_drift_batched(self, X: torch.Tensor, networks) -> torch.Tensor:
        """Compute drift for batched input. Shape: (B, N, d) -> (B, N, d)"""
        B, N, d = X.shape

        X_flat = X.reshape(B * N, d).requires_grad_(True)
        V_vals = networks.V(X_flat)
        grad_V = torch.autograd.grad(V_vals.sum(), X_flat, create_graph=True)[0]
        grad_V = grad_V.reshape(B, N, d)

        diff = compute_pairwise_diff_batched(X)
        distances = torch.norm(diff, dim=-1)
        distances_safe = distances.clone()
        distances_safe[distances_safe < 1e-10] = 1e-10

        dist_flat = distances.flatten().requires_grad_(True)
        Phi_vals = networks.Phi(dist_flat)
        dPhi_dr_flat = torch.autograd.grad(Phi_vals.sum(), dist_flat, create_graph=True)[0]
        dPhi_dr = dPhi_dr_flat.reshape(B, N, N)

        unit_diff = diff / distances_safe.unsqueeze(-1)
        grad_Phi_pairs = dPhi_dr.unsqueeze(-1) * unit_diff

        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        grad_Phi_pairs[:, mask] = 0

        grad_Phi_mean = grad_Phi_pairs.sum(dim=2) / N

        drift = -grad_V - grad_Phi_mean
        return drift

    def compute_laplacian_sum_batched(self, X: torch.Tensor, networks) -> torch.Tensor:
        """Compute Laplacian sum for batched input. Shape: (B, N, d) -> (B, N)"""
        B, N, d = X.shape

        X_flat = X.reshape(B * N, d).requires_grad_(True)
        V_vals = networks.V(X_flat)
        grad_V = torch.autograd.grad(V_vals.sum(), X_flat, create_graph=True)[0]

        laplacian_V = torch.zeros(B * N, device=X.device)
        for i in range(d):
            grad_i = grad_V[:, i]
            grad2 = torch.autograd.grad(grad_i.sum(), X_flat, create_graph=True)[0][:, i]
            laplacian_V = laplacian_V + grad2
        laplacian_V = laplacian_V.reshape(B, N)

        distances = compute_pairwise_distances_batched(X)

        dist_flat = distances.flatten().requires_grad_(True)
        Phi_vals = networks.Phi(dist_flat)
        grad1 = torch.autograd.grad(Phi_vals.sum(), dist_flat, create_graph=True)[0]
        grad2 = torch.autograd.grad(grad1.sum(), dist_flat, create_graph=True)[0]
        laplacian_Phi = grad2.reshape(B, N, N)

        if d > 1:
            distances_safe = distances.clone()
            distances_safe[distances_safe < 1e-10] = 1e-10
            dPhi = grad1.reshape(B, N, N)
            laplacian_Phi = laplacian_Phi + (d - 1) / distances_safe * dPhi

        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        laplacian_Phi[:, mask] = 0

        laplacian_Phi_mean = laplacian_Phi.sum(dim=2) / N

        return laplacian_V + laplacian_Phi_mean

    def compute_energy_batched(self, X: torch.Tensor, networks) -> torch.Tensor:
        """Compute energy for batched input. Shape: (B, N, d) -> (B,)"""
        B, N, d = X.shape

        X_flat = X.reshape(B * N, d)
        V_vals = networks.V(X_flat).reshape(B, N)
        V_mean = V_vals.mean(dim=1)

        distances = compute_pairwise_distances_batched(X)
        Phi_vals = networks.Phi(distances)

        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        Phi_vals[:, mask] = 0

        Phi_mean = Phi_vals.sum(dim=(1, 2)) / (N * N)

        return V_mean + Phi_mean

    def compute_regularization(self, networks, device) -> dict:
        """Compute regularization terms for identifiability."""
        zero_x = torch.zeros(1, self.d, device=device)
        V_at_zero = networks.V(zero_x)
        loss_anchor_V = V_at_zero ** 2

        r_decay_t = torch.tensor([self.r_decay], device=device)
        Phi_at_decay = networks.Phi(r_decay_t)
        loss_decay_Phi = Phi_at_decay ** 2

        return {
            'loss_anchor_V': loss_anchor_V.squeeze(),
            'loss_decay_Phi': loss_decay_Phi.squeeze(),
        }

    def forward(self, data: torch.Tensor, t_snapshots: torch.Tensor, networks) -> tuple:
        """Compute vectorized trajectory-free loss with regularization."""
        M, L, N, d = data.shape
        device = data.device

        dt = t_snapshots[1:] - t_snapshots[:-1]

        X_curr = data[:, :-1, :, :].reshape(M * (L - 1), N, d)
        X_next = data[:, 1:, :, :].reshape(M * (L - 1), N, d)
        dt_expanded = dt.unsqueeze(0).expand(M, -1).reshape(M * (L - 1))

        drift = self.compute_drift_batched(X_curr, networks)
        laplacian_sum = self.compute_laplacian_sum_batched(X_curr, networks)
        E_curr = self.compute_energy_batched(X_curr, networks)
        E_next = self.compute_energy_batched(X_next, networks)

        J_diss_per_pair = (drift ** 2).sum(dim=(1, 2)) / N * dt_expanded
        J_diff_per_pair = self.sigma_sq * laplacian_sum.mean(dim=1) * dt_expanded
        J_energy_per_pair = E_next - E_curr

        n_pairs = M * (L - 1)
        total_diss = J_diss_per_pair.sum() / n_pairs
        total_diff = J_diff_per_pair.sum() / n_pairs
        total_energy_change = J_energy_per_pair.sum() / n_pairs

        residual = total_diss + total_diff - 2 * total_energy_change
        loss_main = residual ** 2

        reg = self.compute_regularization(networks, device)
        loss_reg = (
            self.lambda_anchor * reg['loss_anchor_V'] +
            self.lambda_decay * reg['loss_decay_Phi']
        )

        loss_total = loss_main + loss_reg

        info = {
            'loss': loss_total.item(),
            'loss_main': loss_main.item(),
            'loss_reg': loss_reg.item(),
            'residual': residual.item(),
            'J_diss': total_diss.item(),
            'J_diff': total_diff.item(),
            'J_energy_change': total_energy_change.item(),
        }

        return loss_total, info


def parse_args():
    parser = argparse.ArgumentParser(description='Train NN with constraints + vectorized loss (MVP-1.1)')
    parser.add_argument('--N', type=int, default=10, help='Number of particles')
    parser.add_argument('--d', type=int, default=1, help='Spatial dimension')
    parser.add_argument('--L', type=int, default=20, help='Number of snapshots')
    parser.add_argument('--M', type=int, default=200, help='Number of samples')
    parser.add_argument('--dt', type=float, default=0.01, help='SDE time step')
    parser.add_argument('--T', type=float, default=2.0, help='Total simulation time')
    parser.add_argument('--sigma', type=float, default=0.1, help='Noise strength')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--hidden_dims', type=str, default='64,64', help='Hidden dimensions')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--log_interval', type=int, default=100, help='Log every N epochs')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--results_dir', type=str, default='results/mvp1_1', help='Results directory')
    parser.add_argument('--img_dir', type=str, default='experiments/ips_unlabeled/img', help='Image directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--V_k', type=float, default=1.0, help='Harmonic V constant')
    parser.add_argument('--Phi_A', type=float, default=1.0, help='Gaussian Phi amplitude')
    parser.add_argument('--Phi_sigma', type=float, default=1.0, help='Gaussian Phi width')
    parser.add_argument('--r_ref', type=float, default=5.0, help='Reference distance for Phi(r_ref)=0')
    parser.add_argument('--lambda_anchor', type=float, default=1.0, help='V(0)=0 constraint weight')
    parser.add_argument('--lambda_decay', type=float, default=0.1, help='Phi decay constraint weight')
    return parser.parse_args()


def generate_data(N, d, L, M, dt, T, sigma, seed, V_k, Phi_A, Phi_sigma):
    print("Generating data...")
    V = HarmonicPotential(k=V_k)
    Phi = GaussianInteraction(A=Phi_A, sigma=Phi_sigma)
    simulator = SDESimulator(V=V, Phi=Phi, sigma=sigma, dt=dt)
    data, t_snapshots = simulator.simulate(N=N, d=d, T=T, L=L, M=M, seed=seed)
    print(f"Data shape: {data.shape}")
    return data, t_snapshots


def train(data, t_snapshots, networks, loss_fn, optimizer, epochs, log_interval, device, patience=100):
    data_torch = torch.tensor(data, dtype=torch.float32, device=device)
    t_torch = torch.tensor(t_snapshots, dtype=torch.float32, device=device)

    history = []
    networks.train()

    print("\nTraining (VECTORIZED + CONSTRAINED)...")
    print("-" * 80)

    start_time = time.time()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        epoch_start = time.time()
        optimizer.zero_grad()

        loss, info = loss_fn(data_torch, t_torch, networks)

        loss.backward()
        optimizer.step()

        history.append(info)
        epoch_time = time.time() - epoch_start

        if info['loss'] < best_loss - 1e-7:
            best_loss = info['loss']
            patience_counter = 0
            best_state = {k: v.clone() for k, v in networks.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {info['loss']:.6f} | "
                  f"Main: {info['loss_main']:.6f} | Reg: {info['loss_reg']:.6f} | "
                  f"Res: {info['residual']:.6f} | {epoch_time:.2f}s/ep | Total: {elapsed:.1f}s", flush=True)

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("-" * 80)
    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s ({total_time/(epoch+1):.2f}s/epoch)")

    if best_state is not None:
        networks.load_state_dict(best_state)
        print(f"Restored best model (loss: {best_loss:.6f})")

    return history


def plot_results(networks, history, V_k, Phi_A, Phi_sigma, device, img_dir):
    # Training history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = list(range(1, len(history) + 1))

    ax = axes[0]
    ax.semilogy(epochs, [h['loss'] for h in history], 'b-', label='Total')
    ax.semilogy(epochs, [h['loss_main'] for h in history], 'r--', label='Main')
    ax.semilogy(epochs, [h['loss_reg'] for h in history], 'g--', label='Reg')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, [h['J_diss'] for h in history], 'r-', label='J_diss')
    ax.plot(epochs, [h['J_diff'] for h in history], 'g-', label='J_diff')
    ax.plot(epochs, [h['J_energy_change'] for h in history], 'b-', label='J_energy')
    ax.set_xlabel('Epoch')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, [h['residual'] for h in history], 'k-')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_title('Residual')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'mvp1_1_training_history.png'), dpi=150)
    plt.close()

    # Potential comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x_range = np.linspace(-2, 2, 200)
    V_result = evaluate_V_error(networks, x_range, k=V_k, device=device)

    ax = axes[0, 0]
    ax.plot(x_range, V_result['V_true_centered'], 'r-', lw=2, label='True V')
    ax.plot(x_range, V_result['V_pred_centered'], 'b--', lw=2, label='Learned V')
    ax.set_xlabel('x')
    ax.set_ylabel('V(x) - mean')
    ax.set_title(f'V (rel. error: {V_result["l2_error_rel"]:.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    r_range = np.linspace(0, 3, 200)
    Phi_result = evaluate_Phi_error(networks, r_range, A=Phi_A, sigma=Phi_sigma, device=device)

    ax = axes[0, 1]
    ax.plot(r_range, Phi_result['Phi_true_centered'], 'r-', lw=2, label='True Phi')
    ax.plot(r_range, Phi_result['Phi_pred_centered'], 'b--', lw=2, label='Learned Phi')
    ax.set_xlabel('r')
    ax.set_ylabel('Phi(r) - mean')
    ax.set_title(f'Phi (rel. error: {Phi_result["l2_error_rel"]:.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradients
    true_grad_V = V_k * x_range
    x_torch = torch.tensor(x_range.reshape(-1, 1), dtype=torch.float32, device=device).requires_grad_(True)
    with torch.enable_grad():
        networks.eval()
        V_vals = networks.V(x_torch)
        grad_V = torch.autograd.grad(V_vals.sum(), x_torch)[0].detach().cpu().numpy().flatten()

    ax = axes[1, 0]
    ax.plot(x_range, true_grad_V, 'r-', lw=2, label='True')
    ax.plot(x_range, grad_V, 'b--', lw=2, label='Learned')
    ax.set_xlabel('x')
    ax.set_ylabel('dV/dx')
    ax.set_title('Gradient of V')
    ax.legend()
    ax.grid(True, alpha=0.3)

    true_grad_Phi = -Phi_A * r_range / (Phi_sigma**2) * np.exp(-r_range**2 / (2 * Phi_sigma**2))
    r_torch = torch.tensor(r_range, dtype=torch.float32, device=device)
    with torch.enable_grad():
        grad_Phi = networks.grad_Phi(r_torch).detach().cpu().numpy()

    ax = axes[1, 1]
    ax.plot(r_range, true_grad_Phi, 'r-', lw=2, label='True')
    ax.plot(r_range, grad_Phi, 'b--', lw=2, label='Learned')
    ax.set_xlabel('r')
    ax.set_ylabel('dPhi/dr')
    ax.set_title('Gradient of Phi')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'mvp1_1_potential_comparison.png'), dpi=150)
    plt.close()

    return V_result, Phi_result


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    print("=" * 80)
    print("MVP-1.1: Trajectory-Free Loss + Identifiability Constraints (VECTORIZED)")
    print("=" * 80)
    print(f"\nConfig: N={args.N}, L={args.L}, M={args.M}, sigma={args.sigma}")
    print(f"Constraints: V(0)=0, Phi({args.r_ref})=0")
    print(f"lambda_anchor={args.lambda_anchor}, lambda_decay={args.lambda_decay}")
    print()

    data, t_snapshots = generate_data(
        args.N, args.d, args.L, args.M, args.dt, args.T, args.sigma, args.seed,
        args.V_k, args.Phi_A, args.Phi_sigma,
    )

    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    networks = ConstrainedPotentialNetworks(
        d=args.d, hidden_dims=hidden_dims, r_ref=args.r_ref
    ).to(args.device)
    print(f"Network parameters: {sum(p.numel() for p in networks.parameters())}")

    loss_fn = VectorizedConstrainedLoss(
        sigma=args.sigma, d=args.d,
        lambda_anchor=args.lambda_anchor,
        lambda_decay=args.lambda_decay,
        r_decay=args.r_ref,
    )
    optimizer = optim.Adam(networks.parameters(), lr=args.lr)

    history = train(
        data, t_snapshots, networks, loss_fn, optimizer,
        args.epochs, args.log_interval, args.device, args.patience,
    )

    print("\nEvaluating...")
    V_result, Phi_result = plot_results(
        networks, history, args.V_k, args.Phi_A, args.Phi_sigma,
        args.device, args.img_dir,
    )

    print(f"\n{'='*80}")
    print("Results")
    print(f"{'='*80}")
    print(f"V relative L2 error:   {V_result['l2_error_rel']:.4f} ({V_result['l2_error_rel']*100:.2f}%)")
    print(f"Phi relative L2 error: {Phi_result['l2_error_rel']:.4f} ({Phi_result['l2_error_rel']*100:.2f}%)")

    threshold = 0.10
    V_pass = V_result['l2_error_rel'] < threshold
    Phi_pass = Phi_result['l2_error_rel'] < threshold
    overall_pass = V_pass and Phi_pass

    print(f"\nValidation (< {threshold*100:.0f}%):")
    print(f"  V:   {'PASS' if V_pass else 'FAIL'}")
    print(f"  Phi: {'PASS' if Phi_pass else 'FAIL'}")
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'='*80}")

    metrics = {
        'config': vars(args),
        'V_l2_error_rel': float(V_result['l2_error_rel']),
        'Phi_l2_error_rel': float(Phi_result['l2_error_rel']),
        'final_loss': float(history[-1]['loss']),
        'validation_passed': bool(overall_pass),
    }

    with open(os.path.join(args.results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    torch.save(networks.state_dict(), os.path.join(args.results_dir, 'model.pt'))

    print(f"\nMVP-1.1 COMPLETE: {'PASS' if overall_pass else 'FAIL'}")

    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
