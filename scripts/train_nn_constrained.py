#!/usr/bin/env python
"""Train neural networks with identifiability constraints.

MVP-1.1: Add identifiability constraints to fix V-Phi trade-off.

Key constraints:
1. V(0) = 0 (anchor V at origin)
2. Phi(large_r) -> 0 (Phi decays at infinity)
3. Scale regularization on gradients
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

from core.nn_models import MLP, SymmetricMLP, compute_pairwise_diff, compute_pairwise_distances
from core.trajectory_free_loss import TrajectoryFreeLoss
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
        self.r_ref = r_ref  # Reference distance where Phi -> 0

        # V: R^d -> R (kinetic potential)
        self.V_net = MLP(input_dim=d, hidden_dims=hidden_dims, output_dim=1)

        # Phi: R -> R (interaction potential, depends on distance)
        self.Phi_net = SymmetricMLP(input_dim=1, hidden_dims=hidden_dims, output_dim=1)

        # Precompute reference values
        self._zero_d = None  # Zero vector of dimension d
        self._r_ref_tensor = None

    def _get_zero_d(self, device):
        """Get zero vector of dimension d."""
        if self._zero_d is None or self._zero_d.device != device:
            self._zero_d = torch.zeros(1, self.d, device=device)
        return self._zero_d

    def _get_r_ref_tensor(self, device):
        """Get reference distance tensor."""
        if self._r_ref_tensor is None or self._r_ref_tensor.device != device:
            self._r_ref_tensor = torch.tensor([[self.r_ref]], device=device)
        return self._r_ref_tensor

    def V_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Raw V network output."""
        return self.V_net(x).squeeze(-1)

    def Phi_raw(self, r: torch.Tensor) -> torch.Tensor:
        """Raw Phi network output."""
        r_input = r.unsqueeze(-1) if r.dim() == 0 or r.shape[-1] != 1 else r
        return self.Phi_net(r_input).squeeze(-1)

    def V(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate kinetic potential V(x) with constraint V(0) = 0.

        V(x) = V_raw(x) - V_raw(0)

        Args:
            x: Positions, shape (..., d)
        Returns:
            V values, shape (...)
        """
        V_raw_x = self.V_raw(x)
        V_raw_0 = self.V_raw(self._get_zero_d(x.device)).squeeze()
        return V_raw_x - V_raw_0

    def Phi(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate interaction potential Phi(r) with constraint Phi(r_ref) = 0.

        Phi(r) = Phi_raw(r) - Phi_raw(r_ref)

        Args:
            r: Distances, shape (...)
        Returns:
            Phi values, shape (...)
        """
        Phi_raw_r = self.Phi_raw(r)
        Phi_raw_ref = self.Phi_raw(self._get_r_ref_tensor(r.device)).squeeze()
        return Phi_raw_r - Phi_raw_ref

    def grad_V(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient of V w.r.t. x using AD.

        Note: The constraint V(0)=0 doesn't affect gradient since it's a constant shift.
        """
        x = x.requires_grad_(True)
        V_vals = self.V(x)
        grad = torch.autograd.grad(V_vals.sum(), x, create_graph=True)[0]
        return grad

    def laplacian_V(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian of V using AD."""
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
        """Compute dPhi/dr using AD."""
        r_flat = r.flatten().requires_grad_(True)
        Phi_vals = self.Phi(r_flat)
        grad = torch.autograd.grad(Phi_vals.sum(), r_flat, create_graph=True)[0]
        return grad.reshape(r.shape)

    def laplacian_Phi_1d(self, r: torch.Tensor) -> torch.Tensor:
        """Compute d²Phi/dr² for 1D case using AD."""
        r_flat = r.flatten().requires_grad_(True)
        Phi_vals = self.Phi(r_flat)

        grad1 = torch.autograd.grad(Phi_vals.sum(), r_flat, create_graph=True)[0]
        grad2 = torch.autograd.grad(grad1.sum(), r_flat, create_graph=True)[0]

        return grad2.reshape(r.shape)


class ConstrainedTrajectoryFreeLoss(nn.Module):
    """Trajectory-free loss with identifiability regularization."""

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

        # Regularization weights
        self.lambda_anchor = lambda_anchor
        self.lambda_decay = lambda_decay
        self.lambda_grad_scale = lambda_grad_scale
        self.r_decay = r_decay

    def compute_drift(self, X: torch.Tensor, networks) -> torch.Tensor:
        """Compute drift term."""
        N, d = X.shape

        grad_V = networks.grad_V(X)

        diff = compute_pairwise_diff(X)
        distances = torch.norm(diff, dim=-1)

        distances_safe = distances.clone()
        distances_safe[distances_safe < 1e-10] = 1e-10

        dPhi_dr = networks.grad_Phi(distances)

        unit_diff = diff / distances_safe.unsqueeze(-1)
        grad_Phi_pairs = dPhi_dr.unsqueeze(-1) * unit_diff

        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        grad_Phi_pairs[mask] = 0

        grad_Phi_mean = grad_Phi_pairs.sum(dim=1) / N

        drift = -grad_V - grad_Phi_mean
        return drift

    def compute_laplacian_sum(self, X: torch.Tensor, networks) -> torch.Tensor:
        """Compute sum of Laplacians."""
        N, d = X.shape

        laplacian_V = networks.laplacian_V(X)

        distances = compute_pairwise_distances(X)

        if d == 1:
            laplacian_Phi = networks.laplacian_Phi_1d(distances)
        else:
            distances_safe = distances.clone()
            distances_safe[distances_safe < 1e-10] = 1e-10

            d2Phi = networks.laplacian_Phi_1d(distances)
            dPhi = networks.grad_Phi(distances)
            laplacian_Phi = d2Phi + (d - 1) / distances_safe * dPhi

        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        laplacian_Phi[mask] = 0

        laplacian_Phi_mean = laplacian_Phi.sum(dim=1) / N

        return laplacian_V + laplacian_Phi_mean

    def compute_energy(self, X: torch.Tensor, networks) -> torch.Tensor:
        """Compute energy."""
        N = X.shape[0]

        V_vals = networks.V(X)
        V_mean = V_vals.mean()

        distances = compute_pairwise_distances(X)
        Phi_vals = networks.Phi(distances)

        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        Phi_vals[mask] = 0

        Phi_mean = Phi_vals.sum() / (N * N)

        return V_mean + Phi_mean

    def compute_regularization(self, networks, data: torch.Tensor) -> dict:
        """Compute regularization terms for identifiability.

        Returns dict with individual regularization losses.
        """
        device = data.device
        M, L, N, d = data.shape

        # 1. Anchor constraint: V(0) should already be 0 by construction
        # But we can still add a soft penalty to reinforce it
        zero_x = torch.zeros(1, d, device=device)
        V_at_zero = networks.V(zero_x)
        loss_anchor_V = V_at_zero ** 2

        # 2. Decay constraint: Phi(r_decay) should be 0
        r_decay_t = torch.tensor([[self.r_decay]], device=device)
        Phi_at_decay = networks.Phi(r_decay_t.squeeze())
        loss_decay_Phi = Phi_at_decay ** 2

        # 3. Gradient scale: Prevent gradients from being too large or too small
        # Sample some points from data to compute gradient magnitudes
        X_sample = data[0, 0]  # (N, d)
        grad_V = networks.grad_V(X_sample)
        grad_V_norm = (grad_V ** 2).mean()

        # We want gradients to be on order O(1), penalize if too large
        loss_grad_scale = torch.relu(grad_V_norm - 10.0)  # Penalize if > 10

        # 4. Symmetry verification (soft, for Phi)
        # Phi should satisfy Phi(r) = Phi(-r), but for distances r >= 0
        # We check Phi is smooth at r=0
        r_small = torch.tensor([0.01, 0.02, 0.03], device=device)
        Phi_small = networks.Phi(r_small)
        # Phi should be roughly constant near r=0 (smooth minimum/maximum)
        loss_smoothness = ((Phi_small[1] - Phi_small[0]) - (Phi_small[2] - Phi_small[1])) ** 2

        return {
            'loss_anchor_V': loss_anchor_V,
            'loss_decay_Phi': loss_decay_Phi,
            'loss_grad_scale': loss_grad_scale,
            'loss_smoothness': loss_smoothness,
        }

    def forward(self, data: torch.Tensor, t_snapshots: torch.Tensor, networks) -> tuple:
        """Compute trajectory-free loss with regularization."""
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

                drift = self.compute_drift(X_curr, networks)
                J_diss = (drift ** 2).sum() / N * dt

                laplacian_sum = self.compute_laplacian_sum(X_curr, networks)
                J_diff = self.sigma_sq * laplacian_sum.mean() * dt

                E_curr = self.compute_energy(X_curr, networks)
                E_next = self.compute_energy(X_next, networks)
                J_energy_change = E_next - E_curr

                total_diss = total_diss + J_diss
                total_diff = total_diff + J_diff
                total_energy_change = total_energy_change + J_energy_change

        n_pairs = M * (L - 1)
        total_diss = total_diss / n_pairs
        total_diff = total_diff / n_pairs
        total_energy_change = total_energy_change / n_pairs

        # Main loss: squared residual
        residual = total_diss + total_diff - 2 * total_energy_change
        loss_main = residual ** 2

        # Regularization
        reg = self.compute_regularization(networks, data)

        loss_reg = (
            self.lambda_anchor * reg['loss_anchor_V'] +
            self.lambda_decay * reg['loss_decay_Phi'] +
            self.lambda_grad_scale * reg['loss_grad_scale']
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
            'loss_anchor_V': reg['loss_anchor_V'].item(),
            'loss_decay_Phi': reg['loss_decay_Phi'].item(),
            'loss_grad_scale': reg['loss_grad_scale'].item(),
        }

        return loss_total, info


def parse_args():
    parser = argparse.ArgumentParser(description='Train NN with identifiability constraints (MVP-1.1)')
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
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    # True potential parameters
    parser.add_argument('--V_k', type=float, default=1.0, help='Harmonic V constant')
    parser.add_argument('--Phi_A', type=float, default=1.0, help='Gaussian Phi amplitude')
    parser.add_argument('--Phi_sigma', type=float, default=1.0, help='Gaussian Phi width')
    # Constraint parameters
    parser.add_argument('--r_ref', type=float, default=5.0, help='Reference distance for Phi(r_ref)=0')
    parser.add_argument('--lambda_anchor', type=float, default=1.0, help='V(0)=0 constraint weight')
    parser.add_argument('--lambda_decay', type=float, default=0.1, help='Phi decay constraint weight')
    parser.add_argument('--lambda_grad_scale', type=float, default=0.01, help='Gradient scale constraint weight')
    return parser.parse_args()


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
    networks,
    loss_fn,
    optimizer: optim.Optimizer,
    epochs: int,
    log_interval: int,
    device: str,
    patience: int = 100,
    min_delta: float = 1e-7,
) -> list:
    """Train networks with early stopping."""
    data_torch = torch.tensor(data, dtype=torch.float32, device=device)
    t_torch = torch.tensor(t_snapshots, dtype=torch.float32, device=device)

    history = []
    networks.train()

    print("\nTraining with identifiability constraints...")
    print("-" * 80)

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

        current_loss = info['loss']
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in networks.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {info['loss']:.6f} | "
                  f"Main: {info['loss_main']:.6f} | Reg: {info['loss_reg']:.6f} | "
                  f"Res: {info['residual']:.6f} | Time: {elapsed:.1f}s", flush=True)

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print("-" * 80)
    print(f"Training complete in {time.time() - start_time:.1f}s")

    if best_state is not None:
        networks.load_state_dict(best_state)
        print(f"Restored best model (loss: {best_loss:.6f})")

    return history


def plot_training_history(history: list, save_path: str):
    """Plot training loss history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = list(range(1, len(history) + 1))
    losses = [h['loss'] for h in history]
    loss_main = [h['loss_main'] for h in history]
    loss_reg = [h['loss_reg'] for h in history]
    j_diss = [h['J_diss'] for h in history]
    j_diff = [h['J_diff'] for h in history]
    j_energy = [h['J_energy_change'] for h in history]

    # Total loss
    ax = axes[0]
    ax.semilogy(epochs, np.abs(losses), 'b-', lw=1.5, label='Total')
    ax.semilogy(epochs, np.abs(loss_main), 'r--', lw=1, label='Main')
    ax.semilogy(epochs, np.abs(loss_reg), 'g--', lw=1, label='Reg')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|Loss|')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss components
    ax = axes[1]
    ax.plot(epochs, j_diss, 'r-', label='J_diss', alpha=0.7)
    ax.plot(epochs, j_diff, 'g-', label='J_diff', alpha=0.7)
    ax.plot(epochs, j_energy, 'b-', label='J_energy', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual
    ax = axes[2]
    residuals = [h['residual'] for h in history]
    ax.plot(epochs, residuals, 'k-', lw=1.5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Residual')
    ax.set_title('Weak-form Residual')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot to {save_path}")


def plot_potential_comparison(
    networks,
    V_k: float, Phi_A: float, Phi_sigma: float,
    device: str, save_path: str,
):
    """Plot learned vs true potentials."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # V comparison
    x_range = np.linspace(-2, 2, 200)
    V_result = evaluate_V_error(networks, x_range, k=V_k, device=device)

    ax = axes[0, 0]
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

    ax = axes[0, 1]
    ax.plot(r_range, Phi_result['Phi_true_centered'], 'r-', lw=2, label='True Phi')
    ax.plot(r_range, Phi_result['Phi_pred_centered'], 'b--', lw=2, label='Learned Phi')
    ax.set_xlabel('r')
    ax.set_ylabel('Phi(r) - mean')
    ax.set_title(f'Interaction Potential Phi (rel. error: {Phi_result["l2_error_rel"]:.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # V gradient comparison
    ax = axes[1, 0]
    # True gradient: grad V = k * x
    true_grad_V = V_k * x_range
    # Learned gradient
    x_torch = torch.tensor(x_range.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.enable_grad():
        networks.eval()
        x_torch.requires_grad_(True)
        V_vals = networks.V(x_torch)
        grad_V = torch.autograd.grad(V_vals.sum(), x_torch)[0].detach().cpu().numpy().flatten()

    ax.plot(x_range, true_grad_V, 'r-', lw=2, label='True grad V')
    ax.plot(x_range, grad_V, 'b--', lw=2, label='Learned grad V')
    ax.set_xlabel('x')
    ax.set_ylabel('dV/dx')
    ax.set_title('Gradient of V')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phi gradient comparison
    ax = axes[1, 1]
    # True gradient: grad Phi = -A * r / sigma^2 * exp(...)
    true_grad_Phi = -Phi_A * r_range / (Phi_sigma**2) * np.exp(-r_range**2 / (2 * Phi_sigma**2))
    # Learned gradient
    r_torch = torch.tensor(r_range, dtype=torch.float32, device=device)
    with torch.enable_grad():
        networks.eval()
        grad_Phi = networks.grad_Phi(r_torch).detach().cpu().numpy()

    ax.plot(r_range, true_grad_Phi, 'r-', lw=2, label='True grad Phi')
    ax.plot(r_range, grad_Phi, 'b--', lw=2, label='Learned grad Phi')
    ax.set_xlabel('r')
    ax.set_ylabel('dPhi/dr')
    ax.set_title('Gradient of Phi')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved potential comparison plot to {save_path}")

    return V_result, Phi_result


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    print("=" * 80)
    print("MVP-1.1: Trajectory-Free Loss with Identifiability Constraints")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  N={args.N}, d={args.d}, L={args.L}, M={args.M}")
    print(f"  sigma={args.sigma}, dt={args.dt}, T={args.T}")
    print(f"  V: Harmonic(k={args.V_k})")
    print(f"  Phi: Gaussian(A={args.Phi_A}, sigma={args.Phi_sigma})")
    print(f"  Network: MLP with hidden_dims={args.hidden_dims}")
    print(f"  Training: lr={args.lr}, epochs={args.epochs}")
    print(f"\nConstraints:")
    print(f"  V(0) = 0 (anchor constraint)")
    print(f"  Phi({args.r_ref}) = 0 (decay constraint)")
    print(f"  lambda_anchor={args.lambda_anchor}, lambda_decay={args.lambda_decay}")
    print()

    # Generate data
    data, t_snapshots, data_config = generate_data_with_interaction(
        N=args.N, d=args.d, L=args.L, M=args.M,
        dt=args.dt, T=args.T, sigma=args.sigma, seed=args.seed,
        V_k=args.V_k, Phi_A=args.Phi_A, Phi_sigma=args.Phi_sigma,
    )

    # Initialize constrained networks
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    networks = ConstrainedPotentialNetworks(
        d=args.d, hidden_dims=hidden_dims, r_ref=args.r_ref
    ).to(args.device)
    print(f"\nNetwork parameters: {sum(p.numel() for p in networks.parameters())}")

    # Loss and optimizer
    loss_fn = ConstrainedTrajectoryFreeLoss(
        sigma=args.sigma, d=args.d,
        lambda_anchor=args.lambda_anchor,
        lambda_decay=args.lambda_decay,
        lambda_grad_scale=args.lambda_grad_scale,
        r_decay=args.r_ref,
    )
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

    print(f"\n{'='*80}")
    print("Evaluation Results")
    print(f"{'='*80}")
    print(f"V relative L2 error:   {V_result['l2_error_rel']:.4f} ({V_result['l2_error_rel']*100:.2f}%)")
    print(f"Phi relative L2 error: {Phi_result['l2_error_rel']:.4f} ({Phi_result['l2_error_rel']*100:.2f}%)")
    print(f"{'='*80}")

    # Check pass/fail
    threshold = 0.10  # 10%
    V_pass = V_result['l2_error_rel'] < threshold
    Phi_pass = Phi_result['l2_error_rel'] < threshold
    overall_pass = V_pass and Phi_pass

    print(f"\nValidation (threshold < {threshold*100:.0f}%):")
    print(f"  V error < {threshold*100:.0f}%:   {'PASS' if V_pass else 'FAIL'}")
    print(f"  Phi error < {threshold*100:.0f}%: {'PASS' if Phi_pass else 'FAIL'}")
    print(f"  Overall:        {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'='*80}")

    # Save plots
    plot_training_history(history, os.path.join(args.img_dir, 'mvp1_1_training_history.png'))
    V_result, Phi_result = plot_potential_comparison(
        networks, args.V_k, args.Phi_A, args.Phi_sigma,
        args.device, os.path.join(args.img_dir, 'mvp1_1_potential_comparison.png'),
    )

    # Save metrics
    metrics = {
        'config': {
            'N': args.N, 'd': args.d, 'L': args.L, 'M': args.M,
            'sigma': args.sigma, 'dt': args.dt, 'T': args.T,
            'V_k': args.V_k, 'Phi_A': args.Phi_A, 'Phi_sigma': args.Phi_sigma,
            'hidden_dims': hidden_dims, 'lr': args.lr, 'epochs': args.epochs,
            'r_ref': args.r_ref, 'lambda_anchor': args.lambda_anchor,
            'lambda_decay': args.lambda_decay, 'lambda_grad_scale': args.lambda_grad_scale,
        },
        'V_l2_error_rel': float(V_result['l2_error_rel']),
        'Phi_l2_error_rel': float(Phi_result['l2_error_rel']),
        'final_loss': float(history[-1]['loss']),
        'final_loss_main': float(history[-1]['loss_main']),
        'final_loss_reg': float(history[-1]['loss_reg']),
        'validation_passed': bool(overall_pass),
        'training_history': {
            'loss': [h['loss'] for h in history[::10]],
            'loss_main': [h['loss_main'] for h in history[::10]],
            'loss_reg': [h['loss_reg'] for h in history[::10]],
            'J_diss': [h['J_diss'] for h in history[::10]],
            'J_diff': [h['J_diff'] for h in history[::10]],
            'J_energy_change': [h['J_energy_change'] for h in history[::10]],
        }
    }

    metrics_path = os.path.join(args.results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Save model
    model_path = os.path.join(args.results_dir, 'model.pt')
    torch.save(networks.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    print(f"\n{'='*80}")
    print(f"MVP-1.1 COMPLETE: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'='*80}")

    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
