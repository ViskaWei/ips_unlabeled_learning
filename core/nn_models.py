"""Neural network models for potential learning."""

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """Simple MLP for potential function."""

    def __init__(self, input_dim: int = 1, hidden_dims: list = [64, 64], output_dim: int = 1):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SymmetricMLP(nn.Module):
    """MLP with symmetric constraint: Phi(x) = 0.5 * (Phi_raw(x) + Phi_raw(-x))."""

    def __init__(self, input_dim: int = 1, hidden_dims: list = [64, 64], output_dim: int = 1):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dims, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.mlp(x) + self.mlp(-x))


class PotentialNetworks(nn.Module):
    """Combined networks for V (kinetic potential) and Phi (interaction potential)."""

    def __init__(self, d: int = 1, hidden_dims: list = [64, 64]):
        super().__init__()
        self.d = d

        # V: R^d -> R (kinetic potential)
        self.V_net = MLP(input_dim=d, hidden_dims=hidden_dims, output_dim=1)

        # Phi: R -> R (interaction potential, depends on distance)
        # Input is |x_i - x_j|, so input_dim=1 regardless of d
        self.Phi_net = SymmetricMLP(input_dim=1, hidden_dims=hidden_dims, output_dim=1)

    def V(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate kinetic potential V(x).

        Args:
            x: Positions, shape (..., d)
        Returns:
            V values, shape (...)
        """
        return self.V_net(x).squeeze(-1)

    def Phi(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate interaction potential Phi(r).

        Args:
            r: Distances, shape (...)
        Returns:
            Phi values, shape (...)
        """
        r_input = r.unsqueeze(-1) if r.dim() == 0 or r.shape[-1] != 1 else r
        return self.Phi_net(r_input).squeeze(-1)

    def grad_V(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient of V w.r.t. x using AD.

        Args:
            x: Positions, shape (N, d), requires_grad=True
        Returns:
            Gradient, shape (N, d)
        """
        x = x.requires_grad_(True)
        V_vals = self.V(x)  # shape (N,)
        grad = torch.autograd.grad(
            V_vals.sum(), x, create_graph=True
        )[0]
        return grad  # shape (N, d)

    def laplacian_V(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian of V using AD.

        Args:
            x: Positions, shape (N, d)
        Returns:
            Laplacian, shape (N,)
        """
        x = x.requires_grad_(True)
        V_vals = self.V(x)  # shape (N,)

        # First gradient
        grad = torch.autograd.grad(
            V_vals.sum(), x, create_graph=True
        )[0]  # shape (N, d)

        # Laplacian = sum of second derivatives
        laplacian = torch.zeros(x.shape[0], device=x.device)
        for i in range(self.d):
            grad_i = grad[:, i]
            grad2 = torch.autograd.grad(
                grad_i.sum(), x, create_graph=True
            )[0][:, i]
            laplacian = laplacian + grad2

        return laplacian

    def grad_Phi(self, r: torch.Tensor) -> torch.Tensor:
        """Compute dPhi/dr using AD.

        Args:
            r: Distances, shape (...)
        Returns:
            Gradient, shape (...)
        """
        r_flat = r.flatten().requires_grad_(True)
        Phi_vals = self.Phi(r_flat)
        grad = torch.autograd.grad(
            Phi_vals.sum(), r_flat, create_graph=True
        )[0]
        return grad.reshape(r.shape)

    def laplacian_Phi_1d(self, r: torch.Tensor) -> torch.Tensor:
        """Compute d²Phi/dr² for 1D case using AD.

        Args:
            r: Distances, shape (...)
        Returns:
            Second derivative, shape (...)
        """
        r_flat = r.flatten().requires_grad_(True)
        Phi_vals = self.Phi(r_flat)

        # First derivative
        grad1 = torch.autograd.grad(
            Phi_vals.sum(), r_flat, create_graph=True
        )[0]

        # Second derivative
        grad2 = torch.autograd.grad(
            grad1.sum(), r_flat, create_graph=True
        )[0]

        return grad2.reshape(r.shape)


def compute_pairwise_distances(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances between particles.

    Args:
        X: Particle positions, shape (N, d)
    Returns:
        Distance matrix, shape (N, N)
    """
    # X_i - X_j for all pairs
    diff = X.unsqueeze(0) - X.unsqueeze(1)  # shape (N, N, d)
    distances = torch.norm(diff, dim=-1)  # shape (N, N)
    return distances


def compute_pairwise_diff(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise differences X_i - X_j.

    Args:
        X: Particle positions, shape (N, d)
    Returns:
        Difference tensor, shape (N, N, d)
    """
    return X.unsqueeze(0) - X.unsqueeze(1)
