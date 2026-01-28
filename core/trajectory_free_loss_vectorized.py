"""Vectorized trajectory-free loss for learning interaction potentials.

Optimized version that eliminates the M × (L-1) loop by batching all configurations
together for a single forward pass through the network.

Original: ~25s/epoch due to 3800 iterations (M=200 × L-1=19)
Optimized: Expected ~1-2s/epoch with full vectorization
"""

import torch
import torch.nn as nn
from typing import Tuple
from .nn_models import PotentialNetworks


def compute_pairwise_distances_batched(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances for batched input.

    Args:
        X: Particle positions, shape (B, N, d)
    Returns:
        Distance matrix, shape (B, N, N)
    """
    # X[:, i, :] - X[:, j, :] for all pairs
    diff = X.unsqueeze(2) - X.unsqueeze(1)  # (B, N, N, d)
    distances = torch.norm(diff, dim=-1)  # (B, N, N)
    return distances


def compute_pairwise_diff_batched(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise differences for batched input.

    Args:
        X: Particle positions, shape (B, N, d)
    Returns:
        Difference tensor, shape (B, N, N, d)
    """
    return X.unsqueeze(2) - X.unsqueeze(1)  # (B, N, N, d)


class VectorizedTrajectoryFreeLoss(nn.Module):
    """Fully vectorized trajectory-free loss for IPS potential learning.

    Key optimization: Process all M×(L-1) configurations in a single batch
    instead of iterating one by one.
    """

    def __init__(self, sigma: float = 0.1, d: int = 1):
        super().__init__()
        self.sigma = sigma
        self.sigma_sq = sigma ** 2
        self.d = d

    def compute_drift_batched(
        self,
        X: torch.Tensor,
        networks: PotentialNetworks,
    ) -> torch.Tensor:
        """Compute drift for batched input.

        Args:
            X: Particle positions, shape (B, N, d)
            networks: Potential networks
        Returns:
            Drift, shape (B, N, d)
        """
        B, N, d = X.shape

        # Flatten to (B*N, d) for network forward
        X_flat = X.reshape(B * N, d).requires_grad_(True)

        # Gradient of V: need to compute for all B*N particles
        V_vals = networks.V(X_flat)  # (B*N,)
        grad_V = torch.autograd.grad(
            V_vals.sum(), X_flat, create_graph=True
        )[0]  # (B*N, d)
        grad_V = grad_V.reshape(B, N, d)

        # Pairwise differences and distances
        diff = compute_pairwise_diff_batched(X)  # (B, N, N, d)
        distances = torch.norm(diff, dim=-1)  # (B, N, N)

        # Avoid division by zero
        distances_safe = distances.clone()
        distances_safe[distances_safe < 1e-10] = 1e-10

        # dPhi/dr for all pairs
        dist_flat = distances.flatten().requires_grad_(True)
        Phi_vals = networks.Phi(dist_flat)
        dPhi_dr_flat = torch.autograd.grad(
            Phi_vals.sum(), dist_flat, create_graph=True
        )[0]
        dPhi_dr = dPhi_dr_flat.reshape(B, N, N)  # (B, N, N)

        # grad_Phi(X_i - X_j) = dPhi/dr * (X_i - X_j) / |X_i - X_j|
        unit_diff = diff / distances_safe.unsqueeze(-1)  # (B, N, N, d)
        grad_Phi_pairs = dPhi_dr.unsqueeze(-1) * unit_diff  # (B, N, N, d)

        # Zero out diagonal (self-interaction)
        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        grad_Phi_pairs[:, mask] = 0

        # Mean field: (1/N) sum_j grad_Phi(X_i - X_j)
        grad_Phi_mean = grad_Phi_pairs.sum(dim=2) / N  # (B, N, d)

        # Total drift
        drift = -grad_V - grad_Phi_mean
        return drift

    def compute_laplacian_sum_batched(
        self,
        X: torch.Tensor,
        networks: PotentialNetworks,
    ) -> torch.Tensor:
        """Compute Laplacian sum for batched input.

        Args:
            X: Particle positions, shape (B, N, d)
            networks: Potential networks
        Returns:
            Laplacian sum, shape (B, N)
        """
        B, N, d = X.shape

        # Flatten for network
        X_flat = X.reshape(B * N, d).requires_grad_(True)

        # V values and first gradient
        V_vals = networks.V(X_flat)  # (B*N,)
        grad_V = torch.autograd.grad(
            V_vals.sum(), X_flat, create_graph=True
        )[0]  # (B*N, d)

        # Laplacian of V = sum of second derivatives
        laplacian_V = torch.zeros(B * N, device=X.device)
        for i in range(d):
            grad_i = grad_V[:, i]
            grad2 = torch.autograd.grad(
                grad_i.sum(), X_flat, create_graph=True
            )[0][:, i]
            laplacian_V = laplacian_V + grad2
        laplacian_V = laplacian_V.reshape(B, N)  # (B, N)

        # Pairwise distances
        distances = compute_pairwise_distances_batched(X)  # (B, N, N)

        if d == 1:
            # For 1D: Laplacian(Phi(|x|)) = d²Phi/dr²
            dist_flat = distances.flatten().requires_grad_(True)
            Phi_vals = networks.Phi(dist_flat)

            grad1 = torch.autograd.grad(
                Phi_vals.sum(), dist_flat, create_graph=True
            )[0]
            grad2 = torch.autograd.grad(
                grad1.sum(), dist_flat, create_graph=True
            )[0]
            laplacian_Phi = grad2.reshape(B, N, N)
        else:
            # For higher d: Laplacian(Phi(r)) = d²Phi/dr² + (d-1)/r * dPhi/dr
            distances_safe = distances.clone()
            distances_safe[distances_safe < 1e-10] = 1e-10

            dist_flat = distances.flatten().requires_grad_(True)
            Phi_vals = networks.Phi(dist_flat)

            grad1 = torch.autograd.grad(
                Phi_vals.sum(), dist_flat, create_graph=True
            )[0]
            grad2 = torch.autograd.grad(
                grad1.sum(), dist_flat, create_graph=True
            )[0]

            d2Phi = grad2.reshape(B, N, N)
            dPhi = grad1.reshape(B, N, N)
            laplacian_Phi = d2Phi + (d - 1) / distances_safe * dPhi

        # Zero out diagonal
        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        laplacian_Phi[:, mask] = 0

        # Mean field
        laplacian_Phi_mean = laplacian_Phi.sum(dim=2) / N  # (B, N)

        return laplacian_V + laplacian_Phi_mean

    def compute_energy_batched(
        self,
        X: torch.Tensor,
        networks: PotentialNetworks,
    ) -> torch.Tensor:
        """Compute energy for batched input.

        Args:
            X: Particle positions, shape (B, N, d)
            networks: Potential networks
        Returns:
            Energy, shape (B,)
        """
        B, N, d = X.shape

        # V term
        X_flat = X.reshape(B * N, d)
        V_vals = networks.V(X_flat).reshape(B, N)  # (B, N)
        V_mean = V_vals.mean(dim=1)  # (B,)

        # Phi term
        distances = compute_pairwise_distances_batched(X)  # (B, N, N)
        Phi_vals = networks.Phi(distances)  # (B, N, N)

        # Zero out diagonal
        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        Phi_vals[:, mask] = 0

        # Mean field: (1/N²) sum_{i,j} Phi
        Phi_mean = Phi_vals.sum(dim=(1, 2)) / (N * N)  # (B,)

        return V_mean + Phi_mean

    def forward(
        self,
        data: torch.Tensor,
        t_snapshots: torch.Tensor,
        networks: PotentialNetworks,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute trajectory-free loss with full vectorization.

        Args:
            data: Snapshot data, shape (M, L, N, d)
            t_snapshots: Time points, shape (L,)
            networks: Potential networks
        Returns:
            loss: Total loss (scalar)
            info: Dictionary with component losses
        """
        M, L, N, d = data.shape
        device = data.device

        # Compute dt for each time pair (assume uniform spacing, but handle general case)
        dt = t_snapshots[1:] - t_snapshots[:-1]  # (L-1,)

        # Reshape data: stack all (m, ell) pairs
        # X_curr: all current configurations, shape (M * (L-1), N, d)
        # X_next: all next configurations, shape (M * (L-1), N, d)
        X_curr = data[:, :-1, :, :].reshape(M * (L - 1), N, d)
        X_next = data[:, 1:, :, :].reshape(M * (L - 1), N, d)

        # dt for each pair: repeat M times for each time step
        # Shape: (M * (L-1),)
        dt_expanded = dt.unsqueeze(0).expand(M, -1).reshape(M * (L - 1))

        # Compute all quantities in batch
        drift = self.compute_drift_batched(X_curr, networks)  # (B, N, d)
        laplacian_sum = self.compute_laplacian_sum_batched(X_curr, networks)  # (B, N)
        E_curr = self.compute_energy_batched(X_curr, networks)  # (B,)
        E_next = self.compute_energy_batched(X_next, networks)  # (B,)

        # Dissipation: (1/N) sum_i |drift|² * dt
        J_diss_per_pair = (drift ** 2).sum(dim=(1, 2)) / N * dt_expanded  # (B,)

        # Diffusion: sigma² * (1/N) sum_i [laplacian sum] * dt
        J_diff_per_pair = self.sigma_sq * laplacian_sum.mean(dim=1) * dt_expanded  # (B,)

        # Energy change
        J_energy_per_pair = E_next - E_curr  # (B,)

        # Sum and normalize
        n_pairs = M * (L - 1)
        total_diss = J_diss_per_pair.sum() / n_pairs
        total_diff = J_diff_per_pair.sum() / n_pairs
        total_energy_change = J_energy_per_pair.sum() / n_pairs

        # Loss: squared residual
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


# Alias for backward compatibility
TrajectoryFreeLoss = VectorizedTrajectoryFreeLoss
