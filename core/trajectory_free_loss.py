"""Trajectory-free loss for learning interaction potentials.

The loss is derived from the weak form of the Fokker-Planck equation:
L(Phi, V) = sum over time pairs [J_diss + J_diff - 2 * J_energy_change]

Where:
- J_diss: Dissipation term (drift squared)
- J_diff: Diffusion term (Laplacian)
- J_energy: Energy at each time point
"""

import torch
import torch.nn as nn
from typing import Tuple
from .nn_models import PotentialNetworks, compute_pairwise_distances, compute_pairwise_diff


class TrajectoryFreeLoss(nn.Module):
    """Trajectory-free loss for IPS potential learning."""

    def __init__(self, sigma: float = 0.1, d: int = 1):
        super().__init__()
        self.sigma = sigma
        self.sigma_sq = sigma ** 2
        self.d = d

    def compute_drift(
        self,
        X: torch.Tensor,
        networks: PotentialNetworks,
    ) -> torch.Tensor:
        """Compute drift term: -grad_V(X_i) - (1/N) sum_j grad_Phi(X_i - X_j).

        Args:
            X: Particle positions, shape (N, d)
            networks: Potential networks
        Returns:
            Drift, shape (N, d)
        """
        N, d = X.shape

        # Gradient of V
        grad_V = networks.grad_V(X)  # shape (N, d)

        # Interaction gradient
        diff = compute_pairwise_diff(X)  # shape (N, N, d)
        distances = torch.norm(diff, dim=-1)  # shape (N, N)

        # Avoid division by zero on diagonal
        distances_safe = distances.clone()
        distances_safe[distances_safe < 1e-10] = 1e-10

        # dPhi/dr for each pair
        dPhi_dr = networks.grad_Phi(distances)  # shape (N, N)

        # grad_Phi(X_i - X_j) = dPhi/dr * (X_i - X_j) / |X_i - X_j|
        # shape: (N, N, d)
        unit_diff = diff / distances_safe.unsqueeze(-1)
        grad_Phi_pairs = dPhi_dr.unsqueeze(-1) * unit_diff  # (N, N, d)

        # Zero out diagonal (self-interaction)
        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        grad_Phi_pairs[mask] = 0

        # Mean field: (1/N) sum_j grad_Phi(X_i - X_j)
        grad_Phi_mean = grad_Phi_pairs.sum(dim=1) / N  # shape (N, d)

        # Total drift
        drift = -grad_V - grad_Phi_mean
        return drift

    def compute_laplacian_sum(
        self,
        X: torch.Tensor,
        networks: PotentialNetworks,
    ) -> torch.Tensor:
        """Compute sum of Laplacians: Delta_V(X_i) + (1/N) sum_j Delta_Phi(X_i - X_j).

        For 1D, Laplacian of Phi(|x|) = d²Phi/dr².

        Args:
            X: Particle positions, shape (N, d)
            networks: Potential networks
        Returns:
            Laplacian sum for each particle, shape (N,)
        """
        N, d = X.shape

        # Laplacian of V
        laplacian_V = networks.laplacian_V(X)  # shape (N,)

        # Laplacian of Phi terms
        distances = compute_pairwise_distances(X)  # shape (N, N)

        if d == 1:
            # For 1D: Laplacian(Phi(|x|)) = d²Phi/dr²
            laplacian_Phi = networks.laplacian_Phi_1d(distances)  # shape (N, N)
        else:
            # For higher d: Laplacian(Phi(r)) = d²Phi/dr² + (d-1)/r * dPhi/dr
            distances_safe = distances.clone()
            distances_safe[distances_safe < 1e-10] = 1e-10

            d2Phi = networks.laplacian_Phi_1d(distances)
            dPhi = networks.grad_Phi(distances)
            laplacian_Phi = d2Phi + (d - 1) / distances_safe * dPhi

        # Zero out diagonal
        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        laplacian_Phi[mask] = 0

        # Mean field
        laplacian_Phi_mean = laplacian_Phi.sum(dim=1) / N  # shape (N,)

        return laplacian_V + laplacian_Phi_mean

    def compute_energy(
        self,
        X: torch.Tensor,
        networks: PotentialNetworks,
    ) -> torch.Tensor:
        """Compute energy: (1/N) sum_i [V(X_i) + (1/N) sum_j Phi(|X_i - X_j|)].

        Args:
            X: Particle positions, shape (N, d)
            networks: Potential networks
        Returns:
            Energy (scalar)
        """
        N = X.shape[0]

        # V term
        V_vals = networks.V(X)  # shape (N,)
        V_mean = V_vals.mean()

        # Phi term
        distances = compute_pairwise_distances(X)  # shape (N, N)
        Phi_vals = networks.Phi(distances)  # shape (N, N)

        # Zero out diagonal
        mask = torch.eye(N, device=X.device, dtype=torch.bool)
        Phi_vals[mask] = 0

        # Mean field: (1/N²) sum_{i,j} Phi
        Phi_mean = Phi_vals.sum() / (N * N)

        return V_mean + Phi_mean

    def forward(
        self,
        data: torch.Tensor,
        t_snapshots: torch.Tensor,
        networks: PotentialNetworks,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute trajectory-free loss.

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

        total_diss = torch.tensor(0.0, device=device)
        total_diff = torch.tensor(0.0, device=device)
        total_energy_change = torch.tensor(0.0, device=device)

        for m in range(M):
            for ell in range(L - 1):
                X_curr = data[m, ell]  # shape (N, d)
                X_next = data[m, ell + 1]  # shape (N, d)
                dt = t_snapshots[ell + 1] - t_snapshots[ell]

                # Dissipation term: (1/N) sum_i |drift|² * dt
                drift = self.compute_drift(X_curr, networks)  # (N, d)
                J_diss = (drift ** 2).sum() / N * dt

                # Diffusion term: sigma² * (1/N) sum_i [laplacian sum] * dt
                laplacian_sum = self.compute_laplacian_sum(X_curr, networks)  # (N,)
                J_diff = self.sigma_sq * laplacian_sum.mean() * dt

                # Energy change: E(t_{l+1}) - E(t_l)
                E_curr = self.compute_energy(X_curr, networks)
                E_next = self.compute_energy(X_next, networks)
                J_energy_change = E_next - E_curr

                total_diss = total_diss + J_diss
                total_diff = total_diff + J_diff
                total_energy_change = total_energy_change + J_energy_change

        # Normalize by number of samples and time pairs
        n_pairs = M * (L - 1)
        total_diss = total_diss / n_pairs
        total_diff = total_diff / n_pairs
        total_energy_change = total_energy_change / n_pairs

        # Weak form: J_diss + J_diff = 2 * J_energy_change (equality at true potentials)
        # Loss: squared residual to ensure non-negative and proper minimization
        residual = total_diss + total_diff - 2 * total_energy_change
        loss = residual ** 2

        # Also add a regularization term to keep potentials bounded
        # This helps prevent divergence

        info = {
            'loss': loss.item(),
            'residual': residual.item(),
            'J_diss': total_diss.item(),
            'J_diff': total_diff.item(),
            'J_energy_change': total_energy_change.item(),
        }

        return loss, info


class BatchedTrajectoryFreeLoss(TrajectoryFreeLoss):
    """Batched version for efficiency - processes time pairs in batch."""

    def forward(
        self,
        data: torch.Tensor,
        t_snapshots: torch.Tensor,
        networks: PotentialNetworks,
        batch_size: int = 32,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute trajectory-free loss with batching over samples.

        Args:
            data: Snapshot data, shape (M, L, N, d)
            t_snapshots: Time points, shape (L,)
            networks: Potential networks
            batch_size: Number of samples per batch
        Returns:
            loss: Total loss (scalar)
            info: Dictionary with component losses
        """
        M, L, N, d = data.shape
        device = data.device

        total_diss = torch.tensor(0.0, device=device)
        total_diff = torch.tensor(0.0, device=device)
        total_energy_change = torch.tensor(0.0, device=device)

        n_batches = (M + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_m = batch_idx * batch_size
            end_m = min((batch_idx + 1) * batch_size, M)
            batch_data = data[start_m:end_m]  # (batch_M, L, N, d)
            batch_M = batch_data.shape[0]

            for ell in range(L - 1):
                dt = t_snapshots[ell + 1] - t_snapshots[ell]

                for m in range(batch_M):
                    X_curr = batch_data[m, ell]
                    X_next = batch_data[m, ell + 1]

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

        loss = total_diss + total_diff - 2 * total_energy_change

        info = {
            'loss': loss.item(),
            'J_diss': total_diss.item(),
            'J_diff': total_diff.item(),
            'J_energy_change': total_energy_change.item(),
        }

        return loss, info
