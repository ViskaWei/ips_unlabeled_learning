"""True potential functions for evaluation."""

import numpy as np
import torch


def true_V_harmonic(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    """True kinetic potential: V(x) = 0.5 * k * x^2."""
    return 0.5 * k * np.sum(x**2, axis=-1)


def true_grad_V_harmonic(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Gradient of harmonic potential: grad V = k * x."""
    return k * x


def true_Phi_gaussian(r: np.ndarray, A: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """True interaction potential: Phi(r) = A * exp(-r^2 / (2*sigma^2))."""
    return A * np.exp(-r**2 / (2 * sigma**2))


def true_grad_Phi_gaussian(r: np.ndarray, A: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Gradient of Gaussian interaction: dPhi/dr = -A * r / sigma^2 * exp(...)."""
    return -A * r / (sigma**2) * np.exp(-r**2 / (2 * sigma**2))


def true_V_harmonic_torch(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """True kinetic potential (torch version)."""
    return 0.5 * k * (x**2).sum(dim=-1)


def true_Phi_gaussian_torch(r: torch.Tensor, A: float = 1.0, sigma: float = 1.0) -> torch.Tensor:
    """True interaction potential (torch version)."""
    return A * torch.exp(-r**2 / (2 * sigma**2))


def compute_l2_error(
    pred_func,
    true_func,
    x_range: np.ndarray,
) -> float:
    """Compute relative L2 error between predicted and true functions.

    Args:
        pred_func: Predicted function
        true_func: True function
        x_range: Points to evaluate on
    Returns:
        Relative L2 error
    """
    pred_vals = pred_func(x_range)
    true_vals = true_func(x_range)

    l2_diff = np.sqrt(np.mean((pred_vals - true_vals)**2))
    l2_true = np.sqrt(np.mean(true_vals**2))

    if l2_true < 1e-10:
        return float('inf')

    return l2_diff / l2_true


def evaluate_V_error(
    networks,
    x_range: np.ndarray,
    k: float = 1.0,
    device: str = 'cpu',
) -> dict:
    """Evaluate error on kinetic potential V.

    Args:
        networks: Trained networks
        x_range: 1D array of x values
        k: Harmonic constant
        device: Device
    Returns:
        Dictionary with errors and values
    """
    networks.eval()

    # Prepare input
    if networks.d == 1:
        x_torch = torch.tensor(x_range.reshape(-1, 1), dtype=torch.float32, device=device)
    else:
        x_torch = torch.tensor(x_range, dtype=torch.float32, device=device)

    with torch.no_grad():
        V_pred = networks.V(x_torch).cpu().numpy()

    V_true = true_V_harmonic(x_range.reshape(-1, 1) if networks.d == 1 else x_range, k=k)

    # Remove constant offset (V is defined up to a constant)
    V_pred_centered = V_pred - V_pred.mean()
    V_true_centered = V_true - V_true.mean()

    l2_diff = np.sqrt(np.mean((V_pred_centered - V_true_centered)**2))
    l2_true = np.sqrt(np.mean(V_true_centered**2))
    rel_error = l2_diff / l2_true if l2_true > 1e-10 else float('inf')

    return {
        'V_pred': V_pred,
        'V_true': V_true,
        'V_pred_centered': V_pred_centered,
        'V_true_centered': V_true_centered,
        'l2_error_abs': l2_diff,
        'l2_error_rel': rel_error,
    }


def evaluate_Phi_error(
    networks,
    r_range: np.ndarray,
    A: float = 1.0,
    sigma: float = 1.0,
    device: str = 'cpu',
) -> dict:
    """Evaluate error on interaction potential Phi.

    Args:
        networks: Trained networks
        r_range: 1D array of r values (distances)
        A: Gaussian amplitude
        sigma: Gaussian width
        device: Device
    Returns:
        Dictionary with errors and values
    """
    networks.eval()

    r_torch = torch.tensor(r_range, dtype=torch.float32, device=device)

    with torch.no_grad():
        Phi_pred = networks.Phi(r_torch).cpu().numpy()

    Phi_true = true_Phi_gaussian(r_range, A=A, sigma=sigma)

    # Remove constant offset
    Phi_pred_centered = Phi_pred - Phi_pred.mean()
    Phi_true_centered = Phi_true - Phi_true.mean()

    l2_diff = np.sqrt(np.mean((Phi_pred_centered - Phi_true_centered)**2))
    l2_true = np.sqrt(np.mean(Phi_true_centered**2))
    rel_error = l2_diff / l2_true if l2_true > 1e-10 else float('inf')

    return {
        'Phi_pred': Phi_pred,
        'Phi_true': Phi_true,
        'Phi_pred_centered': Phi_pred_centered,
        'Phi_true_centered': Phi_true_centered,
        'l2_error_abs': l2_diff,
        'l2_error_rel': rel_error,
    }
