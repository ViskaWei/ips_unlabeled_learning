"""Potential functions for IPS simulation."""

import numpy as np
from abc import ABC, abstractmethod


class Potential(ABC):
    """Base class for potential functions."""

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate potential at x."""
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of potential at x."""
        pass


class HarmonicPotential(Potential):
    """Harmonic potential V(x) = 0.5 * k * x^2."""

    def __init__(self, k: float = 1.0):
        self.k = k

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * self.k * np.sum(x**2, axis=-1)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.k * x


class ZeroPotential(Potential):
    """Zero potential (no interaction)."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[:-1]) if x.ndim > 1 else 0.0

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class GaussianInteraction(Potential):
    """Gaussian interaction potential Phi(r) = A * exp(-r^2 / (2*sigma^2))."""

    def __init__(self, A: float = 1.0, sigma: float = 1.0):
        self.A = A
        self.sigma = sigma

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return self.A * np.exp(-r**2 / (2 * self.sigma**2))

    def gradient(self, r: np.ndarray) -> np.ndarray:
        """Gradient w.r.t. r (scalar distance)."""
        return -self.A * r / (self.sigma**2) * np.exp(-r**2 / (2 * self.sigma**2))


class MorsePotential(Potential):
    """Morse potential Phi(r) = D * (1 - exp(-a*(r-r0)))^2."""

    def __init__(self, D: float = 1.0, a: float = 1.0, r0: float = 1.0):
        self.D = D
        self.a = a
        self.r0 = r0

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return self.D * (1 - np.exp(-self.a * (r - self.r0)))**2

    def gradient(self, r: np.ndarray) -> np.ndarray:
        exp_term = np.exp(-self.a * (r - self.r0))
        return 2 * self.D * self.a * (1 - exp_term) * exp_term
