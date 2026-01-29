#!/usr/bin/env python
"""
RKHS Oracle Test: 用真实φ作为唯一basis，验证能否恢复c=1
"""

import numpy as np
from scipy.integrate import simpson
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class GaussianPhi:
    def __init__(self, a=1.0, sigma=1.0):
        self.a = a
        self.sigma = sigma
    
    def __call__(self, r):
        return self.a * np.exp(-r**2 / (2 * self.sigma**2))
    
    def gradient(self, r):
        return -self.a * r / self.sigma**2 * np.exp(-r**2 / (2 * self.sigma**2))


def generate_pde_data(phi_true, x_range=(-5, 5), n_x=200, T=1.0, n_t=100, nu=0.1):
    """生成PDE数据"""
    dx = (x_range[1] - x_range[0]) / (n_x - 1)
    dt = T / n_t
    x_grid = np.linspace(x_range[0], x_range[1], n_x)
    
    # 初始条件
    u = np.exp(-x_grid**2 / 2)
    u = u / simpson(u, dx=dx)
    
    u_data = [u.copy()]
    
    # Laplacian 矩阵
    diag = -2 * np.ones(n_x)
    off_diag = np.ones(n_x - 1)
    L_matrix = (np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / dx**2
    L_matrix[0, :] = 0
    L_matrix[-1, :] = 0
    I_minus_nuL = np.eye(n_x) - nu * dt * L_matrix
    
    for _ in range(1, n_t):
        # K_φ * u
        K_conv = np.zeros(n_x)
        for i, x in enumerate(x_grid):
            diff = x - x_grid
            r = np.abs(diff) + 1e-10
            direction = np.sign(diff)
            grad_phi = phi_true.gradient(r)
            K_conv[i] = simpson(grad_phi * direction * u, dx=dx)
        
        flux = u * K_conv
        div_flux = np.gradient(flux, dx)
        
        rhs = u + dt * div_flux
        rhs[0] = 0
        rhs[-1] = 0
        
        u = np.linalg.solve(I_minus_nuL, rhs)
        u = np.maximum(u, 0)
        u = u / simpson(u, dx=dx)
        
        u_data.append(u.copy())
    
    u_data = np.array(u_data)
    du_dt = np.diff(u_data, axis=0) / dt
    grad_u = np.gradient(u_data, dx, axis=1)
    
    return u_data, du_dt, grad_u, x_grid, dt, dx


def oracle_test(phi_true, u_data, du_dt, grad_u, x_grid, dt, dx, nu):
    """Oracle test: 用φ_true作为唯一basis"""
    n_t, n_x = u_data.shape
    
    A = 0.0  # scalar for single basis
    b = 0.0
    
    for l in range(n_t - 1):
        u = u_data[l]
        
        # 计算 K_φ * u (用真实φ)
        K_conv = np.zeros(n_x)
        for i, x in enumerate(x_grid):
            diff = x - x_grid
            r = np.abs(diff) + 1e-10
            direction = np.sign(diff)
            grad_phi = phi_true.gradient(r)
            K_conv[i] = simpson(grad_phi * direction * u, dx=dx)
        
        # A = ∫ |K_φ * u|² u dx
        A += simpson(K_conv**2 * u, dx=dx) * dt
        
        # 计算 Ψ * u
        Psi_conv = np.zeros(n_x)
        for i, x in enumerate(x_grid):
            diff = x - x_grid
            r = np.abs(diff) + 1e-10
            phi_val = phi_true(r)
            Psi_conv[i] = simpson(phi_val * u, dx=dx)
        
        # b = -∫ [∂_t u (Ψ * u) + ν ∇u · (K_φ * u)] dx
        term1 = simpson(du_dt[l] * Psi_conv, dx=dx)
        term2 = nu * simpson(grad_u[l] * K_conv, dx=dx)
        b -= (term1 + term2) * dt
    
    # 归一化
    A /= (n_t - 1)
    b /= (n_t - 1)
    
    print(f"A = {A:.6e}")
    print(f"b = {b:.6e}")
    print(f"c_opt = b/A = {b/A:.4f}")
    print(f"Expected: c_opt ≈ 1.0")
    
    return b / A


def main():
    print("=" * 60)
    print("RKHS Oracle Test")
    print("=" * 60)
    
    phi_true = GaussianPhi(a=1.0, sigma=1.0)
    
    print("\nGenerating PDE data...")
    u_data, du_dt, grad_u, x_grid, dt, dx = generate_pde_data(
        phi_true, n_x=150, n_t=50, nu=0.1
    )
    print(f"u_data shape: {u_data.shape}")
    
    print("\nRunning Oracle test (single basis = φ_true)...")
    c_opt = oracle_test(phi_true, u_data, du_dt, grad_u, x_grid, dt, dx, nu=0.1)
    
    error = abs(c_opt - 1.0)
    print(f"\nOracle error: |c_opt - 1| = {error:.4f}")
    
    if error < 0.1:
        print("✅ Oracle test PASSED - Error functional 正确")
    else:
        print("❌ Oracle test FAILED - Error functional 有问题")
    
    return c_opt


if __name__ == '__main__':
    main()
