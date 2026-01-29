#!/usr/bin/env python
"""
简化的 RKHS: 用 Gaussian basis 系列，参数化为 φ_j(r) = exp(-r²/(2σ_j²))
目标是学习正确的组合系数
"""

import numpy as np
from scipy.integrate import simpson
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
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


class GaussianBasis:
    """Gaussian basis: φ_j(r) = exp(-r²/(2σ_j²))"""
    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas)
        self.n_basis = len(sigmas)
    
    def __call__(self, r, j=None):
        """返回 φ_j(r) 或所有 basis 的值"""
        r = np.atleast_1d(r)
        if j is not None:
            return np.exp(-r**2 / (2 * self.sigmas[j]**2))
        else:
            # 返回 (n_r, n_basis)
            result = np.zeros((len(r), self.n_basis))
            for j in range(self.n_basis):
                result[:, j] = np.exp(-r**2 / (2 * self.sigmas[j]**2))
            return result
    
    def gradient(self, r, j=None):
        """返回 dφ_j/dr"""
        r = np.atleast_1d(r)
        if j is not None:
            sigma = self.sigmas[j]
            return -r / sigma**2 * np.exp(-r**2 / (2 * sigma**2))
        else:
            result = np.zeros((len(r), self.n_basis))
            for j in range(self.n_basis):
                sigma = self.sigmas[j]
                result[:, j] = -r / sigma**2 * np.exp(-r**2 / (2 * sigma**2))
            return result


def generate_pde_data(phi_true, x_range=(-5, 5), n_x=200, T=1.0, n_t=100, nu=0.1):
    """生成PDE数据"""
    dx = (x_range[1] - x_range[0]) / (n_x - 1)
    dt = T / n_t
    x_grid = np.linspace(x_range[0], x_range[1], n_x)
    
    u = np.exp(-x_grid**2 / 2)
    u = u / simpson(u, dx=dx)
    
    u_data = [u.copy()]
    
    diag = -2 * np.ones(n_x)
    off_diag = np.ones(n_x - 1)
    L_matrix = (np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / dx**2
    L_matrix[0, :] = 0
    L_matrix[-1, :] = 0
    I_minus_nuL = np.eye(n_x) - nu * dt * L_matrix
    
    for _ in range(1, n_t):
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


def fit_gaussian_basis(basis, u_data, du_dt, grad_u, x_grid, dt, dx, nu, lambda_reg=0.01):
    """拟合 Gaussian basis 系数"""
    n_t, n_x = u_data.shape
    n_basis = basis.n_basis
    
    A = np.zeros((n_basis, n_basis))
    b = np.zeros(n_basis)
    
    for l in range(n_t - 1):
        u = u_data[l]
        
        # 对每个 basis 计算 K_φ_j * u
        K_conv_basis = np.zeros((n_x, n_basis))
        for i, x in enumerate(x_grid):
            diff = x - x_grid
            r = np.abs(diff) + 1e-10
            direction = np.sign(diff)
            grad_phi_basis = basis.gradient(r)  # (n_x, n_basis)
            for j in range(n_basis):
                K_conv_basis[i, j] = simpson(grad_phi_basis[:, j] * direction * u, dx=dx)
        
        # A_jk = ∫ (K_φ_j * u)(K_φ_k * u) u dx
        for j in range(n_basis):
            for k in range(n_basis):
                A[j, k] += simpson(K_conv_basis[:, j] * K_conv_basis[:, k] * u, dx=dx) * dt
        
        # b_j = -∫ [∂_t u (Ψ_j * u) + ν ∇u · (K_φ_j * u)] dx
        for j in range(n_basis):
            # Ψ_j * u
            Psi_conv = np.zeros(n_x)
            phi_j_vals = basis(np.abs(x_grid[:, np.newaxis] - x_grid), j)  # (n_x, n_x)
            for i in range(n_x):
                r = np.abs(x_grid[i] - x_grid) + 1e-10
                phi_j = basis(r, j)
                Psi_conv[i] = simpson(phi_j * u, dx=dx)
            
            term1 = simpson(du_dt[l] * Psi_conv, dx=dx)
            term2 = nu * simpson(grad_u[l] * K_conv_basis[:, j], dx=dx)
            b[j] -= (term1 + term2) * dt
    
    A /= (n_t - 1)
    b /= (n_t - 1)
    
    print(f"A diagonal: {np.diag(A)}")
    print(f"b vector: {b}")
    print(f"cond(A): {np.linalg.cond(A):.2e}")
    
    # Tikhonov 正则化
    A_reg = A + lambda_reg * np.eye(n_basis)
    coeffs = np.linalg.solve(A_reg, b)
    
    return coeffs, A, b


def main():
    print("=" * 60)
    print("Gaussian Basis RKHS (Simple Version)")
    print("=" * 60)
    
    # 真实势函数: σ = 1.0
    phi_true = GaussianPhi(a=1.0, sigma=1.0)
    
    # Gaussian basis: 包含真实σ=1.0
    sigmas = [0.5, 0.75, 1.0, 1.25, 1.5]  # 真实的在中间
    basis = GaussianBasis(sigmas)
    
    print(f"\nTrue: Φ(r) = exp(-r²/2) (σ=1.0)")
    print(f"Basis sigmas: {sigmas}")
    print(f"Expected coefficients: [0, 0, 1, 0, 0]")
    
    # 生成数据
    print("\nGenerating PDE data...")
    u_data, du_dt, grad_u, x_grid, dt, dx = generate_pde_data(
        phi_true, n_x=150, n_t=50, nu=0.1
    )
    
    # 拟合
    print("\nFitting Gaussian basis...")
    for lambda_reg in [0.0001, 0.00005, 0.00001, 0.000005]:
        print(f"\n--- λ = {lambda_reg} ---")
        coeffs, A, b = fit_gaussian_basis(
            basis, u_data, du_dt, grad_u, x_grid, dt, dx, nu=0.1, lambda_reg=lambda_reg
        )
        print(f"Coefficients: {coeffs}")
        
        # 计算φ误差
        r = np.linspace(0.1, 3.0, 100)
        phi_pred = sum(coeffs[j] * basis(r, j) for j in range(len(sigmas)))
        phi_true_vals = phi_true(r)
        error = np.sqrt(np.mean((phi_pred - phi_true_vals)**2)) / np.mean(np.abs(phi_true_vals))
        print(f"Φ error: {error:.2%}")
    
    # 绘图
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    r = np.linspace(0.1, 3.0, 100)
    plt.plot(r, phi_true(r), 'b-', lw=2, label='True Φ')
    
    # 用最后一个λ的结果
    phi_pred = sum(coeffs[j] * basis(r, j) for j in range(len(sigmas)))
    plt.plot(r, phi_pred, 'r--', lw=2, label='Learned Φ')
    plt.xlabel('r')
    plt.ylabel('Φ(r)')
    plt.legend()
    plt.title('Interaction Potential')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(coeffs)), coeffs)
    plt.xticks(range(len(coeffs)), [f'σ={s}' for s in sigmas])
    plt.xlabel('Basis')
    plt.ylabel('Coefficient')
    plt.title('Expected: [0,0,1,0,0]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/rkhs/gaussian_basis_results.png', dpi=150)
    plt.close()
    print("\nSaved results to results/rkhs/gaussian_basis_results.png")


if __name__ == '__main__':
    main()
