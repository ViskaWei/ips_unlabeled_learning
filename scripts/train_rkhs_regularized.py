#!/usr/bin/env python
"""
MVP-2.2: RKHS Regularized Trajectory-free Learning

基于 Fei Lu 论文实现 RKHS Tikhonov 正则化：
- Data-adaptive RKHS 而非固定 B-spline
- 通过正则化保证解的唯一性
- 解决 identifiability 问题

参考: Lang & Lu, SIAM J. Sci. Comput. 2022 (Section 3)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from scipy.integrate import simpson
from scipy.linalg import lstsq, svd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class RKHSKernel:
    """Reproducing Kernel for RKHS
    
    使用 Gaussian RBF kernel: K(r, r') = exp(-|r-r'|²/(2h²))
    """
    def __init__(self, bandwidth=0.5, centers=None):
        self.h = bandwidth
        self.centers = centers  # (n_basis,) - kernel centers
        
    def __call__(self, r, r_prime=None):
        """Compute kernel matrix K(r, r') or K(r, centers)"""
        if r_prime is None:
            r_prime = self.centers
        
        # r: (..., ) or (..., n1)
        # r_prime: (n2,)
        r = np.atleast_1d(r)
        r_prime = np.atleast_1d(r_prime)
        
        # Broadcast to compute pairwise distances
        diff = r[..., np.newaxis] - r_prime  # (..., n2)
        K = np.exp(-diff**2 / (2 * self.h**2))
        return K
    
    def gradient(self, r):
        """Compute gradient of kernel basis functions: d/dr K(r, c_j)"""
        # dK/dr = K(r, c) * (c - r) / h²
        r = np.atleast_1d(r)
        diff = self.centers - r[..., np.newaxis]  # (..., n_basis)
        K = self(r)  # (..., n_basis)
        return K * diff / self.h**2


class FeiLuRKHS:
    """Fei Lu 方法 + RKHS 正则化
    
    Error Functional (Eq 2.3):
    E(ψ) = (1/T) ∫∫ [|K_ψ*u|² u + 2∂_t u (Ψ*u) + 2ν ∇u·(K_ψ*u)] dx dt
    
    其中 K_ψ = ∇ψ (interaction kernel)
    """
    
    def __init__(self, n_basis=20, bandwidth=0.5, r_max=5.0, 
                 lambda_reg=1e-4, nu=0.1):
        """
        Args:
            n_basis: RKHS basis 数量
            bandwidth: Gaussian kernel 带宽
            r_max: 最大距离
            lambda_reg: Tikhonov 正则化参数
            nu: 粘性系数 (= σ²/2)
        """
        self.n_basis = n_basis
        self.bandwidth = bandwidth
        self.r_max = r_max
        self.lambda_reg = lambda_reg
        self.nu = nu
        
        # 设置 kernel centers (均匀分布在 [0, r_max])
        self.centers = np.linspace(0.1, r_max, n_basis)
        self.kernel = RKHSKernel(bandwidth, self.centers)
        
        # 系数
        self.coeffs = None
        
    def phi_from_coeffs(self, r, coeffs):
        """从系数计算 φ(r) = Σ c_j K(r, c_j)"""
        K = self.kernel(r)  # (n_r, n_basis)
        return K @ coeffs
    
    def grad_phi_from_coeffs(self, r, coeffs):
        """从系数计算 ∇φ(r) = Σ c_j ∇K(r, c_j)"""
        grad_K = self.kernel.gradient(r)  # (n_r, n_basis)
        return grad_K @ coeffs
    
    def compute_convolution(self, u, x_grid, phi_coeffs):
        """计算 K_φ * u 卷积
        
        K_φ * u(x) = ∫ ∇φ(|x-y|) · (x-y)/|x-y| u(y) dy
        """
        dx = x_grid[1] - x_grid[0]
        n_x = len(x_grid)
        
        result = np.zeros(n_x)
        for i, x in enumerate(x_grid):
            # 计算 x 与所有 y 的距离
            diff = x - x_grid  # (n_x,)
            r = np.abs(diff) + 1e-10
            direction = np.sign(diff)
            
            # ∇φ(r) * direction
            grad_phi = self.grad_phi_from_coeffs(r, phi_coeffs)
            
            # 积分
            result[i] = simpson(grad_phi * direction * u, dx=dx)
        
        return result
    
    def compute_error_functional_matrix(self, u_data, du_dt, grad_u, x_grid, dt):
        """计算离散化的 Error Functional 矩阵形式
        
        E(ψ) = c^T A c - 2 b^T c
        
        其中 A, b 从 u 数据计算
        """
        dx = x_grid[1] - x_grid[0]
        M, L = u_data.shape[0], u_data.shape[1]
        
        # 预计算 basis 在所有 pairwise distances 上的值
        # 对于每个 x, y 对，计算 K(|x-y|, centers) 和 ∇K
        n_x = len(x_grid)
        
        # A 矩阵: n_basis x n_basis
        A = np.zeros((self.n_basis, self.n_basis))
        # b 向量: n_basis
        b = np.zeros(self.n_basis)
        
        for m in range(M):
            for l in range(L):
                u = u_data[m, l]  # (n_x,)
                
                # 对每个 basis function 计算 K_φ_j * u
                K_phi_basis = np.zeros((n_x, self.n_basis))
                
                for i, x in enumerate(x_grid):
                    diff = x - x_grid
                    r = np.abs(diff) + 1e-10
                    direction = np.sign(diff)
                    
                    # ∇K(r, centers) for each basis
                    grad_K = self.kernel.gradient(r)  # (n_x, n_basis)
                    
                    # 对每个 basis，计算卷积
                    for j in range(self.n_basis):
                        K_phi_basis[i, j] = simpson(grad_K[:, j] * direction * u, dx=dx)
                
                # A_jk = ∫ (K_φ_j * u) (K_φ_k * u) u dx
                for j in range(self.n_basis):
                    for k in range(self.n_basis):
                        A[j, k] += simpson(K_phi_basis[:, j] * K_phi_basis[:, k] * u, dx=dx) * dt
                
                # b_j = -∫ [∂_t u (Ψ_j * u) + ν ∇u · (K_φ_j * u)] dx
                # 这里 Ψ_j * u = ∫ φ_j(|x-y|) u(y) dy
                for j in range(self.n_basis):
                    # 计算 Ψ_j * u
                    Psi_conv = np.zeros(n_x)
                    for i, x in enumerate(x_grid):
                        diff = x - x_grid
                        r = np.abs(diff) + 1e-10
                        phi_j = self.kernel(r)[:, j]
                        Psi_conv[i] = simpson(phi_j * u, dx=dx)
                    
                    # ∂_t u term
                    if l < L - 1:
                        du_dt_l = du_dt[m, l]
                        term1 = simpson(du_dt_l * Psi_conv, dx=dx)
                    else:
                        term1 = 0
                    
                    # ∇u · K_φ term
                    grad_u_l = grad_u[m, l]
                    term2 = self.nu * simpson(grad_u_l * K_phi_basis[:, j], dx=dx)
                    
                    b[j] -= (term1 + term2) * dt
        
        # 归一化
        A /= (M * L)
        b /= (M * L)
        
        return A, b
    
    def fit(self, u_data, du_dt, grad_u, x_grid, dt, verbose=True):
        """拟合 RKHS 模型
        
        Args:
            u_data: (M, L, n_x) - 密度数据
            du_dt: (M, L-1, n_x) - 时间导数
            grad_u: (M, L, n_x) - 空间梯度
            x_grid: (n_x,) - 空间网格
            dt: 时间步长
        """
        if verbose:
            print("Computing error functional matrix A, b...")
        
        A, b = self.compute_error_functional_matrix(u_data, du_dt, grad_u, x_grid, dt)
        
        if verbose:
            print(f"A shape: {A.shape}, cond(A): {np.linalg.cond(A):.2e}")
            print(f"A diagonal: {np.diag(A)}")
            print(f"b vector: {b}")
            print(f"A.max(): {A.max():.2e}, b.max(): {np.abs(b).max():.2e}")
        
        # Tikhonov 正则化: (A + λI) c = b
        # 等价于最小化 ||Ac - b||² + λ||c||²
        A_reg = A + self.lambda_reg * np.eye(self.n_basis)
        
        # SVD 求解
        U, s, Vt = svd(A_reg)
        
        if verbose:
            print(f"Singular values: {s[:5]}...")
            print(f"Condition number (regularized): {s[0]/s[-1]:.2e}")
        
        # 截断 SVD 求解
        tol = 1e-10 * s[0]
        s_inv = np.where(s > tol, 1/s, 0)
        self.coeffs = Vt.T @ np.diag(s_inv) @ U.T @ b
        
        if verbose:
            print(f"Coefficients: {self.coeffs[:5]}...")
        
        return self.coeffs
    
    def predict(self, r):
        """预测 φ(r)"""
        if self.coeffs is None:
            raise ValueError("Model not fitted yet")
        return self.phi_from_coeffs(r, self.coeffs)


def generate_pde_data(phi_true, x_range=(-5, 5), n_x=200, T=1.0, n_t=100,
                       nu=0.1, u0_std=1.0, seed=42):
    """生成 PDE 数据 (直接求解 PDE，不是从粒子估计)
    
    ∂_t u = ν Δu + ∇·[u (K_φ * u)]
    
    使用 semi-implicit 方案求解
    """
    np.random.seed(seed)
    
    dx = (x_range[1] - x_range[0]) / (n_x - 1)
    dt = T / n_t
    x_grid = np.linspace(x_range[0], x_range[1], n_x)
    
    # 初始条件: Gaussian
    u = np.exp(-x_grid**2 / (2 * u0_std**2))
    u = u / simpson(u, dx=dx)  # 归一化
    
    # 存储数据
    u_data = np.zeros((n_t, n_x))
    u_data[0] = u
    
    # Laplacian 矩阵 (用于 implicit 扩散)
    diag = -2 * np.ones(n_x)
    off_diag = np.ones(n_x - 1)
    L_matrix = (np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / dx**2
    
    # 边界条件 (Dirichlet: u=0 at boundaries)
    L_matrix[0, :] = 0
    L_matrix[-1, :] = 0
    
    # Identity - ν*dt*L for implicit step
    I_minus_nuL = np.eye(n_x) - nu * dt * L_matrix
    
    for t_idx in range(1, n_t):
        # 计算 K_φ * u (交互项)
        K_conv = np.zeros(n_x)
        for i, x in enumerate(x_grid):
            diff = x - x_grid
            r = np.abs(diff) + 1e-10
            direction = np.sign(diff)
            grad_phi = phi_true.gradient(r)
            K_conv[i] = simpson(grad_phi * direction * u, dx=dx)
        
        # ∇·[u K_conv] = (u K_conv)' (1D)
        flux = u * K_conv
        div_flux = np.gradient(flux, dx)
        
        # Semi-implicit: (I - ν dt L) u^{n+1} = u^n + dt ∇·[u^n K^n]
        rhs = u + dt * div_flux
        rhs[0] = 0  # boundary
        rhs[-1] = 0
        
        u = np.linalg.solve(I_minus_nuL, rhs)
        u = np.maximum(u, 0)  # 保证非负
        u = u / simpson(u, dx=dx)  # 保持归一化
        
        u_data[t_idx] = u
    
    # 计算导数
    du_dt = np.diff(u_data, axis=0) / dt  # (n_t-1, n_x)
    grad_u = np.gradient(u_data, dx, axis=1)  # (n_t, n_x)
    
    # 添加 batch 维度 (M=1 for clean PDE data)
    u_data = u_data[np.newaxis, ...]  # (1, n_t, n_x)
    du_dt = du_dt[np.newaxis, ...]
    grad_u = grad_u[np.newaxis, ...]
    
    return u_data, du_dt, grad_u, x_grid, dt


class GaussianPhi:
    """真实的 Gaussian 势函数"""
    def __init__(self, a=1.0, sigma=1.0):
        self.a = a
        self.sigma = sigma
    
    def __call__(self, r):
        return self.a * np.exp(-r**2 / (2 * self.sigma**2))
    
    def gradient(self, r):
        return -self.a * r / self.sigma**2 * np.exp(-r**2 / (2 * self.sigma**2))


def compute_phi_error(phi_pred, phi_true, r_range=(0.1, 3.0), n_points=100):
    """计算 φ 的相对 L² 误差"""
    r = np.linspace(r_range[0], r_range[1], n_points)
    pred = phi_pred(r)
    true = phi_true(r)
    error = np.sqrt(np.mean((pred - true)**2)) / (np.mean(np.abs(true)) + 1e-8)
    return error


def plot_results(model, phi_true, x_grid, u_data, save_path):
    """绘制结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. φ 对比
    ax = axes[0]
    r = np.linspace(0.1, 3.0, 100)
    phi_pred = model.predict(r)
    phi_true_vals = phi_true(r)
    
    ax.plot(r, phi_true_vals, 'b-', lw=2, label='True Φ')
    ax.plot(r, phi_pred, 'r--', lw=2, label='RKHS Φ')
    ax.set_xlabel('r')
    ax.set_ylabel('Φ(r)')
    ax.set_title('Interaction Potential')
    ax.legend()
    ax.grid(True)
    
    # 2. 密度演化
    ax = axes[1]
    n_t = u_data.shape[1]
    times = [0, n_t//4, n_t//2, 3*n_t//4, n_t-1]
    for t_idx in times:
        ax.plot(x_grid, u_data[0, t_idx], label=f't={t_idx}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Density Evolution')
    ax.legend()
    ax.grid(True)
    
    # 3. 系数
    ax = axes[2]
    ax.bar(range(len(model.coeffs)), model.coeffs)
    ax.set_xlabel('Basis index')
    ax.set_ylabel('Coefficient')
    ax.set_title('RKHS Coefficients')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved results to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_basis', type=int, default=15)
    parser.add_argument('--bandwidth', type=float, default=0.5)
    parser.add_argument('--lambda_reg', type=float, default=1e-3)
    parser.add_argument('--nu', type=float, default=0.1)
    parser.add_argument('--n_x', type=int, default=150)
    parser.add_argument('--n_t', type=int, default=50)
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/rkhs')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MVP-2.2: RKHS Regularized Trajectory-free Learning")
    print("=" * 60)
    
    # 真实势函数
    phi_true = GaussianPhi(a=1.0, sigma=1.0)
    
    print(f"\nTrue potential: Φ(r) = exp(-r²/2)")
    print(f"Config: n_basis={args.n_basis}, bandwidth={args.bandwidth}, λ={args.lambda_reg}")
    
    # 生成 PDE 数据
    print(f"\nGenerating PDE data (nu={args.nu}, n_x={args.n_x}, n_t={args.n_t})...")
    u_data, du_dt, grad_u, x_grid, dt = generate_pde_data(
        phi_true,
        n_x=args.n_x,
        n_t=args.n_t,
        T=args.T,
        nu=args.nu,
        seed=args.seed
    )
    print(f"  u_data shape: {u_data.shape}")
    
    # 创建 RKHS 模型
    model = FeiLuRKHS(
        n_basis=args.n_basis,
        bandwidth=args.bandwidth,
        lambda_reg=args.lambda_reg,
        nu=args.nu
    )
    
    # 拟合
    print(f"\nFitting RKHS model...")
    model.fit(u_data, du_dt, grad_u, x_grid, dt)
    
    # 评估
    error = compute_phi_error(model.predict, phi_true)
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\nΦ relative L² error: {error:.2%}")
    
    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_results(model, phi_true, x_grid, u_data, output_dir / 'rkhs_results.png')
    
    np.savez(
        output_dir / 'rkhs_results.npz',
        coeffs=model.coeffs,
        centers=model.centers,
        error=error,
        config=vars(args)
    )
    
    print(f"\n✅ RKHS fitting complete!")
    print(f"   Φ error: {error:.2%}")
    
    return error


if __name__ == '__main__':
    main()
