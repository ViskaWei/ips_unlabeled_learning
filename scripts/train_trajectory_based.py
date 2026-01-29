#!/usr/bin/env python
"""
MVP-2.1: Trajectory-based MLE Baseline

与trajectory-free方法对比的baseline：
- 需要轨迹信息（知道哪个粒子从哪里到哪里）
- 使用标准MLE方法估计势函数参数

关键思想：
对于SDE: dX = -∇V(X) dt - ∇Φ*μ(X) dt + σ dW
给定轨迹 {X_t, X_{t+dt}}，估计drift:
    drift_observed = (X_{t+dt} - X_t) / dt
    drift_model = -∇V(X_t) - (1/N)Σ_j ∇Φ(X_t^i - X_t^j)
最小二乘: minimize ||drift_observed - drift_model||²
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class ParametricPhi(nn.Module):
    """参数化的交互势函数 Φ(r)"""
    
    def __init__(self, phi_type='gaussian'):
        super().__init__()
        self.phi_type = phi_type
        
        if phi_type == 'gaussian':
            # Φ(r) = a * exp(-r²/(2σ²))
            self.log_a = nn.Parameter(torch.tensor(0.0))
            self.log_sigma = nn.Parameter(torch.tensor(0.0))
        elif phi_type == 'morse':
            # Φ(r) = D * (1 - exp(-α(r-r₀)))²
            self.log_D = nn.Parameter(torch.tensor(0.0))
            self.log_alpha = nn.Parameter(torch.tensor(0.0))
            self.r0 = nn.Parameter(torch.tensor(1.0))
        elif phi_type == 'polynomial':
            # Φ(r) = c₀ + c₁r + c₂r² + c₃r³
            self.coeffs = nn.Parameter(torch.zeros(4))
        else:
            raise ValueError(f"Unknown phi_type: {phi_type}")
    
    def forward(self, r):
        """计算 Φ(r)"""
        if self.phi_type == 'gaussian':
            a = torch.exp(self.log_a)
            sigma = torch.exp(self.log_sigma)
            return a * torch.exp(-r**2 / (2 * sigma**2))
        elif self.phi_type == 'morse':
            D = torch.exp(self.log_D)
            alpha = torch.exp(self.log_alpha)
            return D * (1 - torch.exp(-alpha * (r - self.r0)))**2
        elif self.phi_type == 'polynomial':
            result = self.coeffs[0]
            for i in range(1, len(self.coeffs)):
                result = result + self.coeffs[i] * r**i
            return result
    
    def gradient(self, r):
        """计算 dΦ/dr"""
        r = r.requires_grad_(True)
        phi = self.forward(r)
        grad = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        return grad


class ParametricV(nn.Module):
    """参数化的外势函数 V(x)"""
    
    def __init__(self, v_type='harmonic'):
        super().__init__()
        self.v_type = v_type
        
        if v_type == 'harmonic':
            # V(x) = k/2 * x²
            self.log_k = nn.Parameter(torch.tensor(0.0))
        elif v_type == 'double_well':
            # V(x) = a*(x² - b)²
            self.log_a = nn.Parameter(torch.tensor(0.0))
            self.b = nn.Parameter(torch.tensor(1.0))
        else:
            raise ValueError(f"Unknown v_type: {v_type}")
    
    def forward(self, x):
        """计算 V(x)"""
        if self.v_type == 'harmonic':
            k = torch.exp(self.log_k)
            return 0.5 * k * (x**2).sum(dim=-1)
        elif self.v_type == 'double_well':
            a = torch.exp(self.log_a)
            return a * ((x**2).sum(dim=-1) - self.b)**2
    
    def gradient(self, x):
        """计算 ∇V(x)"""
        x = x.requires_grad_(True)
        v = self.forward(x)
        grad = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        return grad


class TrajectoryBasedEstimator:
    """基于轨迹的MLE估计器"""
    
    def __init__(self, phi_type='gaussian', v_type='harmonic', 
                 learn_v=True, device='cpu'):
        self.device = device
        self.learn_v = learn_v
        
        self.phi = ParametricPhi(phi_type).to(device)
        if learn_v:
            self.v = ParametricV(v_type).to(device)
        else:
            self.v = None
            
    def compute_model_drift(self, X):
        """
        计算模型预测的drift（向量化版本）
        
        Args:
            X: (batch, N, d) 粒子位置
            
        Returns:
            drift: (batch, N, d) 预测的drift
        """
        batch, N, d = X.shape
        
        # 外势梯度
        if self.learn_v and self.v is not None:
            X_flat = X.reshape(-1, d)
            grad_V = self.v.gradient(X_flat).reshape(batch, N, d)
        else:
            grad_V = torch.zeros_like(X)
        
        # 交互势梯度 (向量化)
        # diff[b, i, j, :] = X[b, i, :] - X[b, j, :]
        diff = X.unsqueeze(2) - X.unsqueeze(1)  # (batch, N, N, d)
        r = torch.norm(diff, dim=-1, keepdim=True) + 1e-8  # (batch, N, N, 1)
        
        # dΦ/dr at each pairwise distance
        r_flat = r.squeeze(-1)  # (batch, N, N)
        dphi_dr = self.phi.gradient(r_flat)  # (batch, N, N)
        
        # 方向向量
        direction = diff / r  # (batch, N, N, d)
        
        # 掩码排除i=j
        mask = 1 - torch.eye(N, device=X.device).unsqueeze(0)  # (1, N, N)
        
        # 交互梯度: sum over j
        grad_Phi = (dphi_dr.unsqueeze(-1) * direction * mask.unsqueeze(-1)).sum(dim=2) / N  # (batch, N, d)
        
        drift = -grad_V - grad_Phi
        return drift
    
    def compute_loss(self, X_t, X_tp, dt):
        """
        计算MLE损失
        
        Args:
            X_t: (batch, N, d) t时刻粒子位置
            X_tp: (batch, N, d) t+dt时刻粒子位置
            dt: 时间步长
            
        Returns:
            loss: 标量
        """
        # 观测到的drift（这里需要轨迹信息！）
        drift_obs = (X_tp - X_t) / dt
        
        # 模型预测的drift
        drift_model = self.compute_model_drift(X_t)
        
        # MSE损失
        loss = ((drift_obs - drift_model)**2).mean()
        return loss
    
    def train(self, trajectories, dt, n_epochs=1000, lr=0.01, 
              verbose=True, true_phi=None, true_v=None):
        """
        训练模型
        
        Args:
            trajectories: (M, L, N, d) 轨迹数据，M个独立样本，L个时间步
            dt: 时间步长
            n_epochs: 训练轮数
            lr: 学习率
            
        Returns:
            history: 训练历史
        """
        trajectories = torch.tensor(trajectories, dtype=torch.float32, device=self.device)
        M, L, N, d = trajectories.shape
        
        # 准备训练数据：所有 (X_t, X_{t+dt}) 对
        X_t = trajectories[:, :-1, :, :].reshape(-1, N, d)
        X_tp = trajectories[:, 1:, :, :].reshape(-1, N, d)
        
        print(f"Training data: {X_t.shape[0]} trajectory pairs")
        
        # 优化器
        params = list(self.phi.parameters())
        if self.learn_v and self.v is not None:
            params += list(self.v.parameters())
        optimizer = optim.Adam(params, lr=lr)
        
        history = {'loss': [], 'phi_error': [], 'v_error': []}
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(X_t, X_tp, dt)
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
            
            # 计算误差（如果提供了真实势函数）
            if true_phi is not None:
                phi_err = self._compute_phi_error(true_phi)
                history['phi_error'].append(phi_err)
            
            if verbose and (epoch + 1) % 100 == 0:
                msg = f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}"
                if true_phi is not None:
                    msg += f", Φ error: {history['phi_error'][-1]:.2%}"
                print(msg)
        
        return history
    
    def _compute_phi_error(self, true_phi, r_range=(0.1, 3.0), n_points=100):
        """计算Φ的相对L²误差"""
        r = torch.linspace(r_range[0], r_range[1], n_points, device=self.device)
        
        with torch.no_grad():
            phi_pred = self.phi(r)
            phi_true = torch.tensor([true_phi(ri.item()) for ri in r], 
                                   dtype=torch.float32, device=self.device)
        
        error = torch.sqrt(((phi_pred - phi_true)**2).mean()) / (torch.abs(phi_true).mean() + 1e-8)
        return error.item()
    
    def get_params(self):
        """获取学习到的参数"""
        params = {}
        
        if self.phi.phi_type == 'gaussian':
            params['phi_a'] = torch.exp(self.phi.log_a).item()
            params['phi_sigma'] = torch.exp(self.phi.log_sigma).item()
        elif self.phi.phi_type == 'morse':
            params['phi_D'] = torch.exp(self.phi.log_D).item()
            params['phi_alpha'] = torch.exp(self.phi.log_alpha).item()
            params['phi_r0'] = self.phi.r0.item()
        
        if self.learn_v and self.v is not None:
            if self.v.v_type == 'harmonic':
                params['v_k'] = torch.exp(self.v.log_k).item()
        
        return params


def generate_trajectory_data(N=10, d=1, L=50, M=100, dt=0.01, sigma=0.1,
                             phi_true=None, v_true=None, seed=42):
    """
    生成带轨迹标签的数据（向量化版本）
    
    关键：记录每个粒子的完整轨迹（而不是只有snapshots）
    """
    np.random.seed(seed)
    
    trajectories = np.zeros((M, L, N, d))
    
    # 批量初始化
    X_all = np.random.randn(M, N, d) * 0.5
    trajectories[:, 0] = X_all
    
    for l in range(1, L):
        # 计算drift (向量化)
        grad_V = np.zeros((M, N, d))
        if v_true is not None:
            grad_V = v_true.gradient(X_all.reshape(-1, d)).reshape(M, N, d)
        
        grad_Phi = np.zeros((M, N, d))
        if phi_true is not None:
            # 向量化计算pairwise距离和梯度
            # X_all: (M, N, d)
            # diff[m, i, j, :] = X_all[m, i, :] - X_all[m, j, :]
            diff = X_all[:, :, np.newaxis, :] - X_all[:, np.newaxis, :, :]  # (M, N, N, d)
            r = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-10  # (M, N, N, 1)
            
            # dphi_dr: (M, N, N)
            r_flat = r.squeeze(-1)  # (M, N, N)
            dphi_dr = phi_true.gradient(r_flat)  # (M, N, N)
            
            # 梯度方向: diff / r
            direction = diff / r  # (M, N, N, d)
            
            # 交互梯度: sum over j
            # 排除i=j（对角线）
            mask = 1 - np.eye(N)[np.newaxis, :, :]  # (1, N, N)
            grad_Phi = (dphi_dr[:, :, :, np.newaxis] * direction * mask[:, :, :, np.newaxis]).sum(axis=2) / N  # (M, N, d)
        
        drift = -grad_V - grad_Phi
        noise = sigma * np.sqrt(dt) * np.random.randn(M, N, d)
        X_all = X_all + drift * dt + noise
        trajectories[:, l] = X_all
    
    return trajectories


class GaussianPhi:
    """真实的Gaussian势函数（支持向量化输入）"""
    def __init__(self, a=1.0, sigma=1.0):
        self.a = a
        self.sigma = sigma
    
    def __call__(self, r):
        return self.a * np.exp(-r**2 / (2 * self.sigma**2))
    
    def gradient(self, r):
        """dΦ/dr - 支持任意形状的r"""
        return -self.a * r / self.sigma**2 * np.exp(-r**2 / (2 * self.sigma**2))


class HarmonicV:
    """真实的Harmonic势函数（支持向量化输入）"""
    def __init__(self, k=1.0):
        self.k = k
    
    def __call__(self, x):
        return 0.5 * self.k * np.sum(x**2, axis=-1)
    
    def gradient(self, x):
        """∇V - 支持任意形状，最后一维是空间维度"""
        return self.k * x


def plot_results(estimator, true_phi, history, save_path):
    """绘制结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. 训练曲线
    ax = axes[0]
    ax.semilogy(history['loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True)
    
    # 2. Φ对比
    ax = axes[1]
    r = np.linspace(0.1, 3.0, 100)
    r_tensor = torch.tensor(r, dtype=torch.float32)
    
    with torch.no_grad():
        phi_pred = estimator.phi(r_tensor).numpy()
    phi_true = np.array([true_phi(ri) for ri in r])
    
    ax.plot(r, phi_true, 'b-', lw=2, label='True Φ')
    ax.plot(r, phi_pred, 'r--', lw=2, label='Learned Φ')
    ax.set_xlabel('r')
    ax.set_ylabel('Φ(r)')
    ax.set_title('Interaction Potential')
    ax.legend()
    ax.grid(True)
    
    # 3. Φ误差
    if history.get('phi_error'):
        ax = axes[2]
        ax.semilogy(history['phi_error'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Relative L² Error')
        ax.set_title(f'Φ Error (final: {history["phi_error"][-1]:.2%})')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved results to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--L', type=int, default=100)
    parser.add_argument('--M', type=int, default=200)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learn_v', action='store_true')
    parser.add_argument('--output', type=str, default='results/trajectory_based')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MVP-2.1: Trajectory-based MLE Baseline")
    print("=" * 60)
    
    # 真实势函数
    true_phi = GaussianPhi(a=1.0, sigma=1.0)
    true_v = HarmonicV(k=1.0)
    
    print(f"\nTrue parameters:")
    print(f"  Φ(r) = {true_phi.a} * exp(-r²/(2*{true_phi.sigma}²))")
    print(f"  V(x) = 0.5 * {true_v.k} * x²")
    
    # 生成轨迹数据
    print(f"\nGenerating trajectory data...")
    print(f"  N={args.N}, L={args.L}, M={args.M}, dt={args.dt}, σ={args.sigma}")
    
    trajectories = generate_trajectory_data(
        N=args.N, d=args.d, L=args.L, M=args.M,
        dt=args.dt, sigma=args.sigma,
        phi_true=true_phi, v_true=true_v,
        seed=args.seed
    )
    print(f"  Data shape: {trajectories.shape}")
    
    # 创建估计器
    estimator = TrajectoryBasedEstimator(
        phi_type='gaussian',
        v_type='harmonic',
        learn_v=args.learn_v
    )
    
    # 训练
    print(f"\nTraining...")
    history = estimator.train(
        trajectories, args.dt,
        n_epochs=args.epochs,
        lr=args.lr,
        true_phi=true_phi
    )
    
    # 结果
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    params = estimator.get_params()
    print(f"\nLearned parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nTrue vs Learned:")
    print(f"  Φ amplitude: {true_phi.a:.4f} vs {params['phi_a']:.4f}")
    print(f"  Φ width:     {true_phi.sigma:.4f} vs {params['phi_sigma']:.4f}")
    
    final_error = history['phi_error'][-1] if history.get('phi_error') else None
    print(f"\nFinal Φ relative error: {final_error:.2%}" if final_error else "")
    
    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_results(estimator, true_phi, history, 
                 output_dir / 'trajectory_based_results.png')
    
    # 保存数值结果
    np.savez(
        output_dir / 'trajectory_based_results.npz',
        history_loss=history['loss'],
        history_phi_error=history.get('phi_error', []),
        params=params,
        config=vars(args)
    )
    
    print(f"\n✅ Baseline complete!")
    print(f"   Final Φ error: {final_error:.2%}" if final_error else "")
    
    return final_error


if __name__ == '__main__':
    main()
