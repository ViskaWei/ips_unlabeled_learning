# Vectorized Trajectory-Free Loss

## Problem

原始 `TrajectoryFreeLoss.forward()` 有双层循环：

```python
for m in range(M):           # M=200 samples
    for ell in range(L-1):   # L-1=19 time pairs
        X_curr = data[m, ell]
        drift = self.compute_drift(X_curr, networks)
        ...
```

每个 epoch 循环 **3800 次**，每次都调用神经网络 → **~25-30s/epoch**

## Solution

将所有 M×(L-1) 个配置堆叠成一个大 batch，一次性计算：

```python
# Reshape: (M, L, N, d) → (M*(L-1), N, d)
X_curr = data[:, :-1].reshape(M * (L-1), N, d)  # (3800, 10, 1)
X_next = data[:, 1:].reshape(M * (L-1), N, d)

# 一次性计算所有配置
drift = self.compute_drift_batched(X_curr, networks)        # (3800, N, d)
laplacian = self.compute_laplacian_sum_batched(X_curr, ...)  # (3800, N)
E_curr = self.compute_energy_batched(X_curr, ...)           # (3800,)
E_next = self.compute_energy_batched(X_next, ...)           # (3800,)
```

## Benchmark

| Version    | Time/epoch | Speedup |
|------------|------------|---------|
| Original   | 31.83s     | 1x      |
| Vectorized | 0.67s      | **47x** |

## Files

- `core/trajectory_free_loss_vectorized.py` - 向量化实现
- `scripts/train_nn_vectorized.py` - 使用优化 loss 的训练脚本

## Usage

```bash
# 使用优化版本
python scripts/train_nn_vectorized.py --epochs 1000

# 原始版本（未修改）
python scripts/train_nn.py --epochs 1000
```

## Key Changes

| Function | Input Shape | Output Shape |
|----------|-------------|--------------|
| `compute_drift_batched` | (B, N, d) | (B, N, d) |
| `compute_laplacian_sum_batched` | (B, N, d) | (B, N) |
| `compute_energy_batched` | (B, N, d) | (B,) |

其中 B = M × (L-1) = 3800
