# 📊 MVP-2.0 实验报告：Fei Lu 方法验证

> **ID:** `IPS-20260129-mvp2_0-01`  \
> **Topic:** `ips_unlabeled` | **Phase:** 2 | **Project:** `IPS`  \
> **Author:** Viska Wei | **Date:** 2026-01-29 | **Status:** ❌ FAIL

> 🎯 **目标:** 验证 Fei Lu 的 error functional 在我们的数据上是否有效  \
> 🚀 **结论:** **FAIL** — 实现有问题，A 矩阵条件数过大（~10^11）导致数值不稳定

---

## 📋 Executive Summary

### 核心结论

Fei Lu 方法的初步实现**失败**。主要问题是 normal matrix A 的条件数极大（10^11 量级），导致即使使用 Tikhonov 正则化也无法获得稳定的解。

### 关键发现

| 发现 | 证据 | 影响 |
|------|------|------|
| A 矩阵条件数过大 | cond(A) ~ 10^11 | 数值不稳定 |
| 正则化参数选择困难 | λ_opt ~ 10^-9 或 10^-2 | 系数爆炸或欠拟合 |
| KDE 估计 u(x,t) 可能有误差 | 粒子数据 → KDE | 需验证 KDE 质量 |

### 数字速览

| 实验 | N | M | phi Error | A cond | 状态 |
|------|---|---|-----------|--------|------|
| 小规模 | 50 | 30 | 240% | 2.3e+11 | ❌ |
| 大规模 | 100 | 50 | 674654% | 4.0e+11 | ❌ |

---

## 🧪 实验设计

### 方法概述

参考 Lang & Lu (SIAM J. Sci. Comput. 2022) 实现：

1. **问题设定**：无外势 V，只学 φ
2. **数据处理**：从粒子位置用 KDE 估计 u(x,t)
3. **方法**：Least squares + B-spline + Tikhonov 正则化
4. **Error Functional**：Eq 2.16-2.18

### 配置

| 参数 | 小规模 | 大规模 | 论文参考 |
|------|--------|--------|----------|
| 粒子数 N | 50 | 100 | - |
| 样本数 M_samples | 30 | 50 | - |
| 空间网格 M_grid | 100 | 150 | 300 |
| 时间快照 L | 30 | 50 | - |
| 粘性 ν | 0.1 | 0.5 | 0.01-1.0 |
| B-spline 数量 | 10 | 8 | 10-30 |
| B-spline 阶 | 2 | 2 | 1-3 |
| KDE 带宽 | 0.5 | 0.4 | - |

---

## 📈 结果

### 实验 1：小规模测试

```
N=50, L=30, M_samples=30, M_grid=100, nu=0.1, n_basis=10
```

**结果**：
- A 条件数：2.27e+11
- 最优 λ：1.58e-02
- **phi 相对误差：240.50%**

### 实验 2：大规模测试

```
N=100, L=50, M_samples=50, M_grid=150, nu=0.5, n_basis=8
```

**结果**：
- A 条件数：4.01e+11
- 最优 λ：2.48e-09（太小！）
- 系数：爆炸（max ~23000）
- **phi 相对误差：674653.55%**

---

## 🔍 根因分析

### 为什么条件数这么大？

1. **可能原因 1：KDE 估计不准确**
   - 粒子数 N 不够大，KDE 估计的 u(x,t) 有较大误差
   - 带宽选择不当

2. **可能原因 2：Error functional 实现有误**
   - 双重卷积循环的离散化可能不正确
   - 符号或系数可能有错

3. **可能原因 3：问题本身 ill-posed**
   - 即使是 Fei Lu 的方法，也需要 coercivity condition
   - 我们的数据可能不满足

### 与 Fei Lu 论文的关键差异

| 方面 | Fei Lu | 我们 |
|------|--------|-----|
| 数据 | PDE 解 u(x,t) 直接生成 | 从粒子用 KDE 估计 |
| 数据精度 | 高（PDE solver） | 低（KDE 噪声） |
| A 条件数 | 论文未报告，但算法稳定 | ~10^11 |

---

## 🎯 下一步建议

### 短期（调试当前实现）

1. **验证 KDE 质量**
   - 画出 KDE 估计的 u(x,t) vs 解析解（OU 过程有解析解）
   - 调整带宽

2. **验证 error functional**
   - 用真实的 φ 计算 error functional，应该接近 0
   - 检查离散化误差

3. **增加数据量**
   - N=500, M_samples=100
   - 参考论文 M=300

### 中期（尝试替代方法）

1. **直接用 PDE solver 生成 u(x,t)**
   - 绕过 KDE，直接模拟 mean-field PDE
   - 这是 Fei Lu 论文的做法

2. **尝试其他基函数**
   - 使用 RKHS eigenfunctions（论文建议）
   - 更 data-adaptive

### 长期（回到原始问题）

如果 Fei Lu 方法验证成功，需要：
1. 把 V 加回来
2. 考虑如何处理 V-Φ identifiability

---

## 📎 附录

### A. 代码文件

| 文件 | 说明 |
|------|------|
| `scripts/train_fei_lu_method.py` | MVP-2.0 实现 |
| `results/mvp2_0/metrics.json` | 实验指标 |
| `results/mvp2_0/results.npz` | 详细结果 |

### B. 关键公式

**Error Functional (Eq 2.3)**:
$$\mathcal{E}(\psi) = \frac{1}{T} \int_0^T \int_{\mathbb{R}^d} \left[ |K_\psi * u|^2 u + 2\partial_t u (\Psi * u) + 2\nu \nabla u \cdot (K_\psi * u) \right] dx\, dt$$

**Discrete Error Functional (Eq 2.16)**:
$$\mathcal{E}_{M,L}(\psi) = c^T A_{M,L} c - 2b_{M,L}^T c$$

---

> **报告生成时间**: 2026-01-29
> **实验耗时**: ~7 分钟（数据生成 + 训练）
> **结论**: FAIL，需要调试实现或尝试替代方法
