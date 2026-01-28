# 🍃 Loss Formula Verification and Correction
> **Name:** Loss Formula Verification and Correction  \
> **ID:** `IPS-20260128-mvp1_2-01`  \
> **Topic:** `ips_unlabeled` | **MVP:** MVP-1.2 | **Project:** `IPS`  \
> **Author:** Viska Wei | **Date:** 2026-01-28 | **Status:** ✅ PASS (理论验证)
>
> 🎯 **Target:** 验证 trajectory-free loss 公式的正确性，找出 MVP-1.0/1.1 失败的根因  \
> 🚀 **Decision / Next:** 发现原公式有误并修正，但 identifiability 问题仍存在

---

## ⚡ 核心结论速览

> **一句话**: 发现原 loss 公式存在**符号和系数错误**：正确公式是 `R = J_diss - (σ²/2) J_lap + dE = 0`，而非原来的 `J_diss + σ J_lap - 2 dE = 0`。验证实验确认修正后的公式在真实势函数下残差随 dt→0 而→0。**但修正公式后仍无法学习势函数**，说明存在更根本的 identifiability 问题。

### 0.1 这实验到底在做什么？

$$
X := \underbrace{\text{Loss Formula Verification}}_{\text{是什么}}\ \xrightarrow[\text{通过}]{\ \text{Ito引理推导 + 数值验证 🍎}\ }\ \underbrace{\text{找出正确的弱形式公式 ⭐}}_{\text{用于}}
$$

- **🐻 What**: 验证弱形式 PDE loss 公式的数学正确性
- **🍎 核心机制**: 从 Ito 引理重新推导，用真实势函数验证残差
- **⭐ 目标**: 确定正确的公式系数和符号
- **🩸 Why**: MVP-1.0/1.1 loss→0 但势函数完全错误，怀疑公式本身有问题
- **💧 How**: 用真实 V 和 Φ 计算各项，检查哪个公式使残差→0

### 0.2 关键数字

| Metric | Value | Notes |
|--------|-------|-------|
| 原公式残差 (J_diss + σ*J_lap - 2*dE) | 8.86e-02 | 显著不为零 |
| 修正公式残差 (J_diss - σ²/2*J_lap + dE) | 6.38e-02 → **随 dt→0 而→0** | 收敛行为正确 |
| dt=0.2 时残差 | 1.57e-01 | |
| dt=0.1 时残差 | 7.65e-02 | |
| dt=0.05 时残差 | 3.76e-02 | |
| dt=0.02 时残差 | 1.49e-02 | 线性收敛 |

### 0.3 Links

| Type | Link |
|------|------|
| 🧠 Hub | `experiments/ips_unlabeled/ips_unlabeled_hub.md` § K4 |
| 🗺️ Roadmap | `experiments/ips_unlabeled/ips_unlabeled_roadmap.md` § MVP-1.2 |
| 📋 验证脚本 | `scripts/verify_loss_formula.py`, `scripts/verify_loss_quick.py` |

---

# 1. 🎯 目标

**核心问题**: 为什么 MVP-1.0/1.1 中 loss→0 但势函数完全错误？公式本身是否有问题？

**验证内容**:
1. 原论文 loss 公式的系数是否正确
2. 从 Ito 引理重新推导正确公式
3. 用真实势函数验证：正确公式应使残差→0（期望意义下）

## 1.1 成功标准

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| ✅ 通过 | 找到使残差→0 的公式 | 残差随 dt→0 线性收敛到 0 |
| ❌ 否决 | 所有公式残差都不为 0 | 说明弱形式方法本身有理论问题 |

---

# 2. 🦾 方法

## 2.1 理论推导

### 2.1.1 从 Ito 引理推导正确公式

对于 SDE: $dX = b \, dt + \sigma \, dW$，其中 $b = -\nabla V - \nabla\Phi * \mu$

**Ito 引理** 给出对任意 $f \in C^2$：
$$
d\langle f, \mu \rangle = \langle \nabla f \cdot b + \frac{\sigma^2}{2} \Delta f, \mu \rangle \, dt + \text{martingale}
$$

取 **self-test 函数** $f = V + \Phi * \mu$：
- $\nabla f = \nabla V + \nabla\Phi * \mu$
- $\nabla f \cdot b = -|\nabla V + \nabla\Phi * \mu|^2$（因为 $b = -\nabla f$）
- $\Delta f = \Delta V + \Delta\Phi * \mu$

代入得：
$$
dE = -\langle |\nabla V + \nabla\Phi * \mu|^2, \mu \rangle \, dt + \frac{\sigma^2}{2} \langle \Delta V + \Delta\Phi * \mu, \mu \rangle \, dt + \text{martingale}
$$

整理得**正确的弱形式公式**：
$$
\boxed{J_{diss} - \frac{\sigma^2}{2} J_{lap} + dE = 0 \quad \text{(期望意义下)}}
$$

其中：
- $J_{diss} = \langle |\nabla V + \nabla\Phi * \mu|^2, \mu \rangle \, \Delta t$ （耗散项）
- $J_{lap} = \langle \Delta V + \Delta\Phi * \mu, \mu \rangle \, \Delta t$ （Laplacian 项）
- $dE = E(t+\Delta t) - E(t)$ （能量变化）

### 2.1.2 与原论文公式对比

| 项 | 原论文公式 | 正确公式 | 差异 |
|----|----------|---------|------|
| 耗散项 | $+J_{diss}$ | $+J_{diss}$ | 相同 |
| Laplacian 项 | $+\sigma \cdot J_{lap}$ | $-\frac{\sigma^2}{2} \cdot J_{lap}$ | **符号相反，系数不同** |
| 能量项 | $-2 \cdot dE$ | $+1 \cdot dE$ | **符号相反，系数不同** |

**关键错误**:
1. Laplacian 项前应该是**负号**，不是正号
2. 系数应该是 **σ²/2**，不是 σ
3. 能量项系数应该是 **+1**，不是 -2

## 2.2 数值验证

### 2.2.1 验证方法

用真实势函数计算各项，检验哪个公式使残差为 0：

```python
# 真实势函数
V_true(x) = 0.5 * k * x²      # Harmonic
Φ_true(r) = A * exp(-r²/(2σ²)) # Gaussian

# 计算各项
J_diss = mean(|∇V + ∇Φ*μ|²) * dt
J_lap = mean(ΔV + ΔΦ*μ) * dt
dE = E(t+dt) - E(t)

# 测试公式
residual_original = J_diss + σ * J_lap - 2 * dE  # 原公式
residual_correct = J_diss - (σ²/2) * J_lap + dE   # 正确公式
```

### 2.2.2 实验配置

| 参数 | 值 |
|------|-----|
| N (粒子数) | 10 |
| d (维度) | 1 |
| M (样本数) | 50-500 |
| L (时间快照) | 变化 |
| σ (噪声) | 0.1 |
| V | Harmonic(k=1) |
| Φ | Gaussian(A=1, σ=1) |

---

# 3. 📊 结果

## 3.1 不同公式的残差比较

使用 M=500 样本，L=20 快照：

| 公式 | Mean Residual | Std Residual | t-statistic | p-value |
|------|--------------|--------------|-------------|---------|
| J_diss + σ*J_lap - 2*dE (原) | 8.86e-02 | 6.03e-02 | 143.3 | ~0 |
| J_diss - σ²/2*J_lap + dE (修正) | 6.38e-02 | 1.78e-02 | 349.5 | ~0 |

两个公式的残差都显著不为零，但需要检查**离散时间误差**。

## 3.2 残差随 dt 的收敛行为

**关键测试**：如果公式正确，残差应随 dt→0 而→0。

| dt_snap | Mean Residual | Std Residual |
|---------|--------------|--------------|
| 0.20 | 1.57e-01 | 3.10e-02 |
| 0.10 | 7.65e-02 | 1.47e-02 |
| 0.05 | 3.76e-02 | 7.27e-03 |
| 0.02 | 1.49e-02 | 3.08e-03 |

**观察**: 残差**线性收敛**到 0！这证明修正后的公式是正确的。

```
残差 ≈ O(dt) → dt→0 时残差→0
```

## 3.3 修正公式后的训练结果

用修正后的公式重新训练（MVP-1.2b）：

| 配置 | V Error | Φ Error | Final Loss |
|------|---------|---------|------------|
| N=5, L=10, M=30 | **97.62%** | **128.35%** | ~0 |

**结论**: 即使公式正确，仍然无法学习势函数。问题不在公式，而在 **identifiability**。

---

# 4. 💡 洞见

## 4.1 为什么修正公式后仍然失败？

### 4.1.1 根本原因：Identifiability 问题

弱形式公式 $J_{diss} - \frac{\sigma^2}{2} J_{lap} + dE = 0$ 对于**任何**满足能量平衡的 $(V, \Phi)$ 对都成立，不仅仅是真实的势函数。

**数学解释**：
- 设 $(V_{true}, \Phi_{true})$ 是真实势函数
- 存在无穷多个 $(V', \Phi')$ 使得 $\nabla V' + \nabla\Phi' * \mu = \nabla V_{true} + \nabla\Phi_{true} * \mu$
- 这些 $(V', \Phi')$ 给出相同的漂移场，因此相同的分布演化
- 弱形式 loss 无法区分它们

### 4.1.2 物理直觉

粒子只"感受"到总力 $F = -\nabla V - \nabla\Phi * \mu$，无法区分哪部分来自 V，哪部分来自 Φ。

**类比**：就像只测量合力无法确定各分力一样。

### 4.1.3 为什么添加约束 (V(0)=0, Φ(r_ref)=0) 无效？

这些约束只固定了势函数的**常数项**，但 identifiability 问题在于势函数的**形状**（梯度）。

## 4.2 原论文公式错误的来源分析

对比论文第 157-158 行和 Ito 引理推导：

| 论文写法 | Ito 推导 | 可能原因 |
|---------|---------|---------|
| $+\sigma \int [\Delta V + \Delta\Phi * \mu] \mu \, dx \, \Delta t$ | $-\frac{\sigma^2}{2} \langle \Delta V + \Delta\Phi * \mu, \mu \rangle \, \Delta t$ | 可能是笔误或不同约定 |
| $-2 \int [V + \Phi * \mu] \mu \, dx \big|_{t_\ell}^{t_{\ell+1}}$ | $+[E(t_{\ell+1}) - E(t_\ell)]$ | 符号展开方式不同 |

**注意**：论文提到使用 "automatic reproducing kernel" 方法，可能 RKHS 正则化是解决 identifiability 的关键，但本实验未实现。

---

# 5. 📝 结论 & 下一步

## 5.1 核心发现

1. **✅ 发现并修正公式错误**:
   - 正确公式: $J_{diss} - \frac{\sigma^2}{2} J_{lap} + dE = 0$
   - 原公式的 Laplacian 项符号错误，系数也错误

2. **❌ 修正后仍无法学习**:
   - V error = 97.62%, Φ error = 128.35%
   - 问题是根本性的 identifiability，不是公式 bug

3. **⚠️ 弱形式方法的局限**:
   - 无法仅从分布演化区分不同的 $(V, \Phi)$ 对
   - 需要额外信息或约束

## 5.2 对后续研究的影响

| 影响 | 建议 |
|------|------|
| 代码层面 | 已修正 `trajectory_free_loss.py` 和 `train_nn_constrained.py` |
| 方法论层面 | 纯弱形式方法不可行，需要额外约束或正则化 |
| 理论层面 | 需要研究 identifiability 条件 |

## 5.3 下一步

| 优先级 | 任务 | 说明 |
|--------|------|------|
| 🔴 P0 | MVP-1.3: Φ-only 简化问题 | 假设 V 已知，消除一半 trade-off |
| 🟡 P1 | RKHS 正则化 | 实现论文提到的 automatic kernel |
| 🟢 P2 | 理论分析 | 研究什么条件下有唯一解 |

---

# 6. 📎 附录

## 6.1 验证脚本

### 主要脚本

| 脚本 | 功能 |
|------|------|
| `scripts/verify_loss_formula.py` | 测试不同系数组合 |
| `scripts/verify_loss_formula_v2.py` | 系统搜索最佳公式 |
| `scripts/verify_loss_formula_v3.py` | 统计检验 |
| `scripts/verify_loss_quick.py` | 快速验证 dt 收敛 |

### 复现命令

```bash
cd ~/ips_unlabeled_learning
source /srv/local/tmp/swei20/miniconda3/etc/profile.d/conda.sh
conda activate viska-torch-3

# 验证公式
python3 scripts/verify_loss_quick.py

# 训练（修正公式）
python3 scripts/train_nn_constrained.py --N 5 --L 10 --M 30 --epochs 200
```

## 6.2 代码修改

### `core/trajectory_free_loss.py`

```python
# 修改前
residual = total_diss + total_diff - 2 * total_energy_change

# 修改后
residual = total_diss - self.sigma_sq_half * total_lap + total_energy_change
```

关键变化：
1. 添加 `self.sigma_sq_half = sigma ** 2 / 2`
2. Laplacian 项前加负号
3. 能量项系数改为 +1

## 6.3 数学推导详细步骤

### Ito 引理

对于 $f(X_t)$，Ito 引理给出：
$$
df = \nabla f \cdot dX + \frac{1}{2} \text{tr}(\nabla^2 f \cdot \sigma\sigma^T) \, dt
$$

代入 $dX = b \, dt + \sigma \, dW$：
$$
df = \nabla f \cdot b \, dt + \frac{\sigma^2}{2} \Delta f \, dt + \sigma \nabla f \cdot dW
$$

对经验分布 $\mu = \frac{1}{N} \sum_i \delta_{X_i}$ 取期望：
$$
d\langle f, \mu \rangle = \langle \nabla f \cdot b + \frac{\sigma^2}{2} \Delta f, \mu \rangle \, dt + \text{martingale}
$$

取 $f = V + \Phi * \mu$（self-test 函数）：
- $\nabla f = \nabla V + \nabla\Phi * \mu$
- $b = -\nabla V - \nabla\Phi * \mu = -\nabla f$
- $\nabla f \cdot b = -|\nabla f|^2$

代入得：
$$
dE = -\langle |\nabla V + \nabla\Phi * \mu|^2, \mu \rangle \, dt + \frac{\sigma^2}{2} \langle \Delta V + \Delta\Phi * \mu, \mu \rangle \, dt
$$

即：
$$
J_{diss} \, dt - \frac{\sigma^2}{2} J_{lap} \, dt + dE = \text{martingale} \approx 0
$$

---

> **实验完成时间**: 2026-01-28
