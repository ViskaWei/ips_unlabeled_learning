# 🐼 Fei Lu 理论完整验证报告

> **验证者**: 泡泡 Pi
> **日期**: 2026-01-30
> **状态**: 使用 Fei Lu 原始论文逐条验证

---

## 📚 参考文献

1. **[FoCM 2021]** Lu, Maggioni, Tang. "Learning Interaction Kernels in Stochastic Systems of Interacting Particles from Multiple Trajectories"
2. **[arXiv:2011.10480]** Li & Lu. "On the coercivity condition in the learning of interacting particle systems"

---

## ✅ 验证通过的部分

### 1. 系统设定 (Equation 1.1)

```
dx_{i,t} = (1/N) Σ_j φ(||x_j - x_i||)(x_j - x_i) dt + σ dB_{i,t}
```

**关键定义：**
- `φ: R+ → R` 是交互核
- `Φ'(r) = φ(r)·r` 是 pairwise potential 的导数
- `ρ_T` 是 [0,T] 上所有 pairwise 距离的测度

### 2. 函数空间范数 (Equation 1.6)

```
|||φ||| = ||φ(·)·||_{L²(ρ_T)} = (∫ |φ(r)r|² ρ_T(dr))^{1/2}
```

**重要**: 我们学习的是 `Φ'(r) = φ(r)r`，不是 `φ(r)` 本身！

### 3. Coercivity 定义 (Section 3.1)

**定义**: 假设空间 H 满足 coercivity 条件（常数 c_H > 0）当且仅当对所有 φ ∈ H:

```
E[ ||f_φ(X)||² ] ≥ c_H · |||φ|||²
```

其中 `f_φ = -∇V_φ` 是由 φ 诱导的漂移场。

**等价形式**: 设 G_T 是积分算子

```
(G_T φ)(r) = E[ Σ_{j≠1} φ(r_{1j}) · (r_{1j}/|r_{1j}|) | r_{12} = r ]
```

则 coercivity 等价于 G_T 的正定性:

```
||G_T φ||_{L²(ρ_T)} ≥ c_H · ||φ||_{L²(ρ_T)}
```

### 4. 收敛率 (Theorem 3.2)

**连续时间观测**:
```
|||φ̂ - φ|||² ≲ (1/c_H²) · (log M / M)^{2s/(2s+1)}
```

其中:
- M = 轨迹数量
- s = φ 的 Hölder 光滑度
- 这是 **1D 非参数回归的极小极大最优率**！

---

## ❌ 我们理论中的错误

### 错误 1: Coercivity 常数来源不明

**原文声称**:
```
d=1: c_H ≥ 0.48
d=2: c_H ≥ 0.87
d=3: c_H ≥ 0.73
```

**数值验证结果** (iid Gaussian, N 个粒子):

| N | d=1 |
|---|-----|
| 2 | 1.00 |
| 3 | 0.67 |
| 5 | 0.50 |
| 10 | 0.41 |
| 20 | 0.37 |

**结论**: 
- c_H 依赖于粒子数 N
- 0.48 对应大约 N=4-5 个粒子
- **必须明确 N 的取值**

### 错误 2: 收敛率指数

**原文 Theorem 4**:
```
rate ≥ n^{-2(s-1)/(2s+d)}
```

**问题**:
1. Fei Lu 的率是关于 M（轨迹数），不是 n（样本数）
2. 我们的设定是 trajectory-free，与 Fei Lu 不同
3. 正确的梯度估计率应该是 `n^{-2(s-1)/(2s+d-2)}`

### 错误 3: 条件独立假设

**原文 Proposition 2**:
> "给定 X₁，差值 {r_{1j}}_{j≥2} 条件独立"

**问题**: 
- 这只在 t=0（独立初始化）或 mean-field 极限下成立
- 有交互时粒子耦合，不满足条件独立

---

## 🔄 我们的设定 vs Fei Lu

| 方面 | 我们的设定 | Fei Lu |
|------|-----------|--------|
| **数据类型** | Trajectory-free (快照) | Trajectory-based |
| **学习目标** | V 和 Φ 同时学习 | 只学习 φ |
| **Loss 函数** | Energy dissipation | MLE (负对数似然) |
| **样本** | n = M × L 快照 | M 条轨迹，每条 L 个时间点 |

**关键差异**: 
- Fei Lu 用轨迹信息构造 MLE
- 我们只用快照对，没有轨迹连续性
- 这导致 identifiability 更困难！

---

## 📝 修正建议

### 1. Coercivity 常数

**修正**: 明确说明适用的 N 范围

```latex
\begin{proposition}[Gaussian Coercivity]
For N particles with X_i \sim_{iid} N(0, I_d), the coercivity constant satisfies:
$$c_H \geq \frac{(N-1)}{N^2} \cdot c_0(d)$$
where $c_0(d) > 0$ depends only on dimension.
\end{proposition}
```

### 2. 收敛率

**修正**: 区分不同的 rate

```latex
\begin{theorem}[Corrected Convergence Rate]
For gradient estimation in $L^2(\rho)$ from n samples:
$$E[\|\nabla\hat{\Phi} - \nabla\Phi\|^2] \leq C \cdot n^{-\frac{2(s-1)}{2s+d-2}}$$
This matches the standard nonparametric rate for (s-1)-smooth gradient estimation in d dimensions.
\end{theorem}
```

### 3. 条件独立

**修正**: 明确假设条件

```latex
\begin{assumption}[Initial Distribution]
At time $t=0$, particles are independently distributed: $X_i^0 \sim_{iid} \mu_0$.
The coercivity results apply to this initial distribution.
\end{assumption}
```

---

## 🔬 验证脚本

所有数值验证见:
- `theory/fei_lu_full_verification.py`
- `theory/verify_minimax.py`
- `theory/verify_gaussian_integrals.py`

运行结果保存在:
- `theory/verification_output.txt`

---

## 📋 下一步

1. **重写 theoretical_analysis.tex**
   - 使用 Fei Lu 的精确定义
   - 明确所有假设条件
   - 修正收敛率指数

2. **添加引用**
   - 正确引用 Li & Lu 2021 的 coercivity 结果
   - 区分我们的新贡献 vs 已知结果

3. **数值验证**
   - 运行实验验证理论预测
   - 比较不同 N 下的 coercivity

---

> 🐼 验证完成！理论框架基本正确，但需要修正细节和明确假设。
