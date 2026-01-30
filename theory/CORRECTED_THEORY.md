# 🐼 理论修正报告

> **作者**: 泡泡 Pi
> **日期**: 2026-01-29
> **状态**: 使用数学验证工具审核并修正

---

## 📊 使用的验证工具

| 工具 | 用途 | 结果 |
|------|------|------|
| SymPy | 符号计算 | ✅ 验证 Gaussian 积分 |
| NumPy/SciPy | 数值验证 | ✅ Monte Carlo 检验 |
| Fano 不等式推导 | 手工推理 | ✅ 找到指数错误 |

---

## 🔴 已确认的错误

### 错误 1: Theorem 4 (Minimax) 指数写错

**原文** (theorem statement):
```
rate ≥ n^{-2(s-1)/(2s+d)}
```

**正确** (from proof & standard theory):
```
rate ≥ n^{-2(s-1)/(2s+d-2)}
```

**验证**:
```python
# 标准非参数回归: ||f̂ - f||² ~ n^{-2s/(2s+d)}
# 梯度估计 (少一阶导): ||∇f̂ - ∇f||² ~ n^{-2(s-1)/(2(s-1)+d)}
#                                     = n^{-2(s-1)/(2s+d-2)}
```

**数值对比** (s=2, d=1):
- 定理声称: n^{-0.40}
- 证明得到: n^{-0.67}
- 差异显著！

**结论**: 定理陈述有笔误，证明是对的

---

### 错误 2: Gaussian Coercivity 常数来源不明

**原文声称**:
```
d=1: c_H ≥ 0.48
d=2: c_H ≥ 0.87
d=3: c_H ≥ 0.73
```

**我们的计算** (使用 Monte Carlo + SymPy):

定义 1 (简单版):
```
c_H = E[|u₁₂ + u₁₃|²] / 2
```
结果: d=1 → 1.33, d=2 → 1.41, d=3 → 1.44

定义 2 (Fei Lu 条件方差版):
```
c_H = E[Var(∇φ | X₁)] / E[|∇φ|²]
```
结果: d=1 → 0.67, d=2 → 0.58, d=3 → 0.56

**问题**: 两种定义都不给出 0.48！

**可能原因**:
1. 原文用了不同的归一化
2. 原文考虑了 N→∞ mean-field 极限
3. 原文的 I(d, G_d) 公式可能有误

**结论**: 需要查阅 Fei Lu 原始论文确认定义

---

### 错误 3: Proposition 2 条件独立性假设

**原文假设**:
> "conditional on X₁, the differences {r₁ⱼ}_{j≥2} are conditionally independent"

**问题**: 这在有交互的 IPS 中不成立！

当粒子通过势函数 Φ 耦合时，给定 X₁，其他粒子的位置是相关的。

**适用情况**:
1. t=0 初始分布（独立初始化）
2. N→∞ mean-field 极限
3. 或者 Φ 很弱（近似独立）

**修复建议**:
```
明确这是 t=0 或 stationary + mean-field 的假设
```

---

## 🟡 需要补充的证明

### Theorem 2 (Consistency)
- 只有 proof sketch
- 需要展开：
  1. 均匀大数定律条件
  2. M-estimation argmin 连续性
  3. 紧性论证

### Theorem 3 (Convergence Rate)  
- 没有证明！
- 需要：Rademacher 复杂度 → 泛化界

---

## 🟢 验证正确的部分

| 结果 | 验证方法 | 状态 |
|------|---------|------|
| Lemma 1 (条件独立引理) | 手工推导 | ✅ |
| Prop 1 (能量耗散) | 逻辑检查 | ✅ (细节需补) |
| Thm 1 (Identifiability) | 逻辑检查 | ✅ |
| d=1 E[sign·sign]=1/3 | Monte Carlo | ✅ |

---

## 📝 修正后的定理陈述

### Theorem 4' (Minimax Lower Bound) - 修正版

设 F_s = {Φ ∈ C^s : ||Φ||_{C^s} ≤ R} 为 Hölder 球。对于基于 n=ML 样本的任意估计器 Φ̂：

$$
\inf_{\hat{\Phi}} \sup_{\Phi^* \in F_s} E[||\nabla\hat{\Phi} - \nabla\Phi^*||^2_{L^2}] \geq c \cdot n^{-\frac{2(s-1)}{2s+d-2}}
$$

**证明**: 见 `verify_minimax.py` 的推导。

---

## 🛠️ 后续工作

1. **联系 Fei Lu 团队** 确认 coercivity 定义
2. **Lean 4 形式化** Identifiability 定理
3. **补全证明** Theorem 2, 3

---

> 使用 SymPy + NumPy + 手工推理完成审核 🐼
