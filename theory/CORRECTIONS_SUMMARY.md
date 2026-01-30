# 🐼 理论修正总结

> **日期**: 2026-01-30  
> **验证者**: 泡泡 Pi

---

## ❌ 原始错误

### 1. Coercivity 常数 (已修正 ✅)

**原文 (错误)**:
```
d=1: c_H ≥ 0.48
d=2: c_H ≥ 0.87
d=3: c_H ≥ 0.73
```

**问题**:
- 公式 `I(1,G₁) = (1/(√3·π))√(2π - 6π/5) = √(4/15)` 数学上**不成立**！
  - LHS = 0.291
  - RHS = 0.516
- 数值无法复现

**修正后**:
```
c_H = (2/π) arcsin(1/2) = 1/3 ≈ 0.333 (d=1, 常函数空间)
```

依据: Li & Lu (2021), Definition 1.1

---

### 2. 条件独立假设 (已修正 ✅)

**原文 (错误)**:
> "给定 X_t^1，差值 {r_{1j}}_{j≥2} 条件独立"

**问题**: 只在 t=0 时成立，有交互后粒子相关！

**修正后**:
> "在 t=0 时 i.i.d. 初始化下，条件独立成立。对于 t>0，需要 Li & Lu 的遍历性条件 (Theorem 4.1)。"

---

### 3. 收敛率指数 (已确认正确 ✅)

**Theorem 5 (Total Error Bound)** 中的率：
```
n^{-2(s-1)/(2s+d-2)}
```

这是**正确的**！对应于学习 (s-1)-smooth 的 d 维梯度函数。

---

## ✅ 已验证正确的部分

| 定理 | 状态 |
|------|------|
| Proposition 1 (Energy Dissipation) | ✅ |
| Theorem 1 (Identifiability) | ✅ |
| Definition 2 (Coercivity) | ✅ |
| Theorem 2 (Consistency) | ✅ |
| Theorem 3 (Convergence Rate) | ✅ |
| Theorem 5 (Total Error Bound) | ✅ (率正确) |

---

## 📚 引用来源

1. **Li & Lu (2021)**. "On the coercivity condition in the learning of interacting particle systems". arXiv:2011.10480

2. **Lu, Maggioni, Tang (2021)**. "Learning Interaction Kernels in Stochastic Systems of Interacting Particles from Multiple Trajectories". Foundations of Computational Mathematics.

---

## 📂 修改的文件

- `theory/theoretical_analysis.tex` - 主理论文件
  - 修正 Proposition 4 (Gaussian Coercivity)
  - 修正 Proposition 3 (Gradient Coercivity) 的条件独立假设
  - 移除错误的数值表格
  - 添加正确的引用说明

---

> 🐼 数学正确性是科研的底线！
