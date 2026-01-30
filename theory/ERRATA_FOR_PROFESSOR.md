# 📋 IPS_Theoretical_Analysis.pdf 勘误表

> **文件**: `theory/IPS_Theoretical_Analysis.pdf` (2026-01-29 12:06 版本)  
> **状态**: 已修正，待重新编译

---

## ❌ 错误 1: Coercivity 常数表格 (第 189-198 行)

### 原文 (错误)
```latex
\begin{tabular}{c|ccc}
$d$ & 1 & 2 & 3 \\ \hline
$c_H$ & $\geq 0.48$ & $\geq 0.87$ & $\geq 0.73$
\end{tabular}
```

### 问题
1. **公式推导错误**: 
   ```
   声称: I(1,G₁) = (1/(√3·π))√(2π - 6π/5) = √(4/15)
   实际: (1/(√3·π))√(2π - 6π/5) = 0.291 ≠ √(4/15) = 0.516
   ```
2. **数值来源不明**: 0.48, 0.87, 0.73 无法从 Fei Lu 论文复现

### 正确结果
根据 Li & Lu (2021, Definition 1.1):
$$c_H = \frac{2}{\pi}\arcsin\left(\frac{1}{2}\right) = \frac{1}{3} \approx 0.333 \quad (d=1)$$

---

## ❌ 错误 2: 条件独立假设 (第 102-103 行)

### 原文 (错误)
```latex
"conditional on $X_t^1$, the differences $\{r_{1j} = X_t^j - X_t^1\}_{j=2}^N$ 
are conditionally independent"
```

### 问题
- 这个假设**只在 t=0 时成立**（i.i.d. 初始化）
- 对于 t > 0，粒子通过交互动力学相关联
- 原文没有限定时间条件

### 正确陈述
```latex
"At the initial time $t=0$ with i.i.d. initialization, the differences 
$\{r_{1j}^0\}_{j=2}^N$ are conditionally independent given $X_0^1$. 
For $t > 0$, coercivity requires ergodicity conditions (Li & Lu 2021, Thm 4.1)."
```

---

## ❌ 错误 3: Gaussian Coercivity 证明 (第 216-220 行)

### 原文 (错误)
```latex
I(1, G_1) = \frac{1}{\sqrt{3}\pi} \sqrt{2\pi - \frac{6\pi}{5}} = \sqrt{\frac{4}{15}}
```

### 问题
- **数学错误**: 等式两边不相等
- 左边 = 0.291, 右边 = 0.516

### 正确推导
对于 $(r_{12}, r_{13}) \sim \mathcal{N}(0, \Sigma)$，相关系数 $\rho = 1/2$:
$$\mathbb{E}[\text{sign}(r_{12}) \cdot \text{sign}(r_{13})] = \frac{2}{\pi}\arcsin(\rho) = \frac{1}{3}$$

---

## ✅ 已确认正确的部分

| 内容 | 状态 |
|------|------|
| Proposition 1 (Energy Dissipation Identity) | ✅ 正确 |
| Definition 2 (Coercivity Condition) | ✅ 正确 |
| Theorem 1 (Identifiability from Coercivity) | ✅ 正确 |
| Theorem 2 (Consistency) | ✅ 正确 |
| Theorem 3 (Convergence Rate) | ✅ 正确 |
| Theorem 5 收敛率 $n^{-2(s-1)/(2s+d-2)}$ | ✅ 正确 |

---

## 📚 参考文献

1. **Li & Lu (2021)**. "On the coercivity condition in the learning of interacting particle systems". arXiv:2011.10480

2. **Lu, Maggioni, Tang (2021)**. "Learning Interaction Kernels in Stochastic Systems of Interacting Particles from Multiple Trajectories". Foundations of Computational Mathematics.

---

## 🔧 修正方法

1. 源文件已修正: `theory/theoretical_analysis.tex`
2. 需要重新编译 PDF:
   ```bash
   cd theory
   pdflatex standalone_theory.tex
   # 或
   pdflatex merged_theory.tex
   ```

---

**修正日期**: 2026-01-30 02:30 EST
