# 🐼 Fei Lu 理论完整验证报告 (V2)

> **验证者**: 泡泡 Pi  
> **日期**: 2026-01-30  
> **来源**: 直接从 repo 中的 PDF 提取

---

## 📚 已验证的论文

1. **[FoCM 2021]** Lu, Maggioni, Tang. "Learning Interaction Kernels in Stochastic Systems of Interacting Particles from Multiple Trajectories"
2. **[Li & Lu 2021]** "On the coercivity condition in the learning of interacting particle systems" (arXiv:2011.10480)

---

## ✅ 精确定义提取

### 1. 系统方程 (Equation 1.1)

$$dx_{i,t} = \frac{1}{N} \sum_{j=1}^{N} \phi(\|x_{j,t} - x_{i,t}\|)(x_{j,t} - x_{i,t}) dt + \sigma dB_{i,t}$$

### 2. 能量势函数 (Equation 1.2)

$$V_\phi(X_t) = \frac{1}{2N} \sum_{i,i'} \Phi(\|x_{i,t} - x_{i',t}\|), \quad \text{where } \Phi'(r) = \phi(r) \cdot r$$

### 3. 范数定义 (Equation 1.6)

$$|||\phi||| := \|\phi(\cdot)\cdot\|_{L^2(\rho_T)} = \left( \int_{\mathbb{R}^+} |\phi(r) \cdot r|^2 \rho_T(dr) \right)^{1/2}$$

**重要**: 学习的是 $\Phi'(r) = \phi(r) \cdot r$，不是 $\phi(r)$！

### 4. Coercivity 条件 (Definition 3.1)

**Definition**: 系统满足 coercivity 条件（常数 $c_H > 0$）当且仅当对所有 $\phi \in H$：

$$c_H \cdot |||\phi|||^2 \leq \frac{1}{2\sigma^2 NT} \int_0^T \sum_{i=1}^{N} E\left[ \left| \frac{1}{N} \sum_{i'=1}^{N} \phi(r_{ii'}(t)) \cdot r_{ii'}(t) \right|^2 \right] dt$$

**等价形式 (Li & Lu, Definition 1.1)**:

$$I_T(h) := \frac{1}{T} \int_0^T E\left[ h(|r^t_{12}|) h(|r^t_{13}|) \frac{\langle r^t_{12}, r^t_{13} \rangle}{|r^t_{12}| |r^t_{13}|} \right] dt \geq c_{H,T} \cdot \frac{1}{T} \int_0^T E[h(|r^t_{12}|)^2] dt$$

### 5. 收敛率 (Theorem 3.2 in FoCM)

**连续时间观测**: 设 $\dim(H_n) \asymp \left(\frac{M}{\log M}\right)^{1/(2s+1)}$

$$|||\hat{\phi}_{T,M,H_n} - \phi|||^2 \lesssim \frac{1}{c_H^2} \left( \frac{\log M}{M} \right)^{2s/(2s+1)}$$

**离散时间观测**:

$$|||\hat{\phi}_{L,T,M,H} - \phi||| \leq |||\hat{\phi}_{T,\infty,H} - \phi||| + C\left( \sqrt{\frac{n}{M}} + \sqrt{\Delta t} \right)$$

---

## 🔍 与我们理论的对比

| 方面 | Fei Lu (FoCM 2021) | 我们的 Theorem 4 |
|------|-------------------|-----------------|
| **设定** | Trajectory-based MLE | Trajectory-free energy loss |
| **样本** | M = 轨迹数 | n = 快照数 |
| **率** | $(M/\log M)^{-2s/(2s+1)}$ | $n^{-2(s-1)/(2s+d)}$ ？ |
| **Coercivity** | 依赖假设空间 H | 固定常数 0.48 ？ |

### 关键差异

1. **Fei Lu 学习的是 $\phi(r) \cdot r$**，我们学习的是什么？
2. **Fei Lu 的率与 M（轨迹数）相关**，我们的率与 n（样本数）相关
3. **Coercivity 的 $c_H$** 在 Fei Lu 中是:
   - 依赖假设空间 H 的最小特征值
   - Proposition 3.2: $\lambda_{min}(A_\infty) = c_H$
   - 不是固定数值！

---

## ❓ 我们理论中存疑的部分

### 1. Coercivity 常数 0.48, 0.87, 0.73

**问题**: 这些数字的来源是什么？

**Fei Lu 的说法** (Proposition 3.2):
> "The smallest singular value of $A_\infty$ is $\sigma_{min}(A_\infty) = c_H$"

即 $c_H$ 是 Gram 矩阵的最小特征值，取决于：
- 假设空间 H 的基函数选择
- 测度 $\rho_T$ (pairwise 距离分布)
- 粒子数 N

**Li & Lu (Theorem 4.1)**: 对于某类势函数 Φ，coercivity 成立，但没给出具体常数！

### 2. 收敛率指数

**Fei Lu**: 率是 $2s/(2s+1)$ 关于 M

**我们声称**: 率是 $2(s-1)/(2s+d)$ 关于 n

**问题**:
- 这两个率在什么意义下可比较？
- 如果我们的 n = M × L（样本数 = 轨迹数 × 时间步），那么率应该如何转换？

### 3. 条件独立假设

**我们的 Proposition 2**:
> "给定 $X_1$，差值 $\{r_{1j}\}_{j \geq 2}$ 条件独立"

**问题**: 这只在以下情况成立：
- t = 0 时独立初始化
- Mean-field 极限 (N → ∞)

有交互时，粒子位置相关！

---

## ✅ 可以验证正确的部分

### Li & Lu Theorem 4.1

**定理**: 对于形如 $\Phi(r) = (a + r^\theta)^\gamma$ 的势函数（满足 $a \geq 0, \theta \in (1,2], \gamma \in (0,1], \theta\gamma > 1$），系统从稳态分布出发时，coercivity 成立。

**数值验证**: 我之前的 Monte Carlo 验证了不同 N 下的 coercivity 行为：
- N=2: $c_H \approx 1.00$
- N=5: $c_H \approx 0.50$  
- N=10: $c_H \approx 0.41$

---

## 📝 修正建议

### 1. 明确假设

```latex
\begin{assumption}
We consider snapshot data at $t=0$ with i.i.d. initial positions $X_i \sim \mu_0$.
The coercivity constant $c_H$ depends on $\mu_0$, $N$, and hypothesis space $H$.
\end{assumption}
```

### 2. 修正 Coercivity 声明

不要给出固定数值，而是说明依赖关系：

```latex
\begin{proposition}[Coercivity Scaling]
For $N$ particles with $X_i \sim_{iid} \mathcal{N}(0, I_d)$ and hypothesis space $H_n$ 
with orthonormal basis in $L^2(\rho)$:
$$c_{H_n} = \Theta\left(\frac{1}{N}\right) \text{ as } N \to \infty$$
\end{proposition}
```

### 3. 区分设定

明确说明我们的 trajectory-free 设定与 Fei Lu 的 trajectory-based 设定的区别。

---

## 📂 文件位置

论文 PDF:
- `/home/swei20/ips_unlabeled_learning/Learning interaction kernels in stochastic systems of interacting particles from multiple trajectories.pdf`
- `/home/swei20/ips_unlabeled_learning/On the coercivity condition in the learning of interacting particle systems.pdf`

验证脚本:
- `theory/fei_lu_full_verification.py`
- `theory/verify_minimax.py`

---

> 🐼 下次要先查 repo 里有什么文件！对不起 Viska！
