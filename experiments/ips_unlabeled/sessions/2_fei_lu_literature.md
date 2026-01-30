# Session 2: Fei Lu 文献调研
> **日期**: 2026-01-28 | **类型**: 文献调研 | **触发**: 用户请求

---

## 1. 调研目标

从 [Fei Lu @ JHU](https://math.jhu.edu/~feilu/research.html) 的研究中，寻找与 IPS Unlabeled Learning 项目相关的理论和方法。

---

## 2. Fei Lu 背景

- **职位**: 约翰霍普金斯大学数学系副教授 (2023–至今)
- **研究方向**: 概率论、统计学及其在科学计算中的应用
- **核心主题**: 从数据中学习动力学系统 (Learning dynamics from data)
- **基金**: NSF CAREER Award (DMS-2238486): "Learning kernels in operators from data" — $500,000 (2023–2028)
- **Google Scholar**: 1,097+ 引用

---

## 3. 核心相关论文

### 3.1 Identifiability 理论

#### Paper 1: Identifiability of interaction kernels in mean-field equations
- **作者**: Quanjun Lang, Fei Lu
- **发表**: Foundations of Data Science, 2023
- **链接**: [arXiv:2106.05565](https://arxiv.org/abs/2106.05565)

**核心结论**:
1. Loss functional 的唯一最小值**仅在特定函数空间中保证**
2. 可辨识的函数空间 = **RKHS (再生核希尔伯特空间)** 的闭包
3. **逆问题本质上是 ill-posed**，需要正则化
4. "identifiability holds on two ambient L² spaces **if and only if** the integral operators are strictly positive"
5. **Weighted L² space** 比 unweighted L² space 产生更准确的估计

**关键引用**:
> "The inverse problem is ill-posed in general."
> "Identifiability holds on any subspace of two reproducing kernel Hilbert spaces (RKHS), whose reproducing kernels are intrinsic to the system and are data-adaptive."

---

#### Paper 2: On the coercivity condition in the learning of interacting particle systems
- **作者**: Zhongyang Li, Fei Lu
- **发表**: Stochastic Dynamics
- **链接**: [arXiv:2011.10480](https://arxiv.org/abs/2011.10480)

**核心结论**:
1. **Coercivity condition** 是 identifiability 的数学基础
2. Coercivity ⟺ 积分算子的**严格正定性**
3. 当系统是 **ergodic（遍历的）** 时，coercivity 成立
4. **若 coercivity 不满足，interaction function 不可唯一辨识**

**关键引用**:
> "In the learning of systems of interacting particles or agents, coercivity condition ensures identifiability of the interaction functions, providing the foundation of learning by nonparametric regression."
> "For a class of interaction functions such that the system is ergodic, the integral kernel is strictly positive definite, and hence the coercivity condition holds true."

---

#### Paper 3: On the identifiability of interaction functions in systems of interacting particles
- **作者**: Z. Li, F. Lu, M. Maggioni, S. Tang, C. Zhang
- **发表**: Stochastic Processes and Applications, 132:135–163, 2021

**核心结论**:
1. Coercivity condition 是 identifiability 的**充分条件**
2. 当粒子数 N→∞ 时，coercivity 变为**必要条件**
3. 证明使用了 **Müntz type theorems** 来验证积分核的正定性

---

### 3.2 学习方法

#### Paper 4: Learning interaction kernels in mean-field equations
- **作者**: Quanjun Lang, Fei Lu
- **发表**: SIAM J. Sci. Comput. 44(1), A260–A285, 2022
- **链接**: [arXiv:2010.15694](https://arxiv.org/abs/2010.15694)

**方法**:
- 使用 **discrete space-time observations of the solution** (与我们的 trajectory-free 设定一致)
- **Least squares with regularization** on data-adaptive hypothesis spaces
- 收敛速率 = 数值积分器的阶

**成功案例**:
| 例子 | Kernel 类型 | 结果 |
|------|------------|------|
| Opinion dynamics | Piecewise linear | 成功 |
| Granular media | Quadratic (smooth) | 成功 |
| Aggregation-diffusion | Repulsive-attractive | 成功 |
● 这篇论文太关键了！让我提取实验参数和核心方法。                                  
                                                                                  
  📊 关键实验参数提取                                                             
                                                                                  
  Table 2: 数据生成和推断设置                                                     
  参数: 时间区间                                                                  
  值: [0, T] = [0, 1]                                                             
  说明:                                                                           
  ────────────────────────────────────────                                        
  参数: 空间域                                                                    
  值: Ω = [-10, 10]                                                               
  说明:                                                                           
  ────────────────────────────────────────                                        
  参数: 真实解网格                                                                
  值: dt = 0.001, dx = 20/3000 ≈ 0.0067                                           
  说明: 高精度                                                                    
  ────────────────────────────────────────                                        
  参数: 数据网格                                                                  
  值: Δt = dt, Δx = 10dx                                                          
  说明: M = 300 空间点                                                            
  ────────────────────────────────────────                                        
  参数: 降采样测试                                                                
  值: M ∈ {300, 250, 200, 150, 120, 100, 60, 50, 40, 30}                          
  说明:                                                                           
  三个成功案例的误差水平                                                          
  例子: Cubic                                                                     
  Kernel: φ(r)=3r² (smooth)                                                       
  ν: 1.0                                                                          
  最优维度: 10 (spline)                                                           
  L²(ρ̄_T) 误差: 1.90%                                                             
  RKHS 误差: 0.43%                                                                
  ────────────────────────────────────────                                        
  例子: Opinion                                                                   
  Kernel: piecewise linear                                                        
  ν: 0.1                                                                          
  最优维度: 28 (spline)                                                           
  L²(ρ̄_T) 误差: 36.74%                                                            
  RKHS 误差: 8.10%                                                                
  ────────────────────────────────────────                                        
  例子: Repulsion-attraction                                                      
  Kernel: singular                                                                
  ν: 0.01                                                                         
  最优维度: 30 (spline)                                                           
  L²(ρ̄_T) 误差: 49.06%                                                            
  RKHS 误差: 4.36%                                                                
  关键发现：smooth kernel 可以达到 <2% 误差！                                     
                                                                                  
  ---                                                                             
  🔑 与我们项目的关键区别                                                         
  ┌──────────┬────────────────────────────┬───────────────┐                       
  │   方面   │        Fei Lu 方法         │ 我们当前方法  │                       
  ├──────────┼────────────────────────────┼───────────────┤                       
  │ 学习目标 │ 只学 φ（单一 kernel）      │ 同时学 V 和 Φ │                       
  ├──────────┼────────────────────────────┼───────────────┤                       
  │ 正则化   │ RKHS Tikhonov              │ 无正则化      │                       
  ├──────────┼────────────────────────────┼───────────────┤                       
  │ Loss     │ Quadratic (least squares)  │ 弱形式 PDE    │                       
  ├──────────┼────────────────────────────┼───────────────┤                       
  │ 数据     │ Mean-field solution u(x,t) │ 粒子分布快照  │                       
  └──────────┴────────────────────────────┴───────────────┘                       
  这解释了我们失败的原因：                                                        
  1. 我们尝试同时学 V 和 Φ — 理论上不可行                                         
  2. 我们没有使用 RKHS 正则化 — 导致 ill-posed                                    
                                                                                  
  ---                                                                             
  📐 Convergence Rate (Theorem 3.7)                                               
                                                                                  
  最优收敛率：                                                                    
  $$|\hat{\phi}{n,M,\infty} - \phi|{\mathbb{H}} \lesssim (\Delta x)^{\frac{\alpha 
  s}{s+1}}$$                                                                      
                                                                                  
  其中：                                                                          
  - α = 数值积分器的阶（Riemann sum: α=1, Trapezoidal: α=2）                      
  - s = approximation error 的衰减阶（φ ∈ W^{k,∞} 时 s=k）                        
                                                                                  
  最优维度：n ≈ (Δx)^{-α/(s+1)}                                                   
                                                                                  
  ---                                                                             
  ⚡ 对本项目的直接启示                                                           
                                                                                  
  1. 不要同时学 V 和 Φ — 假设 V 已知，只学 Φ                                      
  2. 实现 RKHS 正则化 — 这是成功的关键                                            
  3. 参考参数配置：                                                               
    - M = 300 空间点                                                              
    - ν = 0.01-1.0                                                                
    - 对 smooth kernel 期望 <2% 误差
---

#### Paper 5: Nonparametric learning of kernels in nonlocal operators
- **作者**: F. Lu, Qingci An, Yue Yu
- **发表**: J. Peridynamics and Nonlocal Modeling, 2023
- **链接**: [arXiv:2205.11006](https://arxiv.org/abs/2205.11006)

**核心发现**:
- Kernel learning 是 **ill-posed 或 ill-defined inverse problem**
- 存在 modeling errors 或 measurement noises 时，estimators **会发散**
- 解决方案: **Data adaptive RKHS Tikhonov regularization**

---

### 3.3 相关工作

#### Paper 6: A data-adaptive prior for Bayesian learning of kernels in operators
- **发表**: JMLR 2024
- **链接**: [JMLR vol.25 no.317](https://jmlr.org/)

---

## 4. 关键发现：与本项目的直接关联

### 4.1 解释我们的失败

我们的 Hub 记录:
- **K1**: Loss→0 但误差 >90%
- **信念2❌**: 弱形式方法无法区分不同的 (V, Φ) 对

**Fei Lu 理论解释**:
> "it is not possible, in general, to identify **both** the confining and interaction potentials from a single-particle observation"

**这直接解释了 MVP-1.0/1.1/1.2 失败的根本原因**。

---

### 4.2 Identifiability 条件总结

| 条件 | 描述 | 我们是否满足 |
|------|------|-------------|
| **Coercivity** | 积分算子严格正定 | ❓ 未验证 |
| **Ergodicity** | 系统是遍历的 | ❓ 需检查 |
| **RKHS 正则化** | 在 RKHS 中优化 | ❌ 未使用 |
| **单一势函数** | 只学 V 或只学 Φ | ❌ 同时学两个 |

---

### 4.3 方法对比

| 组件 | Fei Lu 方法 | 我们当前方法 |
|------|------------|------------|
| **数据** | 轨迹数据 / 分布快照 | 无标签快照 |
| **Loss** | Least squares + RKHS 正则化 | 弱形式 PDE loss |
| **正则化** | Data-adaptive RKHS Tikhonov | 无正则化 |
| **理论保障** | Coercivity → Identifiability | 缺失 |

---

## 5. 行动建议

### 5.1 立即行动 (P0)

1. **不要同时学习 V 和 Φ** — 理论上不可行
   - 方案 A: 固定 V，只学 Φ
   - 方案 B: 使用已知 Φ 的系统验证方法

2. **实现 RKHS 正则化** — 这不是可选的，是必须的
   - 参考: [arXiv:2205.11006](https://arxiv.org/abs/2205.11006)

3. **阅读 identifiability 论文** — 理解何时唯一解存在
   - [arXiv:2106.05565](https://arxiv.org/abs/2106.05565)

### 5.2 后续行动 (P1)

1. **验证 coercivity condition** — 检查我们的系统是否 ergodic
2. **使用 weighted L² space** — 比 unweighted 更准确
3. **多系统联合学习** — 不同 V 的系统共享 Φ

---

## 6. 关键论文链接汇总

| 论文 | 链接 | 重要性 |
|------|------|--------|
| Identifiability (Lang & Lu) | [arXiv:2106.05565](https://arxiv.org/abs/2106.05565) | 核心理论 |
| Coercivity (Li & Lu) | [arXiv:2011.10480](https://arxiv.org/abs/2011.10480) | 核心理论 |
| Mean-field learning (Lang & Lu) | [arXiv:2010.15694](https://arxiv.org/abs/2010.15694) | 方法参考 |
| Nonlocal operators (Lu et al.) | [arXiv:2205.11006](https://arxiv.org/abs/2205.11006) | RKHS 正则化 |
| Identifiability in SPA | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0304414920303951) | 理论补充 |
| Network inference | [arXiv:2402.08412](https://arxiv.org/abs/2402.08412) | 多系统学习 |

---

## 7. 待阅读论文 PDF

- [ ] [arXiv:2106.05565](https://arxiv.org/abs/2106.05565) — identifiability 的详细数学表述
- [ ] [arXiv:2011.10480](https://arxiv.org/abs/2011.10480) — coercivity condition 的具体形式
- [ ] [arXiv:2010.15694](https://arxiv.org/abs/2010.15694) — 实验的具体参数配置 (N, M, L, σ)
- [ ] [arXiv:2205.11006](https://arxiv.org/abs/2205.11006) — RKHS Tikhonov regularization 实现细节

---

> **Session 结论**: Fei Lu 的研究提供了理论基础，解释了我们实验失败的原因（同时学习 V 和 Φ 一般不可行），并指出了解决方案（RKHS 正则化 + 简化问题）。
