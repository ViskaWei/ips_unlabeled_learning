# 🧠 IPS Unlabeled Hub
> **ID:** EXP-20260128-ips_unlabeled-hub | **Status:** ⚠️ 阻塞 (Identifiability 问题)
> **Date:** 2026-01-28 | **Update:** 2026-01-29 (MVP-2.0 Identifiability 问题确认)

| # | 💡 共识[抽象洞见] | 证据 | 决策 |
|---|----------------------------|----------------|------|
| K1 | ❌ **否定**：弱形式 loss 存在根本性 identifiability 问题 | MVP-1.0/1.1/1.2: Loss→0 但误差>90% | 需要完全不同的方法 |
| K2 | ✅ 确认：NN 可稳定计算梯度和 Laplacian (AD) | MVP-1.0: 无数值问题 | AD 方案可继续使用 |
| K3 | ❌ **确认**：V-Φ 可相互补偿导致 loss=0 | MVP-1.1: 添加约束后仍失败 | 约束方法无效 |
| K4 | ✅ **新发现**：原 loss 公式系数有误 | MVP-1.2: 正确公式是 J_diss - (σ²/2)J_lap + dE = 0 | 但修正后仍无法学习 |
| K5 | ❌ **理论确认**：同时学习 V 和 Φ 一般不可行 | Fei Lu 论文: "not possible in general to identify both" | **必须固定其中之一** |
| K6 | ✅ **理论支持**：RKHS 正则化是必须的 | Fei Lu: "inverse problem is ill-posed" | RKHS Tikhonov 正则化 |
| K7 | ✅ **理论支持**：Coercivity condition 保证 identifiability | Li & Lu 2021: ergodic → coercivity | 检查系统是否 ergodic |
| K8 | ✅ **修复**：PDE solver 需要 semi-implicit 方案 | Quadratic kernel CFL=4.5 | ✅ 已修复 |
| K9 | ✅ **确认**：Error functional 公式正确 | Oracle test: c_opt=0.97≈1.0 (稳定 solver) | 公式无误 |
| K10 | ⚠️ **关键发现**：**B-spline 在卷积空间共线** | 真实 φ 在基中仍不唯一恢复 | **需要 RKHS 正则化** |
| K11 | ⚠️ **Identifiability**：不同系数可产生等效 φ | c=[0,1,0] vs c=[-2.2,3.7,-0.9] 误差相近 | 解不唯一 |
| K12 | ❌ **确认：Fundamental Identifiability** | [φ_true, r] 的解不是 [1,0] 而是 [0.63,-0.04] | **PDE 残差最小 ≠ φ 正确** |
| K13 | ✅ **确认：Error functional 公式正确** | Oracle test 通过 (c=0.96≈1.0) | 问题不在公式 |

**🦾 现阶段信念 [≤10条，写"所以呢"]**
- **信念1**：标准方法（MLE/MSE）需要轨迹信息，无法直接应用 → 必须开发 trajectory-free 方法
- **信念2** ❌：~~弱形式 PDE 是绕过轨迹需求的数学基础~~ → **弱形式方法无法区分不同的 (V, Φ) 对**
- **信念3** ✅：神经网络可通过 AD 计算任意阶导数 → AD 计算稳定可靠
- **信念4** ❌：~~轨迹无关方法只需知道分布"整体漂移"与"扩散"~~ → **实验表明仅此不够，V-Φ 可相互抵消**
- **信念5** ❌：~~添加约束（如 V(0)=0）可保证唯一性~~ → **MVP-1.1 表明简单约束无效**
- **信念6**：高维d和大规模N时，需要GPU并行和策略性采样 → 影响训练流程设计
- **信念7** ❗：Loss=0 不保证正确解 → **验证 loss 设计时必须检查势函数形状，不能只看 loss 值**
- **信念8** ❗：原论文 loss 公式有误 → **正确公式是 J_diss - (σ²/2)J_lap + dE = 0，但即使修正也无法解决 identifiability**
- **信念9** ✅ (理论确认)：**必须固定 V 或 Φ 之一** → Fei Lu: "not possible in general to identify both potentials"
- **信念10** ✅ (理论确认)：**必须使用 RKHS 正则化** → Fei Lu: "inverse problem is ill-posed, regularization required"

**👣 下一步最有价值** (2026-01-29 更新 - MVP-2.0 Fundamental Identifiability 确认)
- 🔴 **P0**：**决策点：是否继续 Fei Lu 方向？**
  - ✅ Error functional 公式正确 (Oracle c=0.96)
  - ✅ PDE solver 已修复 (semi-implicit)
  - ❌ **Fundamental identifiability**: 不同 φ 可产生相同 PDE 残差
  - ❌ 正交基、minimal basis 都无法解决
  - **选项 A**: 深入理解 RKHS 正则化如何解决 identifiability
  - **选项 B**: 换方向（如 trajectory-based 方法作为 baseline）
- 🟡 **P1**：如果选 A，需要理解 RKHS
  - [ ] Fei Lu 论文 Section 3 — 如何构造 data-adaptive basis
  - [ ] 理解 "identifiability through regularization" 的数学原理
- 🟡 **P1**：如果选 B，需要实现 baseline
  - [ ] Trajectory-based MLE（需要轨迹标签）
  - [ ] 验证方法论在有标签情况下是否正确
- 🟢 **P2**：**多系统联合学习** — 增加约束来解决 identifiability

> **权威数字**：Best=-；Baseline=-；Δ=-；条件=待定义

| 模型/方法 | 指标值 | 配置 | 备注 |
|-----------|--------|------|------|
| Baseline (真实势函数) | 0% error | Oracle | 上界 |
| Random NN | ~100% error | 未训练 | 下界 |

---

## 1) 🌲 核心假设树
```
🌲 核心: 能否从无标签集合数据学习交互势函数Φ和动力学势函数V？
│
├── Q1: Trajectory-free loss 是否有效？（数学推导层面）
│   ├── Q1.1: 弱形式PDE推导是否正确？ → ⚠️ **存疑** (MVP-1.0: loss→0 但势函数错误)
│   │   └── 发现：系数 -2 可能有误；或缺少 identifiability 约束
│   ├── Q1.2: 离散化误差是否可控？ → ✅ 可控 (收敛表现良好)
│   │   └── 结论：问题不在离散化
│   ├── Q1.3: 损失函数是否有唯一最小值？ → ❌ **否定** (V-Φ 可相互抵消)
│   │   └── 发现：存在多个使 loss=0 的解，需要额外约束
│   └── Q1.4: 收敛速率如何？ → ❓ 无法评估 (因 Q1.3 失败)
│       └── 阻塞：先解决 identifiability 问题
│
├── Q2: 神经网络表示是否合适？（实现层面）
│   ├── Q2.1: NN能否表示势函数类？ → ✅ 能 (MLP 可学习任意平滑势函数)
│   ├── Q2.2: AD计算梯度/Laplacian是否稳定？ → ✅ 稳定 (MVP-1.0: 无数值问题)
│   │   └── 结论：AD 计算可靠，可继续使用
│   ├── Q2.3: 对称性约束是否有效？ → ✅ 有效 (Φ(x)=½(Φ̃(x)+Φ̃(-x)) 工作正常)
│   └── Q2.4: 计算可行性如何？ → ⏳ 待验证 (当前仅测试 N=5,10)
│       └── 子问题：N 和 d 较大时，计算效率如何？
│
├── Q3: 数据需求如何？（数据层面）
│   ├── Q3.1: 需要多少时间快照L？ → ⏳ 待验证
│   ├── Q3.2: 每个快照需要多少样本M？ → ⏳ 待验证
│   ├── Q3.3: 粒子数N如何影响学习？ → ⏳ 待验证
│   └── Q3.4: 数据覆盖度要求？ → ⏳ 待验证
│       └── 子问题：数据分布是否覆盖关键区域（影响可辨识性）？
│
├── Q4: 物理直觉与泛化（物理层面）
│   ├── Q4.1: 能量/耗散/扩散三项的物理意义？ → ⏳ 待验证
│   │   └── 子问题：如何从粒子临近分布演化角度理解弱形式损失？
│   ├── Q4.2: 噪声σ的影响？ → ⏳ 待验证
│   │   └── 子问题：σ较大/较小时，相互作用势如何影响整体分布？
│   ├── Q4.3: 训练是否稳定收敛？ → ⏳ 待验证
│   └── Q4.4: 学到的势函数能否泛化？ → ⏳ 待验证
│
└── Q5: 实际应用约束（约束层面）
    ├── Q5.1: 测量噪声如何处理？ → ⏳ 待验证
    ├── Q5.2: 多模态分布（簇分布）如何处理？ → ⏳ 待验证
    └── Q5.3: V和σ未知时的影响？ → ⏳ 待验证
        └── 子问题：若V或σ也未知，如何区分V与Φ的作用？

Legend: ✅ 已验证 | ❌ 已否定 | 🔆 进行中 | ⏳ 待验证 | 🗑️ 已关闭
```

## 2) 口径冻结（唯一权威）
| 项目 | 规格 |
|---|---|
| 系统类型 | 交互粒子系统 (IPS) with SDE |
| 维度 d | 1D (初始), 扩展到 2D |
| 粒子数 N | 10-100 |
| 时间快照 L | 10-50 |
| 每快照样本 M | 100-1000 |
| 真实势函数 | Φ: Lennard-Jones / Morse; V: Harmonic |
| Metric | Relative L² error on Φ, V, ∇Φ, ∇V |
| 噪声 σ | 0.1 (可调) |
| Seed | 42 |

---

## 3) 当前答案 & 战略推荐

### 3.1 战略推荐
- **推荐路线：Route A (NN + Trajectory-free loss)**
- 需要关闭的 Gate：Gate-1 (Loss有效性), Gate-2 (NN架构)

| Route | 一句话定位 | 当前倾向 | 关键理由 | 需要的 Gate |
|---|---|---|---|---|
| **Route A** | NN + Trajectory-free loss | 🟢 推荐 | 论文核心方法 | Gate-1, Gate-2 |
| Route B | Kernel + Trajectory-free loss | 🟡 | 备选，更有理论保障 | Gate-3 |

### 3.2 分支答案表
| 分支 | 当前答案 | 置信度 | 决策含义 | 证据 |
|---|---|---|---|---|
| Q1 (Loss有效性) | 理论推导合理，待实验验证 | 🟡 | 若验证则继续 | - |
| Q2 (NN表示) | 标准MLP应该足够 | 🟡 | 确定后固定架构 | - |
| Q3 (数据需求) | 未知 | 🔴 | 影响实验设计 | - |

---

## 4) 洞见汇合
| # | 洞见 | 观察 | 解释 | 决策影响 | 证据 |
|---|---|---|---|---|---|
| I1 | 弱形式损失包含三类核心项 | 能量变动、扩散积分、耗散积分 | 分别对应分布整体水平、随机热运动、粒子间斥力/引力 | 需要分别验证每项的有效性 | session-1 |
| I2 | 时间间隔不均匀是实际约束 | ∆tₗ可能不均匀，非小步长极限 | Weak-form积分需处理较大步长下的近似误差 | 实验设计需考虑Δt敏感性 | session-1 |
| I3 | 数据覆盖度影响可辨识性 | 数据分布需覆盖关键区域 | 否则无法保证Φ,V的唯一解 | 数据生成需确保充分覆盖 | session-1 |
| I4 | σ→0时系统趋于确定性 | 扩散项衰减，粒子依势流动 | 需注意弱形式里扩散项的影响 | 实验需测试不同σ值 | session-1 |
| **I5** | **Loss=0 不保证正确解** ❗ | Loss→1e-7 但 V err=162%, Φ err=94% | V-Φ 可相互抵消使 loss=0 | **必须检查势函数形状** | MVP-1.0 |
| **I6** | **V-Φ 存在 identifiability 问题** ❗ | 不同配置下误差分布不同但都失败 | 弱形式 loss 可能有多个使其=0的解 | **需要额外约束** | MVP-1.0 |
| **I7** | **即使固定 V，Φ 仍学不对** ❗ | Φ-only 误差 78%，loss=0.018 | 问题不仅是 V-Φ trade-off | **loss 公式本身可能有问题** | MVP-1.0b |
| **I8** | AD 计算稳定可靠 ✅ | 无数值爆炸/消失 | PyTorch autograd 工作正常 | AD 方案可继续使用 | MVP-1.0 |
| **I9** | **同时学 V 和 Φ 理论上不可行** 📖 | Fei Lu 理论证明 | "not possible in general to identify both" | **必须固定其中之一** | session-2 |
| **I10** | **逆问题 ill-posed，必须正则化** 📖 | Fei Lu: estimators diverge w/o regularization | RKHS 空间中才有唯一解 | **实现 RKHS Tikhonov** | session-2 |
| **I11** | **Coercivity condition 保证 identifiability** 📖 | 积分算子严格正定 ⟺ 唯一解 | Ergodic 系统满足 coercivity | **检查系统是否 ergodic** | session-2 |
| **I12** | **Weighted L² space 优于 unweighted** 📖 | Fei Lu 实验结果 | Data-adaptive measure 更准确 | 考虑 weighted loss | session-2 |

---

## 5) 决策空白（Decision Gaps）
| DG | 我们缺的答案 | 为什么重要 | 什么结果能关闭它 | 决策规则 |
|---|---|---|---|---|
| DG1 | Loss是否在简单系统上work | 验证方法论基础 | 1D系统误差<10% | If work → 扩展; Else → 检查推导 |
| DG2 | NN架构选择 | 影响所有后续实验 | 稳定收敛+低误差 | If MLP work → 固定; Else → 尝试其他 |
| DG3 | 数据量需求 | 影响实验设计 | 确定L,M,N的最小值 | 根据结果设置baseline |

---

## 6) 设计原则

### 6.1 已确认原则
| # | 原则 | 建议 | 适用范围 | 证据 |
|---|---|---|---|---|
| P1 | 使用对称化 Φ(x)=½(Φ̃(x)+Φ̃(-x)) | 做 | 所有交互势学习 | 物理约束 |
| P2 | 使用 AD 计算导数而非有限差分 | 做 | 所有NN实验 | 避免离散误差 |
| P3 | Euler-Maruyama dt=0.01 对平滑势函数足够 | 做 | 数据生成 | MVP-0.0: KL=0.0005 |

### 6.2 待验证原则
| # | 原则 | 初步建议 | 需要验证 |
|---|---|---|---|
| P4 | 能量项需要时间配对 | 使用连续时间快照 | MVP-1.0 |
| P5 | 策略性采样可加快收敛 | 时间批次或空间分层抽样 | MVP-1.0 |
| P6 | 对称化约束提升物理合理性 | Φ(x)=½(Φ̃(x)+Φ̃(-x)) | MVP-1.0 |
| P7 | 先验证1D再扩展到高维 | 从1D开始，逐步到2D/高维 | MVP-1.0 → MVP-3.0 |

### 6.3 关键数字速查
| 指标 | 值 | 条件 | 来源 |
|---|---|---|---|
| 目标误差 | <10% | 相对L²误差 | 论文要求 |
| SDE 验证 KL | 0.0005 | OU baseline | MVP-0.0 |
| SDE 验证方差误差 | 0.42% | OU baseline | MVP-0.0 |

---

## 7) 指针
| 类型 | 路径 | 说明 |
|---|---|---|
| 🗺️ Roadmap | `./ips_unlabeled_roadmap.md` | Decision Gates + MVP 执行 |
| 📗 Experiments | `./exp_*.md` | 单实验报告 |
| 📊 Prompts | `./prompts/` | Coding Prompt |
| 💬 Sessions | `./sessions/` | 对话记录 |
| └─ Session 1 | `./sessions/1.md` | 初始问题框架 |
| └─ Session 2 | `./sessions/2_fei_lu_literature.md` | Fei Lu 文献调研 |

---

## 8) 🔬 Fei Lu 论文综述（文献调研）

> **来源**: [Fei Lu @ JHU](https://math.jhu.edu/~feilu/research.html) | **调研日期**: 2026-01-28
> **详细 Session**: `./sessions/2_fei_lu_literature.md`

### 8.1 关键论文列表

| # | 论文 | 年份 | 核心贡献 | 与本项目关系 |
|---|------|------|---------|-------------|
| **P1** | [Learning interaction kernels in mean-field equations](https://arxiv.org/abs/2010.15694) (Lang & Lu, SIAM J. Sci. Comput.) | 2022 | Mean-field PDE + 分布数据 | **最相关**：与我们的 trajectory-free 设定一致 |
| **P2** | [Identifiability of interaction kernels](https://arxiv.org/abs/2106.05565) (Lang & Lu, FODS) | 2023 | **Identifiability 理论** | 核心理论：解释为什么 loss=0 不保证唯一解 |
| **P3** | [On the coercivity condition](https://arxiv.org/abs/2011.10480) (Li & Lu, Stoch. Dynamics) | 2021 | **Coercivity → Identifiability** | 核心理论：何时有唯一解 |
| **P4** | [On the identifiability of interaction functions](https://www.sciencedirect.com/science/article/pii/S0304414920303951) (Li, Lu, Maggioni et al., SPA) | 2021 | 系统性 identifiability 分析 | 理论基础 |
| **P5** | [Nonparametric learning of kernels in nonlocal operators](https://arxiv.org/abs/2205.11006) (Lu, An, Yu) | 2023 | **RKHS Tikhonov 正则化** | 方法参考：必须的正则化 |
| **P6** | [Network inference](https://arxiv.org/abs/2402.08412) (Lang, Wang, Lu, Maggioni) | 2024 | 多系统联合学习 | 参考：共享 Φ 的多系统 |

### 8.2 核心理论发现

#### ❌ 关键否定结论
> **"It is not possible, in general, to identify both the confining and interaction potentials from a single-particle observation."**

这直接解释了我们 MVP-1.0/1.1/1.2 的失败：**同时学习 V 和 Φ 理论上不可行**。

#### Identifiability 条件 (来自 P2, P3, P4)

| 条件 | 数学表述 | 物理意义 |
|------|---------|---------|
| **Coercivity** | 积分算子 K 严格正定 | 数据足够"丰富" |
| **Ergodicity** | 系统是遍历的 | 长时间后覆盖状态空间 |
| **RKHS 正则化** | 在 RKHS 闭包中优化 | 限制解空间保证唯一性 |

**关键定理** (Li & Lu 2021):
> "Coercivity condition is sufficient for identifiability and becomes **necessary** when N → ∞."

### 8.3 Coercivity Condition 详解

**定义**: Coercivity ⟺ 积分算子严格正定
- 积分核 K(r,r') 必须是 **strictly positive definite**
- 证明方法: **Müntz type theorems**

**何时成立**:
- 系统是 **ergodic** (遍历的)
- Interaction function 满足一定条件使系统 ergodic

**失败时**:
- Loss 有多个使其 = 0 的解
- Estimator 发散 (divergent)
- 无法唯一确定 interaction function

### 8.4 RKHS 正则化 (来自 P5)

**问题**: Kernel learning 是 ill-posed/ill-defined inverse problem
- Modeling errors 或 measurement noises 导致 estimators **发散**

**解决方案**: **Data adaptive RKHS Tikhonov regularization**
- 在 RKHS 中优化而非普通 L²
- 使用 data-adaptive 的 reproducing kernel
- Weighted L² space 优于 unweighted

### 8.5 收敛速率

| 设定 | 收敛速率 | 来源 |
|------|---------|------|
| Mean-field (分布数据) | = 数值积分器的阶 | P1 |
| SDE (轨迹数据) | 1D 非参数回归 min-max rate | P3 |
| 离散化误差 | O(Δt^{1/2}) | P3 |

**关键**: 收敛速率**独立于状态空间维度** (只依赖 interaction 的内在维度)

### 8.6 成功案例 (来自 P1)

| 例子 | Kernel 类型 | 结果 |
|------|------------|------|
| Opinion dynamics | Piecewise linear | ✅ 成功 |
| Granular media | Quadratic (smooth) | ✅ 成功 |
| Aggregation-diffusion | Repulsive-attractive | ✅ 成功 |

### 8.7 对本项目的启示

| 优先级 | 启示 | 具体行动 |
|--------|------|---------|
| 🔴 **P0** | **固定 V 或 Φ 之一** | 同时学习理论上不可行 |
| 🔴 **P0** | **实现 RKHS 正则化** | 这是 identifiability 的必要条件 |
| 🔴 **P0** | **使用 weighted L² space** | 比 unweighted 更准确 |
| 🟡 **P1** | **验证 coercivity** | 检查系统是否 ergodic |
| 🟡 **P1** | **多系统联合学习** | 不同 V 共享 Φ 可能提供额外约束 |

### 8.8 待阅读论文 PDF

- [ ] [arXiv:2106.05565](https://arxiv.org/abs/2106.05565) — identifiability 的详细数学表述
- [ ] [arXiv:2011.10480](https://arxiv.org/abs/2011.10480) — coercivity condition 的具体形式
- [x] [arXiv:2010.15694](https://arxiv.org/abs/2010.15694) — **已详细阅读** (2026-01-28)
- [ ] [arXiv:2205.11006](https://arxiv.org/abs/2205.11006) — RKHS Tikhonov regularization 实现细节

---

## 9) 📖 Fei Lu 论文详细笔记（2026-01-28 详细阅读）

> **来源**: "Learning interaction kernels in mean-field equations of 1st-order systems of interacting particles" (Lang & Lu, SIAM J. Sci. Comput. 2022)

### 9.1 问题设定差异

**Fei Lu 的 Mean-field 方程（无 V）**:
$$\partial_t u = \nu \Delta u + \nabla \cdot [u(K_\phi * u)], \quad x \in \mathbb{R}^d, t > 0$$

**我们的 SDE（有 V 和 Φ）**:
$$dX_t^i = -\nabla V(X_t^i) dt - \frac{1}{N} \sum_j \nabla \Phi(X_t^i - X_t^j) dt + \sigma dB_t^i$$

**关键差异**：
| 方面 | Fei Lu | 我们 |
|------|--------|-----|
| 外势 V | **无** | 有 |
| 数据 | PDE 解 u(x,t) | 粒子位置快照 |
| 学习目标 | 只学 φ | 同时学 V 和 Φ |

### 9.2 Error Functional（Eq 2.3）

$$\mathcal{E}(\psi) = \frac{1}{T} \int_0^T \int_{\mathbb{R}^d} \left[ |K_\psi * u|^2 u + 2\partial_t u (\Psi * u) + 2\nu \nabla u \cdot (K_\psi * u) \right] dx\, dt$$

**优势**（Remark 2.5）:
1. 不需要空间导数 ∇u, Δu
2. 利用 u(·,t) 是概率密度，可以用 Monte Carlo 近似

### 9.3 成功实验配置（Table 2）

| 参数 | 值 | 备注 |
|------|-----|------|
| 空间域 Ω | [-10, 10] | |
| 时间 T | 1.0 | |
| 真实解网格 | dt=0.001, dx=20/3000 | 高精度 |
| 数据网格 | Δx=10dx | M=300 空间点 |
| 粘性 ν | 0.01-1.0 | 对应 σ²/2 |
| B-spline 维度 | 3-50 | L-curve 选择 |

### 9.4 成功案例误差

| 例子 | Kernel | ν | L²(ρ̄_T) 误差 | RKHS 误差 |
|------|--------|---|-------------|-----------|
| **Cubic** | φ(r)=3r² (smooth) | 1.0 | **1.90%** | 0.43% |
| Opinion | piecewise linear | 0.1 | 36.74% | 8.10% |
| Repulsion-attraction | singular | 0.01 | 49.06% | 4.36% |

**关键发现**：smooth kernel 可达 <2% 误差！

### 9.5 Tikhonov 正则化（Section 2.3）

$$\mathcal{E}_\lambda(\psi) = \mathcal{E}(\psi) + \lambda |||\psi|||^2$$

- 正则化范数：RKHS norm 或 H¹ norm
- λ 选择：**L-curve 方法**（最大化曲率）

### 9.6 收敛速率（Theorem 3.7）

$$\|\hat{\phi}_{n,M,\infty} - \phi\|_{\mathbb{H}} \lesssim (\Delta x)^{\frac{\alpha s}{s+1}}$$

- α = 数值积分器阶（Trapezoidal: α=2）
- s = approximation error 衰减阶（B-spline degree p 时 s=p）

### 9.7 对 MVP-2.0 的启示

1. **去掉 V**：设 V=0，只学 φ（论文方程本身无 V）
2. **估计 u(x,t)**：从粒子数据用 KDE 估计
3. **使用 B-spline**：替代 NN
4. **实现 Tikhonov**：必须正则化
5. **参考配置**：M=300, ν=0.1-1.0

---

## 9) 变更日志
| 日期 | 变更 | 影响 |
|---|---|---|
| 2026-01-28 | 创建 Hub | 立项 |
| 2026-01-28 | MVP-0.0 完成: SDE 数据生成器验证通过 | 确认数据生成可靠性，可进入 MVP-1.0 |
| 2026-01-28 | 根据 session-1 完善问题树和洞见 | 补充数学推导、物理直觉、计算机验证三个层面的问题 |
| 2026-01-28 | **Expert Review** 添加 | PI 评审 Phase 1 工作，调整下一步优先级：实验设定调整 > 简化问题 |
| 2026-01-28 | **Fei Lu 论文综述** 添加 | 文献调研：identifiability 理论、convergence rate、RKHS 正则化 |
| 2026-01-28 | **Session-2 文献调研** | 深入调研 Fei Lu 论文：coercivity condition、RKHS 正则化必要性、V-Φ 不可同时学习 |
| 2026-01-28 | **Hub 重大更新** | K5-K7 新增（理论确认）；信念9-10 更新；I9-I12 新增；P0 优先级调整为"固定 V/Φ + RKHS 正则化" |

<details>
<summary><b>附录：术语表</b></summary>

| 术语 | 定义 |
|---|---|
| IPS | Interacting Particle System 交互粒子系统 |
| Φ | Interaction potential 交互势函数 |
| V | Kinetic/external potential 动力学势函数 |
| μ_t^N | 经验分布 (1/N)Σδ_{X_t^i} |
| Trajectory-free | 不需要粒子轨迹配对信息 |
| Weak-form PDE | 弱形式偏微分方程 |

</details>
