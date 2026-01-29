# 🧠 IPS Unlabeled Hub
> **ID:** EXP-20260128-ips_unlabeled-hub | **Status:** ⚠️ 遇阻
> **Date:** 2026-01-28 | **Update:** 2026-01-28 (MVP-1.0 失败)

| # | 💡 共识[抽象洞见] | 证据 | 决策 |
|---|----------------------------|----------------|------|
| K1 | ❌ **否定**：弱形式 loss 存在根本性 identifiability 问题 | MVP-1.0/1.1/1.2: Loss→0 但误差>90% | 需要完全不同的方法 |
| K2 | ✅ 确认：NN 可稳定计算梯度和 Laplacian (AD) | MVP-1.0: 无数值问题 | AD 方案可继续使用 |
| K3 | ❌ **确认**：V-Φ 可相互补偿导致 loss=0 | MVP-1.1: 添加约束后仍失败 | 约束方法无效 |
| K4 | ✅ **新发现**：原 loss 公式系数有误 | MVP-1.2: 正确公式是 J_diss - (σ²/2)J_lap + dE = 0 | 但修正后仍无法学习 |

**🦾 现阶段信念 [≤10条，写"所以呢"]**
- **信念1**：标准方法（MLE/MSE）需要轨迹信息，无法直接应用 → 必须开发 trajectory-free 方法
- **信念2** ❌：~~弱形式 PDE 是绕过轨迹需求的数学基础~~ → **弱形式方法无法区分不同的 (V, Φ) 对**
- **信念3** ✅：神经网络可通过 AD 计算任意阶导数 → AD 计算稳定可靠
- **信念4** ❌：~~轨迹无关方法只需知道分布"整体漂移"与"扩散"~~ → **实验表明仅此不够，V-Φ 可相互抵消**
- **信念5** ❌：~~添加约束（如 V(0)=0）可保证唯一性~~ → **MVP-1.1 表明简单约束无效**
- **信念6**：高维d和大规模N时，需要GPU并行和策略性采样 → 影响训练流程设计
- **信念7** ❗：Loss=0 不保证正确解 → **验证 loss 设计时必须检查势函数形状，不能只看 loss 值**
- **新信念8** ❗：原论文 loss 公式有误 → **正确公式是 J_diss - (σ²/2)J_lap + dE = 0，但即使修正也无法解决 identifiability**
- **新信念9** ❓：可能需要**多系统联合学习**或**已知 V 的简化问题**才能学习 Φ

**👣 下一步最有价值** (根据 Expert Review 2026-01-28 调整)
- 🔴 **P0**：**调整实验设定**（最高优先）：当前 N=5, M=30, σ=0.1 设定下误差太大，无法做有价值的估计
  - ↗️ 增加样本量 M: 200 → 1000+
  - ↘️ 降低噪声 σ: 0.1 → 0.01
  - ↗️ 增加粒子数 N: 5-10 → 50-100
  - 📖 参考论文中成功实验的配置
- 🔴 **P0**：**深入理论分析**：阅读论文中关于 identifiability 的详细讨论，理解唯一解条件
- 🟡 **P1**：考虑 **RKHS 正则化**（automatic kernel）→ 我去年的工作，计划中的另一方法
- 🟡 **P1**：考虑 **多系统联合学习**：不同 V 的系统共享 Φ → NSF proposal 项目之一
- 🟢 **P2**：Route B (Kernel 方法) 作为替代方案 → MVP-2.0

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

---

## 8) 🔬 Fei Lu 论文综述（文献调研）

> **来源**: [Fei Lu @ JHU](https://math.jhu.edu/~feilu/research.html) | **调研日期**: 2026-01-28

### 8.1 关键论文列表

| # | 论文 | 年份 | 核心贡献 | 与本项目关系 |
|---|------|------|---------|-------------|
| **P1** | [Learning interaction kernels in mean-field equations](https://arxiv.org/abs/2010.15694) (Lang & Lu, SIAM J. Sci. Comput.) | 2022 | Mean-field PDE + 分布数据 | **最相关**：与我们的 trajectory-free 设定一致 |
| **P2** | [Identifiability of interaction kernels](https://arxiv.org/abs/2106.05565) (Lang & Lu, FODS) | 2023 | **Identifiability 理论** | 核心理论：解释为什么 loss=0 不保证唯一解 |
| **P3** | [Learning in stochastic systems from multiple trajectories](https://arxiv.org/abs/2007.15174) (Lu, Maggioni & Tang, Found. Comput. Math.) | 2021 | SDE + 轨迹数据 | 参考：convergence rate 分析框架 |
| **P4** | [Heterogeneous systems](https://arxiv.org/abs/1910.04832) (Lu et al., JMLR) | 2021 | 异质系统 | 参考：多系统联合学习 |

### 8.2 Identifiability 核心结论（来自 P2）

**问题**：Quadratic loss 何时有唯一最小值？

**关键结论**：
1. Identifiable function space = **RKHS closure** of the integral operator of inversion
2. Finite particles vs Infinite particles 有**关键区别**
3. Inverse problem is **ill-posed** → **必须正则化**
4. **Weighted L² space** 比 unweighted L² space 产生更准确的估计

### 8.3 Convergence Rate（来自 P1, P3）

**Mean-field case (P1)**：
- 收敛速率 = **数值积分器的阶** (numerical integrator's order)
- 经验误差 E_{M,∞} 收敛速率: `2αs/(s+1)`
- Δt → 0 时最优收敛

**SDE case (P3)**：
- 收敛速率 = **1D 非参数回归的 min-max rate**
- **独立于状态空间维度**（关键优势！）
- 离散化误差 = O(Δt^{1/2})

**理论分析方向**：convergence rate as M → ∞ (样本量增加)

### 8.4 成功案例（来自 P1）

| 例子 | Kernel 类型 | 结果 |
|------|------------|------|
| Opinion dynamics | Piecewise linear | ✅ 成功：highly accurate solutions |
| Granular media | Quadratic (smooth) | ✅ 成功：optimal rate of convergence |
| Aggregation-diffusion | Repulsive-attractive | ✅ 成功：accurate free energy |

**关键**：估计器能 reproduce highly accurate solutions and free energy。

### 8.5 对本项目的启示

| 优先级 | 启示 | 具体行动 |
|--------|------|---------|
| 🔴 **P0** | 必须实现 **RKHS 正则化** | 这是 identifiability 的关键，论文核心方法 |
| 🔴 **P0** | 使用 **weighted L² space** | 比 unweighted 更准确 |
| 🟡 **P1** | 理论分析方向 | convergence rate as M → ∞ |
| 🟡 **P1** | 获取 PDF 详细参数 | 需要具体的 N, M, σ 配置 |

### 8.6 待获取信息

⚠️ **需要阅读论文 PDF**：
- [ ] P1 中实验的具体参数配置 (N, M, L, σ)
- [ ] P2 中 identifiability condition 的详细数学表述
- [ ] P3 中 coercivity condition 的具体形式

---

## 9) 变更日志
| 日期 | 变更 | 影响 |
|---|---|---|
| 2026-01-28 | 创建 Hub | 立项 |
| 2026-01-28 | MVP-0.0 完成: SDE 数据生成器验证通过 | 确认数据生成可靠性，可进入 MVP-1.0 |
| 2026-01-28 | 根据 session-1 完善问题树和洞见 | 补充数学推导、物理直觉、计算机验证三个层面的问题 |
| 2026-01-28 | **Expert Review** 添加 | PI 评审 Phase 1 工作，调整下一步优先级：实验设定调整 > 简化问题 |
| 2026-01-28 | **Fei Lu 论文综述** 添加 | 文献调研：identifiability 理论、convergence rate、RKHS 正则化 |

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
