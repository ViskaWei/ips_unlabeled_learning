# 📊 <TOPIC> Consolidated Summary
> **Name:** TODO | **ID:** `[PROJECT]-[YYYYMMDD]-[topic]-consolidated`  
> **Topic:** `<topic>` | **Merged:** N experiments | **Project:** `VIT`/`SD`  
> **Author:** Viska Wei | **Date:** TODO | **Status:** 🔄/✅
```
💡 一句话总结合并目的  
Focus：<如 "Best config across noise levels">
```

---

# 📑 Table of Contents

- [1. 🎯 Purpose of This Consolidation](#1--purpose-of-this-consolidation)
- [2. 📁 Included Experiments](#2--included-experiments)
- [3. 📊 Core Findings (Key Insights)](#3--core-findings-key-insights)
- [4. 🔍 Detailed Comparison](#4--detailed-comparison)
- [5. 🧪 Failed / Unhelpful Experiments](#5--failed--unhelpful-experiments)
- [6. 🧭 Recommended Best Setting (Current SOTA)](#6--recommended-best-setting-current-sota)
- [7. 📈 Visual Summary](#7--visual-summary)
- [8. 📎 Appendix](#8--appendix)

---

# 1. 🎯 Purpose of This Consolidation

简要说明为什么要把多个子实验合并：

**合并理由**：
- [ ] 多次子实验的结论碎片化，希望统一得到可复现的 summary
- [ ] 需要跨 noise level / 跨参数段 / 跨模型的综合对比
- [ ] 生成可供分享的统一报告

**核心问题**：
> [e.g., "不同 noise level 下的最佳 LightGBM 配置是什么？"]

**预期产出**：
1. 各条件下的 best config 汇总表
2. 关键趋势/规律的提炼
3. 推荐的 SOTA 配置

---

# 2. 📁 Included Experiments

> 自动生成：列出所有被合并的子实验文件

| # | File | Date | Focus | Key Result |
|---|------|------|-------|------------|
| 1 | `exp_xxx_20251201.md` | 2025-12-01 | [描述] | R²=0.XX |
| 2 | `exp_yyy_20251202.md` | 2025-12-02 | [描述] | R²=0.XX |
| 3 | `exp_zzz_20251203.md` | 2025-12-03 | [描述] | R²=0.XX |

### 实验关系图（可选）

```
exp_baseline (基线)
    ├── exp_noise_sweep (噪声扫描)
    │   └── exp_100k_noise (100k 数据量)
    └── exp_tree_limit (树数上限)
```

---

# 3. 📊 Core Findings (Key Insights)

> 用 bulletpoint 形式总结所有 sweep 的核心洞见

### 3.1 ⭐ 一句话总结

> **[e.g., "noise ≤ 0.5 时 lr=0.05 最优，noise > 0.5 时 lr=0.1 更稳健；100k 数据需要 n_estimators=2500+"]**

### 3.2 关键发现

- **最佳参数段**：[e.g., `lr=0.05` consistently dominates at low noise]
- **关键超参影响**：[e.g., learning rate 对性能影响最大]
- **数据量效应**：[e.g., 100k vs 32k 的增益随噪声增大]
- **收益递减点**：[e.g., 树数从 500 → 5000 的增益只有 +0.01 R²]

### 3.3 趋势总结表

| 维度 | 低噪声 (σ≤0.2) | 中噪声 (σ=0.5) | 高噪声 (σ≥1.0) |
|------|---------------|---------------|----------------|
| Best lr | 0.05 | 0.05-0.1 | 0.1 |
| Best n_estimators | 500-1000 | 1000-2500 | 500 |
| Best num_leaves | 31 | 31-63 | 31 |
| R² 范围 | 0.91-0.97 | 0.73-0.76 | 0.45-0.56 |

---

# 4. 🔍 Detailed Comparison

## 4.1 跨 Noise Level 最佳配置

| Noise σ | Best R² | Best lr | n_estimators | num_leaves | Source Exp |
|---------|---------|---------|--------------|------------|------------|
| 0.0 | 0.999 | 0.05 | 5000 | 31 | exp_tree_limit |
| 0.1 | 0.972 | 0.05 | 2218 | 31 | exp_tree_limit |
| 0.2 | 0.932 | 0.05 | 3608 | 31 | exp_tree_limit |
| 0.5 | 0.757 | 0.05 | 3855 | 31 | exp_tree_limit |
| 1.0 | 0.558 | 0.05 | 2140 | 31 | exp_tree_limit |

## 4.2 Sweep: [维度 1, e.g., num_trees]

| num_trees | Best R² | Notes |
|-----------|---------|-------|
| 500 | 0.XXX | fast, stable |
| 5000 | 0.XXX | slight gain, diminishing returns |

→ **Insight**: [e.g., 超过 1000 棵树后增益极小]

## 4.3 Sweep: [维度 2, e.g., learning_rate]

解释趋势 + 引用具体结果：

- lr=0.02 → underfit (R² 低 5-10%)
- lr=0.05 → optimal at low noise
- lr=0.1 → optimal at high noise
- lr=0.3 → unstable, early stopping 过早触发

## 4.4 Sweep: [维度 3, e.g., data_size]

| Data Size | Noise 0.1 | Noise 0.5 | Noise 1.0 | 增益趋势 |
|-----------|-----------|-----------|-----------|---------|
| 32k | 0.946 | 0.674 | 0.451 | baseline |
| 100k | 0.972 | 0.757 | 0.558 | +2.7%~+10% |

→ **Insight**: [e.g., 更多数据在高噪声下价值更高]

---

# 5. 🧪 Failed / Unhelpful Experiments

> 记录不好的结果（哪里失败了 + 为什么）

| 配置 | 结果 | 原因分析 |
|------|------|---------|
| lr=0.3 + n=5000 | R² 最差 | early stopping 过早，只用了 100-300 棵树 |
| num_leaves=127 | 严重 overfit | 模型过于复杂，训练集 R²=0.99 但测试集下降 |
| 100k + n=500 | 被 32k 超越 | 树数限制了 100k 的学习能力 |

### 教训

1. **大模型需要更保守的学习率**：100k 数据下 lr=0.3 完全失效
2. **模型容量要匹配数据量**：100k 需要 n≥2500，否则不如 32k
3. **高噪声下 ensemble 需要控制**：noise=1.0 时 n=500 反而优于 n=1000

---

# 6. 🧭 Recommended Best Setting (Current SOTA)

## 6.1 推荐配置

```python
# SOTA Config for LightGBM log_g Prediction
best_config = {
    # === 核心参数 ===
    'learning_rate': 0.05,      # 低噪声最优；高噪声可调到 0.1
    'n_estimators': 2500,       # 100k 数据的推荐值；32k 可用 1000
    'num_leaves': 31,           # 稳健选择
    'max_depth': 7,             # 或 -1（无限制）
    
    # === 正则化 ===
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    
    # === 训练配置 ===
    'early_stopping_rounds': 100,
    'device_type': 'gpu',       # 如有 GPU
    'random_state': 42,
}
```

## 6.2 按场景推荐

| 场景 | 推荐配置 | 预期 R² |
|------|---------|---------|
| 低噪声 (σ≤0.2) + 100k | lr=0.05, n=2500, leaves=31 | 0.93-0.97 |
| 中噪声 (σ=0.5) + 100k | lr=0.05, n=2500, leaves=31 | 0.75-0.76 |
| 高噪声 (σ≥1.0) + 100k | lr=0.1, n=1500, leaves=31 | 0.55-0.56 |
| 快速实验 (32k) | lr=0.1, n=1000, leaves=31 | 降低 2-5% |

## 6.3 配置选择决策树

```
Start
├── 数据量?
│   ├── 100k+ → n_estimators = 2500
│   └── 32k → n_estimators = 1000
├── 噪声水平?
│   ├── σ ≤ 0.2 → lr = 0.05
│   ├── 0.2 < σ < 1.0 → lr = 0.05 or 0.1
│   └── σ ≥ 1.0 → lr = 0.1
└── 训练时间约束?
    ├── 快速 → n_estimators ÷ 2, early_stopping = 50
    └── 精度优先 → 使用推荐配置
```

---

# 7. 📈 Visual Summary

## 7.1 关键图表引用

| 图表 | 来源 | 要点 |
|------|------|------|
| R² vs Noise | exp_noise_sweep | R² 随 noise 近线性下降 |
| best_iter vs Noise | exp_tree_limit | 100k 需要 2000+ 棵树 |
| Δ R² (100k-32k) | exp_100k_noise | 增益随噪声增大 |

## 7.2 综合对比图（如有）

![Summary Figure](./img/consolidated_summary.png)

---

# 8. 📎 Appendix

## 8.1 完整数值汇总表

> 合并所有实验的关键数值

| Experiment | Noise | lr | n_est | leaves | R² | MAE | best_iter |
|------------|-------|-----|-------|--------|-----|-----|-----------|
| exp_1 | 0.1 | 0.05 | 500 | 31 | 0.964 | 0.039 | 500 |
| exp_1 | 0.5 | 0.10 | 500 | 31 | 0.737 | 0.111 | 500 |
| exp_2 | 0.1 | 0.05 | 5000 | 31 | 0.972 | 0.034 | 2218 |
| ... | ... | ... | ... | ... | ... | ... | ... |

## 8.2 实验时间线

| Date | Experiment | 主要发现 |
|------|------------|---------|
| 2025-12-04 | exp_noise_sweep_lr | lr=0.1 在 n≤100 下最优 |
| 2025-12-05 | exp_100k_noise | 100k + n=500 在各噪声超越 32k |
| 2025-12-07 | exp_tree_limit | 100k 的 tree 上限约 2179 |

## 8.3 开放问题 & 下一步

| 问题 | 优先级 | 建议实验 |
|------|--------|---------|
| num_leaves=63 在 100k 下效果？ | 🟡 P1 | 单独 sweep |
| 混合 noise 训练的鲁棒性？ | 🟢 P2 | multi-noise training |
| LightGBM vs NN 100k 对比？ | 🔴 P0 | exp_nn_vs_lgb |

---

## 🔗 Related Files

| Type | Path | Description |
|------|------|-------------|
| 🧠 Hub | `lightgbm_hub_YYYYMMDD.md` | 智库导航 |
| 🗺️ Roadmap | `lightgbm_roadmap_YYYYMMDD.md` | 实验追踪 |
| 📊 Source Experiments | `exp_*.md` | 合并的源实验 |

---

> **Template Usage**:
> 
> 1. **触发词**: `merge [描述]`
> 2. **自动填充**: §2 Included Experiments 根据匹配自动生成
> 3. **手动整理**: §3-6 根据源实验提取关键信息
> 4. **输出位置**: `logg/[topic]/exp_[topic]_consolidated_YYYYMMDD.md`
