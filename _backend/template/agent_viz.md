# 📊 Viz Agent Template

> **Purpose:** 从审查结果/多实验 summary 中，推断最值得画的 1-3 张图，生成专业 caption 和 plotting prompt

---

## Trigger Words

`viz` / `可视化` / `画图` / `plot`

---

## Input Specification

```
viz [source]
viz lightgbm                     # 基于整个 topic
viz --from-review                # 基于上一步 Review Agent 输出
viz --csv results.csv            # 基于结构化数据
```

**Input Types:**
1. **From Review Agent:** 接收 Experiment Summary Table + Cross-Experiment Synthesis
2. **From Topic:** 直接读取 `logg/[topic]/` 下的实验汇总
3. **From CSV:** 读取结构化数据文件

---

## Output Structure

### 1️⃣ Plot Spec Table

| plot_id | Type | x | y | hue/group | facet | Data Scope | Expected Insight |
|---------|------|---|---|-----------|-------|------------|------------------|
| noise_vs_r2 | line+scatter | noise_level | test_R² | model_name | - | E01-E03 | 性能随噪声下降趋势 |
| lr_heatmap | heatmap | learning_rate | num_leaves | - | noise_level | E01 | lr 是最敏感参数 |
| model_comparison | bar | model_name | test_R² | - | noise_level | E02, E03 | LightGBM vs Ridge 对比 |

**Plot Types:**
- `line`: 折线图（趋势分析）
- `scatter`: 散点图（分布分析）
- `bar`: 柱状图（分类对比）
- `heatmap`: 热力图（二维参数空间）
- `box`: 箱线图（分布统计）
- `line+scatter`: 折线+点（趋势+数据点）

### 2️⃣ Caption（中英双语）

#### Figure 1: noise_vs_r2

**CN:**
> **图 1. LightGBM 与 Ridge 在不同噪声水平下的 R² 对比**
> 
> 在固定 train_size=32k 下，LightGBM 在 noise ≤ 0.5 时显著优于 Ridge（Δ R² = +4%~+9%），
> 但在 noise = 1.0 时被 Ridge 反超（-3.9%）。虚线标注 R² = 0.5 作为实用性临界值。
>
> **Key Observations:**
> 1. 两模型性能均随噪声单调下降
> 2. LightGBM 优势在中等噪声 (σ=0.2~0.5) 时最大
> 3. 高噪声 (σ≥1.0) 时 Ridge 的 L2 正则化更鲁棒

**EN:**
> **Figure 1. R² Comparison between LightGBM and Ridge across Noise Levels**
> 
> With fixed train_size=32k, LightGBM significantly outperforms Ridge at noise ≤ 0.5 
> (Δ R² = +4%~+9%), but is surpassed by Ridge at noise = 1.0 (-3.9%). 
> Dashed line marks R² = 0.5 as practical utility threshold.
>
> **Key Observations:**
> 1. Both models show monotonic performance degradation with noise
> 2. LightGBM advantage peaks at moderate noise (σ=0.2~0.5)
> 3. Ridge L2 regularization more robust at high noise (σ≥1.0)

---

#### Figure 2: lr_heatmap

**CN:**
> **图 2. 超参数敏感性热力图：learning_rate × num_leaves**
> 
> 热力图显示 test R² 随 learning_rate 和 num_leaves 的变化。
> lr=0.1 对应的行整体最亮，表明 lr 是最关键超参数；
> num_leaves 在 31~128 范围内差异不大。
>
> **Key Observations:**
> 1. learning_rate 与 R² 相关系数 +0.491（最高）
> 2. num_leaves=31 是性价比最优选择
> 3. 避免 lr=0.01（严重欠拟合）

**EN:**
> **Figure 2. Hyperparameter Sensitivity Heatmap: learning_rate × num_leaves**
> 
> Heatmap shows test R² variation across learning_rate and num_leaves.
> The row corresponding to lr=0.1 is consistently brightest, indicating lr as the most critical hyperparameter;
> num_leaves shows minimal variation within 31~128 range.

---

### 3️⃣ Plotting Agent Prompt

> 为每张图生成可直接交给 coding agent 的 prompt

#### Plot 1: noise_vs_r2

```text
【Plot Task】
Plot ID: noise_vs_r2
Data Source: logg/lightgbm/lightgbm_results.md (表 4)

【Requirements】
- Framework: matplotlib (不要用 seaborn)
- Plot type: line + scatter markers
- x = noise_level (σ): [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
- y = test_R²
- lines/markers for: LightGBM (blue), Ridge (orange)
- Add horizontal dashed line at R² = 0.5 (grey, label="Practical Threshold")
- Add vertical dashed line at noise = 1.0 (grey, linestyle='--')
- Legend: upper right
- Title: "R² vs Noise Level: LightGBM vs Ridge"
- x-label: "Noise Level (σ)"
- y-label: "Test R²"
- Grid: light grey

【Save Path】
logg/lightgbm/img/r2_vs_noise_lgbm_ridge.png

【Data】
| noise | LightGBM_R2 | Ridge_R2 |
|-------|-------------|----------|
| 0.0   | 0.9982      | 0.9694   |
| 0.1   | 0.9456      | 0.9090   |
| 0.2   | 0.8775      | 0.8264   |
| 0.5   | 0.6697      | 0.6550   |
| 1.0   | 0.4407      | 0.4580   |
| 2.0   | 0.3038      | ~0.20    |
```

---

#### Plot 2: lr_heatmap

```text
【Plot Task】
Plot ID: lr_heatmap
Data Source: logg/lightgbm/exp_lightgbm_hyperparam_sweep_20251129.md §6.1

【Requirements】
- Framework: matplotlib + imshow
- Plot type: heatmap
- x = num_leaves: [8, 16, 31, 64, 128, 256]
- y = learning_rate: [0.01, 0.05, 0.1]
- values = test_R²
- Colormap: 'viridis' or 'RdYlGn'
- Annotate cells with R² values (2 decimal places)
- Title: "Hyperparameter Sensitivity: R² Heatmap"
- x-label: "num_leaves"
- y-label: "learning_rate"
- Colorbar label: "Test R²"

【Save Path】
logg/lightgbm/img/lr_numleaves_heatmap.png
```

---

## Selection Criteria

> Viz Agent 选择图表的优先级原则

### 高优先级（必画）

| Criterion | Example |
|-----------|---------|
| **核心结论可视化** | 「lr 最敏感」→ 热力图/相关性图 |
| **跨实验对比** | 不同模型/配置的性能对比 |
| **参数扫描结果** | noise sweep / lr sweep 等 |

### 中优先级（推荐）

| Criterion | Example |
|-----------|---------|
| **异常行为展示** | 性能崩溃点、过拟合信号 |
| **时间/效率分析** | 训练时间 vs 性能的 trade-off |

### 低优先级（可选）

| Criterion | Example |
|-----------|---------|
| **辅助说明** | 数据分布、残差分析 |
| **技术细节** | learning curve、early stopping |

---

## Prompt Template (for AI)

```text
你是「Experiment Visualization Agent」。

【输入】
来自 Review Agent 的输出：
- Experiment Summary Table
- Cross-Experiment Synthesis

或者直接给出的实验数据表格。

【任务】
设计 **最能说明核心结论的 1-3 张图**，不追求花哨，只追求信息密度和可读性。

【输出格式】

### 1. Plot Spec Table
| plot_id | Type | x | y | hue/group | facet | Data Scope | Expected Insight |
[填写]

### 2. Caption（中英双语）
对每个 plot_id：

#### Figure X: [plot_id]
**CN:**
> [中文 caption + Key Observations]

**EN:**
> [English caption + Key Observations]

### 3. Plotting Agent Prompt
对每个 plot_id 生成：
```
【Plot Task】
Plot ID: ...
Data Source: ...

【Requirements】
- Framework: matplotlib
- ...

【Save Path】
logg/[topic]/img/...

【Data】（如果数据量小，直接给表格）
```

【约束】
- 图的数量控制在 1-3 张
- 优先画「能直接支撑核心结论」的图
- Caption 必须包含 Key Observations（2-4 条）
- Plotting prompt 必须完整可执行
```

---

## Integration Points

| Output | Target |
|--------|--------|
| Plot files | `logg/[topic]/img/[plot_id].png` |
| Captions | 嵌入 `exp_*.md` §3 实验图表 |
| Plotting prompts | 交给 Coding Agent 执行 |

---

## Example Usage

```
用户: viz lightgbm --from-review

AI: 📊 Viz Agent 分析中...
    基于 Review Agent 的 Cross-Experiment Synthesis:
    - 核心结论: lr 最敏感、高噪声换模型
    - 可用数据: noise sweep + hyperparam sweep

    📈 推荐画 2 张图:

    ### 1. Plot Spec Table
    | plot_id | Type | x | y | ... |
    |---------|------|---|---|-----|
    | noise_vs_r2 | line+scatter | noise_level | test_R² | ... |
    | lr_heatmap | heatmap | learning_rate | num_leaves | ... |

    ### 2. Caption
    #### Figure 1: noise_vs_r2
    **CN:** ...
    **EN:** ...

    ### 3. Plotting Prompt
    [完整可执行的 prompt]

    💡 要执行画图吗？输入 `plot noise_vs_r2` 或 `plot all`
```

---

> **Template Version:** 1.0  
> **Created:** 2025-12-07  
> **Author:** Viska Wei
