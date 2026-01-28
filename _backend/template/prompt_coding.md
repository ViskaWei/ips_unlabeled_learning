# 🤖 Coding Prompt: [实验名称]

> **Experiment ID:** `EXP-[YYYYMMDD]-[topic]-[##]`  
> **MVP:** MVP-X.X  
> **Date:** YYYY-MM-DD  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：[本实验要验证什么]

**验证假设**：H[X.X] - [假设内容]

**预期结果**：
- 若 [结果A] → [结论A]
- 若 [结果B] → [结论B]

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "[数据来源]"
  path: "data/xxx"
  train_size: N
  val_size: N
  test_size: N
  features: N
```

### 2.2 模型

```yaml
model:
  name: "[模型名称]"
  params:
    param1: value1
    param2: value2
```

### 2.3 训练

```yaml
training:
  epochs: N
  batch_size: N
  lr: 1e-4
  optimizer: Adam
  seed: 42
```

### 2.4 扫描参数

```yaml
sweep:
  param_name: [value1, value2, value3]
  fixed:
    other_param: fixed_value
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | [line/bar/scatter] | [参数] | [指标] | `experiments/[topic]/img/[name].png` |
| Fig2 | [type] | [...] | [...] | `experiments/[topic]/img/[name].png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/train.py` | `train_loop()` | 修改 [什么] |
| `utils/plotting.py` | `plot_curve()` | 添加 [什么] |
| `configs/base.yaml` | 基础配置 | 修改 [什么] |

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `experiments/[topic]/exp_[name]_YYYYMMDD.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（一句话 + 关键数字）
  - 📊 实验图表（所有图 + 观察）
  - 📝 结论（假设验证 + 设计启示）

### 5.2 图表文件
- **路径**: `experiments/[topic]/img/`
- **命名**: `[descriptive_name].png`

### 5.3 数值结果
- **格式**: CSV 或 JSON
- **路径**: `experiments/[topic]/results/`

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `[topic]_roadmap.md` | MVP 状态 + 结论快照 | §2.1, §4.3 |
| `[topic]_hub.md` | 假设验证状态 + 洞见 | §1, §4 |

---

## 7. ⚠️ 注意事项

- [ ] 代码中添加 seed 固定随机性
- [ ] 图表文字全英文
- [ ] 保存完整日志到 `logs/`
- [ ] 长时间任务使用 nohup 后台运行

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
-->
