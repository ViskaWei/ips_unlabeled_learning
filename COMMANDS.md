# 📖 实验管理系统 - 命令速查手册

> **最后更新**: 2025-01-08  
> **适用仓库**: GiftLive

---

## 🗂️ 目录

- [快捷命令总览](#-快捷命令总览)
- [进度查看](#-进度查看)
- [实验管理](#-实验管理)
- [文档更新](#-文档更新)
- [知识管理](#-知识管理)
- [Shell 命令](#-shell-命令)
- [完整工作流](#-完整工作流)

---

## 🎯 快捷命令总览

| 命令 | 作用 | 输出位置 |
|------|------|----------|
| `?` | 查看进度状态 | 终端输出 |
| `n` | 新建实验计划 | `experiments/[topic]/exp_*.md` |
| `a` | 归档实验结果 | `experiments/[topic]/exp_*.md` |
| `u [exp_id]` | 完整更新实验 | `experiments/[topic]/` |
| `p` | 生成 Coding Prompt | `experiments/[topic]/prompts/` |
| `merge` | 合并相似实验 | `experiments/[topic]/exp_*_consolidated_*.md` |
| `card` | 创建知识卡片 | `experiments/[topic]/card/` |
| `design` | 提取设计原则 | `design/principles.md` |
| `next` | 管理待办 | `status/next_steps.md` |
| `report` | 生成报告 | `reports/drafts/` |
| `session` | GPT 会话归档 | `experiments/[topic]/sessions/` |

---

## ❓ 进度查看

```
触发词: ? | ？ | status | 进度 | 状态
```

**输出内容**:
- 📋 P0/P1 待办任务
- 📦 归档队列状态
- 📝 最近更新的文档
- 📦 自动 Git Commit + Push（可选）

**示例**:
```
用户: ?

AI: 📊 项目状态:
    
    📋 待办任务 (2个):
    🔴 P0: 完成 baseline 实验
    🟡 P1: 整理实验数据
    
    📦 归档队列 (1个):
    1. raw/experiment_report.md → experiments/baseline/
    
    📝 最近更新:
    - experiments/baseline/exp_v1_20250108.md (2h ago)
```

---

## 🔬 实验管理

### 🆕 新建实验
```
触发词: n | N | new | 新建
```

**作用**: 创建 `experiments/[topic]/exp_*.md` 实验框架

**示例**:
```
用户: n 测试新模型架构对性能的影响

AI: 📝 创建实验计划...
    ✅ 已创建: experiments/model/exp_arch_test_20250108.md
    ✅ 已更新: experiments/model/model_roadmap.md §2.1
```

---

### 📦 归档实验
```
触发词: a | A | 归档 | archive
变体: a 1 | a all | s (查看队列)
```

**作用**: 将 `raw/` 中的原始结果归档为结构化 `exp.md`

**示例**:
```
用户: a

AI: 📋 当前归档队列 (2个):
    1. raw/baseline_report.md → experiments/baseline/
    2. raw/feature_analysis.md → experiments/feature/
    要归档哪个？(输入序号或 "all")

用户: 1

AI: ✅ 已创建: experiments/baseline/exp_baseline_20250108.md
```

---

### 🤖 生成 Coding Prompt
```
触发词: p | P | prompt | 生成prompt
```

**作用**: 将 MVP 转化为可执行的 Coding Prompt

**示例**:
```
用户: p 生成模型对比实验的 prompt

AI: 📝 生成 Coding Prompt...
    ✅ 已创建: experiments/model/prompts/coding_prompt_compare_20250108.md
```

**⚠️ 强制规则**:
- ❌ **禁止在 Prompt 中写任何代码**
- ✅ **只写参考代码路径**

---

## 📝 文档更新

### 更新实验
```
触发词: u | U | update | 更新
变体: 
  - u [experiment_id] - 完整更新：补全exp+同步hub/roadmap+git push
  - u [关键词] - 智能追加内容+git push
```

**示例 1 - 完整更新**:
```
用户: u EXP-20250108-model-01

AI: 📝 完整更新实验报告...
    
    📖 Step 2: 审查报告完整性
    | 章节 | 状态 |
    |------|------|
    | ⚡ 核心结论速览 | ✅ |
    | §6.2 实验流程 | ❌ 缺少代码引用 |
    
    🔧 Step 3: 补全遗漏内容
    ✅ 已补充到 §6.2 实验流程记录
    
    📤 Step 4: 同步到 hub.md
    ✅ §2 假设: H1.1 状态 → ✅
    
    📤 Step 5: 同步到 roadmap.md
    ✅ §2.1 实验总览: MVP-1.1 状态 → ✅
    
    📦 Step 6: Git Commit + Push
    ✅ 完成
```

**示例 2 - 智能追加**:
```
用户: u 发现新架构在低数据量下表现更好

AI: 📝 更新文档...
    🔍 匹配到: experiments/model/exp_arch_test_20250108.md
    ✅ 已添加到 §4.1 宏观层洞见
    
    📦 Git Push...
    ✅ 完成
```

---

## 🧠 知识管理

### 📐 设计原则
```
触发词: design | 设计原则 | 原则
```

**作用**: 从各 hub.md 文件中提取设计原则

**示例**:
```
用户: design

AI: 🔍 扫描hub文件中的设计原则...
    
    📁 找到 3 个hub文件
      ✅ model_hub.md: 发现 5 个原则
      ✅ data_hub.md: 发现 3 个原则
    
    📝 追加到 design/principles.md...
    ✅ 已更新
```

---

### 📇 知识卡片
```
触发词: card | 卡片 | kc
```

**定义**: Card 是**可复用的阶段性知识**
- ✅ **做**: 跨多个实验的结构性认知、理论依据、可指导决策的结论
- ❌ **不做**: 指导下一步实验（这是 hub 的职责）

**示例**:
```
用户: card 模型选择最佳实践

AI: 📇 创建知识卡片...
    
    🔍 检索相关实验:
    - experiments/model/exp_arch_test_20250108.md
    - experiments/model/exp_compare_20250107.md
    
    📂 确定位置: experiments/model/card/
    
    ✅ 已创建: experiments/model/card/card_model_selection_20250108.md
```

---

### 📌 下一步计划
```
触发词: next | 下一步 | 计划
变体: next add | next done | next plan
```

**基本操作**:
```
用户: next                    # 查看当前计划
用户: next add P0 完成文档     # 添加 P0 任务
用户: next add P1 测试新模型   # 添加 P1 任务
用户: next done 1             # 完成第 1 个任务
```

**智能计划**:
```
用户: next plan

AI: 🧠 分析最近实验...
    
    💡 建议的下一步:
    🔴 P0: 新架构效果好 → 进行更多测试
    🟡 P1: 写 knowledge card 总结发现
    
    是否添加？(y/n)
```

---

### 💬 GPT 会话归档
```
触发词: session | 会话 | gpt
变体: session new [topic] | session list
```

**作用**: 把 GPT 脑暴对话 → 结构化成可执行实验

**示例**:
```
用户: session new model

AI: 📝 创建新会话...
    ✅ 已创建: experiments/model/sessions/session_20250108.md
```

---

### 🔀 合并实验
```
触发词: merge | 合并 | 整合
```

**用途**: 把多个相似目的的子实验合并成一份综合报告

**示例**:
```
用户: merge model 相关实验

AI: 🔄 合并实验...
    📁 找到 3 个相关实验
    ✅ 已创建: experiments/model/exp_model_consolidated_20250108.md
```

---

## 💻 Shell 命令

### 便捷 Alias (添加到 ~/.bashrc)

```bash
# GiftLive
alias gl='cd ~/GiftLive'

# 实验管理
alias exp-new='echo "使用 n 命令创建新实验"'
alias exp-archive='echo "使用 a 命令归档实验"'
```

---

## 🔄 完整工作流

### 日常工作流

```
┌─────────────────────────────────────────────────────────────┐
│  1. 规划实验                                                 │
│     └─ n [实验描述]                                          │
│                         ↓                                    │
│  2. 执行实验                                                 │
│     └─ python scripts/xxx.py                                │
│                         ↓                                    │
│  3. 查看进度                                                 │
│     └─ ?                                                     │
│                         ↓                                    │
│  4. 归档详细结果                                             │
│     └─ a                                                     │
│                         ↓                                    │
│  5. 更新文档（同步 hub/roadmap）                             │
│     └─ u [exp_id]                                           │
│                         ↓                                    │
│  6. 提炼知识卡片（可选）                                     │
│     └─ card [关键词]                                        │
│                         ↓                                    │
│  7. 生成报告                                                 │
│     └─ report                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 文件位置速查

| 文件 | 路径 |
|------|------|
| 待办清单 | `status/next_steps.md` |
| 归档队列 | `status/archive_queue.md` |
| 报告草稿 | `reports/drafts/` |
| 实验文档 | `experiments/[topic]/` |
| exp 模板 | `_backend/template/exp.md` |
| hub 模板 | `_backend/template/hub.md` |
| 知识卡片模板 | `_backend/template/card.md` |
| 设计原则汇总 | `design/principles.md` |

---

## 📝 Experiment ID 格式

| 格式 | 示例 |
|------|------|
| `EXP-YYYYMMDD-topic-XX` | `EXP-20250108-model-01` |

---

> 💡 **提示**: 将此文件加入书签，随时查阅命令用法
