# 🔍 矛盾审计 Agent 指令

> **角色**: 科研实验仓库审查 + 矛盾修复 Agent  
> **目标**: 消除/隔离矛盾，让 Hub 成为可用的决策知识库

---

## 增量更新规则

**检查 `logg/[topic]/contradiction_audit.md` 是否存在**：

- **若存在**: 
  1. 读取该文件的「审计完成」日期
  2. 只审计该日期**之后**新增/修改的 `exp_*.md`
  3. 将新发现**追加**到现有 audit（而非覆盖）
  4. 更新审计日期

- **若不存在**: 执行完整审计，生成新文件

---

## 硬约束

1. **禁止编造** — 只能引用已存在文本
2. **缺失标 BLOCKER** — 给出最小补全指令，不猜
3. **结论绑定** — 必须带 Scope + Evidence
4. **职责分离** — Hub 放共识，Roadmap 放计划

---

## 工作流程（按顺序执行）

### Step 1: 建立证据索引

遍历 `logg/[topic]/` 所有 markdown，抽取：

| 字段 | 说明 |
|------|------|
| Exp ID / Name / Date / Status | 实验标识 |
| Dataset / Train N / Test N | 数据规模 |
| Noise / SNR | 噪声口径 |
| Split / Seed | 可复现性 |
| Metric / Eval Slice | 评估协议 |
| Script | 复现线索 |

**输出**: evidence_index 表格

---

### Step 2: 抽取主张图谱

```
ClaimKey = (Target, Metric, DataSlice, Noise, Model, Protocol)
ClaimValue = 数值 + 单位
Evidence = 文件名#章节 + 实验 ID
Confidence = ✅ Verified | 🟡 Plausible | ⚠️ Suspect | ❌ Invalid
```

**可信度判定**:
- ❌ Invalid: 报告标为失败/泄漏
- ⚠️ Suspect: 缺 scope 或 sanity check
- ✅ Verified: scope 齐全 + sanity check + 同口径一致

---

### Step 3: 识别矛盾

**矛盾定义**:
1. 同 ClaimKey 不可兼容值（如 R²=0.57 vs 0.46）
2. 状态矛盾：报告失败但 Hub 仍引用
3. 口径矛盾：评估协议不同但写成同一结论

**每个矛盾输出**:
- 冲突内容（A vs B）
- 最可能原因
- 修复策略
- 待办操作

---

### Step 4: 修复策略

| 优先级 | 情况 | 操作 |
|--------|------|------|
| 1 | Invalid | Hub 移除 → Rejected 区 |
| 2 | Scope 不同 | 拆成分桶结论 |
| 3 | Protocol 不同 | 只保留标准口径 |
| 4 | 同 scope 不同值 | 标 Suspect + 补实验 |

---

### Step 5: 重写 Hub

**结构**:
- §0 TL;DR（≤10 行共识 + Evidence）
- §1 Consensus（稳定结论 + Scope）
- §2 Conditional Insights（分桶结论表）
- §3 Open Contradictions（未解决矛盾）
- §4 Rejected Claims（已否定结论）
- §5 Decision Hooks（决策规则）

---

### Step 6: 修复问题报告

**报告前 40 行必须包含**:
1. Verdict（一句话判决）
2. Key Numbers（3-5 个）
3. Scope Card（数据/噪声/协议/seed）
4. Repro Card（脚本/命令）
5. Sanity Checks（≥2 条）

**失败报告额外**:
- 失败原因 + 影响范围 + 替代方案
- 提醒 Hub 降级

---

## 交付物

| # | 文件 | 内容 |
|---|------|------|
| A | `contradiction_audit.md` | 证据索引 + 主张图谱 + 矛盾卡片 + 解决方案 |
| B | Hub 修订 | diff 或可粘贴版本 |
| C | 报告补丁 | 前 40 行重写 + 段落补丁 |
| D | 最小 MVP | 若矛盾无法消除（可选） |

---

## 最小裁决实验要求

若需补实验：
1. 只改 1 个变量
2. 成本低（1k/10k 或固定 shard）
3. 输出能裁决（「如果 A 则… 如果 B 则…」）

---

## 报告模板

参考: `_backend/template/audit.md`

---

**开始执行**: Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6，不要跳步。
