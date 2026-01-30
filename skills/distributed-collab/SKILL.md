---
name: distributed-collab
description: 分布式 Agent 协作工作流。本地 Agent 负责通讯对接（短信、合作者沟通），服务器 Agent 负责计算实验，通过 Telegram 群作为中间层协调。适用于需要隔离隐私、分离通讯与计算职责的科研协作场景。
---

# Distributed Collaboration Workflow

本地 + 服务器双 Agent 协作模式，Telegram 群作为协调层。

## 架构概览

```
┌─────────────────┐    Telegram 群    ┌─────────────────┐
│   本地 Agent    │◄──────────────────►│  服务器 Agent   │
│  (mini/laptop)  │                    │    (server)     │
├─────────────────┤                    ├─────────────────┤
│ • 短信/iMessage │                    │ • 跑实验        │
│ • 合作者对接    │                    │ • 数据处理      │
│ • 日程协调      │                    │ • 代码执行      │
│ • 本地文件      │                    │ • GPU 计算      │
└─────────────────┘                    └─────────────────┘
```

## 职责划分

### 本地 Agent（通讯端）
- 处理短信、iMessage、邮件
- 与合作者沟通（非技术层面）
- 日程安排、会议协调
- 管理本地敏感文件（如有隐私限制）

### 服务器 Agent（计算端）
- 执行实验代码
- 处理大规模数据
- 管理代码仓库
- 生成实验报告和结果

### Telegram 群（协调层）
- 双向消息传递
- 任务分发和状态同步
- 结果汇报
- 人类监督和介入点

## 工作流程

### 1. 任务下发
```
用户 → 本地 Agent: "帮我跑一下 XX 实验"
本地 Agent → Telegram 群: "@服务器Bot 请执行 XX 实验"
服务器 Agent: 收到，开始执行
```

### 2. 状态同步
```
服务器 Agent → Telegram 群: "实验进度 50%，预计 2 小时完成"
本地 Agent: (可选) 转告用户
```

### 3. 结果汇报
```
服务器 Agent → Telegram 群: "实验完成，结果: ..."
本地 Agent → 用户: 转达结果 / 发送通知
```

### 4. 合作者对接
```
合作者 → 短信: "实验结果怎么样？"
本地 Agent → Telegram 群: "合作者询问实验结果"
服务器 Agent: 提供详细数据
本地 Agent → 短信回复: 简洁总结
```

## 隐私边界（可配置）

### 本地端隐私
- 短信内容：可配置是否转发到群
- 联系人信息：默认不暴露
- 本地文件：按需共享

### 服务器端隐私
- 实验数据：默认可共享
- 代码细节：默认可共享
- 凭证/密钥：永不共享

## 配置示例

### 本地 Agent (mini)
```yaml
# TOOLS.md 参考配置
telegram:
  group_id: "-100XXXXXXXXXX"
  role: "通讯端"
  can_mention: ["@服务器Bot"]

privacy:
  forward_sms_content: false  # 只转发摘要
  share_contacts: false
```

### 服务器 Agent (server)
```yaml
# TOOLS.md 参考配置
telegram:
  group_id: "-100XXXXXXXXXX"
  role: "计算端"
  
privacy:
  share_experiment_data: true
  share_code: true
```

## 消息协议（建议）

为了让 Agent 之间高效协作，建议使用结构化消息：

```
[任务] 类型: 实验执行
描述: 运行 train.py --epochs 100
优先级: 普通
期望完成: 今天内
```

```
[状态] 任务: 实验执行
进度: 75%
预计剩余: 30 分钟
```

```
[结果] 任务: 实验执行
状态: 完成
摘要: Accuracy 达到 92%
详情: 见 results/exp_001.md
```

## 扩展场景

- **多服务器**: 一个本地 Agent 对接多个计算节点
- **多合作者**: 不同合作者走不同通讯渠道
- **审批流程**: 敏感操作需人类确认后执行

---

*这套工作流的核心是职责分离和隐私隔离，Telegram 群提供了透明的协调层，人类可以随时介入监督。*
