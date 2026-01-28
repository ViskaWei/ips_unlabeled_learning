# 训练自动化系统 (Training Automation System)

> 让训练流程可控、可追溯、省心

## 🎯 核心理念

1. **前几分钟健康检查**：确认进程没挂、loss 没 NaN、显存没炸
2. **通过后让它自己跑**：不用不停看 log
3. **训练完成自动触发下一步**：eval / 画图 / 汇总
4. **只在关键节点给 Cursor**：summary.json + metrics.csv，而不是完整日志

## 📁 文件结构

```
training/
├── driver.py         # 🚀 主驱动器（一键启动）
├── health_check.py   # 🏥 健康检查模块
├── orchestrator.py   # 🎬 多步骤流水线编排器
├── train_hooks.py    # 🪝 训练脚本钩子
├── post_process.py   # 📊 后处理自动化
├── __init__.py       # 模块入口
└── README.md         # 本文档
```

## 🚀 快速开始

### 方式 1: 使用驱动器（推荐）

```bash
# 进入训练仓库
cd ~/VIT

# 使用配置文件启动
python /home/swei20/Physics_Informed_AI/_backend/scripts/training/driver.py \
    --config configs/exp/moe.yaml \
    --exp-id VIT-20251204-moe-01

# 或使用完整命令
python /home/swei20/Physics_Informed_AI/_backend/scripts/training/driver.py \
    --cmd "python train_nn.py --config configs/nn.yaml" \
    --exp-id VIT-20251204-nn-01

# 自定义健康检查时间（10分钟）
python .../driver.py --config config.yaml --exp-id xxx --health-time 600
```

### 方式 2: 在训练脚本中使用钩子

```python
from training.train_hooks import TrainingHooks

# 创建钩子
hooks = TrainingHooks("VIT-20251204-moe-01", signals_dir="./signals")

# 训练循环
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        loss = train_step(batch)
        
        # 在 warmup 后标记健康
        if epoch == 0 and step == 100:
            if loss < 10.0:
                hooks.mark_healthy(step=step, loss=loss)
            else:
                hooks.mark_failed(f"Loss too high: {loss}")
                return

# 训练结束
hooks.mark_done(metrics={"r2": 0.99, "mae": 0.05})
```

### 方式 3: PyTorch Lightning 集成

```python
from training.train_hooks import TrainingHooks, TrainingHooksCallback
import pytorch_lightning as pl

hooks = TrainingHooks("VIT-20251204-xxx")

trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[
        TrainingHooksCallback(
            hooks,
            warmup_steps=100,
            health_loss_threshold=10.0
        )
    ]
)

trainer.fit(model, dataloader)
```

## 🏥 健康检查详情

检查项目：

| 检查项 | 说明 | 默认阈值 |
|--------|------|---------|
| NaN 检测 | loss/grad 出现 nan | - |
| Loss 爆炸 | loss 突然变很大 | > 1e10 或增长 100 倍 |
| 显存溢出 | OOM 错误 | - |
| CUDA 错误 | GPU 相关错误 | - |
| 学习率 | lr 变为 0 | = 0 |

自定义配置：

```python
from training.health_check import HealthChecker, HealthCheckConfig

config = HealthCheckConfig(
    loss_explosion_threshold=1e8,
    loss_explosion_ratio=50.0,
    check_loss_stagnation=True,
    loss_stagnation_steps=200,
)

checker = HealthChecker("logs/train.log", config)
result = checker.check()
```

## 📡 信号文件约定

```
signals/
├── {exp_id}.healthy    # 健康检查通过
├── {exp_id}.done       # 训练完成
├── {exp_id}.failed     # 训练失败
└── {exp_id}.*.json     # JSON 格式副本
```

信号文件内容示例：

```
# healthy
status: healthy
step: 100
loss: 0.543210
timestamp: 2025-12-04T10:30:00

# done
status: done
return_code: 0
duration: 02:30:45
timestamp: 2025-12-04T13:00:45

# failed
status: failed
reason: Loss explosion: 1e12
timestamp: 2025-12-04T10:35:00
```

## 📊 后处理输出

训练完成后自动生成：

```
results/{exp_id}/
├── metrics.csv      # 训练指标时间序列
├── summary.json     # 实验配置 + 最终结果
└── report_draft.md  # exp.md 报告骨架
```

### metrics.csv

```csv
step,epoch,loss,val_loss,lr,r2,mae
100,0,1.234,1.456,0.001,0.5,0.1
200,0,0.987,1.234,0.001,0.6,0.08
...
```

### summary.json

```json
{
  "exp_id": "VIT-20251204-moe-01",
  "status": "completed",
  "timestamp": "2025-12-04T13:00:45",
  "final_metrics": {"loss": 0.123, "r2": 0.99},
  "best_r2": 0.992,
  "final_r2": 0.990,
  "best_loss": 0.101,
  "final_loss": 0.123
}
```

## 💡 减少 Cursor Token 的使用习惯

### ❌ 不要这样做

```
# 把整个日志贴给 Cursor
cat logs/train.log  # 10000 行...
```

### ✅ 应该这样做

```bash
# 1. 运行后处理
python post_process.py --exp-id xxx --generate-prompt

# 2. 只把摘要给 Cursor
cat results/xxx/summary.json

# 3. 或者让 Cursor 自己读文件
"实验结果在 results/xxx/summary.json，帮我分析并写结论"
```

### 给 Cursor 的 prompt 模板

```
实验 ID: VIT-20251204-moe-01
最终 R²: 0.992
最终 Loss: 0.123

请根据 results/VIT-20251204-moe-01/summary.json 帮我：
1. 总结核心结论（一句话）
2. 提炼关键洞见
3. 给出设计建议
4. 建议下一步实验

不要复述原始数据，只输出精简的分析。
```

## 🔄 完整工作流

```
1. 启动训练
   python driver.py --config config.yaml --exp-id VIT-xxx
   
2. 驱动器自动执行：
   ├─ 启动训练进程
   ├─ 前 5 分钟健康检查
   │   ├─ 通过 → 继续
   │   └─ 失败 → 终止 + 记录原因
   ├─ 等待训练完成
   └─ 自动后处理
       ├─ 提取 metrics.csv
       ├─ 生成 summary.json
       └─ 生成 report_draft.md

3. 给 Cursor 精简信息
   cat results/xxx/summary.json
   
4. 让 Cursor 帮你写报告
   "根据 summary.json 填充 report_draft.md"

5. 归档到知识中心
   a VIT-xxx
```

## 🛠️ Slurm 集群使用

如果在集群上运行，可以使用 job dependency：

```bash
# 提交训练任务
jid_train=$(sbatch train.slurm | awk '{print $4}')

# 提交后处理，依赖训练成功
sbatch --dependency=afterok:$jid_train post_process.slurm
```

train.slurm 示例：

```bash
#!/bin/bash
#SBATCH --job-name=VIT-xxx
#SBATCH --output=logs/%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

export EXP_ID="VIT-20251204-xxx"

# 训练脚本会使用 train_hooks
python train_nn.py --config config.yaml

# 训练结束后标记（如果脚本没有集成 hooks）
if [ $? -eq 0 ]; then
    echo "done" > signals/${EXP_ID}.done
else
    echo "failed" > signals/${EXP_ID}.failed
fi
```

## 📚 相关文档

- [实验归档系统](../../../README.md)
- [exp.md 模板](../../template/exp.md)
- [Coding Prompt 模板](../../template/coding_prompt.md)

---

*最后更新: 2025-12-04*

