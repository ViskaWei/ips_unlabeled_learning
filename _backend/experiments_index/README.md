# 🗂️ Experiment Index

> 跨仓库实验追踪中心

## 文件说明

| 文件 | 用途 |
|------|------|
| `index.csv` | 主索引文件，记录所有实验的元数据 |
| `index.json` | JSON 格式索引（脚本友好） |

## 字段说明

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `experiment_id` | string | 唯一标识符 | `VIT-20251201-corr-attn-01` |
| `project` | enum | 来源仓库 | `VIT` / `BlindSpot` / `Other` |
| `topic` | string | 主题分类 | `cnn`, `swin`, `noise`, `distill` |
| `status` | enum | 实验状态 | `running` / `completed` / `failed` / `aborted` |
| `start_time` | ISO datetime | 开始时间 | `2025-12-01T10:30:00` |
| `end_time` | ISO datetime | 结束时间 | `2025-12-01T12:45:00` |
| `entry_point` | string | 启动脚本 | `scripts/run.py` |
| `config_path` | string | 配置文件路径 | `configs/corr_attention.yaml` |
| `output_path` | string | 结果目录 | `lightning_logs/version_42` |
| `log_path` | string | 日志文件 | `training_full.log` |
| `metrics_summary` | string | 关键指标摘要 | `R2=0.987, RMSE=0.031` |
| `physics_ai_logg_path` | string | 对应 logg 文档 | `logg/cnn/exp_xxx.md` |
| `priority` | enum | 优先级 | `P0` / `P1` / `P2` |
| `next_action` | string | 下一步动作 | `写 exp.md 总结` |
| `notes` | string | 备注 | 任意文本 |

## Experiment ID 命名规范

```
[PROJECT]-[YYYYMMDD]-[topic]-[序号]
```

**示例**:
- `VIT-20251201-cnn-dilated-01`
- `VIT-20251201-swin-attention-01`
- `BS-20251201-latent-probe-01`
- `BS-20251201-encoder-freeze-01`

**Project 前缀**:
- `VIT` - VIT 仓库实验
- `BS` - BlindSpotDenoiser 仓库实验

## 使用方式

### 方式 1: 手动登记

直接编辑 `index.csv`，添加新行。

### 方式 2: 脚本登记

```bash
# 在实验完成后调用
python ~/Physics_Informed_AI/scripts/register_experiment.py \
  --experiment_id "VIT-20251201-cnn-dilated-01" \
  --project VIT \
  --topic cnn \
  --status completed \
  --entry_point "scripts/run.py" \
  --config_path "configs/cnn_dilated.yaml" \
  --output_path "lightning_logs/version_42" \
  --metrics_summary "R2=0.987, MAE=0.031"
```

### 方式 3: 自动扫描

```bash
# 扫描 VIT 仓库，补录历史实验
python scripts/scan_vit_experiments.py --vit-root ~/VIT

# 扫描 BlindSpot 仓库
python scripts/scan_blindspot_experiments.py --blindspot-root ~/BlindSpotDenoiser
```

## 与 logg 的关系

```
experiments_index/index.csv  ←→  logg/[topic]/exp_*.md
          ↑                              ↑
     实验元数据                       知识沉淀
     (When/Where/What)            (Why/Insight/Design)
```

- **index.csv**: 记录实验的「何时、何地、什么配置」
- **logg/**: 记录实验的「为什么、洞见、设计启示」
- 通过 `physics_ai_logg_path` 字段相互链接

