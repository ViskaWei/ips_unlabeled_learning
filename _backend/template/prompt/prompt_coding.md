# 🤖 实验 Coding Prompt

> **日期:** YYYY-MM-DD | **来源:** `logg/[topic]/sessions/session_*.md`

---

## ⚠️ 核心规则

| 规则 | 说明 |
|------|------|
| **nohup 后台运行** | 所有训练必须 `nohup ... &`，>5分钟不持续追踪 |
| **跨仓库用终端** | 写入 Physics_Informed_AI 用 `cat/echo/cp`，禁止 IDE 工具 |
| **图片必须入报告** | 所有图表必须在报告 §3 中引用，路径 `logg/[topic]/img/` |
| **figsize 统一** | 所有图表 `figsize=(6, 5)`，保持一致性 |
| **语言** | Header 英文 \| 正文中文 \| 图表文字英文 |

---

## 🚀 仓库路由

| Topic | 仓库 | 前缀 |
|-------|------|------|
| diffusion | `~/SpecDiffusion` | SD- |
| cnn/swin/ridge/pca/gta/moe | `~/VIT` | VIT- |
| distill/latent/probe | `~/BlindSpotDenoiser` | BS- |

---

## 📋 执行流程

### Step 1: 启动训练

```bash
cd [repo] && source init.sh
nohup python script.py --exp-id [exp_id] > logs/[exp_id].log 2>&1 &
echo $! > logs/[exp_id].pid
```

**确认正常后输出**：
```
✅ 任务已启动 (PID: xxx)
📋 tail -f [repo]/logs/[exp_id].log
⏱️ 预计 ~Xmin，完成后告诉我继续
```

### Step 2: 生成图表
```bash
python plot.py --exp_id [exp_id] --output logg/[topic]/img/
```

### Step 3: 写报告

📄 **模板**: [`_backend/template/exp.md`](./_backend/template/exp.md)

```bash
# 用终端命令写入
cat << 'EOF' > "/home/swei20/Physics_Informed_AI/logg/[topic]/exp/exp_[name]_YYYYMMDD.md"
[按 exp.md 模板填写]
EOF
```

---

## 🗂️ 参考代码（⚠️ 只写路径，禁止写代码）

> **强制规则**：
> - ❌ 禁止在此写任何代码块、代码骨架、示例代码
> - ✅ Agent 执行时必须先阅读下方路径中的代码，理解逻辑后再修改
> - 💡 这样做确保复用已有代码逻辑，避免不一致

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `[仓库路径/script.py]` | `func()` | [修改说明] |

---

## 🎯 实验规格

```yaml
experiment_id: "[PROJECT]-[YYYYMMDD]-[topic]-[##]"
repo_path: "~/VIT"
data: { source: "", path: "", split: N/N/N }
noise: { sigma: 0.1, apply_to: train }
model: { type: "" }
training: { epochs: N, batch: N, lr: 1e-4, seed: 42 }
plots: [{ type: loss_curve, save: "[exp_id]_loss.png" }]
```

---

## ✅ 检查清单

- [ ] 训练完成
- [ ] 图表(英文) + 已在报告 §3 引用
- [ ] 报告(中文)

---

## 🔧 故障排除

| 问题 | 修复 |
|------|------|
| NaN | 降 lr / grad_clip |
| OOM | 减 batch_size |
| Loss爆炸 | 降 lr / warmup |
