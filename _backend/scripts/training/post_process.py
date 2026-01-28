#!/usr/bin/env python3
"""
训练后处理模块 (Post-Processing Module)
=======================================

功能：
- 训练完成后自动执行的处理步骤
- 生成 metrics.csv 汇总
- 生成 summary.json 
- 创建 exp.md 报告骨架（供 Cursor 填充）
- 复制关键结果到知识中心

核心理念：
- **减少给 Cursor 的 token**：只提供精简的 summary，而不是完整日志
- **自动化归档**：生成报告骨架，便于后续填充

输出文件：
- results/{exp_id}/metrics.csv     - 训练指标时间序列
- results/{exp_id}/summary.json    - 实验配置 + 最终结果
- results/{exp_id}/report_draft.md - exp.md 报告骨架

用法：
    python post_process.py --exp-id VIT-20251204-xxx --work-dir ~/VIT

作者: Viska Wei
日期: 2025-12-04
"""

import os
import sys
import json
import csv
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse


class PostProcessor:
    """训练后处理器"""
    
    # 知识中心路径
    KNOWLEDGE_CENTER = Path("/home/swei20/Physics_Informed_AI")
    
    def __init__(
        self,
        exp_id: str,
        work_dir: Path,
        results_dir: Optional[Path] = None,
    ):
        self.exp_id = exp_id
        self.work_dir = Path(work_dir)
        self.results_dir = results_dir or (self.work_dir / "results" / exp_id)
        
        # 确保目录存在
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志和信号目录
        self.logs_dir = self.work_dir / "logs"
        self.signals_dir = self.work_dir / "signals"
        
        # 输出文件
        self.metrics_csv = self.results_dir / "metrics.csv"
        self.summary_json = self.results_dir / "summary.json"
        self.report_draft = self.results_dir / "report_draft.md"
    
    def _log(self, msg: str, level: str = "INFO"):
        """带时间戳的日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
        print(f"[{timestamp}] {prefix} {msg}")
    
    def run(self):
        """执行所有后处理步骤"""
        self._log("开始后处理...")
        
        # 1. 提取训练指标
        self.extract_metrics()
        
        # 2. 生成实验摘要
        self.generate_summary()
        
        # 3. 生成报告骨架
        self.generate_report_draft()
        
        # 4. 同步到知识中心（可选）
        # self.sync_to_knowledge_center()
        
        self._log("后处理完成！", "SUCCESS")
        self._print_output_files()
    
    def extract_metrics(self):
        """从训练日志中提取指标到 CSV"""
        log_file = self.logs_dir / f"{self.exp_id}.log"
        
        if not log_file.exists():
            self._log(f"日志文件不存在: {log_file}", "WARNING")
            return
        
        self._log("提取训练指标...")
        
        # 正则模式：支持多种日志格式
        patterns = {
            "step": re.compile(r'step[=:\s]+(\d+)', re.IGNORECASE),
            "epoch": re.compile(r'epoch[=:\s]+(\d+)', re.IGNORECASE),
            "loss": re.compile(r'(?:train_)?loss[=:\s]+([0-9.eE+-]+)', re.IGNORECASE),
            "val_loss": re.compile(r'val_loss[=:\s]+([0-9.eE+-]+)', re.IGNORECASE),
            "lr": re.compile(r'(?:learning_rate|lr)[=:\s]+([0-9.eE+-]+)', re.IGNORECASE),
            "r2": re.compile(r'r2[=:\s]+([0-9.eE+-]+)', re.IGNORECASE),
            "mae": re.compile(r'mae[=:\s]+([0-9.eE+-]+)', re.IGNORECASE),
        }
        
        rows = []
        current_row = {}
        
        with open(log_file, "r", errors='ignore') as f:
            for line in f:
                for key, pattern in patterns.items():
                    match = pattern.search(line)
                    if match:
                        try:
                            value = float(match.group(1)) if key != "step" and key != "epoch" else int(match.group(1))
                            current_row[key] = value
                        except ValueError:
                            continue
                
                # 每次找到 step 或 epoch 时保存一行
                if ("step" in current_row or "epoch" in current_row) and "loss" in current_row:
                    rows.append(current_row.copy())
                    current_row = {}
        
        if not rows:
            self._log("未能提取到指标", "WARNING")
            return
        
        # 写入 CSV
        fieldnames = ["step", "epoch", "loss", "val_loss", "lr", "r2", "mae"]
        with open(self.metrics_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[fn for fn in fieldnames if any(fn in row for row in rows)])
            writer.writeheader()
            for row in rows:
                writer.writerow({k: v for k, v in row.items() if k in fieldnames})
        
        self._log(f"提取了 {len(rows)} 条指标记录", "SUCCESS")
    
    def generate_summary(self):
        """生成实验摘要 JSON"""
        self._log("生成实验摘要...")
        
        summary = {
            "exp_id": self.exp_id,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
        }
        
        # 读取信号文件中的信息
        done_file = self.signals_dir / f"{self.exp_id}.done"
        if done_file.exists():
            for line in done_file.read_text().strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    summary[key.strip()] = value.strip()
        
        # 从 metrics.csv 提取最终指标
        if self.metrics_csv.exists():
            with open(self.metrics_csv, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    summary["final_metrics"] = {k: float(v) for k, v in last_row.items() if v}
                    
                    # 最佳指标
                    if "loss" in last_row:
                        losses = [float(r["loss"]) for r in rows if "loss" in r and r["loss"]]
                        summary["best_loss"] = min(losses)
                        summary["final_loss"] = float(last_row["loss"])
                    
                    if "r2" in last_row:
                        r2s = [float(r["r2"]) for r in rows if "r2" in r and r["r2"]]
                        summary["best_r2"] = max(r2s)
                        summary["final_r2"] = float(last_row["r2"])
        
        # 写入 JSON
        with open(self.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        
        self._log("摘要已生成", "SUCCESS")
    
    def generate_report_draft(self):
        """生成 exp.md 报告骨架"""
        self._log("生成报告骨架...")
        
        # 读取摘要
        summary = {}
        if self.summary_json.exists():
            with open(self.summary_json) as f:
                summary = json.load(f)
        
        # 从 exp_id 推断主题
        topic = self._infer_topic(self.exp_id)
        date = datetime.now().strftime("%Y%m%d")
        
        # 生成报告骨架
        report = f"""# {self.exp_id} 实验报告

> **实验 ID**: {self.exp_id}
> **状态**: 🔄 待填充
> **日期**: {datetime.now().strftime("%Y-%m-%d")}
> **作者**: Viska Wei

---

## 🔗 上游追溯链接

- **来源会话**: <!-- TODO: 链接到 session.md -->
- **队列入口**: <!-- TODO: 链接到 kanban.md -->

---

## ⚡ 核心结论速览

| 项目 | 内容 |
|------|------|
| **一句话总结** | <!-- TODO --> |
| **假设验证** | <!-- ❌/✅ H?.? --> |
| **关键数字** | R²={summary.get('final_r2', 'TODO')}, Loss={summary.get('final_loss', 'TODO')} |
| **设计启示** | <!-- TODO --> |

---

## 1. 🎯 目标

### 1.1 实验目的
<!-- TODO: 填写实验目的 -->

### 1.2 预期结果
<!-- TODO: 填写预期结果 -->

---

## 2. 🧪 实验设计

### 2.1 数据
<!-- TODO: 数据配置 -->

### 2.2 模型与算法
<!-- TODO: 模型配置 -->

### 2.3 超参数配置
| 超参数 | 值 |
|--------|-----|
| TODO | TODO |

### 2.4 评价指标
- R²
- MAE
- Loss

---

## 3. 📊 实验图表

<!-- TODO: 添加图表 -->

---

## 4. 💡 关键洞见

### 4.1 宏观层洞见
<!-- TODO -->

### 4.2 模型层洞见
<!-- TODO -->

### 4.3 实验层细节洞见
<!-- TODO -->

---

## 5. 📝 结论

### 5.1 核心发现
<!-- TODO -->

### 5.2 关键结论
<!-- TODO -->

### 5.3 设计启示
<!-- TODO -->

### 5.4 物理解释
<!-- TODO -->

### 5.5 关键数字速查
| 指标 | 值 |
|------|-----|
| 最终 R² | {summary.get('final_r2', 'TODO')} |
| 最佳 R² | {summary.get('best_r2', 'TODO')} |
| 最终 Loss | {summary.get('final_loss', 'TODO')} |
| 最佳 Loss | {summary.get('best_loss', 'TODO')} |

### 5.6 下一步工作
<!-- TODO -->

---

## 6. 📎 附录

### 6.1 数值结果表
<!-- 从 metrics.csv 生成 -->

### 6.2 实验流程记录

**执行命令**:
```bash
# TODO: 填写执行命令
```

**关键日志**:
```
# TODO: 粘贴关键日志片段
```

### 6.3 相关文件
- 日志: `{self.logs_dir / f"{self.exp_id}.log"}`
- 指标: `{self.metrics_csv}`
- 摘要: `{self.summary_json}`

---

*报告自动生成于 {datetime.now().isoformat()}*
"""
        
        with open(self.report_draft, "w") as f:
            f.write(report)
        
        self._log("报告骨架已生成", "SUCCESS")
    
    def _infer_topic(self, exp_id: str) -> str:
        """从实验 ID 推断主题"""
        lower_id = exp_id.lower()
        
        topics = {
            "cnn": "cnn",
            "conv": "cnn",
            "dilat": "cnn",
            "moe": "moe",
            "expert": "moe",
            "nn": "NN",
            "mlp": "NN",
            "swin": "swin",
            "vit": "swin",
            "transformer": "swin",
            "ridge": "ridge",
            "linear": "ridge",
            "pca": "pca",
            "distill": "distill",
            "latent": "distill",
            "gta": "gta",
            "global": "gta",
            "diffusion": "diffusion",
            "noise": "noise",
            "lightgbm": "lightgbm",
            "lgbm": "lightgbm",
        }
        
        for keyword, topic in topics.items():
            if keyword in lower_id:
                return topic
        
        return "NN"  # 默认
    
    def sync_to_knowledge_center(self):
        """同步到知识中心（使用终端命令避免跨仓库写入问题）"""
        self._log("同步到知识中心...")
        
        topic = self._infer_topic(self.exp_id)
        target_dir = self.KNOWLEDGE_CENTER / "logg" / topic
        
        # 使用终端命令复制
        import subprocess
        
        # 复制 summary.json
        if self.summary_json.exists():
            subprocess.run([
                "cp", str(self.summary_json),
                str(target_dir / f"{self.exp_id}_summary.json")
            ], check=False)
        
        # 复制报告骨架
        if self.report_draft.exists():
            date = datetime.now().strftime("%Y%m%d")
            subprocess.run([
                "cp", str(self.report_draft),
                str(target_dir / f"exp_{self.exp_id}_{date}.md")
            ], check=False)
        
        self._log("同步完成", "SUCCESS")
    
    def _print_output_files(self):
        """打印输出文件列表"""
        print()
        print("═" * 50)
        print("📁 输出文件:")
        print("═" * 50)
        
        files = [
            (self.metrics_csv, "训练指标 CSV"),
            (self.summary_json, "实验摘要 JSON"),
            (self.report_draft, "报告骨架 MD"),
        ]
        
        for path, desc in files:
            if path.exists():
                size = path.stat().st_size
                print(f"  ✅ {path.name:<25} ({size:,} bytes) - {desc}")
            else:
                print(f"  ⚪ {path.name:<25} (未生成) - {desc}")
        
        print()
        print("💡 提示:")
        print(f"   1. 查看摘要: cat {self.summary_json}")
        print(f"   2. 填充报告: 把 {self.report_draft} 内容给 Cursor")
        print(f"   3. 归档: a {self.exp_id}")
        print("═" * 50)


def generate_cursor_prompt(summary_json: Path, metrics_csv: Path) -> str:
    """
    生成给 Cursor 的精简 prompt
    
    这是核心函数：不把整个日志给 Cursor，只给摘要
    """
    prompt_parts = []
    
    # 读取摘要
    if summary_json.exists():
        with open(summary_json) as f:
            summary = json.load(f)
        
        prompt_parts.append("## 实验摘要\n")
        prompt_parts.append(f"- 实验 ID: {summary.get('exp_id', 'unknown')}")
        prompt_parts.append(f"- 状态: {summary.get('status', 'unknown')}")
        
        if "final_metrics" in summary:
            prompt_parts.append("\n### 最终指标")
            for k, v in summary["final_metrics"].items():
                prompt_parts.append(f"- {k}: {v}")
        
        if "best_r2" in summary:
            prompt_parts.append(f"\n### 关键数字")
            prompt_parts.append(f"- 最佳 R²: {summary['best_r2']}")
            prompt_parts.append(f"- 最终 R²: {summary.get('final_r2', 'N/A')}")
            prompt_parts.append(f"- 最佳 Loss: {summary.get('best_loss', 'N/A')}")
            prompt_parts.append(f"- 最终 Loss: {summary.get('final_loss', 'N/A')}")
    
    # 读取最后几行指标
    if metrics_csv.exists():
        prompt_parts.append("\n### 最后 5 个数据点")
        with open(metrics_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)[-5:]
            if rows:
                headers = rows[0].keys()
                prompt_parts.append("| " + " | ".join(headers) + " |")
                prompt_parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in rows:
                    prompt_parts.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    
    prompt_parts.append("\n---")
    prompt_parts.append("请根据以上信息，帮我：")
    prompt_parts.append("1. 总结核心结论（一句话）")
    prompt_parts.append("2. 提炼关键洞见")
    prompt_parts.append("3. 给出设计建议")
    prompt_parts.append("4. 建议下一步实验")
    
    return "\n".join(prompt_parts)


def parse_args():
    parser = argparse.ArgumentParser(description="训练后处理")
    parser.add_argument("--exp-id", "-e", required=True, help="实验 ID")
    parser.add_argument("--work-dir", "-w", default=os.getcwd(), help="工作目录")
    parser.add_argument("--generate-prompt", action="store_true", help="生成 Cursor prompt")
    return parser.parse_args()


def main():
    args = parse_args()
    
    processor = PostProcessor(
        exp_id=args.exp_id,
        work_dir=Path(args.work_dir),
    )
    
    processor.run()
    
    if args.generate_prompt:
        print("\n" + "=" * 50)
        print("📋 Cursor Prompt (复制以下内容):")
        print("=" * 50)
        prompt = generate_cursor_prompt(processor.summary_json, processor.metrics_csv)
        print(prompt)


if __name__ == "__main__":
    main()

