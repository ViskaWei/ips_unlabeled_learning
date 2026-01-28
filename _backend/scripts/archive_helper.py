#!/usr/bin/env python3
"""
归档助手 (Archive Helper)

用法:
    python scripts/archive_helper.py status   # 查看待归档文件
    python scripts/archive_helper.py scan     # 扫描新文件并更新队列
    python scripts/archive_helper.py --watch  # 持续监听 (后台运行)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 配置
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIRS = ["raw"]  # 统一使用 raw 目录
LOGG_DIR = PROJECT_ROOT / "logg"
QUEUE_FILE = PROJECT_ROOT / "status" / "archive_queue.md"

# 目录映射
DIR_MAPPING = {
    "cnn": "logg/cnn/",
    "nn": "logg/NN/",
    "mlp": "logg/NN/",
    "ridge": "logg/ridge/",
    "pca": "logg/pca/",
    "lightgbm": "logg/lightgbm/",
    "noise": "logg/noise/",
    "train": "logg/train/",
    "val": "logg/train/",
    "distill": "logg/distill/",
    "latent": "logg/distill/",
    "probe": "logg/distill/",
    "pool": "logg/distill/",
    "gta": "logg/gta/",
    "global": "logg/gta/",
}

def infer_target_dir(filename: str) -> str:
    """根据文件名推断目标目录"""
    fname_lower = filename.lower()
    for keyword, target in DIR_MAPPING.items():
        if keyword in fname_lower:
            return target
    return "logg/misc/"  # 默认目录

def get_archived_files() -> set:
    """获取已归档到 logg/ 的文件"""
    archived = set()
    for md_file in LOGG_DIR.rglob("exp_*.md"):
        archived.add(md_file.stem)
    return archived

def scan_raw_files() -> list:
    """扫描 raw_* 目录中的 md 文件"""
    pending = []
    for raw_dir in RAW_DIRS:
        raw_path = PROJECT_ROOT / raw_dir
        if not raw_path.exists():
            continue
        for md_file in raw_path.glob("*.md"):
            target = infer_target_dir(md_file.name)
            pending.append({
                "source": f"{raw_dir}/{md_file.name}",
                "target": target,
                "mtime": datetime.fromtimestamp(md_file.stat().st_mtime)
            })
    return sorted(pending, key=lambda x: x["mtime"], reverse=True)

def print_status():
    """打印当前状态"""
    files = scan_raw_files()
    
    print("\n" + "="*60)
    print("📋 归档队列状态")
    print("="*60)
    
    if not files:
        print("\n✅ 没有待归档的文件！")
        return
    
    print(f"\n📁 待归档文件 ({len(files)} 个):\n")
    print(f"{'序号':<4} {'源文件':<45} {'目标目录':<20}")
    print("-" * 70)
    
    for i, f in enumerate(files, 1):
        print(f"{i:<4} {f['source']:<45} {f['target']:<20}")
    
    print("\n" + "-"*60)
    print("💡 在 Cursor 中说 '归档 [序号]' 来归档指定文件")
    print("💡 或说 '归档 all' 批量处理所有文件")
    print("="*60 + "\n")

def update_queue_file():
    """更新 status/archive_queue.md"""
    files = scan_raw_files()
    today = datetime.now().strftime("%Y-%m-%d")
    
    pending_rows = []
    for f in files:
        priority = "🔴 高" if "FULL" in f["source"].upper() else "🟡 中"
        pending_rows.append(f"| `{f['source']}` | `{f['target']}` | {priority} |")
    
    content = f"""# 📋 归档队列 (Archive Queue)

> **自动更新**: 此文件记录待归档的原始报告
> **使用方法**: 在 Cursor 中说 `归档` 或 `archive`，AI 会自动处理队列

---

## ⏳ 待归档 (Pending)

| 源文件 | 目标目录 | 优先级 |
|--------|----------|--------|
{chr(10).join(pending_rows) if pending_rows else "| - | - | - |"}

---

## ✅ 已归档 (Archived)

| 源文件 | 归档报告 | 归档日期 |
|--------|----------|----------|
| - | - | - |

---

## 📊 统计

- **待归档**: {len(files)}
- **已归档**: 0
- **最后更新**: {today}

---

> **快捷操作**:
> - `归档 [文件名]` - 归档指定文件
> - `归档 all` - 批量归档所有待处理文件
> - `归档状态` - 查看当前队列
"""
    
    QUEUE_FILE.parent.mkdir(exist_ok=True)
    QUEUE_FILE.write_text(content)
    print(f"✅ 已更新: {QUEUE_FILE}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == "status":
        print_status()
    elif cmd == "scan":
        update_queue_file()
        print_status()
    elif cmd == "--watch":
        print("🔄 监听模式暂未实现，请使用 scan 命令")
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)

if __name__ == "__main__":
    main()

