#!/usr/bin/env python3
"""
Auto Report Generator
=====================

Generate weekly/monthly/brief reports by scanning:
1. experiments_index/index.csv for completed experiments
2. logg/**/*.md for updated knowledge documents
3. status/next_steps.md for pending tasks

Usage:
    python generate_report.py --type weekly
    python generate_report.py --type adhoc --start "2025-11-25" --end "2025-12-01"
    python generate_report.py -4h          # Brief report for last 4 hours (for LLM)
    python generate_report.py -24h         # Brief report for last 24 hours
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Constants
PHYSICS_AI_ROOT = Path(__file__).parent.parent
INDEX_CSV = PHYSICS_AI_ROOT / "experiments_index" / "index.csv"
LOGG_DIR = PHYSICS_AI_ROOT / "logg"
NEXT_STEPS_FILE = PHYSICS_AI_ROOT / "status" / "next_steps.md"
REPORTS_DIR = PHYSICS_AI_ROOT / "reports"
LAST_REPORT_FILE = REPORTS_DIR / "last_report.json"
HISTORY_FILE = REPORTS_DIR / "history.csv"
DRAFTS_DIR = REPORTS_DIR / "drafts"


def load_last_report() -> dict:
    """Load the last report metadata."""
    if LAST_REPORT_FILE.exists():
        with open(LAST_REPORT_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_last_report(report_meta: dict):
    """Save report metadata."""
    with open(LAST_REPORT_FILE, 'w') as f:
        json.dump(report_meta, f, indent=2)


def append_to_history(report_meta: dict):
    """Append report to history.csv."""
    history_fields = [
        "report_id", "type", "period_start", "period_end",
        "generated_at", "draft_path", "experiments_count",
        "insights_count", "summary"
    ]
    
    file_exists = HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0
    
    with open(HISTORY_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: report_meta.get(k, '') for k in history_fields})


def get_completed_experiments(since: datetime, until: datetime) -> list[dict]:
    """Get experiments completed in the time window."""
    if not INDEX_CSV.exists():
        return []
    
    experiments = []
    with open(INDEX_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "completed":
                continue
            
            end_time = row.get("end_time", "")
            if not end_time:
                continue
            
            try:
                exp_time = datetime.fromisoformat(end_time.replace("Z", "+00:00").split("+")[0])
                if since <= exp_time <= until:
                    experiments.append(row)
            except:
                pass
    
    return experiments


def get_updated_logg_files(since: datetime, until: datetime) -> list[dict]:
    """Get logg files modified in the time window."""
    if not LOGG_DIR.exists():
        return []
    
    files = []
    for md_file in LOGG_DIR.glob("**/*.md"):
        try:
            mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
            if since <= mtime <= until:
                # Extract title from first line
                title = ""
                content_preview = ""
                conclusions = []
                
                with open(md_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for line in lines:
                        if line.startswith("# "):
                            title = line[2:].strip()
                            break
                    
                    # Extract key conclusions (look for specific patterns)
                    # Pattern 1: "核心结论速览" section
                    if "核心结论速览" in content or "一句话总结" in content:
                        in_conclusion = False
                        for line in lines:
                            if "一句话总结" in line or "Core Finding" in line:
                                in_conclusion = True
                                continue
                            if in_conclusion and line.startswith(">"):
                                conclusions.append(line.strip("> ").strip())
                                break
                            if in_conclusion and line.startswith("##"):
                                break
                    
                    # Pattern 2: Look for key metrics in tables
                    metrics_pattern = re.findall(r'R[²2]\s*[=:]\s*([0-9.]+)', content)
                    if metrics_pattern:
                        best_r2 = max(float(m) for m in metrics_pattern)
                        conclusions.append(f"Best R²={best_r2:.4f}")
                
                # Determine type (main vs exp)
                file_type = "main" if "_main_" in md_file.name else "exp"
                
                # Extract topic from path
                topic = md_file.parent.name if md_file.parent != LOGG_DIR else "general"
                
                files.append({
                    "path": str(md_file.relative_to(PHYSICS_AI_ROOT)),
                    "title": title,
                    "type": file_type,
                    "topic": topic,
                    "mtime": mtime.isoformat(),
                    "conclusions": conclusions
                })
        except:
            pass
    
    return sorted(files, key=lambda x: x["mtime"], reverse=True)


def parse_next_steps() -> dict:
    """Parse next_steps.md for P0 and P1 tasks."""
    if not NEXT_STEPS_FILE.exists():
        return {"P0": [], "P1": []}
    
    tasks = {"P0": [], "P1": []}
    current_priority = None
    
    with open(NEXT_STEPS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect priority section
            if "🔴 P0" in line or "P0 —" in line:
                current_priority = "P0"
            elif "🟡 P1" in line or "P1 —" in line:
                current_priority = "P1"
            elif "🟢 P2" in line or "P2 —" in line:
                current_priority = None  # Stop at P2
            elif "✅ 已完成" in line:
                current_priority = None  # Stop at completed
            
            # Parse task lines
            if current_priority and line.startswith("| [ ]"):
                # Extract task from table row
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    task = {
                        "status": parts[1],
                        "experiment_id": parts[2] if len(parts) > 2 else "",
                        "description": parts[3] if len(parts) > 3 else "",
                    }
                    tasks[current_priority].append(task)
    
    return tasks


def generate_brief_report(
    hours: int,
    period_start: datetime,
    period_end: datetime,
    experiments: list[dict],
    logg_files: list[dict],
    next_steps: dict
) -> str:
    """Generate a brief report optimized for LLM consumption."""
    
    now = datetime.now()
    lines = []
    
    # Header - concise
    lines.append(f"# 🧠 实验进展摘要 (Past {hours}h)")
    lines.append("")
    lines.append(f"**时间范围**: {period_start.strftime('%Y-%m-%d %H:%M')} → {period_end.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**生成时间**: {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Section 1: Core Conclusions (最重要)
    lines.append("## 📌 核心结论")
    lines.append("")
    
    has_conclusions = False
    for logg in logg_files:
        if logg.get("conclusions"):
            has_conclusions = True
            topic = logg.get("topic", "")
            title = logg.get("title", "")
            lines.append(f"### [{topic}] {title}")
            for conclusion in logg["conclusions"]:
                lines.append(f"- {conclusion}")
            lines.append("")
    
    if not has_conclusions:
        lines.append("*无新结论提取*")
        lines.append("")
    
    # Section 2: Updated Documents
    lines.append("## 📝 更新的文档")
    lines.append("")
    
    if logg_files:
        for logg in logg_files[:5]:  # Top 5
            file_type = "📘" if logg["type"] == "main" else "📗"
            lines.append(f"- {file_type} `{logg['path']}` - {logg['title']}")
        lines.append("")
    else:
        lines.append("*无更新*")
        lines.append("")
    
    # Section 3: Experiments (if any)
    if experiments:
        lines.append("## 🔬 完成的实验")
        lines.append("")
        for exp in experiments:
            exp_id = exp.get("experiment_id", "")
            metrics = exp.get("metrics_summary", "N/A")
            lines.append(f"- `{exp_id}`: {metrics}")
        lines.append("")
    
    # Section 4: Pending Tasks
    lines.append("## ⏳ 待办任务")
    lines.append("")
    
    if next_steps.get("P0"):
        lines.append("**🔴 P0 (高优先)**:")
        for task in next_steps["P0"]:
            lines.append(f"- [ ] {task.get('description', '')}")
        lines.append("")
    
    if next_steps.get("P1"):
        lines.append("**🟡 P1 (中优先)**:")
        for task in next_steps["P1"][:3]:  # Top 3
            lines.append(f"- [ ] {task.get('description', '')}")
        lines.append("")
    
    # Section 5: Prompt for LLM
    lines.append("---")
    lines.append("")
    lines.append("## 🤖 给 AI 的提示")
    lines.append("")
    lines.append("基于以上进展，请帮我思考：")
    lines.append("")
    lines.append("1. **结论验证**: 上述核心结论是否合理？有什么需要进一步验证的？")
    lines.append("2. **下一步建议**: 基于当前结果，最值得尝试的下一个实验是什么？")
    lines.append("3. **潜在问题**: 有没有发现任何可能的问题或需要注意的地方？")
    lines.append("4. **优先级调整**: P0/P1 任务的优先级是否需要调整？")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*此报告为 LLM 优化格式，可直接复制给 ChatGPT/Claude 进行分析*")
    
    return "\n".join(lines)


def generate_report_content(
    report_type: str,
    period_start: datetime,
    period_end: datetime,
    experiments: list[dict],
    logg_files: list[dict],
    next_steps: dict
) -> str:
    """Generate the report markdown content."""
    
    now = datetime.now()
    
    # Group experiments by project and topic
    exp_by_project = {}
    for exp in experiments:
        project = exp.get("project", "Other")
        topic = exp.get("topic", "other")
        if project not in exp_by_project:
            exp_by_project[project] = {}
        if topic not in exp_by_project[project]:
            exp_by_project[project][topic] = []
        exp_by_project[project][topic].append(exp)
    
    # Group logg files by topic
    logg_by_topic = {}
    for logg in logg_files:
        topic = logg.get("topic", "other")
        if topic not in logg_by_topic:
            logg_by_topic[topic] = []
        logg_by_topic[topic].append(logg)
    
    # Build report
    lines = []
    
    # Header
    lines.append(f"# 📊 {report_type.capitalize()} Report")
    lines.append("")
    lines.append(f"> **Period**: {period_start.strftime('%Y-%m-%d')} → {period_end.strftime('%Y-%m-%d')}")
    lines.append(f"> **Generated**: {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> **Author**: Viska Wei")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Overview
    lines.append("## 1. Overview")
    lines.append("")
    total_exp = len(experiments)
    total_logg = len(logg_files)
    total_p0 = len(next_steps.get("P0", []))
    
    if total_exp > 0 or total_logg > 0:
        lines.append(f"This period saw **{total_exp} experiments** completed across ")
        lines.append(f"{len(exp_by_project)} projects, with **{total_logg} knowledge documents** updated.")
        if total_p0 > 0:
            lines.append(f" There are **{total_p0} high-priority tasks** pending.")
    else:
        lines.append("No new experiments or knowledge updates in this period.")
    lines.append("")
    
    # Key highlights (top 3 experiments by metrics)
    if experiments:
        lines.append("**Key Highlights**:")
        for exp in experiments[:3]:
            exp_id = exp.get("experiment_id", "")
            metrics = exp.get("metrics_summary", "")
            lines.append(f"- `{exp_id}`: {metrics if metrics else 'N/A'}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Completed Experiments
    lines.append("## 2. New Experiments Completed")
    lines.append("")
    
    if exp_by_project:
        for project, topics in sorted(exp_by_project.items()):
            lines.append(f"### {project}")
            lines.append("")
            
            for topic, exps in sorted(topics.items()):
                lines.append(f"#### {topic}")
                lines.append("")
                lines.append("| Experiment ID | Metrics | Output |")
                lines.append("|---------------|---------|--------|")
                for exp in exps:
                    exp_id = exp.get("experiment_id", "")
                    metrics = exp.get("metrics_summary", "N/A")
                    output = exp.get("output_path", "")
                    lines.append(f"| `{exp_id}` | {metrics[:40]} | {output[:30]}... |")
                lines.append("")
    else:
        lines.append("No new experiments completed in this period.")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Knowledge Updates
    lines.append("## 3. Key Insights & Knowledge Updates")
    lines.append("")
    
    if logg_by_topic:
        for topic, files in sorted(logg_by_topic.items()):
            lines.append(f"### {topic}")
            lines.append("")
            for f in files:
                file_type = "📘 Main" if f["type"] == "main" else "📗 Exp"
                lines.append(f"- {file_type} **{f['title']}** (`{f['path']}`)")
            lines.append("")
    else:
        lines.append("No knowledge documents updated in this period.")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Open Questions & Next Steps
    lines.append("## 4. Open Questions & Next Steps")
    lines.append("")
    
    if next_steps.get("P0"):
        lines.append("### 🔴 P0 — High Priority")
        lines.append("")
        for task in next_steps["P0"]:
            lines.append(f"- [ ] {task.get('description', 'N/A')}")
        lines.append("")
    
    if next_steps.get("P1"):
        lines.append("### 🟡 P1 — Medium Priority")
        lines.append("")
        for task in next_steps["P1"]:
            lines.append(f"- [ ] {task.get('description', 'N/A')}")
        lines.append("")
    
    if not next_steps.get("P0") and not next_steps.get("P1"):
        lines.append("No pending high-priority tasks.")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Appendix
    lines.append("## 5. Appendix")
    lines.append("")
    lines.append("### Quick Links")
    lines.append("")
    lines.append("| Resource | Path |")
    lines.append("|----------|------|")
    lines.append("| Experiment Index | `experiments_index/index.csv` |")
    lines.append("| Next Steps | `status/next_steps.md` |")
    lines.append("| VIT Repository | `~/VIT/` |")
    lines.append("| BlindSpot Repository | `~/BlindSpotDenoiser/` |")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append(f"*Auto-generated by `scripts/generate_report.py` on {now.strftime('%Y-%m-%d %H:%M')}*")
    
    return "\n".join(lines)


def generate_report(
    report_type: str = "weekly",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    hours: Optional[int] = None
) -> str:
    """Generate a report and save it."""
    
    # Determine time window
    now = datetime.now()
    
    # Handle hours-based brief report
    if hours is not None:
        period_end = now
        period_start = now - timedelta(hours=hours)
        report_type = f"brief_{hours}h"
        is_brief = True
    elif start_date and end_date:
        period_start = datetime.fromisoformat(start_date)
        period_end = datetime.fromisoformat(end_date)
        is_brief = False
    elif report_type == "weekly":
        period_end = now
        period_start = now - timedelta(days=7)
        is_brief = False
    elif report_type == "monthly":
        period_end = now
        period_start = now - timedelta(days=30)
        is_brief = False
    else:
        # Default to since last report
        last_report = load_last_report()
        if last_report.get("period_end"):
            period_start = datetime.fromisoformat(last_report["period_end"])
        else:
            period_start = now - timedelta(days=7)
        period_end = now
        is_brief = False
    
    print(f"📊 Generating {report_type} report")
    print(f"   Period: {period_start.strftime('%Y-%m-%d %H:%M')} → {period_end.strftime('%Y-%m-%d %H:%M')}")
    
    # Gather data
    print("\n🔍 Gathering data...")
    experiments = get_completed_experiments(period_start, period_end)
    print(f"   ✓ Found {len(experiments)} completed experiments")
    
    logg_files = get_updated_logg_files(period_start, period_end)
    print(f"   ✓ Found {len(logg_files)} updated logg files")
    
    next_steps = parse_next_steps()
    print(f"   ✓ Found {len(next_steps.get('P0', []))} P0 tasks, {len(next_steps.get('P1', []))} P1 tasks")
    
    # Generate content
    print("\n✍️ Generating report content...")
    
    if is_brief or (hours is not None):
        content = generate_brief_report(
            hours=hours or 4,
            period_start=period_start,
            period_end=period_end,
            experiments=experiments,
            logg_files=logg_files,
            next_steps=next_steps
        )
    else:
        content = generate_report_content(
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            experiments=experiments,
            logg_files=logg_files,
            next_steps=next_steps
        )
    
    # Save draft
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if hours is not None:
        draft_filename = f"brief_{hours}h_{now.strftime('%Y%m%d_%H%M')}.md"
    else:
        draft_filename = f"{report_type}_{period_end.strftime('%Y-%m-%d')}.md"
    
    draft_path = DRAFTS_DIR / draft_filename
    
    with open(draft_path, 'w') as f:
        f.write(content)
    
    print(f"\n✅ Report saved to: {draft_path}")
    
    # Also print content to stdout for easy copy
    if hours is not None:
        print("\n" + "="*60)
        print("📋 COPY THE FOLLOWING TO ChatGPT/Claude:")
        print("="*60 + "\n")
        print(content)
        print("\n" + "="*60)
    
    # Update metadata
    report_id = f"{report_type}-{now.strftime('%Y%m%d_%H%M')}"
    report_meta = {
        "report_id": report_id,
        "type": report_type,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "generated_at": now.isoformat(),
        "draft_path": str(draft_path.relative_to(PHYSICS_AI_ROOT)),
        "experiments_count": len(experiments),
        "insights_count": len(logg_files),
        "summary": f"{len(experiments)} experiments, {len(logg_files)} docs updated"
    }
    
    save_last_report(report_meta)
    append_to_history(report_meta)
    
    print(f"   Updated: reports/last_report.json")
    print(f"   Updated: reports/history.csv")
    
    return str(draft_path)


def parse_hours_arg(arg: str) -> Optional[int]:
    """Parse -Nh argument (e.g., -4h, -24h)."""
    match = re.match(r'^-?(\d+)h$', arg)
    if match:
        return int(match.group(1))
    return None


def main():
    # Check for -Nh style argument first
    if len(sys.argv) == 2:
        hours = parse_hours_arg(sys.argv[1])
        if hours is not None:
            generate_report(hours=hours)
            return
    
    parser = argparse.ArgumentParser(
        description="Generate experiment reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Weekly report
  python generate_report.py --type weekly
  
  # Brief report for last 4 hours (optimized for LLM)
  python generate_report.py -4h
  
  # Brief report for last 24 hours
  python generate_report.py -24h
  
  # Custom date range
  python generate_report.py --type adhoc --start "2025-11-25" --end "2025-12-01"
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["weekly", "monthly", "adhoc"],
        default="weekly",
        help="Report type"
    )
    parser.add_argument(
        "--start", "-s",
        type=str,
        help="Start date (YYYY-MM-DD) for adhoc reports"
    )
    parser.add_argument(
        "--end", "-e",
        type=str,
        help="End date (YYYY-MM-DD) for adhoc reports"
    )
    parser.add_argument(
        "--hours", "-H",
        type=int,
        help="Generate brief report for last N hours (for LLM)"
    )
    
    args = parser.parse_args()
    
    if args.hours:
        generate_report(hours=args.hours)
    elif args.type == "adhoc" and (not args.start or not args.end):
        parser.error("--start and --end are required for adhoc reports")
    else:
        generate_report(
            report_type=args.type,
            start_date=args.start,
            end_date=args.end
        )


if __name__ == "__main__":
    main()
