#!/usr/bin/env python3
"""
设计原则提取脚本 (Design Principles Extractor)

用法:
    python _backend/scripts/extract_design_principles.py    # 提取所有hub文件中新增的设计原则
    python _backend/scripts/extract_design_principles.py --check  # 仅检查有哪些新增原则（不写入）
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# 配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
DESIGN_DIR = PROJECT_ROOT / "design"
PRINCIPLES_FILE = DESIGN_DIR / "principles.md"
LOGG_DIR = PROJECT_ROOT / "logg"

# 设计原则章节匹配模式
SECTION_PATTERNS = [
    r"##\s*6\)\s*设计原则",
    r"##\s*5\)\s*设计原则",
    r"##\s*📐\s*设计原则",
    r"#\s*5\.\s*📐\s*设计原则",
    r"#\s*4\.\s*📐\s*设计原则",
]

TABLE_PATTERNS = [
    r"###\s*6\.1\s*已确认原则",
    r"###\s*5\.1\s*已确认原则",
    r"###\s*4\.1\s*已确认原则",
    r"##\s*\d+\.\d+\s*已确认原则",
]


def find_last_sync_time() -> Optional[datetime]:
    """从principles.md中提取最后同步时间"""
    if not PRINCIPLES_FILE.exists():
        return None
    
    content = PRINCIPLES_FILE.read_text(encoding='utf-8')
    
    # 查找"最后同步"标记
    sync_pattern = r"<!--\s*最后同步[：:]\s*(\d{4}-\d{2}-\d{2})\s*-->"
    match = re.search(sync_pattern, content)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d")
        except:
            pass
    
    # 如果没有标记，尝试从变更日志获取最后日期
    log_pattern = r"\|\s*(\d{4}-\d{2}-\d{2})\s*\|.*\|"
    matches = re.findall(log_pattern, content)
    if matches:
        try:
            return datetime.strptime(matches[-1], "%Y-%m-%d")
        except:
            pass
    
    return None


def get_hub_files() -> List[Path]:
    """获取所有hub文件"""
    hub_files = []
    for hub_file in LOGG_DIR.rglob("*_hub*.md"):
        # 跳过备份文件
        if "copy" in hub_file.name.lower() or "bak" in hub_file.name.lower():
            continue
        hub_files.append(hub_file)
    return sorted(hub_files)


def extract_design_principles_section(content: str) -> Optional[Tuple[str, int, int]]:
    """提取设计原则章节内容，返回(内容, 开始行, 结束行)"""
    lines = content.split('\n')
    
    # 找到设计原则章节的开始
    start_idx = None
    for i, line in enumerate(lines):
        for pattern in SECTION_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                start_idx = i
                break
        if start_idx is not None:
            break
    
    if start_idx is None:
        return None
    
    # 找到章节结束（下一个一级或二级标题，或文件结束）
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith('# ') and i > start_idx + 5:  # 至少读取5行后再判断
            end_idx = i
            break
        if line.startswith('## ') and not any(
            keyword in line.lower() for keyword in ['设计原则', 'design', '原则', '关键数字', '已关闭']
        ) and i > start_idx + 10:  # 更宽松的判断
            # 检查是否是设计原则的子章节
            if not any(pattern.replace(r'\s*', ' ').replace('(', r'\(').replace(')', r'\)') in line.lower() 
                      for pattern in TABLE_PATTERNS):
                end_idx = i
                break
    
    section_content = '\n'.join(lines[start_idx:end_idx])
    return (section_content, start_idx, end_idx)


def extract_principles_from_section(section: str, hub_path: Path) -> List[Dict]:
    """从设计原则章节中提取原则条目"""
    principles = []
    lines = section.split('\n')
    
    current_table = None
    table_start = None
    
    for i, line in enumerate(lines):
        # 检测表格开始
        if '|' in line and ('原则' in line or 'Principle' in line.lower() or '建议' in line):
            current_table = []
            table_start = i
            # 跳过表头分隔线
            if i + 1 < len(lines) and '---' in lines[i + 1]:
                continue
        
        # 提取表格行
        if current_table is not None and '|' in line and '---' not in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 3:  # 至少包含编号、原则、建议
                # 提取编号（可能是P1, R1, M1等格式）
                num = parts[0] if parts[0] else f"P{len(principles)+1}"
                principle = parts[1] if len(parts) > 1 else ""
                recommendation = parts[2] if len(parts) > 2 else ""
                scope = parts[3] if len(parts) > 3 else ""
                evidence = parts[4] if len(parts) > 4 else ""
                
                if principle:  # 确保原则描述不为空
                    principles.append({
                        'num': num,
                        'principle': principle,
                        'recommendation': recommendation,
                        'scope': scope,
                        'evidence': evidence,
                        'hub_file': hub_path.name,
                        'hub_path': str(hub_path.relative_to(PROJECT_ROOT)),
                        'line_num': table_start + len(current_table) + 1 if table_start else i + 1,
                    })
                    current_table.append(parts)
        
        # 检测表格结束（空行或新章节）
        if current_table is not None and (not line.strip() or line.startswith('#')):
            current_table = None
            table_start = None
    
    return principles


def get_file_modify_time(file_path: Path) -> datetime:
    """获取文件最后修改时间"""
    return datetime.fromtimestamp(file_path.stat().st_mtime)


def main():
    check_only = '--check' in sys.argv or '-c' in sys.argv
    
    print("🔍 扫描hub文件中的设计原则...")
    
    last_sync = find_last_sync_time()
    if last_sync:
        print(f"📅 上次同步时间: {last_sync.strftime('%Y-%m-%d')}")
    else:
        print("⚠️  未找到上次同步时间，将提取所有原则")
    
    hub_files = get_hub_files()
    print(f"📁 找到 {len(hub_files)} 个hub文件")
    
    all_new_principles = []
    
    for hub_file in hub_files:
        # 检查文件修改时间
        if last_sync and get_file_modify_time(hub_file) < last_sync:
            continue
        
        try:
            content = hub_file.read_text(encoding='utf-8')
            section_result = extract_design_principles_section(content)
            
            if section_result:
                section_content, start_line, end_line = section_result
                principles = extract_principles_from_section(section_content, hub_file)
                
                if principles:
                    print(f"  ✅ {hub_file.name}: 发现 {len(principles)} 个原则")
                    all_new_principles.extend(principles)
                else:
                    print(f"  ⚠️  {hub_file.name}: 找到设计原则章节但未提取到原则条目")
        
        except Exception as e:
            print(f"  ❌ {hub_file.name}: 处理失败 - {e}")
    
    print(f"\n📊 总共发现 {len(all_new_principles)} 个新增设计原则")
    
    if check_only:
        print("\n📋 新增原则预览（前5个）:")
        for i, p in enumerate(all_new_principles[:5], 1):
            print(f"  {i}. [{p['num']}] {p['principle'][:50]}... (来自 {p['hub_file']})")
        if len(all_new_principles) > 5:
            print(f"  ... 还有 {len(all_new_principles) - 5} 个原则")
        return
    
    if not all_new_principles:
        print("\n✅ 没有发现新增的设计原则")
        return
    
    # TODO: 这里应该将新增原则追加到principles.md
    # 由于格式复杂，暂时只输出信息
    print("\n📝 新增原则详情:")
    for p in all_new_principles:
        print(f"\n  [{p['num']}] {p['principle']}")
        print(f"      建议: {p['recommendation']}")
        print(f"      来源: {p['hub_file']}")
    
    print("\n⚠️  提示: 当前版本仅检测，请手动将新增原则添加到 design/principles.md")
    print(f"   或运行: python {__file__} --sync 来尝试自动同步（待实现）")


if __name__ == "__main__":
    main()

