#!/usr/bin/env python3
"""
多步骤流水线编排器 (Pipeline Orchestrator)
==========================================

功能：
- 基于信号文件的状态机管理
- 支持多步骤流水线（训练 → 评估 → 画图 → 汇总）
- 自动串联任务，无需手动监控

信号文件约定：
- {exp_id}.healthy - 健康检查通过
- {exp_id}.done    - 训练完成
- {exp_id}.failed  - 训练失败

用法：
    # 启动编排器，等待训练完成后执行后续步骤
    python orchestrator.py --exp-id VIT-20251204-xxx
    
    # 指定流水线配置
    python orchestrator.py --exp-id xxx --pipeline eval,plot,summary
    
    # 超时设置
    python orchestrator.py --exp-id xxx --timeout 3600  # 1小时超时

作者: Viska Wei
日期: 2025-12-04
"""

import subprocess
import time
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Optional
import json


@dataclass
class PipelineStep:
    """流水线步骤"""
    name: str
    command: list[str]
    description: str = ""
    depends_on: str = ""  # 依赖的信号文件
    produces: str = ""    # 产生的信号文件
    timeout: int = 3600   # 超时（秒）
    optional: bool = False  # 是否可选


class SignalManager:
    """信号文件管理器"""
    
    def __init__(self, signals_dir: Path):
        self.signals_dir = Path(signals_dir)
        self.signals_dir.mkdir(parents=True, exist_ok=True)
    
    def get_signal_path(self, exp_id: str, signal_type: str) -> Path:
        """获取信号文件路径"""
        return self.signals_dir / f"{exp_id}.{signal_type}"
    
    def wait_for_signal(
        self,
        exp_id: str,
        signal_type: str,
        timeout: Optional[int] = None,
        check_interval: int = 5
    ) -> bool:
        """等待信号文件出现"""
        signal_path = self.get_signal_path(exp_id, signal_type)
        start_time = time.time()
        
        while True:
            if signal_path.exists():
                return True
            
            if timeout is not None and time.time() - start_time > timeout:
                return False
            
            time.sleep(check_interval)
    
    def check_signal(self, exp_id: str, signal_type: str) -> bool:
        """检查信号是否存在"""
        return self.get_signal_path(exp_id, signal_type).exists()
    
    def read_signal(self, exp_id: str, signal_type: str) -> dict:
        """读取信号文件内容"""
        signal_path = self.get_signal_path(exp_id, signal_type)
        if not signal_path.exists():
            return {}
        
        result = {"raw": signal_path.read_text()}
        # 解析简单的 key: value 格式
        for line in result["raw"].strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()
        return result
    
    def write_signal(self, exp_id: str, signal_type: str, content: dict | str):
        """写入信号文件"""
        signal_path = self.get_signal_path(exp_id, signal_type)
        if isinstance(content, dict):
            text = "\n".join(f"{k}: {v}" for k, v in content.items())
        else:
            text = str(content)
        signal_path.write_text(text)
    
    def clear_signal(self, exp_id: str, signal_type: str):
        """清除信号文件"""
        signal_path = self.get_signal_path(exp_id, signal_type)
        if signal_path.exists():
            signal_path.unlink()


class PipelineOrchestrator:
    """流水线编排器"""
    
    def __init__(
        self,
        exp_id: str,
        work_dir: Path,
        signals_dir: Optional[Path] = None,
    ):
        self.exp_id = exp_id
        self.work_dir = Path(work_dir)
        self.signals_dir = signals_dir or (self.work_dir / "signals")
        
        self.signal_manager = SignalManager(self.signals_dir)
        self.logs_dir = self.work_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 流水线步骤
        self.steps: list[PipelineStep] = []
        
        # 运行状态
        self.start_time = None
        self.results: dict[str, dict] = {}
    
    def _log(self, msg: str, level: str = "INFO"):
        """带时间戳的日志输出"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "📝",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "RUNNING": "🚀",
            "WAIT": "⏳",
        }.get(level, "")
        print(f"[{timestamp}] {prefix} {msg}")
    
    def add_step(self, step: PipelineStep):
        """添加流水线步骤"""
        self.steps.append(step)
    
    def add_standard_pipeline(self):
        """添加标准的训练后处理流水线"""
        # 1. 等待训练完成
        self.add_step(PipelineStep(
            name="wait_training",
            command=[],  # 无命令，只等待信号
            description="等待训练完成",
            depends_on="done",
        ))
        
        # 2. 评估
        self.add_step(PipelineStep(
            name="evaluate",
            command=["python", "scripts/evaluate.py", "--exp-id", self.exp_id],
            description="运行评估",
            produces="evaluated",
            optional=True,  # 评估脚本可能不存在
        ))
        
        # 3. 画图
        self.add_step(PipelineStep(
            name="plot",
            command=["python", "scripts/plot_results.py", "--exp-id", self.exp_id],
            description="生成图表",
            produces="plotted",
            optional=True,
        ))
        
        # 4. 生成汇总
        self.add_step(PipelineStep(
            name="summary",
            command=["python", "scripts/make_summary.py", "--exp-id", self.exp_id],
            description="生成汇总报告",
            produces="summarized",
            optional=True,
        ))
    
    def wait_for_training(self, timeout: Optional[int] = None) -> bool:
        """等待训练完成"""
        self._log(f"等待训练完成: {self.exp_id}", "WAIT")
        
        # 首先等待健康检查通过（可选）
        if self.signal_manager.check_signal(self.exp_id, "healthy"):
            self._log("健康检查已通过，等待训练完成...", "SUCCESS")
        else:
            self._log("等待健康检查...", "WAIT")
            ok = self.signal_manager.wait_for_signal(
                self.exp_id, "healthy",
                timeout=600,  # 10分钟内应该通过健康检查
                check_interval=10
            )
            if ok:
                self._log("健康检查通过！", "SUCCESS")
            else:
                self._log("健康检查超时或失败", "WARNING")
        
        # 等待训练完成
        done = self.signal_manager.wait_for_signal(
            self.exp_id, "done",
            timeout=timeout,
            check_interval=30
        )
        
        if done:
            signal_content = self.signal_manager.read_signal(self.exp_id, "done")
            status = signal_content.get("done", "unknown")
            self._log(f"训练完成: {status}", "SUCCESS")
            return status != "failed"
        else:
            # 检查是否失败
            if self.signal_manager.check_signal(self.exp_id, "failed"):
                signal_content = self.signal_manager.read_signal(self.exp_id, "failed")
                reason = signal_content.get("reason", "unknown")
                self._log(f"训练失败: {reason}", "ERROR")
                return False
            
            self._log("训练超时", "ERROR")
            return False
    
    def run_step(self, step: PipelineStep) -> bool:
        """运行单个步骤"""
        self._log(f"[{step.name}] {step.description}", "RUNNING")
        
        # 检查依赖
        if step.depends_on:
            if not self.signal_manager.check_signal(self.exp_id, step.depends_on):
                self._log(f"依赖未满足: {step.depends_on}", "WARNING")
                return False
        
        # 如果没有命令（纯等待步骤）
        if not step.command:
            return True
        
        # 运行命令
        try:
            log_file = self.logs_dir / f"{self.exp_id}_{step.name}.log"
            with open(log_file, "w") as f:
                result = subprocess.run(
                    step.command,
                    cwd=self.work_dir,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=step.timeout,
                )
            
            success = result.returncode == 0
            self.results[step.name] = {
                "success": success,
                "return_code": result.returncode,
                "log_file": str(log_file),
            }
            
            if success:
                self._log(f"[{step.name}] 完成", "SUCCESS")
                if step.produces:
                    self.signal_manager.write_signal(
                        self.exp_id, step.produces,
                        {"status": "done", "timestamp": datetime.now().isoformat()}
                    )
            else:
                self._log(f"[{step.name}] 失败 (exit: {result.returncode})", "ERROR")
            
            return success
            
        except subprocess.TimeoutExpired:
            self._log(f"[{step.name}] 超时", "ERROR")
            self.results[step.name] = {"success": False, "reason": "timeout"}
            return False
        except FileNotFoundError:
            if step.optional:
                self._log(f"[{step.name}] 跳过 (脚本不存在)", "WARNING")
                self.results[step.name] = {"success": True, "skipped": True}
                return True
            else:
                self._log(f"[{step.name}] 脚本不存在", "ERROR")
                self.results[step.name] = {"success": False, "reason": "not_found"}
                return False
        except Exception as e:
            self._log(f"[{step.name}] 错误: {e}", "ERROR")
            self.results[step.name] = {"success": False, "reason": str(e)}
            return False
    
    def run(self, timeout: Optional[int] = None) -> bool:
        """运行完整流水线"""
        self.start_time = time.time()
        
        print()
        print("━" * 60)
        print(f"🎬 流水线编排器 - {self.exp_id}")
        print("━" * 60)
        print(f"工作目录: {self.work_dir}")
        print(f"步骤数: {len(self.steps)}")
        print()
        
        # 1. 等待训练完成
        if not self.wait_for_training(timeout):
            self._print_summary(False)
            return False
        
        # 2. 运行后续步骤
        all_success = True
        for step in self.steps:
            if step.name == "wait_training":
                continue  # 已经处理过了
            
            if not self.run_step(step):
                if not step.optional:
                    all_success = False
                    break
        
        self._print_summary(all_success)
        return all_success
    
    def _print_summary(self, success: bool):
        """打印运行摘要"""
        duration = int(time.time() - self.start_time) if self.start_time else 0
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print()
        print("═" * 60)
        status = "✅ 流水线完成" if success else "❌ 流水线失败"
        print(f"{status}")
        print(f"总耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print()
        
        if self.results:
            print("步骤结果:")
            for name, result in self.results.items():
                if result.get("skipped"):
                    status = "⏭️  跳过"
                elif result.get("success"):
                    status = "✅ 成功"
                else:
                    status = f"❌ 失败 ({result.get('reason', 'unknown')})"
                print(f"  - {name}: {status}")
        
        print("═" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="流水线编排器 - 管理训练后的多步骤流程",
    )
    
    parser.add_argument(
        "--exp-id", "-e",
        required=True,
        help="实验 ID"
    )
    parser.add_argument(
        "--work-dir", "-w",
        default=os.getcwd(),
        help="工作目录"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="总超时时间（秒）"
    )
    parser.add_argument(
        "--pipeline",
        default="eval,plot,summary",
        help="流水线步骤，逗号分隔"
    )
    parser.add_argument(
        "--signals-dir",
        help="信号文件目录"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    signals_dir = Path(args.signals_dir) if args.signals_dir else None
    
    orchestrator = PipelineOrchestrator(
        exp_id=args.exp_id,
        work_dir=Path(args.work_dir),
        signals_dir=signals_dir,
    )
    
    # 添加标准流水线
    orchestrator.add_standard_pipeline()
    
    success = orchestrator.run(timeout=args.timeout)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

