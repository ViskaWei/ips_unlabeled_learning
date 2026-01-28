#!/usr/bin/env python3
"""
训练驱动器 (Training Driver)
============================

功能：
1. 启动训练进程
2. 前 N 分钟进行健康检查（NaN、loss 爆炸、显存溢出等）
3. 健康检查通过后等待训练完成
4. 训练完成后自动触发下一步（eval、画图、生成 summary）

用法：
    python driver.py --config config.yaml --exp-id VIT-20251204-xxx
    python driver.py --cmd "python train.py --config config.yaml" --exp-id VIT-20251204-xxx
    
参数：
    --config: 配置文件路径（自动构建 train 命令）
    --cmd: 完整的训练命令
    --exp-id: 实验 ID
    --health-time: 健康检查时长（秒），默认 300（5分钟）
    --check-interval: 检查间隔（秒），默认 10
    --skip-post: 跳过后处理步骤
    --dry-run: 只显示会执行的命令，不实际运行

示例：
    # 使用配置文件
    python driver.py --config configs/exp/moe_nn.yaml --exp-id VIT-20251204-moe-nn-01
    
    # 使用完整命令
    python driver.py --cmd "python train_nn.py --config configs/nn.yaml" --exp-id VIT-20251204-nn-01
    
    # 自定义健康检查时间
    python driver.py --config config.yaml --exp-id xxx --health-time 600  # 10分钟

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

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from health_check import HealthChecker
from post_process import PostProcessor


class TrainingDriver:
    """训练驱动器：管理训练生命周期"""
    
    def __init__(
        self,
        exp_id: str,
        train_cmd: list[str],
        work_dir: Path,
        health_check_seconds: int = 300,
        check_interval: int = 10,
        skip_post_process: bool = False,
    ):
        self.exp_id = exp_id
        self.train_cmd = train_cmd
        self.work_dir = Path(work_dir)
        self.health_check_seconds = health_check_seconds
        self.check_interval = check_interval
        self.skip_post_process = skip_post_process
        
        # 创建必要目录
        self.logs_dir = self.work_dir / "logs"
        self.signals_dir = self.work_dir / "signals"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径
        self.log_file = self.logs_dir / f"{exp_id}.log"
        self.signal_file = self.signals_dir / f"{exp_id}.done"
        self.healthy_file = self.signals_dir / f"{exp_id}.healthy"
        self.failed_file = self.signals_dir / f"{exp_id}.failed"
        
        # 健康检查器
        self.health_checker = HealthChecker(self.log_file)
        
        # 后处理器
        self.post_processor = PostProcessor(exp_id, self.work_dir)
        
        # 进程句柄
        self.process = None
        self.start_time = None
        
    def _clean_signals(self):
        """清理旧的信号文件"""
        for f in [self.signal_file, self.healthy_file, self.failed_file]:
            if f.exists():
                f.unlink()
                
    def _log(self, msg: str, level: str = "INFO"):
        """带时间戳的日志输出"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "📝",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "RUNNING": "🚀",
        }.get(level, "")
        print(f"[{timestamp}] {prefix} {msg}")
        
    def start_training(self) -> subprocess.Popen:
        """启动训练进程"""
        self._clean_signals()
        self.start_time = time.time()
        
        self._log(f"启动训练: {self.exp_id}", "RUNNING")
        self._log(f"命令: {' '.join(self.train_cmd)}")
        self._log(f"日志: {self.log_file}")
        self._log(f"工作目录: {self.work_dir}")
        
        # 打开日志文件
        log_handle = open(self.log_file, "w")
        
        # 写入训练元信息
        log_handle.write(f"=== Training Started: {datetime.now().isoformat()} ===\n")
        log_handle.write(f"Experiment ID: {self.exp_id}\n")
        log_handle.write(f"Command: {' '.join(self.train_cmd)}\n")
        log_handle.write(f"Working Directory: {self.work_dir}\n")
        log_handle.write("=" * 60 + "\n\n")
        log_handle.flush()
        
        # 启动训练进程
        self.process = subprocess.Popen(
            self.train_cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=self.work_dir,
        )
        
        self._log(f"训练进程启动 (PID: {self.process.pid})", "SUCCESS")
        return self.process
    
    def run_health_check(self) -> bool:
        """运行健康检查（前 N 分钟）"""
        self._log(f"开始健康检查 ({self.health_check_seconds}秒)", "RUNNING")
        
        start_time = time.time()
        healthy = True
        last_status = ""
        
        while time.time() - start_time < self.health_check_seconds:
            # 检查进程是否存活
            if self.process.poll() is not None:
                exit_code = self.process.returncode
                if exit_code == 0:
                    self._log("训练提前完成（正常退出）", "SUCCESS")
                    self._mark_healthy()
                    return True
                else:
                    self._log(f"训练进程异常退出 (exit code: {exit_code})", "ERROR")
                    self._mark_failed(f"Early exit with code {exit_code}")
                    return False
            
            # 执行健康检查
            check_result = self.health_checker.check()
            
            if not check_result.healthy:
                self._log(f"健康检查失败: {check_result.reason}", "ERROR")
                self._log("正在终止训练进程...", "WARNING")
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                self._mark_failed(check_result.reason)
                return False
            
            # 显示进度（只在状态变化时打印）
            elapsed = int(time.time() - start_time)
            remaining = self.health_check_seconds - elapsed
            status = f"健康检查中... {elapsed}s/{self.health_check_seconds}s (剩余 {remaining}s)"
            if check_result.metrics:
                status += f" | {check_result.metrics}"
            
            if status != last_status:
                print(f"\r⏳ {status}", end="", flush=True)
                last_status = status
            
            time.sleep(self.check_interval)
        
        print()  # 换行
        self._log("健康检查通过！", "SUCCESS")
        self._mark_healthy()
        return True
    
    def wait_for_completion(self) -> int:
        """等待训练完成"""
        self._log("等待训练完成...", "RUNNING")
        
        while self.process.poll() is None:
            # 定期显示进度
            elapsed = int(time.time() - self.start_time)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # 读取最新的 loss（如果有）
            latest_info = self.health_checker.get_latest_metrics()
            status = f"训练中... {hours:02d}:{minutes:02d}:{seconds:02d}"
            if latest_info:
                status += f" | {latest_info}"
            
            print(f"\r🚀 {status}", end="", flush=True)
            time.sleep(60)  # 每分钟更新一次
        
        print()  # 换行
        return self.process.returncode
    
    def run_post_process(self):
        """运行后处理步骤"""
        if self.skip_post_process:
            self._log("跳过后处理步骤", "WARNING")
            return
        
        self._log("开始后处理...", "RUNNING")
        self.post_processor.run()
        self._log("后处理完成", "SUCCESS")
    
    def _mark_healthy(self):
        """标记健康检查通过"""
        self.healthy_file.write_text(
            f"healthy\n"
            f"timestamp: {datetime.now().isoformat()}\n"
            f"elapsed: {int(time.time() - self.start_time)}s\n"
        )
        
    def _mark_failed(self, reason: str):
        """标记失败"""
        self.failed_file.write_text(
            f"failed\n"
            f"reason: {reason}\n"
            f"timestamp: {datetime.now().isoformat()}\n"
            f"elapsed: {int(time.time() - self.start_time)}s\n"
        )
        
    def _mark_done(self, success: bool, return_code: int):
        """标记训练完成"""
        duration = int(time.time() - self.start_time)
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        status = "done" if success else "failed"
        self.signal_file.write_text(
            f"{status}\n"
            f"return_code: {return_code}\n"
            f"duration: {hours:02d}:{minutes:02d}:{seconds:02d}\n"
            f"timestamp: {datetime.now().isoformat()}\n"
        )
    
    def run(self) -> bool:
        """运行完整训练流程"""
        print()
        print("━" * 60)
        print(f"🎯 训练驱动器 - {self.exp_id}")
        print("━" * 60)
        print()
        
        try:
            # 1. 启动训练
            self.start_training()
            print()
            
            # 2. 健康检查
            if not self.run_health_check():
                self._log("由于健康检查失败，训练已停止", "ERROR")
                self._print_failure_summary()
                return False
            print()
            
            # 3. 等待完成
            return_code = self.wait_for_completion()
            print()
            
            # 4. 检查结果
            success = return_code == 0
            self._mark_done(success, return_code)
            
            if success:
                self._log(f"训练成功完成！", "SUCCESS")
                self._print_success_summary()
                
                # 5. 后处理
                self.run_post_process()
            else:
                self._log(f"训练失败 (exit code: {return_code})", "ERROR")
                self._print_failure_summary()
            
            return success
            
        except KeyboardInterrupt:
            self._log("收到中断信号，正在停止训练...", "WARNING")
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self._mark_failed("User interrupted")
            return False
        except Exception as e:
            self._log(f"发生错误: {e}", "ERROR")
            if self.process and self.process.poll() is None:
                self.process.terminate()
            self._mark_failed(str(e))
            raise
    
    def _print_success_summary(self):
        """打印成功摘要"""
        duration = int(time.time() - self.start_time)
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print()
        print("╔" + "═" * 58 + "╗")
        print("║" + " " * 20 + "✅ 训练完成！" + " " * 20 + "║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  实验 ID:  {self.exp_id:<45}║")
        print(f"║  耗时:     {hours:02d}:{minutes:02d}:{seconds:02d}" + " " * 40 + "║")
        print(f"║  日志:     {str(self.log_file)[-43:]:<45}║")
        print("╠" + "═" * 58 + "╣")
        print("║  📊 下一步:                                             ║")
        print("║    1. 查看结果: results/{exp_id}/                       ║")
        print("║    2. 归档报告: a {exp_id}                              ║")
        print("╚" + "═" * 58 + "╝")
        print()
        
    def _print_failure_summary(self):
        """打印失败摘要，包含调试信息供 Agent 继续修复"""
        # 读取失败原因
        fail_reason = "unknown"
        if self.failed_file.exists():
            content = self.failed_file.read_text()
            for line in content.split("\n"):
                if line.startswith("reason:"):
                    fail_reason = line.split(":", 1)[1].strip()
                    break
        
        # 读取日志最后几行
        log_tail = ""
        if self.log_file.exists():
            try:
                lines = self.log_file.read_text().strip().split("\n")
                log_tail = "\n".join(lines[-20:])
            except:
                pass
        
        print()
        print("╔" + "═" * 70 + "╗")
        print("║" + " " * 25 + "❌ 训练失败" + " " * 26 + "║")
        print("╠" + "═" * 70 + "╣")
        print(f"║  实验 ID:  {self.exp_id:<57}║")
        print(f"║  失败原因: {fail_reason[:55]:<57}║")
        print("╠" + "═" * 70 + "╣")
        print("║  📋 调试信息:                                                        ║")
        print(f"║    日志文件: {str(self.log_file):<55}║")
        print(f"║    信号文件: {str(self.failed_file):<55}║")
        print("╚" + "═" * 70 + "╝")
        print()
        
        # 输出日志尾部供 Agent 分析
        print("━" * 72)
        print("📄 日志最后 20 行 (供 Agent 分析):")
        print("━" * 72)
        if log_tail:
            print(log_tail)
        else:
            print("(日志为空或无法读取)")
        print("━" * 72)
        print()
        
        # 根据失败原因给出修复建议
        print("💡 修复建议:")
        print("━" * 72)
        
        if "nan" in fail_reason.lower():
            print("""
🔍 NaN 检测 - 可能原因:
   1. 学习率过高 → 尝试降低 lr (如 1e-4 → 1e-5)
   2. 梯度爆炸 → 添加 gradient clipping
   3. 数据问题 → 检查输入数据是否有 inf/nan
   4. Loss 函数问题 → 检查 loss 计算逻辑

📝 建议的修复命令:
   # 降低学习率重试
   python driver.py --cmd "python train.py --lr 1e-5" --exp-id {exp_id}-fix1
   
   # 添加梯度裁剪
   在配置中添加: grad_clip: 1.0
""".format(exp_id=self.exp_id))
        
        elif "oom" in fail_reason.lower() or "memory" in fail_reason.lower():
            print("""
🔍 显存溢出 (OOM) - 可能原因:
   1. Batch size 过大 → 减小 batch_size
   2. 模型过大 → 减少层数/隐藏维度
   3. 序列过长 → 减少 max_length

📝 建议的修复命令:
   # 减小 batch size
   python driver.py --cmd "python train.py --batch-size 16" --exp-id {exp_id}-fix1
   
   # 使用 gradient accumulation
   python driver.py --cmd "python train.py --batch-size 8 --grad-accum 4" --exp-id {exp_id}-fix1
""".format(exp_id=self.exp_id))
        
        elif "cuda" in fail_reason.lower():
            print("""
🔍 CUDA 错误 - 可能原因:
   1. GPU 驱动问题 → 重启或检查 nvidia-smi
   2. 数据类型不匹配 → 检查 tensor dtype
   3. 设备不一致 → 确保所有 tensor 在同一设备

📝 调试命令:
   nvidia-smi  # 检查 GPU 状态
   python -c "import torch; print(torch.cuda.is_available())"
""")
        
        elif "loss" in fail_reason.lower() and "explo" in fail_reason.lower():
            print("""
🔍 Loss 爆炸 - 可能原因:
   1. 学习率过高 → 降低 lr
   2. 权重初始化问题 → 检查初始化方法
   3. 数据未归一化 → 添加 normalization

📝 建议的修复:
   # 使用学习率预热
   在配置中添加: warmup_steps: 100
""")
        
        else:
            print(f"""
🔍 未知错误: {fail_reason}

📝 通用调试步骤:
   1. 查看完整日志: cat {self.log_file}
   2. 检查配置文件
   3. 尝试更小的数据集/模型进行验证
""")
        
        print("━" * 72)
        print()
        print("🤖 Agent 下一步:")
        print("   请根据以上信息分析失败原因，修改配置后重新运行实验。")
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练驱动器 - 健康检查 + 训练 + 后处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置文件
  python driver.py --config configs/exp/moe.yaml --exp-id VIT-20251204-moe-01

  # 使用完整命令
  python driver.py --cmd "python train_nn.py --config configs/nn.yaml" \\
                   --exp-id VIT-20251204-nn-01

  # 自定义健康检查时间（10分钟）
  python driver.py --config config.yaml --exp-id xxx --health-time 600
        """
    )
    
    # 训练命令（二选一）
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument(
        "--config", "-c",
        help="配置文件路径（自动构建 python scripts/run.py -f CONFIG 命令）"
    )
    cmd_group.add_argument(
        "--cmd",
        help="完整的训练命令（用引号包裹）"
    )
    
    # 必需参数
    parser.add_argument(
        "--exp-id", "-e",
        required=True,
        help="实验 ID，如 VIT-20251204-moe-01"
    )
    
    # 可选参数
    parser.add_argument(
        "--work-dir", "-w",
        default=os.getcwd(),
        help="工作目录（默认当前目录）"
    )
    parser.add_argument(
        "--health-time",
        type=int,
        default=300,
        help="健康检查时长（秒），默认 300（5分钟）"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=10,
        help="健康检查间隔（秒），默认 10"
    )
    parser.add_argument(
        "--skip-post",
        action="store_true",
        help="跳过后处理步骤"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示会执行的命令，不实际运行"
    )
    
    return parser.parse_args()


def build_train_cmd(args) -> list[str]:
    """构建训练命令"""
    if args.cmd:
        # 使用用户提供的完整命令
        import shlex
        return shlex.split(args.cmd)
    else:
        # 从配置文件构建命令
        return ["python", "scripts/run.py", "-f", args.config]


def main():
    args = parse_args()
    
    # 构建训练命令
    train_cmd = build_train_cmd(args)
    
    # Dry run 模式
    if args.dry_run:
        print("🔍 Dry Run 模式 - 不会实际执行")
        print()
        print(f"实验 ID:     {args.exp_id}")
        print(f"训练命令:   {' '.join(train_cmd)}")
        print(f"工作目录:   {args.work_dir}")
        print(f"健康检查:   {args.health_time}秒")
        print(f"检查间隔:   {args.check_interval}秒")
        print(f"后处理:     {'跳过' if args.skip_post else '启用'}")
        return 0
    
    # 创建驱动器并运行
    driver = TrainingDriver(
        exp_id=args.exp_id,
        train_cmd=train_cmd,
        work_dir=Path(args.work_dir),
        health_check_seconds=args.health_time,
        check_interval=args.check_interval,
        skip_post_process=args.skip_post,
    )
    
    success = driver.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

