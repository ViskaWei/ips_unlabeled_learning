#!/usr/bin/env python3
"""
健康检查模块 (Health Check Module)
==================================

功能：
- 检测训练日志中的异常情况
- 支持多种检查规则：NaN、loss 爆炸、显存溢出、学习率异常等
- 可配置的检查策略

检查项：
1. NaN 检测 - loss/grad 中出现 nan
2. Loss 爆炸 - loss 突然变得很大
3. Loss 停滞 - loss 长时间不下降
4. 显存溢出 - OOM (Out of Memory) 错误
5. CUDA 错误 - GPU 相关错误
6. 学习率异常 - lr 变为 0 或异常值

用法：
    from health_check import HealthChecker
    
    checker = HealthChecker("logs/train.log")
    result = checker.check()
    
    if not result.healthy:
        print(f"问题: {result.reason}")

作者: Viska Wei
日期: 2025-12-04
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    healthy: bool = True
    reason: str = ""
    metrics: str = ""  # 最新的指标摘要
    details: dict = field(default_factory=dict)


@dataclass
class HealthCheckConfig:
    """健康检查配置"""
    # NaN 检测
    check_nan: bool = True
    
    # Loss 爆炸检测
    check_loss_explosion: bool = True
    loss_explosion_threshold: float = 1e10  # loss 超过此值认为爆炸
    loss_explosion_ratio: float = 100.0  # 单步 loss 增长超过 100 倍
    
    # Loss 停滞检测
    check_loss_stagnation: bool = False  # 默认关闭，因为可能误判
    loss_stagnation_steps: int = 100  # 连续 N 步 loss 不下降
    loss_stagnation_threshold: float = 0.001  # 下降幅度阈值
    
    # 显存检测
    check_oom: bool = True
    
    # CUDA 错误检测
    check_cuda_error: bool = True
    
    # 学习率检测
    check_lr_zero: bool = True
    
    # 读取日志的字节数（避免读取过大文件）
    max_log_bytes: int = 100_000  # 100KB


class HealthChecker:
    """训练健康检查器"""
    
    def __init__(
        self,
        log_file: Path | str,
        config: Optional[HealthCheckConfig] = None
    ):
        self.log_file = Path(log_file)
        self.config = config or HealthCheckConfig()
        
        # 编译正则表达式（提高效率）
        self._compile_patterns()
        
        # 历史 loss 值（用于检测爆炸/停滞）
        self.loss_history: list[float] = []
        
    def _compile_patterns(self):
        """编译检查用的正则表达式"""
        # NaN 检测模式（更严格，避免误判命令参数）
        self.nan_patterns = [
            # loss=nan 或 loss: nan（必须是值位置）
            re.compile(r'loss\s*[=:]\s*nan\b', re.IGNORECASE),
            # grad=nan 或 gradient nan
            re.compile(r'grad(?:ient)?\s*[=:]\s*nan\b', re.IGNORECASE),
            # NaN 作为独立输出（如 "tensor is NaN"）
            re.compile(r'(?:tensor|value|output).*\bnan\b', re.IGNORECASE),
            # "nan" 紧跟数字格式（如 nan, 0.123）
            re.compile(r'\bnan\s*[,\]]', re.IGNORECASE),
        ]
        
        # OOM 检测模式
        self.oom_patterns = [
            re.compile(r'out\s*of\s*memory', re.IGNORECASE),
            re.compile(r'CUDA\s*out\s*of\s*memory', re.IGNORECASE),
            re.compile(r'RuntimeError.*allocate', re.IGNORECASE),
            re.compile(r'OOM', re.IGNORECASE),
        ]
        
        # CUDA 错误检测模式
        self.cuda_error_patterns = [
            re.compile(r'CUDA\s*error', re.IGNORECASE),
            re.compile(r'cudnn\s*error', re.IGNORECASE),
            re.compile(r'NCCL\s*error', re.IGNORECASE),
            re.compile(r'device-side assert', re.IGNORECASE),
        ]
        
        # Loss 提取模式（支持多种格式）
        self.loss_patterns = [
            # loss=0.123 or loss: 0.123
            re.compile(r'loss\s*[=:]\s*([0-9.eE+-]+)', re.IGNORECASE),
            # train_loss=0.123
            re.compile(r'train_loss\s*[=:]\s*([0-9.eE+-]+)', re.IGNORECASE),
            # Loss: 0.123
            re.compile(r'Loss\s*[=:]\s*([0-9.eE+-]+)'),
        ]
        
        # 学习率提取模式
        self.lr_patterns = [
            re.compile(r'lr\s*[=:]\s*([0-9.eE+-]+)', re.IGNORECASE),
            re.compile(r'learning_rate\s*[=:]\s*([0-9.eE+-]+)', re.IGNORECASE),
        ]
        
        # Step/Epoch 提取模式
        self.step_patterns = [
            re.compile(r'step\s*[=:]\s*(\d+)', re.IGNORECASE),
            re.compile(r'epoch\s*[=:]\s*(\d+)', re.IGNORECASE),
            re.compile(r'iter\s*[=:]\s*(\d+)', re.IGNORECASE),
        ]
    
    def _read_log_tail(self) -> str:
        """读取日志文件的最后部分"""
        if not self.log_file.exists():
            return ""
        
        try:
            file_size = self.log_file.stat().st_size
            if file_size <= self.config.max_log_bytes:
                return self.log_file.read_text(errors='ignore')
            else:
                # 只读取最后部分
                with open(self.log_file, 'r', errors='ignore') as f:
                    f.seek(file_size - self.config.max_log_bytes)
                    f.readline()  # 跳过可能的不完整行
                    return f.read()
        except Exception as e:
            return f"Error reading log: {e}"
    
    def _extract_losses(self, text: str) -> list[float]:
        """从日志中提取 loss 值"""
        losses = []
        for pattern in self.loss_patterns:
            for match in pattern.finditer(text):
                try:
                    loss = float(match.group(1))
                    if not math.isnan(loss) and not math.isinf(loss):
                        losses.append(loss)
                except (ValueError, IndexError):
                    continue
        return losses
    
    def _extract_lr(self, text: str) -> Optional[float]:
        """从日志中提取学习率"""
        for pattern in self.lr_patterns:
            matches = list(pattern.finditer(text))
            if matches:
                try:
                    return float(matches[-1].group(1))
                except (ValueError, IndexError):
                    continue
        return None
    
    def _extract_step(self, text: str) -> Optional[int]:
        """从日志中提取当前步数"""
        for pattern in self.step_patterns:
            matches = list(pattern.finditer(text))
            if matches:
                try:
                    return int(matches[-1].group(1))
                except (ValueError, IndexError):
                    continue
        return None
    
    def check(self) -> HealthCheckResult:
        """执行健康检查"""
        result = HealthCheckResult()
        
        # 读取日志
        log_text = self._read_log_tail()
        if not log_text:
            # 日志文件还没有内容，认为正常
            return result
        
        # 1. NaN 检测
        if self.config.check_nan:
            for pattern in self.nan_patterns:
                if pattern.search(log_text):
                    result.healthy = False
                    result.reason = "检测到 NaN (loss 或 gradient 出现 nan)"
                    return result
        
        # 2. OOM 检测
        if self.config.check_oom:
            for pattern in self.oom_patterns:
                if pattern.search(log_text):
                    result.healthy = False
                    result.reason = "检测到显存溢出 (Out of Memory)"
                    return result
        
        # 3. CUDA 错误检测
        if self.config.check_cuda_error:
            for pattern in self.cuda_error_patterns:
                if pattern.search(log_text):
                    result.healthy = False
                    result.reason = "检测到 CUDA 错误"
                    return result
        
        # 4. Loss 爆炸检测
        if self.config.check_loss_explosion:
            losses = self._extract_losses(log_text)
            if losses:
                latest_loss = losses[-1]
                self.loss_history.extend(losses)
                
                # 检查绝对值
                if latest_loss > self.config.loss_explosion_threshold:
                    result.healthy = False
                    result.reason = f"Loss 爆炸: {latest_loss:.2e} > {self.config.loss_explosion_threshold:.2e}"
                    return result
                
                # 检查增长率
                if len(self.loss_history) >= 2:
                    prev_loss = self.loss_history[-2]
                    if prev_loss > 0 and latest_loss / prev_loss > self.config.loss_explosion_ratio:
                        result.healthy = False
                        result.reason = f"Loss 突增: {prev_loss:.4f} → {latest_loss:.4f} (增长 {latest_loss/prev_loss:.1f} 倍)"
                        return result
        
        # 5. 学习率检测
        if self.config.check_lr_zero:
            lr = self._extract_lr(log_text)
            if lr is not None and lr == 0:
                result.healthy = False
                result.reason = "学习率变为 0"
                return result
        
        # 6. Loss 停滞检测（可选）
        if self.config.check_loss_stagnation:
            if len(self.loss_history) >= self.config.loss_stagnation_steps:
                recent = self.loss_history[-self.config.loss_stagnation_steps:]
                if max(recent) - min(recent) < self.config.loss_stagnation_threshold:
                    result.healthy = False
                    result.reason = f"Loss 停滞: 连续 {self.config.loss_stagnation_steps} 步无明显变化"
                    return result
        
        # 构建指标摘要
        result.metrics = self._build_metrics_summary(log_text)
        
        return result
    
    def _build_metrics_summary(self, log_text: str) -> str:
        """构建指标摘要字符串"""
        parts = []
        
        # Step
        step = self._extract_step(log_text)
        if step is not None:
            parts.append(f"step={step}")
        
        # Loss
        losses = self._extract_losses(log_text)
        if losses:
            parts.append(f"loss={losses[-1]:.4f}")
        
        # LR
        lr = self._extract_lr(log_text)
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        
        return " | ".join(parts) if parts else ""
    
    def get_latest_metrics(self) -> str:
        """获取最新指标（用于进度显示）"""
        log_text = self._read_log_tail()
        if not log_text:
            return ""
        return self._build_metrics_summary(log_text)


# 便捷函数
def quick_health_check(log_file: str | Path) -> tuple[bool, str]:
    """
    快速健康检查
    
    返回: (is_healthy, reason)
    """
    checker = HealthChecker(log_file)
    result = checker.check()
    return result.healthy, result.reason


if __name__ == "__main__":
    # 测试用例
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python health_check.py <log_file>")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    checker = HealthChecker(log_file)
    result = checker.check()
    
    if result.healthy:
        print(f"✅ 健康检查通过")
        if result.metrics:
            print(f"   当前指标: {result.metrics}")
    else:
        print(f"❌ 健康检查失败: {result.reason}")
    
    sys.exit(0 if result.healthy else 1)

