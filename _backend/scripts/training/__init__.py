"""
训练自动化模块 (Training Automation Module)
============================================

提供训练生命周期管理的工具：
- 健康检查
- 信号文件状态机
- 训练钩子
- 后处理自动化

主要组件：
- driver.py: 训练驱动器（健康检查 + 完整训练 + 后处理）
- health_check.py: 健康检查模块
- orchestrator.py: 多步骤流水线编排器
- train_hooks.py: 训练脚本钩子
- post_process.py: 后处理自动化

用法：
    # 方式 1: 使用驱动器（推荐）
    python -m training.driver --config config.yaml --exp-id VIT-20251204-xxx
    
    # 方式 2: 在训练脚本中使用钩子
    from training.train_hooks import TrainingHooks
    hooks = TrainingHooks("VIT-20251204-xxx")
    hooks.mark_healthy(step=100, loss=0.5)
    hooks.mark_done(metrics={"r2": 0.99})
    
    # 方式 3: 使用编排器管理多步骤流水线
    python -m training.orchestrator --exp-id VIT-20251204-xxx
"""

from .health_check import HealthChecker, HealthCheckResult, HealthCheckConfig
from .train_hooks import TrainingHooks
from .post_process import PostProcessor

__all__ = [
    "HealthChecker",
    "HealthCheckResult", 
    "HealthCheckConfig",
    "TrainingHooks",
    "PostProcessor",
]

