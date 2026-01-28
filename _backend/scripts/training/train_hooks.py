#!/usr/bin/env python3
"""
训练钩子模块 (Training Hooks)
==============================

在训练脚本中使用这些钩子来标记训练状态，
让外部编排器/驱动器知道训练进度。

信号文件约定：
- {exp_id}.healthy - 健康检查通过（在 warmup 后调用）
- {exp_id}.done    - 训练完成
- {exp_id}.failed  - 训练失败

用法（在你的 train.py 中）：

    from train_hooks import TrainingHooks
    
    hooks = TrainingHooks("VIT-20251204-xxx", signals_dir="./signals")
    
    # 训练循环
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            # ... 训练代码 ...
            
            # 在 warmup 后标记健康
            if epoch == 0 and step == warmup_steps:
                if loss < threshold:
                    hooks.mark_healthy(step=step, loss=loss)
                else:
                    hooks.mark_failed(f"Loss too high: {loss}")
                    return
    
    # 训练结束
    hooks.mark_done(metrics={"r2": 0.99, "mae": 0.05})

PyTorch Lightning 集成：

    class MyLightningModule(pl.LightningModule):
        def __init__(self, hooks: TrainingHooks):
            self.hooks = hooks
            self.warmup_done = False
        
        def training_step(self, batch, batch_idx):
            # ... 训练代码 ...
            
            # 检查健康
            if not self.warmup_done and self.global_step >= 100:
                if loss < 10.0:
                    self.hooks.mark_healthy(step=self.global_step, loss=loss.item())
                    self.warmup_done = True
            
            return loss
        
        def on_train_end(self):
            self.hooks.mark_done()

作者: Viska Wei
日期: 2025-12-04
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
import json


class TrainingHooks:
    """训练钩子 - 用于标记训练状态"""
    
    def __init__(
        self,
        exp_id: str,
        signals_dir: str | Path = "./signals",
        auto_env: bool = True,
    ):
        """
        初始化训练钩子
        
        Args:
            exp_id: 实验 ID
            signals_dir: 信号文件目录
            auto_env: 是否从环境变量读取 exp_id（如果未指定）
        """
        # 尝试从环境变量获取 exp_id
        if auto_env and not exp_id:
            exp_id = os.environ.get("EXP_ID", "unknown")
        
        self.exp_id = exp_id
        self.signals_dir = Path(signals_dir)
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        
        # 状态标志
        self._healthy_marked = False
        self._done_marked = False
    
    def _get_signal_path(self, signal_type: str) -> Path:
        """获取信号文件路径"""
        return self.signals_dir / f"{self.exp_id}.{signal_type}"
    
    def _write_signal(self, signal_type: str, content: dict):
        """写入信号文件"""
        content["timestamp"] = datetime.now().isoformat()
        content["exp_id"] = self.exp_id
        
        signal_path = self._get_signal_path(signal_type)
        
        # 写入简单格式（便于 shell 读取）
        with open(signal_path, "w") as f:
            for key, value in content.items():
                f.write(f"{key}: {value}\n")
        
        # 同时写入 JSON 格式（便于程序读取）
        json_path = signal_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(content, f, indent=2)
    
    def mark_healthy(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        **extra_info
    ):
        """
        标记健康检查通过
        
        通常在 warmup 后、loss 稳定时调用。
        
        Args:
            step: 当前步数
            epoch: 当前 epoch
            loss: 当前 loss 值
            **extra_info: 额外信息
        """
        if self._healthy_marked:
            return  # 避免重复标记
        
        content = {"status": "healthy"}
        if step is not None:
            content["step"] = step
        if epoch is not None:
            content["epoch"] = epoch
        if loss is not None:
            content["loss"] = f"{loss:.6f}"
        content.update(extra_info)
        
        self._write_signal("healthy", content)
        self._healthy_marked = True
        
        print(f"[TrainingHooks] ✅ Marked healthy at step={step}, loss={loss}")
    
    def mark_done(
        self,
        success: bool = True,
        metrics: Optional[dict] = None,
        **extra_info
    ):
        """
        标记训练完成
        
        Args:
            success: 是否成功完成
            metrics: 最终指标字典
            **extra_info: 额外信息
        """
        if self._done_marked:
            return
        
        content = {
            "status": "done" if success else "failed",
        }
        
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    content[f"metric_{key}"] = f"{value:.6f}"
                else:
                    content[f"metric_{key}"] = str(value)
        
        content.update(extra_info)
        
        self._write_signal("done", content)
        self._done_marked = True
        
        status = "✅ done" if success else "❌ failed"
        print(f"[TrainingHooks] {status}")
        if metrics:
            print(f"[TrainingHooks] Final metrics: {metrics}")
    
    def mark_failed(self, reason: str, **extra_info):
        """
        标记训练失败
        
        Args:
            reason: 失败原因
            **extra_info: 额外信息
        """
        content = {
            "status": "failed",
            "reason": reason,
        }
        content.update(extra_info)
        
        self._write_signal("failed", content)
        print(f"[TrainingHooks] ❌ Failed: {reason}")
    
    def log_metrics(
        self,
        step: int,
        metrics: dict,
        log_file: Optional[str | Path] = None
    ):
        """
        记录训练指标到 CSV（可选功能）
        
        Args:
            step: 当前步数
            metrics: 指标字典
            log_file: 日志文件路径
        """
        if log_file is None:
            log_file = self.signals_dir.parent / "logs" / f"{self.exp_id}_metrics.csv"
        
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入 CSV
        write_header = not log_file.exists()
        
        with open(log_file, "a") as f:
            if write_header:
                headers = ["step"] + list(metrics.keys())
                f.write(",".join(headers) + "\n")
            
            values = [str(step)] + [
                f"{v:.6f}" if isinstance(v, float) else str(v)
                for v in metrics.values()
            ]
            f.write(",".join(values) + "\n")
    
    @property
    def is_healthy(self) -> bool:
        """检查是否已标记为健康"""
        return self._healthy_marked or self._get_signal_path("healthy").exists()
    
    @property
    def is_done(self) -> bool:
        """检查是否已完成"""
        return self._done_marked or self._get_signal_path("done").exists()


# ============================================================
# PyTorch Lightning 回调
# ============================================================

try:
    import pytorch_lightning as pl
    
    class TrainingHooksCallback(pl.Callback):
        """
        PyTorch Lightning 回调 - 自动管理训练钩子
        
        用法：
            hooks = TrainingHooks("VIT-20251204-xxx")
            trainer = pl.Trainer(
                callbacks=[TrainingHooksCallback(hooks, warmup_steps=100)]
            )
        """
        
        def __init__(
            self,
            hooks: TrainingHooks,
            warmup_steps: int = 100,
            health_loss_threshold: float = 100.0,
        ):
            super().__init__()
            self.hooks = hooks
            self.warmup_steps = warmup_steps
            self.health_loss_threshold = health_loss_threshold
            self._health_checked = False
        
        def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
        ):
            # 健康检查
            if not self._health_checked and trainer.global_step >= self.warmup_steps:
                # 获取当前 loss
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
                if hasattr(loss, "item"):
                    loss = loss.item()
                
                if loss < self.health_loss_threshold:
                    self.hooks.mark_healthy(
                        step=trainer.global_step,
                        epoch=trainer.current_epoch,
                        loss=loss
                    )
                else:
                    self.hooks.mark_failed(
                        f"Loss too high after warmup: {loss} > {self.health_loss_threshold}"
                    )
                
                self._health_checked = True
        
        def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
            # 收集最终指标
            metrics = {}
            if hasattr(trainer, "callback_metrics"):
                for key, value in trainer.callback_metrics.items():
                    if hasattr(value, "item"):
                        metrics[key] = value.item()
                    else:
                        metrics[key] = float(value)
            
            self.hooks.mark_done(success=True, metrics=metrics)
        
        def on_exception(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            exception: BaseException
        ):
            self.hooks.mark_failed(f"Exception: {type(exception).__name__}: {exception}")

except ImportError:
    # PyTorch Lightning 未安装
    pass


# ============================================================
# 便捷函数
# ============================================================

def get_hooks_from_env() -> TrainingHooks:
    """从环境变量创建 TrainingHooks"""
    exp_id = os.environ.get("EXP_ID", "unknown")
    signals_dir = os.environ.get("SIGNALS_DIR", "./signals")
    return TrainingHooks(exp_id, signals_dir)


# ============================================================
# 示例代码
# ============================================================

if __name__ == "__main__":
    # 示例：模拟训练过程
    import time
    import random
    
    hooks = TrainingHooks("TEST-example-001")
    
    print("模拟训练过程...")
    
    # 模拟 warmup
    for step in range(100):
        loss = 10.0 - step * 0.08 + random.random() * 0.5
        
        if step == 50:  # warmup 后标记健康
            hooks.mark_healthy(step=step, loss=loss)
        
        if step % 20 == 0:
            print(f"  Step {step}, loss={loss:.4f}")
            hooks.log_metrics(step, {"loss": loss, "lr": 0.001})
        
        time.sleep(0.01)
    
    # 标记完成
    hooks.mark_done(metrics={"final_loss": loss, "r2": 0.95})
    
    print("\n检查信号文件:")
    for f in hooks.signals_dir.glob(f"{hooks.exp_id}.*"):
        print(f"  {f.name}:")
        print(f"    {f.read_text()}")

