#!/bin/bash
# ============================================================
# 训练启动器 (Training Launcher)
# ============================================================
# 便捷脚本，封装 driver.py 的常用参数
#
# 用法:
#   ./train.sh <exp_id> <config_or_cmd>
#   ./train.sh VIT-20251204-moe-01 configs/exp/moe.yaml
#   ./train.sh VIT-20251204-nn-01 "python train_nn.py --config config.yaml"
#
# 环境变量:
#   HEALTH_TIME - 健康检查时长（秒），默认 300
#   WORK_DIR    - 工作目录，默认当前目录
#   SKIP_POST   - 设为 1 跳过后处理
# ============================================================

set -e

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_PY="$SCRIPT_DIR/driver.py"

# 参数检查
if [ $# -lt 2 ]; then
    echo "用法: $0 <exp_id> <config_or_cmd>"
    echo ""
    echo "示例:"
    echo "  $0 VIT-20251204-moe-01 configs/exp/moe.yaml"
    echo "  $0 VIT-20251204-nn-01 \"python train_nn.py --config config.yaml\""
    echo ""
    echo "环境变量:"
    echo "  HEALTH_TIME - 健康检查时长（秒），默认 300"
    echo "  WORK_DIR    - 工作目录，默认当前目录"
    echo "  SKIP_POST   - 设为 1 跳过后处理"
    exit 1
fi

EXP_ID="$1"
CONFIG_OR_CMD="$2"

# 默认值
HEALTH_TIME="${HEALTH_TIME:-300}"
WORK_DIR="${WORK_DIR:-$(pwd)}"
SKIP_POST="${SKIP_POST:-0}"

# 构建参数
ARGS=(
    --exp-id "$EXP_ID"
    --work-dir "$WORK_DIR"
    --health-time "$HEALTH_TIME"
)

# 判断是配置文件还是命令
if [[ "$CONFIG_OR_CMD" == *.yaml ]] || [[ "$CONFIG_OR_CMD" == *.yml ]]; then
    ARGS+=(--config "$CONFIG_OR_CMD")
else
    ARGS+=(--cmd "$CONFIG_OR_CMD")
fi

# 是否跳过后处理
if [ "$SKIP_POST" = "1" ]; then
    ARGS+=(--skip-post)
fi

# 打印信息
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 训练启动器"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "实验 ID:     $EXP_ID"
echo "配置/命令:  $CONFIG_OR_CMD"
echo "工作目录:   $WORK_DIR"
echo "健康检查:   ${HEALTH_TIME}秒"
echo "后处理:     $([ "$SKIP_POST" = "1" ] && echo "跳过" || echo "启用")"
echo ""

# 执行
exec python "$DRIVER_PY" "${ARGS[@]}"

