#!/bin/bash
# 检查实验是否完成
# 用法: check_exp.sh [exp_id] [repo_path]

EXP_ID=${1:-""}
REPO_PATH=${2:-"$HOME/VIT"}

if [ -z "$EXP_ID" ]; then
    echo "用法: check_exp.sh <exp_id> [repo_path]"
    echo "示例: check_exp.sh VIT-20251204-cnn-01 ~/VIT"
    exit 1
fi

SIGNAL_FILE="$REPO_PATH/signals/${EXP_ID}.done"
LOG_FILE="$REPO_PATH/logs/${EXP_ID}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 检查实验: $EXP_ID"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "$SIGNAL_FILE" ]; then
    echo "✅ 训练已完成!"
    echo "   完成时间: $(cat $SIGNAL_FILE)"
    echo ""
    echo "📊 下一步："
    echo "   1. 生成图表"
    echo "   2. 撰写报告到 /home/swei20/Physics_Informed_AI/logg/"
    echo ""
    echo "💡 可以继续执行后续步骤了"
    exit 0
else
    echo "⏳ 训练进行中..."
    echo ""
    if [ -f "$LOG_FILE" ]; then
        echo "📄 最近日志 (tail -15):"
        echo "---"
        tail -15 "$LOG_FILE"
        echo "---"
    fi
    echo ""
    echo "💡 训练完成后会自动创建: $SIGNAL_FILE"
    exit 1
fi

