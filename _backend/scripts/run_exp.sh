#!/bin/bash
# 实验执行器：后台运行 + 完成后通知
# 用法: run_exp.sh <exp_id> <训练命令>
# 示例: run_exp.sh VIT-20251204-moe-01 "python train.py --config config.yaml"

set -e

EXP_ID=${1:-"exp"}
shift
TRAIN_CMD="$@"

if [ -z "$TRAIN_CMD" ]; then
    echo "用法: run_exp.sh <exp_id> <训练命令>"
    echo "示例: run_exp.sh VIT-20251204-moe-01 python train.py --config config.yaml"
    exit 1
fi

# 创建目录
mkdir -p signals logs

LOG_FILE="logs/${EXP_ID}.log"
SIGNAL_FILE="signals/${EXP_ID}.done"
START_TIME=$(date +%s)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动实验: $EXP_ID"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 命令: $TRAIN_CMD"
echo "📄 日志: $LOG_FILE"
echo "🔔 信号: $SIGNAL_FILE"
echo ""

# 后台执行训练
nohup bash -c "
    echo '=== 训练开始: $(date) ===' 
    $TRAIN_CMD
    EXIT_CODE=\$?
    END_TIME=\$(date +%s)
    DURATION=\$(( END_TIME - $START_TIME ))
    
    if [ \$EXIT_CODE -eq 0 ]; then
        echo 'done' > '$SIGNAL_FILE'
        echo \"duration: \${DURATION}s\" >> '$SIGNAL_FILE'
        
        # ═══════════════════════════════════════
        # 完成通知（多种方式）
        # ═══════════════════════════════════════
        
        # 1. 终端响铃
        echo -e '\a\a\a'
        
        # 2. 醒目打印
        echo ''
        echo '╔═══════════════════════════════════════════════════════════╗'
        echo '║  ✅ 训练完成！                                            ║'
        echo '║  实验ID: $EXP_ID                                          ║'
        echo '║  👉 输入: check $EXP_ID                                   ║'
        echo '╚═══════════════════════════════════════════════════════════╝'
        echo ''
        
        # 3. 桌面通知（如果有图形界面）
        which notify-send >/dev/null 2>&1 && notify-send '✅ 训练完成' '$EXP_ID 已完成，输入 check 继续'
        
        # 4. 写入提示文件（方便 Agent 发现）
        echo '✅ $EXP_ID 训练完成，请输入 check $EXP_ID 继续' > /tmp/exp_notification.txt
    else
        echo 'failed' > '$SIGNAL_FILE'
        echo \"exit_code: \$EXIT_CODE\" >> '$SIGNAL_FILE'
        echo ''
        echo '╔═══════════════════════════════════════════════════════════╗'
        echo '║  ❌ 训练失败！                                            ║'
        echo '║  查看日志: tail -50 $LOG_FILE                             ║'
        echo '╚═══════════════════════════════════════════════════════════╝'
    fi
" > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ 训练已在后台启动 (PID: $PID)"
echo ""
echo "📊 监控命令:"
echo "   tail -f $LOG_FILE        # 实时日志"
echo "   check_exp.sh $EXP_ID     # 检查状态"
echo ""
echo "💡 训练完成后会自动通知，届时输入: check $EXP_ID"

