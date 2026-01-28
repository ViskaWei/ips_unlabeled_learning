#!/bin/bash
# 监控实验完成状态（可选：完成后自动复制 check 命令）
# 用法: watch_exp.sh <exp_id> [repo_path]
# 在另一个终端运行，完成后会提醒你

EXP_ID=${1:-""}
REPO_PATH=${2:-"$PWD"}
CHECK_INTERVAL=${3:-300}  # 检查间隔（秒），默认 5 分钟

if [ -z "$EXP_ID" ]; then
    echo "用法: watch_exp.sh <exp_id> [repo_path] [interval]"
    exit 1
fi

SIGNAL_FILE="$REPO_PATH/signals/${EXP_ID}.done"
LOG_FILE="$REPO_PATH/logs/${EXP_ID}.log"

echo "👀 监控实验: $EXP_ID"
echo "   信号文件: $SIGNAL_FILE"
echo "   检查间隔: $((CHECK_INTERVAL/60)) 分钟"
echo "   按 Ctrl+C 退出"
echo ""

while true; do
    if [ -f "$SIGNAL_FILE" ]; then
        # 完成通知
        echo -e '\a\a\a'  # 响铃
        clear
        echo ""
        echo "╔═══════════════════════════════════════════════════════════════╗"
        echo "║                                                               ║"
        echo "║   ✅✅✅  训练完成！✅✅✅                                    ║"
        echo "║                                                               ║"
        echo "║   实验 ID: $EXP_ID"
        echo "║   完成时间: $(cat $SIGNAL_FILE)"
        echo "║                                                               ║"
        echo "║   👉 在 Cursor 中输入:                                        ║"
        echo "║                                                               ║"
        echo "║      check $EXP_ID                                            ║"
        echo "║                                                               ║"
        echo "╚═══════════════════════════════════════════════════════════════╝"
        echo ""
        
        # 桌面通知
        which notify-send >/dev/null 2>&1 && \
            notify-send -u critical "✅ 训练完成" "$EXP_ID - 输入 check 继续"
        
        # macOS 通知
        which osascript >/dev/null 2>&1 && \
            osascript -e "display notification \"$EXP_ID 训练完成\" with title \"✅ 实验完成\" sound name \"Glass\""
        
        exit 0
    fi
    
    # 显示进度
    printf "\r⏳ 等待中... [%s] " "$(date +%H:%M:%S)"
    
    # 如果有日志，显示最后一行
    if [ -f "$LOG_FILE" ]; then
        LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null | cut -c1-50)
        printf "| %s" "$LAST_LINE"
    fi
    
    sleep $CHECK_INTERVAL
done

