#!/bin/bash
#
# Sync Experiments Script
# =======================
# 
# Scan VIT and BlindSpot repositories and sync to experiment index.
#
# Usage:
#   ./sync_experiments.sh           # Full sync
#   ./sync_experiments.sh --dry-run # Preview only
#   ./sync_experiments.sh --since "2025-11-28"  # Since date
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHYSICS_AI_ROOT="$(dirname "$SCRIPT_DIR")"

VIT_ROOT="${VIT_ROOT:-$HOME/VIT}"
BLINDSPOT_ROOT="${BLINDSPOT_ROOT:-$HOME/BlindSpotDenoiser}"

# Parse arguments
DRY_RUN=""
SINCE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-n)
            DRY_RUN="--dry-run"
            shift
            ;;
        --since|-s)
            SINCE="--since $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🔄 Syncing experiments to Physics_Informed_AI"
echo "   VIT_ROOT: $VIT_ROOT"
echo "   BLINDSPOT_ROOT: $BLINDSPOT_ROOT"
echo ""

# Scan VIT
if [ -d "$VIT_ROOT" ]; then
    echo "📁 Scanning VIT repository..."
    python "$SCRIPT_DIR/scan_vit_experiments.py" \
        --vit-root "$VIT_ROOT" \
        $SINCE $DRY_RUN
    echo ""
else
    echo "⚠️  VIT repository not found at $VIT_ROOT"
fi

# Scan BlindSpot
if [ -d "$BLINDSPOT_ROOT" ]; then
    echo "📁 Scanning BlindSpotDenoiser repository..."
    python "$SCRIPT_DIR/scan_blindspot_experiments.py" \
        --blindspot-root "$BLINDSPOT_ROOT" \
        $SINCE $DRY_RUN
    echo ""
else
    echo "⚠️  BlindSpotDenoiser repository not found at $BLINDSPOT_ROOT"
fi

echo "✅ Sync complete!"
echo ""
echo "📊 Index location: $PHYSICS_AI_ROOT/experiments_index/index.csv"
echo "📝 Next steps: $PHYSICS_AI_ROOT/status/next_steps.md"
