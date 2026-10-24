#!/usr/bin/env bash
#
# jetson_cleanup.sh — Run on the Jetson AFTER pulling results back locally.
# Removes the GAIA repo, compiled artifacts, and shell history traces.
#
# Usage: bash scripts/jetson_cleanup.sh [--repo-dir /path/to/GAIA]
#
set -euo pipefail

REPO_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "=== GAIA Jetson Cleanup ==="
echo "Target directory: $REPO_DIR"
echo ""

# Safety check
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "ERROR: $REPO_DIR is not a git repository. Aborting."
    exit 1
fi

read -rp "This will PERMANENTLY DELETE $REPO_DIR and clean shell history. Continue? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

# 1. Remove compiled binaries that might exist outside the repo
echo "[1/5] Removing any stray compiled binaries..."
find /tmp -maxdepth 2 -name "gpu_power_poc_v3*" -o -name "gpu_victim_kernel*" 2>/dev/null | xargs rm -f 2>/dev/null || true

# 2. Remove CUDA compilation cache
echo "[2/5] Clearing CUDA caches..."
rm -rf ~/.nv/ComputeCache 2>/dev/null || true

# 3. Remove pip packages installed for this project (optional)
echo "[3/5] Removing project pip packages..."
pip3 uninstall -y nvidia-ml-py 2>/dev/null || true

# 4. Remove the repo directory
echo "[4/5] Removing repository..."
rm -rf "$REPO_DIR"
echo "  Deleted: $REPO_DIR"

# 5. Scrub shell history
echo "[5/5] Cleaning shell history..."
# Remove lines referencing GAIA, gpu_power, gpu_victim, gpu_fault, gpu_defense
if [ -f ~/.bash_history ]; then
    sed -i '/GAIA\|gpu_power_poc\|gpu_victim_kernel\|gpu_fault_assessor\|gpu_defense_filters\|closed_loop_mitigator\|gaia_master_suite/Id' ~/.bash_history
fi
# Clear current session history
history -c 2>/dev/null || true

echo ""
echo "=== Cleanup complete ==="
echo "Verify with: ls $REPO_DIR (should show 'No such file or directory')"
echo "Start a new shell session to clear in-memory history."
