#!/usr/bin/env bash
# Run os-std-combo: OSMonitor (monitor-replan) + CycleMonitor (extension1) at the same time.
#
# Usage:
#   ./run_combo.sh           # full 144-sample run
#   ./run_combo.sh smoke     # 20-sample smoke test (indices 0-19)

set -e

MODE="${1:-full}"

if [[ "$MODE" == "smoke" ]]; then
  echo "[$(date '+%F %T')] === Running os-std-combo (smoke, 20 samples) ==="
  agentrl-eval \
    --no-interactive \
    -c http://localhost:5020/api \
    -u https://api.openai.com/v1 \
    -m gpt-5-mini \
    --concurrency 4 \
    -n 1 \
    --indices-range 0-19 \
    -o results \
    os-std-combo
else
  echo "[$(date '+%F %T')] === Running os-std-combo (full 144) ==="
  agentrl-eval \
    --no-interactive \
    -c http://localhost:5020/api \
    -u https://api.openai.com/v1 \
    -m gpt-5-mini \
    --concurrency 4 \
    -n 1 \
    -o results \
    os-std-combo
fi

echo "[$(date '+%F %T')] === Done ==="
