#!/usr/bin/env bash
set -e

echo "[$(date '+%F %T')] === Running baseline os-std (144) ==="
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5-mini \
  --concurrency 4 \
  -n 1 \
  -o results \
  os-std

echo "[$(date '+%F %T')] === Running os-std-extension1 (144) ==="
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5-mini \
  --concurrency 4 \
  -n 1 \
  -o results \
  os-std-extension1

echo "[$(date '+%F %T')] === Done ==="
