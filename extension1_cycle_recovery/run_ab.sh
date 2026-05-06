#!/usr/bin/env bash
set -e

agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5-mini \
  --indices-range "0-39" \
  --concurrency 1 \
  -n 1 \
  -o results \
  os-std

agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5-mini \
  --indices-range "0-39" \
  --concurrency 1 \
  -n 1 \
  -o results \
  os-std-extension1
