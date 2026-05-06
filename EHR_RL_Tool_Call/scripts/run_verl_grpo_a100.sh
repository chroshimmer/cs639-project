#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets}"
export HF_HOME="${HF_HOME:-/tmp/hf_home}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

DATASET="${DATASET:-data/rolling_sepsis_trajectories.json}"
DB_PATH="${DB_PATH:-mimic/mimic4_dk.db}"
SPLIT_PATH="${SPLIT_PATH:-data/rl_tool_call/sepsis_split_seed7.json}"
VERL_DATA_DIR="${VERL_DATA_DIR:-data/rl_tool_call/verl_sepsis_core}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/sepsis_core_qwen4b_verl_grpo}"
CONFIG_NAME="${CONFIG_NAME:-sepsis_core_grpo_qwen4b_a100}"

prepare_data() {
  if [[ ! -f "$SPLIT_PATH" ]]; then
    python3 -m sepsis_mvp.cli make-split \
      --dataset "$DATASET" \
      --output "$SPLIT_PATH" \
      --train-size 70 \
      --val-size 14 \
      --test-size 14 \
      --seed 7
  fi

  python3 -m sepsis_mvp.verl_export \
    --db-path "$DB_PATH" \
    --dataset "$DATASET" \
    --split "$SPLIT_PATH" \
    --output-dir "$VERL_DATA_DIR" \
    --tool-scope sepsis_core \
    --max-step-interactions 3 \
    --output-format parquet
}

run_verl() {
  python3 -m verl.trainer.main_ppo \
    --config-path "$ROOT_DIR/configs/verl" \
    --config-name "$CONFIG_NAME" \
    'hydra.searchpath=[pkg://verl.trainer.config]' \
    data.train_files="$VERL_DATA_DIR/train.parquet" \
    data.val_files="$VERL_DATA_DIR/val.parquet" \
    actor_rollout_ref.model.path="$MODEL" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    "$@"
}

case "$MODE" in
  prepare)
    prepare_data
    ;;
  smoke)
    prepare_data
    run_verl \
      data.train_max_samples=1 \
      data.val_max_samples=1 \
      data.train_batch_size=1 \
      data.val_batch_size=1 \
      data.max_prompt_length=4096 \
      actor_rollout_ref.rollout.n=1 \
      actor_rollout_ref.rollout.response_length=64 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
      actor_rollout_ref.actor.ppo_mini_batch_size=1 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
      trainer.total_epochs=1 \
      trainer.val_before_train=false \
      trainer.test_freq=-1 \
      trainer.save_freq=-1 \
      trainer.logger='[console]' \
      trainer.default_local_dir=checkpoints/smoke_sepsis_core_qwen4b_verl_grpo \
      "$@"
    ;;
  full)
    prepare_data
    run_verl "$@"
    ;;
  eval)
    ADAPTER="${ADAPTER:-}"
    if [[ -z "$ADAPTER" ]]; then
      echo "Set ADAPTER=/path/to/final/peft/adapter before running eval." >&2
      exit 2
    fi
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" QWEN_CUDA_DEVICE=0 python3 -m sepsis_mvp.cli run \
      --db-path "$DB_PATH" \
      --dataset "$DATASET" \
      --split "$SPLIT_PATH" \
      --split-name test \
      --agent qwen \
      --model "$MODEL" \
      --adapter "$ADAPTER" \
      --task-mode single \
      --tool-backend official \
      --tool-scope sepsis_core \
      --protocol rolling_toolbox_with_history \
      --rollouts-output data/rl_tool_call/test_verl_grpo_qwen4b_rollouts.json \
      --trajectory-output data/rl_tool_call/test_verl_grpo_qwen4b_trajectories.jsonl \
      --evaluation-output data/rl_tool_call/test_verl_grpo_qwen4b_eval.json \
      --events-output data/rl_tool_call/test_verl_grpo_qwen4b_events.jsonl \
      "$@"
    ;;
  *)
    echo "Usage: $0 {prepare|smoke|full|eval} [extra verl/eval args]" >&2
    exit 2
    ;;
esac
