# RL Tool-Calling Training Readiness

Status checked: 2026-04-30

## Bottom Line

The project is **ready to move to a clean A100 for the final verl GRPO smoke/full run**.

The remaining blocker on this machine is GPU availability, not missing code. A minimal verl trainer launch reached Ray startup, config validation, dataset loading, Qwen model loading, and LoRA injection, then failed during FSDP wrapping because GPU 0 had only about 384 MiB free. Other GPUs were also occupied.

What is ready:

- single-task rolling sepsis dataset: `data/rolling_sepsis_trajectories.json`
- official MIMIC DuckDB tool backend: `mimic/mimic4_dk.db`
- prompt-driven Qwen tool-calling baseline runner
- reward-policy proxy runner
- deterministic rollout/evaluation artifacts
- offline rollout reward scorer: `src/sepsis_mvp/rl_reward.py`
- rollout environment that can be reused as the basis for an RL environment: `src/sepsis_mvp/environment.py`

What is now implemented in code:

- stay-level split CLI: `make-split`
- split filtering for rollout, prompt-baseline, and reward scoring commands
- `--tool-scope sepsis_core|shared` for `rolling_toolbox_with_history`
- LoRA adapter loading for Qwen evaluation through `--adapter`
- oracle trace export CLI: `export-sft-traces`
- optional LoRA SFT script: `python3 -m sepsis_mvp.train_sft`
- lightweight grouped policy-gradient script: `python3 -m sepsis_mvp.train_rl`
- final verl parquet export: `python3 -m sepsis_mvp.verl_export`
- final verl Agent Loop: `src/sepsis_mvp/verl_agent_loop.py`
- final verl reward hook: `src/sepsis_mvp/verl_reward.py`
- one-A100 verl GRPO config: `configs/verl/sepsis_core_grpo_qwen4b_a100.yaml`

Validated locally:

- `python3 -m py_compile` for the verl/export/reward/CLI modules
- direct reward function smoke
- parquet export for train/val/test
- `verl.utils.dataset.RLHFDataset` can read the exported parquet with native chat prompts and dict `extra_info`
- Hydra config composes against installed `verl==0.7.1`
- minimal trainer launch reaches FSDP/LoRA model initialization
- unit suite: `PYTHONPATH=src python3 -m unittest discover -s tests` runs 60 tests successfully

Treat `src/sepsis_mvp/train_rl.py` as a smoke/ablation trainer only. The final solution is `verl` Agent Loop + GRPO; see `verl_final_solution.md`.

## Existing Commands That Work Today

Prompt-driven Qwen tool-calling baseline:

```bash
CUDA_VISIBLE_DEVICES=0 QWEN_CUDA_DEVICE=0 PYTHONPATH=src python3 -m sepsis_mvp.cli run \
  --db-path mimic/mimic4_dk.db \
  --dataset data/rolling_sepsis_trajectories.json \
  --agent qwen \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --task-mode single \
  --tool-backend official \
  --protocol rolling_toolbox_with_history \
  --rollouts-output data/rl_tool_call/full_prompt_tool_qwen4b_rollouts.json \
  --trajectory-output data/rl_tool_call/full_prompt_tool_qwen4b_trajectories.jsonl \
  --evaluation-output data/rl_tool_call/full_prompt_tool_qwen4b_eval.json \
  --events-output data/rl_tool_call/full_prompt_tool_qwen4b_events.jsonl \
  --resume
```

Reward-policy proxy:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli run \
  --db-path mimic/mimic4_dk.db \
  --dataset data/rolling_sepsis_trajectories.json \
  --agent heuristic \
  --task-mode single \
  --tool-backend official \
  --protocol rolling_toolbox_with_history \
  --rollouts-output data/rl_tool_call/full_rl_policy_proxy_rollouts.json \
  --trajectory-output data/rl_tool_call/full_rl_policy_proxy_trajectories.jsonl \
  --evaluation-output data/rl_tool_call/full_rl_policy_proxy_eval.json \
  --events-output data/rl_tool_call/full_rl_policy_proxy_events.jsonl
```

Offline reward scoring for saved rollouts:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli score-rollouts \
  --dataset data/rolling_sepsis_trajectories.json \
  --rollouts data/rl_tool_call/full_prompt_tool_qwen4b_rollouts.json \
  --output data/rl_tool_call/full_prompt_tool_qwen4b_rewards.json
```

## Training CLI

Create stay-level split:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli make-split \
  --dataset data/rolling_sepsis_trajectories.json \
  --output data/rl_tool_call/sepsis_split_seed7.json \
  --train-size 70 \
  --val-size 14 \
  --test-size 14 \
  --seed 7
```

Export oracle traces for optional SFT warm start:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli export-sft-traces \
  --db-path mimic/mimic4_dk.db \
  --dataset data/rolling_sepsis_trajectories.json \
  --split data/rl_tool_call/sepsis_split_seed7.json \
  --split-name train \
  --tool-backend official \
  --tool-scope sepsis_core \
  --output data/rl_tool_call/sepsis_core_oracle_sft_train.jsonl
```

Run LoRA SFT warm start:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 -m sepsis_mvp.train_sft \
  --model Qwen/Qwen3-1.7B \
  --train data/rl_tool_call/sepsis_core_oracle_sft_train.jsonl \
  --validation data/rl_tool_call/sepsis_core_oracle_sft_val.jsonl \
  --output-dir checkpoints/sepsis_core_qwen17b_sft_lora \
  --max-seq-length 2048 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate 2e-4 \
  --num-train-epochs 5 \
  --fp16
```

Run lightweight RL fine-tuning from the SFT adapter. This is not the final method, only a smoke/ablation path:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 -m sepsis_mvp.train_rl \
  --model Qwen/Qwen3-1.7B \
  --adapter checkpoints/sepsis_core_qwen17b_sft_lora \
  --train-traces data/rl_tool_call/sepsis_core_oracle_sft_train.jsonl \
  --validation-traces data/rl_tool_call/sepsis_core_oracle_sft_val.jsonl \
  --output-dir checkpoints/sepsis_core_qwen17b_grpo_lora \
  --updates 500 \
  --batch-size 2 \
  --group-size 4 \
  --max-new-tokens 128 \
  --learning-rate 5e-6 \
  --fp16
```

Run lightweight RL directly from the base model without SFT. This is not the final method, only a smoke/ablation path:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 -m sepsis_mvp.train_rl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --train-traces data/rl_tool_call/sepsis_core_oracle_sft_train.jsonl \
  --validation-traces data/rl_tool_call/sepsis_core_oracle_sft_val.jsonl \
  --output-dir checkpoints/sepsis_core_qwen4b_grpo_lora \
  --updates 800 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 128 \
  --learning-rate 5e-6 \
  --bf16 \
  --gradient-checkpointing
```

Evaluate trained checkpoint on held-out test stays:

```bash
CUDA_VISIBLE_DEVICES=0 QWEN_CUDA_DEVICE=0 PYTHONPATH=src python3 -m sepsis_mvp.cli run \
  --db-path mimic/mimic4_dk.db \
  --dataset data/rolling_sepsis_trajectories.json \
  --split data/rl_tool_call/sepsis_split_seed7.json \
  --split-name test \
  --agent qwen \
  --model Qwen/Qwen3-1.7B \
  --adapter checkpoints/sepsis_core_qwen17b_grpo_lora \
  --task-mode single \
  --tool-backend official \
  --tool-scope sepsis_core \
  --protocol rolling_toolbox_with_history \
  --rollouts-output data/rl_tool_call/test_rl_qwen17b_grpo_rollouts.json \
  --trajectory-output data/rl_tool_call/test_rl_qwen17b_grpo_trajectories.jsonl \
  --evaluation-output data/rl_tool_call/test_rl_qwen17b_grpo_eval.json \
  --events-output data/rl_tool_call/test_rl_qwen17b_grpo_events.jsonl
```

## Why SFT Is Optional

SFT is not conceptually required for RL. A policy can start from the base LLM and be optimized directly with the reward function.

For this project, SFT is recommended as an engineering stabilizer, not as a scientific requirement:

- The smaller Qwen model already showed malformed JSON and `<think>` prose in smoke tests.
- RL rollouts are expensive if most samples fail schema validation before reaching meaningful clinical decisions.
- The action space is highly structured: emit one JSON tool call or one JSON final action.
- SFT can teach the model the grammar and the infection-to-SOFA transition before RL starts optimizing timing and tool efficiency.

Direct RL is still valid if the training stack can tolerate a high initial invalid-output rate. In that setup, the reward must strongly penalize malformed JSON, invalid tool names, and final positive actions without evidence. The risk is sample inefficiency, not invalid methodology.

Recommended comparison:

1. Prompted tool-calling Qwen baseline.
2. SFT-only model, if SFT is used.
3. RL-trained model initialized from base Qwen.
4. RL-trained model initialized from SFT, if compute allows.
5. Reward-policy proxy as a non-learned upper/reference policy.

## Suggested A100 First Run

For one A100, start with `Qwen/Qwen3-4B-Instruct-2507` if the model is available in the target environment. Use bf16 on A100.

Install:

```bash
python3 -m pip install -r requirements-training.txt
```

For the final verl path, use:

```bash
python3 -m pip install -r requirements-verl.txt
```

Export verl parquet:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.verl_export \
  --db-path mimic/mimic4_dk.db \
  --dataset data/rolling_sepsis_trajectories.json \
  --split data/rl_tool_call/sepsis_split_seed7.json \
  --output-dir data/rl_tool_call/verl_sepsis_core \
  --tool-scope sepsis_core \
  --max-step-interactions 3 \
  --output-format parquet
```

Minimal verl smoke on a clean A100:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src HF_DATASETS_CACHE=/tmp/hf_datasets HF_HOME=/tmp/hf_home HYDRA_FULL_ERROR=1 \
python3 -m verl.trainer.main_ppo \
  --config-path "$PWD/configs/verl" \
  --config-name sepsis_core_grpo_qwen4b_a100 \
  'hydra.searchpath=[pkg://verl.trainer.config]' \
  data.train_files=data/rl_tool_call/verl_sepsis_core/train.parquet \
  data.val_files=data/rl_tool_call/verl_sepsis_core/val.parquet \
  data.train_max_samples=1 \
  data.val_max_samples=1 \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=4096 \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.response_length=64 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  trainer.default_local_dir=checkpoints/smoke_sepsis_core_qwen4b_verl_grpo \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.logger='[console]'
```

Create split and traces:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli make-split \
  --dataset data/rolling_sepsis_trajectories.json \
  --output data/rl_tool_call/sepsis_split_seed7.json \
  --train-size 70 \
  --val-size 14 \
  --test-size 14 \
  --seed 7

PYTHONPATH=src python3 -m sepsis_mvp.cli export-sft-traces \
  --db-path mimic/mimic4_dk.db \
  --dataset data/rolling_sepsis_trajectories.json \
  --split data/rl_tool_call/sepsis_split_seed7.json \
  --split-name train \
  --tool-scope sepsis_core \
  --output data/rl_tool_call/sepsis_core_oracle_sft_train.jsonl

PYTHONPATH=src python3 -m sepsis_mvp.cli export-sft-traces \
  --db-path mimic/mimic4_dk.db \
  --dataset data/rolling_sepsis_trajectories.json \
  --split data/rl_tool_call/sepsis_split_seed7.json \
  --split-name val \
  --tool-scope sepsis_core \
  --output data/rl_tool_call/sepsis_core_oracle_sft_val.jsonl
```

Optional SFT:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 -m sepsis_mvp.train_sft \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --train data/rl_tool_call/sepsis_core_oracle_sft_train.jsonl \
  --validation data/rl_tool_call/sepsis_core_oracle_sft_val.jsonl \
  --output-dir checkpoints/sepsis_core_qwen4b_sft_lora \
  --max-seq-length 2048 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --num-train-epochs 3 \
  --bf16 \
  --gradient-checkpointing
```

Lightweight RL:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 -m sepsis_mvp.train_rl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter checkpoints/sepsis_core_qwen4b_sft_lora \
  --train-traces data/rl_tool_call/sepsis_core_oracle_sft_train.jsonl \
  --validation-traces data/rl_tool_call/sepsis_core_oracle_sft_val.jsonl \
  --output-dir checkpoints/sepsis_core_qwen4b_rl_lora \
  --updates 800 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 128 \
  --learning-rate 5e-6 \
  --bf16 \
  --gradient-checkpointing
```

Held-out test evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 QWEN_CUDA_DEVICE=0 PYTHONPATH=src python3 -m sepsis_mvp.cli run \
  --db-path mimic/mimic4_dk.db \
  --dataset data/rolling_sepsis_trajectories.json \
  --split data/rl_tool_call/sepsis_split_seed7.json \
  --split-name test \
  --agent qwen \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter checkpoints/sepsis_core_qwen4b_rl_lora \
  --task-mode single \
  --tool-backend official \
  --tool-scope sepsis_core \
  --protocol rolling_toolbox_with_history \
  --rollouts-output data/rl_tool_call/test_rl_qwen4b_rollouts.json \
  --trajectory-output data/rl_tool_call/test_rl_qwen4b_trajectories.jsonl \
  --evaluation-output data/rl_tool_call/test_rl_qwen4b_eval.json \
  --events-output data/rl_tool_call/test_rl_qwen4b_events.jsonl
```
