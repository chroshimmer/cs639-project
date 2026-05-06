# A100 verl GRPO Runbook

This is the recommended next step for the RL tool-calling project: direct
`Qwen/Qwen3-4B-Instruct-2507 -> verl Agent Loop GRPO`, with no SFT warm start.

SFT remains optional. Use it only if direct GRPO wastes too many samples on
invalid JSON or never learns the infection-to-SOFA transition.

## Inputs

Expected files on the A100 machine:

- `data/rolling_sepsis_trajectories.json`
- `mimic/mimic4_dk.db`
- this repo's `src/`, `configs/`, `scripts/`, and `requirements-verl.txt`

The run script creates these if missing:

- `data/rl_tool_call/sepsis_split_seed7.json`
- `data/rl_tool_call/verl_sepsis_core/train.parquet`
- `data/rl_tool_call/verl_sepsis_core/val.parquet`
- `data/rl_tool_call/verl_sepsis_core/test.parquet`

## Install

Use a clean Python environment on the A100 machine, then install:

```bash
python3 -m pip install -r requirements-verl.txt
python3 -m pip install --no-deps --force-reinstall torchao==0.17.0
```

The pinned stack is centered on:

- `verl==0.7.1`
- `ray==2.55.1`
- `sglang==0.5.10.post1`
- `torchao==0.17.0`
- `pyarrow==24.0.0`

`sglang==0.5.10.post1` declares `torchao==0.9.0`, so `torchao==0.17.0` must be
installed as a post-install override. Earlier local testing reached LoRA
injection only after using this newer torchao version.

## One-Command Smoke

Run this first on a clean A100:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_verl_grpo_a100.sh smoke
```

This command:

1. creates the 70/14/14 stay-level split if needed,
2. exports verl parquet data,
3. runs a one-sample, one-rollout GRPO trainer smoke,
4. disables validation and checkpoint saving for speed.

Expected duration: about 5-20 minutes, mostly depending on model download/cache
state and verl/sglang startup.

## Full Direct GRPO

After smoke passes:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_verl_grpo_a100.sh full
```

This uses `configs/verl/sepsis_core_grpo_qwen4b_a100.yaml`:

- model: `Qwen/Qwen3-4B-Instruct-2507`
- actor: LoRA rank 16, bf16-compatible A100 path
- algorithm: GRPO, no critic
- rollout: async sglang, `n=4`
- data: core sepsis toolbox only
- epochs: 3
- output: `checkpoints/sepsis_core_qwen4b_verl_grpo`

Expected duration for the current 70-stay training split: roughly 2-6 hours on
a clean A100. Budget half a day if the remote environment still needs package,
CUDA, model-cache, or verl Agent Loop adjustments.

## A100 40GB Safer Full Run

If the remote A100 is 40GB or memory is tight, start with a smaller full run:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_verl_grpo_a100.sh full \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
```

If that works, increase `actor_rollout_ref.rollout.n` back toward `4`.

## Evaluation

verl checkpoint layout can vary. After training, locate the final PEFT adapter
directory under `checkpoints/sepsis_core_qwen4b_verl_grpo`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 ADAPTER=/path/to/final_adapter ./scripts/run_verl_grpo_a100.sh eval
```

Outputs:

- `data/rl_tool_call/test_verl_grpo_qwen4b_eval.json`
- `data/rl_tool_call/test_verl_grpo_qwen4b_rollouts.json`
- `data/rl_tool_call/test_verl_grpo_qwen4b_trajectories.jsonl`
- `data/rl_tool_call/test_verl_grpo_qwen4b_events.jsonl`

Compare the result against:

- `data/rl_tool_call/full_prompt_tool_qwen4b_eval.json`
- `data/rl_tool_call/full_rl_policy_proxy_eval.json`

Primary metrics to watch:

- step macro F1
- alert F1
- infection and alert timing MAE
- necessary SOFA-call coverage
- unsupported positive-action rate
- average tool calls per step

## Manual Commands

The script is the preferred interface. The equivalent manual full command is:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src HF_DATASETS_CACHE=/tmp/hf_datasets HF_HOME=/tmp/hf_home HYDRA_FULL_ERROR=1 \
python3 -m verl.trainer.main_ppo \
  --config-path "$PWD/configs/verl" \
  --config-name sepsis_core_grpo_qwen4b_a100 \
  'hydra.searchpath=[pkg://verl.trainer.config]' \
  data.train_files=data/rl_tool_call/verl_sepsis_core/train.parquet \
  data.val_files=data/rl_tool_call/verl_sepsis_core/val.parquet \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
  trainer.default_local_dir=checkpoints/sepsis_core_qwen4b_verl_grpo
```

Do not add `actor_rollout_ref.model.lora_adapter_path=...` for the first run;
that would switch the experiment from direct RL to SFT-initialized RL.
