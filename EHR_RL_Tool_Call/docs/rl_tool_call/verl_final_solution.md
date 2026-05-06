# Final RL Fine-Tuning Solution: verl Agent Loop GRPO

Status: implemented and ready for a clean A100 smoke/full run.

## Decision

Use **verl Agent Loop + GRPO** as the final RL fine-tuning path.

Do not use `src/sepsis_mvp/train_rl.py` as the final result. That script is only a lightweight smoke/ablation trainer over exported states. It can prove that LoRA checkpoints can be produced and re-evaluated, but it does not train the full multi-turn tool-calling policy against the live environment.

The final solution should train the model in the same interaction regime used at evaluation time:

1. Model receives current checkpoint prompt plus rolling history.
2. Model emits a JSON tool call or final action.
3. Environment executes the official MIMIC tool.
4. Tool result is appended to the conversation.
5. Model continues until it emits a final action or hits the max interaction limit.
6. Reward is computed from final action correctness, evidence sufficiency, timing, tool coverage, tool efficiency, and schema validity.
7. GRPO updates the policy from grouped rollouts.

## Why verl

verl is the right target because it is designed for LLM RL at scale and now supports the agentic pieces this task needs:

- GRPO avoids training a separate critic/value model, which is simpler and cheaper for a one-A100 first run.
- Agent Loop supports user-defined multi-turn rollout logic with LLM calls, tools, and environment interaction.
- Async rollout support reduces GPU idle time while the environment is doing CPU/DB tool calls.
- Custom reward functions are first-class in the config.

This matches the project goal better than a hand-written local policy-gradient loop.

## Why GRPO Instead Of PPO First

Use GRPO first.

Reasons:

- No separate critic model, so lower memory and lower implementation complexity.
- Grouped sampling is natural for this task: for the same checkpoint state, sample several candidate tool/action continuations and reward the better ones.
- The reward is mostly outcome/evidence based, not dense token-level human preference scoring.
- On one A100, GRPO gives the best chance of fitting Qwen 4B with LoRA/QLoRA and useful rollout throughput.

Use PPO later only if GRPO is unstable or if we need a value function for longer-horizon credit assignment.

## Model Choice For One A100

Preferred first run:

- `Qwen/Qwen3-4B-Instruct-2507`
- LoRA or QLoRA
- bf16
- core sepsis toolbox only

Fallback:

- `Qwen/Qwen3-1.7B`
- LoRA
- bf16 or fp16

The 4B model is preferable because it already worked as the prompt-tool baseline and follows JSON better than the 1.7B model. Use 1.7B only for fast pipeline debugging or if the A100 environment has memory/runtime problems.

## SFT Position

SFT is optional, not required.

Final comparison should ideally include both:

1. `base -> verl GRPO`
2. `base -> SFT warm start -> verl GRPO`

The SFT warm start is an engineering stabilizer:

- It teaches the exact JSON grammar.
- It teaches the infection-to-SOFA transition.
- It reduces wasted RL samples from malformed output.

But the main scientific claim should be based on the RL-tuned checkpoint, not on SFT alone.

## Required Implementation Pieces

Already implemented:

- stay-level split CLI
- `sepsis_core` tool scope
- prompt/eval adapter loading with `--adapter`
- oracle SFT trace export
- LoRA SFT entrypoint
- lightweight RL smoke trainer

Implemented now:

1. **verl dataset export**
   - Output parquet files for train/val/test.
   - Include raw chat prompt fields required by Agent Loop.
   - Include trajectory id, stay id, step index, t_hour, ground-truth label, rolling-history seed fields, and reward metadata.
   - Implementation: `src/sepsis_mvp/verl_export.py`

2. **verl Agent Loop**
   - Implement a `SepsisToolAgentLoop` class.
   - It owns the per-rollout checkpoint state.
   - It calls the async LLM server each turn.
   - It parses JSON tool calls/final actions.
   - It executes `query_suspicion_of_infection` and `query_sofa` against the official DuckDB backend.
   - It appends compact tool outputs to the conversation.
   - It stops after final action or max interactions.
   - It returns token ids and response mask in the format expected by verl Agent Loop.
   - Initial implementation: `src/sepsis_mvp/verl_agent_loop.py`

3. **verl reward function**
   - Refactor `src/sepsis_mvp/rl_reward.py` into a per-rollout reward function callable by verl.
   - Reward terms:
     - action correctness
     - valid JSON/schema
     - positive decision evidence support
     - necessary infection tool coverage
     - necessary SOFA tool coverage
     - repeated/low-utility tool penalties
     - safety checks for stay_id/t_hour
     - trajectory timing bonus when training by full stay
   - Initial implementation: `src/sepsis_mvp/verl_reward.py`

4. **verl config**
   - GRPO
   - LoRA/QLoRA actor
   - no critic
   - KL loss against reference policy
   - async rollout mode
   - Agent Loop enabled
   - `data.return_raw_chat=True`
   - one A100 resource settings
   - initial config path: `configs/verl/sepsis_core_grpo_qwen4b_a100.yaml`

5. **checkpoint evaluation bridge**
   - Load the trained verl/PEFT checkpoint through existing `--adapter`.
   - Re-run frozen test split with the exact same `rolling_toolbox_with_history` protocol.

Still needs A100 validation:

- Confirm final checkpoint/adapter save directory emitted by verl.
- Run a 1-2 batch smoke train on an actually free GPU before launching the full run.

## Final CLI Shape

Install final training stack on the A100 machine:

```bash
python3 -m pip install -r requirements-verl.txt
python3 -m pip install --no-deps --force-reinstall torchao==0.17.0
```

The current pinned stack is centered on `verl==0.7.1`, `ray==2.55.1`, `sglang==0.5.10.post1`, `trl==0.9.6`, and `torchao==0.17.0`. `sglang` declares an older `torchao` pin, but PEFT LoRA injection requires the newer `torchao`; the local smoke reached LoRA injection only after upgrading `torchao`.

## Smoke Status

Local smoke passed:

- Python syntax check for verl export/reward/agent-loop modules.
- Parquet export path for train/val/test.
- `verl.utils.dataset.RLHFDataset` loads the exported parquet with native chat prompts and dict `extra_info`.
- Reward function direct call.
- Agent Loop class imports and registers as `sepsis_tool_agent_loop`.
- Hydra config composes against the installed `verl==0.7.1` schema.
- Minimal trainer launch starts Ray, validates config, loads the dataset, loads Qwen weights, and applies LoRA.
- Existing unit suite: `PYTHONPATH=src python3 -m unittest discover -s tests` passes 60 tests.

Local blocker:

- The trainer smoke fails after LoRA injection with CUDA OOM during FSDP wrapping because the available local GPUs are already occupied. The observed failure was GPU 0 with about 384 MiB free and another process using about 34 GiB. This is a resource blocker, not a missing-code blocker.

Create split:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli make-split \
  --dataset data/rolling_sepsis_trajectories.json \
  --output data/rl_tool_call/sepsis_split_seed7.json \
  --train-size 70 \
  --val-size 14 \
  --test-size 14 \
  --seed 7
```

Export verl parquet data:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.verl_export \
  --db-path mimic/mimic4_dk.db \
  --dataset data/rolling_sepsis_trajectories.json \
  --split data/rl_tool_call/sepsis_split_seed7.json \
  --output-dir data/rl_tool_call/verl_sepsis_core \
  --tool-scope sepsis_core \
  --max-step-interactions 3
```

Optional SFT warm start:

```bash
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

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 -m sepsis_mvp.train_sft \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --train data/rl_tool_call/sepsis_core_oracle_sft_train.jsonl \
  --validation data/rl_tool_call/sepsis_core_oracle_sft_val.jsonl \
  --output-dir checkpoints/sepsis_core_qwen4b_sft_lora \
  --bf16 \
  --gradient-checkpointing
```

Run verl GRPO:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src HF_DATASETS_CACHE=/tmp/hf_datasets HF_HOME=/tmp/hf_home HYDRA_FULL_ERROR=1 \
python3 -m verl.trainer.main_ppo \
  --config-path "$PWD/configs/verl" \
  --config-name sepsis_core_grpo_qwen4b_a100 \
  'hydra.searchpath=[pkg://verl.trainer.config]' \
  data.train_files=data/rl_tool_call/verl_sepsis_core/train.parquet \
  data.val_files=data/rl_tool_call/verl_sepsis_core/val.parquet \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
  actor_rollout_ref.model.lora_adapter_path=checkpoints/sepsis_core_qwen4b_sft_lora \
  trainer.default_local_dir=checkpoints/sepsis_core_qwen4b_verl_grpo
```

Recommended first A100 verl smoke, before the full run:

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
  trainer.default_local_dir=checkpoints/smoke_sepsis_core_qwen4b_verl_grpo
```

Evaluate frozen test split:

```bash
CUDA_VISIBLE_DEVICES=0 QWEN_CUDA_DEVICE=0 PYTHONPATH=src python3 -m sepsis_mvp.cli run \
  --db-path mimic/mimic4_dk.db \
  --dataset data/rolling_sepsis_trajectories.json \
  --split data/rl_tool_call/sepsis_split_seed7.json \
  --split-name test \
  --agent qwen \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter checkpoints/sepsis_core_qwen4b_verl_grpo/final_adapter \
  --task-mode single \
  --tool-backend official \
  --tool-scope sepsis_core \
  --protocol rolling_toolbox_with_history \
  --rollouts-output data/rl_tool_call/test_verl_grpo_qwen4b_rollouts.json \
  --trajectory-output data/rl_tool_call/test_verl_grpo_qwen4b_trajectories.jsonl \
  --evaluation-output data/rl_tool_call/test_verl_grpo_qwen4b_eval.json \
  --events-output data/rl_tool_call/test_verl_grpo_qwen4b_events.jsonl
```

The exact checkpoint path may differ depending on verl save format. The evaluation command should point `--adapter` at the final PEFT adapter directory.

## Reporting Table

Final report should compare:

| Run | Model | Init | RL framework | Tool scope | Split | Notes |
|---|---|---|---|---|---|---|
| Prompt tool baseline | Qwen 4B | base | none | shared or sepsis_core | test | prompt-only tool policy |
| SFT only | Qwen 4B | SFT LoRA | none | sepsis_core | test | optional stabilizer |
| Direct GRPO | Qwen 4B | base | verl GRPO | sepsis_core | test | primary RL result |
| SFT + GRPO | Qwen 4B | SFT LoRA | verl GRPO | sepsis_core | test | likely strongest |
| Reward-policy proxy | heuristic | none | none | sepsis_core | test | non-learned reference |

Primary metrics:

- macro F1
- alert F1
- infection transition exact/MAE
- alert transition exact/MAE
- necessary infection-call coverage
- necessary SOFA-call coverage
- unsupported positive-action rate
- repeated call rate
- average tool calls per step
