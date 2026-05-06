# RL Tool-Calling Full Run Summary

Date: 2026-04-30

Dataset: `data/rolling_sepsis_trajectories.json`

Database: `mimic/mimic4_dk.db`

Task: rolling sepsis state prediction over 98 MIMIC-IV stays, with actions:

- `keep_monitoring`
- `infection_suspect`
- `trigger_sepsis_alert`

## Experiment Outputs

| Experiment | Eval | Rollouts | Rewards | Events |
|---|---|---|---|---|
| Qwen prompt-driven tool calling baseline | `data/rl_tool_call/full_prompt_tool_qwen4b_eval.json` | `data/rl_tool_call/full_prompt_tool_qwen4b_rollouts.json` | `data/rl_tool_call/full_prompt_tool_qwen4b_rewards.json` | `data/rl_tool_call/full_prompt_tool_qwen4b_events.jsonl` |
| Reward-policy proxy for RL tool calling | `data/rl_tool_call/full_rl_policy_proxy_eval.json` | `data/rl_tool_call/full_rl_policy_proxy_rollouts.json` | `data/rl_tool_call/full_rl_policy_proxy_rewards.json` | `data/rl_tool_call/full_rl_policy_proxy_events.jsonl` |

## Headline Metrics

| Metric | Qwen prompt tool baseline | Reward-policy proxy |
|---|---:|---:|
| Trajectories | 98 | 98 |
| Step accuracy | 0.4446 | 0.7988 |
| Step macro F1 | 0.3391 | 0.7234 |
| Infection exact timing | 0.1837 | 0.4388 |
| Infection MAE hours | 8.6022 | 2.0351 |
| Sepsis alert exact timing | 0.3878 | 0.6122 |
| Sepsis alert MAE hours | 6.9048 | 1.8367 |
| Infection grounding rate | 0.5479 | 1.0000 |
| Alert grounding rate | 0.0000 | 1.0000 |
| Avg tool calls per step | 0.2391 | 1.6195 |
| Steps without tool calls | 0.7609 | 0.0000 |
| Necessary infection-call coverage | 0.2892 | 1.0000 |
| Necessary SOFA-call coverage | 0.0000 | 1.0000 |
| Unsupported positive-action rate | 0.5372 | 0.0000 |
| Mean trajectory reward | 0.2008 | 1.1468 |
| Mean step reward | 0.3046 | 1.0508 |

## Tool Use

Qwen prompt-driven baseline:

- `query_suspicion_of_infection`: 164
- `query_sofa`: 0

Reward-policy proxy:

- `query_suspicion_of_infection`: 686
- `query_sofa`: 425

## Interpretation

The prompt-driven Qwen baseline frequently failed to maintain or act on intermediate diagnostic state. It made sparse tool calls, skipped tools on 76.09% of steps, never called SOFA, and produced alert predictions with zero alert-grounding coverage. This directly matches the intended baseline limitation: prompt instruction alone did not reliably make the model track the infection-to-organ-dysfunction decision chain.

The reward-policy proxy is not a trained RL checkpoint. It is the current executable proxy for the intended RL-improved policy: a reward-shaped tool policy that explicitly optimizes evidence coverage, timing, and tool efficiency. It establishes the target behavior and validates the environment, tools, metrics, and reward path before integrating a true verl training loop.

## Commands

Qwen prompt-driven tool baseline:

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

Reward scoring:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli score-rollouts \
  --dataset data/rolling_sepsis_trajectories.json \
  --rollouts data/rl_tool_call/full_prompt_tool_qwen4b_rollouts.json \
  --output data/rl_tool_call/full_prompt_tool_qwen4b_rewards.json

PYTHONPATH=src python3 -m sepsis_mvp.cli score-rollouts \
  --dataset data/rolling_sepsis_trajectories.json \
  --rollouts data/rl_tool_call/full_rl_policy_proxy_rollouts.json \
  --output data/rl_tool_call/full_rl_policy_proxy_rewards.json
```
