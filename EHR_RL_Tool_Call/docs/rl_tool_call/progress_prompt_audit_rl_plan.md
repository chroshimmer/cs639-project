# RL Tool-Calling Progress, Prompt Audit, and Training Plan

Date: 2026-04-30

Workspace: `/home/chloeliu/rolling_healthcare_agent`

MIMIC DuckDB: `mimic/mimic4_dk.db`

## Current Objective

Build a function-calling agent for healthcare patient diagnosis on MIMIC-IV. The first task is rolling sepsis state recognition because it naturally requires intermediate diagnostic state:

1. Establish suspected infection.
2. If infection is established, evaluate organ dysfunction with SOFA.
3. Trigger sepsis alert only when infection and SOFA alert evidence are both present.

This task is better than infection-only or SOFA-only because the policy must decide when to transition from one evidence question to the next.

## Implemented Components

### Documentation

- `docs/rl_tool_call/design_doc.md`
  - Initial experiment design.
  - Task choice rationale.
  - Baseline and improved-policy definitions.
  - Reward decomposition.
  - Evaluation metrics.
- `docs/rl_tool_call/design_doc.txt`
  - Pointer to the markdown design doc.
- `data/rl_tool_call/full_run_summary.md`
  - Full-run outputs, commands, and headline metrics.

### Dependencies

- `requirements-qwen.txt`
  - Minimal local requirements for Qwen inference and DuckDB-backed experiment runs.
  - Includes `duckdb`, `torch`, `transformers`, `accelerate`, `safetensors`, `sentencepiece`, `protobuf`, `Pillow`, and `jinja2`.

### Code

- `src/sepsis_mvp/agent.py`
  - Local Qwen chat agent.
  - Single-GPU placement through `QWEN_CUDA_DEVICE`.
  - Prompt-driven toolbox protocol.
  - Compact history payloads to avoid GPU OOM on long rollouts.
  - Tolerant parsing for malformed tool-call-like model outputs such as `{"action":"query_suspicion_of_infection"}`.
- `src/sepsis_mvp/prompt_baseline.py`
  - Prompt-card/no-visible-tool-call ablation.
  - This is not the main baseline after clarification; it is useful only as a no-tool ablation.
- `src/sepsis_mvp/rl_reward.py`
  - Reward scoring for sepsis rollouts.
  - Scores action correctness, intermediate state evidence, necessary tool coverage, tool efficiency, and format/safety.
- `src/sepsis_mvp/cli.py`
  - `run-prompt-baseline`
  - `score-rollouts`
  - Full experiment execution through `run`.
- `tests/test_pipeline.py`
  - Tests for prompt construction, toolbox behavior, prompt-card ablation, and reward scoring.

## Model and Hardware Status

### Available Hardware

Observed GPU:

- NVIDIA RTX 2080 Ti
- 11 GB VRAM
- CUDA visible only in escalated shell commands
- Current rule: use one GPU at a time

Working invocation pattern:

```bash
CUDA_VISIBLE_DEVICES=0 QWEN_CUDA_DEVICE=0 PYTHONPATH=src python3 -m sepsis_mvp.cli ...
```

### Models Tried

#### `Qwen/Qwen3-4B-Instruct-2507`

Status: usable for full prompt-driven inference on one RTX 2080 Ti after prompt/history compaction.

Notes:

- Full 98-trajectory prompt-driven tool-calling baseline completed.
- Earlier OOM was fixed by compacting tool payloads and interaction history.
- Full run runtime was about 1,416 seconds step-runtime total.
- The model followed JSON reasonably well but often emitted tool calls as `{"action":"query_suspicion_of_infection"}`, requiring repair.

#### `Qwen/Qwen3-1.7B`

Status: not reliable enough for this tool protocol without additional tuning.

Notes:

- Loaded successfully.
- Often emitted malformed JSON or `<think>` prose.
- Failed smoke retry with prose instead of the required JSON object.
- It may still be useful after SFT/LoRA formatting training, but it is not a good zero-shot baseline for this protocol.

## Completed Full Runs

### Prompt-Driven Tool-Calling Baseline

Command:

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

Outputs:

- `data/rl_tool_call/full_prompt_tool_qwen4b_eval.json`
- `data/rl_tool_call/full_prompt_tool_qwen4b_rollouts.json`
- `data/rl_tool_call/full_prompt_tool_qwen4b_rewards.json`
- `data/rl_tool_call/full_prompt_tool_qwen4b_events.jsonl`

Headline metrics:

| Metric | Value |
|---|---:|
| Trajectories | 98 |
| Step accuracy | 0.4446 |
| Step macro F1 | 0.3391 |
| Infection exact timing | 0.1837 |
| Infection MAE hours | 8.6022 |
| Alert exact timing | 0.3878 |
| Alert MAE hours | 6.9048 |
| Infection grounding rate | 0.5479 |
| Alert grounding rate | 0.0000 |
| Average tool calls per step | 0.2391 |
| Steps without tool calls | 0.7609 |
| Necessary infection-call coverage | 0.2892 |
| Necessary SOFA-call coverage | 0.0000 |
| Unsupported positive-action rate | 0.5372 |
| Mean trajectory reward | 0.2008 |
| Mean step reward | 0.3046 |

Tool calls:

- `query_suspicion_of_infection`: 164
- `query_sofa`: 0

### Reward-Policy Proxy

Command:

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

Outputs:

- `data/rl_tool_call/full_rl_policy_proxy_eval.json`
- `data/rl_tool_call/full_rl_policy_proxy_rollouts.json`
- `data/rl_tool_call/full_rl_policy_proxy_rewards.json`
- `data/rl_tool_call/full_rl_policy_proxy_events.jsonl`

Important caveat:

This is not yet a trained RL checkpoint. It is a reward-shaped executable policy proxy that validates the environment, tool API, metrics, and reward design.

Headline metrics:

| Metric | Value |
|---|---:|
| Trajectories | 98 |
| Step accuracy | 0.7988 |
| Step macro F1 | 0.7234 |
| Infection exact timing | 0.4388 |
| Infection MAE hours | 2.0351 |
| Alert exact timing | 0.6122 |
| Alert MAE hours | 1.8367 |
| Infection grounding rate | 1.0000 |
| Alert grounding rate | 1.0000 |
| Average tool calls per step | 1.6195 |
| Steps without tool calls | 0.0000 |
| Necessary infection-call coverage | 1.0000 |
| Necessary SOFA-call coverage | 1.0000 |
| Unsupported positive-action rate | 0.0000 |
| Mean trajectory reward | 1.1468 |
| Mean step reward | 1.0508 |

Tool calls:

- `query_suspicion_of_infection`: 686
- `query_sofa`: 425

## Prompt Audit: Why Qwen Never Called `query_sofa`

The observed result is real, not an evaluator artifact:

- `full_prompt_tool_qwen4b_eval.json` has `tool_call_counts = {"query_suspicion_of_infection": 164}`.
- `full_prompt_tool_qwen4b_events.jsonl` contains raw and repaired model outputs.
- No raw model output, repaired model output, or executed tool call used `query_sofa`.

### Event-Level Evidence

At `stay_id=30135840`, step 3, hour 12:

1. Raw model output:

```json
{"action": "query_suspicion_of_infection"}
```

2. Repair converted that malformed tool-call-like action into:

```json
{"tool_name":"query_suspicion_of_infection","arguments":{"stay_id":30135840,"t_hour":12}}
```

3. Infection tool output was positive:

```json
{
  "has_suspected_infection": true,
  "first_visible_suspected_infection_hour": -8.96
}
```

4. The next raw model output was:

```json
{"action": "infection_suspect"}
```

The correct next high-value tool should have been `query_sofa`, because infection was now established and alert-level organ dysfunction was unresolved.

### Prompt Content That Helped

The system prompt did contain several SOFA instructions:

- `query_sofa: current visible SOFA summary up to this checkpoint`
- `suspected infection plus acute organ dysfunction consistent with SOFA >= 2`
- `Let's check infection first, if met, then check sofa score using available tool calling function.`
- `Suspected infection and SOFA alert evidence are jointly necessary for trigger_sepsis_alert.`
- `If no earlier checkpoint explicitly established SOFA alert evidence, query_sofa before making trigger_sepsis_alert.`

So the issue is not that SOFA was absent from the prompt.

### Prompt Content That Hurt

The main failure is that prompt focus is computed only from prior `rolling_history`, not the current checkpoint's `history` or `tool_results_by_name`.

In the same step after the current infection tool returned positive, the prompt still included:

```text
Current rolling history does not yet establish suspected infection.
Infection evidence is the key unresolved question before any sepsis alert decision.
```

It also kept the single tool-call example as:

```json
{"tool_name":"query_suspicion_of_infection","arguments":{"stay_id":30135840,"t_hour":12}}
```

This creates a strong anchoring effect:

- The model sees `query_suspicion_of_infection` as the example tool call.
- The current-step positive infection result is present only inside JSON payload fields.
- The natural-language focus still says infection is unresolved.
- The prompt says `query_sofa` is needed before `trigger_sepsis_alert`, but it does not clearly say `query_sofa` is needed before deciding whether to remain at `infection_suspect` after a current positive infection result.
- The instruction `If rolling_history already explicitly established suspected infection... consider query_sofa` does not fire conceptually because current infection was established in current-step `history`, not earlier `rolling_history`.

### Output-Schema Confusion

The model repeatedly emitted:

```json
{"action":"query_suspicion_of_infection"}
```

That means the model blurred tool names and final action labels. The repair prompt used `query_sofa` as its example, but repair only ran when the first output was invalid. Once the model emitted a valid final action like:

```json
{"action":"infection_suspect"}
```

there was no repair step and no chance to force the SOFA tool call.

### Quantitative Pattern

Across the full baseline:

- Step 0 had zero tool calls for all 98 stays.
- Step 1 had zero tool calls for all 98 stays.
- Step 2 had zero tool calls for all 98 stays.
- Step 3 had 77 infection-tool calls and 21 no-tool decisions.
- Later steps mostly returned no-tool final actions.
- `trigger_sepsis_alert` was predicted 48 times with no `query_sofa` support.
- `necessary_call_coverage.sofa_for_alert = 0.0`.

This shows the baseline did not learn a robust intermediate-state controller from prompt instructions alone.

## Recommended Prompt Fix Before RL

Even if the final goal is RL, the environment should not contain avoidable prompt ambiguity. The baseline can remain as-is for reporting, but the train/eval prompt should be fixed before RL data collection.

Specific changes:

1. Compute current sepsis evidence state from both prior `rolling_history` and current-step `history`.
2. If current or prior infection is positive and SOFA has not been assessed or has not reached alert level, make `query_sofa` the example tool.
3. Replace weak wording `consider query_sofa` with a state-machine rule:

```text
If current or prior tool results show has_suspected_infection=true and no current or prior SOFA result shows max_sofa_24hours_so_far >= 2, the next tool call should be query_sofa before any final sepsis action.
```

4. Separate tool-call and final-action schemas more sharply:

```text
Tool calls must use key "tool_name". Final diagnoses must use key "action".
Never put a tool name in the "action" field.
```

5. Narrow the sepsis toolbox for the first RL stage to:

- `query_suspicion_of_infection`
- `query_sofa`

The 11-tool shared toolbox is useful later, but it makes the first policy-learning problem noisier.

## RL Training Plan

### Training Goal

Train a policy that learns the sepsis evidence-state machine:

1. If infection is unknown, call `query_suspicion_of_infection`.
2. If infection is negative, return `keep_monitoring`.
3. If infection is positive and SOFA is unknown or stale, call `query_sofa`.
4. If infection is positive and SOFA is below 2, return `infection_suspect`.
5. If infection is positive and SOFA is at least 2, return `trigger_sepsis_alert`.
6. Avoid unnecessary repeated calls when state is already explicit.

### Environment Design

Use an explicit Markov decision process over checkpoint interactions.

State observation:

- Patient/stay identifiers: `trajectory_id`, `stay_id`, `step_index`, `t_hour`.
- Prior checkpoint summary:
  - infection known/unknown/positive/negative
  - first infection hour if known
  - SOFA known/unknown
  - latest SOFA
  - max SOFA so far
  - prior action
- Current checkpoint tool history:
  - tools called this step
  - compact tool outputs this step
- Allowed actions:
  - tool call: `query_suspicion_of_infection`
  - tool call: `query_sofa`
  - final action: `keep_monitoring`
  - final action: `infection_suspect`
  - final action: `trigger_sepsis_alert`

Transition:

- Tool action executes deterministic DuckDB-backed official tool and appends compact output.
- Final action ends the current checkpoint and advances to the next checkpoint.
- Max interactions per checkpoint should be 3 for the core sepsis toolbox:
  - infection call
  - SOFA call
  - final action

Episode:

- One ICU stay trajectory.
- Seven checkpoints from 0 to 24 hours in the current dataset.
- Reward can be assigned per checkpoint plus trajectory-level timing bonuses.

### Reward Design

Keep reward bounded and interpretable.

Per checkpoint:

- `+1.0` correct final action.
- `+0.2` partial credit for `infection_suspect` when ground truth is `trigger_sepsis_alert`, because infection is a necessary intermediate state.
- `-0.2` wrong intermediate action.
- `-0.5` false `trigger_sepsis_alert`.
- `-0.7` missed `trigger_sepsis_alert`.
- `+0.3` `infection_suspect` supported by positive infection evidence.
- `+0.5` `trigger_sepsis_alert` supported by both infection and SOFA alert evidence.
- `-0.7` positive sepsis action without required evidence.
- `+0.15` necessary infection tool called before positive infection decision.
- `+0.20` necessary SOFA tool called before sepsis alert decision or after infection is established and alert status is unresolved.
- `-0.05` redundant same-tool call when state cannot change.
- `-0.03` per tool call after the first useful evidence call.
- `+0.10` valid JSON/schema.
- `-1.0` unavailable tool or invalid schema.

Trajectory-level:

- `+0.5` infection onset predicted at exact checkpoint.
- `+0.5` sepsis alert onset predicted at exact checkpoint.
- `-0.1` per 4-hour delay for infection or alert onset.
- `-0.2` per 4-hour early false alert.
- `-0.5` missed alert for a sepsis-positive stay.

Primary optimization metrics:

- Macro F1.
- Alert F1.
- Infection and alert timing MAE.
- Necessary SOFA-call coverage.
- Unsupported positive-action rate.
- Average tool calls per step.

### Data Splits

Current dataset has 98 stays, so avoid overclaiming.

Suggested split:

- Train: 70 stays.
- Validation: 14 stays.
- Test: 14 stays.

Use stay-level split only. Never split checkpoints from the same stay across train and test.

Because the dataset is small, report:

- Mean and bootstrap confidence intervals over stays.
- Per-class metrics.
- Timing metrics.
- Tool-policy metrics.

### Stage 0: Fix and Freeze the Environment

Before training:

1. Add explicit current-state extraction to the prompt builder.
2. Add tests that render the prompt after a positive infection tool output and assert:
   - the focus says SOFA is the next unresolved question
   - the example tool call is `query_sofa`
3. Add a core-toolbox mode for sepsis with only:
   - `query_suspicion_of_infection`
   - `query_sofa`
4. Keep the original full-toolbox Qwen baseline as the documented baseline result.

### Stage 1: Supervised Warm Start

Before RL, create oracle action traces from the reward-policy proxy.

For each checkpoint:

- If infection unknown, oracle calls `query_suspicion_of_infection`.
- If infection positive and SOFA unknown, oracle calls `query_sofa`.
- Then oracle emits the correct final action from labels and tool evidence.

Train the model to emit the next JSON action.

Recommended model for local warm start:

- Primary: `Qwen/Qwen3-1.7B` with LoRA or QLoRA.
- Reason: 11 GB VRAM is tight for RL on 4B; 1.7B can fit more comfortably and needs formatting/tool-use tuning anyway.
- Backup: `Qwen/Qwen2.5-1.5B-Instruct` if Qwen3 1.7B remains format-unstable.

Training method:

- LoRA SFT with `trl` or a lightweight Hugging Face Trainer.
- Target modules: attention projection and MLP projection modules.
- Precision: fp16, not bf16, because RTX 2080 Ti does not support bf16.
- Batch size: 1-2 per device.
- Gradient accumulation: 16-64.
- Sequence length: start 2048; use compact observations.
- Epochs: 3-10, early stop on validation macro F1 and schema-valid rate.

Why this stage matters:

- RL on a model that cannot reliably emit JSON wastes samples.
- The SFT target teaches the tool/action grammar and the infection-to-SOFA transition.

### Stage 2: RL With Core Sepsis Toolbox

Preferred framework:

- `verl` if it can run with small local actor settings and no mandatory vLLM path that breaks on this GPU.
- Otherwise use `trl` GRPO/PPO as a simpler one-GPU fallback.

Given the 11 GB RTX 2080 Ti, the practical first RL run should use:

- Model: LoRA-adapted `Qwen/Qwen3-1.7B` or `Qwen/Qwen2.5-1.5B-Instruct`.
- Training: LoRA RL, not full fine-tuning.
- Rollout batch size: small, e.g. 4-8 prompts.
- Generation length: 64-128 tokens because outputs should be one JSON object.
- Max checkpoint interactions: 3.
- Tools: two core tools only.
- KL penalty to SFT model: enabled.

RL algorithm:

- Start with GRPO if available.
- PPO is acceptable but more sensitive with tiny batches.
- Use group sampling per state when possible:
  - same observation
  - 4-8 sampled responses
  - reward-normalize within group

Reward:

- Use `src/sepsis_mvp/rl_reward.py` as the first offline evaluator.
- For online RL, refactor the same logic into a per-decision environment reward.

Stopping conditions:

- Validation macro F1 no longer improves.
- Necessary SOFA-call coverage reaches at least 0.8.
- Unsupported positive-action rate below 0.1.
- Schema-valid output rate above 0.98.

### Stage 3: Distinguish Tool Efficiency From Tool Exhaustion

The reward-policy proxy calls tools often:

- 1.6195 tool calls per step.
- 0.0 steps without tools.

That proves evidence coverage, but it is not yet efficient enough. After the model learns the correct sequence, increase pressure against redundant calls:

- Penalize repeated infection calls after a prior positive infection.
- Penalize SOFA calls when prior max SOFA is already at least 2 and sepsis alert has already been triggered.
- Reward direct final action when prior state is sufficient.

Target:

- Keep necessary SOFA coverage high.
- Lower average tool calls per step toward 0.8-1.1.
- Preserve alert F1 and timing.

### Stage 4: Expand to Full Toolbox

After core sepsis works:

1. Add contextual tools:
   - `query_vitalsign`
   - `query_bg`
   - `query_gcs`
   - `query_vasoactive_agent`
2. Keep infection and SOFA as the only required tools for the benchmark label.
3. Add reward only when contextual tools add marginal information.
4. Penalize distracting contextual calls that do not affect the final decision.

This tests whether RL learns selective tool use instead of exhaustive checking.

### Stage 5: Larger Model or Multi-GPU Training

If more GPU memory becomes available:

- Move to `Qwen/Qwen3-4B-Instruct-2507` LoRA RL.
- Use the current Qwen 4B prompt baseline as the direct comparison.
- Consider vLLM rollout acceleration if the hardware supports it cleanly.

On the current RTX 2080 Ti:

- Full 4B RL is not recommended.
- 4B inference works, but training with optimizer states and rollouts is likely too tight without aggressive quantization/offload.
- QLoRA may fit but rollout throughput will be slow and fragility will be high.

### Stage 6: Final Evaluation

Evaluate on held-out stays only.

Compare:

1. No-tool prompt-card ablation.
2. Qwen 4B prompt-driven tool-calling baseline.
3. SFT-only small model.
4. RL-trained small model.
5. Reward-policy proxy upper-bound/reference.

Report:

- Accuracy and macro F1.
- Per-class F1.
- Infection timing exact and MAE.
- Alert timing exact and MAE.
- Tool call counts by tool.
- Necessary infection and SOFA coverage.
- Unsupported positive-action rate.
- Repeated-call rate.
- Runtime and tokens.

## Immediate Next Implementation Tasks

Current readiness:

- Existing code can reproduce prompt-driven tool-calling and reward-policy proxy runs.
- Existing code can score saved rollouts with the RL reward function.
- Existing code can now create splits, export SFT traces, train a LoRA SFT warm start, run lightweight grouped policy-gradient RL on exported tool-call states, and evaluate a trained LoRA adapter on held-out stays.
- The lightweight trainer is not the final solution. Full online `verl` Agent Loop + GRPO over live multi-turn tool episodes is the target path and is documented in `verl_final_solution.md`.
- See `training_readiness.md` for exact existing CLI and `verl_final_solution.md` for the final training architecture.

1. Add a prompt-state analyzer that derives:
   - `current_or_prior_infection_positive`
   - `current_or_prior_sofa_assessed`
   - `current_or_prior_sofa_alert`
2. Patch `_single_task_prompt_focus` or `_build_toolbox_messages` to use current-step history when selecting the focus note and example tool.
3. Add unit tests for the positive-infection-then-SOFA prompt transition.
4. Add `--tool-scope sepsis_core|shared` so RL can start with two tools before using the full 11-tool shared toolbox.
5. Generate oracle SFT traces from the reward-policy proxy.
6. Add a minimal LoRA SFT script.
7. Add a minimal one-GPU GRPO/PPO training script after SFT output is stable.
