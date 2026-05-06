# RL Tool-Calling Experiments For Healthcare Diagnosis

## Status

Initial implementation started.

Implemented so far:

- direct prompt-card no-tool ablation runner
- prompted Qwen tool-calling baseline through the existing `run` command
- offline RL reward scorer for saved sepsis rollouts
- single-GPU default for local Qwen loading

Implemented after the initial baseline/proxy pass:

- train/validation/test split CLI
- sepsis core tool-scope CLI
- SFT warm start export
- LoRA SFT entrypoint
- lightweight grouped policy-gradient RL entrypoint
- LoRA checkpoint evaluation CLI

Not implemented yet:

- verl environment adapter
- full online PPO/GRPO training loop over live multi-turn episodes

Current readiness note:

- The benchmark, official tools, rollout runner, prompt-tool baseline, reward-policy proxy, and offline reward scorer are ready.
- Lightweight LoRA RL training is ready only as a smoke/ablation run. The final target is `verl` Agent Loop + GRPO. See `verl_final_solution.md` for the paper-quality training plan.

Database target:

- `/home/chloeliu/rolling_healthcare_agent/mimic/mimic4_dk.db`

Existing benchmark assets this design builds on:

- `src/sepsis_mvp/agent.py`
- `src/sepsis_mvp/environment.py`
- `src/sepsis_mvp/tools.py`
- `src/sepsis_mvp/schemas.py`
- `docs/sepsis_longitudinal_toolbox_design.md`
- `docs/official_single_vs_toolbox_history_qwen3_30b.md`

## Executive Decision

Use **single-task rolling sepsis escalation** as the first RL tool-calling task.

The best first task is not infection-only, antibiotic prescription extraction, or isolated SOFA scoring. Those are useful subproblems, but each is too narrow to stress a tool-calling policy:

- Infection-only mostly tests one evidence lookup: antibiotic-culture suspicion of infection.
- Antibiotic or prescription detection mostly tests medication retrieval and temporal matching.
- SOFA or organ dysfunction alone mostly tests score/state lookup.
- Sepsis requires a staged diagnostic policy: detect suspected infection, then check whether organ dysfunction reaches alert-level severity.

Sepsis is therefore the strongest first target because it naturally rewards:

- choosing the right evidence tool at the right time
- avoiding redundant infection checks after infection is already established
- checking SOFA when infection is known but sepsis alert status is unresolved
- preserving longitudinal history across checkpoints
- balancing early alerting against false or unsupported escalation

Recommended first label space:

```json
["keep_monitoring", "infection_suspect", "trigger_sepsis_alert"]
```

Recommended checkpoint horizon:

- ICU stay checkpoints every 4 hours
- horizon: 0 to 24 hours
- one trajectory per ICU stay

## Research Question

Can RL improve a Qwen function-calling diagnosis agent so that it is more accurate, better timed, and more tool-efficient than a direct prompt baseline?

The important comparison is not only accuracy. The improved agent should also show better clinical evidence acquisition behavior:

- fewer low-value calls
- fewer repeated calls
- better coverage of necessary calls at transition points
- fewer positive decisions without evidence
- improved alert timing

## Experiment 1: Baseline Prompted Tool-Calling Qwen

### Purpose

Measure how well a Qwen model can solve the sepsis task when prompted to use tools, but without RL.

This is the direct prompt baseline requested by the user: the model receives instructions and available function schemas in the prompt, then chooses tool calls and final actions by prompting alone.

The key limitation this baseline is intended to expose is that a direct prompt model can miss or mishandle the **intermediate state**:

- infection evidence can already be visible
- SOFA alert-level organ dysfunction may not be visible yet
- the correct action is then `infection_suspect`, not `keep_monitoring` and not `trigger_sepsis_alert`

The improved RL tool-calling policy should learn to make this intermediate state explicit by first establishing infection and then checking organ dysfunction only when needed.

### Model

Use Qwen as the base model. Keep the model family consistent with existing repo usage:

- default local target: `Qwen/Qwen3.5-9B`
- larger comparison target if available: `Qwen/Qwen3-30B-A3B-Instruct-2507`

The exact model should be recorded in every run artifact.

### Input Contract

Use the existing `rolling_toolbox_with_history` protocol:

- Qwen sees the current checkpoint.
- Qwen sees compact prior rolling history.
- Qwen sees available tool names and tool-use instructions.
- Qwen can emit a tool-call JSON object or final action JSON.
- There is no RL and no learned policy update.

The baseline uses the same tool interface as the RL experiment. This isolates the effect of RL from the effect of simply enabling function calls.

### Output Contract

The model returns exactly one JSON object:

```json
{"action": "keep_monitoring"}
```

Valid values:

- `keep_monitoring`
- `infection_suspect`
- `trigger_sepsis_alert`

### Baseline Metrics

Report:

- step accuracy
- macro F1
- infection transition exact match
- infection transition mean absolute error
- infection transition early/late/missed rates
- sepsis alert exact match
- sepsis alert mean absolute error
- sepsis alert early/late/missed rates
- invalid JSON rate
- abstention or fallback rate, if implemented

Tool-efficiency metrics are meaningful for this baseline because the model is actually calling tools from prompt instructions.

Implemented command:

```bash
CUDA_VISIBLE_DEVICES=0 QWEN_CUDA_DEVICE=0 PYTHONPATH=src python3 -m sepsis_mvp.cli run \
  --db-path /home/chloeliu/rolling_healthcare_agent/mimic/mimic4_dk.db \
  --dataset <single-task-sepsis-dataset> \
  --agent qwen \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --task-mode single \
  --tool-backend official \
  --protocol rolling_toolbox_with_history \
  --rollouts-output data/rl_tool_call/sepsis_prompt_tool_baseline_rollouts.json \
  --evaluation-output data/rl_tool_call/sepsis_prompt_tool_baseline_eval.json \
  --events-output data/rl_tool_call/sepsis_prompt_tool_baseline_events.jsonl
```

No-tool prompt-card ablation command:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli run-prompt-baseline \
  --concepts data/sample_concepts.json \
  --dataset data/sample_trajectories.json \
  --agent rule \
  --sample-size 1
```

The prompt-card ablation intentionally records `tool_calls = 0` because DB queries are used only to build the hidden evidence card. It should not be used as the primary prompted tool-calling baseline.

## Experiment 2: Improved RL Tool-Calling Agent

### Purpose

Train a Qwen policy to decide when to call tools, which tools to call, and when to stop and diagnose.

The goal is not just higher accuracy. The goal is a better longitudinal diagnostic policy.

### Starting Protocol

Use the existing benchmark protocol:

```text
rolling_toolbox_with_history
```

This protocol already supports:

- per-checkpoint interaction loops
- zero or more tool calls before final action
- rolling history
- Qwen function-call style JSON output
- tool-efficiency evaluation

### Tool Surface

Start with the existing sepsis toolbox:

- `query_suspicion_of_infection`
- `query_sofa`
- `query_kdigo_stage`
- `query_ventilation_status`
- `query_urine_output_rate`
- `query_vasoactive_agent`
- `query_vitalsign`
- `query_bg`
- `query_gcs`
- `query_antibiotic`
- `query_invasive_line`

For the first RL run, use a restricted core set unless there is a reason to study distractor tools:

- `query_suspicion_of_infection`
- `query_sofa`
- `query_kdigo_stage`
- `query_ventilation_status`

Rationale:

- `query_suspicion_of_infection` and `query_sofa` are directly required.
- `query_kdigo_stage` and `query_ventilation_status` are clinically plausible organ dysfunction context and useful distractors.
- The larger 11-tool surface is better as a phase-2 stress test after the reward and rollout infrastructure are stable.

### Agent Action Space

At each interaction turn, the policy emits one of:

```json
{"tool_name": "query_sofa", "arguments": {"stay_id": 30000000, "t_hour": 8}}
```

or:

```json
{"action": "trigger_sepsis_alert"}
```

Set a hard cap:

- `max_tool_calls_per_step = 6` for the core 4-tool setup
- `max_tool_calls_per_step = 10` for the full 11-tool setup

If the cap is exceeded, force a final action or assign a truncation penalty.

### RL Framework Recommendation

Use **verl** for the main implementation if the available training environment supports it.

Reasons:

- it is built for LLM RL training
- it supports PPO/GRPO-style workflows used for tool-use and reasoning policies
- it can train Qwen-family models
- it separates rollout generation, reward computation, and policy optimization cleanly

Recommended first algorithm:

- GRPO if available and stable in the local stack
- PPO as fallback

If verl integration becomes too heavy for the first implementation, use a simpler staged path:

1. Generate trajectories with zero-shot Qwen toolbox prompting.
2. Convert high-reward trajectories into supervised fine-tuning data.
3. Train a Qwen LoRA adapter with SFT.
4. Use verl RL on top of that adapter.

This staged path is lower risk than starting PPO/GRPO from a weak tool policy.

### Reward Design

Use a scalar reward per checkpoint plus optional trajectory-level bonuses.

Recommended first reward:

```text
R = R_action + R_timing + R_evidence + R_efficiency + R_format + R_safety
```

#### 1. Action Accuracy Reward

Per checkpoint:

- `+1.0` if predicted action equals ground truth action
- `+0.2` for clinically adjacent partial credit:
  - predicted `infection_suspect` when ground truth is `trigger_sepsis_alert`
  - predicted `trigger_sepsis_alert` when ground truth is `infection_suspect`
- `-0.5` for false positive alert when ground truth is `keep_monitoring`
- `-0.7` for missed alert when ground truth is `trigger_sepsis_alert`

The missed-alert penalty should be larger than a generic classification error because delayed sepsis escalation is clinically important.

#### 2. Transition Timing Reward

At trajectory end:

- `+1.0` if infection onset is predicted at the correct checkpoint
- `+1.0` if sepsis alert onset is predicted at the correct checkpoint
- subtract `0.1 * abs(predicted_hour - gt_hour) / 4` for onset timing error
- `-1.0` for missed sepsis alert when ground truth alert exists
- `-0.5` for early sepsis alert before evidence exists

Keep timing reward bounded so it does not dominate step accuracy.

#### 3. Evidence Sufficiency Reward

Per checkpoint:

- `+0.3` if `infection_suspect` is supported by current or prior infection evidence
- `+0.5` if `trigger_sepsis_alert` is supported by both infection and SOFA alert evidence
- `-0.7` if a positive action is returned without sufficient evidence
- `-0.5` if `trigger_sepsis_alert` is returned without a SOFA check and no prior SOFA alert evidence exists

This maps directly to existing `positive_action_without_sufficient_evidence_rate` and grounding logic.

#### 4. Tool Efficiency Reward

Per tool call:

- `-0.03` base cost per tool call
- `+0.10` if the call has marginal utility
- `-0.10` if the call repeats already established evidence
- `-0.20` for infection calls after infection is already established
- `+0.15` for a necessary infection call at infection transition
- `+0.20` for a necessary SOFA call before correct sepsis alert

Do not make tool cost too high initially. A strong cost too early will train a model that under-calls tools and misses alerts.

#### 5. Format Reward

Per turn:

- `+0.1` valid JSON
- `+0.1` valid tool name or valid final action
- `-0.5` malformed JSON
- `-0.5` invalid tool name
- `-0.5` invalid action

This is useful in early RL because invalid structured output otherwise wastes rollouts.

#### 6. Safety and Leakage Reward

For this derived-tool experiment, leakage risk is lower than raw SQL because the agent only uses named tools. Still enforce:

- `-1.0` for any attempt to call unavailable tools
- `-1.0` for arguments with mismatched `stay_id` or `t_hour`
- `-1.0` for emitting free-text clinical recommendations instead of benchmark action JSON

### Reward Normalization

Keep rewards bounded approximately between `-2` and `+3` per checkpoint.

For trajectory-level training, normalize by number of checkpoints:

```text
trajectory_reward = mean(step_rewards) + bounded_transition_bonus
```

This avoids giving longer trajectories systematically larger absolute rewards.

Implemented offline scoring command:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli score-rollouts \
  --dataset <single-task-sepsis-dataset> \
  --rollouts data/rl_tool_call/sepsis_toolbox_zeroshot_rollouts.json \
  --output data/rl_tool_call/sepsis_toolbox_zeroshot_rewards.json
```

Important: this scorer is designed for tool-calling rollouts. Direct prompt-card baseline rollouts have no visible tool outputs, so evidence-sufficiency reward terms will intentionally penalize unsupported positive actions even if the hidden prompt card contained the evidence. Use standard classification and timing metrics for the prompt-only baseline.

## Dataset Split

Use stay-level splits to avoid leakage across checkpoints from the same patient stay.

Recommended split:

- train: 70 percent of stays
- validation: 15 percent
- test: 15 percent

If the current sepsis dataset has only around 100 trajectories, prefer:

- train: 60 stays
- validation: 20 stays
- test: 20 stays

Keep the test split fixed for all experiments.

Recommended fixed evaluation cohorts:

- full sepsis test split
- balanced subset with roughly equal positive-alert and no-alert stays
- transition-heavy subset where infection or alert begins within the 24h horizon

## Ground Truth

Use the existing official derived concept labels as the primary ground truth:

- suspected infection from `mimiciv_derived.suspicion_of_infection`
- organ dysfunction from `mimiciv_derived.sofa`
- sepsis alert from suspected infection plus SOFA threshold logic already encoded in the benchmark dataset

The MIMIC database should be read-only during rollout and evaluation.

## Evaluation Table

The final report should compare:

| Experiment | Model | Protocol | RL? | Tools? | Step Acc | Macro F1 | Infection Exact | Alert Exact | Alert Missed | Calls/Step | Zero-Call Rate | Repeated Call Rate | Marginal Utility |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No-tool prompt-card ablation | Qwen | prompt_card | no | hidden card only | | | | | | n/a | n/a | n/a | n/a |
| Prompted tool-calling baseline | Qwen | rolling_toolbox_with_history | no | yes | | | | | | | | | |
| RL toolbox agent | Qwen + RL | rolling_toolbox_with_history | yes | yes | | | | | | | | | |

Primary success criteria:

- RL improves alert timing or alert recall without a large false-positive increase.
- RL improves or matches macro F1 over zero-shot toolbox baseline.
- RL reduces `positive_action_without_sufficient_evidence_rate`.
- RL improves `necessary_call_coverage.sofa_for_alert`.
- RL keeps average tool calls per step below the old fixed two-tool baseline.

## Implementation Plan

### Phase 0: Design and Audit

- Confirm available MIMIC tables in `/home/chloeliu/rolling_healthcare_agent/mimic/mimic4_dk.db`.
- Confirm the sepsis rolling dataset path used for training and evaluation.
- Decide whether the first RL tool surface is 4 tools or 11 tools.
- Freeze train/validation/test stay splits.

### Phase 1: Baseline Direct Prompt

Add a prompt-only runner that:

- loads the sepsis trajectories
- builds `patient_checkpoint_card` observations
- calls Qwen once per checkpoint
- records JSON actions
- evaluates with existing single-task metrics

Artifacts:

- `data/rl_tool_call/sepsis_prompt_baseline_rollouts.json`
- `data/rl_tool_call/sepsis_prompt_baseline_eval.json`

### Phase 2: Zero-Shot Toolbox Baseline

Run existing Qwen agent with:

- task: single sepsis
- backend: official
- protocol: `rolling_toolbox_with_history`
- same test split as baseline

Artifacts:

- `data/rl_tool_call/sepsis_toolbox_zeroshot_rollouts.json`
- `data/rl_tool_call/sepsis_toolbox_zeroshot_eval.json`

### Phase 3: RL Environment Adapter

Create a thin RL adapter around `BenchmarkEnvironment`:

- `reset(trajectory_id, step_index)`
- `observe()`
- `step(model_json)`
- `execute_tool(tool_call)`
- `finalize(action)`
- `compute_reward(step_record, prior_state, gt)`

The adapter should reuse existing tool runtimes and evaluation code rather than duplicating clinical logic.

### Phase 4: Offline Warm Start

Before RL:

- collect zero-shot toolbox rollouts
- score every step with the reward function
- create SFT examples from high-reward interactions
- fine-tune a Qwen LoRA adapter to stabilize JSON and tool-use format

This phase is optional but recommended.

### Phase 5: verl RL

Train with verl:

- initialize from base Qwen or SFT adapter
- generate rollouts against the environment adapter
- compute reward from action correctness, timing, evidence, and efficiency
- update policy with GRPO/PPO
- validate every fixed number of updates on held-out stays

Stop based on validation composite score, not training reward.

### Phase 6: Final Evaluation

Evaluate on the frozen test split:

- no-tool prompt-card ablation
- prompted tool-calling baseline
- RL toolbox agent

Also run qualitative bad-case analysis:

- false early alerts
- missed alerts
- infection transition misses
- repeated low-utility calls
- unsupported positive decisions

## Composite Model Selection Metric

Use validation score:

```text
score =
  0.35 * macro_f1
+ 0.20 * alert_exact_match
- 0.15 * alert_missed_rate
+ 0.10 * infection_exact_match
+ 0.10 * necessary_sofa_call_coverage
+ 0.05 * marginal_utility_of_call_rate
- 0.05 * repeated_tool_call_rate
- 0.05 * positive_action_without_sufficient_evidence_rate
```

This keeps clinical correctness primary while still selecting for tool-use quality.

## Why Not Start With Other Tasks?

### Infection-Only

Good warmup, weak main task.

It is useful for validating:

- JSON format
- infection evidence lookup
- transition timing reward

But it mostly requires one tool, so RL may learn a trivial policy: call infection tool once or reuse positive history.

### Antibiotic or Prescription Detection

Good auxiliary extraction task, not a full diagnosis task.

It is valuable for pretraining or debugging infection evidence, but the final output is too close to database retrieval.

### Organ Dysfunction or SOFA Alone

Good scoring task, incomplete diagnostic task.

It tests severity recognition but does not require combining infection and dysfunction evidence.

### AKI Non-Monotonic Current State

Strong phase-2 task.

It is useful because AKI can improve or worsen, so the agent cannot treat all positive states as permanent. However, the first RL experiment should use sepsis because existing reports already identify a concrete tool-policy failure: the toolbox agent improves timing and efficiency but misses too many final alerts.

## Open Questions Before Implementation

1. Which exact Qwen checkpoint should be the canonical baseline model?
2. Is the first RL run allowed to use only the 4-tool core toolbox, or should it include the full 11-tool toolbox from the start?
3. Where is the canonical sepsis rolling CSV in this workspace?
4. Should the prompt-only baseline receive compact derived evidence cards, or should it receive only text summaries generated from raw table snippets?
5. What compute budget is available for verl training?

## Recommended First Implementation Scope

Implement only:

- single-task sepsis
- official tool backend
- 4-tool core toolbox
- Qwen direct-prompt baseline
- Qwen zero-shot toolbox baseline
- reward computation and offline scoring

Then implement verl training after the reward can score saved rollouts deterministically.
