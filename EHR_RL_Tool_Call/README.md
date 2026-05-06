# Rolling ICU Surveillance MVP

This repo contains a rolling monitoring benchmark pipeline on MIMIC-IV concept-layer data.

Implemented task modes:

- single-task sepsis escalation
- single-task AKI current-state tracking
- multi-task escalation for:
  - sepsis
  - AKI
  - respiratory support

The agent never sees raw vitals, labs, meds, or procedures directly. It only interacts with derived concept tools.

## Current datasets

Packaged datasets live under [/Users/chloe/Documents/New project/rolling_monitor_dataset](/Users/chloe/Documents/New project/rolling_monitor_dataset):

- [/Users/chloe/Documents/New project/rolling_monitor_dataset/sepsis](/Users/chloe/Documents/New project/rolling_monitor_dataset/sepsis)
- [/Users/chloe/Documents/New project/rolling_monitor_dataset/aki](/Users/chloe/Documents/New project/rolling_monitor_dataset/aki)
- [/Users/chloe/Documents/New project/rolling_monitor_dataset/aki_non_monotonic](/Users/chloe/Documents/New project/rolling_monitor_dataset/aki_non_monotonic)
- [/Users/chloe/Documents/New project/rolling_monitor_dataset/respiratory_support](/Users/chloe/Documents/New project/rolling_monitor_dataset/respiratory_support)
- [/Users/chloe/Documents/New project/rolling_monitor_dataset/multitask](/Users/chloe/Documents/New project/rolling_monitor_dataset/multitask)

The main shared multi-task cohort is:

- [/Users/chloe/Documents/New project/rolling_monitor_dataset/multitask/rolling_multitask.csv](/Users/chloe/Documents/New project/rolling_monitor_dataset/multitask/rolling_multitask.csv)
- [/Users/chloe/Documents/New project/rolling_monitor_dataset/multitask/trajectory_schema.json](/Users/chloe/Documents/New project/rolling_monitor_dataset/multitask/trajectory_schema.json)

## Tool layer

The pipeline now supports three tool backends:

- `official`: current DuckDB wrappers over MIMIC derived concepts
- `autoformalized`: generated Python concept functions from [/Users/chloe/Documents/New project/autoformalized_library](/Users/chloe/Documents/New project/autoformalized_library)
- `zeroshot_raw`: checkpoint-scoped raw-table Python execution against MIMIC-IV without `mimiciv_derived`

The derived-concept backends expose the same tool names:

- `query_suspicion_of_infection`
- `query_sofa`
- `query_kdigo_stage`
- `query_ventilation_status`

For the `official` backend, these are backed by:

- `mimiciv_derived.suspicion_of_infection`
- `mimiciv_derived.sofa`
- `mimiciv_derived.kdigo_stages`
- `mimiciv_derived.ventilation`

For the `zeroshot_raw` backend, the agent receives a checkpoint-scoped Python session with:

- `query_db(sql, params=None)` over raw `mimiciv_icu.*` and `mimiciv_hosp.*` views
- raw sepsis guidance from [/Users/chloe/Documents/New project/baseline/sepsis_guideline.yaml](/Users/chloe/Documents/New project/baseline/sepsis_guideline.yaml)
- no access to `mimiciv_derived`

## Modes

The runner has two independent mode choices:

- `--task-mode`
  - `auto`: infer from the dataset you pass
  - `single`: require a single-task dataset
  - `multitask`: require a multitask dataset
- `--tool-backend`
  - `official`: visible tools come from official derived DuckDB concepts
  - `autoformalized`: visible tools come from generated functions in `autoformalized_library`
  - `zeroshot_raw`: the model writes Python/SQL against checkpoint-scoped raw MIMIC-IV views

Important:

- `single` does not choose the event type by itself
- the event type comes from the dataset

Examples:

- `rolling_monitor_dataset/sepsis/rolling_sepsis.csv` + `--task-mode single`
  runs single-task sepsis
- `rolling_monitor_dataset/aki/rolling_aki.csv` + `--task-mode single`
  runs single-task AKI
- `rolling_monitor_dataset/aki_non_monotonic/rolling_aki_non_monotonic.csv` + `--task-mode single`
  runs single-task non-monotonic AKI current-state tracking
- `rolling_monitor_dataset/respiratory_support/rolling_respiratory_support.csv` + `--task-mode single`
  runs single-task respiratory support
- `rolling_monitor_dataset/multitask/rolling_multitask.csv` + `--task-mode multitask`
  runs joint sepsis + AKI + respiratory support monitoring

If you use `--task-mode auto`, the pipeline infers this from the dataset format and task metadata.

## Agent contract

Single-task sepsis mode returns:

```json
{"action":"infection_suspect"}
```

Single-task non-monotonic AKI mode returns:

```json
{"action":"aki_stage_2"}
```

Multi-task mode returns:

```json
{
  "task_actions": {
    "sepsis": "infection_suspect",
    "aki": "suspect_aki",
    "respiratory_support": "room_air_or_low_support"
  }
}
```

Tool calls always use:

```json
{"tool_name":"query_kdigo_stage","arguments":{"stay_id":123,"t_hour":8}}
```

## Quick start

### 1. Build trajectories from a CSV package

Single-task sepsis:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli build-dataset \
  --rolling-csv /Users/chloe/Documents/New\ project/rolling_monitor_dataset/sepsis/rolling_sepsis.csv \
  --output data/rolling_sepsis_trajectories.json
```

Shared multi-task:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli build-dataset \
  --rolling-csv /Users/chloe/Documents/New\ project/rolling_monitor_dataset/multitask/rolling_multitask.csv \
  --output data/rolling_multitask_trajectories.json
```

Single-task non-monotonic AKI:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli build-dataset \
  --rolling-csv /Users/chloe/Documents/New\ project/rolling_monitor_dataset/aki_non_monotonic/rolling_aki_non_monotonic.csv \
  --output data/rolling_aki_non_monotonic_trajectories.json
```

### 2. Smoke test with the heuristic agent

This is the recommended pre-GPU check.

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli run \
  --db-path /Users/chloe/Desktop/healthcare/mimic-iv-3.1/buildmimic/duckdb/mimic4_dk.db \
  --dataset /Users/chloe/Documents/New\ project/rolling_monitor_dataset/multitask/rolling_multitask.csv \
  --agent heuristic \
  --task-mode multitask \
  --tool-backend official \
  --sample-size 5 \
  --events-output data/multitask_events.jsonl \
  --trajectory-output data/multitask_trajectories.jsonl \
  --rollouts-output data/multitask_rollouts.json \
  --evaluation-output data/multitask_eval.json
```

To resume a long run without reprocessing completed stays, reuse the same output paths and add `--resume`:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli run \
  --db-path /Users/chloe/Desktop/healthcare/mimic-iv-3.1/buildmimic/duckdb/mimic4_dk.db \
  --dataset /Users/chloe/Documents/New\ project/rolling_monitor_dataset/multitask/rolling_multitask.csv \
  --agent qwen \
  --task-mode multitask \
  --tool-backend official \
  --trajectory-output data/qwen_multitask_trajectories.jsonl \
  --rollouts-output data/qwen_multitask_rollouts.json \
  --evaluation-output data/qwen_multitask_eval.json \
  --resume
```

`--resume` checks existing completed rollouts in `--trajectory-output` first, then `--rollouts-output`, skips any matching `trajectory_id`s in the current dataset, and evaluates on the combined old-plus-new set.

Single-task sepsis smoke test:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli run \
  --concepts data/sample_concepts.json \
  --dataset data/sample_trajectories.json \
  --agent heuristic \
  --task-mode single \
  --tool-backend official \
  --sample-size 1 \
  --evaluation-output data/sample_eval.json
```

Single-task non-monotonic AKI smoke test:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli run \
  --db-path /Users/chloe/Desktop/healthcare/mimic-iv-3.1/buildmimic/duckdb/mimic4_dk.db \
  --dataset /Users/chloe/Documents/New\ project/rolling_monitor_dataset/aki_non_monotonic/rolling_aki_non_monotonic.csv \
  --agent heuristic \
  --task-mode single \
  --tool-backend official \
  --sample-size 3 \
  --events-output data/aki_non_monotonic_events.jsonl \
  --trajectory-output data/aki_non_monotonic_trajectories.jsonl \
  --rollouts-output data/aki_non_monotonic_rollouts.json \
  --evaluation-output data/aki_non_monotonic_eval.json
```

### 3. Run the local Qwen model

```bash
export QWEN_MODEL="Qwen/Qwen3.5-9B"
export QWEN_OFFLINE=0

PYTHONPATH=src python3 -m sepsis_mvp.cli run \
  --db-path /Users/chloe/Desktop/healthcare/mimic-iv-3.1/buildmimic/duckdb/mimic4_dk.db \
  --dataset /Users/chloe/Documents/New\ project/rolling_monitor_dataset/multitask/rolling_multitask.csv \
  --agent qwen \
  --task-mode multitask \
  --tool-backend official \
  --model Qwen/Qwen3.5-9B \
  --temperature 0.0 \
  --top-p 0.95 \
  --max-new-tokens 250 \
  --sample-size 10 \
  --events-output data/qwen_multitask_events.jsonl \
  --trajectory-output data/qwen_multitask_trajectories.jsonl \
  --rollouts-output data/qwen_multitask_rollouts.json \
  --evaluation-output data/qwen_multitask_eval.json
```

To run the same benchmark with generated concept functions instead of official derived tables:

```bash
PYTHONPATH=src python3 -m sepsis_mvp.cli run \
  --db-path /Users/chloe/Desktop/healthcare/mimic-iv-3.1/buildmimic/duckdb/mimic4_dk.db \
  --dataset /Users/chloe/Documents/New\ project/rolling_monitor_dataset/multitask/rolling_multitask.csv \
  --agent qwen \
  --task-mode multitask \
  --tool-backend autoformalized \
  --autoformalized-library /Users/chloe/Documents/New\ project/autoformalized_library \
  --model Qwen/Qwen3.5-9B \
  --sample-size 10 \
  --events-output data/qwen_multitask_autoform_events.jsonl \
  --trajectory-output data/qwen_multitask_autoform_trajectories.jsonl \
  --rollouts-output data/qwen_multitask_autoform_rollouts.json \
  --evaluation-output data/qwen_multitask_autoform_eval.json
```

## Debug outputs

Useful flags:

- `--sample-size N`: run only the first `N` trajectories
- `--resume`: skip already-completed `trajectory_id`s found in the existing rollout files and continue with the remaining trajectories
- `--task-mode auto|single|multitask`: validate the dataset against a requested task layout
- `--tool-backend official|autoformalized`: choose the visible concept backend
- `--autoformalized-library path`: root folder for generated functions when using the autoformalized backend
- `--events-output path.jsonl`: append every step start, tool call, tool output, action, and trajectory completion
- `--events-output path.jsonl` also captures raw Qwen outputs, repair outputs, and any controller-forced tool corrections
- `--trajectory-output path.jsonl`: append each completed stay rollout immediately
- `--rollouts-output path.json`: write the final full in-memory rollout list at the end
- `--evaluation-output path.json`: save the final evaluation summary with task mode, backend, sample size, and metrics

Resume behavior:

- `--trajectory-output` is the preferred checkpoint source because it is appended after every completed stay
- if both `--trajectory-output` and `--rollouts-output` exist, the runner uses the trajectory JSONL first and fills any missing IDs from the final rollouts JSON
- if `--resume` is set and all sampled trajectories are already complete, the runner skips model execution and just writes the merged evaluation summary

This means partial progress survives long runs and crashes.

For single-task runs, the saved metrics are task-specific and detailed:

- sepsis:
  - `transition_timing.infection`
  - `transition_timing.sepsis_alert`
  - `tool_grounding.infection_predictions_grounded_rate`
  - `tool_grounding.alert_predictions_grounded_rate`
- AKI:
  - `transition_timing.aki_suspect`
  - `transition_timing.aki_alert`
  - `tool_grounding.suspect_predictions_grounded_rate`
  - `tool_grounding.alert_predictions_grounded_rate`
- non-monotonic AKI:
  - `step_level` over `no_aki`, `aki_stage_1`, `aki_stage_2`, `aki_stage_3`
  - `state_change.worsening`
  - `state_change.recovery`
  - `state_change.exact_path_match_rate`
  - `tool_grounding.stage1_predictions_grounded_rate`
  - `tool_grounding.stage2_predictions_grounded_rate`
  - `tool_grounding.stage3_predictions_grounded_rate`
- respiratory support:
  - `transition_timing.medium_support`
  - `transition_timing.invasive_support`
  - `tool_grounding.medium_support_predictions_grounded_rate`
  - `tool_grounding.invasive_support_predictions_grounded_rate`

## Prompting notes

The local Qwen path is prompt-tuned for strict JSON output:

- no free-text reasoning
- no `<think>` tags
- single-task and multitask prompts are now generated separately
- both prompt modes share a basic clinical guidance block:
  sepsis uses infection evidence plus SOFA>=2 as alert-level guidance,
  AKI uses KDIGO stage 1 vs stage>=2,
  respiratory support uses low vs HFNC/NIV vs invasive/trach mapping
- fixed-key `task_actions` object in multitask mode
- one-shot repair retry if the first generation is not valid JSON
- deterministic required-tool guard based on the monitored task set
- if Qwen skips a required tool, repeats a previous tool, or keeps calling tools after all required tools are done, the controller repairs or corrects the step instead of silently falling back to baseline labels

## Query checks

Live query checks were run successfully against the DuckDB for:

- one stay positive for all three tasks: `30294009`
- one stay negative for all three tasks: `30009797`

Observed behavior matched expectations:

- pre-ICU infection can already be visible at `t_hour=0`
- SOFA, KDIGO, and ventilation are properly time-gated by checkpoint
- the respiratory wrapper reports both current and highest support seen so far

## Verification

Local verification completed with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

and a CLI smoke run on the refactored official single-task path.
