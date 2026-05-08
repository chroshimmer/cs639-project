# Diagnosing and Improving Long-Horizon OS Agents

This repository is a CS639 course project built on the function-calling version of [THUDM/AgentBench](https://github.com/THUDM/AgentBench).  We focus on **OS interaction tasks** and study whether a lightweight inference-time monitor can improve LLM agents on both standard OS tasks and long-horizon OS tasks.

The main system in this repo is **monitor-replan**: a rule-based, trace-aware prompting pipeline that sits on top of the same acting model.  It does **not** train a new model.  Instead, it watches the task trajectory, detects failure risks, and injects targeted supervision only when needed.

This README documents the main monitor-replan project only.  It does not cover the separate project extensions.

---

## Project summary

### Problem

LLM agents can solve some short OS tasks, but their success rate often drops as interaction horizons grow.  Longer trajectories make it harder to preserve the task goal, maintain state, verify changes, and avoid loops or premature termination.

### Goal

We evaluate whether an inference-time monitor can help OS agents:

- stay grounded in shell evidence,
- avoid planning drift,
- recover from repeated or weak-signal actions,
- verify state changes before finishing,
- and remain robust as tasks become more long-horizon.

### Main result

Across the models we tested, the monitor improves performance on both the original AgentBench OS benchmark and a reconstructed HORIZON-style safe subset.

| Benchmark | Models tested | Baseline -> w/ Monitor | Gain range |
|---|---:|---:|---:|
| Original AgentBench OS | 5 models | standard 144-task benchmark | +1.4 to +21.6 pts |
| HORIZON-style safe subset | 3 models | 146-task long-horizon subset | +15.8 to +35.6 pts |

The largest gains occur on weaker or mid-tier base models and on long-horizon tasks, suggesting that monitor-replan is most useful when the base agent is likely to lose task focus or skip verification.

---

## Benchmarks

### 1. Original AgentBench OS

The standard benchmark is the public AgentBench FC OS task set used by the project baseline.

- Task name: `os-std`
- Monitored task name: `os-std-monitor-replan`
- Size: 144 OS tasks
- Metric: success rate reported by `agentrl-eval`
- Environment: containerized OS interaction environment

### 2. HORIZON-style safe subset

We also evaluate on a conservative long-horizon subset reconstructed from the public HORIZON OS release.

The public HORIZON parquet contains prompts / trajectories, but it does not directly provide runnable AgentBench-style OS initialization and evaluation configs.  To keep automatic evaluation valid, we only keep augmented prompts whose final objective remains compatible with their own AgentBench base task.

- Task name: `os-horizon-all`
- Monitored task name: `os-horizon-all-monitor-replan`
- Size: 146 tasks
  - `cf0`: 46 base tasks
  - `cf1`-`cf10`: 10 compatible augmented tasks per tier
- Important caveat: this is a **HORIZON-style safe subset**, not the official HORIZON OS evaluator.  The original AgentBench final evaluator is reused, so final-task compatibility is checked, but the added HORIZON intermediate constraints are not fully oracle-checked.

Per-tier task names are also available:

```text
os-horizon-cf0 ... os-horizon-cf10
os-horizon-cf0-monitor-replan ... os-horizon-cf10-monitor-replan
```

`cf0` is the unaugmented HORIZON base subset derived from AgentBench.  `cf1` is the first augmented long-horizon level, and `cf10` is the longest level used here.

---

## Method: monitor-replan

The monitor is an inference-time control layer over the same base model.  The base model still acts through AgentBench's normal tools, but the monitor maintains a compact trajectory state and decides whether to inject extra guidance.

### High-level pipeline

```text
Task + history
    -> mode inference
    -> trace monitor
    -> typed intervention
    -> next model action
```

The monitor uses deterministic heuristics, not a learned classifier or a second LLM.

### 1. Task + trajectory state

For each task, the monitor keeps a compact background state containing:

- the original task goal,
- evaluation type,
- recent shell commands and observations,
- command type: inspection / mutation / mixed / unknown,
- error tags and weak-signal indicators,
- known target paths,
- blocked commands,
- intervention history,
- last mutation round,
- last verification round.

This state is not directly shown to the model at every turn.  When the monitor decides intervention is needed, part of this state is summarized into a short working plan or recovery prompt.

### 2. Mode inference

The monitor first classifies the task into one of three modes:

| Mode | Typical task | Desired behavior |
|---|---|---|
| `answer` | count, path, yes/no, output, file content | use short read-only commands, then call `answer_action` |
| `state` | chmod, rename, modify, create, delete, install | mutate minimally, verify, then finish |
| `hybrid` | implement or modify something, then report a result | implement -> verify -> answer |

This is a rule-based keyword heuristic.  For example, phrases like `how many`, `count`, `what is`, `full path`, and `yes or no` suggest answer-mode tasks.  Phrases like `chmod`, `rename`, `make writable`, `delete`, `modify`, or `install` suggest state-change tasks.  If both kinds of signals appear, the task is treated as hybrid.

### 3. Trace monitoring

After each model action, the monitor checks whether the trajectory is likely to be going wrong.  It detects patterns such as:

- repeated commands or repeated observations,
- broad search that does not reduce uncertainty,
- large multi-line bash scripts issued too early,
- weak or empty shell evidence,
- environment mutation during a pure answer task,
- mutation without follow-up verification,
- answer or finish attempts without sufficient grounding,
- tool-call protocol problems.

### 4. Typed intervention

When risk is detected, the monitor injects targeted supervision.  The intervention depends on the failure type.

| Failure pattern | Intervention |
|---|---|
| Answer/query drift | read-only hint, direct evidence gathering, use `answer_action` |
| State mutation without verification | require explicit verification before finish |
| Hybrid task confusion | enforce implement -> verify -> answer sequencing |
| Loop or stall | stop repeating command; choose a new direct check |
| Oversized script | block brittle multi-line bash; request one short command first |
| Weak grounding | gather direct evidence before answering |
| Unsafe finish / answer | commit gate blocks termination |

The monitor is intentionally selective.  It does not inject a plan every turn; it intervenes only when the trajectory shows clear risk.

---

## Key files

| File | Purpose |
|---|---|
| `src/server/tasks/os_interaction/monitoring.py` | monitor state, task typing, trace analysis, recovery prompts, commit gate, oversized-script guard |
| `src/server/tasks/os_interaction/task.py` | OS task worker, task loading, initial hint injection, tool handling, trace recording, integration with monitor |
| `configs/tasks/os.yaml` | task entries for baseline, monitor, HORIZON safe subset, and per-tier HORIZON runs |
| `extra/docker-compose.os-only.yml` | controller + redis + OS workers for the standard and HORIZON tasks |
| `data/os_interaction/data_horizon/` | reconstructed HORIZON-style safe subset data |
| `results/` | run logs, `results.jsonl`, and per-task traces |

---

## Setup

These instructions are written for the UW-Madison CSL environment, where rootless Docker is used and sudo is not available.  On a normal Linux machine with rootful Docker, some CSL-specific steps may not be necessary.

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/chroshimmer/cs639-project.git
cd cs639-project

conda create -n cs639-project python=3.11 -y
conda activate cs639-project

pip install -r requirements.txt
pip install -U agentrl-eval openai anthropic
```

### 2. Start Docker on CSL

```bash
systemctl --user start docker.service
```

### 3. Build local OS environment images

```bash
docker build -t local-os/default  -f ./data/os_interaction/res/dockerfiles/default  data/os_interaction/res/dockerfiles
docker build -t local-os/packages -f ./data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles
docker build -t local-os/ubuntu   -f ./data/os_interaction/res/dockerfiles/ubuntu   data/os_interaction/res/dockerfiles
```

### 4. Start the OS-only stack

```bash
docker compose -f extra/docker-compose.os-only.yml up --build -d
```

Sanity check:

```bash
curl -s http://localhost:5020/api/get_tasks | head
```

Check that the worker can start a sample and returns non-empty `messages`:

```bash
curl -s -H 'Content-Type: application/json' \
  -d '{"name":"os-std","index":0,"custom_task":null}' \
  http://localhost:5020/api/start_sample | head -c 300; echo
```

Check the HORIZON worker:

```bash
curl -s "http://localhost:5020/api/get_indices?name=os-horizon-all" | head -c 300; echo
```

If `start_sample` returns an empty `messages` array, verify that `configs/tasks/os.yaml` places the Redis settings under `env_options`, for example:

```yaml
env_options:
  network_name: agentbench-fc_default
  state_driver: redis
  state_options:
    connection:
      host: redis
```

---

## Running evaluations

Set the API key expected by `agentrl-eval`:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
```

### Original AgentBench OS

Baseline:

```bash
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5-mini \
  --concurrency 4 \
  -n 1 \
  -o results \
  os-std
```

Monitor-replan:

```bash
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5-mini \
  --concurrency 4 \
  -n 1 \
  -o results \
  os-std-monitor-replan
```

### HORIZON-style safe subset

Baseline:

```bash
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5-mini \
  --concurrency 4 \
  -n 1 \
  -o results/horizon_baseline \
  os-horizon-all
```

Monitor-replan:

```bash
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5-mini \
  --concurrency 4 \
  -n 1 \
  -o results/horizon_monitor \
  os-horizon-all-monitor-replan
```

Run a single tier:

```bash
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5-mini \
  --concurrency 2 \
  -n 1 \
  -o results/horizon_cf10_baseline \
  os-horizon-cf10
```

---

## Optional: Gemini / Vertex AI runs

The Gemini experiments were run through Vertex AI's OpenAI-compatible endpoint.  Because `agentrl-eval` uses an OpenAI-style client, the Google OAuth access token is passed through the `OPENAI_API_KEY` environment variable.

```bash
export PROJECT_ID="YOUR_GCP_PROJECT"
export LOCATION="global"
export VERTEX_OPENAI_BASE_URL="https://aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/endpoints/openapi"
export OPENAI_API_KEY="$(gcloud auth print-access-token)"

agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u "$VERTEX_OPENAI_BASE_URL" \
  -m google/gemini-3-flash-preview \
  --concurrency 2 \
  -n 1 \
  -o results/gemini3_flash_os_std \
  os-std
```

The access token from `gcloud auth print-access-token` is short-lived.  For long overnight runs, split evaluation by index ranges and refresh the token before each segment.

---

## Results

### Original AgentBench OS: 144 tasks

| Model | Baseline | w/ Monitor | Gain |
|---|---:|---:|---:|
| GPT-5-mini | 28.7% | 50.3% | +21.6 pts |
| GPT-5.4 | 48.6% | 56.2% | +7.6 pts |
| GPT-5.4-pro | 48.6% | 50.0% | +1.4 pts |
| Gemini 3 Flash | 24.3% | 36.1% | +11.8 pts |
| Gemini 3.1 Pro | 45.1% | 55.6% | +10.4 pts |

Takeaway: the monitor improves every tested model on the original benchmark.  The largest absolute gain is on GPT-5-mini, suggesting that the monitor helps most when the base model is weaker.

### HORIZON-style safe subset: 146 tasks

| Model | Baseline | w/ Monitor | Gain |
|---|---:|---:|---:|
| GPT-5-mini | 50.0% | 85.6% | +35.6 pts |
| Gemini 3 Flash | 42.5% | 67.8% | +25.3 pts |
| Gemini 3.1 Pro | 76.7% | 92.5% | +15.8 pts |

Takeaway: monitor-replan improves all three models on long-horizon tasks.  Gains are largest for weaker or mid-tier models, but even Gemini 3.1 Pro improves substantially.

### Tier-wise HORIZON results

Each value is success rate on that tier.  `cf0` has 46 tasks; `cf1`-`cf10` have 10 tasks each, so tier-level fluctuations should be interpreted cautiously.

| Tier | GPT-5-mini Base | GPT-5-mini Mon | Gemini Flash Base | Gemini Flash Mon | Gemini 3.1 Pro Base | Gemini 3.1 Pro Mon |
|---|---:|---:|---:|---:|---:|---:|
| cf0 | 69.6% | 76.1% | 65.2% | 69.6% | 84.8% | 87.0% |
| cf1 | 70.0% | 90.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| cf2 | 40.0% | 90.0% | 40.0% | 80.0% | 80.0% | 100.0% |
| cf3 | 30.0% | 100.0% | 30.0% | 90.0% | 60.0% | 90.0% |
| cf4 | 30.0% | 80.0% | 40.0% | 50.0% | 70.0% | 90.0% |
| cf5 | 40.0% | 90.0% | 60.0% | 80.0% | 70.0% | 100.0% |
| cf6 | 50.0% | 100.0% | 0.0% | 70.0% | 50.0% | 100.0% |
| cf7 | 30.0% | 100.0% | 20.0% | 50.0% | 70.0% | 100.0% |
| cf8 | 40.0% | 90.0% | 50.0% | 70.0% | 100.0% | 80.0% |
| cf9 | 40.0% | 80.0% | 10.0% | 60.0% | 70.0% | 100.0% |
| cf10 | 40.0% | 80.0% | 30.0% | 40.0% | 60.0% | 90.0% |

---

## Failure reason analysis

We also analyzed original AgentBench OS failures using a HORIZON-style failure taxonomy.

Key observations:

- **Planning errors decrease with the monitor.**  Planning error drops for all OpenAI models in the failure-reason analysis: GPT-5-mini drops from roughly 34% to 22%, GPT-5.4 from 19% to 7%, and GPT-5.4-pro from 13% to 6%.
- **Ill-defined instructions remain a bottleneck.**  Stronger models fail less from execution/planning issues and more from ambiguous task interpretation, which the current monitor does not fully solve.
- **Stronger models fail differently.**  For weaker models, the monitor mainly reduces execution and planning drift.  For stronger models, the remaining failures are more often tied to task ambiguity or benchmark wording.

---

## Output format

Each `agentrl-eval` run writes a timestamped directory under the specified output directory.  Typical contents include:

```text
results/<run-name>/
  run.log
  results.jsonl
  <index>-<run>-<session>-<timestamp>/trace.json
```

For monitor-replan runs, traces may include injected monitor messages such as:

- initial task-mode hints,
- `[WORKING PLAN]`,
- `[Monitor]` recovery prompts,
- commit-gate messages.

---

## Limitations

- The monitor is heuristic and hand-designed.  It does not learn when to intervene.
- Task-mode inference and trajectory-risk detection are rule-based, not LLM-judged.
- The HORIZON-style safe subset is not the official HORIZON OS evaluator.
- The safe subset preserves final-task compatibility with AgentBench evaluation, but does not fully oracle-check all HORIZON-added intermediate constraints.
- `cf1`-`cf10` each contain only 10 tasks, so per-tier success rates have high variance.
- Some failures still require deeper long-horizon replanning than a lightweight prompt-layer monitor can provide.

---

## Current takeaway

Monitor-replan acts as a lightweight trajectory controller for OS agents.  It improves standard AgentBench OS performance across all tested models and gives larger gains on the HORIZON-style long-horizon safe subset.  The main contribution is not a new base model, but a targeted inference-time control layer that keeps the agent grounded, phase-aware, and less likely to finish or answer without evidence.
