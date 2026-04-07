# CS639 Project: Long-Horizon OS Tasks

This repo/project is based on **THUDM/AgentBench (AgentBench FC / function-calling version)**.  
This course project focuses on **OS interaction tasks**, specifically Long-Horizon OS tasks.

> Note: This setup was made to run on **UW–Madison CSL machines** (rootless Docker, no sudo).  
> Some changes below may be **CSL-specific** and not necessary on a normal local Linux + rootful Docker setup.

---

## 1) What changed (CSL-specific)

Some edits were made to make AgentBench FC (OS) runnable on CSL machines:

- **Edited Dockerfiles for OS env images**
  - `data/os_interaction/res/dockerfiles/default`
  - `data/os_interaction/res/dockerfiles/packages`  
  (Goal: avoid Ubuntu base image install / permission issues under rootless Docker; use a more stable base + non-interactive installs.)

- **Edited OS task config (Redis host)**
  - `configs/tasks/os.yaml`  
  Changed Redis host from a hard-coded IP (e.g., `172.17.0.1`) to the compose service name **`redis`** for container-to-container networking.

- **Edited OS task worker image Dockerfile**
  - `src/server/tasks/os_interaction/Dockerfile`  
  (Goal: make the OS task worker container compatible with CSL rootless Docker constraints.)

- **Added an OS-only docker compose file**
  - `extra/docker-compose.os-only.yml`  
  (Goal: start only what we need for OS tasks: controller + redis + `os_interaction-std` worker.)

- **Added a lightweight monitor + replanner path for OS tasks**
  - `src/server/tasks/os_interaction/monitoring.py`
  - `src/server/tasks/os_interaction/task.py`
  - `configs/tasks/os.yaml`  
  (Goal: keep the original `os-std` baseline intact, while adding `os-std-monitor-replan` with state memory, loop/stall detection, commit gating, and recovery prompts.)

These edits are intended for CSL machines; on other systems you may not need them.

---

## 2) Quick Start (on a  CSL machine)

### Step 0 — Clone
```bash
git clone https://github.com/chroshimmer/cs639-project.git
cd cs639-project
```

### Step 1 — Python env + install deps
```bash
conda create -n cs639-project python=3.11 -y
conda activate cs639-project

pip install -r requirements.txt
pip install -U agentrl-eval openai anthropic
```

### Step 2 — Start rootless Docker (CSL)
```bash
systemctl --user start docker.service
```

### Step 3 — Build OS env images (local-os/*)
```bash
docker build -t local-os/default  -f ./data/os_interaction/res/dockerfiles/default  data/os_interaction/res/dockerfiles
docker build -t local-os/packages -f ./data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles
docker build -t local-os/ubuntu   -f ./data/os_interaction/res/dockerfiles/ubuntu   data/os_interaction/res/dockerfiles
```

### Step 4 — Start OS-only stack (controller + redis + os worker)
```bash
docker compose -f extra/docker-compose.os-only.yml up --build -d
```

Sanity check (controller reachable):
```bash
curl -s http://localhost:5020/api/get_tasks | head
```

Sanity check (start_sample returns non-empty `messages`):
```bash
curl -s -H 'Content-Type: application/json' \
  -d '{"name":"os-std","index":0,"custom_task":null}' \
  http://localhost:5020/api/start_sample | head -c 300; echo
```

Optional sanity check for the monitored variant:
```bash
curl -s -H 'Content-Type: application/json' \
  -d '{"name":"os-std-monitor-replan","index":0,"custom_task":null}' \
  http://localhost:5020/api/start_sample | head -c 300; echo
```

### Step 5 — Run evaluation

Set API key (AgentRL evaluator uses this variable reliably):
```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
```

Small test:
```bash
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5.1 \
  --indices-range "0-4" \
  --concurrency 1 \
  -n 1 \
  -o results \
  os-std
```

Monitor + replanner variant:
```bash
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5.1 \
  --concurrency 4 \
  -n 1 \
  -o results \
  os-std-monitor-replan
```

Full run (144 indices):
```bash
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5.1 \
  --concurrency 4 \
  -n 1 \
  -o results \
  os-std
```

Outputs go to:
- `results/<model>-os-std-YYYYMMDDHHMM/`
  - `run.log`
  - `results.jsonl`
  - per-index traces (e.g., `*/trace.json`)

The `os-std-monitor-replan` traces additionally contain injected `[STATE MEMORY]` and `[Monitor]` messages, which makes later trajectory analysis easier.

---

## 3) Current results (preliminary)

All runs are stored under the `results/` directory in this repo.

Initial reproduction (CSL machines):

- **gpt-5.1**: SR ≈ **38.8%**
- **gpt-5-mini**: SR ≈ **28.7%**

(“SR” here refers to the success rate metric reported by `agentrl-eval` at the end of `run.log`.)

---

## 4) Progress update: monitor-assisted long-horizon improvements

This section documents the **post-baseline monitor-assisted improvements** that were added after the initial CSL reproduction.

### Motivation

The original `os-std` baseline keeps the agent loop simple, but long-horizon failures often come from:

- **task-type confusion**  
  query-style tasks (for example, *how many*, *which path*, *what is the output*) being treated like state-change tasks

- **over-long bash steps**  
  the model sometimes emits a large multi-line script too early, which is expensive, brittle, and often yields weak evidence

- **loop / stall behavior**  
  repeated commands, repeated observations, or broad exploration without reducing uncertainty

- **premature finish / answer**  
  the model may try to finish without verification, or answer without grounded shell evidence

- **tool-calling protocol mistakes**  
  some model responses produce multiple tool calls or awkward plain-text answers when a tool call is required

### What we added

The monitored variant (`os-std-monitor-replan`) now includes the following components.

#### A. Goal-aware task typing

We added a lightweight goal heuristic in `src/server/tasks/os_interaction/monitoring.py` that classifies each task into one of three modes:

- **answer**
- **state**
- **hybrid**

This is important because many OS tasks are not pure state-change tasks. Some only require a grounded answer, while others require a small implementation step **followed by** a grounded answer.

Examples:

- **answer**: “how many …”, “tell me …”, “full path …”, “what will be the output …”
- **state**: “change permission …”, “rename …”, “make file writable …”
- **hybrid**: “write a bash script … then calculate / output / report …”

This task typing is used by the monitor to decide:
- whether mutation is appropriate at all,
- whether the next phase should be diagnose / mutate / verify / answer,
- how aggressive the monitor should be.

#### B. Initial task hint

At the beginning of each sample, the worker injects a short **task hint** that tells the model whether the task is mainly:
- query/answer,
- state-change, or
- hybrid.

This is meant to bias the agent toward:
- **short read-only commands** for query tasks,
- **minimal state changes + verification** for state tasks,
- **implement -> verify -> answer** for hybrid tasks.

#### C. Conditional working-plan injection

Earlier monitor versions injected a plan too often, which sometimes improved robustness but also increased token cost and over-constrained the agent.

The current version only injects `[WORKING PLAN]` conditionally, mainly when:
- the latest step failed,
- the latest step changed state but still needs verification,
- the agent recently triggered a monitor intervention,
- or the trajectory looks suspicious.

This reduced prompt overhead and made the monitor much less intrusive.

#### D. Loop / stall detection

The monitor keeps a short trajectory state and checks for:
- repeated commands with nearly identical outputs,
- suspicious empty inspection steps,
- repeated low-signal behavior over a short window.

When needed, it injects a recovery prompt that is aware of:
- the current task mode,
- the inferred failure type,
- the remaining subgoal.

#### E. Oversized-bash guard

A new pre-execution guard blocks **large multi-line bash scripts** in situations where they are usually harmful:
- query tasks,
- hybrid tasks in diagnose / answer phases,
- state tasks during early diagnosis.

Instead of allowing a large brittle script, the monitor tells the model to:
- narrow with one short command first,
- avoid large pipelines unless the task explicitly requires creating a script/file.

#### F. Commit / answer gating

The monitor blocks:
- `finish_action` on answer-style tasks,
- finishing without verification on state tasks,
- finishing hybrid tasks before the final output is grounded.

This helps enforce:
- **state tasks**: mutate -> verify -> finish
- **answer tasks**: observe -> answer
- **hybrid tasks**: implement -> verify -> answer / finish

#### G. Protocol-safe tool handling

In `src/server/tasks/os_interaction/task.py`, tool handling was tightened so that:
- assistant tool calls are always followed by tool responses in the proper order,
- extra tool calls in the same turn are explicitly handled,
- plain-text atomic answers can be auto-converted into an answer result in limited cases,
- bash execution failures are wrapped and surfaced more safely.

This reduced model/protocol errors substantially in the latest run.

---

## 5) Files changed for the monitored variant

The main files involved in the current monitor-assisted pipeline are:

- `src/server/tasks/os_interaction/monitoring.py`
- `src/server/tasks/os_interaction/task.py`
- `configs/tasks/os.yaml`

At a high level:

- **`monitoring.py`**
  - trajectory state
  - task typing (`answer` / `state` / `hybrid`)
  - loop/stall detection
  - working-plan generation
  - recovery prompt generation
  - commit/answer gating
  - oversized-bash guard

- **`task.py`**
  - inject initial task hint
  - inject working plan only when needed
  - process tool calls safely
  - record shell execution into trajectory state
  - auto-handle some grounded atomic answers

- **`os.yaml`**
  - keeps the original `os-std`
  - adds the monitored variant `os-std-monitor-replan`
  - uses a lighter monitor config:
    - `round_limit: 14`
    - `max_interventions: 2`
    - `cooldown_rounds: 2`

---

## 6) Running the latest monitored variant

After code changes, rebuild the OS-only stack:

```bash
docker compose -f extra/docker-compose.os-only.yml down
docker compose -f extra/docker-compose.os-only.yml up --build -d
```

Small smoke test:

```bash
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5-mini \
  --indices-range "0-19" \
  --concurrency 1 \
  -n 1 \
  -o results \
  os-std-monitor-replan
```

Full run:

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

---

## 7) Latest experimental results

### Baseline reproduction

Initial CSL reproduction:

- **gpt-5.1**: SR ≈ **38.8%**
- **gpt-5-mini**: SR ≈ **28.7%**

### Intermediate monitored version

A later monitor-assisted version (full 144-task run) with **gpt-5-mini** produced:

- **valid-task success rate**: **38.46%** (`55 / 143`)
- **overall end-to-end success rate**: **38.19%** (`55 / 144`)
- **non-completed samples**:
  - `1` server error
  - `1` task limit reached

Token usage from `run.log`:

- **1.6M tokens total**
  - 1.2M input
  - 437.3k output
  - 349.6k thinking

### Latest monitored version

The latest full run of `os-std-monitor-replan` with **gpt-5-mini** produced:

- **valid-task success rate**: **50.35%** (`72 / 143`)
- **overall end-to-end success rate**: **50.00%** (`72 / 144`)
- **non-completed samples**:
  - `1` server error
  - `0` task errors
  - `0` model errors
  - `0` task-limit failures

Token usage from `run.log`:

- **833.7k tokens total**
  - 593.7k input
  - 240.1k output
  - 201.3k thinking

### Why both numbers are reported

`agentrl-eval` reports a **valid-task average**, which excludes some failed runs that do not complete in the standard way (for example, server-side errors).  
For transparency, we also report the **overall end-to-end success rate over all 144 tasks**.

In the latest run:

- evaluator-reported valid average: **50.35%**
- true full-run success over all 144 tasks: **50.00%**

---

## 8) Observations from the latest traces

From the recent trace analysis, the biggest improvements appear to come from:

- **better task-type routing**
  - query tasks are less likely to be pushed into unnecessary mutations
- **less monitor over-injection**
  - trajectories are shorter and cleaner
- **safer tool-call handling**
  - protocol-related model errors were largely eliminated
- **blocking oversized scripts**
  - the agent is nudged toward short, local evidence-gathering steps

A useful side effect is **lower token usage**:
- the latest run achieved a much higher success rate than the intermediate monitor version,
- while using **roughly half** as many total tokens.

---

## 9) Remaining known issue

There is still **one persistent server error** in the latest full run.

Observed symptom:
- an `InteractResponse` validation error indicating that the response `messages` field was empty

This issue does **not** look like a general monitor/protocol failure, because:
- it appears only once,
- it also appeared in an earlier monitored run,
- and the broader protocol-error pattern disappeared in the latest version.

So the current working hypothesis is that this is a **sample-specific controller/session edge case**, rather than the main algorithmic bottleneck.

---

## 10) Current takeaway

At this point, the monitor-assisted variant is no longer just a minor scaffold around the baseline.  
It functions as a lightweight **trajectory controller** for long-horizon OS tasks:

- infer task type,
- encourage short grounded evidence gathering,
- block oversized low-value actions,
- prevent premature finish/answer,
- and recover from loops or weak-signal behavior.

On the latest full run with **gpt-5-mini**, this improved the project from the original **28.7%** baseline to about **50% overall**, while also reducing protocol/task errors and lowering token usage.

This is still a preliminary course-project result, but it is a strong signal that **lightweight inference-time trajectory control** can substantially improve long-horizon OS-agent robustness without any RL training.
