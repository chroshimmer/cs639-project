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
