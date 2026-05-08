# Extension 1: Cycle Detection and Recovery Bundle

This folder packages the Extension 1 artifacts needed for review and reproducibility.
It is prepared for readers such as course staff, teammates, and external reviewers who need a clear, self-contained reference for what was implemented and how to reproduce the reported behavior.
Specifically, this extension adds a state-aware cycle detection mechanism during OS task execution to identify when the agent gets stuck repeating actions.  
When a cycle is detected, the system triggers recovery and replanning to reduce wasted steps and improve end-to-end task completion stability.

This bundle is intended to support:
- fast technical review of the Extension 1 changes;
- straightforward handoff with minimal repository exploration;
- reproducibility with explicit run commands and file mapping to the main project.

---

## 1) Relationship to the original project

`extension1_cycle_recovery/` is **not** a standalone repo by default.  
It is a snapshot-style package that mirrors selected files from the main project.

- Original project root: `cs639-project/`
- Extension bundle root: `cs639-project/extension1_cycle_recovery/`

Think of this folder as:
- a structured backup of your Extension 1 changes/results;
- a reproducibility helper;
- a "drop-in reference" for which files were modified.

**Integration note (important):** `cycle_monitor.py` lives **here**, under `extension1_cycle_recovery/src/...`. In many checkouts the **repository root** `src/server/tasks/os_interaction/` does **not** ship Extension~1 by default (no `cycle_monitor.py` until you copy from this bundle per §4). Running `os-std-extension1` / `os-std-combo` requires those overlays.

---

## 2) What is included in this package

Current mirrored content includes:
- `configs/tasks/os.yaml`
- `extra/docker-compose.os-only.yml`
- `src/server/tasks/os_interaction/task.py`
- `src/server/tasks/os_interaction/cycle_monitor.py`
- `reports/Extension 1 – State-Aware Cycle Detection & Recovery.md`
- selected full-run result folders under `results/...`:
  - `gpt-5-mini-os-std-202605031258` (baseline)
  - `gpt-5-mini-os-std-extension1-202605031322` (extension1)
  - `gpt-5-mini-os-std-combo-202605031445` (combo v2)
- run helpers: `run_combo.sh`, `run_ab.sh`, `run_full_ab.sh`
- `tests/test_cycle_monitor.py` (unit tests for cycle scoring / intervention logic)

**Not bundled here (large / separate run):** the full `gpt-5-mini` **monitor-replan** directory used for four-way comparisons in the Extension~1 report lives under the **main repo** at `results/gpt-5-mini-os-std-monitor-replan-202604070041/` (April batch). Copy it in if you want a single folder that holds all four raw runs.

These paths are preserved under `extension1_cycle_recovery/` to make diffing and restoration simple.

---

## 3) Reproduce Extension 1 (recommended workflow)

Use the **main project root** (`cs639-project/`) as the execution root.
Do not run from inside `extension1_cycle_recovery/` unless you explicitly create a separate runnable repo.

### Step A. Environment and dependencies

From project root:

```bash
conda create -n cs639-project python=3.11 -y
conda activate cs639-project
pip install -r requirements.txt
pip install -U agentrl-eval openai anthropic
```

### Step B. Start Docker services (OS-only stack)

```bash
systemctl --user start docker.service
docker compose -f extra/docker-compose.os-only.yml up --build -d
```

Controller check:

```bash
curl -s http://localhost:5020/api/get_tasks
```

### Step C. Run evaluations

Set API key:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
```

Run examples:

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

```bash
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5-mini \
  --concurrency 4 \
  -n 1 \
  -o results \
  os-std-extension1
```

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

```bash
agentrl-eval \
  --no-interactive \
  -c http://localhost:5020/api \
  -u https://api.openai.com/v1 \
  -m gpt-5-mini \
  --concurrency 4 \
  -n 1 \
  -o results \
  os-std-combo
```

---

## 4) If you want to "apply Extension 1" onto another clean copy

If you have a clean project checkout and want to use this Extension 1 version, replace files in the clean copy with these files from `extension1_cycle_recovery/`:

- `extension1_cycle_recovery/configs/tasks/os.yaml` -> `configs/tasks/os.yaml`
- `extension1_cycle_recovery/extra/docker-compose.os-only.yml` -> `extra/docker-compose.os-only.yml`
- `extension1_cycle_recovery/src/server/tasks/os_interaction/task.py` -> `src/server/tasks/os_interaction/task.py`
- `extension1_cycle_recovery/src/server/tasks/os_interaction/cycle_monitor.py` -> `src/server/tasks/os_interaction/cycle_monitor.py`

Optional documentation replacement:

- `extension1_cycle_recovery/reports/Extension 1 – State-Aware Cycle Detection & Recovery.md`  
  as your latest report version under `reports/`.

Optional scripts/logs/results:

- Copy from `extension1_cycle_recovery/run_combo.sh`, `run_ab.sh`, `run_full_ab.sh`, and `extension1_cycle_recovery/results/...` as needed.
- For GitHub hygiene, prefer uploading summarized reports over full raw traces.

---

## 5) Suggested validation after replacement

After replacing files, run:

```bash
python -m py_compile src/server/tasks/os_interaction/task.py src/server/tasks/os_interaction/cycle_monitor.py
```

Then do:
- a small smoke run (`--indices-range "0-19"`),
- one full run (144 tasks),
- and compare `overall.json` / `run.log` to your bundled results.

---

## 6) Notes

- This package stores copied artifacts; the canonical runnable project remains the root repository.
- Before grading or external review, **diff** `extension1_cycle_recovery/src/...` and `configs/...` against the repo root and refresh copies if the main branch moved (avoid stale mirrors).
- Keep filenames with spaces/special dash quoted in shell commands, e.g.:
  - `"reports/Extension 1 – State-Aware Cycle Detection & Recovery.md"`
