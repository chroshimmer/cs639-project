# Extension 1 Results Summary

This summary covers the three full runs currently retained in `extension1_cycle_recovery/results/`.

## 1) Retained full runs

- Baseline: `gpt-5-mini-os-std-202605031258`
- Extension 1: `gpt-5-mini-os-std-extension1-202605031322`
- Combo v2: `gpt-5-mini-os-std-combo-202605031445`

## 2) Core metrics (from `results.jsonl`)

| Config | total | completed | success (`metric_reward=1`) | success / completed |
|---|---:|---:|---:|---:|
| baseline (`os-std`) | 144 | 143 | 49 | 0.3427 |
| extension1 (`os-std-extension1`) | 144 | 142 | 56 | 0.3944 |
| combo v2 (`os-std-combo`) | 144 | 140 | 68 | 0.4857 |

**Four-way comparison (monitor-replan):** not stored under `extension1_cycle_recovery/results/`. From the main repo run `results/gpt-5-mini-os-std-monitor-replan-202604070041/`: completed-only success rate tracks **72/143 ≈ 50.35%** (same convention as `results.jsonl` aggregated rows in the Extension~1 report §5.1).

## 3) Status breakdown

- Baseline (`os-std-202605031258`)
  - `completed`: 143
  - `server error`: 1
- Extension 1 (`os-std-extension1-202605031322`)
  - `completed`: 142
  - `server error`: 2
- Combo v2 (`os-std-combo-202605031445`)
  - `completed`: 140
  - `server error`: 1
  - `task limit reached`: 3

## 4) Important metric-note for combo v2

There are two valid reporting conventions in your materials:

- **`results.jsonl` completed-only denominator**  
  - `68 / 140 = 0.4857`
- **Evaluator `run.log` valid denominator**  
  - `68 / 143 = 0.4755`

Why both appear:
- `run.log` can count non-completed but still "valid" outcomes in its `valid` denominator;
- completed-only summaries only use `status=completed` as denominator.

For presentations, keep one convention consistently and add a one-line note about denominator definition.
