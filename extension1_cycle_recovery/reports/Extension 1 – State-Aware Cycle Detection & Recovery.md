# Extension 1 - State-Aware Cycle Detection and Recovery

## Executive Summary

This is the cleaned final report for Extension 1 based on May 3 experiments.

Main findings:
- Extension 1 improves over baseline when used standalone.
- Monitor-replan remains the strongest single setup on headline score.
- Naive stacking hurts; coordinated stacking helps but still does not beat standalone monitor-replan.

## 1. Goal

Extension 1 targets loop-like behavior in OS tasks:
- repeated similar commands;
- repeated similar observations;
- no meaningful progress.

It is a lightweight runtime monitor:
- no changes to dataset/checker/environment;
- only inject recovery guidance when cycle signals exceed thresholds.

## 2. Final Implementation

Active code paths:
- `src/server/tasks/os_interaction/cycle_monitor.py`
- `src/server/tasks/os_interaction/task.py`
- `configs/tasks/os.yaml`

Task variants used:
- `os-std` (baseline)
- `os-std-extension1` (cycle monitor only)
- `os-std-monitor-replan` (teammate monitor system)
- `os-std-combo` (both enabled)

## 3. Coordination Update for Combo

To reduce double intervention when both monitors are enabled, we added:
- `note_external_intervention(state, turn_id)` in `CycleMonitor`.

`task.py` calls this when `OSMonitor` intervenes (commit block, complex bash block, replanning prompt), so cycle monitor yields during cooldown instead of stacking another meta instruction immediately.

## 4. Evaluation Setup

Common settings for May 3 runs:
- model: `gpt-5-mini`
- controller: `http://localhost:5020/api`
- task split: OS standard 144 tasks
- concurrency: 4
- runs: 1

Run directories in this analysis:
- baseline: `results/gpt-5-mini-os-std-202605031258/`
- extension1: `results/gpt-5-mini-os-std-extension1-202605031322/`
- monitor-replan: `results/gpt-5-mini-os-std-monitor-replan-202604070041/`
- combo v1: `results/gpt-5-mini-os-std-combo-202605031408/`
- combo v2 (coordinated): `results/gpt-5-mini-os-std-combo-202605031445/`

## 5. Results

### 5.1 Raw metrics

| Config | valid | success | avg |
|---|---:|---:|---:|
| baseline (os-std) | 143 | 49 | 0.3427 |
| extension1 (cycle) | 142 | 56 | 0.3944 |
| monitor-replan | 143 | 72 | 0.5035 |
| combo v1 (naive stack) | 142 | 66 | 0.4648 |
| combo v2 (coordinated) | 143 | 68 | 0.4755 |

### 5.2 Common subset (141 shared indices)

| Config | success | avg | delta vs baseline |
|---|---:|---:|---:|
| baseline | 48 | 0.3404 | - |
| extension1 | 56 | 0.3972 | +5.67 pts |
| monitor-replan | 71 | 0.5035 | +16.31 pts |
| combo v1 | 66 | 0.4681 | +12.77 pts |
| combo v2 | 67 | 0.4752 | +13.48 pts |

### 5.3 Interpretation

- Extension 1 is a positive standalone improvement over baseline.
- Naive stacking is below standalone monitor-replan.
- Coordination improves combo (`v2 > v1`), but combo still trails standalone monitor-replan.

## 6. Why Combo Still Trails Monitor-Replan

Current evidence suggests:
- monitor-replan already covers much of long-horizon control behavior;
- cycle monitor adds limited incremental signal when monitor-replan is active;
- run-to-run variance remains meaningful at this sample size.

So in this setup, Extension 1 is more useful as a standalone/lightweight guard than a strong additive gain on top of monitor-replan.

## 7. Practical Conclusion

Under current settings:
- best headline score: standalone monitor-replan;
- best lightweight interpretable add-on: Extension 1 standalone;
- combo with coordination is viable, but not superior to standalone monitor-replan yet.

## 8. Next Steps

1. Re-run monitor-replan in the same time window as combo v2 for tighter apples-to-apples comparison.
2. Run repeated trials (multiple runs) to reduce variance sensitivity.
3. If integrating further, merge non-overlapping cycle signals into monitor-replan internals instead of stacking two separate intervention channels.