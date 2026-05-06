from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/Users/chloe/Documents/New project")
DATASET_PATH = ROOT / "rolling_monitor_dataset" / "multitask" / "rolling_multitask.csv"

OFFICIAL_DIR = ROOT / "result" / "official_multi_Qwen3-30B-A3B-Instruct-2507"
AUTO_DIR = ROOT / "result" / "auto_multi_Qwen3-30B-A3B-Instruct-2507"

OFFICIAL_TRAJ_PATH = OFFICIAL_DIR / "qwen_multitask_trajectories.jsonl"
AUTO_TRAJ_PATH = AUTO_DIR / "auto_qwen_multitask_trajectories.jsonl"

OFFICIAL_EVAL_PATH = OFFICIAL_DIR / "qwen_multitask_eval.json"
AUTO_EVAL_PATH = AUTO_DIR / "auto_qwen_multitask_eval.json"
COMPARISON_JSON_PATH = ROOT / "result" / "multitask_official_vs_auto_comparison.json"
REPORT_PATH = ROOT / "docs" / "official_vs_auto_multitask_report.md"

TASK_LABEL_SPACES = {
    "sepsis": ["keep_monitoring", "infection_suspect", "trigger_sepsis_alert"],
    "aki": ["keep_monitoring", "suspect_aki", "trigger_aki_alert"],
    "respiratory_support": [
        "room_air_or_low_support",
        "high_flow_or_noninvasive_support",
        "invasive_vent_required",
    ],
}

TASK_BASELINE_ACTION = {
    "sepsis": "keep_monitoring",
    "aki": "keep_monitoring",
    "respiratory_support": "room_air_or_low_support",
}

TASK_TOOLS = {
    "sepsis": {"query_suspicion_of_infection", "query_sofa"},
    "aki": {"query_kdigo_stage"},
    "respiratory_support": {"query_ventilation_status"},
}

TASK_TRANSITION_ACTIONS = {
    "sepsis": ["infection_suspect", "trigger_sepsis_alert"],
    "aki": ["suspect_aki", "trigger_aki_alert"],
    "respiratory_support": ["high_flow_or_noninvasive_support", "invasive_vent_required"],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_dataset_stays(path: Path) -> int:
    stay_ids: set[str] = set()
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stay_ids.add(row["stay_id"])
    return len(stay_ids)


def safe_round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def first_gt_action_hour(rollout: dict[str, Any], task_name: str, action: str) -> int | None:
    for step in rollout["steps"]:
        if step["gt_task_actions"][task_name] == action:
            return step["t_hour"]
    return None


def timing_metrics(gt_hours: list[int | None], pred_hours: list[int | None]) -> dict[str, Any]:
    exact = 0
    early = 0
    late = 0
    missed = 0
    abs_errors: list[int] = []

    positives = sum(gt is not None for gt in gt_hours)
    for gt, pred in zip(gt_hours, pred_hours):
        if gt is None:
            continue
        if pred is None:
            missed += 1
            continue
        if pred == gt:
            exact += 1
        elif pred < gt:
            early += 1
        else:
            late += 1
        abs_errors.append(abs(pred - gt))

    return {
        "positives": positives,
        "exact_match_rate": safe_round(exact / positives) if positives else None,
        "mean_absolute_error_hours": safe_round(sum(abs_errors) / len(abs_errors)) if abs_errors else None,
        "early_rate": safe_round(early / positives) if positives else None,
        "late_rate": safe_round(late / positives) if positives else None,
        "missed_rate": safe_round(missed / positives) if positives else None,
    }


def evaluate_multitask_rollouts(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = 0
    joint_correct = 0

    per_task_counts: dict[str, int] = defaultdict(int)
    per_task_correct: dict[str, int] = defaultdict(int)
    per_task_confusion: dict[str, Counter[tuple[str, str | None]]] = defaultdict(Counter)
    grounding = {task_name: {"num": 0, "den": 0} for task_name in TASK_LABEL_SPACES}
    gt_counts = {task_name: Counter() for task_name in TASK_LABEL_SPACES}
    pred_counts = {task_name: Counter() for task_name in TASK_LABEL_SPACES}

    transitions: dict[str, dict[str, dict[str, list[int | None]]]] = {
        task_name: {
            action: {"gt": [], "pred": []}
            for action in TASK_TRANSITION_ACTIONS[task_name]
        }
        for task_name in TASK_LABEL_SPACES
    }

    for rollout in rollouts:
        for step in rollout["steps"]:
            total_steps += 1
            gt = step["gt_task_actions"]
            pred = step["predicted_task_actions"]
            if gt == pred:
                joint_correct += 1

            tool_names = {call["tool_name"] for call in step["tool_calls"]}
            for task_name, gt_action in gt.items():
                pred_action = pred.get(task_name)
                per_task_counts[task_name] += 1
                gt_counts[task_name][gt_action] += 1
                pred_counts[task_name][pred_action] += 1
                per_task_confusion[task_name][(gt_action, pred_action)] += 1
                if gt_action == pred_action:
                    per_task_correct[task_name] += 1
                if pred_action is not None and pred_action != TASK_BASELINE_ACTION[task_name]:
                    grounding[task_name]["den"] += 1
                    if tool_names & TASK_TOOLS[task_name]:
                        grounding[task_name]["num"] += 1

        first_predicted = rollout.get("first_predicted_task_hours") or {}
        for task_name, actions in TASK_TRANSITION_ACTIONS.items():
            for action in actions:
                transitions[task_name][action]["gt"].append(first_gt_action_hour(rollout, task_name, action))
                transitions[task_name][action]["pred"].append((first_predicted.get(task_name) or {}).get(action))

    per_task: dict[str, Any] = {}
    for task_name, label_space in TASK_LABEL_SPACES.items():
        per_class: dict[str, Any] = {}
        for action in label_space:
            tp = per_task_confusion[task_name][(action, action)]
            fp = sum(per_task_confusion[task_name][(gt, action)] for gt in label_space if gt != action)
            fn = sum(per_task_confusion[task_name][(action, pred)] for pred in label_space if pred != action)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            per_class[action] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1(tp, fp, fn), 4),
            }

        per_task[task_name] = {
            "accuracy": safe_round(per_task_correct[task_name] / per_task_counts[task_name]) if per_task_counts[task_name] else None,
            "macro_f1": round(sum(metrics["f1"] for metrics in per_class.values()) / len(label_space), 4),
            "per_class": per_class,
            "grounded_rate": safe_round(grounding[task_name]["num"] / grounding[task_name]["den"])
            if grounding[task_name]["den"]
            else None,
            "gt_counts": dict(gt_counts[task_name]),
            "pred_counts": dict(pred_counts[task_name]),
        }

    return {
        "joint_step_accuracy": safe_round(joint_correct / total_steps) if total_steps else None,
        "per_task": per_task,
        "transition_timing": {
            task_name: {
                action: timing_metrics(values["gt"], values["pred"])
                for action, values in task_transitions.items()
            }
            for task_name, task_transitions in transitions.items()
        },
    }


def paired_correctness(
    official_rollouts: list[dict[str, Any]],
    auto_rollouts: list[dict[str, Any]],
    overlap_ids: list[str],
) -> dict[str, Any]:
    official_by_id = {rollout["trajectory_id"]: rollout for rollout in official_rollouts}
    auto_by_id = {rollout["trajectory_id"]: rollout for rollout in auto_rollouts}

    joint_step = Counter()
    joint_trajectory = Counter()
    per_task = {task_name: Counter() for task_name in TASK_LABEL_SPACES}

    for trajectory_id in overlap_ids:
        official = official_by_id[trajectory_id]
        auto = auto_by_id[trajectory_id]
        official_steps = {step["t_hour"]: step for step in official["steps"]}
        auto_steps = {step["t_hour"]: step for step in auto["steps"]}

        official_perfect = True
        auto_perfect = True
        for t_hour in sorted(official_steps.keys() & auto_steps.keys()):
            gt = official_steps[t_hour]["gt_task_actions"]
            official_pred = official_steps[t_hour]["predicted_task_actions"]
            auto_pred = auto_steps[t_hour]["predicted_task_actions"]

            official_joint = official_pred == gt
            auto_joint = auto_pred == gt
            joint_step[(official_joint, auto_joint)] += 1
            if not official_joint:
                official_perfect = False
            if not auto_joint:
                auto_perfect = False

            for task_name in TASK_LABEL_SPACES:
                official_task = official_pred[task_name] == gt[task_name]
                auto_task = auto_pred[task_name] == gt[task_name]
                per_task[task_name][(official_task, auto_task)] += 1

        joint_trajectory[(official_perfect, auto_perfect)] += 1

    return {
        "joint_step": {f"official_{key[0]}__auto_{key[1]}": value for key, value in joint_step.items()},
        "joint_trajectory": {f"official_{key[0]}__auto_{key[1]}": value for key, value in joint_trajectory.items()},
        "per_task": {
            task_name: {f"official_{key[0]}__auto_{key[1]}": value for key, value in counts.items()}
            for task_name, counts in per_task.items()
        },
    }


def step_tool_map(step: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        call["tool_name"]: output
        for call, output in zip(step["tool_calls"], step["tool_outputs"])
    }


def summarize_distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": round(sum(values) / len(values), 4),
        "median": round(ordered[len(ordered) // 2], 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def backend_divergence(
    official_rollouts: list[dict[str, Any]],
    auto_rollouts: list[dict[str, Any]],
    overlap_ids: list[str],
) -> dict[str, Any]:
    official_by_id = {rollout["trajectory_id"]: rollout for rollout in official_rollouts}
    auto_by_id = {rollout["trajectory_id"]: rollout for rollout in auto_rollouts}

    counts = Counter()
    examples: dict[str, list[Any]] = defaultdict(list)
    infection_hour_deltas: list[float] = []
    sofa_deltas: list[float] = []

    for trajectory_id in overlap_ids:
        official = official_by_id[trajectory_id]
        auto = auto_by_id[trajectory_id]
        official_steps = {step["t_hour"]: step for step in official["steps"]}
        auto_steps = {step["t_hour"]: step for step in auto["steps"]}

        for t_hour in sorted(official_steps.keys() & auto_steps.keys()):
            counts["steps"] += 1
            official_tools = step_tool_map(official_steps[t_hour])
            auto_tools = step_tool_map(auto_steps[t_hour])

            official_infection = official_tools["query_suspicion_of_infection"]
            auto_infection = auto_tools["query_suspicion_of_infection"]
            official_sofa = official_tools["query_sofa"]
            auto_sofa = auto_tools["query_sofa"]
            official_aki = official_tools["query_kdigo_stage"]
            auto_aki = auto_tools["query_kdigo_stage"]
            official_resp = official_tools["query_ventilation_status"]
            auto_resp = auto_tools["query_ventilation_status"]

            if official_infection["has_suspected_infection"] != auto_infection["has_suspected_infection"]:
                counts["infection_flag_disagree"] += 1
                if len(examples["infection_flag_disagree"]) < 5:
                    examples["infection_flag_disagree"].append(
                        {
                            "trajectory_id": trajectory_id,
                            "t_hour": t_hour,
                            "official_flag": official_infection["has_suspected_infection"],
                            "auto_flag": auto_infection["has_suspected_infection"],
                            "official_hour": official_infection.get("first_visible_suspected_infection_hour"),
                            "auto_hour": auto_infection.get("first_visible_suspected_infection_hour"),
                        }
                    )

            if auto_infection.get("evidence") and not auto_infection.get("has_suspected_infection"):
                counts["auto_infection_internal_contradiction"] += 1
                if len(examples["auto_infection_internal_contradiction"]) < 5:
                    examples["auto_infection_internal_contradiction"].append(
                        {
                            "trajectory_id": trajectory_id,
                            "t_hour": t_hour,
                            "evidence_count": len(auto_infection["evidence"]),
                            "auto_hour": auto_infection.get("first_visible_suspected_infection_hour"),
                        }
                    )

            official_hour = official_infection.get("first_visible_suspected_infection_hour")
            auto_hour = auto_infection.get("first_visible_suspected_infection_hour")
            if official_hour is not None and auto_hour is not None:
                infection_hour_deltas.append(auto_hour - official_hour)
            elif official_hour is not None or auto_hour is not None:
                counts["infection_hour_presence_disagree"] += 1

            official_sofa_score = official_sofa.get("latest_sofa_24hours")
            auto_sofa_score = auto_sofa.get("latest_sofa_24hours")
            if official_sofa_score is not None and auto_sofa_score is not None:
                delta = auto_sofa_score - official_sofa_score
                sofa_deltas.append(delta)
                if delta >= 2:
                    counts["auto_sofa_ge2_higher"] += 1
                if delta <= -2:
                    counts["auto_sofa_ge2_lower"] += 1

            if official_aki.get("latest_aki_stage_smoothed") != auto_aki.get("latest_aki_stage_smoothed"):
                counts["aki_stage_disagree"] += 1
                if len(examples["aki_stage_disagree"]) < 5:
                    examples["aki_stage_disagree"].append(
                        {
                            "trajectory_id": trajectory_id,
                            "t_hour": t_hour,
                            "official_stage": official_aki.get("latest_aki_stage_smoothed"),
                            "auto_stage": auto_aki.get("latest_aki_stage_smoothed"),
                        }
                    )

            if official_resp.get("current_support_level") != auto_resp.get("current_support_level"):
                counts["resp_support_disagree"] += 1
                if len(examples["resp_support_disagree"]) < 5:
                    examples["resp_support_disagree"].append(
                        {
                            "trajectory_id": trajectory_id,
                            "t_hour": t_hour,
                            "official_level": official_resp.get("current_support_level"),
                            "auto_level": auto_resp.get("current_support_level"),
                            "auto_raw_status": auto_resp.get("current_status_raw"),
                        }
                    )

    return {
        **counts,
        "infection_hour_delta_auto_minus_official": summarize_distribution(infection_hour_deltas),
        "sofa_delta_auto_minus_official": summarize_distribution(sofa_deltas),
        "examples": examples,
    }


def build_eval_summary(
    tool_backend: str,
    completed_rollouts: list[dict[str, Any]],
    dataset_size: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_mode": "multitask",
        "tool_backend": tool_backend,
        "dataset": str(DATASET_PATH),
        "num_trajectories": len(completed_rollouts),
        "sample_size": dataset_size,
        "agent": "qwen",
        "metrics": {
            "joint_step_accuracy": metrics["joint_step_accuracy"],
            "per_task": metrics["per_task"],
        },
        "metadata_note": (
            "Reconstructed from saved multitask trajectory JSONL because the original eval summary was not "
            f"saved. Metrics cover {len(completed_rollouts)} completed trajectories out of the "
            f"{dataset_size}-stay multitask dataset."
        ),
    }


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def file_link(path: Path) -> str:
    return str(path)


def build_report(
    dataset_size: int,
    official_rollouts: list[dict[str, Any]],
    auto_rollouts: list[dict[str, Any]],
    official_full: dict[str, Any],
    auto_full: dict[str, Any],
    official_overlap: dict[str, Any],
    auto_overlap: dict[str, Any],
    overlap_ids: list[str],
    official_only_ids: list[str],
    auto_only_ids: list[str],
    paired: dict[str, Any],
    divergence: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Official vs Autoformalized Multitask Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This report compares the saved multitask Qwen runs under:")
    lines.append("")
    lines.append(f"- official visible concepts: `{file_link(OFFICIAL_DIR)}`")
    lines.append(f"- autoformalized visible concepts: `{file_link(AUTO_DIR)}`")
    lines.append("")
    lines.append(f"Reference multitask dataset: `{file_link(DATASET_PATH)}`")
    lines.append("")
    lines.append("## Important Caveat")
    lines.append("")
    lines.append("This is not a clean full-cohort comparison from the saved artifacts alone.")
    lines.append("")
    lines.append(f"- intended multitask cohort size: `{dataset_size}` stays")
    lines.append(f"- official saved trajectories: `{len(official_rollouts)}`")
    lines.append(f"- auto saved trajectories: `{len(auto_rollouts)}`")
    lines.append(f"- overlap on exact trajectory id: `{len(overlap_ids)}`")
    lines.append(f"- official-only saved trajectories: `{len(official_only_ids)}`")
    lines.append(f"- auto-only saved trajectories: `{len(auto_only_ids)}`")
    lines.append("")
    lines.append("So there are two fair views:")
    lines.append("")
    lines.append("- saved-run view: each backend scored on everything that was actually saved")
    lines.append("- matched-overlap view: both backends scored only on the shared saved trajectories")
    lines.append("")
    lines.append("The matched-overlap view is the right one for backend comparison. The saved-run view is still useful for completeness and provenance.")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("The official backend is the stronger multitask system overall, and that conclusion survives the matched-overlap check.")
    lines.append("")
    lines.append("Why:")
    lines.append("")
    lines.append(
        f"- much better joint multitask accuracy on the matched overlap: `{fmt(official_overlap['joint_step_accuracy'])}` vs `{fmt(auto_overlap['joint_step_accuracy'])}`"
    )
    lines.append(
        f"- decisive AKI advantage on the matched overlap: `{fmt(official_overlap['per_task']['aki']['accuracy'])}` vs `{fmt(auto_overlap['per_task']['aki']['accuracy'])}`"
    )
    lines.append(
        f"- better respiratory support accuracy on the matched overlap: `{fmt(official_overlap['per_task']['respiratory_support']['accuracy'])}` vs `{fmt(auto_overlap['per_task']['respiratory_support']['accuracy'])}`"
    )
    lines.append("- both backends are fully grounded in the narrow benchmark sense, so the gap is not caused by tool refusal")
    lines.append("")
    lines.append("The sepsis head is more nuanced than the joint result:")
    lines.append("")
    lines.append(
        f"- auto is slightly higher on matched sepsis step accuracy: `{fmt(auto_overlap['per_task']['sepsis']['accuracy'])}` vs `{fmt(official_overlap['per_task']['sepsis']['accuracy'])}`"
    )
    lines.append(
        f"- but official preserves the intermediate `infection_suspect` state better: recall `{fmt(official_overlap['per_task']['sepsis']['per_class']['infection_suspect']['recall'])}` vs `{fmt(auto_overlap['per_task']['sepsis']['per_class']['infection_suspect']['recall'])}`"
    )
    lines.append("- in practice, auto is more terminal-alert-oriented on sepsis, while official is materially cleaner on AKI and somewhat cleaner on respiratory support")
    lines.append("")
    lines.append("The root cause is concept-layer instability, not missing tool use:")
    lines.append("")
    lines.append(f"- infection flag disagreement on overlap steps: `{divergence['infection_flag_disagree']}/{divergence['steps']}`")
    lines.append(
        f"- auto infection internal contradictions (`evidence` present while flag is false): `{divergence['auto_infection_internal_contradiction']}` steps"
    )
    lines.append(f"- AKI stage disagreement: `{divergence['aki_stage_disagree']}/{divergence['steps']}` steps")
    lines.append(f"- respiratory support disagreement: `{divergence['resp_support_disagree']}/{divergence['steps']}` steps")
    lines.append("")
    lines.append("## 1. Saved-Run Results")
    lines.append("")
    lines.append("These numbers use every completed trajectory found in each folder, even though the folders are not equally complete.")
    lines.append("")
    lines.append("### Official")
    lines.append("")
    lines.append(f"- joint step accuracy: `{fmt(official_full['joint_step_accuracy'])}`")
    lines.append(f"- sepsis accuracy / macro F1: `{fmt(official_full['per_task']['sepsis']['accuracy'])}` / `{fmt(official_full['per_task']['sepsis']['macro_f1'])}`")
    lines.append(f"- AKI accuracy / macro F1: `{fmt(official_full['per_task']['aki']['accuracy'])}` / `{fmt(official_full['per_task']['aki']['macro_f1'])}`")
    lines.append(
        f"- respiratory accuracy / macro F1: `{fmt(official_full['per_task']['respiratory_support']['accuracy'])}` / `{fmt(official_full['per_task']['respiratory_support']['macro_f1'])}`"
    )
    lines.append("")
    lines.append("### Autoformalized")
    lines.append("")
    lines.append(f"- joint step accuracy: `{fmt(auto_full['joint_step_accuracy'])}`")
    lines.append(f"- sepsis accuracy / macro F1: `{fmt(auto_full['per_task']['sepsis']['accuracy'])}` / `{fmt(auto_full['per_task']['sepsis']['macro_f1'])}`")
    lines.append(f"- AKI accuracy / macro F1: `{fmt(auto_full['per_task']['aki']['accuracy'])}` / `{fmt(auto_full['per_task']['aki']['macro_f1'])}`")
    lines.append(
        f"- respiratory accuracy / macro F1: `{fmt(auto_full['per_task']['respiratory_support']['accuracy'])}` / `{fmt(auto_full['per_task']['respiratory_support']['macro_f1'])}`"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- official is stronger on all three tasks in the saved-run view")
    lines.append("- but this view is still confounded by the missing saved trajectories, especially the smaller auto run")
    lines.append("")
    lines.append("## 2. Matched-Overlap Results")
    lines.append("")
    lines.append(f"The rest of the report uses the `{len(overlap_ids)}` shared trajectories.")
    lines.append("")
    lines.append("### Headline Metrics")
    lines.append("")
    lines.append(
        f"- joint step accuracy: official `{fmt(official_overlap['joint_step_accuracy'])}` vs auto `{fmt(auto_overlap['joint_step_accuracy'])}`"
    )
    lines.append(
        f"- sepsis accuracy: official `{fmt(official_overlap['per_task']['sepsis']['accuracy'])}` vs auto `{fmt(auto_overlap['per_task']['sepsis']['accuracy'])}`"
    )
    lines.append(
        f"- AKI accuracy: official `{fmt(official_overlap['per_task']['aki']['accuracy'])}` vs auto `{fmt(auto_overlap['per_task']['aki']['accuracy'])}`"
    )
    lines.append(
        f"- respiratory accuracy: official `{fmt(official_overlap['per_task']['respiratory_support']['accuracy'])}` vs auto `{fmt(auto_overlap['per_task']['respiratory_support']['accuracy'])}`"
    )
    lines.append("")
    lines.append("### Paired Correctness")
    lines.append("")
    lines.append(
        f"- joint steps where both are correct: `{paired['joint_step'].get('official_True__auto_True', 0)}`"
    )
    lines.append(
        f"- joint steps where only official is correct: `{paired['joint_step'].get('official_True__auto_False', 0)}`"
    )
    lines.append(
        f"- joint steps where only auto is correct: `{paired['joint_step'].get('official_False__auto_True', 0)}`"
    )
    lines.append(
        f"- trajectories solved perfectly only by official: `{paired['joint_trajectory'].get('official_True__auto_False', 0)}`"
    )
    lines.append(
        f"- trajectories solved perfectly only by auto: `{paired['joint_trajectory'].get('official_False__auto_True', 0)}`"
    )
    lines.append("")
    lines.append("Task-specific paired view:")
    lines.append("")
    lines.append(
        f"- sepsis steps only-official-correct / only-auto-correct: `{paired['per_task']['sepsis'].get('official_True__auto_False', 0)}` / `{paired['per_task']['sepsis'].get('official_False__auto_True', 0)}`"
    )
    lines.append(
        f"- AKI steps only-official-correct / only-auto-correct: `{paired['per_task']['aki'].get('official_True__auto_False', 0)}` / `{paired['per_task']['aki'].get('official_False__auto_True', 0)}`"
    )
    lines.append(
        f"- respiratory steps only-official-correct / only-auto-correct: `{paired['per_task']['respiratory_support'].get('official_True__auto_False', 0)}` / `{paired['per_task']['respiratory_support'].get('official_False__auto_True', 0)}`"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- the multitask gap is driven mostly by AKI, with respiratory support a clear secondary contributor")
    lines.append("- sepsis is closer: auto gets some extra terminal alerts right, while official better retains the intermediate surveillance state")
    lines.append("")
    lines.append("## 3. Transition Timing")
    lines.append("")
    lines.append("### Sepsis")
    lines.append("")
    lines.append(
        f"- infection_suspect exact match: official `{fmt(official_overlap['transition_timing']['sepsis']['infection_suspect']['exact_match_rate'])}` vs auto `{fmt(auto_overlap['transition_timing']['sepsis']['infection_suspect']['exact_match_rate'])}`"
    )
    lines.append(
        f"- infection_suspect missed rate: official `{fmt(official_overlap['transition_timing']['sepsis']['infection_suspect']['missed_rate'])}` vs auto `{fmt(auto_overlap['transition_timing']['sepsis']['infection_suspect']['missed_rate'])}`"
    )
    lines.append(
        f"- trigger_sepsis_alert exact match: official `{fmt(official_overlap['transition_timing']['sepsis']['trigger_sepsis_alert']['exact_match_rate'])}` vs auto `{fmt(auto_overlap['transition_timing']['sepsis']['trigger_sepsis_alert']['exact_match_rate'])}`"
    )
    lines.append(
        f"- trigger_sepsis_alert missed rate: official `{fmt(official_overlap['transition_timing']['sepsis']['trigger_sepsis_alert']['missed_rate'])}` vs auto `{fmt(auto_overlap['transition_timing']['sepsis']['trigger_sepsis_alert']['missed_rate'])}`"
    )
    lines.append("")
    lines.append("Sepsis takeaway:")
    lines.append("")
    lines.append("- official is much better at surfacing the intermediate infection stage at the right checkpoint")
    lines.append("- auto is more willing to jump to the terminal alert state, which helps some late-sepsis labels but harms the surveillance ladder")
    lines.append("")
    lines.append("### AKI")
    lines.append("")
    lines.append(
        f"- suspect_aki exact match: official `{fmt(official_overlap['transition_timing']['aki']['suspect_aki']['exact_match_rate'])}` vs auto `{fmt(auto_overlap['transition_timing']['aki']['suspect_aki']['exact_match_rate'])}`"
    )
    lines.append(
        f"- suspect_aki missed rate: official `{fmt(official_overlap['transition_timing']['aki']['suspect_aki']['missed_rate'])}` vs auto `{fmt(auto_overlap['transition_timing']['aki']['suspect_aki']['missed_rate'])}`"
    )
    lines.append(
        f"- trigger_aki_alert exact match: official `{fmt(official_overlap['transition_timing']['aki']['trigger_aki_alert']['exact_match_rate'])}` vs auto `{fmt(auto_overlap['transition_timing']['aki']['trigger_aki_alert']['exact_match_rate'])}`"
    )
    lines.append(
        f"- trigger_aki_alert missed rate: official `{fmt(official_overlap['transition_timing']['aki']['trigger_aki_alert']['missed_rate'])}` vs auto `{fmt(auto_overlap['transition_timing']['aki']['trigger_aki_alert']['missed_rate'])}`"
    )
    lines.append("")
    lines.append("AKI takeaway:")
    lines.append("")
    lines.append("- official is dramatically better at both stage-1 suspicion timing and stage-2-or-3 alert timing")
    lines.append("- auto misses over half of the intermediate AKI transitions and roughly two thirds of the severe AKI alerts on the overlap")
    lines.append("")
    lines.append("### Respiratory Support")
    lines.append("")
    lines.append(
        f"- invasive support exact match: official `{fmt(official_overlap['transition_timing']['respiratory_support']['invasive_vent_required']['exact_match_rate'])}` vs auto `{fmt(auto_overlap['transition_timing']['respiratory_support']['invasive_vent_required']['exact_match_rate'])}`"
    )
    lines.append(
        f"- invasive support missed rate: official `{fmt(official_overlap['transition_timing']['respiratory_support']['invasive_vent_required']['missed_rate'])}` vs auto `{fmt(auto_overlap['transition_timing']['respiratory_support']['invasive_vent_required']['missed_rate'])}`"
    )
    lines.append("")
    lines.append("Respiratory takeaway:")
    lines.append("")
    lines.append("- official is effectively perfect on the matched overlap")
    lines.append("- auto is still strong, but it introduces avoidable support-level drift")
    lines.append("")
    lines.append("## 4. Tool-Output Divergence")
    lines.append("")
    lines.append("Both agents always call the same tools, so the real question is what those tools expose.")
    lines.append("")
    lines.append("### Infection Tool")
    lines.append("")
    lines.append(f"- infection flag disagreement: `{divergence['infection_flag_disagree']}/{divergence['steps']}` steps")
    lines.append(
        f"- infection hour present in only one backend: `{divergence['infection_hour_presence_disagree']}/{divergence['steps']}` steps"
    )
    lines.append(
        f"- auto evidence-present / flag-false contradictions: `{divergence['auto_infection_internal_contradiction']}`"
    )
    lines.append("")
    lines.append("This is the clearest autoformalized failure mode. The backend often exposes infection evidence or a reconstructed first-visible hour, but still keeps `has_suspected_infection = false`. That breaks the benchmark contract at the concept level before the policy even acts.")
    lines.append("")
    lines.append("### SOFA Tool")
    lines.append("")
    lines.append(
        f"- auto SOFA at least 2 points higher than official: `{divergence['auto_sofa_ge2_higher']}` steps"
    )
    lines.append(
        f"- auto SOFA at least 2 points lower than official: `{divergence['auto_sofa_ge2_lower']}` steps"
    )
    lines.append(
        f"- auto minus official SOFA delta: mean `{fmt(divergence['sofa_delta_auto_minus_official']['mean'])}`, range `{fmt(divergence['sofa_delta_auto_minus_official']['min'])}` to `{fmt(divergence['sofa_delta_auto_minus_official']['max'])}`"
    )
    lines.append("")
    lines.append("The issue is not a simple monotone bias. Auto SOFA is unstable in both directions, which is consistent with it being a visible-prefix recomputation rather than a faithful wrapper over the official hourly rolling concept.")
    lines.append("")
    lines.append("### AKI Tool")
    lines.append("")
    lines.append(f"- AKI stage disagreement: `{divergence['aki_stage_disagree']}/{divergence['steps']}` steps")
    lines.append("")
    lines.append("The overlap examples show both early auto stage-1 overcalls and missed official-positive stages. That lines up with the large AKI timing and accuracy gap.")
    lines.append("")
    lines.append("### Respiratory Tool")
    lines.append("")
    lines.append(f"- respiratory support disagreement: `{divergence['resp_support_disagree']}/{divergence['steps']}` steps")
    lines.append("")
    lines.append("The dominant pattern is auto overcalling invasive support when raw strings such as `Endotracheal tube` remain visible, even when the official concept has already normalized the support state back down.")
    lines.append("")
    lines.append("## 5. Bottom Line")
    lines.append("")
    lines.append("The official backend should remain the reference multitask visible-concept layer.")
    lines.append("")
    lines.append("The strongest reasons are:")
    lines.append("")
    lines.append("- it wins clearly on the only fair summary metric here: matched-overlap joint accuracy")
    lines.append("- it is far more reliable on AKI, which is where multitask performance diverges most")
    lines.append("- it avoids the auto backend's infection inconsistency and respiratory support overcalls")
    lines.append("")
    lines.append("The autoformalized backend is not uniformly worse. On the matched overlap it is roughly competitive on sepsis macro F1 and slightly higher on sepsis step accuracy. But that comes with a more alert-heavy policy, much weaker intermediate infection recall, and large concept-level instability in AKI and respiratory support.")
    lines.append("")
    lines.append("So the right conclusion is:")
    lines.append("")
    lines.append("- autoformalized is promising as a concept-generation experiment")
    lines.append("- but in its current form it is not yet a drop-in replacement for the official multitask concept layer")
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append(f"- `{file_link(OFFICIAL_EVAL_PATH)}`")
    lines.append(f"- `{file_link(AUTO_EVAL_PATH)}`")
    lines.append(f"- `{file_link(COMPARISON_JSON_PATH)}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    dataset_size = count_dataset_stays(DATASET_PATH)
    official_rollouts = load_jsonl(OFFICIAL_TRAJ_PATH)
    auto_rollouts = load_jsonl(AUTO_TRAJ_PATH)

    official_full = evaluate_multitask_rollouts(official_rollouts)
    auto_full = evaluate_multitask_rollouts(auto_rollouts)

    official_ids = {rollout["trajectory_id"] for rollout in official_rollouts}
    auto_ids = {rollout["trajectory_id"] for rollout in auto_rollouts}
    overlap_ids = sorted(official_ids & auto_ids)
    official_only_ids = sorted(official_ids - auto_ids)
    auto_only_ids = sorted(auto_ids - official_ids)

    official_overlap_rollouts = [rollout for rollout in official_rollouts if rollout["trajectory_id"] in overlap_ids]
    auto_overlap_rollouts = [rollout for rollout in auto_rollouts if rollout["trajectory_id"] in overlap_ids]

    official_overlap = evaluate_multitask_rollouts(official_overlap_rollouts)
    auto_overlap = evaluate_multitask_rollouts(auto_overlap_rollouts)
    paired = paired_correctness(official_rollouts, auto_rollouts, overlap_ids)
    divergence = backend_divergence(official_rollouts, auto_rollouts, overlap_ids)

    official_eval = build_eval_summary("official", official_rollouts, dataset_size, official_full)
    auto_eval = build_eval_summary("autoformalized", auto_rollouts, dataset_size, auto_full)

    OFFICIAL_EVAL_PATH.write_text(json.dumps(official_eval, indent=2))
    AUTO_EVAL_PATH.write_text(json.dumps(auto_eval, indent=2))

    comparison_payload = {
        "dataset_size": dataset_size,
        "official_completed_trajectories": len(official_rollouts),
        "auto_completed_trajectories": len(auto_rollouts),
        "overlap_trajectories": len(overlap_ids),
        "official_only_trajectories": official_only_ids,
        "auto_only_trajectories": auto_only_ids,
        "official_full_metrics": official_full,
        "auto_full_metrics": auto_full,
        "official_overlap_metrics": official_overlap,
        "auto_overlap_metrics": auto_overlap,
        "paired_correctness": paired,
        "backend_divergence": divergence,
    }
    COMPARISON_JSON_PATH.write_text(json.dumps(comparison_payload, indent=2))

    report = build_report(
        dataset_size=dataset_size,
        official_rollouts=official_rollouts,
        auto_rollouts=auto_rollouts,
        official_full=official_full,
        auto_full=auto_full,
        official_overlap=official_overlap,
        auto_overlap=auto_overlap,
        overlap_ids=overlap_ids,
        official_only_ids=official_only_ids,
        auto_only_ids=auto_only_ids,
        paired=paired,
        divergence=divergence,
    )
    REPORT_PATH.write_text(report)

    print(json.dumps(
        {
            "official_eval": str(OFFICIAL_EVAL_PATH),
            "auto_eval": str(AUTO_EVAL_PATH),
            "comparison_json": str(COMPARISON_JSON_PATH),
            "report": str(REPORT_PATH),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
