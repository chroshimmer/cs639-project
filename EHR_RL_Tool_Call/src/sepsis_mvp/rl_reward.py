from __future__ import annotations

from collections import Counter
from typing import Any

from .environment import (
    _empty_toolbox_state,
    _tool_call_has_marginal_utility,
    _update_toolbox_state_from_output,
)
from .schemas import Trajectory, TrajectoryRollout


def _action_reward(gt: str | None, pred: str | None) -> float:
    if pred == gt:
        return 1.0
    if {gt, pred} == {"infection_suspect", "trigger_sepsis_alert"}:
        return 0.2
    if gt == "keep_monitoring" and pred == "trigger_sepsis_alert":
        return -0.5
    if gt == "trigger_sepsis_alert" and pred != "trigger_sepsis_alert":
        return -0.7
    return -0.2


def _evidence_reward(pred: str | None, state: dict[str, Any]) -> tuple[float, dict[str, bool]]:
    flags = {
        "positive_action_without_sufficient_evidence": False,
        "alert_without_sofa_evidence": False,
    }
    if pred == "infection_suspect":
        if state["infection_positive"]:
            return 0.3, flags
        flags["positive_action_without_sufficient_evidence"] = True
        return -0.7, flags
    if pred == "trigger_sepsis_alert":
        if state["infection_positive"] and state["sofa_alert"]:
            return 0.5, flags
        flags["positive_action_without_sufficient_evidence"] = True
        if not state["sofa_alert"]:
            flags["alert_without_sofa_evidence"] = True
        return -0.7, flags
    return 0.0, flags


def _timing_bonus(gt_hour: int | None, pred_hour: int | None, *, missed_penalty: float, early_penalty: float) -> float:
    if gt_hour is None and pred_hour is None:
        return 0.0
    if gt_hour is None and pred_hour is not None:
        return early_penalty
    if gt_hour is not None and pred_hour is None:
        return missed_penalty
    assert gt_hour is not None and pred_hour is not None
    if gt_hour == pred_hour:
        return 1.0
    return max(-1.0, -0.1 * (abs(pred_hour - gt_hour) / 4.0))


def _trajectory_timing_reward(trajectory: Trajectory, rollout: TrajectoryRollout) -> dict[str, float]:
    transitions = trajectory.transitions or {}
    predicted = (rollout.first_predicted_task_hours or {}).get("sepsis", {})
    infection_pred = predicted.get("infection_suspect", rollout.first_predicted_infection_hour)
    alert_pred = predicted.get("trigger_sepsis_alert", rollout.first_predicted_alert_hour)
    infection_reward = _timing_bonus(
        transitions.get("infection_start_hour"),
        infection_pred,
        missed_penalty=-0.5,
        early_penalty=-0.3,
    )
    alert_reward = _timing_bonus(
        transitions.get("sepsis_start_hour"),
        alert_pred,
        missed_penalty=-1.0,
        early_penalty=-0.5,
    )
    return {
        "infection_timing_reward": round(infection_reward, 4),
        "alert_timing_reward": round(alert_reward, 4),
        "trajectory_timing_reward": round(infection_reward + alert_reward, 4),
    }


def score_sepsis_rollout(
    trajectory: Trajectory,
    rollout: TrajectoryRollout,
    *,
    include_timing: bool = True,
) -> dict[str, Any]:
    if trajectory.primary_task_name() != "sepsis":
        raise ValueError("Sepsis reward scorer only supports single-task sepsis trajectories.")

    state = _empty_toolbox_state()
    step_scores: list[dict[str, Any]] = []
    counters = Counter()

    for step in rollout.steps:
        before_step_state = dict(state)
        tool_reward = 0.0
        useful_calls = 0
        repeated_calls = 0
        seen_this_step: set[str] = set()
        current_outputs: dict[str, dict[str, Any]] = {}

        for call_payload, output in zip(step.tool_calls or [], step.tool_outputs or []):
            tool_name = call_payload.get("tool_name")
            if not tool_name:
                continue
            current_outputs[tool_name] = output
            pre_call_state = dict(state)
            tool_reward -= 0.03
            if tool_name in seen_this_step:
                repeated_calls += 1
                tool_reward -= 0.1
            if tool_name == "query_suspicion_of_infection" and pre_call_state["infection_positive"]:
                tool_reward -= 0.2
            if _tool_call_has_marginal_utility(tool_name, output, pre_call_state):
                useful_calls += 1
                tool_reward += 0.1
            _update_toolbox_state_from_output(state, tool_name, output)
            seen_this_step.add(tool_name)

        pred = step.predicted_action
        action_reward = _action_reward(step.gt_action, pred)
        evidence_reward, evidence_flags = _evidence_reward(pred, state)
        necessary_reward = 0.0

        if pred in {"infection_suspect", "trigger_sepsis_alert"} and not before_step_state["infection_positive"]:
            counters["necessary_infection_total"] += 1
            if "query_suspicion_of_infection" in current_outputs:
                counters["necessary_infection_covered"] += 1
                necessary_reward += 0.15
            else:
                necessary_reward -= 0.15
        if pred == "trigger_sepsis_alert" and not before_step_state["sofa_alert"]:
            counters["necessary_sofa_total"] += 1
            if "query_sofa" in current_outputs:
                counters["necessary_sofa_covered"] += 1
                necessary_reward += 0.2
            else:
                necessary_reward -= 0.5

        format_reward = 0.1 if pred in {"keep_monitoring", "infection_suspect", "trigger_sepsis_alert"} else -0.5
        safety_reward = 0.0
        for call_payload in step.tool_calls or []:
            args = call_payload.get("arguments") or {}
            if args.get("stay_id") not in {None, trajectory.stay_id}:
                safety_reward -= 1.0
            if args.get("t_hour") not in {None, step.t_hour}:
                safety_reward -= 1.0

        total = action_reward + evidence_reward + tool_reward + necessary_reward + format_reward + safety_reward
        total = max(-2.0, min(3.0, total))
        if evidence_flags["positive_action_without_sufficient_evidence"]:
            counters["positive_action_without_sufficient_evidence"] += 1
        if evidence_flags["alert_without_sofa_evidence"]:
            counters["alert_without_sofa_evidence"] += 1
        counters["steps"] += 1
        counters["tool_calls"] += len(step.tool_calls or [])
        counters["useful_tool_calls"] += useful_calls
        counters["repeated_calls"] += repeated_calls

        step_scores.append(
            {
                "step_index": step.step_index,
                "t_hour": step.t_hour,
                "gt_action": step.gt_action,
                "predicted_action": pred,
                "reward": round(total, 4),
                "components": {
                    "action": round(action_reward, 4),
                    "evidence": round(evidence_reward, 4),
                    "tool_efficiency": round(tool_reward, 4),
                    "necessary_call": round(necessary_reward, 4),
                    "format": round(format_reward, 4),
                    "safety": round(safety_reward, 4),
                },
                "state_after_step": {
                    "infection_positive": state["infection_positive"],
                    "sofa_alert": state["sofa_alert"],
                    "max_sofa": state["max_sofa"],
                },
            }
        )

    mean_step_reward = sum(item["reward"] for item in step_scores) / len(step_scores) if step_scores else 0.0
    timing = _trajectory_timing_reward(trajectory, rollout) if include_timing else {
        "infection_timing_reward": 0.0,
        "alert_timing_reward": 0.0,
        "trajectory_timing_reward": 0.0,
    }
    trajectory_reward = mean_step_reward + (0.25 * timing["trajectory_timing_reward"])
    return {
        "trajectory_id": rollout.trajectory_id,
        "stay_id": rollout.stay_id,
        "mean_step_reward": round(mean_step_reward, 4),
        "trajectory_reward": round(max(-2.0, min(3.0, trajectory_reward)), 4),
        "timing": timing,
        "counters": dict(counters),
        "steps": step_scores,
    }


def score_sepsis_rollouts(
    trajectories: list[Trajectory],
    rollouts: list[TrajectoryRollout],
) -> dict[str, Any]:
    trajectory_by_id = {trajectory.trajectory_id: trajectory for trajectory in trajectories}
    scored = [
        score_sepsis_rollout(trajectory_by_id[rollout.trajectory_id], rollout)
        for rollout in rollouts
        if rollout.trajectory_id in trajectory_by_id
    ]
    if not scored:
        return {"num_trajectories": 0, "mean_trajectory_reward": 0.0, "trajectory_scores": []}
    total_counters = Counter()
    for item in scored:
        total_counters.update(item["counters"])
    return {
        "num_trajectories": len(scored),
        "mean_trajectory_reward": round(
            sum(item["trajectory_reward"] for item in scored) / len(scored),
            4,
        ),
        "mean_step_reward": round(
            sum(item["mean_step_reward"] for item in scored) / len(scored),
            4,
        ),
        "aggregate_counters": dict(total_counters),
        "trajectory_scores": scored,
    }

