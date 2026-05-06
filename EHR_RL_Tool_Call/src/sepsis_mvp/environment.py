from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import time
from typing import Any

from .agent import Agent
from .schemas import (
    ActionDecision,
    AgentStepInput,
    CODE_EXEC_TOOL_NAME,
    SHARED_TOOLBOX_TOOL_NAMES,
    SEPSIS_CORE_TOOL_NAMES,
    SQL_EXEC_TOOL_NAME,
    TASK_BASELINE_ACTION,
    TASK_LABEL_SPACES,
    TASK_TOOL_NAMES,
    TASK_TRANSITION_FIELDS,
    Trajectory,
    TrajectoryRollout,
    StepRecord,
    ToolCall,
)
from .tools import ToolRuntime


class BenchmarkEnvironment:
    SUPPORTED_PROTOCOLS = {"rolling_no_history", "rolling_with_history", "rolling_toolbox_with_history"}

    def __init__(
        self,
        trajectories: list[Trajectory],
        tool_runtime: ToolRuntime,
        *,
        max_tool_calls_per_step: int = 4,
        event_callback: Any | None = None,
        tool_backend: str = "official",
        tool_scope: str = "shared",
        task_mode: str = "auto",
        protocol: str = "rolling_no_history",
    ) -> None:
        self.trajectories = trajectories
        self.tool_runtime = tool_runtime
        self.max_tool_calls_per_step = max_tool_calls_per_step
        self.event_callback = event_callback
        self.tool_backend = tool_backend
        self.tool_scope = tool_scope
        self.task_mode = task_mode
        if protocol not in self.SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"Unsupported protocol '{protocol}'. Supported protocols: {sorted(self.SUPPORTED_PROTOCOLS)}"
            )
        self.protocol = protocol

    @staticmethod
    def _is_surveillance_trajectory(trajectory: Trajectory) -> bool:
        return trajectory.primary_task_name() == "general_icu_surveillance"

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event)

    def run_trajectory(self, trajectory: Trajectory, agent: Agent) -> TrajectoryRollout:
        steps: list[StepRecord] = []
        first_predicted_infection_hour = None
        first_predicted_alert_hour = None
        first_predicted_task_hours: dict[str, dict[str, int | None]] = defaultdict(dict)
        task_names = trajectory.resolved_task_names()
        task_mode = self._resolve_task_mode(trajectory)

        self._emit(
            {
                "event_type": "trajectory_start",
                "trajectory_id": trajectory.trajectory_id,
                "stay_id": trajectory.stay_id,
                "task_names": task_names,
                "task_mode": task_mode,
                "tool_backend": self.tool_backend,
            }
        )

        available_tools = self._available_tools_for_protocol(trajectory)
        rolling_history: list[dict[str, Any]] = []
        use_rolling_history = (
            self.protocol in {"rolling_with_history", "rolling_toolbox_with_history"}
            and (self.tool_backend != "zeroshot_python" or self._is_surveillance_trajectory(trajectory))
        )

        for step_index, checkpoint in enumerate(trajectory.checkpoints):
            step_started = time.perf_counter()
            instruction = self._instruction_for_trajectory(trajectory)
            step_rolling_history = (
                list(rolling_history)
                if use_rolling_history
                else []
            )
            step_max_interactions = self._max_step_interactions_for_protocol(available_tools)
            step_input = AgentStepInput(
                trajectory_id=trajectory.trajectory_id,
                stay_id=trajectory.stay_id,
                step_index=step_index,
                t_hour=checkpoint.t_hour,
                available_tools=available_tools,
                instruction=instruction,
                task_names=task_names,
                task_variant=trajectory.task_variant,
                label_spaces=trajectory.label_spaces,
                task_mode=task_mode,
                tool_backend=self.tool_backend,
                max_step_interactions=step_max_interactions,
                protocol=self.protocol,
                rolling_history=step_rolling_history,
            )
            history: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []
            tool_outputs: list[dict[str, Any]] = []
            step_resource_usage = _empty_step_resource_usage()
            primary_task = trajectory.primary_task_name()
            predicted_action: str | None = TASK_BASELINE_ACTION.get(primary_task)
            predicted_task_actions: dict[str, str] | None = None
            predicted_surveillance: dict[str, Any] | None = None
            step_session_id = None
            if hasattr(self.tool_runtime, "start_step_session"):
                step_session_id = self.tool_runtime.start_step_session(
                    stay_id=trajectory.stay_id,
                    t_hour=checkpoint.t_hour,
                )

            self._emit(
                {
                    "event_type": "step_start",
                    "trajectory_id": trajectory.trajectory_id,
                    "stay_id": trajectory.stay_id,
                    "step_index": step_index,
                    "t_hour": checkpoint.t_hour,
                    "gt_action": checkpoint.state_label,
                    "gt_task_actions": checkpoint.task_labels,
                    "protocol": self.protocol,
                    "rolling_history_length": len(step_rolling_history),
                    "available_tools": available_tools,
                }
            )

            try:
                for _ in range(step_max_interactions + 1):
                    agent_call_started = time.perf_counter()
                    response = agent.next_response(
                        step_input=step_input.to_dict(),
                        history=history,
                        available_tools=step_input.available_tools,
                    )
                    step_resource_usage["agent_calls"] += 1
                    step_resource_usage["agent_runtime_sec"] += time.perf_counter() - agent_call_started
                    _merge_agent_response_metrics(step_resource_usage, _pop_agent_response_metrics(agent))
                    if isinstance(response, ToolCall):
                        arguments = dict(response.arguments)
                        if response.tool_name in {CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME} and step_session_id is not None:
                            arguments["session_id"] = step_session_id
                        tool_call_started = time.perf_counter()
                        output = self.tool_runtime.execute(response.tool_name, arguments)
                        step_resource_usage["tool_calls"] += 1
                        step_resource_usage["tool_runtime_sec"] += time.perf_counter() - tool_call_started
                        call_payload = {"tool_name": response.tool_name, "arguments": arguments}
                        self._emit(
                            {
                                "event_type": "tool_call",
                                "trajectory_id": trajectory.trajectory_id,
                                "stay_id": trajectory.stay_id,
                                "step_index": step_index,
                                "t_hour": checkpoint.t_hour,
                                "tool_name": response.tool_name,
                                "arguments": arguments,
                            }
                        )
                        history.append({"type": "tool_call", "tool_name": response.tool_name, "payload": call_payload})
                        history.append({"type": "tool_output", "tool_name": response.tool_name, "payload": output})
                        tool_calls.append(call_payload)
                        tool_outputs.append(output)
                        self._emit(
                            {
                                "event_type": "tool_output",
                                "trajectory_id": trajectory.trajectory_id,
                                "stay_id": trajectory.stay_id,
                                "step_index": step_index,
                                "t_hour": checkpoint.t_hour,
                                "tool_name": response.tool_name,
                                "output": output,
                            }
                        )
                        continue
                    if isinstance(response, ActionDecision):
                        predicted_action = response.action
                        predicted_task_actions = response.task_actions
                        predicted_surveillance = response.surveillance
                        self._emit(
                            {
                                "event_type": "action",
                                "trajectory_id": trajectory.trajectory_id,
                                "stay_id": trajectory.stay_id,
                                "step_index": step_index,
                                "t_hour": checkpoint.t_hour,
                                "gt_action": checkpoint.state_label,
                                "gt_task_actions": checkpoint.task_labels,
                                "gt_surveillance": checkpoint.surveillance_labels,
                                "predicted_action": predicted_action,
                                "predicted_task_actions": predicted_task_actions,
                                "predicted_surveillance": predicted_surveillance,
                            }
                        )
                        break
                    raise TypeError(f"Unsupported agent response: {response}")
            finally:
                if hasattr(self.tool_runtime, "close_step_session"):
                    self.tool_runtime.close_step_session(step_session_id)

            if self._is_surveillance_trajectory(trajectory):
                predicted_action = (predicted_surveillance or {}).get("global_action", predicted_action)
            elif not trajectory.is_multitask():
                if (
                    predicted_action is not None
                    and predicted_action != TASK_BASELINE_ACTION.get(primary_task)
                ):
                    first_predicted_task_hours[primary_task].setdefault(predicted_action, checkpoint.t_hour)
                if primary_task in {"sepsis", "infection_only"}:
                    if predicted_action == "infection_suspect" and first_predicted_infection_hour is None:
                        first_predicted_infection_hour = checkpoint.t_hour
                    if primary_task == "sepsis" and predicted_action == "trigger_sepsis_alert":
                        if first_predicted_infection_hour is None:
                            first_predicted_infection_hour = checkpoint.t_hour
                        if first_predicted_alert_hour is None:
                            first_predicted_alert_hour = checkpoint.t_hour
            else:
                for task_name, action in (predicted_task_actions or {}).items():
                    if action != TASK_BASELINE_ACTION.get(task_name):
                        first_predicted_task_hours[task_name].setdefault(action, checkpoint.t_hour)

            steps.append(
                StepRecord(
                    step_index=step_index,
                    t_hour=checkpoint.t_hour,
                    gt_action=checkpoint.state_label,
                    predicted_action=predicted_action,
                    gt_task_actions=checkpoint.task_labels,
                    predicted_task_actions=predicted_task_actions,
                    gt_surveillance=checkpoint.surveillance_labels,
                    predicted_surveillance=predicted_surveillance,
                    tool_calls=tool_calls,
                    tool_outputs=tool_outputs,
                    resource_usage=_finalize_step_resource_usage(
                        step_resource_usage,
                        step_runtime_sec=time.perf_counter() - step_started,
                    ),
                )
            )
            history_entry = _build_rolling_history_entry(
                trajectory=trajectory,
                checkpoint=checkpoint,
                step_index=step_index,
                tool_outputs=tool_outputs,
                predicted_surveillance=predicted_surveillance,
            )
            if (
                history_entry is not None
                and use_rolling_history
            ):
                rolling_history.append(history_entry)

        rollout = TrajectoryRollout(
            trajectory_id=trajectory.trajectory_id,
            stay_id=trajectory.stay_id,
            steps=steps,
            first_predicted_infection_hour=first_predicted_infection_hour,
            first_predicted_alert_hour=first_predicted_alert_hour,
            first_predicted_task_hours=dict(first_predicted_task_hours) if first_predicted_task_hours else None,
            resource_usage=_aggregate_rollout_resource_usage(steps),
        )
        self._emit(
            {
                "event_type": "trajectory_complete",
                "trajectory_id": trajectory.trajectory_id,
                "stay_id": trajectory.stay_id,
                "task_mode": task_mode,
                "tool_backend": self.tool_backend,
                "protocol": self.protocol,
                "first_predicted_infection_hour": first_predicted_infection_hour,
                "first_predicted_alert_hour": first_predicted_alert_hour,
                "first_predicted_task_hours": rollout.first_predicted_task_hours or None,
                "num_steps": len(steps),
            }
        )
        return rollout

    def run_all(self, agent: Agent) -> list[TrajectoryRollout]:
        return [self.run_trajectory(trajectory, agent) for trajectory in self.trajectories]

    def _resolve_task_mode(self, trajectory: Trajectory) -> str:
        if self._is_surveillance_trajectory(trajectory):
            inferred = "surveillance"
        else:
            inferred = "multitask" if trajectory.is_multitask() else "single"
        if self.task_mode == "auto":
            return inferred
        if self.task_mode != inferred:
            raise ValueError(
                f"Dataset/task-mode mismatch: trajectory {trajectory.trajectory_id} is {inferred}, "
                f"but runner requested {self.task_mode}."
        )
        return self.task_mode

    def _available_tools_for_protocol(self, trajectory: Trajectory) -> list[str]:
        if self.tool_backend == "zeroshot_python":
            return [CODE_EXEC_TOOL_NAME]
        if self.tool_backend in {"zeroshot_sql", "zeroshot_raw"}:
            if trajectory.primary_task_name() == "infection_only":
                return [SQL_EXEC_TOOL_NAME]
            return [CODE_EXEC_TOOL_NAME]
        if self.protocol == "rolling_toolbox_with_history":
            if self.tool_scope == "sepsis_core" and trajectory.primary_task_name() == "sepsis" and not trajectory.is_multitask():
                return list(SEPSIS_CORE_TOOL_NAMES)
            return list(SHARED_TOOLBOX_TOOL_NAMES)
        return trajectory.resolved_tool_names()

    def _max_step_interactions_for_protocol(self, available_tools: list[str]) -> int:
        if self.protocol == "rolling_toolbox_with_history":
            return max(self.max_tool_calls_per_step, len(available_tools) + 2)
        return self.max_tool_calls_per_step

    def _instruction_for_trajectory(self, trajectory: Trajectory) -> str:
        if self._is_surveillance_trajectory(trajectory):
            return (
                "Use the checkpoint-scoped DuckDB session to gather evidence if needed. "
                "Then return one final surveillance decision with global_action, suspected_conditions, alerts, priority, "
                "recommended_next_tools, rationale, and checkpoint_summary."
            )
        if trajectory.is_multitask():
            return "Use tools if needed. Then output one decision for each monitored task."
        task_name = trajectory.primary_task_name()
        label_space = (trajectory.label_spaces or {}).get(task_name, TASK_LABEL_SPACES[task_name])
        return (
            f"Use tools if needed. Then output exactly one action for task '{task_name}': "
            + ", ".join(label_space)
            + "."
        )


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_rollouts(
    trajectories: list[Trajectory],
    rollouts: list[TrajectoryRollout],
    *,
    protocol: str | None = None,
) -> dict[str, Any]:
    if trajectories and trajectories[0].primary_task_name() == "general_icu_surveillance":
        metrics = evaluate_surveillance_rollouts(trajectories, rollouts)
    elif trajectories and trajectories[0].is_multitask():
        metrics = evaluate_multitask_rollouts(trajectories, rollouts)
    else:
        metrics = evaluate_single_task_rollouts(trajectories, rollouts, protocol=protocol)
    if protocol == "rolling_toolbox_with_history" and trajectories:
        if not trajectories[0].is_multitask() and trajectories[0].primary_task_name() == "sepsis":
            metrics["toolbox_efficiency"] = evaluate_sepsis_toolbox_efficiency(rollouts)
        else:
            metrics["toolbox_efficiency"] = evaluate_generic_toolbox_efficiency(rollouts)
    metrics["resource_usage"] = evaluate_resource_usage(rollouts)
    return metrics


def _set_f1(gt_items: list[str], pred_items: list[str]) -> tuple[float, float, float]:
    gt = set(gt_items)
    pred = set(pred_items)
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if tp + fp else 1.0 if not pred and not gt else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0 if not pred and not gt else 0.0
    return precision, recall, _f1(tp, fp, fn)


def evaluate_surveillance_rollouts(
    trajectories: list[Trajectory],
    rollouts: list[TrajectoryRollout],
) -> dict[str, Any]:
    trajectory_by_id = {trajectory.trajectory_id: trajectory for trajectory in trajectories}
    total_steps = 0
    global_action_correct = 0
    priority_correct = 0
    exact_suspected_match = 0
    exact_alert_match = 0
    suspected_f1_total = 0.0
    alerts_f1_total = 0.0
    alerts_precision_total = 0.0
    alerts_recall_total = 0.0
    first_alert_timing_pairs: list[tuple[int | None, int | None]] = []

    for rollout in rollouts:
        trajectory = trajectory_by_id[rollout.trajectory_id]
        gt_first_alert_hour = None
        pred_first_alert_hour = None
        for checkpoint, step in zip(trajectory.checkpoints, rollout.steps):
            total_steps += 1
            gt = checkpoint.surveillance_labels or {}
            pred = step.predicted_surveillance or {}
            gt_action = gt.get("global_action")
            pred_action = pred.get("global_action")
            if gt_action == pred_action:
                global_action_correct += 1
            gt_priority = gt.get("priority")
            pred_priority = pred.get("priority")
            if gt_priority == pred_priority:
                priority_correct += 1

            gt_suspected = gt.get("suspected_conditions") or []
            pred_suspected = pred.get("suspected_conditions") or []
            gt_alerts = gt.get("alerts") or []
            pred_alerts = pred.get("alerts") or []

            if set(gt_suspected) == set(pred_suspected):
                exact_suspected_match += 1
            if set(gt_alerts) == set(pred_alerts):
                exact_alert_match += 1

            _, _, suspected_f1 = _set_f1(gt_suspected, pred_suspected)
            alert_precision, alert_recall, alert_f1 = _set_f1(gt_alerts, pred_alerts)
            suspected_f1_total += suspected_f1
            alerts_f1_total += alert_f1
            alerts_precision_total += alert_precision
            alerts_recall_total += alert_recall

            if gt_first_alert_hour is None and gt_alerts:
                gt_first_alert_hour = checkpoint.t_hour
            if pred_first_alert_hour is None and pred_alerts:
                pred_first_alert_hour = checkpoint.t_hour

        first_alert_timing_pairs.append((gt_first_alert_hour, pred_first_alert_hour))

    timing_errors = [
        pred - gt
        for gt, pred in first_alert_timing_pairs
        if gt is not None and pred is not None
    ]
    false_early_alerts = sum(
        1
        for gt, pred in first_alert_timing_pairs
        if gt is not None and pred is not None and pred < gt
    )
    missed_alert_trajectories = sum(1 for gt, pred in first_alert_timing_pairs if gt is not None and pred is None)

    return {
        "task_name": "general_icu_surveillance",
        "step_level": {
            "global_action_accuracy": round(global_action_correct / total_steps, 4) if total_steps else 0.0,
            "priority_accuracy": round(priority_correct / total_steps, 4) if total_steps else 0.0,
            "suspected_conditions_exact_match": round(exact_suspected_match / total_steps, 4) if total_steps else 0.0,
            "alerts_exact_match": round(exact_alert_match / total_steps, 4) if total_steps else 0.0,
            "suspected_conditions_macro_f1": round(suspected_f1_total / total_steps, 4) if total_steps else 0.0,
            "alerts_macro_precision": round(alerts_precision_total / total_steps, 4) if total_steps else 0.0,
            "alerts_macro_recall": round(alerts_recall_total / total_steps, 4) if total_steps else 0.0,
            "alerts_macro_f1": round(alerts_f1_total / total_steps, 4) if total_steps else 0.0,
        },
        "timing": {
            "first_alert_mean_error_hours": round(sum(timing_errors) / len(timing_errors), 4) if timing_errors else None,
            "first_alert_mean_abs_error_hours": round(sum(abs(err) for err in timing_errors) / len(timing_errors), 4)
            if timing_errors
            else None,
            "false_early_alert_trajectories": false_early_alerts,
            "missed_alert_trajectories": missed_alert_trajectories,
        },
    }


def evaluate_single_task_rollouts(
    trajectories: list[Trajectory],
    rollouts: list[TrajectoryRollout],
    *,
    protocol: str | None = None,
) -> dict[str, Any]:
    trajectory_by_id = {trajectory.trajectory_id: trajectory for trajectory in trajectories}
    if not trajectories:
        return {}
    task_name = trajectories[0].primary_task_name()
    label_space = trajectories[0].label_spaces.get(task_name, TASK_LABEL_SPACES[task_name]) if trajectories[0].label_spaces else TASK_LABEL_SPACES[task_name]
    if _is_non_monotonic_aki_task(trajectories[0], label_space):
        return evaluate_single_task_current_state_rollouts(trajectories, rollouts)
    total_steps = 0
    correct_steps = 0
    confusion = Counter()
    per_class = {}
    transition_fields = TASK_TRANSITION_FIELDS.get(task_name, {})
    gt_hours_by_action = {action: [] for action in transition_fields}
    pred_hours_by_action = {action: [] for action in transition_fields}
    grounding_counts = _single_task_grounding_counters(task_name, label_space)

    use_history_aware_grounding = protocol == "rolling_toolbox_with_history"

    for rollout in rollouts:
        trajectory = trajectory_by_id[rollout.trajectory_id]
        prior_state = _empty_toolbox_state() if use_history_aware_grounding else None
        for step in rollout.steps:
            total_steps += 1
            if step.gt_action == step.predicted_action:
                correct_steps += 1
            confusion[(step.gt_action, step.predicted_action)] += 1
            has_exec_tool = any(
                call["tool_name"] in {CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME}
                for call in (step.tool_calls or [])
            )
            if use_history_aware_grounding and prior_state is not None and not has_exec_tool:
                current_state = dict(prior_state)
                for call_payload, output in zip(step.tool_calls or [], step.tool_outputs or []):
                    _update_toolbox_state_from_output(current_state, call_payload["tool_name"], output)
                _update_single_task_grounding_from_state(
                    task_name,
                    step.predicted_action,
                    grounding_counts,
                    current_state,
                    label_space,
                )
                prior_state = current_state
            else:
                _update_single_task_grounding(
                    task_name,
                    step.predicted_action,
                    step.tool_calls,
                    grounding_counts,
                    label_space,
                )

        predicted_hours = (rollout.first_predicted_task_hours or {}).get(task_name, {})
        for action, transition_field in transition_fields.items():
            gt_hours_by_action[action].append(trajectory.transitions.get(transition_field))
            pred_hours_by_action[action].append(predicted_hours.get(action))

    for action in label_space:
        tp = confusion[(action, action)]
        fp = sum(confusion[(gt, action)] for gt in label_space if gt != action)
        fn = sum(confusion[(action, pred)] for pred in label_space if pred != action)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[action] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(_f1(tp, fp, fn), 4),
        }

    macro_f1 = round(sum(metrics["f1"] for metrics in per_class.values()) / len(label_space), 4)
    transition_timing = {
        action: _timing_metrics(gt_hours_by_action[action], pred_hours_by_action[action])
        for action in transition_fields
    }
    transition_timing = _format_single_task_transition_timing(task_name, transition_timing)

    return {
        "task_name": task_name,
        "step_level": {
            "accuracy": round(correct_steps / total_steps, 4) if total_steps else 0.0,
            "macro_f1": macro_f1,
            "per_class": per_class,
        },
        "transition_timing": transition_timing,
        "tool_grounding": _finalize_single_task_grounding(grounding_counts),
    }


def evaluate_single_task_current_state_rollouts(
    trajectories: list[Trajectory],
    rollouts: list[TrajectoryRollout],
) -> dict[str, Any]:
    trajectory_by_id = {trajectory.trajectory_id: trajectory for trajectory in trajectories}
    task_name = trajectories[0].primary_task_name()
    label_space = trajectories[0].label_spaces.get(task_name, TASK_LABEL_SPACES[task_name]) if trajectories[0].label_spaces else TASK_LABEL_SPACES[task_name]
    total_steps = 0
    correct_steps = 0
    confusion = Counter()
    per_class = {}
    change_counts = {
        "worsening": {"tp": 0, "fp": 0, "fn": 0},
        "recovery": {"tp": 0, "fp": 0, "fn": 0},
    }
    exact_path_matches = 0
    grounding_counts = _single_task_grounding_counters(task_name, label_space)

    for rollout in rollouts:
        trajectory = trajectory_by_id[rollout.trajectory_id]
        gt_path: list[str] = []
        pred_path: list[str] = []
        for step in rollout.steps:
            total_steps += 1
            if step.gt_action == step.predicted_action:
                correct_steps += 1
            confusion[(step.gt_action, step.predicted_action)] += 1
            _update_single_task_grounding(task_name, step.predicted_action, step.tool_calls, grounding_counts, label_space)
            if step.gt_action is not None:
                gt_path.append(step.gt_action)
            if step.predicted_action is not None:
                pred_path.append(step.predicted_action)

        if gt_path == pred_path:
            exact_path_matches += 1

        for idx in range(1, min(len(gt_path), len(pred_path))):
            gt_delta = _state_order(label_space, gt_path[idx]) - _state_order(label_space, gt_path[idx - 1])
            pred_delta = _state_order(label_space, pred_path[idx]) - _state_order(label_space, pred_path[idx - 1])
            _update_change_counts(change_counts["worsening"], gt_delta > 0, pred_delta > 0)
            _update_change_counts(change_counts["recovery"], gt_delta < 0, pred_delta < 0)

    for action in label_space:
        tp = confusion[(action, action)]
        fp = sum(confusion[(gt, action)] for gt in label_space if gt != action)
        fn = sum(confusion[(action, pred)] for pred in label_space if pred != action)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[action] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(_f1(tp, fp, fn), 4),
        }

    return {
        "task_name": task_name,
        "task_variant": trajectories[0].task_variant,
        "step_level": {
            "accuracy": round(correct_steps / total_steps, 4) if total_steps else 0.0,
            "macro_f1": round(sum(metrics["f1"] for metrics in per_class.values()) / len(label_space), 4),
            "per_class": per_class,
        },
        "state_change": {
            "worsening": _event_metrics(change_counts["worsening"]),
            "recovery": _event_metrics(change_counts["recovery"]),
            "exact_path_match_rate": round(exact_path_matches / len(rollouts), 4) if rollouts else 0.0,
        },
        "tool_grounding": _finalize_single_task_grounding(grounding_counts),
    }


def evaluate_multitask_rollouts(trajectories: list[Trajectory], rollouts: list[TrajectoryRollout]) -> dict[str, Any]:
    total_steps = 0
    joint_correct = 0
    per_task_counts = defaultdict(int)
    per_task_correct = defaultdict(int)
    per_task_confusion = defaultdict(Counter)
    grounding = {
        "sepsis": {"num": 0, "den": 0},
        "aki": {"num": 0, "den": 0},
        "respiratory_support": {"num": 0, "den": 0},
    }

    for rollout in rollouts:
        for step in rollout.steps:
            total_steps += 1
            gt = step.gt_task_actions or {}
            pred = step.predicted_task_actions or {}
            if gt == pred:
                joint_correct += 1
            for task_name, gt_action in gt.items():
                pred_action = pred.get(task_name)
                per_task_counts[task_name] += 1
                if gt_action == pred_action:
                    per_task_correct[task_name] += 1
                per_task_confusion[task_name][(gt_action, pred_action)] += 1
                if pred_action is not None and pred_action != TASK_BASELINE_ACTION[task_name]:
                    grounding[task_name]["den"] += 1
                    if _is_grounded(task_name, step.tool_calls):
                        grounding[task_name]["num"] += 1

    per_task_metrics = {}
    for task_name, counts in per_task_counts.items():
        label_space = TASK_LABEL_SPACES[task_name]
        class_metrics = {}
        for action in label_space:
            tp = per_task_confusion[task_name][(action, action)]
            fp = sum(per_task_confusion[task_name][(gt, action)] for gt in label_space if gt != action)
            fn = sum(per_task_confusion[task_name][(action, pred)] for pred in label_space if pred != action)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            class_metrics[action] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(_f1(tp, fp, fn), 4),
            }
        per_task_metrics[task_name] = {
            "accuracy": round(per_task_correct[task_name] / counts, 4) if counts else 0.0,
            "macro_f1": round(sum(metric["f1"] for metric in class_metrics.values()) / len(label_space), 4),
            "per_class": class_metrics,
            "grounded_rate": round(grounding[task_name]["num"] / grounding[task_name]["den"], 4)
            if grounding[task_name]["den"]
            else None,
        }

    return {
        "joint_step_accuracy": round(joint_correct / total_steps, 4) if total_steps else 0.0,
        "per_task": per_task_metrics,
    }

def _is_grounded(task_name: str, tool_calls: list[dict[str, Any]]) -> bool:
    tools = {call["tool_name"] for call in tool_calls}
    if task_name == "sepsis":
        return bool(tools & {"query_suspicion_of_infection", "query_sofa", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME})
    if task_name == "infection_only":
        return bool(tools & {"query_suspicion_of_infection", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME})
    if task_name == "aki":
        return bool(tools & {"query_kdigo_stage", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME})
    if task_name == "respiratory_support":
        return bool(tools & {"query_ventilation_status", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME})
    return False


def _single_task_grounding_spec(
    task_name: str,
    label_space: list[str] | None = None,
) -> dict[str, tuple[str, set[str]]]:
    if task_name == "sepsis":
        return {
            "infection_suspect": (
                "infection_predictions_grounded_rate",
                {"query_suspicion_of_infection", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME},
            ),
            "trigger_sepsis_alert": (
                "alert_predictions_grounded_rate",
                {"query_suspicion_of_infection", "query_sofa", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME},
            ),
        }
    if task_name == "infection_only":
        return {
            "infection_suspect": (
                "infection_predictions_grounded_rate",
                {"query_suspicion_of_infection", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME},
            ),
        }
    if task_name == "aki":
        if label_space and "aki_stage_1" in label_space:
            return {
                "aki_stage_1": ("stage1_predictions_grounded_rate", {"query_kdigo_stage", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME}),
                "aki_stage_2": ("stage2_predictions_grounded_rate", {"query_kdigo_stage", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME}),
                "aki_stage_3": ("stage3_predictions_grounded_rate", {"query_kdigo_stage", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME}),
            }
        return {
            "suspect_aki": ("suspect_predictions_grounded_rate", {"query_kdigo_stage", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME}),
            "trigger_aki_alert": ("alert_predictions_grounded_rate", {"query_kdigo_stage", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME}),
        }
    if task_name == "respiratory_support":
        return {
            "high_flow_or_noninvasive_support": (
                "medium_support_predictions_grounded_rate",
                {"query_ventilation_status", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME},
            ),
            "invasive_vent_required": (
                "invasive_support_predictions_grounded_rate",
                {"query_ventilation_status", CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME},
            ),
        }
    return {}


def _single_task_grounding_counters(
    task_name: str,
    label_space: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    return {
        metric_name: {"num": 0, "den": 0}
        for metric_name, _tools in _single_task_grounding_spec(task_name, label_space).values()
    }


def _empty_toolbox_state() -> dict[str, Any]:
    return {
        "infection_assessed": False,
        "infection_positive": False,
        "sofa_assessed": False,
        "sofa_alert": False,
        "max_sofa": None,
        "kdigo_assessed": False,
        "max_kdigo": None,
        "vent_assessed": False,
        "max_support_rank": None,
    }


def _update_single_task_grounding(
    task_name: str,
    predicted_action: str | None,
    tool_calls: list[dict[str, Any]],
    grounding_counts: dict[str, dict[str, int]],
    label_space: list[str] | None = None,
) -> None:
    if predicted_action is None:
        return
    tools = {call["tool_name"] for call in tool_calls}
    for action_name, (metric_name, required_tools) in _single_task_grounding_spec(task_name, label_space).items():
        if predicted_action != action_name:
            continue
        grounding_counts[metric_name]["den"] += 1
        if tools & required_tools:
            grounding_counts[metric_name]["num"] += 1


def _update_single_task_grounding_from_state(
    task_name: str,
    predicted_action: str | None,
    grounding_counts: dict[str, dict[str, int]],
    current_state: dict[str, Any],
    label_space: list[str] | None = None,
) -> None:
    if predicted_action is None:
        return
    spec = _single_task_grounding_spec(task_name, label_space)
    if predicted_action not in spec:
        return
    metric_name, _required_tools = spec[predicted_action]
    grounding_counts[metric_name]["den"] += 1

    grounded = False
    if task_name == "sepsis":
        if predicted_action == "infection_suspect":
            grounded = bool(current_state["infection_positive"])
        elif predicted_action == "trigger_sepsis_alert":
            grounded = bool(current_state["infection_positive"] and current_state["sofa_alert"])
    elif task_name == "infection_only":
        grounded = bool(current_state["infection_positive"])
    elif task_name == "aki":
        stage = current_state.get("max_kdigo")
        if predicted_action == "suspect_aki":
            grounded = stage is not None and stage >= 1
        elif predicted_action == "trigger_aki_alert":
            grounded = stage is not None and stage >= 2
    elif task_name == "respiratory_support":
        support_rank = current_state.get("max_support_rank")
        if predicted_action == "high_flow_or_noninvasive_support":
            grounded = support_rank is not None and support_rank >= 1
        elif predicted_action == "invasive_vent_required":
            grounded = support_rank is not None and support_rank >= 2

    if grounded:
        grounding_counts[metric_name]["num"] += 1


def _finalize_single_task_grounding(grounding_counts: dict[str, dict[str, int]]) -> dict[str, float | None]:
    return {
        metric_name: round(counts["num"] / counts["den"], 4) if counts["den"] else None
        for metric_name, counts in grounding_counts.items()
    }


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _empty_step_resource_usage() -> dict[str, Any]:
    return {
        "agent_calls": 0,
        "tool_calls": 0,
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "agent_runtime_sec": 0.0,
        "model_runtime_sec": 0.0,
        "tool_runtime_sec": 0.0,
    }


def _pop_agent_response_metrics(agent: Agent) -> dict[str, Any]:
    if hasattr(agent, "pop_last_response_metrics"):
        metrics = agent.pop_last_response_metrics()
        return metrics if isinstance(metrics, dict) else {}
    return {}


def _merge_agent_response_metrics(step_resource_usage: dict[str, Any], metrics: dict[str, Any]) -> None:
    step_resource_usage["model_calls"] += int(metrics.get("model_calls", 0) or 0)
    step_resource_usage["prompt_tokens"] += int(metrics.get("prompt_tokens", 0) or 0)
    step_resource_usage["completion_tokens"] += int(metrics.get("completion_tokens", 0) or 0)
    step_resource_usage["total_tokens"] += int(metrics.get("total_tokens", 0) or 0)
    step_resource_usage["model_runtime_sec"] += float(metrics.get("model_runtime_sec", 0.0) or 0.0)


def _finalize_step_resource_usage(
    step_resource_usage: dict[str, Any],
    *,
    step_runtime_sec: float,
) -> dict[str, Any]:
    finalized = dict(step_resource_usage)
    finalized["agent_runtime_sec"] = round(finalized["agent_runtime_sec"], 6)
    finalized["model_runtime_sec"] = round(finalized["model_runtime_sec"], 6)
    finalized["tool_runtime_sec"] = round(finalized["tool_runtime_sec"], 6)
    finalized["step_runtime_sec"] = round(step_runtime_sec, 6)
    return finalized


def _aggregate_rollout_resource_usage(steps: list[StepRecord]) -> dict[str, Any]:
    totals = _empty_step_resource_usage()
    total_step_runtime_sec = 0.0
    for step in steps:
        usage = step.resource_usage or {}
        totals["agent_calls"] += int(usage.get("agent_calls", 0) or 0)
        totals["tool_calls"] += int(usage.get("tool_calls", 0) or 0)
        totals["model_calls"] += int(usage.get("model_calls", 0) or 0)
        totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        totals["agent_runtime_sec"] += float(usage.get("agent_runtime_sec", 0.0) or 0.0)
        totals["model_runtime_sec"] += float(usage.get("model_runtime_sec", 0.0) or 0.0)
        totals["tool_runtime_sec"] += float(usage.get("tool_runtime_sec", 0.0) or 0.0)
        total_step_runtime_sec += float(usage.get("step_runtime_sec", 0.0) or 0.0)
    totals["agent_runtime_sec"] = round(totals["agent_runtime_sec"], 6)
    totals["model_runtime_sec"] = round(totals["model_runtime_sec"], 6)
    totals["tool_runtime_sec"] = round(totals["tool_runtime_sec"], 6)
    totals["total_step_runtime_sec"] = round(total_step_runtime_sec, 6)
    totals["num_steps"] = len(steps)
    return totals


def evaluate_resource_usage(rollouts: list[TrajectoryRollout]) -> dict[str, Any]:
    num_rollouts = len(rollouts)
    all_steps = [step for rollout in rollouts for step in rollout.steps]
    num_steps = len(all_steps)
    total_agent_calls = sum(int((step.resource_usage or {}).get("agent_calls", 0) or 0) for step in all_steps)
    total_tool_calls = sum(int((step.resource_usage or {}).get("tool_calls", 0) or 0) for step in all_steps)
    total_model_calls = sum(int((step.resource_usage or {}).get("model_calls", 0) or 0) for step in all_steps)
    total_prompt_tokens = sum(int((step.resource_usage or {}).get("prompt_tokens", 0) or 0) for step in all_steps)
    total_completion_tokens = sum(int((step.resource_usage or {}).get("completion_tokens", 0) or 0) for step in all_steps)
    total_tokens = sum(int((step.resource_usage or {}).get("total_tokens", 0) or 0) for step in all_steps)
    total_agent_runtime = sum(float((step.resource_usage or {}).get("agent_runtime_sec", 0.0) or 0.0) for step in all_steps)
    total_model_runtime = sum(float((step.resource_usage or {}).get("model_runtime_sec", 0.0) or 0.0) for step in all_steps)
    total_tool_runtime = sum(float((step.resource_usage or {}).get("tool_runtime_sec", 0.0) or 0.0) for step in all_steps)
    total_step_runtime = sum(float((step.resource_usage or {}).get("step_runtime_sec", 0.0) or 0.0) for step in all_steps)

    return {
        "totals": {
            "num_trajectories": num_rollouts,
            "num_steps": num_steps,
            "agent_calls": total_agent_calls,
            "tool_calls": total_tool_calls,
            "model_calls": total_model_calls,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "agent_runtime_sec": round(total_agent_runtime, 6),
            "model_runtime_sec": round(total_model_runtime, 6),
            "tool_runtime_sec": round(total_tool_runtime, 6),
            "step_runtime_sec": round(total_step_runtime, 6),
        },
        "step_level": {
            "avg_agent_calls_per_step": round(total_agent_calls / num_steps, 4) if num_steps else 0.0,
            "avg_tool_calls_per_step": round(total_tool_calls / num_steps, 4) if num_steps else 0.0,
            "avg_prompt_tokens_per_step": round(total_prompt_tokens / num_steps, 4) if num_steps else 0.0,
            "avg_completion_tokens_per_step": round(total_completion_tokens / num_steps, 4) if num_steps else 0.0,
            "avg_total_tokens_per_step": round(total_tokens / num_steps, 4) if num_steps else 0.0,
            "avg_agent_runtime_sec_per_step": round(total_agent_runtime / num_steps, 6) if num_steps else 0.0,
            "avg_model_runtime_sec_per_step": round(total_model_runtime / num_steps, 6) if num_steps else 0.0,
            "avg_tool_runtime_sec_per_step": round(total_tool_runtime / num_steps, 6) if num_steps else 0.0,
            "avg_step_runtime_sec": round(total_step_runtime / num_steps, 6) if num_steps else 0.0,
        },
        "trajectory_level": {
            "avg_steps_per_trajectory": round(num_steps / num_rollouts, 4) if num_rollouts else 0.0,
            "avg_total_tokens_per_trajectory": round(total_tokens / num_rollouts, 4) if num_rollouts else 0.0,
            "avg_total_runtime_sec_per_trajectory": round(total_step_runtime / num_rollouts, 6) if num_rollouts else 0.0,
        },
    }


def _support_level_rank(level: str | None) -> int | None:
    if level == "room_air_or_low_support":
        return 0
    if level == "high_flow_or_noninvasive_support":
        return 1
    if level == "invasive_vent_required":
        return 2
    return None


def _update_toolbox_state_from_output(state: dict[str, Any], tool_name: str, output: dict[str, Any]) -> None:
    if tool_name == "query_suspicion_of_infection":
        state["infection_assessed"] = True
        if output.get("has_suspected_infection"):
            state["infection_positive"] = True
    elif tool_name == "query_sofa":
        state["sofa_assessed"] = True
        max_sofa = output.get("max_sofa_24hours_so_far")
        latest_sofa = output.get("latest_sofa_24hours")
        sofa_value = max_sofa if max_sofa is not None else latest_sofa
        if sofa_value is not None:
            previous = state.get("max_sofa")
            state["max_sofa"] = sofa_value if previous is None else max(previous, sofa_value)
        if (state.get("max_sofa") or 0) >= 2:
            state["sofa_alert"] = True
    elif tool_name == "query_kdigo_stage":
        state["kdigo_assessed"] = True
        stage = output.get("max_aki_stage_smoothed_so_far")
        if stage is None:
            stage = output.get("latest_aki_stage_smoothed")
        if stage is not None:
            previous = state.get("max_kdigo")
            state["max_kdigo"] = stage if previous is None else max(previous, stage)
    elif tool_name == "query_ventilation_status":
        state["vent_assessed"] = True
        level = output.get("highest_support_level_so_far") or output.get("current_support_level")
        rank = _support_level_rank(level)
        if rank is not None:
            previous = state.get("max_support_rank")
            state["max_support_rank"] = rank if previous is None else max(previous, rank)


def _tool_call_has_marginal_utility(
    tool_name: str,
    output: dict[str, Any],
    prior_state: dict[str, Any],
) -> bool:
    if tool_name in {CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME}:
        return bool(
            output.get("ok")
            and (
                output.get("result") is not None
                or bool(output.get("stdout"))
            )
        )
    if tool_name == "query_suspicion_of_infection":
        return (not prior_state["infection_assessed"]) or (
            not prior_state["infection_positive"] and bool(output.get("has_suspected_infection"))
        )
    if tool_name == "query_sofa":
        latest_sofa = output.get("latest_sofa_24hours")
        max_sofa = output.get("max_sofa_24hours_so_far")
        sofa_value = max_sofa if max_sofa is not None else latest_sofa
        return (not prior_state["sofa_assessed"]) or (
            sofa_value is not None and (prior_state["max_sofa"] is None or sofa_value > prior_state["max_sofa"])
        )
    if tool_name == "query_kdigo_stage":
        stage = output.get("max_aki_stage_smoothed_so_far")
        if stage is None:
            stage = output.get("latest_aki_stage_smoothed")
        return (not prior_state["kdigo_assessed"]) or (
            stage is not None and (prior_state["max_kdigo"] is None or stage > prior_state["max_kdigo"])
        )
    if tool_name == "query_ventilation_status":
        level = output.get("highest_support_level_so_far") or output.get("current_support_level")
        rank = _support_level_rank(level)
        return (not prior_state["vent_assessed"]) or (
            rank is not None
            and (prior_state["max_support_rank"] is None or rank > prior_state["max_support_rank"])
        )
    if tool_name == "query_urine_output_rate":
        return bool(
            output.get("min_6hr_rate_mL_kg_hr") is not None
            or output.get("has_oliguria")
            or output.get("has_severe_oliguria")
        )
    if tool_name == "query_vasoactive_agent":
        return bool(output.get("received_vasoactive") or output.get("active_agents"))
    if tool_name == "query_vitalsign":
        return bool(
            output.get("latest_charttime") is not None
            or output.get("has_tachycardia")
            or output.get("has_hypotension")
        )
    if tool_name == "query_bg":
        return bool(
            output.get("peak_lactate") is not None
            or output.get("min_pH") is not None
            or output.get("worst_pao2fio2ratio") is not None
        )
    if tool_name == "query_gcs":
        return bool(output.get("min_gcs") is not None or output.get("has_severe_impairment"))
    if tool_name == "query_antibiotic":
        return bool(output.get("received_antibiotics") or output.get("distinct_antibiotic_count"))
    if tool_name == "query_invasive_line":
        return bool(output.get("has_invasive_line") or output.get("lines_present"))
    return False


def evaluate_generic_toolbox_efficiency(rollouts: list[TrajectoryRollout]) -> dict[str, Any]:
    total_steps = 0
    total_tool_calls = 0
    execution_calls_total = 0
    execution_calls_success = 0
    execution_calls_with_result = 0
    steps_without_calls = 0
    repeated_tool_calls = 0
    marginal_utility_total = 0
    marginal_utility_positive = 0
    call_counts_by_tool = Counter()
    useful_call_counts_by_tool = Counter()

    for rollout in rollouts:
        prior_state = _empty_toolbox_state()
        called_tools_seen: set[str] = set()
        for step in rollout.steps:
            total_steps += 1
            tool_calls = step.tool_calls or []
            tool_outputs = step.tool_outputs or []
            if not tool_calls:
                steps_without_calls += 1
            for call_payload, output in zip(tool_calls, tool_outputs):
                tool_name = call_payload["tool_name"]
                pre_call_state = dict(prior_state)
                total_tool_calls += 1
                call_counts_by_tool[tool_name] += 1
                if tool_name in {CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME}:
                    execution_calls_total += 1
                    if output.get("ok"):
                        execution_calls_success += 1
                    if output.get("result") is not None or output.get("stdout"):
                        execution_calls_with_result += 1
                if tool_name in called_tools_seen:
                    repeated_tool_calls += 1
                is_useful = _tool_call_has_marginal_utility(tool_name, output, pre_call_state)
                marginal_utility_total += 1
                if is_useful:
                    marginal_utility_positive += 1
                    useful_call_counts_by_tool[tool_name] += 1
                _update_toolbox_state_from_output(prior_state, tool_name, output)
                called_tools_seen.add(tool_name)

    return {
        "avg_tool_calls_per_step": round(total_tool_calls / total_steps, 4) if total_steps else 0.0,
        "steps_without_tool_calls_rate": _rate(steps_without_calls, total_steps),
        "repeated_tool_call_rate": _rate(repeated_tool_calls, total_tool_calls),
        "execution_success_rate": _rate(execution_calls_success, execution_calls_total),
        "execution_informative_rate": _rate(execution_calls_with_result, execution_calls_total),
        "marginal_utility_of_call_rate": _rate(marginal_utility_positive, marginal_utility_total),
        "tool_call_counts": dict(call_counts_by_tool),
        "tool_marginal_utility_rate": {
            tool_name: _rate(useful_call_counts_by_tool[tool_name], call_counts_by_tool[tool_name])
            for tool_name in sorted(call_counts_by_tool)
        },
    }


def evaluate_sepsis_toolbox_efficiency(rollouts: list[TrajectoryRollout]) -> dict[str, Any]:
    total_steps = 0
    total_tool_calls = 0
    execution_calls_total = 0
    execution_calls_success = 0
    execution_calls_with_result = 0
    steps_without_calls = 0
    repeated_tool_calls = 0
    infection_calls_total = 0
    infection_calls_after_positive = 0
    positive_actions_total = 0
    positive_actions_without_sufficient_evidence = 0
    necessary_infection_total = 0
    necessary_infection_covered = 0
    necessary_sofa_total = 0
    necessary_sofa_covered = 0
    marginal_utility_total = 0
    marginal_utility_positive = 0
    call_counts_by_tool = Counter()
    useful_call_counts_by_tool = Counter()

    for rollout in rollouts:
        prior_state = _empty_toolbox_state()
        called_tools_seen: set[str] = set()

        for step in rollout.steps:
            total_steps += 1
            tool_calls = step.tool_calls or []
            tool_outputs = step.tool_outputs or []
            step_state_before = dict(prior_state)
            if not tool_calls:
                steps_without_calls += 1

            current_step_outputs: dict[str, dict[str, Any]] = {}
            for call_payload, output in zip(tool_calls, tool_outputs):
                tool_name = call_payload["tool_name"]
                pre_call_state = dict(prior_state)
                current_step_outputs[tool_name] = output
                total_tool_calls += 1
                call_counts_by_tool[tool_name] += 1
                if tool_name in {CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME}:
                    execution_calls_total += 1
                    if output.get("ok"):
                        execution_calls_success += 1
                    if output.get("result") is not None or output.get("stdout"):
                        execution_calls_with_result += 1
                if tool_name in called_tools_seen:
                    repeated_tool_calls += 1
                if tool_name == "query_suspicion_of_infection":
                    infection_calls_total += 1
                    if pre_call_state["infection_positive"]:
                        infection_calls_after_positive += 1
                is_useful = _tool_call_has_marginal_utility(tool_name, output, pre_call_state)
                marginal_utility_total += 1
                if is_useful:
                    marginal_utility_positive += 1
                    useful_call_counts_by_tool[tool_name] += 1
                _update_toolbox_state_from_output(prior_state, tool_name, output)
                called_tools_seen.add(tool_name)

            predicted_action = step.predicted_action
            if predicted_action in {"infection_suspect", "trigger_sepsis_alert"}:
                positive_actions_total += 1
                if predicted_action == "infection_suspect":
                    sufficient = prior_state["infection_positive"]
                else:
                    sufficient = prior_state["infection_positive"] and prior_state["sofa_alert"]
                if not sufficient:
                    positive_actions_without_sufficient_evidence += 1

                if (
                    not step_state_before["infection_positive"]
                    and "query_suspicion_of_infection" not in current_step_outputs
                ):
                    necessary_infection_total += 1
                elif not step_state_before["infection_positive"]:
                    necessary_infection_total += 1
                    necessary_infection_covered += 1

                if predicted_action == "trigger_sepsis_alert":
                    if not step_state_before["sofa_alert"] and "query_sofa" not in current_step_outputs:
                        necessary_sofa_total += 1
                    elif not step_state_before["sofa_alert"]:
                        necessary_sofa_total += 1
                        necessary_sofa_covered += 1

    return {
        "avg_tool_calls_per_step": round(total_tool_calls / total_steps, 4) if total_steps else 0.0,
        "steps_without_tool_calls_rate": _rate(steps_without_calls, total_steps),
        "repeated_tool_call_rate": _rate(repeated_tool_calls, total_tool_calls),
        "execution_success_rate": _rate(execution_calls_success, execution_calls_total),
        "execution_informative_rate": _rate(execution_calls_with_result, execution_calls_total),
        "repeated_infection_call_after_positive_rate": _rate(
            infection_calls_after_positive,
            infection_calls_total,
        ),
        "positive_action_without_sufficient_evidence_rate": _rate(
            positive_actions_without_sufficient_evidence,
            positive_actions_total,
        ),
        "necessary_call_coverage": {
            "infection": _rate(necessary_infection_covered, necessary_infection_total),
            "sofa_for_alert": _rate(necessary_sofa_covered, necessary_sofa_total),
        },
        "marginal_utility_of_call_rate": _rate(marginal_utility_positive, marginal_utility_total),
        "tool_call_counts": dict(call_counts_by_tool),
        "tool_marginal_utility_rate": {
            tool_name: _rate(useful_call_counts_by_tool[tool_name], call_counts_by_tool[tool_name])
            for tool_name in sorted(call_counts_by_tool)
        },
    }


def _format_single_task_transition_timing(
    task_name: str,
    transition_timing: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if task_name == "sepsis":
        return {
            "infection": transition_timing.get("infection_suspect"),
            "sepsis_alert": transition_timing.get("trigger_sepsis_alert"),
        }
    if task_name == "infection_only":
        return {
            "infection": transition_timing.get("infection_suspect"),
        }
    if task_name == "aki":
        return {
            "aki_suspect": transition_timing.get("suspect_aki"),
            "aki_alert": transition_timing.get("trigger_aki_alert"),
        }
    if task_name == "respiratory_support":
        return {
            "medium_support": transition_timing.get("high_flow_or_noninvasive_support"),
            "invasive_support": transition_timing.get("invasive_vent_required"),
        }
    return transition_timing


def _is_non_monotonic_aki_task(trajectory: Trajectory, label_space: list[str]) -> bool:
    return (
        trajectory.primary_task_name() == "aki"
        and (
            trajectory.task_variant == "non_monotonic_current_state"
            or "aki_stage_1" in label_space
        )
    )


def _state_order(label_space: list[str], label: str) -> int:
    try:
        return label_space.index(label)
    except ValueError:
        return -1


def _update_change_counts(counter: dict[str, int], gt_event: bool, pred_event: bool) -> None:
    if gt_event and pred_event:
        counter["tp"] += 1
    elif not gt_event and pred_event:
        counter["fp"] += 1
    elif gt_event and not pred_event:
        counter["fn"] += 1


def _event_metrics(counter: dict[str, int]) -> dict[str, float]:
    tp = counter["tp"]
    fp = counter["fp"]
    fn = counter["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_f1(tp, fp, fn), 4),
    }


def _timing_metrics(gt_hours: list[int | None], pred_hours: list[int | None]) -> dict[str, Any]:
    exact_matches = 0
    abs_errors = []
    early = 0
    late = 0
    missed = 0
    for gt, pred in zip(gt_hours, pred_hours):
        if gt == pred:
            exact_matches += 1
        if gt is None and pred is None:
            abs_errors.append(0)
            continue
        if gt is None and pred is not None:
            early += 1
            abs_errors.append(pred)
            continue
        if gt is not None and pred is None:
            missed += 1
            continue
        assert gt is not None and pred is not None
        abs_errors.append(abs(pred - gt))
        if pred < gt:
            early += 1
        elif pred > gt:
            late += 1
    return {
        "exact_match_rate": round(exact_matches / len(gt_hours), 4) if gt_hours else 0.0,
        "mean_absolute_error_hours": round(sum(abs_errors) / len(abs_errors), 4) if abs_errors else None,
        "early_rate": round(early / len(gt_hours), 4) if gt_hours else 0.0,
        "late_rate": round(late / len(gt_hours), 4) if gt_hours else 0.0,
        "missed_rate": round(missed / len(gt_hours), 4) if gt_hours else 0.0,
    }


def rollout_to_dicts(rollouts: list[TrajectoryRollout]) -> list[dict[str, Any]]:
    return [asdict(rollout) for rollout in rollouts]


def _latest_step_tool_output(
    tool_outputs: list[dict[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    return next((item for item in reversed(tool_outputs) if key in item), None)


def _compact_infection_history(output: dict[str, Any] | None) -> dict[str, Any] | None:
    if output is None:
        return None
    evidence_preview = []
    for item in (output.get("evidence") or [])[:2]:
        evidence_preview.append(
            {
                "antibiotic": item.get("antibiotic"),
                "antibiotic_time": item.get("antibiotic_time"),
                "culture_time": item.get("culture_time"),
                "specimen": item.get("specimen"),
            }
        )
    return {
        "infection": output.get("has_suspected_infection"),
        "infection_first_visible_hour": output.get("first_visible_suspected_infection_hour"),
        "infection_first_visible_time": output.get("first_visible_suspected_infection_time"),
        "evidence": evidence_preview,
    }


def _compact_sofa_history(output: dict[str, Any] | None) -> dict[str, Any] | None:
    if output is None:
        return None
    return {
        "sofa_score": output.get("latest_sofa_24hours"),
        "sofa_hr": output.get("latest_visible_hr"),
        "max_sofa_score_so_far": output.get("max_sofa_24hours_so_far"),
    }


def _compact_aki_history(output: dict[str, Any] | None) -> dict[str, Any] | None:
    if output is None:
        return None
    return {
        "current_aki_state_label": output.get("current_aki_state_label"),
        "current_aki_state_stage": output.get("current_aki_state_stage"),
        "max_aki_stage_smoothed_so_far": output.get("max_aki_stage_smoothed_so_far"),
        "has_stage2_or_higher": output.get("has_stage2_or_higher"),
        "has_stage3_or_crrt": output.get("has_stage3_or_crrt"),
    }


def _compact_vent_history(output: dict[str, Any] | None) -> dict[str, Any] | None:
    if output is None:
        return None
    return {
        "current_support_level": output.get("current_support_level"),
        "highest_support_level_so_far": output.get("highest_support_level_so_far"),
        "has_medium_support": output.get("has_medium_support"),
        "has_invasive_support": output.get("has_invasive_support"),
    }


def _compact_contextual_tool_history(tool_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    contextual: dict[str, Any] = {}
    urine = _latest_step_tool_output(tool_outputs, "min_6hr_rate_mL_kg_hr")
    if urine is not None:
        contextual["urine_output_rate"] = {
            "min_6hr_rate_mL_kg_hr": urine.get("min_6hr_rate_mL_kg_hr"),
            "has_oliguria": urine.get("has_oliguria"),
            "has_severe_oliguria": urine.get("has_severe_oliguria"),
        }
    vaso = _latest_step_tool_output(tool_outputs, "received_vasoactive")
    if vaso is not None:
        contextual["vasoactive_agent"] = {
            "received_vasoactive": vaso.get("received_vasoactive"),
            "active_agents": vaso.get("active_agents"),
        }
    vitals = _latest_step_tool_output(tool_outputs, "has_tachycardia")
    if vitals is not None:
        contextual["vitalsign"] = {
            "latest_heart_rate": vitals.get("latest_heart_rate"),
            "latest_mbp": vitals.get("latest_mbp"),
            "has_tachycardia": vitals.get("has_tachycardia"),
            "has_hypotension": vitals.get("has_hypotension"),
        }
    bg = _latest_step_tool_output(tool_outputs, "peak_lactate")
    if bg is not None:
        contextual["bg"] = {
            "peak_lactate": bg.get("peak_lactate"),
            "min_pH": bg.get("min_pH"),
            "has_elevated_lactate": bg.get("has_elevated_lactate"),
            "has_acidosis": bg.get("has_acidosis"),
        }
    gcs = _latest_step_tool_output(tool_outputs, "min_gcs")
    if gcs is not None:
        contextual["gcs"] = {
            "min_gcs": gcs.get("min_gcs"),
            "has_severe_impairment": gcs.get("has_severe_impairment"),
        }
    antibiotic = _latest_step_tool_output(tool_outputs, "received_antibiotics")
    if antibiotic is not None:
        contextual["antibiotic"] = {
            "received_antibiotics": antibiotic.get("received_antibiotics"),
            "distinct_antibiotic_count": antibiotic.get("distinct_antibiotic_count"),
        }
    invasive_line = _latest_step_tool_output(tool_outputs, "has_invasive_line")
    if invasive_line is not None:
        contextual["invasive_line"] = {
            "has_invasive_line": invasive_line.get("has_invasive_line"),
            "lines_present": invasive_line.get("lines_present"),
        }
    return contextual


def _build_rolling_history_entry(
    *,
    trajectory: Trajectory,
    checkpoint: Any,
    step_index: int,
    tool_outputs: list[dict[str, Any]],
    predicted_surveillance: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    task_name = trajectory.primary_task_name()
    if task_name == "general_icu_surveillance":
        summary = (predicted_surveillance or {}).get("checkpoint_summary")
        if not summary:
            suspected = ", ".join((predicted_surveillance or {}).get("suspected_conditions") or []) or "none"
            alerts = ", ".join((predicted_surveillance or {}).get("alerts") or []) or "none"
            global_action = (predicted_surveillance or {}).get("global_action") or checkpoint.state_label or "continue_monitoring"
            priority = (predicted_surveillance or {}).get("priority") or "low"
            summary = f"action={global_action}; suspects={suspected}; alerts={alerts}; priority={priority}"
        return {
            "task_name": task_name,
            "step_index": step_index,
            "t_hour": checkpoint.t_hour,
            "summary": summary,
        }
    infection = _compact_infection_history(
        _latest_step_tool_output(tool_outputs, "has_suspected_infection")
    ) or {
        "infection": None,
        "infection_first_visible_hour": None,
        "infection_first_visible_time": None,
        "evidence": [],
    }
    sofa = _compact_sofa_history(
        _latest_step_tool_output(tool_outputs, "latest_sofa_24hours")
    ) or {
        "sofa_score": None,
        "sofa_hr": None,
        "max_sofa_score_so_far": None,
    }
    aki = _compact_aki_history(
        _latest_step_tool_output(tool_outputs, "current_aki_state_label")
    ) or {
        "current_aki_state_label": None,
        "current_aki_state_stage": None,
        "max_aki_stage_smoothed_so_far": None,
        "has_stage2_or_higher": None,
        "has_stage3_or_crrt": None,
    }
    vent = _compact_vent_history(
        _latest_step_tool_output(tool_outputs, "current_support_level")
    ) or {
        "current_support_level": None,
        "highest_support_level_so_far": None,
        "has_medium_support": None,
        "has_invasive_support": None,
    }
    contextual = _compact_contextual_tool_history(tool_outputs)
    if trajectory.is_multitask():
        return {
            "task_name": "multitask",
            "step_index": step_index,
            "t_hour": checkpoint.t_hour,
            "infection": infection["infection"],
            "sofa_score": sofa["sofa_score"],
            "current_aki_state_label": aki["current_aki_state_label"],
            "max_aki_stage_smoothed_so_far": aki["max_aki_stage_smoothed_so_far"],
            "current_support_level": vent["current_support_level"],
            "highest_support_level_so_far": vent["highest_support_level_so_far"],
            "contextual_tools": contextual,
        }
    if task_name == "sepsis":
        return {
            "task_name": task_name,
            "step_index": step_index,
            "t_hour": checkpoint.t_hour,
            "sofa_score": sofa["sofa_score"],
            "sofa_hr": sofa["sofa_hr"],
            "max_sofa_score_so_far": sofa["max_sofa_score_so_far"],
            "infection": infection["infection"],
            "infection_first_visible_hour": infection["infection_first_visible_hour"],
            "infection_first_visible_time": infection["infection_first_visible_time"],
            "evidence": infection["evidence"],
            "contextual_tools": contextual,
        }
    if task_name == "infection_only":
        return {
            "task_name": task_name,
            "step_index": step_index,
            "t_hour": checkpoint.t_hour,
            "infection": infection["infection"],
            "infection_first_visible_hour": infection["infection_first_visible_hour"],
            "infection_first_visible_time": infection["infection_first_visible_time"],
            "evidence": infection["evidence"],
            "contextual_tools": contextual,
        }
    if task_name == "aki":
        return {
            "task_name": task_name,
            "step_index": step_index,
            "t_hour": checkpoint.t_hour,
            "current_aki_state_label": aki["current_aki_state_label"],
            "current_aki_state_stage": aki["current_aki_state_stage"],
            "max_aki_stage_smoothed_so_far": aki["max_aki_stage_smoothed_so_far"],
            "has_stage2_or_higher": aki["has_stage2_or_higher"],
            "has_stage3_or_crrt": aki["has_stage3_or_crrt"],
            "contextual_tools": contextual,
        }
    if task_name == "respiratory_support":
        return {
            "task_name": task_name,
            "step_index": step_index,
            "t_hour": checkpoint.t_hour,
            "current_support_level": vent["current_support_level"],
            "highest_support_level_so_far": vent["highest_support_level_so_far"],
            "has_medium_support": vent["has_medium_support"],
            "has_invasive_support": vent["has_invasive_support"],
            "contextual_tools": contextual,
        }
    return None
