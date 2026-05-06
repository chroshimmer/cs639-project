from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .agent import LocalQwenChat, _coerce_agent_output, _extract_json_object
from .environment import evaluate_rollouts
from .schemas import ActionDecision, StepRecord, ToolCall, Trajectory, TrajectoryRollout
from .tools import ToolRuntime


PROMPT_BASELINE_PROTOCOL = "prompt_card"


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _summarize_infection(payload: dict[str, Any]) -> dict[str, Any]:
    evidence = payload.get("evidence") or []
    first = evidence[0] if evidence else {}
    return {
        "suspected_infection": "visible" if payload.get("has_suspected_infection") else "not_visible",
        "first_visible_suspected_infection_hour": payload.get("first_visible_suspected_infection_hour"),
        "first_visible_suspected_infection_time": payload.get("first_visible_suspected_infection_time"),
        "evidence_count": len(evidence),
        "first_evidence": {
            "antibiotic": first.get("antibiotic"),
            "antibiotic_time": first.get("antibiotic_time"),
            "culture_time": first.get("culture_time"),
            "specimen": first.get("specimen"),
            "positive_culture": first.get("positive_culture"),
        }
        if first
        else None,
    }


def _summarize_sofa(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "latest_visible_hr": payload.get("latest_visible_hr"),
        "latest_sofa_24hours": payload.get("latest_sofa_24hours"),
        "max_sofa_24hours_so_far": payload.get("max_sofa_24hours_so_far"),
        "alert_level_sofa_visible": bool((payload.get("max_sofa_24hours_so_far") or 0) >= 2),
        "latest_components": payload.get("latest_components") or {},
    }


def build_patient_checkpoint_card(
    *,
    trajectory: Trajectory,
    step_index: int,
    infection_output: dict[str, Any],
    sofa_output: dict[str, Any],
    rolling_history: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = trajectory.checkpoints[step_index]
    return {
        "protocol": PROMPT_BASELINE_PROTOCOL,
        "task_name": trajectory.primary_task_name(),
        "trajectory_id": trajectory.trajectory_id,
        "stay_id": trajectory.stay_id,
        "step_index": step_index,
        "t_hour": checkpoint.t_hour,
        "label_space": ["keep_monitoring", "infection_suspect", "trigger_sepsis_alert"],
        "rolling_history": rolling_history,
        "current_checkpoint_summary": {
            "infection": _summarize_infection(infection_output),
            "sofa": _summarize_sofa(sofa_output),
        },
    }


def _history_entry_from_card(card: dict[str, Any], action: str | None) -> dict[str, Any]:
    infection = card["current_checkpoint_summary"]["infection"]
    sofa = card["current_checkpoint_summary"]["sofa"]
    return {
        "task_name": "sepsis",
        "step_index": card["step_index"],
        "t_hour": card["t_hour"],
        "predicted_action": action,
        "infection": infection["suspected_infection"] == "visible",
        "infection_first_visible_hour": infection.get("first_visible_suspected_infection_hour"),
        "sofa_score": sofa.get("latest_sofa_24hours"),
        "max_sofa_score_so_far": sofa.get("max_sofa_24hours_so_far"),
        "sofa_alert": sofa.get("alert_level_sofa_visible"),
    }


def _build_prompt_messages(card: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = (
        "You are an ICU rolling sepsis diagnosis agent.\n"
        "You receive a compact evidence card instead of function tools.\n"
        "The key limitation of this baseline is that you must infer the intermediate state from the card: "
        "suspected infection can be present before full sepsis alert criteria are met.\n"
        "Use this staged policy: if infection is not visible, return keep_monitoring; "
        "if infection is visible but SOFA alert evidence is not visible, return infection_suspect; "
        "if infection is visible and max SOFA is 2 or higher, return trigger_sepsis_alert.\n"
        "Do not output reasoning, analysis, markdown, or extra text.\n"
        "Return exactly one JSON object with one key: action.\n"
        "Valid actions: keep_monitoring, infection_suspect, trigger_sepsis_alert."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({"patient_checkpoint_card": card}, indent=2, default=_json_default)},
    ]


@dataclass(slots=True)
class PromptCardQwenAgent:
    model: str = "Qwen/Qwen3.5-9B"
    temperature: float = 0.0
    top_p: float = 0.95
    max_new_tokens: int = 160
    trace_callback: Callable[[dict[str, Any]], None] | None = field(default=None, repr=False)
    client: LocalQwenChat = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.client = LocalQwenChat(
            model_ref=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )

    def decide(self, card: dict[str, Any]) -> tuple[ActionDecision, dict[str, Any]]:
        messages = _build_prompt_messages(card)
        text, stats = self.client.generate_with_stats(messages)
        if self.trace_callback is not None:
            self.trace_callback(
                {
                    "event_type": "prompt_baseline_model_output_raw",
                    "trajectory_id": card["trajectory_id"],
                    "stay_id": card["stay_id"],
                    "step_index": card["step_index"],
                    "t_hour": card["t_hour"],
                    "output": text,
                }
            )
        try:
            response = _coerce_agent_output(_extract_json_object(text))
            if isinstance(response, ToolCall):
                raise ValueError("Prompt-card baseline does not allow tool calls.")
            decision = response
        except Exception:
            decision = ActionDecision(action="keep_monitoring")
            stats["fallback_action_used"] = True
        else:
            stats["fallback_action_used"] = False
        return decision, stats


@dataclass(slots=True)
class PromptCardRuleAgent:
    """Deterministic card reader used for smoke tests and reward debugging."""

    def decide(self, card: dict[str, Any]) -> tuple[ActionDecision, dict[str, Any]]:
        infection = card["current_checkpoint_summary"]["infection"]["suspected_infection"] == "visible"
        sofa_alert = bool(card["current_checkpoint_summary"]["sofa"]["alert_level_sofa_visible"])
        if infection and sofa_alert:
            action = "trigger_sepsis_alert"
        elif infection:
            action = "infection_suspect"
        else:
            action = "keep_monitoring"
        return ActionDecision(action=action), {"fallback_action_used": False}


def run_prompt_card_baseline(
    *,
    trajectories: list[Trajectory],
    tool_runtime: ToolRuntime,
    agent: PromptCardQwenAgent | PromptCardRuleAgent,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[TrajectoryRollout], dict[str, Any]]:
    rollouts: list[TrajectoryRollout] = []
    for trajectory in trajectories:
        if trajectory.primary_task_name() != "sepsis" or trajectory.is_multitask():
            raise ValueError("Prompt-card baseline currently supports only single-task sepsis trajectories.")
        steps: list[StepRecord] = []
        rolling_history: list[dict[str, Any]] = []
        first_infection_hour = None
        first_alert_hour = None
        first_task_hours: dict[str, dict[str, int | None]] = {"sepsis": {}}
        for step_index, checkpoint in enumerate(trajectory.checkpoints):
            started = time.perf_counter()
            infection_output = tool_runtime.execute(
                "query_suspicion_of_infection",
                {"stay_id": trajectory.stay_id, "t_hour": checkpoint.t_hour},
            )
            sofa_output = tool_runtime.execute(
                "query_sofa",
                {"stay_id": trajectory.stay_id, "t_hour": checkpoint.t_hour},
            )
            card = build_patient_checkpoint_card(
                trajectory=trajectory,
                step_index=step_index,
                infection_output=infection_output,
                sofa_output=sofa_output,
                rolling_history=list(rolling_history),
            )
            decision_started = time.perf_counter()
            decision, stats = agent.decide(card)
            predicted_action = decision.action or "keep_monitoring"
            if predicted_action not in {"keep_monitoring", "infection_suspect", "trigger_sepsis_alert"}:
                predicted_action = "keep_monitoring"
                stats["fallback_action_used"] = True

            if predicted_action == "infection_suspect" and first_infection_hour is None:
                first_infection_hour = checkpoint.t_hour
            if predicted_action == "trigger_sepsis_alert":
                if first_infection_hour is None:
                    first_infection_hour = checkpoint.t_hour
                if first_alert_hour is None:
                    first_alert_hour = checkpoint.t_hour
            if predicted_action != "keep_monitoring":
                first_task_hours["sepsis"].setdefault(predicted_action, checkpoint.t_hour)

            resource_usage = {
                "agent_calls": 1,
                "tool_calls": 0,
                "model_calls": 1 if stats.get("total_tokens") is not None else 0,
                "prompt_tokens": int(stats.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(stats.get("completion_tokens", 0) or 0),
                "total_tokens": int(stats.get("total_tokens", 0) or 0),
                "agent_runtime_sec": round(time.perf_counter() - decision_started, 6),
                "model_runtime_sec": round(float(stats.get("generation_runtime_sec", 0.0) or 0.0), 6),
                "tool_runtime_sec": 0.0,
                "step_runtime_sec": round(time.perf_counter() - started, 6),
                "hidden_card_tool_calls": 2,
                "fallback_action_used": bool(stats.get("fallback_action_used")),
            }
            steps.append(
                StepRecord(
                    step_index=step_index,
                    t_hour=checkpoint.t_hour,
                    gt_action=checkpoint.state_label,
                    predicted_action=predicted_action,
                    tool_calls=[],
                    tool_outputs=[],
                    resource_usage=resource_usage,
                )
            )
            if event_callback is not None:
                event_callback(
                    {
                        "event_type": "prompt_baseline_action",
                        "trajectory_id": trajectory.trajectory_id,
                        "stay_id": trajectory.stay_id,
                        "step_index": step_index,
                        "t_hour": checkpoint.t_hour,
                        "gt_action": checkpoint.state_label,
                        "predicted_action": predicted_action,
                        "patient_checkpoint_card": card,
                    }
                )
            rolling_history.append(_history_entry_from_card(card, predicted_action))
        rollouts.append(
            TrajectoryRollout(
                trajectory_id=trajectory.trajectory_id,
                stay_id=trajectory.stay_id,
                steps=steps,
                first_predicted_infection_hour=first_infection_hour,
                first_predicted_alert_hour=first_alert_hour,
                first_predicted_task_hours=first_task_hours,
            )
        )
    return rollouts, evaluate_rollouts(trajectories, rollouts, protocol="rolling_no_history")
