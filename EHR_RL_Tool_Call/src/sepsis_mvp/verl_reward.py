from __future__ import annotations

import json
from typing import Any

try:
    from .schemas import SEPSIS_ACTIONS
except ImportError:
    from sepsis_mvp.schemas import SEPSIS_ACTIONS


VALID_TOOLS = {"query_suspicion_of_infection", "query_sofa"}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def score_completion(completion: str, ground_truth: str | None, extra_info: dict[str, Any] | None = None) -> float:
    payload = _extract_json_object(completion)
    if payload is None:
        return -1.0

    reward = 0.1
    extra_info = extra_info or {}
    available_tools = set(extra_info.get("available_tools") or VALID_TOOLS)
    rolling_history = extra_info.get("rolling_history") or []
    if not rolling_history and extra_info.get("rolling_history_json"):
        try:
            rolling_history = json.loads(extra_info["rolling_history_json"])
        except (TypeError, json.JSONDecodeError):
            rolling_history = []

    if "tool_name" in payload:
        tool_name = payload.get("tool_name")
        args = payload.get("arguments") or {}
        if tool_name not in available_tools:
            return -1.0
        reward += 0.2
        if args.get("stay_id") == extra_info.get("stay_id"):
            reward += 0.1
        else:
            reward -= 0.4
        if args.get("t_hour") == extra_info.get("t_hour"):
            reward += 0.1
        else:
            reward -= 0.4
        if tool_name == "query_suspicion_of_infection":
            reward += 0.1
        if tool_name == "query_sofa":
            infection_known = any(item.get("infection") is True for item in rolling_history)
            reward += 0.25 if infection_known else -0.05
        return max(-1.0, min(1.0, reward))

    action = payload.get("action")
    if action not in SEPSIS_ACTIONS:
        return -1.0
    if action == ground_truth:
        reward += 1.0
    elif {action, ground_truth} == {"infection_suspect", "trigger_sepsis_alert"}:
        reward += 0.2
    elif ground_truth == "trigger_sepsis_alert" and action != "trigger_sepsis_alert":
        reward -= 0.7
    elif action == "trigger_sepsis_alert" and ground_truth == "keep_monitoring":
        reward -= 0.5
    else:
        reward -= 0.2

    infection_known = any(item.get("infection") is True for item in rolling_history)
    sofa_alert_known = any((item.get("max_sofa_score_so_far") or item.get("sofa_score") or 0) >= 2 for item in rolling_history)
    if action == "infection_suspect":
        reward += 0.2 if infection_known else -0.4
    if action == "trigger_sepsis_alert":
        reward += 0.3 if infection_known and sofa_alert_known else -0.6
    return max(-1.0, min(1.5, reward))


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: str | dict[str, Any] | None = None,
    **_: Any,
) -> float:
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            extra_info = {}
    return score_completion(solution_str or "", ground_truth, extra_info if isinstance(extra_info, dict) else {})
