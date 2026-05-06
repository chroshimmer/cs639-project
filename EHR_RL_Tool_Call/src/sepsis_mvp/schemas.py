from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


SEPSIS_ACTIONS = (
    "keep_monitoring",
    "infection_suspect",
    "trigger_sepsis_alert",
)

INFECTION_ONLY_ACTIONS = (
    "keep_monitoring",
    "infection_suspect",
)

AKI_ACTIONS = (
    "keep_monitoring",
    "suspect_aki",
    "trigger_aki_alert",
)

AKI_NON_MONOTONIC_ACTIONS = (
    "no_aki",
    "aki_stage_1",
    "aki_stage_2",
    "aki_stage_3",
)

RESP_SUPPORT_ACTIONS = (
    "room_air_or_low_support",
    "high_flow_or_noninvasive_support",
    "invasive_vent_required",
)

ACTIONS = tuple(
    dict.fromkeys(
        SEPSIS_ACTIONS + INFECTION_ONLY_ACTIONS + AKI_ACTIONS + AKI_NON_MONOTONIC_ACTIONS + RESP_SUPPORT_ACTIONS
    )
)

TASK_NAMES = (
    "sepsis",
    "infection_only",
    "aki",
    "respiratory_support",
    "general_icu_surveillance",
)

TASK_MODES = (
    "auto",
    "single",
    "multitask",
    "surveillance",
)

TOOL_BACKENDS = (
    "official",
    "autoformalized",
    "zeroshot_python",
    "zeroshot_sql",
    "zeroshot_raw",
)

SURVEILLANCE_GLOBAL_ACTIONS = (
    "continue_monitoring",
    "escalate",
)

SURVEILLANCE_PRIORITY_LEVELS = (
    "low",
    "medium",
    "high",
)

CODE_EXEC_TOOL_NAME = "run_python"
SQL_EXEC_TOOL_NAME = "run_sql"

DEFAULT_TOOL_NAMES = [
    "query_suspicion_of_infection",
    "query_sofa",
]

SHARED_TOOLBOX_TOOL_NAMES = [
    "query_suspicion_of_infection",
    "query_sofa",
    "query_kdigo_stage",
    "query_ventilation_status",
    "query_urine_output_rate",
    "query_vasoactive_agent",
    "query_vitalsign",
    "query_bg",
    "query_gcs",
    "query_antibiotic",
    "query_invasive_line",
]

SEPSIS_CORE_TOOL_NAMES = [
    "query_suspicion_of_infection",
    "query_sofa",
]

SEPSIS_TOOLBOX_TOOL_NAMES = list(SHARED_TOOLBOX_TOOL_NAMES)

MULTITASK_TOOL_NAMES = [
    "query_suspicion_of_infection",
    "query_sofa",
    "query_kdigo_stage",
    "query_ventilation_status",
]

TASK_LABEL_SPACES: dict[str, list[str]] = {
    "sepsis": list(SEPSIS_ACTIONS),
    "infection_only": list(INFECTION_ONLY_ACTIONS),
    "aki": list(AKI_ACTIONS),
    "respiratory_support": list(RESP_SUPPORT_ACTIONS),
}

TASK_TOOL_NAMES: dict[str, list[str]] = {
    "sepsis": ["query_suspicion_of_infection", "query_sofa"],
    "infection_only": ["query_suspicion_of_infection"],
    "aki": ["query_kdigo_stage"],
    "respiratory_support": ["query_ventilation_status"],
}

TASK_TRANSITION_FIELDS: dict[str, dict[str, str]] = {
    "sepsis": {
        "infection_suspect": "infection_start_hour",
        "trigger_sepsis_alert": "sepsis_start_hour",
    },
    "infection_only": {
        "infection_suspect": "infection_start_hour",
    },
    "aki": {
        "suspect_aki": "aki_stage1_start_hour",
        "trigger_aki_alert": "aki_stage23_start_hour",
    },
    "respiratory_support": {
        "high_flow_or_noninvasive_support": "medium_support_start_hour",
        "invasive_vent_required": "invasive_support_start_hour",
    },
}

TASK_BASELINE_ACTION: dict[str, str] = {
    "sepsis": SEPSIS_ACTIONS[0],
    "infection_only": INFECTION_ONLY_ACTIONS[0],
    "aki": AKI_ACTIONS[0],
    "respiratory_support": RESP_SUPPORT_ACTIONS[0],
}


@dataclass(slots=True)
class Checkpoint:
    t_hour: int
    state_label: str | None = None
    task_labels: dict[str, str] | None = None
    surveillance_labels: dict[str, Any] | None = None
    checkpoint_time: str | None = None
    terminal: bool | None = None
    terminal_by_task: dict[str, bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Trajectory:
    trajectory_id: str
    stay_id: int
    subject_id: int
    hadm_id: int
    anchor: str
    step_hours: int
    horizon_hours: int
    transitions: dict[str, Any]
    checkpoints: list[Checkpoint]
    icu_intime: str | None = None
    icu_outtime: str | None = None
    icu_los_hours: float | None = None
    is_sepsis: bool | None = None
    task_name: str | None = None
    task_variant: str | None = None
    task_names: list[str] | None = None
    tool_names: list[str] | None = None
    label_spaces: dict[str, list[str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoints"] = [checkpoint.to_dict() for checkpoint in self.checkpoints]
        return payload

    def is_multitask(self) -> bool:
        return bool(self.task_names and len(self.task_names) > 1)

    def is_single_task(self) -> bool:
        return not self.is_multitask()

    def resolved_task_names(self) -> list[str]:
        if self.task_names:
            return self.task_names
        if self.task_name:
            return [self.task_name]
        return ["sepsis"]

    def primary_task_name(self) -> str:
        return self.resolved_task_names()[0]

    def resolved_tool_names(self) -> list[str]:
        if self.tool_names:
            return self.tool_names
        if self.is_multitask():
            return list(MULTITASK_TOOL_NAMES)
        return list(TASK_TOOL_NAMES.get(self.primary_task_name(), DEFAULT_TOOL_NAMES))


@dataclass(slots=True)
class ICUStay:
    stay_id: int
    subject_id: int
    hadm_id: int
    icu_intime: datetime


@dataclass(slots=True)
class SuspicionOfInfectionRow:
    stay_id: int
    suspected_infection_time: datetime
    antibiotic: str | None = None
    antibiotic_time: datetime | None = None
    culture_time: datetime | None = None
    specimen: str | None = None
    positive_culture: int | None = None


@dataclass(slots=True)
class Sepsis3Row:
    stay_id: int
    sepsis_time: datetime


@dataclass(slots=True)
class SofaRow:
    stay_id: int
    hr: int
    sofa_24hours: int
    respiration_24hours: int | None = None
    coagulation_24hours: int | None = None
    liver_24hours: int | None = None
    cardiovascular_24hours: int | None = None
    cns_24hours: int | None = None
    renal_24hours: int | None = None

    def components(self) -> dict[str, int | None]:
        return {
            "respiration_24hours": self.respiration_24hours,
            "coagulation_24hours": self.coagulation_24hours,
            "liver_24hours": self.liver_24hours,
            "cardiovascular_24hours": self.cardiovascular_24hours,
            "cns_24hours": self.cns_24hours,
            "renal_24hours": self.renal_24hours,
        }


@dataclass(slots=True)
class AgentStepInput:
    trajectory_id: str
    stay_id: int
    step_index: int
    t_hour: int
    available_tools: list[str]
    instruction: str
    task_names: list[str] | None = None
    task_variant: str | None = None
    label_spaces: dict[str, list[str]] | None = None
    task_mode: str | None = None
    tool_backend: str | None = None
    max_step_interactions: int | None = None
    protocol: str | None = None
    rolling_history: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ActionDecision:
    action: str | None = None
    task_actions: dict[str, str] | None = None
    surveillance: dict[str, Any] | None = None


@dataclass(slots=True)
class StepRecord:
    step_index: int
    t_hour: int
    gt_action: str | None = None
    predicted_action: str | None = None
    gt_task_actions: dict[str, str] | None = None
    predicted_task_actions: dict[str, str] | None = None
    gt_surveillance: dict[str, Any] | None = None
    predicted_surveillance: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    resource_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrajectoryRollout:
    trajectory_id: str
    stay_id: int
    steps: list[StepRecord]
    first_predicted_infection_hour: int | None = None
    first_predicted_alert_hour: int | None = None
    first_predicted_task_hours: dict[str, dict[str, int | None]] | None = None
    resource_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
