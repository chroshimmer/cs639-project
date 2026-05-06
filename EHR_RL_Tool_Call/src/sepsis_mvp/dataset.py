from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .schemas import (
    AKI_ACTIONS,
    AKI_NON_MONOTONIC_ACTIONS,
    DEFAULT_TOOL_NAMES,
    INFECTION_ONLY_ACTIONS,
    MULTITASK_TOOL_NAMES,
    RESP_SUPPORT_ACTIONS,
    SEPSIS_ACTIONS,
    TASK_BASELINE_ACTION,
    TASK_LABEL_SPACES,
    TASK_TOOL_NAMES,
    Checkpoint,
    ICUStay,
    Sepsis3Row,
    SofaRow,
    SuspicionOfInfectionRow,
    Trajectory,
)


MVP_ACTIONS = set(SEPSIS_ACTIONS)


@dataclass(slots=True)
class CSVLoadResult:
    trajectories: list[Trajectory]
    included_trajectories: int
    skipped_trajectories: int
    skipped_reasons: dict[str, int]


def parse_datetime(value: str | None) -> datetime | None:
    if value in (None, ""):
        return None
    return datetime.fromisoformat(value)


def load_concept_tables(path: str | Path) -> dict[str, list[Any]]:
    data = json.loads(Path(path).read_text())
    icustays = [
        ICUStay(
            stay_id=int(row["stay_id"]),
            subject_id=int(row["subject_id"]),
            hadm_id=int(row["hadm_id"]),
            icu_intime=parse_datetime(row["icu_intime"]),
        )
        for row in data["icustays"]
    ]
    suspicion = [
        SuspicionOfInfectionRow(
            stay_id=int(row["stay_id"]),
            suspected_infection_time=parse_datetime(row["suspected_infection_time"]),
            antibiotic=row.get("antibiotic"),
            antibiotic_time=parse_datetime(row.get("antibiotic_time")),
            culture_time=parse_datetime(row.get("culture_time")),
            specimen=row.get("specimen"),
            positive_culture=row.get("positive_culture"),
        )
        for row in data.get("suspicion_of_infection", [])
    ]
    sepsis = [
        Sepsis3Row(
            stay_id=int(row["stay_id"]),
            sepsis_time=parse_datetime(row["sepsis_time"]),
        )
        for row in data.get("sepsis3", [])
    ]
    sofa = [
        SofaRow(
            stay_id=int(row["stay_id"]),
            hr=int(row["hr"]),
            sofa_24hours=int(row["sofa_24hours"]),
            respiration_24hours=row.get("respiration_24hours"),
            coagulation_24hours=row.get("coagulation_24hours"),
            liver_24hours=row.get("liver_24hours"),
            cardiovascular_24hours=row.get("cardiovascular_24hours"),
            cns_24hours=row.get("cns_24hours"),
            renal_24hours=row.get("renal_24hours"),
        )
        for row in data.get("sofa", [])
    ]
    return {
        "icustays": icustays,
        "suspicion_of_infection": suspicion,
        "sepsis3": sepsis,
        "sofa": sofa,
    }


def save_trajectories(trajectories: list[Trajectory], path: str | Path) -> None:
    payload = [trajectory.to_dict() for trajectory in trajectories]
    Path(path).write_text(json.dumps(payload, indent=2))


def load_trajectories(path: str | Path) -> list[Trajectory]:
    raw = json.loads(Path(path).read_text())
    trajectories = []
    for item in raw:
        trajectories.append(
            Trajectory(
                trajectory_id=item["trajectory_id"],
                stay_id=int(item["stay_id"]),
                subject_id=int(item["subject_id"]),
                hadm_id=int(item["hadm_id"]),
                anchor=item["anchor"],
                step_hours=int(item["step_hours"]),
                horizon_hours=int(item["horizon_hours"]),
                transitions=item["transitions"],
                checkpoints=[Checkpoint(**checkpoint) for checkpoint in item["checkpoints"]],
                icu_intime=item.get("icu_intime"),
                icu_outtime=item.get("icu_outtime"),
                icu_los_hours=item.get("icu_los_hours"),
                is_sepsis=item.get("is_sepsis"),
                task_name=item.get("task_name"),
                task_variant=item.get("task_variant"),
                task_names=item.get("task_names"),
                tool_names=item.get("tool_names"),
                label_spaces=item.get("label_spaces"),
            )
        )
    return trajectories


def load_dataset_auto(
    path: str | Path,
    *,
    strict_mvp: bool = True,
) -> list[Trajectory]:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        with dataset_path.open() as handle:
            reader = csv.DictReader(handle)
            first_row = next(reader)
        if "global_action" in first_row and "suspected_conditions" in first_row and "alerts" in first_row:
            return load_general_surveillance_csv_dataset(dataset_path).trajectories
        if first_row.get("task_name") == "infection_only":
            return load_single_task_csv_dataset(dataset_path, task_name="infection_only").trajectories
        if "sepsis_label" in first_row:
            return load_multitask_csv_dataset(dataset_path).trajectories
        if "aki_stage1_start_hour" in first_row:
            return load_single_task_csv_dataset(dataset_path, task_name="aki").trajectories
        if "current_aki_stage_smoothed" in first_row or "path_family" in first_row:
            return load_single_task_csv_dataset(dataset_path, task_name="aki").trajectories
        if "medium_support_start_hour" in first_row or "invasive_support_start_hour" in first_row:
            return load_single_task_csv_dataset(dataset_path, task_name="respiratory_support").trajectories
        return load_rolling_csv_dataset(dataset_path, strict_mvp=strict_mvp).trajectories
    if suffix == ".json":
        return load_trajectories(dataset_path)
    raise ValueError(f"Unsupported dataset format for {dataset_path}. Expected .csv or .json.")


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    if str(value).strip().lower() in {"", "null", "none", "nan"}:
        return None
    return int(float(value))


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    if str(value).strip().lower() in {"", "null", "none", "nan"}:
        return None
    return float(value)


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "null", "none", "nan"}:
        return None
    return normalized == "true" or normalized == "1"


def _parse_pipe_list(value: str | None) -> list[str]:
    if value is None:
        return []
    value = value.strip()
    if value.lower() in {"", "null", "none", "nan"}:
        return []
    return [item for item in value.split("|") if item]


def _trajectory_skip_reason(rows: list[dict[str, str]]) -> str | None:
    labels = {row["state_label"] for row in rows}
    if not labels.issubset(MVP_ACTIONS):
        return "unsupported_labels"

    first = rows[0]
    infection_start_hour = _parse_optional_int(first.get("infection_start_hour"))
    sepsis_start_hour = _parse_optional_int(first.get("sepsis_start_hour"))
    if (
        infection_start_hour is not None
        and sepsis_start_hour is not None
        and infection_start_hour > sepsis_start_hour
    ):
        return "infection_after_sepsis"
    return None


def load_rolling_csv_dataset(
    path: str | Path,
    *,
    strict_mvp: bool = True,
) -> CSVLoadResult:
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped_rows[row["trajectory_id"]].append(row)

    trajectories: list[Trajectory] = []
    skipped_reasons: dict[str, int] = defaultdict(int)

    for trajectory_id, rows in sorted(grouped_rows.items()):
        rows.sort(key=lambda row: int(row["t_hour"]))
        if strict_mvp:
            reason = _trajectory_skip_reason(rows)
            if reason is not None:
                skipped_reasons[reason] += 1
                continue

        first = rows[0]
        checkpoints = [
            Checkpoint(
                t_hour=int(row["t_hour"]),
                state_label=row["state_label"],
                checkpoint_time=row.get("checkpoint_time") or None,
                terminal=_parse_bool(row.get("terminal")),
            )
            for row in rows
        ]
        step_hours = checkpoints[1].t_hour - checkpoints[0].t_hour if len(checkpoints) > 1 else 4
        horizon_hours = checkpoints[-1].t_hour

        transitions = {
            "infection_start_hour": _parse_optional_int(first.get("infection_start_hour")),
            "sepsis_start_hour": _parse_optional_int(first.get("sepsis_start_hour")),
            "infection_start_time": first.get("infection_start_time") or None,
            "sepsis_start_time": first.get("sepsis_start_time") or None,
        }
        organ_dysfunction_start_hour = _parse_optional_int(first.get("organ_dysfunction_start_hour"))
        organ_dysfunction_start_time = first.get("organ_dysfunction_start_time") or None
        if organ_dysfunction_start_hour is not None or organ_dysfunction_start_time is not None:
            transitions["organ_dysfunction_start_hour"] = organ_dysfunction_start_hour
            transitions["organ_dysfunction_start_time"] = organ_dysfunction_start_time

        trajectories.append(
            Trajectory(
                trajectory_id=trajectory_id,
                stay_id=int(first["stay_id"]),
                subject_id=int(first["subject_id"]),
                hadm_id=int(first["hadm_id"]),
                anchor="icu_intime",
                step_hours=step_hours,
                horizon_hours=horizon_hours,
                transitions=transitions,
                checkpoints=checkpoints,
                icu_intime=first.get("icu_intime") or None,
                icu_outtime=first.get("icu_outtime") or None,
                icu_los_hours=_parse_optional_float(first.get("icu_los_hours")),
                is_sepsis=_parse_bool(first.get("is_sepsis")),
                task_name="sepsis",
                task_names=["sepsis"],
                tool_names=list(DEFAULT_TOOL_NAMES),
                label_spaces={"sepsis": list(SEPSIS_ACTIONS)},
            )
        )

    return CSVLoadResult(
        trajectories=trajectories,
        included_trajectories=len(trajectories),
        skipped_trajectories=sum(skipped_reasons.values()),
        skipped_reasons=dict(skipped_reasons),
    )


def load_multitask_csv_dataset(path: str | Path) -> CSVLoadResult:
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped_rows[row["trajectory_id"]].append(row)

    trajectories: list[Trajectory] = []
    for trajectory_id, rows in sorted(grouped_rows.items()):
        rows.sort(key=lambda row: int(row["t_hour"]))
        first = rows[0]
        checkpoints = [
            Checkpoint(
                t_hour=int(row["t_hour"]),
                checkpoint_time=row.get("checkpoint_time") or None,
                task_labels={
                    "sepsis": row["sepsis_label"],
                    "aki": row["aki_label"],
                    "respiratory_support": row["respiratory_support_label"],
                },
                terminal_by_task={
                    "sepsis": _parse_bool(row.get("sepsis_terminal")) or False,
                    "aki": _parse_bool(row.get("aki_terminal")) or False,
                    "respiratory_support": _parse_bool(row.get("respiratory_support_terminal")) or False,
                },
                terminal=_parse_bool(row.get("terminal_any")),
            )
            for row in rows
        ]
        step_hours = checkpoints[1].t_hour - checkpoints[0].t_hour if len(checkpoints) > 1 else 4
        horizon_hours = checkpoints[-1].t_hour

        trajectories.append(
            Trajectory(
                trajectory_id=trajectory_id,
                stay_id=int(first["stay_id"]),
                subject_id=int(first["subject_id"]),
                hadm_id=int(first["hadm_id"]),
                anchor="icu_intime",
                step_hours=step_hours,
                horizon_hours=horizon_hours,
                transitions={
                    "sepsis": {
                        "infection_start_hour": _parse_optional_int(first.get("infection_start_hour")),
                        "sepsis_start_hour": _parse_optional_int(first.get("sepsis_start_hour")),
                        "infection_start_time": first.get("infection_start_time") or None,
                        "sepsis_start_time": first.get("sepsis_start_time") or None,
                    },
                    "aki": {
                        "aki_stage1_start_hour": _parse_optional_int(first.get("aki_stage1_start_hour")),
                        "aki_stage23_start_hour": _parse_optional_int(first.get("aki_stage23_start_hour")),
                        "aki_stage1_start_time": first.get("aki_stage1_start_time") or None,
                        "aki_stage23_start_time": first.get("aki_stage23_start_time") or None,
                    },
                    "respiratory_support": {
                        "medium_support_start_hour": _parse_optional_int(first.get("medium_support_start_hour")),
                        "invasive_support_start_hour": _parse_optional_int(first.get("invasive_support_start_hour")),
                        "medium_support_start_time": first.get("medium_support_start_time") or None,
                        "invasive_support_start_time": first.get("invasive_support_start_time") or None,
                    },
                },
                checkpoints=checkpoints,
                icu_intime=first.get("icu_intime") or None,
                icu_outtime=first.get("icu_outtime") or None,
                icu_los_hours=_parse_optional_float(first.get("icu_los_hours")),
                is_sepsis=_parse_bool(first.get("is_sepsis")),
                task_name="multitask",
                task_names=["sepsis", "aki", "respiratory_support"],
                tool_names=list(MULTITASK_TOOL_NAMES),
                label_spaces={task: labels[:] for task, labels in TASK_LABEL_SPACES.items()},
            )
        )

    return CSVLoadResult(
        trajectories=trajectories,
        included_trajectories=len(trajectories),
        skipped_trajectories=0,
        skipped_reasons={},
    )


def load_general_surveillance_csv_dataset(path: str | Path) -> CSVLoadResult:
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped_rows[row["trajectory_id"]].append(row)

    trajectories: list[Trajectory] = []
    for trajectory_id, rows in sorted(grouped_rows.items()):
        rows.sort(key=lambda row: int(row["t_hour"]))
        first = rows[0]
        checkpoints = []
        for row in rows:
            checkpoint_summary = row.get("checkpoint_summary")
            surveillance_labels = {
                "global_action": row["global_action"],
                "suspected_conditions": _parse_pipe_list(row.get("suspected_conditions")),
                "alerts": _parse_pipe_list(row.get("alerts")),
                "priority": row.get("priority") or None,
                "recommended_next_tools": _parse_pipe_list(row.get("recommended_next_tools")),
                "rationale": row.get("rationale") or None,
                "checkpoint_summary": checkpoint_summary or None,
            }
            checkpoints.append(
                Checkpoint(
                    t_hour=int(row["t_hour"]),
                    state_label=row["global_action"],
                    surveillance_labels=surveillance_labels,
                    checkpoint_time=row.get("checkpoint_time") or None,
                    terminal=_parse_bool(row.get("terminal")),
                )
            )

        step_hours = checkpoints[1].t_hour - checkpoints[0].t_hour if len(checkpoints) > 1 else 4
        horizon_hours = checkpoints[-1].t_hour
        trajectories.append(
            Trajectory(
                trajectory_id=trajectory_id,
                stay_id=int(first["stay_id"]),
                subject_id=int(first["subject_id"]),
                hadm_id=int(first["hadm_id"]),
                anchor="icu_intime",
                step_hours=step_hours,
                horizon_hours=horizon_hours,
                transitions={},
                checkpoints=checkpoints,
                icu_intime=first.get("icu_intime") or None,
                icu_outtime=first.get("icu_outtime") or None,
                icu_los_hours=_parse_optional_float(first.get("icu_los_hours")),
                task_name="general_icu_surveillance",
                task_variant="surveillance_v1",
                task_names=["general_icu_surveillance"],
                tool_names=[],
                label_spaces={},
            )
        )

    return CSVLoadResult(
        trajectories=trajectories,
        included_trajectories=len(trajectories),
        skipped_trajectories=0,
        skipped_reasons={},
    )


def load_single_task_csv_dataset(path: str | Path, *, task_name: str) -> CSVLoadResult:
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped_rows[row["trajectory_id"]].append(row)

    if task_name == "aki":
        is_non_monotonic = any(
            key in grouped_rows[next(iter(grouped_rows))]
            for key in ("current_aki_stage_smoothed", "path_family")
        )
        if is_non_monotonic:
            transition_keys = {
                "path_family": "path_family",
                "path_0_24": "path_0_24",
                "max_stage_24h": "max_stage_24h",
                "has_up_24h": "has_up_24h",
                "has_down_24h": "has_down_24h",
                "num_changes_24h": "num_changes_24h",
            }
        else:
            transition_keys = {
                "aki_stage1_start_hour": "aki_stage1_start_hour",
                "aki_stage23_start_hour": "aki_stage23_start_hour",
                "aki_stage1_start_time": "aki_stage1_start_time",
                "aki_stage23_start_time": "aki_stage23_start_time",
            }
    elif task_name == "respiratory_support":
        transition_keys = {
            "medium_support_start_hour": "medium_support_start_hour",
            "invasive_support_start_hour": "invasive_support_start_hour",
            "medium_support_start_time": "medium_support_start_time",
            "invasive_start_time": "invasive_support_start_time",
            "invasive_support_start_time": "invasive_support_start_time",
        }
    elif task_name == "infection_only":
        transition_keys = {
            "infection_start_hour": "infection_start_hour",
            "infection_start_time": "infection_start_time",
        }
    else:
        raise ValueError(f"Unsupported single-task CSV type: {task_name}")

    trajectories: list[Trajectory] = []
    for trajectory_id, rows in sorted(grouped_rows.items()):
        rows.sort(key=lambda row: int(row["t_hour"]))
        first = rows[0]
        checkpoints = [
            Checkpoint(
                t_hour=int(row["t_hour"]),
                state_label=row["state_label"],
                checkpoint_time=row.get("checkpoint_time") or None,
                terminal=_parse_bool(row.get("terminal")),
            )
            for row in rows
        ]
        step_hours = checkpoints[1].t_hour - checkpoints[0].t_hour if len(checkpoints) > 1 else 4
        horizon_hours = checkpoints[-1].t_hour
        transitions = {}
        for source_key, target_key in transition_keys.items():
            if source_key.endswith("_hour"):
                transitions[target_key] = _parse_optional_int(first.get(source_key))
            elif source_key in {"max_stage_24h", "num_changes_24h"}:
                transitions[target_key] = _parse_optional_int(first.get(source_key))
            elif source_key in {"has_up_24h", "has_down_24h"}:
                transitions[target_key] = _parse_bool(first.get(source_key))
            else:
                transitions[target_key] = first.get(source_key) or None

        task_variant = None
        label_space = list(TASK_LABEL_SPACES[task_name])
        if task_name == "aki" and "current_aki_stage_smoothed" in first:
            task_variant = "non_monotonic_current_state"
            label_space = list(AKI_NON_MONOTONIC_ACTIONS)
        elif task_name == "infection_only":
            label_space = list(INFECTION_ONLY_ACTIONS)

        trajectories.append(
            Trajectory(
                trajectory_id=trajectory_id,
                stay_id=int(first["stay_id"]),
                subject_id=int(first["subject_id"]),
                hadm_id=int(first["hadm_id"]),
                anchor="icu_intime",
                step_hours=step_hours,
                horizon_hours=horizon_hours,
                transitions=transitions,
                checkpoints=checkpoints,
                icu_intime=first.get("icu_intime") or None,
                icu_outtime=first.get("icu_outtime") or None,
                icu_los_hours=_parse_optional_float(first.get("icu_los_hours")),
                task_name=task_name,
                task_variant=task_variant,
                task_names=[task_name],
                tool_names=list(TASK_TOOL_NAMES[task_name]),
                label_spaces={task_name: label_space},
            )
        )

    return CSVLoadResult(
        trajectories=trajectories,
        included_trajectories=len(trajectories),
        skipped_trajectories=0,
        skipped_reasons={},
    )


def hour_offset(anchor: datetime, event_time: datetime | None) -> float | None:
    if event_time is None:
        return None
    return (event_time - anchor).total_seconds() / 3600.0


def checkpoint_hour_for_event(event_hour: float | None, step_hours: int, horizon_hours: int) -> int | None:
    if event_hour is None:
        return None
    snapped = int(math.ceil(max(0.0, event_hour) / step_hours) * step_hours)
    return min(snapped, horizon_hours)


def label_for_checkpoint(
    checkpoint_hour: int,
    infection_start_hour: int | None,
    sepsis_start_hour: int | None,
) -> str:
    if infection_start_hour is None or checkpoint_hour < infection_start_hour:
        return "keep_monitoring"
    if sepsis_start_hour is None or checkpoint_hour < sepsis_start_hour:
        return "infection_suspect"
    return "trigger_sepsis_alert"


def infection_only_label_for_checkpoint(
    checkpoint_hour: int,
    infection_start_hour: int | None,
) -> str:
    if infection_start_hour is None or checkpoint_hour < infection_start_hour:
        return "keep_monitoring"
    return "infection_suspect"


def build_dataset(
    concept_tables: dict[str, list[Any]],
    *,
    step_hours: int = 4,
    horizon_hours: int = 24,
    sample_sepsis: int | None = None,
    sample_non_sepsis: int | None = None,
    seed: int = 7,
) -> list[Trajectory]:
    stays = {row.stay_id: row for row in concept_tables["icustays"]}
    suspicion_by_stay: dict[int, list[SuspicionOfInfectionRow]] = defaultdict(list)
    for row in concept_tables["suspicion_of_infection"]:
        suspicion_by_stay[row.stay_id].append(row)
    sepsis_by_stay: dict[int, list[Sepsis3Row]] = defaultdict(list)
    for row in concept_tables["sepsis3"]:
        sepsis_by_stay[row.stay_id].append(row)

    sepsis_stay_ids = sorted(sepsis_by_stay.keys())
    non_sepsis_stay_ids = sorted(stay_id for stay_id in stays if stay_id not in sepsis_by_stay)

    rng = random.Random(seed)
    if sample_sepsis is not None and sample_sepsis < len(sepsis_stay_ids):
        sepsis_stay_ids = sorted(rng.sample(sepsis_stay_ids, sample_sepsis))
    if sample_non_sepsis is not None and sample_non_sepsis < len(non_sepsis_stay_ids):
        non_sepsis_stay_ids = sorted(rng.sample(non_sepsis_stay_ids, sample_non_sepsis))

    selected_stay_ids = sepsis_stay_ids + non_sepsis_stay_ids
    trajectories: list[Trajectory] = []

    for stay_id in selected_stay_ids:
        stay = stays[stay_id]
        suspicion_rows = sorted(
            suspicion_by_stay.get(stay_id, []),
            key=lambda row: row.suspected_infection_time,
        )
        sepsis_rows = sorted(sepsis_by_stay.get(stay_id, []), key=lambda row: row.sepsis_time)

        first_suspicion_time = suspicion_rows[0].suspected_infection_time if suspicion_rows else None
        first_sepsis_time = sepsis_rows[0].sepsis_time if sepsis_rows else None

        infection_start_hour = checkpoint_hour_for_event(
            hour_offset(stay.icu_intime, first_suspicion_time),
            step_hours,
            horizon_hours,
        )
        sepsis_start_hour = checkpoint_hour_for_event(
            hour_offset(stay.icu_intime, first_sepsis_time),
            step_hours,
            horizon_hours,
        )

        checkpoints = []
        for t_hour in range(0, horizon_hours + 1, step_hours):
            checkpoints.append(
                Checkpoint(
                    t_hour=t_hour,
                    state_label=label_for_checkpoint(t_hour, infection_start_hour, sepsis_start_hour),
                )
            )

        trajectories.append(
            Trajectory(
                trajectory_id=f"mimiciv_stay_{stay_id}",
                stay_id=stay.stay_id,
                subject_id=stay.subject_id,
                hadm_id=stay.hadm_id,
                anchor="icu_intime",
                step_hours=step_hours,
                horizon_hours=horizon_hours,
                transitions={
                    "infection_start_hour": infection_start_hour,
                    "sepsis_start_hour": sepsis_start_hour,
                },
                checkpoints=checkpoints,
                task_name="sepsis",
                task_names=["sepsis"],
                tool_names=list(DEFAULT_TOOL_NAMES),
                label_spaces={"sepsis": list(SEPSIS_ACTIONS)},
            )
        )

    return sorted(trajectories, key=lambda trajectory: trajectory.stay_id)
