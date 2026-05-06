from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agent import _build_toolbox_messages
from .cli import _filter_trajectories_by_split
from .dataset import load_dataset_auto
from .environment import _build_rolling_history_entry, _empty_toolbox_state, _update_toolbox_state_from_output
from .schemas import SHARED_TOOLBOX_TOOL_NAMES
from .tools import build_tool_runtime


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _write_records(records: list[dict[str, Any]], path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        path.write_text("".join(json.dumps(record, default=_json_default) + "\n" for record in records))
        return
    if output_format == "parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise SystemExit("Parquet export requires pandas.") from exc
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise SystemExit("Parquet export requires pyarrow. Install requirements-verl.txt on the training machine.") from exc
        pd.DataFrame(records).to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {output_format}")


def _record_for_checkpoint(
    *,
    trajectory: Any,
    checkpoint: Any,
    step_index: int,
    rolling_history: list[dict[str, Any]],
    available_tools: list[str],
    db_path: str,
    max_step_interactions: int,
) -> dict[str, Any]:
    step_input = {
        "trajectory_id": trajectory.trajectory_id,
        "stay_id": trajectory.stay_id,
        "step_index": step_index,
        "t_hour": checkpoint.t_hour,
        "available_tools": available_tools,
        "instruction": "Use tools if needed. Then output exactly one action for task 'sepsis'.",
        "task_names": ["sepsis"],
        "label_spaces": trajectory.label_spaces or {"sepsis": ["keep_monitoring", "infection_suspect", "trigger_sepsis_alert"]},
        "task_mode": "single",
        "tool_backend": "official",
        "max_step_interactions": max_step_interactions,
        "protocol": "rolling_toolbox_with_history",
        "rolling_history": list(rolling_history),
    }
    messages = _build_toolbox_messages(step_input, history=[], available_tools=available_tools)
    extra_info = {
        "trajectory_id": trajectory.trajectory_id,
        "stay_id": trajectory.stay_id,
        "step_index": step_index,
        "t_hour": checkpoint.t_hour,
        "gt_action": checkpoint.state_label,
        "transitions_json": json.dumps(trajectory.transitions, default=_json_default),
        "rolling_history_json": json.dumps(rolling_history, default=_json_default),
        "db_path": db_path,
        "tool_scope": "sepsis_core" if available_tools == ["query_suspicion_of_infection", "query_sofa"] else "shared",
        "available_tools": available_tools,
        "max_step_interactions": max_step_interactions,
    }
    return {
        "data_source": "rolling_sepsis_toolbox",
        "prompt": messages,
        "raw_prompt": messages,
        "agent_name": "sepsis_tool_agent_loop",
        "reward_model": {"style": "rule", "ground_truth": checkpoint.state_label},
        "ground_truth": checkpoint.state_label,
        "extra_info": extra_info,
        "trajectory_id": trajectory.trajectory_id,
        "stay_id": trajectory.stay_id,
        "step_index": step_index,
        "t_hour": checkpoint.t_hour,
    }


def export_split(args: argparse.Namespace, split_name: str, output_path: Path) -> dict[str, Any]:
    trajectories = load_dataset_auto(args.dataset, strict_mvp=not args.include_out_of_scope)
    trajectories = _filter_trajectories_by_split(trajectories, args.split, split_name)
    unsupported = [
        trajectory.trajectory_id
        for trajectory in trajectories
        if trajectory.is_multitask() or trajectory.primary_task_name() != "sepsis"
    ]
    if unsupported:
        raise SystemExit(f"verl export supports only single-task sepsis. First unsupported: {unsupported[:5]}")

    runtime = build_tool_runtime(tool_backend="official", db_path=args.db_path)
    available_tools = (
        ["query_suspicion_of_infection", "query_sofa"]
        if args.tool_scope == "sepsis_core"
        else list(SHARED_TOOLBOX_TOOL_NAMES)
    )
    records = []
    for trajectory in trajectories:
        rolling_history: list[dict[str, Any]] = []
        state = _empty_toolbox_state()
        for step_index, checkpoint in enumerate(trajectory.checkpoints):
            records.append(
                _record_for_checkpoint(
                    trajectory=trajectory,
                    checkpoint=checkpoint,
                    step_index=step_index,
                    rolling_history=rolling_history,
                    available_tools=available_tools,
                    db_path=args.db_path,
                    max_step_interactions=args.max_step_interactions,
                )
            )
            tool_outputs: list[dict[str, Any]] = []
            if not state["infection_positive"]:
                output = runtime.execute(
                    "query_suspicion_of_infection",
                    {"stay_id": trajectory.stay_id, "t_hour": checkpoint.t_hour},
                )
                tool_outputs.append(output)
                _update_toolbox_state_from_output(state, "query_suspicion_of_infection", output)
            if state["infection_positive"] and not state["sofa_alert"]:
                output = runtime.execute(
                    "query_sofa",
                    {"stay_id": trajectory.stay_id, "t_hour": checkpoint.t_hour},
                )
                tool_outputs.append(output)
                _update_toolbox_state_from_output(state, "query_sofa", output)
            history_entry = _build_rolling_history_entry(
                trajectory=trajectory,
                checkpoint=checkpoint,
                step_index=step_index,
                tool_outputs=tool_outputs,
            )
            if history_entry is not None:
                rolling_history.append(history_entry)
    _write_records(records, output_path, args.output_format)
    return {"split": split_name, "output": str(output_path), "records": len(records), "trajectories": len(trajectories)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export sepsis tool-call data for verl Agent Loop GRPO.")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tool-scope", choices=["sepsis_core", "shared"], default="sepsis_core")
    parser.add_argument("--max-step-interactions", type=int, default=3)
    parser.add_argument("--output-format", choices=["parquet", "jsonl"], default="parquet")
    parser.add_argument("--include-out-of-scope", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    suffix = "parquet" if args.output_format == "parquet" else "jsonl"
    summaries = [
        export_split(args, split_name, output_dir / f"{split_name}.{suffix}")
        for split_name in ("train", "val", "test")
    ]
    manifest = {
        "dataset": args.dataset,
        "db_path": args.db_path,
        "split": args.split,
        "tool_scope": args.tool_scope,
        "max_step_interactions": args.max_step_interactions,
        "output_format": args.output_format,
        "splits": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    print(json.dumps(manifest, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
