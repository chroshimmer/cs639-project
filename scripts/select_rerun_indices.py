#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

QUERY_HINTS = (
    "how many", "what is", "tell me", "full path", "find out", "determine",
    "calculate", "count", "single integer", "answer as an integer", "yes or no",
    "what will be the output", "locate"
)

PROTOCOL_HINTS = (
    "tool_calls must be followed by tool messages",
    "tool call",
    "multiple tool",
    "to_openai_chat_completion_input",
    "dict has no attribute",
)

OVER_INTERVENTION_HINTS = (
    "[WORKING PLAN]",
    "[Monitor] Recovery required.",
)

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def compact_ranges(indices: list[int]) -> list[str]:
    if not indices:
        return []
    indices = sorted(set(indices))
    out = []
    start = prev = indices[0]
    for x in indices[1:]:
        if x == prev + 1:
            prev = x
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = x
    out.append(f"{start}-{prev}" if start != prev else str(start))
    return out

def load_trace_text(run_dir: Path, idx: int) -> str:
    candidates = [
        run_dir / str(idx) / "trace.json",
        run_dir / str(idx) / "messages.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    for p in run_dir.rglob("trace.json"):
        if p.parent.name == str(idx):
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    return ""

def entry_index(entry: dict) -> int | None:
    for key in ("index", "idx", "sample_index"):
        if key in entry:
            try:
                return int(entry[key])
            except Exception:
                pass
    return None

def entry_status(entry: dict) -> str:
    for key in ("status", "sample_status", "result"):
        if key in entry:
            return str(entry[key]).lower()
    return ""

def entry_success(entry: dict) -> bool | None:
    for key in ("success", "passed", "completed"):
        if key in entry and isinstance(entry[key], bool):
            return entry[key]
    reward = entry.get("reward")
    if isinstance(reward, (int, float)):
        if reward == 1:
            return True
        if reward == 0:
            return False
    return None

def failed(entry: dict) -> bool:
    success = entry_success(entry)
    status = entry_status(entry)
    if success is False:
        return True
    if any(k in status for k in ("task error", "model error", "server error", "task limit reached", "failed")):
        return True
    reward = entry.get("reward")
    if isinstance(reward, (int, float)) and reward == 0:
        return True
    return False

def likely_improved(entry: dict, run_dir: Path) -> bool:
    idx = entry_index(entry)
    if idx is None:
        return False

    status = entry_status(entry)
    success = entry_success(entry)
    if success is True:
        return False

    text = json.dumps(entry, ensure_ascii=False).lower()
    if any(k in status for k in ("task error", "model error", "server error")):
        return True
    if "task limit reached" in status:
        return True

    trace = load_trace_text(run_dir, idx).lower()
    hay = text + "\n" + trace

    if any(q in hay for q in QUERY_HINTS) and "task type: state-change task" in hay:
        return True
    if sum(1 for h in OVER_INTERVENTION_HINTS if h in hay) >= 2:
        return True
    if any(p in hay for p in PROTOCOL_HINTS):
        return True

    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--mode", choices=["failed", "likely_improved"], default="failed")
    ap.add_argument("--format", choices=["ranges", "list", "bash_loop"], default="ranges")
    ap.add_argument("--task-name", default="os-std-monitor-replan")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--controller", default="http://localhost:5020/api")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--output-dir", default="results")
    args = ap.parse_args()

    results_jsonl = args.run_dir / "results.jsonl"
    if not results_jsonl.exists():
        raise SystemExit(f"Could not find {results_jsonl}")

    chosen = []
    for entry in read_jsonl(results_jsonl):
        idx = entry_index(entry)
        if idx is None:
            continue
        if args.mode == "failed":
            if failed(entry):
                chosen.append(idx)
        else:
            if likely_improved(entry, args.run_dir):
                chosen.append(idx)

    chosen = sorted(set(chosen))

    if args.format == "list":
        print(" ".join(str(i) for i in chosen))
        return
    if args.format == "ranges":
        print(",".join(compact_ranges(chosen)))
        return

    print("for IDX in " + " ".join(str(i) for i in chosen) + "; do")
    print("  agentrl-eval \\")
    print("    --no-interactive \\")
    print(f"    -c {args.controller} \\")
    print(f"    -u {args.base_url} \\")
    print(f"    -m {args.model} \\")
    print("    --concurrency 1 \\")
    print("    -n 1 \\")
    print(f"    -o {args.output_dir} \\")
    print('    --indices-range "$IDX-$IDX" \\')
    print(f"    {args.task_name}")
    print("done")

if __name__ == "__main__":
    main()
