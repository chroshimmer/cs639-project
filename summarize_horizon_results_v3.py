import json
import sys
import csv
from pathlib import Path
from collections import defaultdict

def cf_from_index(idx):
    idx = int(idx)
    if 0 <= idx <= 45:
        return 0
    if 46 <= idx <= 145:
        return 1 + (idx - 46) // 10
    return None

def extract_index(obj):
    idx = obj.get("index")

    if isinstance(idx, int):
        return idx

    if isinstance(idx, str) and idx.isdigit():
        return int(idx)

    if isinstance(idx, dict):
        if idx.get("int_value") is not None:
            return int(idx["int_value"])
        if idx.get("str_value") is not None and str(idx["str_value"]).isdigit():
            return int(idx["str_value"])

    return None

def extract_success(obj):
    # agentrl-eval current schema
    for k in ["metric_success_rate", "metric_score", "metric_reward"]:
        if k in obj and obj[k] is not None:
            return float(obj[k]) > 0.5

    # fallback fields
    for k in ["success", "correct", "passed"]:
        if k in obj:
            v = obj[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return float(v) > 0.5
            if isinstance(v, str):
                return v.strip().lower() in {"true", "success", "passed", "correct", "1"}

    return None

def find_result_jsonl(root):
    root = Path(root)
    files = sorted(root.rglob("results.jsonl"))
    if not files:
        raise FileNotFoundError(f"No results.jsonl found under {root}")

    # If there are multiple runs, use all of them. Better practice: pass the exact run dir.
    return files

def summarize(root):
    stats = defaultdict(lambda: {"n": 0, "correct": 0})
    rows = []
    unparsed = []

    for path in find_result_jsonl(root):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                idx = extract_index(obj)
                ok = extract_success(obj)

                if idx is None or ok is None:
                    unparsed.append((str(path), line_no, obj))
                    continue

                cf = cf_from_index(idx)
                if cf is None:
                    unparsed.append((str(path), line_no, obj))
                    continue

                stats[cf]["n"] += 1
                stats[cf]["correct"] += int(ok)

                rows.append({
                    "index": idx,
                    "tier": f"cf{cf}",
                    "success": int(ok),
                    "metric_success_rate": obj.get("metric_success_rate"),
                    "metric_score": obj.get("metric_score"),
                    "metric_reward": obj.get("metric_reward"),
                    "status": obj.get("status"),
                    "raw_trace": obj.get("raw_trace"),
                    "source": str(path),
                })

    return stats, rows, unparsed

def print_table(label, stats):
    print(f"\n=== {label} ===")
    print(f"{'tier':<6}{'n':>6}{'correct':>10}{'accuracy':>12}")
    print("-" * 34)

    total_n = 0
    total_correct = 0

    for cf in range(11):
        n = stats[cf]["n"]
        c = stats[cf]["correct"]
        acc = 100 * c / n if n else 0
        print(f"cf{cf:<4}{n:>6}{c:>10}{acc:>11.1f}%")
        total_n += n
        total_correct += c

    total_acc = 100 * total_correct / total_n if total_n else 0
    print("-" * 34)
    print(f"{'all':<6}{total_n:>6}{total_correct:>10}{total_acc:>11.1f}%")

def write_csv(root, rows):
    out = Path(root) / "tier_summary_rows.csv"
    if not rows:
        return

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["index"]))

    print(f"Wrote per-task CSV: {out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_horizon_results_v3.py RESULTS_DIR [RESULTS_DIR2 ...]")
        sys.exit(1)

    for root in sys.argv[1:]:
        stats, rows, unparsed = summarize(root)
        print_table(root, stats)
        write_csv(root, rows)

        if unparsed:
            print(f"\nWarning: {len(unparsed)} unparsed rows")
            path, line_no, obj = unparsed[0]
            print("First unparsed row:")
            print("path:", path)
            print("line:", line_no)
            print(json.dumps(obj, indent=2)[:2000])
