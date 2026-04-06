"""
Bucket OS benchmark runs into 9 HORIZON categories using Azure OpenAI.

Usage:
    Create a .env file with:
        AZURE_OPENAI_API_KEY=your_api_key_here
        AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
        AZURE_OPENAI_DEPLOYMENT=gpt-4o
        AZURE_OPENAI_API_VERSION=2024-02-01
    python bucket_failures.py /path/to/target_directory
"""

import json
import sys
import argparse
import csv
import os
import time
from pathlib import Path
from openai import AzureOpenAI
import matplotlib.pyplot as plt
from dotenv import load_dotenv

SUCCESS               = "SUCCESS"
ENV_DISTURBANCE       = "ENV_DISTURBANCE"
UNDETECTED_CHANGE     = "UNDETECTED_CHANGE"
ILL_DEFINED_INSTR     = "ILL_DEFINED_INSTR"
PARTIAL_UNDERSTANDING = "PARTIAL_UNDERSTANDING"
CATASTROPHIC_FORGET   = "CATASTROPHIC_FORGET"
FALSE_ASSUMPTIONS     = "FALSE_ASSUMPTIONS"
PLANNING_ERROR        = "PLANNING_ERROR"
SYSTEM_ERROR          = "SYSTEM_ERROR"

ALL_BUCKETS = [
    SUCCESS,
    ENV_DISTURBANCE, UNDETECTED_CHANGE, ILL_DEFINED_INSTR,
    PARTIAL_UNDERSTANDING, CATASTROPHIC_FORGET, FALSE_ASSUMPTIONS,
    PLANNING_ERROR,
    SYSTEM_ERROR
]

HORIZON_FAILURES = ALL_BUCKETS[1:8]

CLASSIFY_SYSTEM = f"""You are an expert diagnostic AI evaluating LLM agents on OS tasks.
Given a condensed trace of an agent interacting with a bash shell, classify the PRIMARY failure reason into EXACTLY ONE of these categories:

{', '.join(HORIZON_FAILURES)}

Return ONLY the category name. Do not include any other text, punctuation, or explanation."""

def compress_trace(trace_data: list) -> str:
    """
    Parses the trace schema to extract roles, goals, tools, and OS outputs.
    Condenses the trace to maximize API efficiency.
    """
    condensed = []
    for turn in trace_data:
        role = turn.get("role")
        if not role:
            role = turn.get("type", "unknown")
            
        if role in ["system", "user"]:
            content = turn.get("content", "")
            label = "MONITOR" if isinstance(content, str) and content.startswith("[Monitor]") else role.upper()
            if isinstance(content, str) and content.startswith("[STATE MEMORY]"):
                label = "STATE_MEMORY"
            condensed.append(f"[{label}]: {content}")
            
        elif role == "function_call":
            name = turn.get("name", "unknown_tool")
            args = turn.get("arguments", "")
            condensed.append(f"[AGENT_ACTION -> {name}]: {args}")
            
        elif role == "tool":
            content = turn.get("content", "").strip()
            label = "MONITOR" if content.startswith("[Monitor]") else "OS_FEEDBACK"
            condensed.append(f"[{label}]: {content}")
            
    return "\n".join(condensed)

def classify_with_azure(trace_text: str, client: AzureOpenAI, deployment: str) -> str:
    """Calls the Azure OpenAI chat API to categorize the trace deterministically."""
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": f"TRACE:\n{trace_text[-30000:]}"},
            ],
            temperature=0.0,
            max_tokens=32,
        )

        prediction = str(response.choices[0].message.content).strip().upper()

        for bucket in HORIZON_FAILURES:
            if bucket in prediction:
                return bucket

        return SYSTEM_ERROR

    except Exception as e:
        print(f"  [warn] Azure OpenAI API exception: {e}", file=sys.stderr)
        time.sleep(2)
        return SYSTEM_ERROR

def get_trace_data(raw_trace_path: str, target_dir: Path) -> list | None:
    """Loads and parses the raw trace JSON from the target directory."""
    if not raw_trace_path:
        return None
        
    full_path = target_dir / raw_trace_path
    if full_path.exists():
        try:
            with open(full_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def process_record(record: dict, target_dir: Path, client: AzureOpenAI, deployment: str) -> str:
    """Routes the record to the correct bucket based on status, reward, and trace."""
    status = record.get("status", "")
    reward = record.get("metric_reward")

    if reward is None:
        return SYSTEM_ERROR

    if status in ["task limit reached", "task error", "server error", "model error"]:
        return SYSTEM_ERROR

    if status == "completed" and float(reward) == 1.0:
        return SUCCESS

    if status == "completed" and float(reward) == 0.0:
        raw_trace = record.get("raw_trace")
        trace_data = get_trace_data(str(raw_trace), target_dir)
        
        if trace_data:
            print(f"  Classifying trace: {raw_trace}...", file=sys.stderr)
            compressed = compress_trace(trace_data)
            return classify_with_azure(compressed, client, deployment)
            
        return SYSTEM_ERROR

    return SYSTEM_ERROR

def plot_results(counts: dict, output_path: Path):
    """Generates and saves a bar chart of the bucket counts."""
    buckets = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(buckets, values, color='#4A90E2', edgecolor='black')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), str(yval), ha='center', va='bottom', fontsize=10)

    plt.title('OS Agent Long-Horizon Failure Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Failure Category', fontsize=12)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path)
    print(f"\nSaved bar chart to {output_path}")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Bucket benchmark runs via Azure OpenAI and generate a report.")
    parser.add_argument("target_dir", help="Path to the directory containing results.jsonl and trace folders")
    args = parser.parse_args()

    # ── Validate required env vars ────────────────────────────────────────────
    api_key   = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint  = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    missing = [v for v, val in [("AZURE_OPENAI_API_KEY", api_key), ("AZURE_OPENAI_ENDPOINT", endpoint)] if not val]
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        print("Run:\n  export AZURE_OPENAI_API_KEY='your_key'", file=sys.stderr)
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'", file=sys.stderr)
        sys.exit(1)

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=str(endpoint),
        api_version=api_version,
    )

    target_dir = Path(args.target_dir)
    results_file = target_dir / "results.jsonl"
    output_csv = target_dir / "results.csv"
    output_plot = target_dir / "results_chart.png"

    if not results_file.exists():
        print(f"ERROR: Could not find {results_file}", file=sys.stderr)
        sys.exit(1)

    bucketed_records = []
    counts = {bucket: 0 for bucket in ALL_BUCKETS}
    total_runs = 0

    print(f"Loading {results_file}...", file=sys.stderr)
    with open(results_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            total_runs += 1
            record = json.loads(line)
            
            bucket = process_record(record, target_dir, client, deployment)
            record["horizon_bucket"] = bucket
            counts[bucket] += 1
            bucketed_records.append(record)

    # ── Summary Output ────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"HORIZON 9-BUCKET SUMMARY (Model: {deployment})")
    print("="*50)
    
    for bucket in ALL_BUCKETS:
        print(f"{bucket:<25} | {counts[bucket]:>5}")
        
    print("-" * 50)
    print(f"{'TOTAL PROCESSED':<25} | {total_runs:>5}")

    # ── CSV Export ────────────────────────────────────────────────────────────
    fields = ["session_id", "status", "metric_reward", "raw_trace", "horizon_bucket", "ts_start", "ts_end", "task", "index_value"]
    with open(output_csv, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in bucketed_records:
            rec["index_value"] = rec.get("index", {}).get("int_value", "")
            writer.writerow(rec)
    print(f"Saved detailed results CSV to {output_csv}")

    # ── Plot Generation ───────────────────────────────────────────────────────
    plot_results(counts, output_plot)

if __name__ == "__main__":
    main()
