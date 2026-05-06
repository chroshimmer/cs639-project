from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


DATASETS = [
    ("sepsis_final", "rolling_sepsis_final"),
    ("aki_final", "rolling_aki_final"),
    ("respiratory_support_final", "rolling_respiratory_support_final"),
    ("multitask_final", "rolling_multitask_final"),
    ("aki_non_monotonic_final", "rolling_aki_non_monotonic_final"),
]

STRATUM_COLUMNS = [
    "sepsis_stratum",
    "aki_stratum",
    "respiratory_stratum",
    "path_family",
]


def _read_sql(path: Path) -> str:
    return path.read_text().strip().rstrip(";")


def _fetch_pairs(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, object]]:
    rows = con.execute(sql).fetchall()
    cols = [desc[0] for desc in con.description]
    return [dict(zip(cols, row)) for row in rows]


def export_dataset(con: duckdb.DuckDBPyConnection, dataset_dir: Path, csv_stem: str) -> None:
    sql = _read_sql(dataset_dir / "dataset_sql.sql")
    view_name = f"v_{dataset_dir.name}"
    con.execute(f"CREATE OR REPLACE TEMP VIEW {view_name} AS {sql}")

    full_csv = dataset_dir / f"{csv_stem}.csv"
    con.execute(f"COPY {view_name} TO '{full_csv}' (HEADER, DELIMITER ',')")

    for split in ("train", "dev", "test"):
        split_csv = dataset_dir / f"{csv_stem}_{split}.csv"
        con.execute(
            f"COPY (SELECT * FROM {view_name} WHERE split = '{split}') TO '{split_csv}' (HEADER, DELIMITER ',')"
        )

    summary: dict[str, object] = {}
    summary["rows"] = con.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
    summary["trajectories"] = con.execute(f"SELECT COUNT(DISTINCT trajectory_id) FROM {view_name}").fetchone()[0]
    summary["rows_by_split"] = _fetch_pairs(
        con,
        f"""
        SELECT split, COUNT(*) AS rows
        FROM {view_name}
        GROUP BY 1
        ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'dev' THEN 2 ELSE 3 END
        """,
    )
    summary["trajectories_by_split"] = _fetch_pairs(
        con,
        f"""
        SELECT split, COUNT(DISTINCT trajectory_id) AS trajectories
        FROM {view_name}
        GROUP BY 1
        ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'dev' THEN 2 ELSE 3 END
        """,
    )

    columns = {
        row[1]
        for row in con.execute(f"PRAGMA table_info('{view_name}')").fetchall()
    }

    for stratum_col in STRATUM_COLUMNS:
        if stratum_col in columns:
            summary[f"{stratum_col}_counts"] = _fetch_pairs(
                con,
                f"""
                SELECT split, {stratum_col}, COUNT(DISTINCT trajectory_id) AS trajectories
                FROM {view_name}
                GROUP BY 1, 2
                ORDER BY split, {stratum_col}
                """,
            )

    multitask_cols = {"sepsis_positive", "aki_positive", "respiratory_support_positive"}
    if multitask_cols.issubset(columns):
        summary["multitask_cell_counts"] = _fetch_pairs(
            con,
            f"""
            SELECT
              split,
              sepsis_positive,
              aki_positive,
              respiratory_support_positive,
              COUNT(DISTINCT trajectory_id) AS trajectories
            FROM {view_name}
            GROUP BY 1, 2, 3, 4
            ORDER BY split, sepsis_positive DESC, aki_positive DESC, respiratory_support_positive DESC
            """,
        )

    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final benchmark dataset packages from SQL specs.")
    parser.add_argument("--db-path", required=True, help="Path to the MIMIC DuckDB database.")
    parser.add_argument(
        "--base-dir",
        default="rolling_monitor_dataset",
        help="Base directory containing the *_final dataset folders.",
    )
    args = parser.parse_args()

    con = duckdb.connect(args.db_path, read_only=True)
    base_dir = Path(args.base_dir)

    for dataset_name, csv_stem in DATASETS:
        export_dataset(con, base_dir / dataset_name, csv_stem)


if __name__ == "__main__":
    main()
