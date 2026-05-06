from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta
import io
from pathlib import Path
import re
import traceback
from typing import Any
from uuid import uuid4

from .schemas import CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME


SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "next": next,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
}

READ_ONLY_SQL_PREFIXES = ("SELECT", "WITH", "DESCRIBE", "SHOW", "PRAGMA")
FORBIDDEN_SQL_PATTERNS = (
    (re.compile(r"\bsource\s*\.", re.IGNORECASE), "Direct access to source.* is not allowed. Query the checkpoint-scoped mimiciv_* views instead."),
)


@dataclass(slots=True)
class _StayContext:
    stay_id: int
    subject_id: int
    hadm_id: int
    intime: Any
    visible_until: Any


@dataclass(slots=True)
class _PythonSession:
    context: _StayContext
    conn: Any
    namespace: dict[str, Any]


class ZeroShotRawDuckDBRuntime:
    def __init__(self, db_path: str | Path) -> None:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("Zero-shot raw runtime requires the 'duckdb' package.") from exc
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Zero-shot raw runtime requires the 'numpy' package.") from exc
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Zero-shot raw runtime requires the 'pandas' package.") from exc

        self.duckdb = duckdb
        self.np = np
        self.pd = pd
        self.db_path = str(db_path)
        self.connection = duckdb.connect(self.db_path, read_only=True)
        self._validate_required_tables()
        self._sessions: dict[str, _PythonSession] = {}

    def _validate_required_tables(self) -> None:
        required = {
            "mimiciv_icu.icustays",
            "mimiciv_icu.chartevents",
            "mimiciv_icu.inputevents",
            "mimiciv_icu.outputevents",
            "mimiciv_icu.d_items",
            "mimiciv_hosp.admissions",
            "mimiciv_hosp.labevents",
            "mimiciv_hosp.d_labitems",
            "mimiciv_hosp.microbiologyevents",
            "mimiciv_hosp.prescriptions",
        }
        rows = self.connection.execute(
            """
            SELECT table_schema || '.' || table_name AS full_name
            FROM information_schema.tables
            """
        ).fetchall()
        available = {row[0] for row in rows}
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(
                "Missing required DuckDB relations for zero-shot raw tools: " + ", ".join(missing)
            )

    def _sql_string(self, value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def _sql_timestamp(self, value: Any) -> str:
        return f"TIMESTAMP {self._sql_string(value.strftime('%Y-%m-%d %H:%M:%S'))}"

    def _sql_date(self, value: Any) -> str:
        return f"DATE {self._sql_string(value.strftime('%Y-%m-%d'))}"

    def _stay_context(self, stay_id: int, t_hour: int) -> _StayContext:
        row = self.connection.execute(
            """
            SELECT stay_id, subject_id, hadm_id, intime
            FROM mimiciv_icu.icustays
            WHERE stay_id = ?
            """,
            [stay_id],
        ).fetchone()
        if row is None:
            raise ValueError(f"stay_id {stay_id} not found in mimiciv_icu.icustays")
        visible_until = row[3] + timedelta(hours=t_hour)
        return _StayContext(
            stay_id=int(row[0]),
            subject_id=int(row[1]),
            hadm_id=int(row[2]),
            intime=row[3],
            visible_until=visible_until,
        )

    def _checkpoint_connection(self, context: _StayContext):
        conn = self.duckdb.connect()
        conn.execute(f"ATTACH {self._sql_string(self.db_path)} AS source (READ_ONLY)")
        for schema in ("mimiciv_icu", "mimiciv_hosp"):
            conn.execute(f"CREATE SCHEMA {schema}")

        visible_until = self._sql_timestamp(context.visible_until)
        visible_date = self._sql_date(context.visible_until)

        conn.execute(
            f"""
            CREATE VIEW mimiciv_icu.icustays AS
            SELECT
                stay_id,
                subject_id,
                hadm_id,
                intime,
                CASE WHEN outtime <= {visible_until} THEN outtime ELSE {visible_until} END AS outtime,
                first_careunit,
                last_careunit,
                los
            FROM source.mimiciv_icu.icustays
            WHERE stay_id = {context.stay_id}
            """
        )
        conn.execute("CREATE VIEW mimiciv_icu.d_items AS SELECT * FROM source.mimiciv_icu.d_items")
        conn.execute(
            f"""
            CREATE VIEW mimiciv_icu.chartevents AS
            SELECT *
            FROM source.mimiciv_icu.chartevents
            WHERE stay_id = {context.stay_id}
              AND charttime <= {visible_until}
            """
        )
        conn.execute(
            f"""
            CREATE VIEW mimiciv_icu.inputevents AS
            SELECT *
            FROM source.mimiciv_icu.inputevents
            WHERE stay_id = {context.stay_id}
              AND starttime <= {visible_until}
            """
        )
        conn.execute(
            f"""
            CREATE VIEW mimiciv_icu.outputevents AS
            SELECT *
            FROM source.mimiciv_icu.outputevents
            WHERE stay_id = {context.stay_id}
              AND charttime <= {visible_until}
            """
        )
        conn.execute(
            f"""
            CREATE VIEW mimiciv_hosp.admissions AS
            SELECT *
            FROM source.mimiciv_hosp.admissions
            WHERE hadm_id = {context.hadm_id}
            """
        )
        conn.execute("CREATE VIEW mimiciv_hosp.d_labitems AS SELECT * FROM source.mimiciv_hosp.d_labitems")
        conn.execute(
            f"""
            CREATE VIEW mimiciv_hosp.labevents AS
            SELECT *
            FROM source.mimiciv_hosp.labevents
            WHERE hadm_id = {context.hadm_id}
              AND charttime <= {visible_until}
            """
        )
        conn.execute(
            f"""
            CREATE VIEW mimiciv_hosp.microbiologyevents AS
            SELECT *
            FROM source.mimiciv_hosp.microbiologyevents
            WHERE hadm_id = {context.hadm_id}
              AND COALESCE(charttime, CAST(chartdate AS TIMESTAMP)) <= {visible_until}
            """
        )
        conn.execute(
            f"""
            CREATE VIEW mimiciv_hosp.prescriptions AS
            SELECT *
            FROM source.mimiciv_hosp.prescriptions
            WHERE hadm_id = {context.hadm_id}
              AND starttime <= {visible_until}
            """
        )
        return conn

    def start_step_session(self, *, stay_id: int, t_hour: int) -> str:
        context = self._stay_context(stay_id, t_hour)
        conn = self._checkpoint_connection(context)
        session_id = str(uuid4())
        self._sessions[session_id] = _PythonSession(
            context=context,
            conn=conn,
            namespace=self._build_namespace(conn, context),
        )
        return session_id

    def close_step_session(self, session_id: str | None) -> None:
        if session_id is None:
            return
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.conn.close()

    def _build_namespace(self, conn: Any, context: _StayContext) -> dict[str, Any]:
        def query_db(sql: str, params: Any = None):
            self._validate_user_sql(sql, tool_name="query_db")
            if params is None:
                return conn.execute(sql).fetchdf()
            return conn.execute(sql, params).fetchdf()

        return {
            "__builtins__": SAFE_BUILTINS,
            "pd": self.pd,
            "np": self.np,
            "datetime": datetime,
            "timedelta": timedelta,
            "query_db": query_db,
            "stay_id": context.stay_id,
            "subject_id": context.subject_id,
            "hadm_id": context.hadm_id,
            "icu_intime": context.intime,
            "visible_until": context.visible_until,
            "t_hour": int((context.visible_until - context.intime).total_seconds() // 3600),
        }

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in {CODE_EXEC_TOOL_NAME, SQL_EXEC_TOOL_NAME}:
            raise ValueError(f"Unknown tool: {tool_name}")
        session_id = arguments.get("session_id")
        if not session_id or session_id not in self._sessions:
            raise ValueError(f"{tool_name} requires a valid session_id.")
        session = self._sessions[session_id]
        if tool_name == CODE_EXEC_TOOL_NAME:
            code = arguments.get("code")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("run_python requires non-empty Python code.")
            return self._execute_code(session, code)
        sql = arguments.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("run_sql requires non-empty SQL.")
        return self._execute_sql(session, sql)

    def _validate_user_sql(self, sql: str, *, tool_name: str) -> None:
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError(f"{tool_name} requires a non-empty SQL string.")
        first_token = sql.lstrip().split(None, 1)[0].upper()
        if first_token not in READ_ONLY_SQL_PREFIXES:
            raise ValueError(
                f"{tool_name} is read-only. Use SELECT/WITH/DESCRIBE/SHOW/PRAGMA statements only."
            )
        for pattern, message in FORBIDDEN_SQL_PATTERNS:
            if pattern.search(sql):
                raise ValueError(message)

    def _execute_code(self, session: _PythonSession, code: str) -> dict[str, Any]:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        session.namespace.pop("RESULT", None)
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, session.namespace)
        except Exception as exc:
            return {
                "backend": "zeroshot_raw",
                "ok": False,
                "stdout": self._truncate(stdout_buffer.getvalue()),
                "stderr": self._truncate(stderr_buffer.getvalue()),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": self._truncate(traceback.format_exc(), limit=5000),
            }

        result = session.namespace.get("RESULT")
        return {
            "backend": "zeroshot_raw",
            "ok": True,
            "stdout": self._truncate(stdout_buffer.getvalue()),
            "stderr": self._truncate(stderr_buffer.getvalue()),
            "result": self._summarize_result(result),
        }

    def _execute_sql(self, session: _PythonSession, sql: str) -> dict[str, Any]:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                self._validate_user_sql(sql, tool_name="run_sql")
                result = session.conn.execute(sql).fetchdf()
        except Exception as exc:
            return {
                "backend": "zeroshot_raw",
                "ok": False,
                "stdout": self._truncate(stdout_buffer.getvalue()),
                "stderr": self._truncate(stderr_buffer.getvalue()),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": self._truncate(traceback.format_exc(), limit=5000),
            }
        return {
            "backend": "zeroshot_raw",
            "ok": True,
            "stdout": self._truncate(stdout_buffer.getvalue()),
            "stderr": self._truncate(stderr_buffer.getvalue()),
            "result": self._summarize_result(result),
        }

    def _truncate(self, text: str | None, *, limit: int = 3000) -> str:
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[: limit - 15] + "\n...[truncated]"

    def _summarize_result(self, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, self.pd.DataFrame):
            preview = value.head(5).copy()
            preview = preview.where(self.pd.notnull(preview), None)
            return {
                "kind": "dataframe",
                "rows": int(len(value)),
                "columns": [str(col) for col in value.columns.tolist()],
                "head": self._json_safe(preview.to_dict(orient="records")),
            }
        if isinstance(value, self.pd.Series):
            preview = value.head(10).copy()
            preview = preview.where(self.pd.notnull(preview), None)
            return {
                "kind": "series",
                "length": int(len(value)),
                "head": self._json_safe(preview.to_dict()),
            }
        if isinstance(value, (list, tuple)):
            preview = list(value[:10])
            return {"kind": type(value).__name__, "length": len(value), "preview": self._json_safe(preview)}
        if isinstance(value, dict):
            preview = {str(key): value[key] for key in list(value)[:10]}
            return {"kind": "dict", "preview": self._json_safe(preview)}
        return {
            "kind": type(value).__name__,
            "repr": self._truncate(repr(value), limit=1500),
        }

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if hasattr(value, "item"):
            try:
                return self._json_safe(value.item())
            except Exception:
                pass
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        if self.pd.isna(value):
            return None
        return self._truncate(repr(value), limit=500)
