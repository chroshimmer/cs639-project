from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import traceback
from typing import Any
from uuid import uuid4

from .duckdb_session import DuckDBCodeSession
from .schemas import CODE_EXEC_TOOL_NAME


ALLOWED_RAW_RELATIONS = {
    ("mimiciv_icu", "icustays"),
    ("mimiciv_icu", "chartevents"),
    ("mimiciv_icu", "inputevents"),
    ("mimiciv_icu", "outputevents"),
    ("mimiciv_icu", "d_items"),
    ("mimiciv_hosp", "admissions"),
    ("mimiciv_hosp", "labevents"),
    ("mimiciv_hosp", "d_labitems"),
    ("mimiciv_hosp", "microbiologyevents"),
    ("mimiciv_hosp", "prescriptions"),
}


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
    session: DuckDBCodeSession


class ZeroShotPythonDuckDBRuntime:
    def __init__(
        self,
        db_path: str | Path,
        *,
        guidelines_dir: str | Path | None = None,
        functions_dir: str | Path | None = None,
        session_profile: str = "raw",
    ) -> None:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("Zero-shot python backend requires the 'duckdb' package.") from exc
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Zero-shot python backend requires the 'pandas' package.") from exc

        self.duckdb = duckdb
        self.pd = pd
        self.db_path = str(db_path)
        self.guidelines_dir = str(guidelines_dir) if guidelines_dir is not None else None
        self.functions_dir = str(functions_dir) if functions_dir is not None else None
        self.session_profile = session_profile
        self.connection = duckdb.connect(self.db_path, read_only=True)
        self._validate_required_tables()
        self._sessions: dict[str, _PythonSession] = {}

    def _validate_required_tables(self) -> None:
        required = {f"{schema}.{table}" for schema, table in ALLOWED_RAW_RELATIONS}
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
                "Missing required DuckDB relations for zero-shot python backend: " + ", ".join(missing)
            )

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

    def _create_session(self, context: _StayContext) -> DuckDBCodeSession:
        allowed_relations = set(ALLOWED_RAW_RELATIONS) if self.session_profile == "raw" else None
        return DuckDBCodeSession(
            db_path=self.db_path,
            subject_id=context.subject_id,
            hadm_id=context.hadm_id,
            stay_id=context.stay_id,
            visible_until=context.visible_until,
            time_bounded_views=True,
            allowed_relations=allowed_relations,
            guidelines_dir=self.guidelines_dir,
            functions_dir=self.functions_dir,
        )

    def start_step_session(self, *, stay_id: int, t_hour: int) -> str:
        context = self._stay_context(stay_id, t_hour)
        session_id = str(uuid4())
        self._sessions[session_id] = _PythonSession(
            context=context,
            session=self._create_session(context),
        )
        return session_id

    def close_step_session(self, session_id: str | None) -> None:
        if session_id is None:
            return
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.session.close()

    def _reset_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.session.close()
        session.session = self._create_session(session.context)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name != CODE_EXEC_TOOL_NAME:
            raise ValueError(f"Unknown tool: {tool_name}")
        session_id = arguments.get("session_id")
        if not session_id or session_id not in self._sessions:
            raise ValueError(f"{tool_name} requires a valid session_id.")
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("run_python requires non-empty Python code.")
        return self._execute_code(session_id, code)

    def _execute_code(self, session_id: str, code: str) -> dict[str, Any]:
        session = self._sessions[session_id]
        runtime_session = session.session
        runtime_session.namespace.pop("RESULT", None)

        try:
            compile(code, "<model>", "exec")
        except Exception as exc:
            return {
                "backend": "zeroshot_python",
                "ok": False,
                "stdout": "",
                "stderr": "",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": self._truncate(traceback.format_exc(), limit=5000),
                "query_stats": runtime_session.get_query_stats(),
            }

        output = runtime_session.execute(code)
        stdout, stderr = self._split_output(output)
        query_stats = runtime_session.get_query_stats()

        if output.startswith("BLOCKED: "):
            return {
                "backend": "zeroshot_python",
                "ok": False,
                "stdout": "",
                "stderr": "",
                "error_type": "UnsafeCodeError",
                "error_message": output[len("BLOCKED: ") :],
                "traceback": "",
                "query_stats": query_stats,
            }

        if output.startswith("TIMEOUT: "):
            self._reset_session(session_id)
            return {
                "backend": "zeroshot_python",
                "ok": False,
                "stdout": "",
                "stderr": "",
                "error_type": "CodeExecutionTimeout",
                "error_message": output[len("TIMEOUT: ") :],
                "traceback": "",
                "query_stats": query_stats,
            }

        if output.startswith("Error:\n"):
            error_type, error_message = self._parse_traceback(output[len("Error:\n") :])
            return {
                "backend": "zeroshot_python",
                "ok": False,
                "stdout": stdout,
                "stderr": stderr,
                "error_type": error_type,
                "error_message": error_message,
                "traceback": self._truncate(output[len("Error:\n") :], limit=5000),
                "query_stats": query_stats,
            }

        result = runtime_session.namespace.get("RESULT")
        return {
            "backend": "zeroshot_python",
            "ok": True,
            "stdout": stdout,
            "stderr": stderr,
            "result": self._summarize_result(result),
            "query_stats": query_stats,
        }

    def _split_output(self, output: str) -> tuple[str, str]:
        marker = "\nStderr:\n"
        if marker in output:
            stdout, stderr = output.split(marker, 1)
            return self._truncate(stdout), self._truncate(stderr)
        if output.startswith("Error:\n") or output.startswith("BLOCKED: ") or output.startswith("TIMEOUT: "):
            return "", ""
        return self._truncate(output), ""

    def _parse_traceback(self, tb: str) -> tuple[str, str]:
        lines = [line.strip() for line in tb.splitlines() if line.strip()]
        for line in reversed(lines):
            if ":" in line:
                error_type, error_message = line.split(":", 1)
                error_type = error_type.strip()
                error_message = error_message.strip()
                if error_type:
                    return error_type, error_message
        return "ExecutionError", self._truncate(tb, limit=1000)

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
