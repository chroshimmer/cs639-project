from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import inspect
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable


AUTOFORM_TOOL_TO_FUNCTION = MappingProxyType(
    {
        "query_suspicion_of_infection": "suspicion_of_infection",
        "query_sofa": "sofa",
        "query_kdigo_stage": "kdigo_stages",
        "query_ventilation_status": "ventilation",
        "query_urine_output_rate": "urine_output_rate",
        "query_vasoactive_agent": "vasoactive_agent",
        "query_vitalsign": "vitalsign",
        "query_bg": "bg",
        "query_gcs": "gcs",
        "query_antibiotic": "antibiotic",
        "query_invasive_line": "invasive_line",
    }
)


def _aki_state_label_from_stage(stage: int | None) -> str | None:
    if stage is None:
        return None
    return {
        0: "no_aki",
        1: "aki_stage_1",
        2: "aki_stage_2",
        3: "aki_stage_3",
    }.get(int(stage), "aki_stage_3")


@dataclass(slots=True)
class _StayContext:
    stay_id: int
    subject_id: int
    hadm_id: int
    intime: Any
    visible_until: Any


class AutoformalizedDuckDBToolRuntime:
    def __init__(self, db_path: str, library_path: str | Path) -> None:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError(
                "Autoformalized runtime requires the 'duckdb' Python package to be installed."
            ) from exc
        try:
            import pandas  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Autoformalized runtime requires 'pandas' because generated functions return DataFrames."
            ) from exc

        self.duckdb = duckdb
        self.db_path = db_path
        self.library_path = Path(library_path)
        self.function_dir = self.library_path / "functions"
        if not self.function_dir.exists():
            raise RuntimeError(f"Autoformalized function directory not found: {self.function_dir}")

        self.connection = duckdb.connect(db_path, read_only=True)
        self._validate_required_tables()
        self._function_namespaces: dict[str, dict[str, Any]] = {}

    def _validate_required_tables(self) -> None:
        required = {
            "mimiciv_icu.icustays",
            "mimiciv_icu.chartevents",
            "mimiciv_icu.inputevents",
            "mimiciv_icu.outputevents",
            "mimiciv_icu.d_items",
            "mimiciv_hosp.microbiologyevents",
            "mimiciv_hosp.prescriptions",
            "mimiciv_hosp.procedures_icd",
            "mimiciv_hosp.labevents",
            "mimiciv_hosp.d_labitems",
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
                "Missing required DuckDB relations for autoformalized tools: " + ", ".join(missing)
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
        conn.execute(
            f"ATTACH {self._sql_string(self.db_path)} AS source (READ_ONLY)"
        )
        for schema in ("mimiciv_icu", "mimiciv_hosp", "mimiciv_derived"):
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
            WHERE subject_id = {context.subject_id}
              AND charttime <= {visible_until}
            """
        )
        conn.execute(
            f"""
            CREATE VIEW mimiciv_hosp.labevents AS
            SELECT *
            FROM source.mimiciv_hosp.labevents
            WHERE subject_id = {context.subject_id}
              AND charttime <= {visible_until}
            """
        )
        conn.execute("CREATE VIEW mimiciv_hosp.d_labitems AS SELECT * FROM source.mimiciv_hosp.d_labitems")
        conn.execute(
            f"""
            CREATE VIEW mimiciv_hosp.microbiologyevents AS
            SELECT *
            FROM source.mimiciv_hosp.microbiologyevents
            WHERE subject_id = {context.subject_id}
              AND hadm_id = {context.hadm_id}
              AND COALESCE(charttime, CAST(chartdate AS TIMESTAMP)) <= {visible_until}
            """
        )
        conn.execute(
            f"""
            CREATE VIEW mimiciv_hosp.prescriptions AS
            SELECT *
            FROM source.mimiciv_hosp.prescriptions
            WHERE subject_id = {context.subject_id}
              AND hadm_id = {context.hadm_id}
              AND starttime <= {visible_until}
            """
        )
        conn.execute(
            f"""
            CREATE VIEW mimiciv_hosp.procedures_icd AS
            SELECT *
            FROM source.mimiciv_hosp.procedures_icd
            WHERE subject_id = {context.subject_id}
              AND hadm_id = {context.hadm_id}
              AND (chartdate IS NULL OR chartdate <= {visible_date})
            """
        )
        return conn

    def _load_namespace(self, function_key: str) -> dict[str, Any]:
        if function_key in self._function_namespaces:
            return self._function_namespaces[function_key]
        path = self.function_dir / f"{function_key}.py"
        if not path.exists():
            raise RuntimeError(f"Autoformalized function file not found: {path}")
        namespace: dict[str, Any] = {}
        exec(path.read_text(), namespace)
        final_function = namespace.get("FINAL_FUNCTION")
        if not callable(final_function):
            raise RuntimeError(f"FINAL_FUNCTION missing or invalid in {path}")
        self._function_namespaces[function_key] = namespace
        return namespace

    def _run_function(self, tool_name: str, stay_id: int, t_hour: int) -> tuple[dict[str, Any], _StayContext]:
        context = self._stay_context(stay_id, t_hour)
        function_key = AUTOFORM_TOOL_TO_FUNCTION[tool_name]
        namespace = self._load_namespace(function_key)
        conn = self._checkpoint_connection(context)
        try:
            namespace["query_db"] = lambda sql: conn.execute(sql).fetchdf()
            final_function = namespace["FINAL_FUNCTION"]
            kwargs: dict[str, Any] = {}
            for param_name in inspect.signature(final_function).parameters:
                if param_name == "stay_id":
                    kwargs[param_name] = stay_id
                elif param_name == "subject_id":
                    kwargs[param_name] = context.subject_id
                elif param_name == "hadm_id":
                    kwargs[param_name] = context.hadm_id
            result = final_function(**kwargs)
        finally:
            conn.close()
        if not isinstance(result, dict):
            raise RuntimeError(f"Autoformalized function {function_key} returned non-dict output")
        return result, context

    def _iso_or_none(self, value: Any) -> str | None:
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    def _to_datetime_or_none(self, value: Any):
        if value is None:
            return None
        try:
            import pandas as pd
        except ImportError:
            return value
        return pd.to_datetime(value)

    def query_suspicion_of_infection(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        result, context = self._run_function("query_suspicion_of_infection", stay_id, t_hour)
        culture_df = result.get("culture_orders")
        antibiotic_info = result.get("antibiotic_info") or {}
        antibiotic_df = antibiotic_info.get("antibiotic_administrations")

        earliest_time = None
        evidence: list[dict[str, Any]] = []

        if culture_df is not None and hasattr(culture_df, "empty") and not culture_df.empty:
            for _, row in culture_df.head(5).iterrows():
                culture_time = self._to_datetime_or_none(row.get("culture_time"))
                if earliest_time is None or (culture_time is not None and culture_time < earliest_time):
                    earliest_time = culture_time
                evidence.append(
                    {
                        "antibiotic": None,
                        "antibiotic_time": None,
                        "culture_time": self._iso_or_none(culture_time),
                        "specimen": row.get("spec_type_desc"),
                        "positive_culture": None,
                    }
                )

        if antibiotic_df is not None and hasattr(antibiotic_df, "empty") and not antibiotic_df.empty:
            for _, row in antibiotic_df.head(5).iterrows():
                admin_time = self._to_datetime_or_none(row.get("admin_time"))
                if earliest_time is None or (admin_time is not None and admin_time < earliest_time):
                    earliest_time = admin_time
                evidence.append(
                    {
                        "antibiotic": row.get("antibiotic_name"),
                        "antibiotic_time": self._iso_or_none(admin_time),
                        "culture_time": None,
                        "specimen": None,
                        "positive_culture": None,
                    }
                )

        first_hour = None
        if earliest_time is not None:
            first_hour = round((earliest_time - context.intime).total_seconds() / 3600.0, 2)

        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "backend": "autoformalized",
            "has_suspected_infection": bool(result.get("has_suspicion_of_infection")),
            "first_visible_suspected_infection_hour": first_hour,
            "first_visible_suspected_infection_time": self._iso_or_none(earliest_time),
            "evidence": evidence,
        }

    def query_sofa(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        result, _context = self._run_function("query_sofa", stay_id, t_hour)
        total_score = result.get("total_score")
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "backend": "autoformalized",
            "latest_visible_hr": t_hour,
            "latest_sofa_24hours": total_score,
            "max_sofa_24hours_so_far": total_score,
            "latest_components": {
                "respiration_24hours": result.get("respiration_score"),
                "coagulation_24hours": result.get("coagulation_score"),
                "liver_24hours": result.get("liver_score"),
                "cardiovascular_24hours": result.get("cardiovascular_score"),
                "cns_24hours": result.get("brain_score"),
                "renal_24hours": result.get("kidney_score"),
            },
        }

    def query_kdigo_stage(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        result, context = self._run_function("query_kdigo_stage", stay_id, t_hour)
        stage = result.get("kdigo_stage")
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "backend": "autoformalized",
            "latest_charttime": self._iso_or_none(context.visible_until),
            "latest_aki_stage": stage,
            "latest_aki_stage_smoothed": stage,
            "current_aki_state_label": _aki_state_label_from_stage(stage),
            "current_aki_state_stage": stage,
            "current_aki_state_source": "latest_aki_stage_smoothed",
            "latest_components": {
                "aki_stage_creat": result.get("kdigo_stage_creatinine"),
                "aki_stage_uo": result.get("kdigo_stage_uo"),
                "aki_stage_crrt": 3 if result.get("has_stage_3") else 0,
            },
            "max_aki_stage_so_far": stage,
            "max_aki_stage_smoothed_so_far": stage,
            "has_stage1_or_higher": bool(result.get("has_aki")),
            "has_stage2_or_higher": bool(result.get("has_stage_2_or_higher")),
            "has_stage3_or_crrt": bool(result.get("has_stage_3")),
        }

    def query_ventilation_status(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        result, _context = self._run_function("query_ventilation_status", stay_id, t_hour)
        if result.get("invasive_mechanical_ventilation"):
            level = "invasive_vent_required"
        elif result.get("non_invasive_mechanical_ventilation") or result.get("high_flow_nasal_cannula"):
            level = "high_flow_or_noninvasive_support"
        else:
            level = "room_air_or_low_support"
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "backend": "autoformalized",
            "current_status_raw": result.get("ventilation_details", {}).get("oxygen_delivery_devices_used", []),
            "current_support_level": level,
            "highest_support_level_so_far": level,
            "first_medium_support_time": None,
            "first_invasive_support_time": None,
            "has_medium_support": level in {"high_flow_or_noninvasive_support", "invasive_vent_required"},
            "has_invasive_support": level == "invasive_vent_required",
        }

    def query_urine_output_rate(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        result, context = self._run_function("query_urine_output_rate", stay_id, t_hour)
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "backend": "autoformalized",
            "latest_charttime": self._iso_or_none(context.visible_until),
            "weight_kg": result.get("weight_kg"),
            "min_6hr_rate_mL_kg_hr": result.get("min_6hr_rate_mL_kg_hr"),
            "min_12hr_rate_mL_kg_hr": None,
            "min_24hr_rate_mL_kg_hr": None,
            "has_oliguria": bool(result.get("has_oliguria")),
            "has_severe_oliguria": bool(result.get("has_severe_oliguria")),
        }

    def query_vasoactive_agent(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        result, _context = self._run_function("query_vasoactive_agent", stay_id, t_hour)
        details = result.get("agent_details") or {}
        active_agents = []
        for agent_name, detail in details.items():
            last_time = self._to_datetime_or_none(detail.get("last_time"))
            if last_time is None or last_time >= _context.visible_until:
                active_agents.append(agent_name)
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "backend": "autoformalized",
            "received_vasoactive": bool(result.get("received_vasoactive")),
            "agents_received": sorted(result.get("agents_received") or []),
            "active_agents": sorted(active_agents),
            "first_vasoactive_time": self._iso_or_none(result.get("first_time")),
            "latest_vasoactive_endtime": self._iso_or_none(result.get("last_time")),
        }

    def query_vitalsign(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        result, context = self._run_function("query_vitalsign", stay_id, t_hour)
        vital_df = result.get("vital_signs_data")
        latest_row = None
        if vital_df is not None and hasattr(vital_df, "empty") and not vital_df.empty:
            latest_row = vital_df.sort_values("charttime").iloc[-1]
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "backend": "autoformalized",
            "latest_charttime": self._iso_or_none(latest_row["charttime"]) if latest_row is not None else self._iso_or_none(context.visible_until),
            "latest_heart_rate": latest_row["valuenum"] if latest_row is not None and latest_row.get("vital_type") == "heart_rate" else None,
            "latest_mbp": None,
            "latest_resp_rate": None,
            "latest_temperature": None,
            "latest_spo2": None,
            "has_tachycardia": bool(result.get("has_tachycardia")),
            "has_hypotension": bool(result.get("has_hypotension")),
        }

    def query_bg(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        result, context = self._run_function("query_bg", stay_id, t_hour)
        measurements = result.get("all_measurements")
        latest_charttime = None
        worst_pf = None
        if measurements is not None and hasattr(measurements, "empty") and not measurements.empty:
            latest_charttime = measurements["charttime"].max()
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "backend": "autoformalized",
            "latest_charttime": self._iso_or_none(latest_charttime) or self._iso_or_none(context.visible_until),
            "peak_lactate": result.get("peak_lactate"),
            "min_pH": result.get("min_pH"),
            "max_pH": result.get("max_pH"),
            "has_elevated_lactate": bool(result.get("has_elevated_lactate")),
            "has_acidosis": bool(result.get("has_acidosis")),
            "has_severe_acidosis": bool(result.get("has_severe_acidosis")),
            "worst_pao2fio2ratio": worst_pf,
        }

    def query_gcs(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        result, _context = self._run_function("query_gcs", stay_id, t_hour)
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "backend": "autoformalized",
            "latest_charttime": self._iso_or_none(result.get("last_gcs_time")),
            "min_gcs": result.get("min_gcs"),
            "min_gcs_all": result.get("min_gcs_all"),
            "max_gcs": result.get("max_gcs"),
            "has_severe_impairment": bool(result.get("has_severe_impairment")),
            "has_verbal_unresponsive": bool(result.get("has_verbal_unresponsive")),
        }

    def query_antibiotic(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        result, _context = self._run_function("query_antibiotic", stay_id, t_hour)
        antibiotic_df = result.get("antibiotic_administrations")
        first_time = None
        if antibiotic_df is not None and hasattr(antibiotic_df, "empty") and not antibiotic_df.empty:
            first_time = antibiotic_df["admin_time"].min()
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "backend": "autoformalized",
            "received_antibiotics": bool(result.get("received_antibiotics")),
            "distinct_antibiotic_count": int(result.get("distinct_antibiotic_count") or 0),
            "distinct_antibiotics": sorted(result.get("distinct_antibiotics") or []),
            "first_antibiotic_time": self._iso_or_none(first_time),
        }

    def query_invasive_line(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        result, _context = self._run_function("query_invasive_line", stay_id, t_hour)
        first_times = [
            detail.get("first_seen")
            for detail in (result.get("line_details") or {}).values()
            if detail.get("first_seen")
        ]
        first_line_time = min(first_times) if first_times else None
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "backend": "autoformalized",
            "has_invasive_line": bool(result.get("lines_present")),
            "lines_present": sorted(result.get("lines_present") or []),
            "first_line_time": self._iso_or_none(first_line_time),
            "line_details": result.get("line_details") or {},
        }

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "query_suspicion_of_infection":
            return self.query_suspicion_of_infection(**arguments)
        if tool_name == "query_sofa":
            return self.query_sofa(**arguments)
        if tool_name == "query_kdigo_stage":
            return self.query_kdigo_stage(**arguments)
        if tool_name == "query_ventilation_status":
            return self.query_ventilation_status(**arguments)
        if tool_name == "query_urine_output_rate":
            return self.query_urine_output_rate(**arguments)
        if tool_name == "query_vasoactive_agent":
            return self.query_vasoactive_agent(**arguments)
        if tool_name == "query_vitalsign":
            return self.query_vitalsign(**arguments)
        if tool_name == "query_bg":
            return self.query_bg(**arguments)
        if tool_name == "query_gcs":
            return self.query_gcs(**arguments)
        if tool_name == "query_antibiotic":
            return self.query_antibiotic(**arguments)
        if tool_name == "query_invasive_line":
            return self.query_invasive_line(**arguments)
        raise ValueError(f"Unknown tool: {tool_name}")
