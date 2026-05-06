from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

from .schemas import ICUStay, SofaRow, SuspicionOfInfectionRow


RESP_SUPPORT_RANK = {
    "None": 0,
    "SupplementalOxygen": 0,
    "HFNC": 1,
    "NonInvasiveVent": 1,
    "InvasiveVent": 2,
    "Tracheostomy": 2,
}

RESP_SUPPORT_LABELS = {
    0: "room_air_or_low_support",
    1: "high_flow_or_noninvasive_support",
    2: "invasive_vent_required",
}


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _aki_state_label_from_stage(stage: int | None) -> str | None:
    if stage is None:
        return None
    return {
        0: "no_aki",
        1: "aki_stage_1",
        2: "aki_stage_2",
        3: "aki_stage_3",
    }.get(int(stage), "aki_stage_3")


class ToolRuntime(Protocol):
    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ...


class ConceptToolRuntime:
    def __init__(self, concept_tables: dict[str, list[Any]]):
        self.stays: dict[int, ICUStay] = {stay.stay_id: stay for stay in concept_tables["icustays"]}
        self.suspicion_rows: dict[int, list[SuspicionOfInfectionRow]] = defaultdict(list)
        self.sofa_rows: dict[int, list[SofaRow]] = defaultdict(list)
        for row in concept_tables.get("suspicion_of_infection", []):
            self.suspicion_rows[row.stay_id].append(row)
        for row in concept_tables.get("sofa", []):
            self.sofa_rows[row.stay_id].append(row)
        for stay_rows in self.suspicion_rows.values():
            stay_rows.sort(key=lambda row: row.suspected_infection_time)
        for stay_rows in self.sofa_rows.values():
            stay_rows.sort(key=lambda row: row.hr)

    def _visible_until(self, stay_id: int, t_hour: int):
        stay = self.stays[stay_id]
        return stay.icu_intime + timedelta(hours=t_hour)

    def query_suspicion_of_infection(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        visible_until = self._visible_until(stay_id, t_hour)
        visible_rows = [
            row
            for row in self.suspicion_rows.get(stay_id, [])
            if row.suspected_infection_time <= visible_until
        ]
        if not visible_rows:
            return {
                "stay_id": stay_id,
                "t_hour": t_hour,
                "has_suspected_infection": False,
                "first_visible_suspected_infection_hour": None,
                "first_visible_suspected_infection_time": None,
                "evidence": [],
            }

        first = visible_rows[0]
        anchor = self.stays[stay_id].icu_intime
        first_hour = round((first.suspected_infection_time - anchor).total_seconds() / 3600.0, 2)
        evidence = []
        for row in visible_rows:
            evidence.append(
                {
                    "antibiotic": row.antibiotic,
                    "antibiotic_time": row.antibiotic_time.isoformat() if row.antibiotic_time else None,
                    "culture_time": row.culture_time.isoformat() if row.culture_time else None,
                    "specimen": row.specimen,
                    "positive_culture": row.positive_culture,
                }
            )
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "has_suspected_infection": True,
            "first_visible_suspected_infection_hour": first_hour,
            "first_visible_suspected_infection_time": first.suspected_infection_time.isoformat(),
            "evidence": evidence,
        }

    def query_sofa(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        visible_rows = [row for row in self.sofa_rows.get(stay_id, []) if row.hr <= t_hour]
        if not visible_rows:
            return {
                "stay_id": stay_id,
                "t_hour": t_hour,
                "latest_visible_hr": None,
                "latest_sofa_24hours": None,
                "max_sofa_24hours_so_far": None,
                "latest_components": {},
            }

        latest = visible_rows[-1]
        max_sofa = max(row.sofa_24hours for row in visible_rows)
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "latest_visible_hr": latest.hr,
            "latest_sofa_24hours": latest.sofa_24hours,
            "max_sofa_24hours_so_far": max_sofa,
            "latest_components": latest.components(),
        }

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "query_suspicion_of_infection":
            return self.query_suspicion_of_infection(**arguments)
        if tool_name == "query_sofa":
            return self.query_sofa(**arguments)
        if tool_name == "query_kdigo_stage":
            return {
                "stay_id": arguments["stay_id"],
                "t_hour": arguments["t_hour"],
                "latest_charttime": None,
                "latest_aki_stage": None,
                "latest_aki_stage_smoothed": None,
                "current_aki_state_label": None,
                "current_aki_state_stage": None,
                "current_aki_state_source": "latest_aki_stage_smoothed",
                "max_aki_stage_so_far": None,
                "max_aki_stage_smoothed_so_far": None,
                "has_stage1_or_higher": False,
                "has_stage2_or_higher": False,
                "has_stage3_or_crrt": False,
            }
        if tool_name == "query_ventilation_status":
            return {
                "stay_id": arguments["stay_id"],
                "t_hour": arguments["t_hour"],
                "current_status_raw": None,
                "current_support_level": "room_air_or_low_support",
                "highest_support_level_so_far": "room_air_or_low_support",
                "first_medium_support_time": None,
                "first_invasive_support_time": None,
                "has_medium_support": False,
                "has_invasive_support": False,
            }
        if tool_name == "query_urine_output_rate":
            return {
                "stay_id": arguments["stay_id"],
                "t_hour": arguments["t_hour"],
                "latest_charttime": None,
                "weight_kg": None,
                "min_6hr_rate_mL_kg_hr": None,
                "min_12hr_rate_mL_kg_hr": None,
                "min_24hr_rate_mL_kg_hr": None,
                "has_oliguria": False,
                "has_severe_oliguria": False,
            }
        if tool_name == "query_vasoactive_agent":
            return {
                "stay_id": arguments["stay_id"],
                "t_hour": arguments["t_hour"],
                "received_vasoactive": False,
                "agents_received": [],
                "active_agents": [],
                "first_vasoactive_time": None,
                "latest_vasoactive_endtime": None,
            }
        if tool_name == "query_vitalsign":
            return {
                "stay_id": arguments["stay_id"],
                "t_hour": arguments["t_hour"],
                "latest_charttime": None,
                "latest_heart_rate": None,
                "latest_mbp": None,
                "latest_resp_rate": None,
                "latest_temperature": None,
                "latest_spo2": None,
                "has_tachycardia": False,
                "has_hypotension": False,
            }
        if tool_name == "query_bg":
            return {
                "stay_id": arguments["stay_id"],
                "t_hour": arguments["t_hour"],
                "latest_charttime": None,
                "peak_lactate": None,
                "min_pH": None,
                "max_pH": None,
                "has_elevated_lactate": False,
                "has_acidosis": False,
                "has_severe_acidosis": False,
                "worst_pao2fio2ratio": None,
            }
        if tool_name == "query_gcs":
            return {
                "stay_id": arguments["stay_id"],
                "t_hour": arguments["t_hour"],
                "latest_charttime": None,
                "min_gcs": None,
                "min_gcs_all": None,
                "max_gcs": None,
                "has_severe_impairment": False,
                "has_verbal_unresponsive": False,
            }
        if tool_name == "query_antibiotic":
            return {
                "stay_id": arguments["stay_id"],
                "t_hour": arguments["t_hour"],
                "received_antibiotics": False,
                "distinct_antibiotic_count": 0,
                "distinct_antibiotics": [],
                "first_antibiotic_time": None,
            }
        if tool_name == "query_invasive_line":
            return {
                "stay_id": arguments["stay_id"],
                "t_hour": arguments["t_hour"],
                "has_invasive_line": False,
                "lines_present": [],
                "first_line_time": None,
            }
        raise ValueError(f"Unknown tool: {tool_name}")


class DuckDBConceptToolRuntime:
    def __init__(self, db_path: str):
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB runtime requires the 'duckdb' package to be installed.") from exc
        self.db_path = db_path
        self.connection = duckdb.connect(db_path, read_only=True)
        self._validate_required_tables()

    def _validate_required_tables(self) -> None:
        required = {
            "mimiciv_icu.icustays",
            "mimiciv_derived.suspicion_of_infection",
            "mimiciv_derived.sofa",
            "mimiciv_derived.kdigo_stages",
            "mimiciv_derived.ventilation",
            "mimiciv_derived.urine_output_rate",
            "mimiciv_derived.vasoactive_agent",
            "mimiciv_derived.vitalsign",
            "mimiciv_derived.bg",
            "mimiciv_derived.gcs",
            "mimiciv_derived.antibiotic",
            "mimiciv_derived.invasive_line",
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
                "Missing required DuckDB relations for the MVP tool layer: " + ", ".join(missing)
            )

    def _intime(self, stay_id: int):
        row = self.connection.execute(
            """
            SELECT intime
            FROM mimiciv_icu.icustays
            WHERE stay_id = ?
            """,
            [stay_id],
        ).fetchone()
        if row is None:
            raise ValueError(f"stay_id {stay_id} not found in mimiciv_icu.icustays")
        return row[0]

    def _visible_until(self, stay_id: int, t_hour: int):
        return self._intime(stay_id) + timedelta(hours=t_hour)

    def query_suspicion_of_infection(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        visible_until = self._visible_until(stay_id, t_hour)
        rows = self.connection.execute(
            """
            SELECT
                antibiotic,
                antibiotic_time,
                suspected_infection_time,
                culture_time,
                specimen,
                positive_culture
            FROM mimiciv_derived.suspicion_of_infection
            WHERE stay_id = ?
              AND suspected_infection = 1
              AND suspected_infection_time IS NOT NULL
              AND suspected_infection_time <= ?
            ORDER BY suspected_infection_time, antibiotic_time
            """,
            [stay_id, visible_until],
        ).fetchall()
        if not rows:
            return {
                "stay_id": stay_id,
                "t_hour": t_hour,
                "has_suspected_infection": False,
                "first_visible_suspected_infection_hour": None,
                "first_visible_suspected_infection_time": None,
                "evidence": [],
            }

        first = rows[0]
        intime = self._intime(stay_id)
        first_hour = round((first[2] - intime).total_seconds() / 3600.0, 2)
        evidence = [
            {
                "antibiotic": antibiotic,
                "antibiotic_time": antibiotic_time.isoformat() if antibiotic_time else None,
                "culture_time": culture_time.isoformat() if culture_time else None,
                "specimen": specimen,
                "positive_culture": positive_culture,
            }
            for antibiotic, antibiotic_time, _, culture_time, specimen, positive_culture in rows
        ]
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "has_suspected_infection": True,
            "first_visible_suspected_infection_hour": first_hour,
            "first_visible_suspected_infection_time": first[2].isoformat(),
            "evidence": evidence,
        }

    def query_sofa(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        latest = self.connection.execute(
            """
            SELECT
                hr,
                sofa_24hours,
                respiration_24hours,
                coagulation_24hours,
                liver_24hours,
                cardiovascular_24hours,
                cns_24hours,
                renal_24hours
            FROM mimiciv_derived.sofa
            WHERE stay_id = ?
              AND hr <= ?
            ORDER BY hr DESC
            LIMIT 1
            """,
            [stay_id, t_hour],
        ).fetchone()
        if latest is None:
            return {
                "stay_id": stay_id,
                "t_hour": t_hour,
                "latest_visible_hr": None,
                "latest_sofa_24hours": None,
                "max_sofa_24hours_so_far": None,
                "latest_components": {},
            }

        max_sofa = self.connection.execute(
            """
            SELECT MAX(sofa_24hours)
            FROM mimiciv_derived.sofa
            WHERE stay_id = ?
              AND hr <= ?
            """,
            [stay_id, t_hour],
        ).fetchone()[0]
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "latest_visible_hr": int(latest[0]),
            "latest_sofa_24hours": latest[1],
            "max_sofa_24hours_so_far": max_sofa,
            "latest_components": {
                "respiration_24hours": latest[2],
                "coagulation_24hours": latest[3],
                "liver_24hours": latest[4],
                "cardiovascular_24hours": latest[5],
                "cns_24hours": latest[6],
                "renal_24hours": latest[7],
            },
        }

    def query_kdigo_stage(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        visible_until = self._visible_until(stay_id, t_hour)
        latest = self.connection.execute(
            """
            SELECT
                charttime,
                aki_stage,
                aki_stage_smoothed,
                aki_stage_creat,
                aki_stage_uo,
                aki_stage_crrt
            FROM mimiciv_derived.kdigo_stages
            WHERE stay_id = ?
              AND charttime <= ?
            ORDER BY charttime DESC
            LIMIT 1
            """,
            [stay_id, visible_until],
        ).fetchone()
        if latest is None:
            return {
                "stay_id": stay_id,
                "t_hour": t_hour,
                "latest_charttime": None,
                "latest_aki_stage": None,
                "latest_aki_stage_smoothed": None,
                "current_aki_state_label": None,
                "current_aki_state_stage": None,
                "current_aki_state_source": "latest_aki_stage_smoothed",
                "max_aki_stage_so_far": None,
                "max_aki_stage_smoothed_so_far": None,
                "has_stage1_or_higher": False,
                "has_stage2_or_higher": False,
                "has_stage3_or_crrt": False,
            }

        max_stage, max_stage_smoothed = self.connection.execute(
            """
            SELECT
                MAX(aki_stage),
                MAX(aki_stage_smoothed)
            FROM mimiciv_derived.kdigo_stages
            WHERE stay_id = ?
              AND charttime <= ?
            """,
            [stay_id, visible_until],
        ).fetchone()
        latest_smoothed = latest[2] or 0
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "latest_charttime": latest[0].isoformat() if latest[0] else None,
            "latest_aki_stage": latest[1],
            "latest_aki_stage_smoothed": latest[2],
            "current_aki_state_label": _aki_state_label_from_stage(latest[2]),
            "current_aki_state_stage": latest[2],
            "current_aki_state_source": "latest_aki_stage_smoothed",
            "latest_components": {
                "aki_stage_creat": latest[3],
                "aki_stage_uo": latest[4],
                "aki_stage_crrt": latest[5],
            },
            "max_aki_stage_so_far": max_stage,
            "max_aki_stage_smoothed_so_far": max_stage_smoothed,
            "has_stage1_or_higher": bool(latest_smoothed >= 1),
            "has_stage2_or_higher": bool(latest_smoothed >= 2),
            "has_stage3_or_crrt": bool((max_stage_smoothed or 0) >= 3 or (latest[5] or 0) >= 3),
        }

    def query_ventilation_status(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        visible_until = self._visible_until(stay_id, t_hour)
        rows = self.connection.execute(
            """
            SELECT starttime, endtime, ventilation_status
            FROM mimiciv_derived.ventilation
            WHERE stay_id = ?
              AND starttime <= ?
            ORDER BY starttime
            """,
            [stay_id, visible_until],
        ).fetchall()
        current_rank = 0
        current_status = "SupplementalOxygen"
        highest_rank = 0
        first_medium_time = None
        first_invasive_time = None

        for starttime, endtime, status in rows:
            rank = RESP_SUPPORT_RANK.get(status, 0)
            highest_rank = max(highest_rank, rank)
            if rank >= 1 and first_medium_time is None:
                first_medium_time = starttime
            if rank >= 2 and first_invasive_time is None:
                first_invasive_time = starttime
            if starttime <= visible_until and (endtime is None or endtime >= visible_until):
                current_rank = rank
                current_status = status

        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "current_status_raw": current_status,
            "current_support_level": RESP_SUPPORT_LABELS[current_rank],
            "highest_support_level_so_far": RESP_SUPPORT_LABELS[highest_rank],
            "first_medium_support_time": _iso_or_none(first_medium_time),
            "first_invasive_support_time": _iso_or_none(first_invasive_time),
            "has_medium_support": bool(first_medium_time is not None),
            "has_invasive_support": bool(first_invasive_time is not None),
        }

    def query_urine_output_rate(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        visible_until = self._visible_until(stay_id, t_hour)
        latest = self.connection.execute(
            """
            SELECT
                charttime,
                weight,
                uo_mlkghr_6hr,
                uo_mlkghr_12hr,
                uo_mlkghr_24hr
            FROM mimiciv_derived.urine_output_rate
            WHERE stay_id = ?
              AND charttime <= ?
            ORDER BY charttime DESC
            LIMIT 1
            """,
            [stay_id, visible_until],
        ).fetchone()
        if latest is None:
            return {
                "stay_id": stay_id,
                "t_hour": t_hour,
                "latest_charttime": None,
                "weight_kg": None,
                "min_6hr_rate_mL_kg_hr": None,
                "min_12hr_rate_mL_kg_hr": None,
                "min_24hr_rate_mL_kg_hr": None,
                "has_oliguria": False,
                "has_severe_oliguria": False,
            }
        mins = self.connection.execute(
            """
            SELECT
                MIN(uo_mlkghr_6hr),
                MIN(uo_mlkghr_12hr),
                MIN(uo_mlkghr_24hr)
            FROM mimiciv_derived.urine_output_rate
            WHERE stay_id = ?
              AND charttime <= ?
            """,
            [stay_id, visible_until],
        ).fetchone()
        min_6hr, min_12hr, min_24hr = mins
        severe_values = [value for value in (min_6hr, min_12hr, min_24hr) if value is not None]
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "latest_charttime": _iso_or_none(latest[0]),
            "weight_kg": latest[1],
            "min_6hr_rate_mL_kg_hr": min_6hr,
            "min_12hr_rate_mL_kg_hr": min_12hr,
            "min_24hr_rate_mL_kg_hr": min_24hr,
            "has_oliguria": bool(any(value is not None and value < 0.5 for value in severe_values)),
            "has_severe_oliguria": bool(any(value is not None and value < 0.3 for value in severe_values)),
        }

    def query_vasoactive_agent(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        visible_until = self._visible_until(stay_id, t_hour)
        rows = self.connection.execute(
            """
            SELECT starttime, endtime, dopamine, epinephrine, norepinephrine, phenylephrine, vasopressin, dobutamine, milrinone
            FROM mimiciv_derived.vasoactive_agent
            WHERE stay_id = ?
              AND starttime <= ?
            ORDER BY starttime
            """,
            [stay_id, visible_until],
        ).fetchall()
        agent_names = [
            "dopamine",
            "epinephrine",
            "norepinephrine",
            "phenylephrine",
            "vasopressin",
            "dobutamine",
            "milrinone",
        ]
        received_agents: set[str] = set()
        active_agents: set[str] = set()
        first_time = None
        latest_endtime = None
        for row in rows:
            starttime, endtime, *rates = row
            present = [agent_names[idx] for idx, rate in enumerate(rates) if rate is not None and rate > 0]
            if not present:
                continue
            received_agents.update(present)
            if first_time is None:
                first_time = starttime
            if latest_endtime is None or (endtime is not None and endtime > latest_endtime):
                latest_endtime = endtime
            if starttime <= visible_until and (endtime is None or endtime >= visible_until):
                active_agents.update(present)
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "received_vasoactive": bool(received_agents),
            "agents_received": sorted(received_agents),
            "active_agents": sorted(active_agents),
            "first_vasoactive_time": _iso_or_none(first_time),
            "latest_vasoactive_endtime": _iso_or_none(latest_endtime),
        }

    def query_vitalsign(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        visible_until = self._visible_until(stay_id, t_hour)
        latest = self.connection.execute(
            """
            SELECT charttime, heart_rate, mbp, resp_rate, temperature, spo2
            FROM mimiciv_derived.vitalsign
            WHERE stay_id = ?
              AND charttime <= ?
            ORDER BY charttime DESC
            LIMIT 1
            """,
            [stay_id, visible_until],
        ).fetchone()
        if latest is None:
            return {
                "stay_id": stay_id,
                "t_hour": t_hour,
                "latest_charttime": None,
                "latest_heart_rate": None,
                "latest_mbp": None,
                "latest_resp_rate": None,
                "latest_temperature": None,
                "latest_spo2": None,
                "has_tachycardia": False,
                "has_hypotension": False,
            }
        has_tachycardia, has_hypotension = self.connection.execute(
            """
            SELECT
                MAX(CASE WHEN heart_rate > 100 THEN 1 ELSE 0 END),
                MAX(CASE WHEN mbp < 65 THEN 1 ELSE 0 END)
            FROM mimiciv_derived.vitalsign
            WHERE stay_id = ?
              AND charttime <= ?
            """,
            [stay_id, visible_until],
        ).fetchone()
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "latest_charttime": _iso_or_none(latest[0]),
            "latest_heart_rate": latest[1],
            "latest_mbp": latest[2],
            "latest_resp_rate": latest[3],
            "latest_temperature": latest[4],
            "latest_spo2": latest[5],
            "has_tachycardia": bool(has_tachycardia),
            "has_hypotension": bool(has_hypotension),
        }

    def query_bg(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        visible_until = self._visible_until(stay_id, t_hour)
        latest = self.connection.execute(
            """
            SELECT charttime, lactate, ph, pao2fio2ratio
            FROM mimiciv_derived.bg
            WHERE subject_id = (SELECT subject_id FROM mimiciv_icu.icustays WHERE stay_id = ?)
              AND hadm_id = (SELECT hadm_id FROM mimiciv_icu.icustays WHERE stay_id = ?)
              AND charttime <= ?
            ORDER BY charttime DESC
            LIMIT 1
            """,
            [stay_id, stay_id, visible_until],
        ).fetchone()
        if latest is None:
            return {
                "stay_id": stay_id,
                "t_hour": t_hour,
                "latest_charttime": None,
                "peak_lactate": None,
                "min_pH": None,
                "max_pH": None,
                "has_elevated_lactate": False,
                "has_acidosis": False,
                "has_severe_acidosis": False,
                "worst_pao2fio2ratio": None,
            }
        peak_lactate, min_ph, max_ph, worst_pf = self.connection.execute(
            """
            SELECT MAX(lactate), MIN(ph), MAX(ph), MIN(pao2fio2ratio)
            FROM mimiciv_derived.bg
            WHERE subject_id = (SELECT subject_id FROM mimiciv_icu.icustays WHERE stay_id = ?)
              AND hadm_id = (SELECT hadm_id FROM mimiciv_icu.icustays WHERE stay_id = ?)
              AND charttime <= ?
            """,
            [stay_id, stay_id, visible_until],
        ).fetchone()
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "latest_charttime": _iso_or_none(latest[0]),
            "peak_lactate": peak_lactate,
            "min_pH": min_ph,
            "max_pH": max_ph,
            "has_elevated_lactate": bool(peak_lactate is not None and peak_lactate >= 2),
            "has_acidosis": bool(min_ph is not None and min_ph < 7.35),
            "has_severe_acidosis": bool(min_ph is not None and min_ph <= 7.20),
            "worst_pao2fio2ratio": worst_pf,
        }

    def query_gcs(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        visible_until = self._visible_until(stay_id, t_hour)
        latest = self.connection.execute(
            """
            SELECT charttime, gcs, gcs_motor, gcs_verbal, gcs_eyes
            FROM mimiciv_derived.gcs
            WHERE stay_id = ?
              AND charttime <= ?
            ORDER BY charttime DESC
            LIMIT 1
            """,
            [stay_id, visible_until],
        ).fetchone()
        if latest is None:
            return {
                "stay_id": stay_id,
                "t_hour": t_hour,
                "latest_charttime": None,
                "min_gcs": None,
                "min_gcs_all": None,
                "max_gcs": None,
                "has_severe_impairment": False,
                "has_verbal_unresponsive": False,
            }
        min_gcs, max_gcs, has_verbal = self.connection.execute(
            """
            SELECT MIN(gcs), MAX(gcs), MAX(CASE WHEN gcs_verbal = 1 THEN 1 ELSE 0 END)
            FROM mimiciv_derived.gcs
            WHERE stay_id = ?
              AND charttime <= ?
            """,
            [stay_id, visible_until],
        ).fetchone()
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "latest_charttime": _iso_or_none(latest[0]),
            "min_gcs": min_gcs,
            "min_gcs_all": min_gcs,
            "max_gcs": max_gcs,
            "latest_components": {
                "gcs_motor": latest[2],
                "gcs_verbal": latest[3],
                "gcs_eyes": latest[4],
            },
            "has_severe_impairment": bool(min_gcs is not None and min_gcs <= 8),
            "has_verbal_unresponsive": bool(has_verbal),
        }

    def query_antibiotic(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        visible_until = self._visible_until(stay_id, t_hour)
        rows = self.connection.execute(
            """
            SELECT antibiotic, route, starttime, stoptime
            FROM mimiciv_derived.antibiotic
            WHERE stay_id = ?
              AND starttime <= ?
            ORDER BY starttime
            """,
            [stay_id, visible_until],
        ).fetchall()
        antibiotics = []
        first_time = None
        for antibiotic, route, starttime, stoptime in rows:
            antibiotics.append({"antibiotic": antibiotic, "route": route, "starttime": _iso_or_none(starttime), "stoptime": _iso_or_none(stoptime)})
            if first_time is None:
                first_time = starttime
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "received_antibiotics": bool(rows),
            "distinct_antibiotic_count": len({row[0] for row in rows if row[0]}),
            "distinct_antibiotics": sorted({row[0] for row in rows if row[0]}),
            "first_antibiotic_time": _iso_or_none(first_time),
            "antibiotics": antibiotics[:5],
        }

    def query_invasive_line(self, stay_id: int, t_hour: int) -> dict[str, Any]:
        visible_until = self._visible_until(stay_id, t_hour)
        rows = self.connection.execute(
            """
            SELECT line_type, line_site, starttime, endtime
            FROM mimiciv_derived.invasive_line
            WHERE stay_id = ?
              AND starttime <= ?
            ORDER BY starttime
            """,
            [stay_id, visible_until],
        ).fetchall()
        first_time = rows[0][2] if rows else None
        return {
            "stay_id": stay_id,
            "t_hour": t_hour,
            "has_invasive_line": bool(rows),
            "lines_present": sorted({row[0] for row in rows if row[0]}),
            "first_line_time": _iso_or_none(first_time),
            "line_details": [
                {
                    "line_type": line_type,
                    "line_site": line_site,
                    "starttime": _iso_or_none(starttime),
                    "endtime": _iso_or_none(endtime),
                }
                for line_type, line_site, starttime, endtime in rows[:5]
            ],
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


def build_tool_runtime(
    *,
    tool_backend: str,
    db_path: str | None = None,
    concepts: str | Path | None = None,
    autoformalized_library: str | Path | None = None,
    guidelines_dir: str | Path | None = None,
    functions_dir: str | Path | None = None,
    zeroshot_session_profile: str = "raw",
) -> ToolRuntime:
    if tool_backend == "official":
        if db_path:
            return DuckDBConceptToolRuntime(db_path)
        if concepts is None:
            raise SystemExit("Official backend requires either --db-path or --concepts")
        from .dataset import load_concept_tables

        return ConceptToolRuntime(load_concept_tables(concepts))

    if tool_backend == "autoformalized":
        if not db_path:
            raise SystemExit("Autoformalized backend requires --db-path")
        library_root = Path(autoformalized_library or "autoformalized_library")
        if not library_root.is_absolute():
            library_root = Path.cwd() / library_root
        from .autoformalized import AutoformalizedDuckDBToolRuntime

        return AutoformalizedDuckDBToolRuntime(db_path=db_path, library_path=library_root)

    if tool_backend == "zeroshot_python":
        if not db_path:
            raise SystemExit("Zero-shot python backend requires --db-path")
        from .zeroshot_python import ZeroShotPythonDuckDBRuntime

        return ZeroShotPythonDuckDBRuntime(
            db_path=db_path,
            guidelines_dir=guidelines_dir,
            functions_dir=functions_dir,
            session_profile=zeroshot_session_profile,
        )

    if tool_backend in {"zeroshot_sql", "zeroshot_raw"}:
        if not db_path:
            raise SystemExit("Zero-shot SQL backend requires --db-path")
        from .zeroshot_raw import ZeroShotRawDuckDBRuntime

        return ZeroShotRawDuckDBRuntime(db_path=db_path)

    raise SystemExit(f"Unsupported tool backend: {tool_backend}")
