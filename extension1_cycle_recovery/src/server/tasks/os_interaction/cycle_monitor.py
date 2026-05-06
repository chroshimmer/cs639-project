from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


RECOVERY_NOTICE = """[RECOVERY NOTICE]
Your recent actions and observations are highly similar to previous turns, suggesting that the current strategy may not be making progress.

Before your next action:
1. Summarize what you have already tried.
2. Explain why the previous strategy did not help.
3. Choose a different strategy or inspect a different source of information.
4. Do not repeat a command similar to the recent commands unless you have a clear reason.

Continue with exactly one valid action in the required format."""


_SHELL_WRAPPER_RE = re.compile(
    r"^\s*(?:sudo\s+)?(?:/bin/)?(?:bash|sh)\s+-c\s+(['\"])(?P<body>.*)\1\s*$",
    re.IGNORECASE | re.DOTALL,
)
_PATH_RE = re.compile(r"(?:^|[\s:=])(/[A-Za-z0-9._~+\-/]+)")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")


@dataclass
class TurnRecord:
    turn_id: int
    raw_agent_response: str
    action_type: str
    command: Optional[str]
    observation: str
    observation_truncated: bool
    error_type: Optional[str] = None
    action_category: Optional[str] = None
    state_signature: Optional[str] = None


@dataclass
class CycleDecision:
    triggered: bool
    reason: str = ""
    action_similarity: float = 0.0
    observation_similarity: float = 0.0
    cycle_score: float = 0.0
    repeated_turns: int = 0
    matched_turn_ids: List[int] = field(default_factory=list)
    recovery_notice: str = RECOVERY_NOTICE


@dataclass
class CycleInterventionEvent:
    sample_id: str
    turn_id: int
    trigger_reason: str
    action_similarity: float
    observation_similarity: float
    cycle_score: float
    recent_commands: List[str]
    injected_recovery_notice: str
    matched_turn_ids: List[int]
    eventually_succeeds: Optional[bool] = None
    final_status: Optional[str] = None


@dataclass
class CycleMonitorState:
    sample_id: str
    enabled: bool
    turns: List[TurnRecord] = field(default_factory=list)
    interventions: List[CycleInterventionEvent] = field(default_factory=list)
    pending_notice: Optional[str] = None
    last_intervention_turn: Optional[int] = None

    def recent_commands(self, limit: int = 5) -> List[str]:
        commands = [record.command for record in self.turns if record.command]
        return commands[-limit:]


class CycleMonitor:
    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        self.enabled = bool(
            config.get("enable_cycle_recovery", config.get("enabled", False))
        )
        self.window_size = int(config.get("cycle_window_size", config.get("window_size", 5)))
        self.similarity_threshold = float(
            config.get("cycle_similarity_threshold", config.get("similarity_threshold", 0.8))
        )
        self.cycle_score_threshold = float(
            config.get("cycle_score_threshold", config.get("score_threshold", 0.8))
        )
        self.min_repeated_turns = int(
            config.get("cycle_min_repeated_turns", config.get("min_repeated_turns", 2))
        )
        self.cooldown_turns = int(
            config.get("cycle_cooldown_turns", config.get("cooldown_turns", 2))
        )
        self.max_interventions = int(
            config.get("cycle_max_interventions", config.get("max_interventions", 2))
        )
        self.use_observation_similarity = bool(
            config.get("cycle_use_observation_similarity", True)
        )
        self.use_action_similarity = bool(config.get("cycle_use_action_similarity", True))
        self.use_state_signature = bool(config.get("cycle_use_state_signature", True))
        self.observation_char_limit = int(config.get("cycle_observation_char_limit", 2000))
        self.log_path = config.get("cycle_log_path", config.get("log_path", "cycle_recovery_logs.jsonl"))

    def create_state(self, sample_id: Any = "unknown") -> CycleMonitorState:
        return CycleMonitorState(sample_id=str(sample_id), enabled=self.enabled)

    def make_turn_record(
        self,
        *,
        turn_id: int,
        raw_agent_response: Optional[str],
        action_type: str,
        command: Optional[str],
        observation: Optional[str],
        observation_truncated: bool = False,
    ) -> TurnRecord:
        observation_text = observation or ""
        return TurnRecord(
            turn_id=turn_id,
            raw_agent_response=raw_agent_response or "",
            action_type=action_type,
            command=command,
            observation=observation_text,
            observation_truncated=observation_truncated,
            error_type=self._detect_error_type(observation_text),
            action_category=self.classify_action(action_type, command),
            state_signature=self.state_signature(command, observation_text),
        )

    def record_turn(
        self,
        state: CycleMonitorState,
        record: TurnRecord,
    ) -> CycleDecision:
        state.turns.append(record)
        max_history = max(self.window_size * 2, self.window_size + 1)
        state.turns = state.turns[-max_history:]

        if not state.enabled:
            return CycleDecision(triggered=False, reason="disabled")
        decision = self._detect_cycle(state, record)
        if not decision.triggered:
            return decision

        state.last_intervention_turn = record.turn_id
        state.pending_notice = decision.recovery_notice
        state.interventions.append(
            CycleInterventionEvent(
                sample_id=state.sample_id,
                turn_id=record.turn_id,
                trigger_reason=decision.reason,
                action_similarity=decision.action_similarity,
                observation_similarity=decision.observation_similarity,
                cycle_score=decision.cycle_score,
                recent_commands=state.recent_commands(),
                injected_recovery_notice=decision.recovery_notice,
                matched_turn_ids=decision.matched_turn_ids,
            )
        )
        return decision

    def consume_pending_notice(self, state: CycleMonitorState) -> Optional[str]:
        notice = state.pending_notice
        state.pending_notice = None
        return notice

    def note_external_intervention(self, state: CycleMonitorState, turn_id: int) -> None:
        """Yield to a foreign monitor (e.g. OSMonitor) that just intervened on `turn_id`.

        This bumps `last_intervention_turn` so the internal cooldown logic suppresses
        cycle_monitor's own recovery notice for `self.cooldown_turns` turns, avoiding
        the agent receiving two stacked meta-instructions on the same / adjacent turns.
        Any pending notice that was about to be injected is also dropped.
        """
        if not state.enabled:
            return
        if state.last_intervention_turn is None or turn_id > state.last_intervention_turn:
            state.last_intervention_turn = turn_id
        state.pending_notice = None

    def build_summary(self, state: CycleMonitorState) -> Dict[str, Any]:
        return {
            "enabled": state.enabled,
            "turns_seen": len(state.turns),
            "interventions": len(state.interventions),
            "recent_commands": state.recent_commands(),
            "last_intervention_turn": state.last_intervention_turn,
        }

    def finalize(
        self,
        state: CycleMonitorState,
        *,
        final_status: str,
        eventually_succeeds: bool,
    ) -> None:
        if not state.enabled or not state.interventions:
            return
        for event in state.interventions:
            event.final_status = final_status
            event.eventually_succeeds = eventually_succeeds
            self._append_log(asdict(event))

    def _detect_cycle(
        self,
        state: CycleMonitorState,
        current: TurnRecord,
    ) -> CycleDecision:
        if len(state.interventions) >= self.max_interventions:
            return CycleDecision(triggered=False, reason="max_interventions_reached")
        if (
            state.last_intervention_turn is not None
            and current.turn_id - state.last_intervention_turn < self.cooldown_turns
        ):
            return CycleDecision(triggered=False, reason="cooldown")

        window = state.turns[-(self.window_size + 1):-1]
        if len(window) < self.min_repeated_turns:
            return CycleDecision(triggered=False, reason="insufficient_history")

        matched_turn_ids: List[int] = []
        action_sims: List[float] = []
        observation_sims: List[float] = []
        scores: List[float] = []

        for previous in window:
            action_sim = self.command_similarity(current.command, previous.command)
            obs_sim = self.observation_similarity(current.observation, previous.observation)
            no_progress = self.no_progress_score(current, previous)
            score = self.cycle_score(action_sim, obs_sim, no_progress)
            action_gate = (not self.use_action_similarity) or action_sim >= self.similarity_threshold
            obs_gate = (not self.use_observation_similarity) or obs_sim >= self.similarity_threshold
            if score >= self.cycle_score_threshold and (action_gate or obs_gate):
                matched_turn_ids.append(previous.turn_id)
                action_sims.append(action_sim)
                observation_sims.append(obs_sim)
                scores.append(score)

        if len(matched_turn_ids) < self.min_repeated_turns:
            return CycleDecision(triggered=False, reason="below_threshold")

        return CycleDecision(
            triggered=True,
            reason="similar_action_observation_no_progress",
            action_similarity=max(action_sims) if action_sims else 0.0,
            observation_similarity=max(observation_sims) if observation_sims else 0.0,
            cycle_score=max(scores) if scores else 0.0,
            repeated_turns=len(matched_turn_ids),
            matched_turn_ids=matched_turn_ids,
        )

    def cycle_score(self, action_similarity: float, observation_similarity: float, no_progress_score: float) -> float:
        action_component = action_similarity if self.use_action_similarity else 0.0
        obs_component = observation_similarity if self.use_observation_similarity else 0.0
        if self.use_action_similarity and self.use_observation_similarity:
            return 0.4 * action_component + 0.4 * obs_component + 0.2 * no_progress_score
        if self.use_action_similarity:
            return 0.8 * action_component + 0.2 * no_progress_score
        if self.use_observation_similarity:
            return 0.8 * obs_component + 0.2 * no_progress_score
        return no_progress_score

    def no_progress_score(self, current: TurnRecord, previous: TurnRecord) -> float:
        if self._is_modification_then_verification(previous, current):
            return 0.0
        if self.use_state_signature and self._has_new_state_signal(current, previous):
            return 0.0
        action_sim = self.command_similarity(current.command, previous.command)
        obs_sim = self.observation_similarity(current.observation, previous.observation)
        if action_sim >= self.similarity_threshold and obs_sim >= self.similarity_threshold:
            return 1.0
        if obs_sim >= max(self.similarity_threshold, 0.86):
            return 1.0
        return 0.0

    def command_similarity(self, left: Optional[str], right: Optional[str]) -> float:
        left_norm = self.normalize_command(left)
        right_norm = self.normalize_command(right)
        if not left_norm and not right_norm:
            return 1.0
        if not left_norm or not right_norm:
            return 0.0
        return SequenceMatcher(None, left_norm, right_norm).ratio()

    def observation_similarity(self, left: Optional[str], right: Optional[str]) -> float:
        left_norm = self.normalize_observation(left)
        right_norm = self.normalize_observation(right)
        if not left_norm and not right_norm:
            return 1.0
        if not left_norm or not right_norm:
            return 0.0
        return SequenceMatcher(None, left_norm, right_norm).ratio()

    def normalize_command(self, command: Optional[str]) -> str:
        command = command or ""
        command = command.strip()
        wrapper = _SHELL_WRAPPER_RE.match(command)
        if wrapper:
            command = wrapper.group("body")
        command = re.sub(r"\s+", " ", command)
        command = re.sub(r"^(?:sudo\s+)?(?:/bin/)?(?:bash|sh)\s+-lc\s+", "", command, flags=re.IGNORECASE)
        return command.strip().lower()

    def normalize_observation(self, observation: Optional[str]) -> str:
        observation = observation or ""
        observation = re.sub(r"\s+", " ", observation.strip())
        return observation[: self.observation_char_limit]

    def classify_action(self, action_type: str, command: Optional[str]) -> str:
        action_type = (action_type or "").lower()
        command_norm = self.normalize_command(command)
        first = self._first_command(command_norm)
        if action_type in {"commit", "finish", "answer", "finish_action", "answer_action"}:
            return "answer_or_finish"
        if first in {"ls", "find", "pwd", "tree"}:
            return "exploration"
        if first in {"cat", "grep", "head", "tail", "wc", "stat", "file", "du"}:
            return "inspection"
        if first in {"chmod", "chown", "mv", "cp", "rm", "touch", "mkdir"}:
            return "modification"
        if first in {"sed", "awk"} and self._contains_write(command or ""):
            return "modification"
        if self._contains_write(command or "") or first == "tee":
            return "modification"
        return "other"

    def state_signature(self, command: Optional[str], observation: Optional[str]) -> Optional[str]:
        text = f"{command or ''}\n{observation or ''}"
        paths = _PATH_RE.findall(text)
        numbers = _NUMBER_RE.findall(text)
        tokens = [
            token.lower()
            for token in _WORD_RE.findall(text)
            if len(token) >= 4 and not token.startswith("-")
        ]
        signature_parts = sorted(set(paths))[:8] + sorted(set(numbers))[:8] + sorted(set(tokens))[:12]
        return "|".join(signature_parts) if signature_parts else None

    def _has_new_state_signal(self, current: TurnRecord, previous: TurnRecord) -> bool:
        current_sig = set((current.state_signature or "").split("|")) if current.state_signature else set()
        previous_sig = set((previous.state_signature or "").split("|")) if previous.state_signature else set()
        return bool(current_sig - previous_sig)

    def _is_modification_then_verification(self, previous: TurnRecord, current: TurnRecord) -> bool:
        return (
            previous.action_category == "modification"
            and current.action_category in {"exploration", "inspection"}
        )

    def _detect_error_type(self, observation: str) -> Optional[str]:
        lower = (observation or "").lower()
        if "permission denied" in lower or "operation not permitted" in lower:
            return "permission"
        if "no such file" in lower or "cannot access" in lower or "not found" in lower:
            return "missing_file"
        if "command not found" in lower:
            return "bad_command"
        if "syntax error" in lower or "usage:" in lower:
            return "parse_error"
        if "timed out" in lower or "timeout" in lower:
            return "timeout"
        return None

    def _append_log(self, event: Dict[str, Any]) -> None:
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    @staticmethod
    def _first_command(command: str) -> str:
        command = re.split(r"[;&|()\n]", command or "", maxsplit=1)[0].strip()
        if not command:
            return ""
        parts = command.split()
        while parts and "=" in parts[0] and not parts[0].startswith((">", "<")):
            parts.pop(0)
        return parts[0] if parts else ""

    @staticmethod
    def _contains_write(command: str) -> bool:
        return bool(re.search(r"(?:^|[\s;])(?:>|>>)\s*\S+|\btee\b|\bsed\s+-i\b", command or ""))