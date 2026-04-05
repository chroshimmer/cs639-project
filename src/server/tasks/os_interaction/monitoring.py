from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional


_PATH_PATTERN = re.compile(r"(/[\w.\-~/]+)")

_INSPECTION_HINTS = (
    "cat",
    "grep",
    "awk",
    "sed",
    "find",
    "ls",
    "head",
    "tail",
    "wc",
    "stat",
    "pwd",
    "readlink",
    "sort",
    "uniq",
    "cut",
    "tr",
    "ps",
    "env",
    "which",
    "whereis",
    "file",
    "printf",
)

_MUTATION_HINTS = (
    "chmod",
    "chown",
    "touch",
    "mkdir",
    "rm",
    "mv",
    "cp",
    "ln",
    "tee",
    "sed -i",
    "python",
    "python3",
    "perl",
    "patch",
    "git",
    "tar",
    "unzip",
    "service",
    "systemctl",
    "kill",
    "pkill",
)

_VERIFICATION_HINTS = (
    "ls",
    "stat",
    "test",
    "grep",
    "cat",
    "head",
    "tail",
    "wc",
    "find",
    "readlink",
    "ps",
    "systemctl",
    "service",
    "file",
    "ss",
    "netstat",
)

_COMMAND_STOPWORDS = {
    "sudo",
    "bash",
    "sh",
    "cat",
    "grep",
    "awk",
    "sed",
    "find",
    "ls",
    "head",
    "tail",
    "wc",
    "stat",
    "pwd",
    "readlink",
    "sort",
    "uniq",
    "cut",
    "tr",
    "ps",
    "env",
    "which",
    "whereis",
    "file",
    "printf",
    "chmod",
    "chown",
    "touch",
    "mkdir",
    "rm",
    "mv",
    "cp",
    "ln",
    "tee",
    "python",
    "python3",
    "perl",
    "patch",
    "git",
    "tar",
    "unzip",
    "service",
    "systemctl",
    "kill",
    "pkill",
    "echo",
    "true",
    "false",
}

_ERROR_RULES = (
    ("permission", ("permission denied", "operation not permitted", "read-only file system")),
    ("missing_file", ("no such file", "cannot access", "not found")),
    ("bad_command", ("command not found", "not recognized")),
    ("service_state", ("not running", "failed to start", "inactive (dead)")),
    ("parse_error", ("syntax error", "unexpected eof", "invalid option", "usage:")),
    ("timeout", ("timed out", "timeout")),
)

_REASON_TEXT = {
    "loop": "you are repeating the same command and getting nearly the same feedback",
    "stall": "recent steps are suspicious and are not adding new evidence",
    "complex_empty": "a complex inspection command returned empty output, so the evidence is weak",
    "finish_on_match": "this task expects an exact answer, so finishing without answering is unsafe",
    "no_shell_evidence": "you have not collected any shell evidence yet",
    "latest_step_failed": "the latest shell step still shows an unresolved error",
    "latest_step_empty": "the latest read-only step returned empty output",
    "missing_verification": "a state-changing step happened, but the final state was not verified",
    "finish_without_state_change": "you are trying to finish a state-change task without a clear state change",
    "no_grounding_for_answer": "the answer is not grounded in a clear observation yet",
}


def normalize_command(command: str) -> str:
    return " ".join((command or "").strip().split())


def output_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left or "", right or "").ratio()


def extract_error_tags(text: str) -> List[str]:
    lower = (text or "").lower()
    tags: List[str] = []
    for tag, patterns in _ERROR_RULES:
        if any(pattern in lower for pattern in patterns):
            tags.append(tag)
    return tags


def _extract_paths(*texts: str) -> List[str]:
    paths: List[str] = []
    for text in texts:
        for match in _PATH_PATTERN.findall(text or ""):
            if match not in paths:
                paths.append(match)
    return paths


def _first_nonempty_lines(text: str, limit: int = 2) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return "empty output"
    return " | ".join(lines[:limit])[:220]


def _command_complexity(command: str) -> int:
    return len(re.findall(r"\|\||&&|[|;\n]", command or ""))


def _has_shell_hint(normalized_command: str, hint: str) -> bool:
    pattern = rf"(^|[;\s|&]){re.escape(hint)}($|[;\s|&])"
    return re.search(pattern, normalized_command) is not None


def _classify_command(command: str) -> str:
    lowered = normalize_command(command).lower()
    has_mutation = any(_has_shell_hint(lowered, hint) for hint in _MUTATION_HINTS)
    has_mutation = has_mutation or bool(re.search(r"(^|[^0-9])>>?\s*[^&\s]", command or ""))
    has_inspection = any(_has_shell_hint(lowered, hint) for hint in _INSPECTION_HINTS)

    if has_mutation and has_inspection:
        return "mixed"
    if has_mutation:
        return "mutation"
    if has_inspection:
        return "inspection"
    return "unknown"


def _command_keywords(command: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z0-9_./:-]+", (command or "").lower()))
    return {
        token
        for token in tokens
        if len(token) >= 3 and not token.startswith("-") and token not in _COMMAND_STOPWORDS
    }


def _get_latest_mutation_step(state: "TrajectoryState") -> Optional["StepRecord"]:
    for record in reversed(state.recent_steps[:-1]):
        if record.command_type in {"mutation", "mixed"}:
            return record
    return None


def _looks_like_verification_command(command: str) -> bool:
    lowered = normalize_command(command).lower()
    return any(_has_shell_hint(lowered, hint) for hint in _VERIFICATION_HINTS)


def _is_related_to_latest_mutation(command: str, state: "TrajectoryState") -> bool:
    mutation_step = _get_latest_mutation_step(state)
    if mutation_step is None:
        return False

    current_paths = set(_extract_paths(command))
    mutation_paths = set(_extract_paths(mutation_step.command, mutation_step.output))
    if current_paths and mutation_paths and current_paths.intersection(mutation_paths):
        return True

    current_keywords = _command_keywords(command)
    mutation_keywords = _command_keywords(mutation_step.command + "\n" + mutation_step.output)
    return bool(current_keywords and mutation_keywords and current_keywords.intersection(mutation_keywords))


@dataclass
class StepRecord:
    round_id: int
    command: str
    normalized_command: str
    output: str
    output_excerpt: str
    command_type: str
    error_tags: List[str] = field(default_factory=list)
    informative: bool = False
    suspicious: bool = False
    output_was_truncated: bool = False


@dataclass
class TrajectoryState:
    goal: str
    evaluation_type: str
    recent_steps: List[StepRecord] = field(default_factory=list)
    known_paths: List[str] = field(default_factory=list)
    blocked_commands: List[str] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)
    intervention_codes: List[str] = field(default_factory=list)
    last_mutation_round: Optional[int] = None
    last_verification_round: Optional[int] = None
    last_intervention_round: Optional[int] = None

    def add_step(self, step: StepRecord) -> None:
        self.recent_steps.append(step)
        self.recent_steps = self.recent_steps[-6:]

    def add_paths(self, paths: List[str]) -> None:
        for path in paths:
            if path not in self.known_paths:
                self.known_paths.append(path)
        self.known_paths = self.known_paths[-6:]

    def remember_blocked_command(self, command: str) -> None:
        normalized = normalize_command(command)
        if normalized and normalized not in self.blocked_commands:
            self.blocked_commands.append(normalized)
        self.blocked_commands = self.blocked_commands[-4:]

    def has_productive_observation(self) -> bool:
        return any(step.informative and not step.error_tags for step in self.recent_steps)

    def has_verification_after_mutation(self) -> bool:
        return (
            self.last_mutation_round is not None
            and self.last_verification_round is not None
            and self.last_verification_round > self.last_mutation_round
        )


@dataclass
class MonitorIntervention:
    reason_codes: List[str]
    summary: str
    user_message: str
    tool_message: Optional[str] = None


class OSReplanner:
    def build_memory_card(self, state: TrajectoryState) -> Optional[str]:
        if not state.recent_steps:
            return None

        lines = ["[STATE MEMORY]", f"Goal: {state.goal}"]
        if state.evaluation_type == "match":
            lines.append("Need: return an exact answer after collecting direct evidence.")
        else:
            lines.append("Need: change the OS state if required, then verify it before finishing.")

        lines.append("Recent commands:")
        for idx, step in enumerate(state.recent_steps[-3:], start=1):
            lines.append(f"{idx}. {step.normalized_command[:140]}")

        lines.append("Key observations:")
        for idx, step in enumerate(state.recent_steps[-3:], start=1):
            observation = step.output_excerpt
            if step.error_tags:
                observation = f"{observation} [tags: {', '.join(step.error_tags)}]"
            lines.append(f"{idx}. {observation}")

        if state.known_paths:
            lines.append(f"Known paths: {', '.join(state.known_paths[-4:])}")
        if state.blocked_commands:
            lines.append("Do not repeat:")
            for command in state.blocked_commands[-2:]:
                lines.append(f"- {command[:140]}")
        if state.last_mutation_round is not None:
            verified = "yes" if state.has_verification_after_mutation() else "no"
            lines.append(f"Verified after latest state change: {verified}")

        return "\n".join(lines)

    def build_recovery_prompt(
        self,
        state: TrajectoryState,
        reason_codes: List[str],
        latest_step: StepRecord,
    ) -> str:
        reasons = [f"- {_REASON_TEXT[code]}" for code in reason_codes if code in _REASON_TEXT]
        if not reasons:
            reasons = ["- recent evidence is weak, so you should re-check your plan"]

        lines = [
            "[Monitor] Potential long-horizon stall detected.",
            "Why this was flagged:",
            *reasons[:3],
            f"Latest observation: {latest_step.output_excerpt}",
            "Replan before the next tool call.",
            "1. Do not repeat a blocked command unless new evidence appears.",
            "2. Pick one next action type: diagnose the state, change the state, or verify the state.",
            "3. Prefer a simpler command that gives direct evidence.",
        ]
        return "\n".join(lines)

    def build_commit_block_prompt(
        self,
        state: TrajectoryState,
        action_name: str,
        reason_codes: List[str],
    ) -> str:
        action_label = "finish" if action_name == "finish_action" else "answer"
        reasons = [f"- {_REASON_TEXT[code]}" for code in reason_codes if code in _REASON_TEXT]
        lines = [
            f"[Monitor] Proposed {action_label} was blocked.",
            "Why this was flagged:",
            *reasons[:3],
            "Before the next completion attempt:",
            "1. Gather one more direct observation or verification step.",
            "2. If the last step failed, diagnose or correct that failure first.",
            "3. Only answer/finish after the evidence matches the task goal.",
        ]
        return "\n".join(lines)


class OSMonitor:
    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.max_interventions = int(config.get("max_interventions", 3))
        self.cooldown_rounds = int(config.get("cooldown_rounds", 1))
        self.loop_similarity_threshold = float(config.get("loop_similarity_threshold", 0.88))
        self.replanner = OSReplanner()

    def create_state(self, goal: str, evaluation_type: str) -> TrajectoryState:
        return TrajectoryState(goal=goal, evaluation_type=evaluation_type)

    def build_memory_card(self, state: TrajectoryState) -> Optional[str]:
        if not self.enabled:
            return None
        return self.replanner.build_memory_card(state)

    def record_bash(
        self,
        state: TrajectoryState,
        round_id: int,
        command: str,
        output: str,
        *,
        output_was_truncated: bool = False,
    ) -> tuple[StepRecord, Optional[MonitorIntervention]]:
        normalized_command = normalize_command(command)
        command_type = _classify_command(command)
        error_tags = extract_error_tags(output)
        informative = bool((output or "").strip())
        suspicious = bool(error_tags) or (command_type != "mutation" and not informative)

        step = StepRecord(
            round_id=round_id,
            command=command,
            normalized_command=normalized_command,
            output=output,
            output_excerpt=_first_nonempty_lines(output),
            command_type=command_type,
            error_tags=error_tags,
            informative=informative,
            suspicious=suspicious,
            output_was_truncated=output_was_truncated,
        )

        state.add_step(step)
        state.add_paths(_extract_paths(command, output))

        if command_type in {"mutation", "mixed"}:
            state.last_mutation_round = round_id
            state.last_verification_round = None
        elif command_type in {"inspection", "unknown"} and informative:
            if state.last_mutation_round is not None:
                if (
                        _looks_like_verification_command(command)
                        and _is_related_to_latest_mutation(command, state)
                ):
                    state.last_verification_round = round_id
            elif _looks_like_verification_command(command):
                state.last_verification_round = round_id

        intervention = None
        if self.enabled:
            reason_codes = self._detect_step_level_issues(state, step)
            if reason_codes:
                intervention = self._make_bash_intervention(state, step, reason_codes)

        return step, intervention

    def check_commit(
        self,
        state: TrajectoryState,
        action_name: str,
        proposed_answer: Optional[str] = None,
    ) -> Optional[MonitorIntervention]:
        if not self.enabled:
            return None

        reason_codes: List[str] = []

        if action_name == "finish_action" and state.evaluation_type == "match":
            reason_codes.append("finish_on_match")

        if not state.recent_steps:
            reason_codes.append("no_shell_evidence")
        else:
            latest_step = state.recent_steps[-1]
            if latest_step.error_tags:
                reason_codes.append("latest_step_failed")
            if latest_step.command_type != "mutation" and not latest_step.informative:
                reason_codes.append("latest_step_empty")

        if action_name == "finish_action" and state.evaluation_type == "check":
            if state.last_mutation_round is None:
                reason_codes.append("finish_without_state_change")
            elif not state.has_verification_after_mutation():
                reason_codes.append("missing_verification")

        if action_name == "answer_action" and not state.has_productive_observation():
            answer_text = (proposed_answer or "").strip()
            if answer_text != "0":
                reason_codes.append("no_grounding_for_answer")

        reason_codes = self._dedupe(reason_codes)
        if not reason_codes or not self._can_intervene(state, state.recent_steps[-1].round_id if state.recent_steps else 0):
            return None

        state.intervention_codes.extend(reason_codes)
        state.interventions.append(", ".join(reason_codes))
        state.last_intervention_round = state.recent_steps[-1].round_id if state.recent_steps else 0

        summary = "; ".join(_REASON_TEXT[code] for code in reason_codes if code in _REASON_TEXT)
        tool_message = f"[Monitor] {action_name} blocked: {summary}."
        user_message = self.replanner.build_commit_block_prompt(state, action_name, reason_codes)
        return MonitorIntervention(
            reason_codes=reason_codes,
            summary=summary,
            tool_message=tool_message,
            user_message=user_message,
        )

    def build_summary(self, state: TrajectoryState) -> dict:
        return {
            "enabled": self.enabled,
            "steps_seen": len(state.recent_steps),
            "interventions": len(state.interventions),
            "blocked_commands": state.blocked_commands,
            "known_paths": state.known_paths,
            "verification_after_mutation": state.has_verification_after_mutation(),
            "intervention_codes": state.intervention_codes,
        }

    def _detect_step_level_issues(
            self,
            state: TrajectoryState,
            step: StepRecord,
    ) -> List[str]:
        reason_codes: List[str] = []

        previous_step = state.recent_steps[-2] if len(state.recent_steps) >= 2 else None

        similar_repeat_found = False
        for record in state.recent_steps[-4:-1]:
            if (
                    record.normalized_command == step.normalized_command
                    and output_similarity(step.output, record.output) >= self.loop_similarity_threshold
            ):
                similar_repeat_found = True
                break

        if (
                previous_step
                and step.normalized_command == previous_step.normalized_command
                and output_similarity(step.output, previous_step.output) >= self.loop_similarity_threshold
        ):
            reason_codes.append("loop")
            state.remember_blocked_command(step.command)
        elif similar_repeat_found:
            reason_codes.append("loop")
            state.remember_blocked_command(step.command)

        if (
                step.command_type in {"inspection", "mixed", "unknown"}
                and not step.informative
                and _command_complexity(step.command) >= 2
        ):
            reason_codes.append("complex_empty")

        recent_window = state.recent_steps[-4:]
        suspicious_count = sum(1 for record in recent_window if record.suspicious)
        if len(recent_window) >= 3 and suspicious_count >= 3:
            reason_codes.append("stall")
        elif len(state.recent_steps) >= 2 and all(record.suspicious for record in state.recent_steps[-2:]):
            reason_codes.append("stall")

        return self._dedupe(reason_codes)

    def _make_bash_intervention(
        self,
        state: TrajectoryState,
        step: StepRecord,
        reason_codes: List[str],
    ) -> Optional[MonitorIntervention]:
        if not self._can_intervene(state, step.round_id):
            return None

        state.intervention_codes.extend(reason_codes)
        state.interventions.append(", ".join(reason_codes))
        state.last_intervention_round = step.round_id

        summary = "; ".join(_REASON_TEXT[code] for code in reason_codes if code in _REASON_TEXT)
        user_message = self.replanner.build_recovery_prompt(state, reason_codes, step)
        return MonitorIntervention(reason_codes=reason_codes, summary=summary, user_message=user_message)

    def _can_intervene(self, state: TrajectoryState, round_id: int) -> bool:
        if len(state.interventions) >= self.max_interventions:
            return False
        if state.last_intervention_round is None:
            return True
        return (round_id - state.last_intervention_round) >= self.cooldown_rounds

    @staticmethod
    def _dedupe(items: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped
