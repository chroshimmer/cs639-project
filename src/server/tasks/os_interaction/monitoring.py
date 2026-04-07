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
    "du",
    "realpath",
)

_MUTATION_HINTS = (
    "chmod",
    "chown",
    "chgrp",
    "touch",
    "mkdir",
    "rm",
    "mv",
    "cp",
    "ln",
    "tee",
    "sed -i",
    "patch",
    "groupadd",
    "groupmod",
    "useradd",
    "usermod",
    "userdel",
    "setfacl",
)

_STATE_CHANGE_PATTERNS = (
    r"\bchmod\b",
    r"\bchown\b",
    r"\bchgrp\b",
    r"\btouch\b",
    r"\bmkdir\b",
    r"\brm\b",
    r"\bmv\b",
    r"\bcp\b",
    r"\bln\b",
    r"\btee\b",
    r"\bsed\s+-i\b",
    r"\bpatch\b",
    r"\bgroupadd\b",
    r"\bgroupmod\b",
    r"\buseradd\b",
    r"\busermod\b",
    r"\buserdel\b",
    r"\bsetfacl\b",
    r"\binstall\b",
    r"\bapt(?:-get)?\s+install\b",
    r"\b(?:yum|dnf|pip|pip3)\s+install\b",
    r"\bsystemctl\s+(?:start|stop|restart|reload|enable|disable|daemon-reload)\b",
    r"\bservice\s+\S+\s+(?:start|stop|restart|reload)\b",
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

_REASON_TEXT.update({
    "finish_on_answer_task": "this task asks for a grounded answer, so finishing without returning the answer is unsafe",
    "hybrid_answer_pending": "the implementation looks incomplete or the final output is not yet grounded",
})


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
    has_inspection = any(_has_shell_hint(lowered, hint) for hint in _INSPECTION_HINTS)

    has_mutation = any(re.search(pattern, lowered) for pattern in _STATE_CHANGE_PATTERNS)
    has_mutation = has_mutation or _contains_file_write_redirection(command)

    if _looks_like_pure_diagnostic_script(command):
        has_mutation = False

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


def _looks_like_setup_or_bootstrap(command: str) -> bool:
    cmd = normalize_command(command).lower()
    hints = (
        "mkdir",
        "touch",
        "cat >",
        "cat >>",
        "<<",
        "python - <<",
        "python3 - <<",
        "chmod",
        "chown",
        "cp ",
        "mv ",
        "tar ",
        "unzip ",
        "setup",
        "init",
        "bootstrap",
        "mktemp",
    )
    return any(hint in cmd for hint in hints)


def _contains_file_write_redirection(command: str) -> bool:
    cmd = command or ""
    if re.search(r"(^|[;\n])\s*(?:cat|printf|echo|awk|sed|perl|python|python3)\b[^\n]*>(?![&0-9])\s*\S+", cmd):
        return True
    if re.search(r"(^|[;\n])\s*>\s*\S+", cmd):
        return True
    return False


def _is_large_multiline_script(command: str) -> bool:
    cmd = command or ""
    line_count = cmd.count("\n") + 1
    return len(cmd) > 350 or line_count > 10 or _command_complexity(cmd) > 14


def _looks_like_pure_diagnostic_script(command: str) -> bool:
    cmd = normalize_command(command).lower()
    if _contains_file_write_redirection(command):
        return False
    if any(re.search(pattern, cmd) for pattern in _STATE_CHANGE_PATTERNS):
        return False
    diag_hints = (
        "whoami", "id", "groups", "ls", "stat", "find", "grep", "cat", "head", "tail",
        "wc", "pwd", "which", "whereis", "command -v", "getent", "readlink", "realpath",
        "sudo -n", "sudo -l", "systemctl status",
    )
    return _is_large_multiline_script(command) and any(hint in cmd for hint in diag_hints)


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
    def _infer_task_mode_from_goal(self, goal: str) -> str:
        goal_l = (goal or "").lower()

        strong_impl_patterns = (
            "write a bash script",
            "write a shell script",
            "implement a command line tool",
            "implement command line tool",
            "implement ",
            "create a directory",
            "generate ",
            "set all files",
            "set all directories",
            "fix ",
            "make ",
            "permission",
            "read-only",
            "readable",
            "writable",
            "rename ",
            "modify ",
            "change ",
            "delete ",
            "remove ",
            "install ",
            "uninstall ",
            "append ",
            "replace ",
            "copy ",
            "move ",
            "start ",
            "stop ",
            "restart ",
        )
        answer_patterns = (
            "how many",
            "what is",
            "calculate",
            "determine",
            "count",
            "single integer",
            "return an integer",
            "answer as an integer",
            "yes or no",
            "total number",
            "number of lines",
            "number of files",
            "number of directories",
            "total occurrences",
            "sum of",
            "tell me",
            "full path",
            "which file",
            "which directory",
            "what will be the output",
            "output if i execute",
            "max number",
            "find out",
            "locate ",
            "path of",
        )

        strong_impl_hit = any(pattern in goal_l for pattern in strong_impl_patterns)
        answer_hit = any(pattern in goal_l for pattern in answer_patterns)

        if strong_impl_hit and answer_hit:
            return "hybrid"
        if strong_impl_hit:
            return "state"
        if answer_hit:
            return "answer"

        if "script" in goal_l or "tool" in goal_l:
            return "state"
        if "path" in goal_l or "output" in goal_l or "count" in goal_l or "integer" in goal_l:
            return "answer"
        return "state"

    def _infer_phase(self, state: TrajectoryState) -> str:
        if not state.recent_steps:
            return "diagnose"

        task_mode = self._infer_task_mode_from_goal(state.goal)
        latest = state.recent_steps[-1]

        if task_mode == "answer":
            if latest.error_tags:
                return "diagnose"
            if not state.has_productive_observation():
                return "diagnose"
            return "answer"

        if task_mode == "hybrid":
            if state.last_mutation_round is None:
                if latest.error_tags:
                    return "diagnose"
                return "mutate"
            if not state.has_verification_after_mutation():
                return "verify"
            if latest.error_tags:
                return "diagnose"
            if state.has_productive_observation():
                return "answer"
            return "diagnose"

        # state task
        if state.last_mutation_round is None:
            if latest.error_tags:
                return "diagnose"
            if latest.command_type in {"inspection", "unknown"} and not state.has_productive_observation():
                return "diagnose"
            return "mutate"

        if not state.has_verification_after_mutation():
            return "verify"

        if latest.error_tags:
            return "diagnose"

        return "mutate"

    def _target_hints(self, state: TrajectoryState) -> List[str]:
        hints: List[str] = []

        latest_mutation = _get_latest_mutation_step(state)
        if latest_mutation:
            hints.extend(_extract_paths(latest_mutation.command, latest_mutation.output))

        if not hints:
            hints.extend(_extract_paths(state.goal))

        if not hints and state.known_paths:
            hints.extend(state.known_paths[-4:])

        goal_keywords = list(_command_keywords(state.goal))
        for kw in goal_keywords[:4]:
            if kw not in hints:
                hints.append(kw)

        filtered = []
        for item in hints:
            if item and item not in filtered and item.lower() not in {"and", "the", "full", "path", "output"}:
                filtered.append(item)
        return filtered[:5]

    def _completed_evidence(self, state: TrajectoryState) -> List[str]:
        evidence: List[str] = []

        for step in state.recent_steps[-4:]:
            if step.informative:
                prefix = "error seen" if step.error_tags else "observed"
                evidence.append(f"{prefix}: {step.output_excerpt}")

        if state.last_mutation_round is not None:
            evidence.append(f"state-changing step already happened at round {state.last_mutation_round}")

        if state.has_verification_after_mutation():
            evidence.append("latest state change has been verified")

        deduped: List[str] = []
        for item in evidence:
            if item not in deduped:
                deduped.append(item)
        return deduped[:4]

    def _remaining_subgoals(self, state: TrajectoryState, phase: str) -> List[str]:
        goals: List[str] = []
        task_mode = self._infer_task_mode_from_goal(state.goal)

        if task_mode == "answer":
            if not state.has_productive_observation():
                goals.append("collect one direct shell observation that contains the answer")
            goals.append("extract the exact answer string/number/path from evidence")
            goals.append("do not change the environment just to make the answer easier")
            goals.append("if the recent shell output already contains the answer, commit now")
            return goals[:4]

        if task_mode == "hybrid":
            if state.last_mutation_round is None:
                goals.append("perform the required setup or implementation step")
            elif not state.has_verification_after_mutation():
                goals.append("verify that the required artifact/state now exists and behaves correctly")
            if not state.has_productive_observation():
                goals.append("collect one direct shell observation of the resulting output/value/path")
            goals.append("finish only after the implementation is verified and the final output is grounded")
            phase_hint = {
                "diagnose": "next step should narrow uncertainty before changing anything",
                "mutate": "next step should make one minimal required implementation change",
                "verify": "next step should directly test the artifact you just created or changed",
                "answer": "if the recent output already contains the requested result, commit now",
            }
            goals.append(phase_hint.get(phase, "next step should be simple and targeted"))
            return goals[:4]

        # state task
        if not state.has_productive_observation():
            goals.append("diagnose the current state of the target")
        if state.last_mutation_round is None:
            goals.append("apply exactly one state-changing command that moves toward the goal")
        elif not state.has_verification_after_mutation():
            goals.append("verify the final state on the same target after the mutation")
        goals.append("finish only after the verification output matches the task goal")
        phase_hint = {
            "diagnose": "next step should reduce uncertainty, not guess",
            "mutate": "next step should change only one thing, then verify",
            "verify": "next step should directly confirm the changed target",
            "answer": "next step should produce the exact grounded answer",
        }
        goals.append(phase_hint.get(phase, "next step should be simple and targeted"))
        return goals[:4]

    def _pick_failure_type(
        self,
        reason_codes: List[str],
        latest_step: Optional[StepRecord],
        state: Optional[TrajectoryState] = None,
    ) -> str:
        latest_tags = set(latest_step.error_tags if latest_step else [])
        task_mode = self._infer_task_mode_from_goal(state.goal) if state else "state"

        if "missing_verification" in reason_codes:
            return "missing_verification"
        if "loop" in reason_codes or "stall" in reason_codes:
            return "ineffective_search"
        if "missing_file" in latest_tags:
            return "wrong_target"
        if "permission" in latest_tags:
            return "permission_barrier"
        if "bad_command" in latest_tags or "parse_error" in latest_tags:
            return "bad_command"
        if "service_state" in latest_tags:
            return "state_mismatch"
        if "no_grounding_for_answer" in reason_codes or "finish_on_answer_task" in reason_codes:
            return "ungrounded_answer"
        if "latest_step_empty" in reason_codes or "complex_empty" in reason_codes:
            return "weak_signal"
        if task_mode == "answer":
            return "ungrounded_answer"
        if task_mode == "hybrid":
            return "generic_replan"
        return "generic_replan"

    def _typed_recovery_actions(self, failure_type: str, state: TrajectoryState) -> List[str]:
        target_hints = self._target_hints(state)
        target_text = ", ".join(target_hints[:3]) if target_hints else "the intended target"
        task_mode = self._infer_task_mode_from_goal(state.goal)

        if task_mode == "answer":
            templates = {
                "ineffective_search": [
                    "Stop broad exploration and do not repeat a blocked command.",
                    f"Run one short diagnostic command that directly answers the question about {target_text}.",
                    "Do not mutate the environment for a query-style task.",
                ],
                "wrong_target": [
                    "Your current target may be wrong.",
                    f"Re-identify the target first using a direct read-only check around {target_text}.",
                    "Do not mutate anything until the target identity is confirmed.",
                ],
                "ungrounded_answer": [
                    "The answer is not yet grounded in a direct shell observation.",
                    "Collect one exact observation that contains the final answer.",
                    "If the recent shell output already contains the answer, commit now.",
                ],
                "weak_signal": [
                    "The latest evidence is weak or empty.",
                    "Use a simpler read-only command that gives direct evidence on the target.",
                    "Do not change the environment just to make the answer easier.",
                ],
                "generic_replan": [
                    "Re-state the question in one sentence before the next tool call.",
                    "Choose one short read-only command that directly narrows the answer.",
                    "Commit as soon as the answer is grounded.",
                ],
            }
            return templates.get(failure_type, templates["generic_replan"])

        if task_mode == "hybrid":
            templates = {
                "ineffective_search": [
                    "Stop broad exploration and do not repeat a blocked command.",
                    "Decide whether you are still implementing, verifying, or ready to return the result.",
                    f"Choose one short command that advances only the current phase on {target_text}.",
                ],
                "missing_verification": [
                    "A required implementation step likely happened, but it has not been verified yet.",
                    f"Run one direct verification command on {target_text}.",
                    "Once the artifact or output is verified, commit promptly.",
                ],
                "generic_replan": [
                    "Re-state the remaining implementation goal before the next tool call.",
                    "Do exactly one of these next: implement, verify, or extract the final result.",
                    "Avoid exploratory commands that do not advance the current phase.",
                ],
            }
            return templates.get(failure_type, templates["generic_replan"])

        # state task
        templates = {
            "ineffective_search": [
                "Stop broad exploration and do not repeat a blocked command.",
                f"Pick one target and narrow the search around {target_text}.",
                "Run exactly one short diagnostic command before any further mutation.",
            ],
            "wrong_target": [
                "Your current target may be wrong.",
                f"Re-identify the target first using a direct check around {target_text}.",
                "Do not mutate anything until the target identity is confirmed.",
            ],
            "permission_barrier": [
                "The last step failed because of permissions or write constraints.",
                "Diagnose who owns the target and what the current mode/state is.",
                "Choose one corrective command, then verify immediately.",
            ],
            "bad_command": [
                "The previous command form is likely invalid for this environment.",
                "Switch to a simpler standard shell command.",
                "Prefer one direct check or one direct corrective action, not a chained command.",
            ],
            "state_mismatch": [
                "The environment state does not match your assumption.",
                "Refresh the current state with one direct status check.",
                "Only then choose whether to mutate or verify.",
            ],
            "missing_verification": [
                "A state change likely happened, but it has not been verified yet.",
                f"Run one direct verification command on {target_text}.",
                "Do not finish until the verification output matches the goal.",
            ],
            "weak_signal": [
                "The latest evidence is weak or empty.",
                "Use a simpler command that gives direct evidence on the target.",
                "Avoid complex pipelines until the state is clear.",
            ],
            "generic_replan": [
                "Re-state the remaining subgoal before the next tool call.",
                "Choose exactly one next action type: diagnose, mutate, or verify.",
                "Prefer a simple command with direct evidence.",
            ],
        }
        return templates.get(failure_type, templates["generic_replan"])

    def build_memory_card(self, state: TrajectoryState) -> Optional[str]:
        if not state.recent_steps:
            return None

        phase = self._infer_phase(state)
        task_mode = self._infer_task_mode_from_goal(state.goal)
        targets = self._target_hints(state)
        completed = self._completed_evidence(state)
        remaining = self._remaining_subgoals(state, phase)

        lines = [
            "[WORKING PLAN]",
            f"Goal: {state.goal}",
            f"Current phase: {phase}",
        ]

        if task_mode == "answer":
            lines.append("Task type: answer/query task. Ground the answer in shell evidence and avoid changing the environment.")
        elif task_mode == "hybrid":
            lines.append("Task type: implementation-plus-answer task. First make or verify the required artifact/state, then ground the final output/value before finishing.")
        else:
            lines.append("Task type: state-change task. Change state only if needed, then verify before finish.")

        if targets:
            lines.append(f"Target hints: {', '.join(targets)}")

        lines.append("Completed evidence:")
        for item in completed[:4]:
            lines.append(f"- {item}")

        lines.append("Remaining subgoals:")
        for item in remaining[:4]:
            lines.append(f"- {item}")

        lines.append("Recent commands:")
        for idx, step in enumerate(state.recent_steps[-3:], start=1):
            lines.append(f"{idx}. {step.normalized_command[:140]}")

        lines.append("Recent observations:")
        for idx, step in enumerate(state.recent_steps[-3:], start=1):
            obs = step.output_excerpt
            if step.error_tags:
                obs = f"{obs} [tags: {', '.join(step.error_tags)}]"
            lines.append(f"{idx}. {obs}")

        if state.blocked_commands:
            lines.append("Blocked patterns:")
            for command in state.blocked_commands[-2:]:
                lines.append(f"- {command[:140]}")

        if state.last_mutation_round is not None:
            verified = "yes" if state.has_verification_after_mutation() else "no"
            lines.append(f"Verified after latest mutation: {verified}")

        lines.append("Next-step policy:")
        if task_mode == "answer":
            lines.append("- Execute exactly one short read-only diagnostic command next.")
            lines.append("- Prefer ls/stat/find/wc/grep/cat over any state-changing command.")
            lines.append("- If recent shell output already contains the answer, commit now.")
        elif task_mode == "hybrid":
            if phase == "mutate":
                lines.append("- Execute exactly one implementation step next.")
                lines.append("- After implementation, the following step should verify the artifact or output.")
            elif phase == "verify":
                lines.append("- Execute exactly one direct verification or test command next.")
                lines.append("- If the verified output already answers the question, commit now.")
            elif phase == "answer":
                lines.append("- If the recent shell output already contains the requested value/path/output, commit now.")
                lines.append("- Do not perform extra changes once the implementation is verified.")
            else:
                lines.append("- Use one short diagnostic command to decide what is still missing.")
                lines.append("- Avoid broad exploration that does not advance implement/verify/answer.")
        else:
            if phase == "diagnose":
                lines.append("- Execute exactly one diagnostic command next.")
                lines.append("- Prefer ls/stat/find/ps/systemctl status over broad search or repeated mutation.")
            elif phase == "mutate":
                lines.append("- Execute exactly one corrective mutation next.")
                lines.append("- After mutation, the following step should be verification.")
            elif phase == "verify":
                lines.append("- Execute exactly one direct verification command on the changed target.")
                lines.append("- Do not finish before the verification output matches the goal.")
            else:
                lines.append("- Commit only with the exact grounded result.")
                lines.append("- Do not finish without direct evidence in the recent shell output.")

        return "\n".join(lines)

    def build_recovery_prompt(
        self,
        state: TrajectoryState,
        reason_codes: List[str],
        latest_step: StepRecord,
    ) -> str:
        failure_type = self._pick_failure_type(reason_codes, latest_step, state)
        typed_actions = self._typed_recovery_actions(failure_type, state)
        remaining = self._remaining_subgoals(state, self._infer_phase(state))
        task_mode = self._infer_task_mode_from_goal(state.goal)

        reason_lines = [f"- {_REASON_TEXT[code]}" for code in reason_codes if code in _REASON_TEXT]
        if not reason_lines:
            reason_lines = ["- recent evidence is weak, so you should re-check your plan"]

        lines = [
            "[Monitor] Recovery required.",
            f"Detected failure type: {failure_type}",
            "Why this was flagged:",
            *reason_lines[:3],
            f"Latest observation: {latest_step.output_excerpt}",
            "Do this before the next tool call:",
        ]
        for item in typed_actions[:3]:
            lines.append(f"- {item}")

        lines.append("Remaining subgoals right now:")
        for item in remaining[:3]:
            lines.append(f"- {item}")

        if task_mode == "answer":
            lines.append("Constraint: the next tool call should be a single read-only diagnostic step or a grounded answer.")
        elif task_mode == "hybrid":
            lines.append("Constraint: the next tool call should advance exactly one phase: implement, verify, or return the grounded result.")
        else:
            lines.append("Constraint: the next tool call should perform exactly one action type: diagnose, mutate, verify, or answer.")
        return "\n".join(lines)

    def build_commit_block_prompt(
        self,
        state: TrajectoryState,
        action_name: str,
        reason_codes: List[str],
    ) -> str:
        action_label = "finish" if action_name == "finish_action" else "answer"
        latest_step = state.recent_steps[-1] if state.recent_steps else None
        failure_type = self._pick_failure_type(reason_codes, latest_step, state)
        typed_actions = self._typed_recovery_actions(failure_type, state)
        remaining = self._remaining_subgoals(state, self._infer_phase(state))

        reason_lines = [f"- {_REASON_TEXT[code]}" for code in reason_codes if code in _REASON_TEXT]

        lines = [
            f"[Monitor] Proposed {action_label} was blocked.",
            f"Detected failure type: {failure_type}",
            "Why this was flagged:",
            *reason_lines[:3],
            "Before the next completion attempt:",
        ]
        for item in typed_actions[:2]:
            lines.append(f"- {item}")

        lines.append("Still missing:")
        for item in remaining[:3]:
            lines.append(f"- {item}")

        lines.append(f"Do not call {action_label} again until one missing subgoal is resolved by direct shell evidence.")
        return "\n".join(lines)



class OSMonitor:
    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.max_interventions = int(config.get("max_interventions", 2))
        self.cooldown_rounds = int(config.get("cooldown_rounds", 2))
        self.loop_similarity_threshold = float(config.get("loop_similarity_threshold", 0.88))
        self.replanner = OSReplanner()

    def create_state(self, goal: str, evaluation_type: str) -> TrajectoryState:
        return TrajectoryState(goal=goal, evaluation_type=evaluation_type)

    def infer_task_mode(self, goal: str, evaluation_type: str = "check") -> str:
        return self.replanner._infer_task_mode_from_goal(goal)

    def build_initial_hint(self, goal: str, evaluation_type: str = "check") -> Optional[str]:
        if not self.enabled:
            return None
        task_mode = self.infer_task_mode(goal, evaluation_type)
        if task_mode == "answer":
            return (
                "[Task hint]\n"
                "This is mainly an answer/query task. Prefer one short read-only command at a time. "
                "Avoid long multi-line scripts. As soon as shell output contains the exact answer, call answer_action."
            )
        if task_mode == "hybrid":
            return (
                "[Task hint]\n"
                "This task mixes implementation and answering. Make the minimal required change first, then verify it, "
                "then return the final grounded value/path/output. Avoid oversized diagnostic scripts."
            )
        return (
            "[Task hint]\n"
            "This is mainly a state-change task. Prefer minimal changes and short verification commands. "
            "Avoid large diagnostic scripts unless a short check already failed."
        )

    def should_block_complex_bash(self, state: TrajectoryState, command: str) -> Optional[str]:
        if not self.enabled:
            return None
        task_mode = self.replanner._infer_task_mode_from_goal(state.goal)
        phase = self.replanner._infer_phase(state)
        if not _is_large_multiline_script(command):
            return None
        if task_mode == "answer":
            return (
                "[Monitor] This bash script is too large for a query task. "
                "Use one short read-only command next, and avoid multi-line scripts or set -euo pipefail."
            )
        if task_mode == "hybrid" and phase in {"diagnose", "answer"}:
            return (
                "[Monitor] The current step is too large. Narrow the task with one short command first. "
                "Only use a larger implementation command after the target/output is clear."
            )
        if task_mode == "state" and phase == "diagnose":
            return (
                "[Monitor] The diagnostic script is too large. Use one short command to narrow the issue first, "
                "then mutate or verify."
            )
        return None

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
        suppress_empty_signal = (
            not informative
            and not error_tags
            and (command_type in {"mutation", "mixed"} or _looks_like_setup_or_bootstrap(command))
        )
        suspicious = bool(error_tags) or (
            command_type in {"inspection", "unknown"}
            and not informative
            and not suppress_empty_signal
            and not _looks_like_pure_diagnostic_script(command)
        )

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

        mutation_like = command_type in {"mutation", "mixed"} and not _looks_like_pure_diagnostic_script(command)
        if mutation_like:
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
        task_mode = self.replanner._infer_task_mode_from_goal(state.goal)

        if action_name == "finish_action" and task_mode == "answer":
            reason_codes.append("finish_on_answer_task")

        if not state.recent_steps:
            reason_codes.append("no_shell_evidence")
        else:
            latest_step = state.recent_steps[-1]
            if latest_step.error_tags:
                reason_codes.append("latest_step_failed")
            if latest_step.command_type in {"inspection", "unknown"} and not latest_step.informative:
                reason_codes.append("latest_step_empty")

        if task_mode in {"state", "hybrid"} and action_name == "finish_action":
            if state.last_mutation_round is None:
                reason_codes.append("finish_without_state_change")
            elif not state.has_verification_after_mutation():
                reason_codes.append("missing_verification")
            elif task_mode == "hybrid" and not state.has_productive_observation():
                reason_codes.append("hybrid_answer_pending")

        if task_mode == "answer" and not state.has_productive_observation():
            answer_text = (proposed_answer or "").strip()
            if answer_text != "0":
                reason_codes.append("no_grounding_for_answer")

        reason_codes = self._dedupe(reason_codes)
        current_round = state.recent_steps[-1].round_id if state.recent_steps else 0
        if not reason_codes or not self._can_intervene(state, current_round):
            return None

        state.intervention_codes.extend(reason_codes)
        state.interventions.append(", ".join(reason_codes))
        state.last_intervention_round = current_round

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
            step.round_id >= 2
            and step.command_type in {"inspection", "unknown"}
            and not step.informative
            and not step.error_tags
            and not _looks_like_setup_or_bootstrap(step.command)
            and not _looks_like_pure_diagnostic_script(step.command)
            and _command_complexity(step.command) >= 2
        ):
            reason_codes.append("complex_empty")

        recent_window = state.recent_steps[-4:]
        suspicious_count = sum(1 for record in recent_window if record.suspicious)
        if len(recent_window) >= 4 and suspicious_count >= 4:
            reason_codes.append("stall")
        elif (
            len(state.recent_steps) >= 4
            and all(record.suspicious for record in state.recent_steps[-4:])
            and not any(record.informative and not record.error_tags for record in state.recent_steps[-4:])
        ):
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

