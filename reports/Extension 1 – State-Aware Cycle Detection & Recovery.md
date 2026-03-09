# Extension 1 – State-Aware Cycle Detection & Recovery

## 1. Background & Goal

According to the **AgentBench (Liu et al., 2023)** study, LLMs often suffer from "Reasoning Stalls" in long-horizon OS tasks. When a command returns a non-informative output, agents frequently retry the exact same action. Our Extension 1 addresses this by implementing an external **Inference-Time Interceptor** to break these action loops, a method inspired by reactive control loops in autonomous systems.

------

## 2. Logic of Cycle Detection

We track the agent's command history $C = \{c_1, \dots, c_t\}$ to detect loops. A cycle is flagged if the last $k$ commands are identical. We set $k=3$ as the default threshold; this allows the model a chance to retry after minor errors (like typos) while preventing infinite repetitions that would cause a Task Limit Exceeded (TLE) error.

------

## 3. Implementation: Locating and Modifying the Core Engine

Since the `agentrl` library is installed as a third-party package within the Conda environment, the modification requires locating the live source code.

### 3.1 Locating `openai.py`

To find the exact file responsible for API dispatching, we use the following Python command:

Bash

```
python3 -c "import agentrl.eval.client.openai as o; print(o.__file__)"
```

This returns the absolute path to the client implementation, typically located in:

```
/home/your-name/miniconda/envs/cs639-project/lib/python3.11/site-packages/agentrl/eval/client/openai.py
```

### 3.2 Code Modification

The integration involves injecting the interceptor into the `OpenAIClient` class:

1. **Import**: `from .openai_extension import apply_cycle_detection_recovery`
2. **Hook Point**: In `_query_chat_completions`, we insert the call after the message format conversion but before the `client.chat.completions.create` call. This ensures we are intercepting the final payload being sent to the LLM.

------

## 4. Key Enhancements & Methods

Our Extension 1 introduces several robust methods to ensure the intervention is both accurate and effective:

- **Handling command variations**: The `_normalize_bash_command` function ensures that syntactic variations (e.g., `ls -a` vs `ls   -a`) are mapped to the same string representation $c_i$, preventing the model from bypassing detection with trivial spacing changes.
- **Pydantic-dict compatibility**: The `_to_message_dict` helper allows the interceptor to work seamlessly whether the `agentrl` library passes raw Python dictionaries or Pydantic `MessageRecord` models.
- **Parsing recent history**: `_collect_recent_bash_commands` filters the trajectory to only analyze `assistant` tool calls, ignoring system prompts or previous user interventions that might skew the detection.
- **Attention-shift recovery**: By injecting the warning as a `user` message, we leverage the model's training to prioritize recent user instructions, effectively forcing an "attention shift" to resolve the loop.

------

##  Appendix: Full Implementation Code (`openai_extension.py`)

Python

```
# Cycle Detection for Extension 1

from __future__ import annotations

import json
import logging
from typing import Any, List, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CYCLE_THRESHOLD = 3

DEFAULT_RECOVERY_PROMPT = (
    "[SYSTEM WARNING]: Cycle Detected. You have repeatedly issued the exact same command. "
    "Stop and analyze the OS error. Try a completely different approach."
)


def _normalize_bash_command(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    return " ".join(raw.strip().split())


def _get_bash_command_from_tool_call(tool_call: Dict[str, Any]) -> Optional[str]:
    try:
        func = tool_call.get("function") or {}
        if func.get("name") != "bash_action":
            return None
        args_str = func.get("arguments")
        if args_str is None:
            return None
        if isinstance(args_str, dict):
            args = args_str
        else:
            args = json.loads(args_str)
        if not isinstance(args, dict):
            return None
        values = list(args.values())
        if not values:
            return None
        cmd = values[0]
        if isinstance(cmd, str):
            return _normalize_bash_command(cmd)
        return None
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        logger.debug("openai_extension: skip tool_call parse error %s", e)
        return None


def _to_message_dict(msg: Any) -> Optional[Dict[str, Any]]:
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "dict"):
        return msg.dict()
    return None


def _collect_recent_bash_commands(messages: List[Dict[str, Any]], max_lookback: int = 20) -> List[str]:
    commands: List[str] = []
    for raw in messages:
        msg = _to_message_dict(raw) if not isinstance(raw, dict) else raw
        if not msg:
            continue
        role = (msg.get("role") or "").strip().lower()
        if role != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            tc_dict = _to_message_dict(tc) if not isinstance(tc, dict) else tc
            if not tc_dict:
                continue
            cmd = _get_bash_command_from_tool_call(tc_dict)
            if cmd is not None:
                commands.append(cmd)
                break
    return commands[-max_lookback:] if max_lookback else commands


def detect_cycle(
    messages: List[Dict[str, Any]],
    *,
    threshold: int = DEFAULT_CYCLE_THRESHOLD,
    max_lookback: int = 20,
) -> bool:
    recent = _collect_recent_bash_commands(messages, max_lookback=max_lookback)
    if len(recent) < threshold:
        return False
    last_n = recent[-threshold:]
    first = last_n[0]
    return all(c == first for c in last_n)


def apply_cycle_detection_recovery(
    messages: List[Dict[str, Any]],
    *,
    threshold: int = DEFAULT_CYCLE_THRESHOLD,
    recovery_prompt: Optional[str] = None,
    max_lookback: int = 20,
    inject_as_user: bool = True,
) -> List[Dict[str, Any]]:
    if not messages:
        return messages

    if detect_cycle(messages, threshold=threshold, max_lookback=max_lookback):
        prompt = recovery_prompt or DEFAULT_RECOVERY_PROMPT
        role = "user" if inject_as_user else "system"
        injection: Dict[str, Any] = {"role": role, "content": prompt}
        logger.info("openai_extension: cycle detected, injecting recovery prompt (role=%s)", role)
        return list(messages) + [injection]

    return messages


def get_last_bash_commands_for_debug(messages: List[Dict[str, Any]], n: int = 10) -> List[str]:
    return _collect_recent_bash_commands(messages, max_lookback=n)

```

