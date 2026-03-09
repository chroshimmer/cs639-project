# Extension 1 – State-Aware Cycle Detection & Recovery

## 1. Motivation & Theoretical Foundation

According to the **AgentBench (Liu et al., 2023)** study, LLMs often suffer from "Reasoning Stalls" in long-horizon OS tasks. When a command returns a non-informative output, agents frequently retry the exact same action. Our Extension 1 addresses this by implementing an external **Inference-Time Interceptor** to break these action loops, a method inspired by reactive control loops in autonomous systems.

------

## 2. Logic of Cycle Detection

We define the state of an agent's trajectory at time $t$ by the sequence of issued bash commands $C = \{c_1, c_2, \dots, c_t\}$. To detect a "hallucination loop," we implement a heuristic where a cycle is flagged if the following condition holds for a given threshold $k$:

$$\forall i \in [t-k+1, t], \quad c_i = c_{t}$$

In our implementation, we set $k=3$ (`DEFAULT_CYCLE_THRESHOLD`). This threshold is chosen to allow the model a minimum degree of self-correction (e.g., retrying after a minor typo) while preventing catastrophic **Task Limit Exceeded (TLE)** failures caused by infinite repetition.

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

### 3.2 Integration "Surgery"

The integration involves injecting the interceptor into the `OpenAIClient` class:

1. **Import**: `from .openai_extension import apply_cycle_detection_recovery`
2. **Hook Point**: In `_query_chat_completions`, we insert the call after the message format conversion but before the `client.chat.completions.create` call. This ensures we are intercepting the final payload being sent to the LLM.

------

## 4. Key Enhancements & Methods

Our Extension 1 introduces several robust methods to ensure the intervention is both accurate and effective:

- **Semantic Normalization**: The `_normalize_bash_command` function ensures that syntactic variations (e.g., `ls -a` vs `ls   -a`) are mapped to the same string representation $c_i$, preventing the model from bypassing detection with trivial spacing changes.
- **Pydantic-Dict Compatibility**: The `_to_message_dict` helper allows the interceptor to work seamlessly whether the `agentrl` library passes raw Python dictionaries or Pydantic `MessageRecord` models.
- **Stateful Trajectory Parsing**: `_collect_recent_bash_commands` filters the trajectory to only analyze `assistant` tool calls, ignoring system prompts or previous user interventions that might skew the detection.
- **Attention-Shift Recovery**: By injecting the warning as a `user` message, we leverage the model's training to prioritize recent user instructions, effectively forcing an "attention shift" to resolve the loop.

------

##  Appendix: Full Implementation Code (`openai_extension.py`)

Python

```
"""
State-Aware Cycle Detection & Recovery (Extension 1).

Lightweight, inference-time intervention for os-std / Long-Horizon OS tasks.
Designed to plug into agentrl.eval.client.openai.OpenAIClient._query_chat_completions
right before client.chat.completions.create() to break hallucination loops where
the agent repeatedly issues the same failing bash_action.

Message format: expects the same list as passed to create() — i.e. after
MessageRecord.convert_all(messages, to='openai_chat_completion_input') and
after trim_images/resize_images. Each item is a dict with at least "role",
and optionally "content", "tool_calls" (assistant), "tool_call_id" (tool).
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Dict, Optional

logger = logging.getLogger(__name__)

# Default threshold: same bash_action command 3 times consecutively = cycle
DEFAULT_CYCLE_THRESHOLD = 3

# Prompt injected to break the loop (user message so the model attends to it)
DEFAULT_RECOVERY_PROMPT = (
    "[SYSTEM WARNING]: Cycle Detected. You have repeatedly issued the exact same command. "
    "Stop and analyze the OS error. Try a completely different approach."
)


def _normalize_bash_command(raw: str) -> Optional[str]:
    """
    Normalize a bash command string for comparison (strip whitespace, collapse spaces).
    Returns None if parsing fails.
    """
    if not isinstance(raw, str):
        return None
    return " ".join(raw.strip().split())


def _get_bash_command_from_tool_call(tool_call: Dict[str, Any]) -> Optional[str]:
    """
    Parse a single tool_call dict and return the bash command string for bash_action, else None.

    Compatible with OpenAI Chat Completions input (openai_chat_completion_input from agentrl):
        tool_call["function"]["name"] == "bash_action"
        tool_call["function"]["arguments"] == JSON string, e.g. {"thought": "...", "command": "cd /tmp"}
    Matches os_interaction task: arguments = list(json.loads(...).values()); content = arguments[0].
    """
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
    """Normalize to dict: already-dict or agentrl MessageRecord-like (model_dump/dict)."""
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "dict"):
        return msg.dict()
    return None


def _collect_recent_bash_commands(messages: List[Dict[str, Any]], max_lookback: int = 20) -> List[str]:
    """
    Walk messages in order and collect the last N bash_action commands (same order as in history).
    Only considers assistant messages that contain at least one bash_action tool_call;
    we take the first bash_action in each such message (os-std typically one tool call per turn).
    """
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
    """
    Returns True if the last `threshold` bash_action commands are identical (cycle).
    """
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
    """
    Intercept the messages payload before client.chat.completions.create().
    If the same bash_action command appears `threshold` times consecutively,
    append a recovery user (or system) message to break the loop.

    Args:
        messages: OpenAI chat completion input (e.g. after MessageRecord.convert_all(..., to='openai_chat_completion_input')).
        threshold: Number of consecutive identical bash commands to consider a cycle (default 3).
        recovery_prompt: Override the injected warning text.
        max_lookback: How many recent bash commands to consider (default 20).
        inject_as_user: If True, append a "user" message; else append a "system" message.

    Returns:
        The same list, or a new list with one extra message if cycle detected. Does not mutate input.
    """
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
    """Convenience helper for logging/debug: return the last n bash commands in history."""
    return _collect_recent_bash_commands(messages, max_lookback=n)


# ---------------------------------------------------------------------------
# Integration with agentrl/eval/client/openai.py (OpenAIClient._query_chat_completions)
#
# 1) Add import at top of openai.py:
#    from src.client.openai_extension import apply_cycle_detection_recovery
#
# 2) In _query_chat_completions, after convert_all and trim_images/resize_images,
#    and right before client.chat.completions.create(...), insert:
#
#        messages = MessageRecord.convert_all(messages, to='openai_chat_completion_input')
#        if self.max_images is not None:
#            messages = trim_images(messages, self.max_images)
#        if self.image_size is not None:
#            messages = resize_images(messages, self.image_size)
#
#        messages = apply_cycle_detection_recovery(messages)   # <-- add this line
#
#        response = await client.chat.completions.create(
#            messages=messages,
#            model=model,
#            ...
#        )
#
# If a cycle is detected, one extra user message is appended; otherwise messages unchanged.

```

