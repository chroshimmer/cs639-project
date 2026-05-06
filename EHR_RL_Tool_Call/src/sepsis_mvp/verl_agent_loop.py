from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from .tools import build_tool_runtime


def _patch_sglang_http_server_launch() -> None:
    """Keep verl 0.7.1 compatible with newer sglang 0.5.10 HTTP internals."""
    try:
        from sglang.srt.entrypoints import engine, http_server
    except Exception:
        return
    if hasattr(http_server, "_launch_subprocesses"):
        return

    def _launch_subprocesses(*args: Any, **kwargs: Any):
        return engine.Engine._launch_subprocesses(*args, **kwargs)

    http_server._launch_subprocesses = _launch_subprocesses


_patch_sglang_http_server_launch()


def _load_agent_loop_types():
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
    except Exception:
        try:
            from verl.utils.agent_loop import AgentLoopBase, AgentLoopOutput, register
        except Exception as exc:
            raise RuntimeError(
                "Could not import verl AgentLoopBase/AgentLoopOutput. "
                "Install a verl version with Agent Loop support, then adjust this import path if the API moved."
            ) from exc
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
    except Exception:
        AgentLoopMetrics = dict
    try:
        from verl.utils.time_counter import simple_timer
    except Exception:
        from contextlib import contextmanager

        @contextmanager
        def simple_timer(name: str, metrics: dict[str, Any]):
            yield

    return AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register, simple_timer


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


try:
    _AgentLoopBase, _AgentLoopOutput, _AgentLoopMetrics, _register, _simple_timer = _load_agent_loop_types()
except RuntimeError:
    _AgentLoopBase = object
    _AgentLoopOutput = None
    _AgentLoopMetrics = dict

    def _register(name: str):
        def decorator(cls):
            return cls

        return decorator

    from contextlib import contextmanager

    @contextmanager
    def _simple_timer(name: str, metrics: dict[str, Any]):
        yield


@_register("sepsis_tool_agent_loop")
class SepsisToolAgentLoop(_AgentLoopBase):
    """verl Agent Loop for one sepsis checkpoint with official MIMIC tools.

    The exact verl Agent Loop API is still alpha. This class keeps the task logic
    isolated so only the generate/output glue should need adjustment if verl moves
    import paths or method names.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(super(), "__init__"):
            try:
                super().__init__(*args, **kwargs)
            except TypeError:
                pass
        self._runtime_cache: dict[str, Any] = {}

    def _runtime(self, db_path: str):
        if db_path not in self._runtime_cache:
            self._runtime_cache[db_path] = build_tool_runtime(tool_backend="official", db_path=db_path)
        return self._runtime_cache[db_path]

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _encode_tool_response(self, output: dict[str, Any]) -> list[int]:
        text = "\n" + json.dumps(output, default=str, separators=(",", ":")) + "\n"
        return self.tokenizer.encode(text, add_special_tokens=False)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        if _AgentLoopOutput is None:
            raise RuntimeError("verl Agent Loop is not importable in this environment.")

        raw_prompt = kwargs.get("raw_prompt") or kwargs.get("prompt")
        messages = json.loads(raw_prompt) if isinstance(raw_prompt, str) else raw_prompt
        if not isinstance(messages, list):
            raise ValueError("SepsisToolAgentLoop requires raw_prompt/prompt to contain chat messages.")

        extra_info = kwargs.get("extra_info")
        extra = json.loads(extra_info) if isinstance(extra_info, str) else dict(extra_info or {})
        db_path = extra["db_path"]
        stay_id = int(extra["stay_id"])
        t_hour = int(extra["t_hour"])
        max_turns = int(extra.get("max_step_interactions") or 3)
        available_tools = set(extra.get("available_tools") or ["query_suspicion_of_infection", "query_sofa"])
        runtime = self._runtime(db_path)

        prompt_ids = await self.apply_chat_template(list(messages))
        current_prompt_ids = list(prompt_ids)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        turn_scores: list[float] = []
        tool_rewards: list[float] = []
        metrics: dict[str, Any] = {"tool_calls": 0.0}
        response_length = int(getattr(self.rollout_config, "response_length", 512))
        num_preempted = -1

        for _ in range(max_turns + 1):
            with _simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=current_prompt_ids,
                    sampling_params=sampling_params,
                )
            num_preempted = output.num_preempted if getattr(output, "num_preempted", None) is not None else num_preempted
            generated_ids = list(output.token_ids)
            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))
            if getattr(output, "log_probs", None):
                response_logprobs.extend(output.log_probs)
            text = self._decode(generated_ids)
            payload = _extract_json_object(text)
            if not payload:
                break
            if "action" in payload:
                break
            tool_name = payload.get("tool_name")
            args = payload.get("arguments") or {}
            if tool_name not in available_tools:
                break
            args["stay_id"] = stay_id
            args["t_hour"] = t_hour
            with _simple_timer("tool_calls", metrics):
                tool_output = runtime.execute(tool_name, args)
            metrics["tool_calls"] = float(metrics.get("tool_calls", 0.0)) + 1.0
            tool_ids = self._encode_tool_response(tool_output)
            response_ids.extend(tool_ids)
            response_mask.extend([0] * len(tool_ids))
            turn_scores.append(0.0)
            tool_rewards.append(0.0)
            current_prompt_ids = prompt_ids + response_ids
            if len(response_ids) >= response_length:
                break

        metrics["num_preempted"] = num_preempted
        response_logprobs_out = response_logprobs[:response_length] if response_logprobs else None
        return _AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:response_length],
            response_mask=response_mask[:response_length],
            response_logprobs=response_logprobs_out,
            num_turns=int(metrics.get("tool_calls", 0.0)) + 1,
            metrics=metrics,
            extra_fields={"turn_scores": turn_scores, "tool_rewards": tool_rewards},
        )
