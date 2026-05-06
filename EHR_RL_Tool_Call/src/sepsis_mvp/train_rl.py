from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VALID_ACTIONS = {"keep_monitoring", "infection_suspect", "trigger_sepsis_alert"}
VALID_TOOLS = {"query_suspicion_of_infection", "query_sofa"}


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _reward_completion(text: str, oracle_text: str) -> float:
    payload = _extract_json_object(text)
    try:
        oracle = json.loads(oracle_text)
    except json.JSONDecodeError:
        oracle = {}
    if payload is None:
        return -1.0
    reward = 0.1
    if "tool_name" in payload:
        if payload.get("tool_name") in VALID_TOOLS and isinstance(payload.get("arguments"), dict):
            reward += 0.4
        else:
            reward -= 0.5
    elif "action" in payload:
        if payload.get("action") in VALID_ACTIONS:
            reward += 0.4
        else:
            reward -= 0.5
    else:
        reward -= 0.5
    if _normalize_json(payload) == _normalize_json(oracle):
        reward += 1.0
    elif payload.get("action") == oracle.get("action") or payload.get("tool_name") == oracle.get("tool_name"):
        reward += 0.3
    return max(-1.0, min(1.5, reward))


def _token_logprobs(model: Any, input_ids: Any, attention_mask: Any, prompt_len: int):
    import torch
    import torch.nn.functional as F

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    logprobs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    token_positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    completion_mask = token_positions >= max(prompt_len - 1, 0)
    completion_mask = completion_mask & (attention_mask[:, 1:].bool())
    denom = completion_mask.sum(dim=1).clamp_min(1)
    return (logprobs * completion_mask).sum(dim=1) / denom


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight grouped policy-gradient fine-tuning on exported sepsis tool-call states. "
            "This is a practical one-GPU GRPO-style warm RL loop, not a full verl/PPO trainer."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="Optional SFT LoRA adapter to initialize from.")
    parser.add_argument("--train-traces", required=True, help="JSONL from `sepsis_mvp.cli export-sft-traces`.")
    parser.add_argument("--validation-traces")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import random
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "train_rl requires torch, transformers, and peft. Install training dependencies on the GPU machine."
        ) from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
        )
        model = get_peft_model(model, lora)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.cuda()
    model.train()
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate)
    records = _load_jsonl(args.train_traces)
    if not records:
        raise SystemExit(f"No records found in {args.train_traces}")

    for update in range(1, args.updates + 1):
        batch = random.sample(records, k=min(args.batch_size, len(records)))
        losses = []
        all_rewards = []
        for record in batch:
            prompt_messages = record.get("prompt_messages") or record["messages"][:-1]
            oracle = record["completion"]
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_seq_length).to(model.device)
            group_logps = []
            group_rewards = []
            for _ in range(args.group_size):
                with torch.no_grad():
                    generated = model.generate(
                        **prompt,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                completion_ids = generated[0, prompt["input_ids"].shape[1] :]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                reward = _reward_completion(completion, oracle)
                attention_mask = torch.ones_like(generated, device=model.device)
                logp = _token_logprobs(model, generated, attention_mask, prompt["input_ids"].shape[1])
                group_logps.append(logp[0])
                group_rewards.append(reward)
            rewards = torch.tensor(group_rewards, dtype=torch.float32, device=model.device)
            advantages = rewards - rewards.mean()
            if rewards.numel() > 1 and float(rewards.std()) > 1e-6:
                advantages = advantages / rewards.std()
            logps = torch.stack(group_logps)
            losses.append(-(advantages.detach() * logps).mean())
            all_rewards.extend(group_rewards)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % 10 == 0:
            mean_reward = sum(all_rewards) / len(all_rewards)
            print(json.dumps({"update": update, "loss": round(float(loss.detach().cpu()), 6), "mean_reward": round(mean_reward, 4)}), flush=True)
        if update % 100 == 0:
            model.save_pretrained(Path(args.output_dir) / f"checkpoint-{update}")
            tokenizer.save_pretrained(Path(args.output_dir) / f"checkpoint-{update}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

