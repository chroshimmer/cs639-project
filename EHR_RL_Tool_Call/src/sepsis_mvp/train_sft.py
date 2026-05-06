from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class ChatTraceDataset:
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_seq_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        messages = record["messages"]
        prompt_messages = record.get("prompt_messages") or messages[:-1]
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False,
        )
        prompt = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False,
        )
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = list(input_ids)
        prompt_len = min(len(prompt["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA SFT warm start for sepsis tool-call traces.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "train_sft requires torch, transformers, and peft. Install training dependencies on the GPU machine."
        ) from exc

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_dataset = ChatTraceDataset(_load_jsonl(args.train), tokenizer, args.max_seq_length)
    eval_dataset = (
        ChatTraceDataset(_load_jsonl(args.validation), tokenizer, args.max_seq_length)
        if args.validation
        else None
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_total_limit=3,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

