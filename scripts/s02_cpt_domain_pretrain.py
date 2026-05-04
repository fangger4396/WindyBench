#!/usr/bin/env python3
"""
CPT (Continuous Domain Pre-training) for wind-turbine domain text.

This script performs causal LM continued pretraining on unlabeled domain corpus,
matching the paper's Stage-1 EAA adaptation before SFT.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def read_corpus(paths: List[str]) -> List[str]:
    docs: List[str] = []
    for p in paths:
        path = Path(p)
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        if "text" in obj:
                            docs.append(str(obj["text"]))
                        elif "content" in obj:
                            docs.append(str(obj["content"]))
                    elif isinstance(obj, str):
                        docs.append(obj)
        else:
            docs.append(path.read_text(encoding="utf-8"))
    return [d.strip() for d in docs if d.strip()]


def pack_docs(docs: List[str], tokenizer: AutoTokenizer, block_size: int) -> Dataset:
    ids: List[int] = []
    for d in docs:
        ids.extend(tokenizer(d, add_special_tokens=True)["input_ids"])
    chunks = [ids[i : i + block_size] for i in range(0, len(ids) - block_size, block_size)]
    data = {
        "input_ids": chunks,
        "attention_mask": [[1] * len(c) for c in chunks],
    }
    return Dataset.from_dict(data)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPT continued pretraining for wind-turbine domain text.")
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--corpus", nargs="+", required=True, help="Input corpus files (.txt or .jsonl with text/content).")
    p.add_argument("--output-dir", required=True, help="Output directory for CPT LoRA adapter and tokenizer.")
    p.add_argument("--use-4bit", action="store_true")
    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"device_map": "auto"}
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
        )
    model_kwargs["torch_dtype"] = torch.bfloat16 if args.use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules,
        ),
    )
    model.print_trainable_parameters()

    docs = read_corpus(args.corpus)
    if not docs:
        raise ValueError("No domain corpus text found.")
    train_ds = pack_docs(docs, tokenizer, args.block_size)
    print(f"CPT chunks: {len(train_ds)}")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.use_bf16,
        fp16=not args.use_bf16,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved CPT adapter: {args.output_dir}")


if __name__ == "__main__":
    main()
