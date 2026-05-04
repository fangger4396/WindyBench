#!/usr/bin/env python3
"""
LoRA/QLoRA fine-tuning for Llama-7B style causal LMs on QA JSONL dataset.

Expected dataset record format (one JSON object per line):
{
  "question": "...",
  "answer": "Cause Analysis: ...\\nAction Recommendations: ...\\nRisk Evaluation: ...",
  ...
}

Example:
python3 s04_finetune_lora_llama7b.py \
  --model-name-or-path meta-llama/Llama-2-7b-hf \
  --train-jsonl qa_dataset_train.jsonl \
  --eval-jsonl qa_dataset_val.jsonl \
  --output-dir outputs/llama7b-lora \
  --use-4bit \
  --num-train-epochs 3 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "You are a wind-turbine diagnostic assistant. "
    "Based on SCADA anomaly evidence, answer the question with technical and concise reasoning.\n\n"
    "### Question:\n{question}\n\n"
    "### Response:\n"
)

INSTR_INPUT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input_text}\n\n"
    "### Response:\n"
)


@dataclass
class Sample:
    prompt: str
    answer: str


@dataclass
class CausalLMCollator:
    """Pad input_ids/attention_mask and labels (with -100) to same batch length."""

    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [f["labels"] for f in features]
        model_features = [
            {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        seq_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            if len(lab) < seq_len:
                lab = lab + [self.label_pad_token_id] * (seq_len - len(lab))
            else:
                lab = lab[:seq_len]
            padded_labels.append(lab)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Llama-7B with LoRA/QLoRA on QA JSONL.")
    p.add_argument("--model-name-or-path", required=True, help="HF model name or local path")
    p.add_argument("--train-jsonl", required=True, help="Training JSONL file")
    p.add_argument("--eval-jsonl", default=None, help="Optional eval JSONL file")
    p.add_argument("--test-jsonl", default=None, help="Optional test JSONL file (not used for training)")
    p.add_argument("--output-dir", required=True, help="Output dir for LoRA adapter and tokenizer")
    p.add_argument("--auto-split", action="store_true", help="Auto split train-jsonl into train/eval/test")
    p.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio when --auto-split")
    p.add_argument("--eval-ratio", type=float, default=0.15, help="Eval split ratio when --auto-split")

    p.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")
    p.add_argument("--use-bf16", action="store_true", help="Enable bf16 training")
    p.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-scheduler-type", default="cosine")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--per-device-eval-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--gradient-checkpointing", action="store_true")

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="LoRA target module names",
    )

    p.add_argument("--train-on-inputs", action="store_true", help="If set, do not mask prompt tokens")
    p.add_argument("--packing", action="store_true", help="Concatenate samples before chunking")
    p.add_argument("--report-to", default="none", choices=["none", "tensorboard", "wandb"])
    p.add_argument("--merge-and-save", action="store_true", help="Merge LoRA into base model and save")
    p.add_argument("--merged-model-dir", default=None, help="Output dir for merged full model")
    p.add_argument(
        "--dataset-format",
        default="auto",
        choices=["auto", "qa", "cot"],
        help="Input JSONL format: qa(question/answer), cot(instruction/input/output), or auto detect.",
    )
    return p.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    required = ["Cause Analysis:", "Action Recommendations:", "Risk Evaluation:"]
    if all(k in answer for k in required):
        return answer
    return (
        "Cause Analysis:\n"
        + answer
        + "\n\nAction Recommendations:\n"
        + "Inspect affected subsystem and execute OEM maintenance procedures.\n\n"
        + "Risk Evaluation:\n"
        + "Persistent anomaly increases downtime and secondary component failure risk."
    )


def _build_from_qa(rows: List[Dict[str, Any]]) -> List[Sample]:
    out: List[Sample] = []
    for r in rows:
        q = str(r.get("question", "")).strip()
        a = str(r.get("answer", "")).strip()
        if not q or not a:
            continue
        prompt = PROMPT_TEMPLATE.format(question=q)
        out.append(Sample(prompt=prompt, answer=normalize_answer(a)))
    return out


def _build_from_cot(rows: List[Dict[str, Any]]) -> List[Sample]:
    out: List[Sample] = []
    for r in rows:
        instruction = str(r.get("instruction", "")).strip()
        input_text = str(r.get("input", "")).strip()
        output = str(r.get("output", "")).strip()
        if not instruction or not output:
            continue
        prompt = INSTR_INPUT_TEMPLATE.format(instruction=instruction, input_text=input_text)
        out.append(Sample(prompt=prompt, answer=output))
    return out


def build_samples(rows: List[Dict[str, Any]], dataset_format: str = "auto") -> List[Sample]:
    if dataset_format == "qa":
        return _build_from_qa(rows)
    if dataset_format == "cot":
        return _build_from_cot(rows)

    # auto: prefer explicit CoT fields, fallback to QA fields
    if rows:
        first = rows[0]
        if "instruction" in first and "output" in first:
            return _build_from_cot(rows)
    return _build_from_qa(rows)


def maybe_pack_samples(samples: List[Sample], sep: str = "\n\n") -> List[Sample]:
    # Minimal packing placeholder to keep behavior deterministic.
    # True token-level packing is typically done with TRL; here we keep one sample per row.
    return samples


def make_hf_dataset(samples: List[Sample]) -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [s.prompt for s in samples],
            "answer": [s.answer for s in samples],
            "text": [s.prompt + s.answer for s in samples],
        }
    )


def tokenize_and_mask(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
    train_on_inputs: bool,
) -> Dataset:
    def _tokenize(ex: Dict[str, Any]) -> Dict[str, Any]:
        full_text = ex["text"]
        prompt_text = ex["prompt"]

        tok_full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        labels = tok_full["input_ids"][:]

        if not train_on_inputs:
            tok_prompt = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
            )
            prompt_len = min(len(tok_prompt["input_ids"]), len(labels))
            labels[:prompt_len] = [-100] * prompt_len

        tok_full["labels"] = labels
        return tok_full

    return dataset.map(_tokenize, remove_columns=dataset.column_names, desc="Tokenizing")


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config: Optional[BitsAndBytesConfig] = None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # For torchrun/DDP, pin each process to one GPU to avoid cross-device loss tensors.
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        model_kwargs: Dict[str, Any] = {"device_map": {"": local_rank}}
    else:
        model_kwargs = {"device_map": "auto"}

    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.use_bf16 else torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    random.seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main_process = rank == 0
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = read_jsonl(args.train_jsonl)
    if not all_rows:
        raise ValueError(f"No valid rows found in train file: {args.train_jsonl}")

    train_rows: List[Dict[str, Any]]
    eval_rows: List[Dict[str, Any]]
    test_rows: List[Dict[str, Any]]

    if args.auto_split:
        if not (0.0 < args.train_ratio < 1.0):
            raise ValueError("--train-ratio must be in (0,1)")
        if not (0.0 <= args.eval_ratio < 1.0):
            raise ValueError("--eval-ratio must be in [0,1)")
        if args.train_ratio + args.eval_ratio >= 1.0:
            raise ValueError("--train-ratio + --eval-ratio must be < 1.0")

        rows = all_rows[:]
        random.shuffle(rows)
        n = len(rows)
        n_train = int(n * args.train_ratio)
        n_eval = int(n * args.eval_ratio)

        # Keep each split non-empty when possible.
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_eval = max(1, min(n - n_train - 1, n_eval))
        n_test = n - n_train - n_eval
        if n_test < 0:
            n_test = 0
            n_eval = n - n_train

        train_rows = rows[:n_train]
        eval_rows = rows[n_train : n_train + n_eval]
        test_rows = rows[n_train + n_eval :]
    else:
        train_rows = all_rows
        eval_rows = read_jsonl(args.eval_jsonl) if args.eval_jsonl else []
        test_rows = read_jsonl(args.test_jsonl) if args.test_jsonl else []

    train_samples = build_samples(train_rows, dataset_format=args.dataset_format)
    eval_samples = build_samples(eval_rows, dataset_format=args.dataset_format)
    test_samples = build_samples(test_rows, dataset_format=args.dataset_format)
    if args.packing:
        train_samples = maybe_pack_samples(train_samples)

    print(f"Train samples: {len(train_samples)}")
    print(f"Eval samples: {len(eval_samples)}")
    print(f"Test samples: {len(test_samples)}")

    model, tokenizer = load_model_and_tokenizer(args)

    train_ds = make_hf_dataset(train_samples)
    eval_ds = make_hf_dataset(eval_samples) if eval_samples else None

    train_tok = tokenize_and_mask(
        train_ds, tokenizer=tokenizer, max_length=args.max_length, train_on_inputs=args.train_on_inputs
    )
    eval_tok = (
        tokenize_and_mask(
            eval_ds, tokenizer=tokenizer, max_length=args.max_length, train_on_inputs=args.train_on_inputs
        )
        if eval_ds is not None
        else None
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps if eval_tok is not None else None,
        evaluation_strategy="steps" if eval_tok is not None else "no",
        bf16=args.use_bf16,
        fp16=not args.use_bf16,
        report_to=args.report_to,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False if world_size > 1 else None,
    )

    collator = CausalLMCollator(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )

    trainer.train()

    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if is_main_process:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved LoRA adapter to: {args.output_dir}")

    if args.merge_and_save and is_main_process:
        if not args.merged_model_dir:
            raise ValueError("--merge-and-save requires --merged-model-dir")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
            device_map="auto",
        )
        merged = PeftModel.from_pretrained(base_model, args.output_dir)
        merged = merged.merge_and_unload()
        Path(args.merged_model_dir).mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(args.merged_model_dir)
        tokenizer.save_pretrained(args.merged_model_dir)
        print(f"Saved merged full model to: {args.merged_model_dir}")


if __name__ == "__main__":
    train(parse_args())
