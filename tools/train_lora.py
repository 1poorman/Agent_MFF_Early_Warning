#!/usr/bin/env python3
"""Native-chat-template SFT runner (LoRA by default, or finetuning_type: full).

This runner intentionally does not depend on a LLaMA-Factory template fork. It
uses the tokenizer shipped with the base model, which keeps training and vLLM
inference on the same chat-template path for custom remote-code models.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    required = ("model_name_or_path", "train_file", "output_dir")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ids(value: Any) -> list[int]:
    """Normalize tokenizer output across transformers 4.x/5.x backends."""
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "ids"):
        value = value.ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def apply_template(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool,
                   enable_thinking: bool) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    try:
        return _ids(tokenizer.apply_chat_template(messages, **kwargs))
    except TypeError as exc:
        # Older custom tokenizers may not expose enable_thinking in the Jinja
        # template. The config still records the intended serving mode.
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        return _ids(tokenizer.apply_chat_template(messages, **kwargs))


def encode_row(row: dict[str, Any], tokenizer, max_length: int, enable_thinking: bool) -> dict[str, list[int]]:
    messages = row["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("each row must end with an assistant message")
    prompt_messages = messages[:-1]
    prompt_ids = apply_template(tokenizer, prompt_messages,
                                add_generation_prompt=True,
                                enable_thinking=enable_thinking)
    full_ids = apply_template(tokenizer, messages,
                              add_generation_prompt=False,
                              enable_thinking=enable_thinking)

    # A mismatch must fail: shortening this boundary could supervise prompt
    # tokens or an unintended reasoning prefix.
    prefix = len(prompt_ids)
    if full_ids[:prefix] != prompt_ids:
        raise ValueError("chat template prompt is not an exact prefix of the full conversation")

    full_ids = full_ids[:max_length]
    labels = [-100] * min(prefix, len(full_ids))
    labels.extend(full_ids[len(labels):])
    if len(labels) != len(full_ids):
        labels = labels[:len(full_ids)]
    if not any(label != -100 for label in labels):
        raise ValueError("row has no assistant tokens after truncation")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def load_jsonl(path: str, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def build_training_args(cfg: dict[str, Any], has_cuda: bool) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    args: dict[str, Any] = {
        "output_dir": cfg["output_dir"],
        "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 8)),
        "num_train_epochs": float(cfg.get("num_train_epochs", 3.0)),
        "learning_rate": float(cfg.get("learning_rate", 1e-4)),
        "lr_scheduler_type": cfg.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.1)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 1.0)),
        "logging_steps": int(cfg.get("logging_steps", 10)),
        "logging_first_step": True,
        "disable_tqdm": bool(cfg.get("disable_tqdm", True)),
        "save_steps": int(cfg.get("save_steps", 200)),
        "save_total_limit": int(cfg.get("save_total_limit", 3)),
        "logging_dir": str(Path(cfg["output_dir"]) / "logs"),
        "report_to": [],
        "remove_unused_columns": False,
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", True)),
        "seed": int(cfg.get("seed", 42)),
        "data_seed": int(cfg.get("seed", 42)),
        "dataloader_num_workers": int(cfg.get("dataloader_num_workers", 2)),
        "save_strategy": "steps",
        "bf16": bool(cfg.get("bf16", True)) and has_cuda,
        "fp16": bool(cfg.get("fp16", False)) and has_cuda,
    }
    if cfg.get("eval_file"):
        args[evaluation_key] = "steps"
        args["eval_steps"] = int(cfg.get("eval_steps", 200))
    else:
        args[evaluation_key] = "no"
    if "optim" in signature:
        args["optim"] = cfg.get("optim", "adamw_torch")
    if "ddp_find_unused_parameters" in signature:
        args["ddp_find_unused_parameters"] = False
    return TrainingArguments(**args)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def data_manifest(path: str, rows: list, encoded: Dataset) -> dict[str, Any]:
    lengths = [len(row["input_ids"]) for row in encoded]
    supervised = [sum(x != -100 for x in row["labels"]) for row in encoded]
    return {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "samples_used": len(rows),
        "kind_counts": dict(Counter(row.get("meta", {}).get("kind") for row in rows)),
        "sequence_tokens": {"min": min(lengths), "max": max(lengths),
                            "mean": sum(lengths) / len(lengths), "total": sum(lengths)},
        "supervised_tokens": {"min": min(supervised), "max": max(supervised),
                              "mean": sum(supervised) / len(supervised), "total": sum(supervised)},
    }


class ProgressLog(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.path = output_dir / "progress.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        row = {"time_utc": datetime.now(timezone.utc).isoformat(),
               "step": state.global_step, "epoch": state.epoch, **(logs or {})}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def cuda_peaks() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    return {
        "device": torch.cuda.get_device_name(),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def main() -> int:
    started = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-eval-samples", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="load/tokenize data and print trainable params")
    ap.add_argument("--output-dir", help="override output_dir without changing the experiment YAML")
    args = ap.parse_args()
    cfg = load_config(args.config)
    finetuning_type = cfg.get("finetuning_type", "lora")
    if finetuning_type not in ("lora", "full"):
        raise ValueError(f"unsupported finetuning_type: {finetuning_type}")
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    output_dir = Path(cfg["output_dir"])
    if not args.dry_run and output_dir.exists() and any(output_dir.iterdir()) and not cfg.get("resume_from_checkpoint"):
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}; choose --output-dir")
    seed_everything(int(cfg.get("seed", 42)))

    model_path = str(cfg["model_name_or_path"])
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"base model path does not exist: {model_path}. "
            "Mount/download the model before starting LoRA training."
        )
    has_cuda = torch.cuda.is_available()
    print(f"[env] torch={torch.__version__} cuda={has_cuda} devices={torch.cuda.device_count()}", flush=True)
    print(f"[model] {model_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=bool(cfg.get("trust_remote_code", True))
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    enable_thinking = bool(cfg.get("enable_thinking", False))
    max_length = int(cfg.get("cutoff_len", cfg.get("max_seq_length", 4096)))

    train_rows = load_jsonl(cfg["train_file"], args.max_train_samples)
    if not train_rows:
        raise ValueError("training dataset is empty")
    train_data = Dataset.from_list([
        encode_row(row, tokenizer, max_length, enable_thinking) for row in train_rows
    ])
    eval_data = None
    if cfg.get("eval_file"):
        eval_rows = load_jsonl(cfg["eval_file"], args.max_eval_samples)
        eval_data = Dataset.from_list([
            encode_row(row, tokenizer, max_length, enable_thinking) for row in eval_rows
        ])
    print(f"[data] train={len(train_data)} eval={len(eval_data) if eval_data else 0} max_len={max_length}", flush=True)

    dtype = torch.bfloat16 if bool(cfg.get("bf16", True)) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if bool(cfg.get("gradient_checkpointing", True)):
        model.enable_input_require_grads()

    if finetuning_type == "lora":
        target_modules = cfg.get("target_modules", "all-linear")
        if isinstance(target_modules, str) and "," in target_modules:
            target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]
        peft_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=int(cfg.get("lora_rank", 32)),
            lora_alpha=int(cfg.get("lora_alpha", 64)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
    else:
        # Keep full-model trainable weights BF16; FP32 promotion exceeds 24 GiB.
        model.requires_grad_(True)
        print(f"[full] trainable parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    if args.dry_run:
        print(f"[dry-run] tokenizer/model/{finetuning_type} setup succeeded", flush=True)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    packages = {}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets", "llamafactory"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {
        "finetuning_type": finetuning_type,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "python": platform.python_version(),
        "packages": packages, "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "base_dtype": str(dtype), "trainable_parameters": trainable,
        "total_parameters_with_adapter": total, "trainable_fraction": trainable / total,
        "trainable_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
        "adapter_modules": [name for name, module in model.named_modules() if hasattr(module, "lora_A")],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "train": data_manifest(cfg["train_file"], train_rows, train_data),
    }
    if eval_data is not None:
        manifest["val"] = data_manifest(cfg["eval_file"], eval_rows, eval_data)
    write_json(output_dir / "run_manifest.json", manifest)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
        pad_to_multiple_of=8 if has_cuda else None,
    )
    trainer = Trainer(
        model=model,
        args=build_training_args(cfg, has_cuda),
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[ProgressLog(output_dir)],
    )
    if has_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    if has_cuda:
        torch.cuda.synchronize()
    train_wall = time.perf_counter() - train_started
    train_memory = cuda_peaks()
    trainer.save_model(str(output_dir))
    trainer.save_state()
    tokenizer.save_pretrained(str(output_dir))
    metrics = dict(result.metrics)
    metrics.update({"train_samples": len(train_data), "eval_samples": len(eval_data) if eval_data else 0})
    metrics.update({"global_step": trainer.state.global_step, "train_wall_seconds": train_wall,
                    "cuda_memory_train_including_periodic_eval": train_memory})
    write_json(output_dir / f"{finetuning_type}_train_metrics.json", metrics)
    if eval_data:
        if has_cuda:
            torch.cuda.reset_peak_memory_stats()
        eval_metrics = trainer.evaluate()
        eval_metrics["cuda_memory"] = cuda_peaks()
        write_json(output_dir / "eval_metrics.json", eval_metrics)
    manifest.update({"finished_utc": datetime.now(timezone.utc).isoformat(),
                     "main_wall_seconds": time.perf_counter() - started, "status": "completed"})
    write_json(output_dir / "run_manifest.json", manifest)
    print(f"[done] {finetuning_type} weights saved to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
