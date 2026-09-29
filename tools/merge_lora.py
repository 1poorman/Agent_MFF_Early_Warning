#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model for vLLM serving."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing export: {out}")
    out.mkdir(parents=True, exist_ok=True)
    # Merging uses CPU regardless of visible GPUs; export precision is explicit.
    dtype = getattr(torch, args.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(out)
    (out / "lora_merge_manifest.json").write_text(
        json.dumps({"base_model": args.base_model, "adapter": args.adapter,
                    "device": "cpu", "dtype": args.dtype,
                    "adapter_sha256": hashlib.sha256(
                        (Path(args.adapter) / "adapter_model.safetensors").read_bytes()).hexdigest(),
                    "parameters": sum(p.numel() for p in merged.parameters()),
                    "weight_dtypes": sorted({str(p.dtype) for p in merged.parameters()}),
                    "wall_seconds": time.perf_counter() - started}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"merged model saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
