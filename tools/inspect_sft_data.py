#!/usr/bin/env python3
"""Describe frozen SFT splits, duplicates, token lengths, and loss-mask parity."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

from transformers import AutoTokenizer

from train_lora import apply_template, encode_row, load_jsonl


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def describe(values: list[int]) -> dict:
    ordered = sorted(values)
    return {"min": min(values), "max": max(values), "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": ordered[round(0.95 * (len(ordered) - 1))], "total": sum(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/huachenghao/models/MiniCPM5-2B")
    parser.add_argument("--data-dir", default="data/sft")
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--check-lf-template", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    template = lf_tokenizer = None
    if args.check_lf_template:
        sys.path.insert(0, str(Path.home() / "codes/LLaMA-Factory/src"))
        from llamafactory.data.template import get_template_and_fix_tokenizer
        from llamafactory.hparams import DataArguments
        lf_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        template = get_template_and_fix_tokenizer(
            lf_tokenizer, DataArguments(template="minicpm5", enable_thinking=False))

    result = {"model": args.model, "cutoff_len": args.cutoff_len,
              "enable_thinking": False, "splits": {}, "cross_split": {}}
    all_fingerprints, all_seeds = {}, {}
    labels_for_input = defaultdict(set)
    mismatches = []
    for split in ("train", "val", "eval"):
        path = Path(args.data_dir) / f"{split}.jsonl"
        rows = load_jsonl(str(path))
        fingerprints, lengths, response_lengths, prompt_lengths = [], [], [], []
        truncated = 0
        for index, row in enumerate(rows):
            messages = row["messages"]
            fingerprint = digest(json.dumps(messages[:-1], ensure_ascii=False,
                                            sort_keys=True).encode())
            fingerprints.append(fingerprint)
            labels_for_input[fingerprint].add(row["meta"]["gold"])
            encoded = encode_row(row, tokenizer, args.cutoff_len, False)
            full = apply_template(tokenizer, messages, add_generation_prompt=False,
                                  enable_thinking=False)
            length = len(full)
            prompt_length = sum(x == -100 for x in encoded["labels"])
            lengths.append(length)
            prompt_lengths.append(prompt_length)
            response_lengths.append(sum(x != -100 for x in encoded["labels"]))
            truncated += length > args.cutoff_len
            if template is not None:
                system = messages[0]["content"] if messages[0]["role"] == "system" else ""
                rest = messages[1:] if messages[0]["role"] == "system" else messages
                prompt, response = template.encode_oneturn(lf_tokenizer, rest, system=system)
                expected_ids = (prompt + response)[:args.cutoff_len]
                expected_labels = ([-100] * len(prompt) + response)[:args.cutoff_len]
                if encoded["input_ids"] != expected_ids or encoded["labels"] != expected_labels:
                    mismatches.append({"split": split, "index": index})
        seeds = set(row["meta"]["seed"] for row in rows)
        all_fingerprints[split], all_seeds[split] = fingerprints, seeds
        result["splits"][split] = {
            "path": str(path.resolve()), "sha256": digest(path.read_bytes()),
            "bytes": path.stat().st_size, "rows": len(rows),
            "kinds": dict(Counter(row["meta"]["kind"] for row in rows)),
            "gold": dict(Counter(row["meta"]["gold"] for row in rows)), "seeds": sorted(seeds),
            "unique_inputs": len(set(fingerprints)),
            "duplicate_input_rows": len(rows) - len(set(fingerprints)),
            "full_sequence_tokens": describe(lengths), "prompt_tokens": describe(prompt_lengths),
            "supervised_tokens": describe(response_lengths), "truncated_rows": truncated,
        }
    for a, b in (("train", "val"), ("train", "eval"), ("val", "eval")):
        overlap = set(all_fingerprints[a]) & set(all_fingerprints[b])
        result["cross_split"][f"{a}_{b}"] = {
            "shared_unique_inputs": len(overlap),
            f"{a}_rows_with_shared_input": sum(f in overlap for f in all_fingerprints[a]),
            f"{b}_rows_with_shared_input": sum(f in overlap for f in all_fingerprints[b]),
            "shared_seeds": sorted(all_seeds[a] & all_seeds[b]),
        }
    result["conflicting_gold_inputs"] = sum(len(labels) > 1 for labels in labels_for_input.values())
    result["lf_parity"] = {"checked": args.check_lf_template,
                           "rows": sum(x["rows"] for x in result["splits"].values()) if template else 0,
                           "mismatches": mismatches}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
