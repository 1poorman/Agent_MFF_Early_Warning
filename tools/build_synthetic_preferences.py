#!/usr/bin/env python3
"""Build an explicitly synthetic DPO baseline and a matching GRPO prompt split.

Gold answers come from the simulator. Negatives are verified by RewardScorer;
these are neither human preferences nor AI-judge labels. Existing rollout-based
preference construction remains a separate experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.build_preference_data import SyntheticMaker, as_record, input_key, splice_json
from tools.rl_reward import RewardScorer, parse_output


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def build(source, excluded, seed=42, holdout=0.1, min_margin=0.05):
    if not 0 < holdout < 1 or not 0 < min_margin <= 1:
        raise ValueError("holdout and min_margin must be positive fractions")
    keys = [input_key(r["messages"]) for r in source]
    if len(keys) != len(set(keys)):
        raise ValueError("source has duplicate inputs")
    forbidden = {input_key(r["messages"]) for r in excluded}
    if set(keys) & forbidden:
        raise ValueError("source overlaps an excluded evaluation/SFT dataset")
    source_seeds = {r["meta"]["seed"] for r in source}
    if source_seeds & {r["meta"]["seed"] for r in excluded}:
        raise ValueError("source and excluded datasets share simulation seeds")
    rng = random.Random(seed)
    scorer = RewardScorer()
    maker = SyntheticMaker(scorer, rng)
    maker.faults = sorted({row["meta"]["gold"] for row in source})
    strata = defaultdict(list)
    for i, row in enumerate(source):
        strata[row["meta"]["kind"]].append(i)
    held = set()
    for indices in strata.values():
        rng.shuffle(indices)
        held.update(indices[:max(1, round(len(indices) * holdout))])
    outputs = {k: [] for k in ("prefs_train", "prefs_heldout", "prompts_train", "prompts_heldout")}
    counts = Counter()
    for i, row in enumerate(source):
        split = "heldout" if i in held else "train"
        messages = [m for m in row["messages"] if m["role"] != "assistant"]
        prompt = next(m["content"] for m in messages if m["role"] == "user")
        chosen = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        parsed = parse_output(chosen)
        if parsed is None:
            raise ValueError(f"invalid simulator JSON at index {i}")
        # Both sides use identical JSON serialization; indentation must not reveal the label.
        chosen = splice_json(chosen, parsed)
        gold = row["meta"]["gold"]
        cs = scorer.score(prompt, chosen, gold)
        if not cs.correct or not cs.json_ok or not cs.faith_ok:
            raise ValueError(f"invalid simulator gold at index {i}")
        # One pair per input, cycling corruption types to avoid a length-only task.
        kinds = list(maker.KINDS)
        kinds = kinds[i % len(kinds):] + kinds[:i % len(kinds)]
        for kind in kinds:
            rejected = maker.make(kind, chosen, gold, prompt)
            if rejected is None:
                continue
            rs = scorer.score(prompt, rejected, gold)
            margin = cs.total - rs.total
            if margin < min_margin:
                continue
            group = {"idx": i, "messages": messages, **row["meta"]}
            cmeta = {**asdict(cs), "reward": cs.total}
            rmeta = {**asdict(rs), "reward": rs.total}
            pair = as_record(group, chosen, rejected, margin, "gold", "synthetic",
                             cmeta, rmeta, synthetic_kind=kind)
            pair["feedback_source"] = "simulator_gold_and_rule_verified_corruption"
            pair["input_sha256"] = hashlib.sha256(keys[i].encode()).hexdigest()
            outputs[f"prefs_{split}"].append(pair)
            outputs[f"prompts_{split}"].append({"messages": messages, "meta": row["meta"]})
            counts[kind] += 1
            break
        else:
            raise ValueError(f"no verified negative at index {i}")
    for rows in outputs.values():
        rng.shuffle(rows)
    return outputs, dict(counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/rl/prompts_gold.jsonl")
    ap.add_argument("--exclude", nargs="+", default=["data/sft/train.jsonl", "data/sft/val.jsonl",
                                                       "data/sft/eval.jsonl", "data/rl/eval.jsonl"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--min-margin", type=float, default=0.05)
    args = ap.parse_args()
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    excluded = [row for path in args.exclude for row in read_rows(path)]
    outputs, counts = build(read_rows(args.source), excluded, args.seed, args.holdout, args.min_margin)
    out.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, rows in outputs.items():
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "source": args.source, "seed": args.seed,
                "feedback_source": "synthetic_rules", "human_labels": 0, "ai_judge_labels": 0,
                "min_margin": args.min_margin, "holdout": args.holdout,
                "counts": {k: len(v) for k, v in outputs.items()}, "corruptions": counts,
                "sha256": digests,
                "inputs_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                  for p in [args.source, *args.exclude]},
                "leakage": {"input_overlap": 0, "simulation_seed_overlap": 0,
                            "split_unit": "unique system+user input, stratified by kind"}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
