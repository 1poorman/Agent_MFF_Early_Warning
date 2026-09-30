#!/usr/bin/env python3
"""Build episode-isolated synthetic preferences from hard_faults_v1 train/val."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hard_fault_metrics import HardFaultScorer


def serialize(value):
    return json.dumps(value, ensure_ascii=False)


def corrupt(answer, mode, scorer, rng):
    rejected = copy.deepcopy(answer)
    if mode == "wrong_cause":
        alternatives = sorted(scorer.names - set(answer["root_causes"]))
        rejected["root_causes"] = [rng.choice(alternatives)]
    elif mode == "omit_or_add_cause":
        if len(answer["root_causes"]) > 1:
            rejected["root_causes"].pop()
        else:
            alternatives = sorted(scorer.names - set(answer["root_causes"]))
            rejected["root_causes"].append(rng.choice(alternatives))
    elif mode == "wrong_uncertainty":
        rejected["needs_more_data"] = not rejected["needs_more_data"]
    elif mode == "fabricated_evidence":
        rejected["evidence"][0] = re.sub(r"[-+]?\d+(?:\.\d+)?", "99999.123", rejected["evidence"][0], count=1)
    else:
        raise ValueError(mode)
    rejected["root_causes"].sort()
    rejected["root_cause"] = " + ".join(rejected["root_causes"]) if rejected["root_causes"] else scorer.ontology[
        "uncertain_label" if rejected["needs_more_data"] else "normal_label"]
    return rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/sft_hard_v1")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "logs/hard_minicpm5_20260929/rl_data")
    args = ap.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    scorer = HardFaultScorer(args.data_dir / "ontology.json")
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    rng = random.Random(42)
    modes = ["wrong_cause", "omit_or_add_cause", "wrong_uncertainty", "fabricated_evidence"]
    outputs, inputs, episodes, seeds, counts = {}, {}, {}, {}, {}
    for split in ("train", "val"):
        path = args.data_dir / f"{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"][split]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        built = []
        inputs[split], episodes[split], seeds[split] = set(), set(), set()
        for index, row in enumerate(rows):
            messages = row["messages"][:-1]
            user = messages[1]["content"]
            answer = json.loads(row["messages"][-1]["content"])
            gold = {k: answer[k] for k in ("root_causes", "needs_more_data")}
            mode = modes[index % len(modes)]
            rejected = corrupt(answer, mode, scorer, rng)
            chosen_text, rejected_text = serialize(answer), serialize(rejected)
            chosen_score = scorer.score(user, chosen_text, gold)
            rejected_score = scorer.score(user, rejected_text, gold)
            assert chosen_score.total > 0.9999 and chosen_score.total - rejected_score.total >= 0.05
            assert chosen_text != rejected_text
            key = serialize(messages)
            assert key not in inputs[split]
            inputs[split].add(key)
            episodes[split].add(row["meta"]["episode_id"])
            seeds[split].add(row["meta"]["seed"])
            built.append({"messages": messages, "prompt": user, "chosen": chosen_text,
                          "rejected": rejected_text, "source": "synthetic_rules",
                          "corruption": mode, "gold_target": gold, "meta": row["meta"],
                          "reward_margin": chosen_score.total - rejected_score.total})
        counts[split] = {"pairs": len(built), "corruptions": dict(Counter(r["corruption"] for r in built))}
        outputs[split] = built
    assert not inputs["train"] & inputs["val"]
    assert not episodes["train"] & episodes["val"] and not seeds["train"] & seeds["val"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, rows in outputs.items():
        path = args.out_dir / f"prefs_{split}.jsonl"
        path.write_text("".join(serialize(r) + "\n" for r in rows))
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"feedback_source": "synthetic_rules", "human_labels": 0, "ai_judge_labels": 0,
              "counts": counts, "sha256": hashes, "source_data_sha256": manifest["sha256"],
              "input_overlap": 0, "seed_overlap": 0, "episode_overlap": 0,
              "eval_note": "Frozen eval rows were not read or used to build preferences.",
              "reward_script_sha256": hashlib.sha256((ROOT / "tools/hard_fault_metrics.py").read_bytes()).hexdigest()}
    (args.out_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
