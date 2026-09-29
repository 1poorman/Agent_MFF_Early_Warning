#!/usr/bin/env python3
"""Compare any number of completed runs on one frozen behavioral dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def calibration(rows):
    valid = [r for r in rows if isinstance(r.get("conf_raw"), (int, float))
             and math.isfinite(r["conf_raw"]) and 0 <= r["conf_raw"] <= 1]
    if not valid:
        return {"n": 0, "brier": None, "ece_10": None}
    brier = mean((r["conf_raw"] - int(r["correct_raw"])) ** 2 for r in valid)
    ece = 0
    for i in range(10):
        sub = [r for r in valid if min(int(r["conf_raw"] * 10), 9) == i]
        if sub:
            ece += len(sub) / len(valid) * abs(mean(r["conf_raw"] for r in sub)
                                               - mean(r["correct_raw"] for r in sub))
    return {"n": len(valid), "brier": brier, "ece_10": ece}


def summarize(dataset, arms):
    samples = read_rows(dataset)
    digest = hashlib.sha256(Path(dataset).read_bytes()).hexdigest()
    unique = {}
    for i, sample in enumerate(samples):
        key = json.dumps([m for m in sample["messages"] if m["role"] != "assistant"],
                         ensure_ascii=False, sort_keys=True)
        unique.setdefault(key, i)
    output = {"dataset": str(dataset), "dataset_sha256": digest,
              "n": len(samples), "unique_inputs": len(unique), "arms": {}}
    reference_inference = None
    for name, directory in arms.items():
        directory = Path(directory)
        manifest = json.loads((directory / "evaluation_manifest.json").read_text())
        if (manifest["status"] != "completed" or manifest["dataset_sha256"] != digest
                or not manifest["temporary_service_stopped"] or manifest["runs"] < 1):
            raise ValueError(f"incomplete or incompatible evaluation: {directory}")
        if reference_inference is None:
            reference_inference = manifest["inference"]
        if manifest["inference"] != reference_inference:
            raise ValueError(f"inference settings differ: {name}")
        results = []
        for run in range(1, manifest["runs"] + 1):
            rows = read_rows(directory / f"run{run}.jsonl")
            if len(rows) != len(samples):
                raise ValueError(f"incomplete run: {name}/{run}")
            for i, (row, sample) in enumerate(zip(rows, samples)):
                if (row["idx"] != i or row["gold"] != sample["meta"]["gold"]
                        or row["kind"] != sample["meta"]["kind"] or row.get("error")):
                    raise ValueError(f"misaligned or failed result: {name}/{run}/{i}")
            faults = [r for r in rows if r["kind"] in ("clear", "boundary", "composite", "misleading")]
            antihall = [r for r in rows if r["kind"] == "antihall"]
            normal = [r for r in rows if r["kind"] == "transition"]
            valid_hallu = [r for r in rows if r["kind"] != "antihall"]
            dedup = [rows[i] for i in unique.values()]
            stats = {"fault_accuracy": mean(r["correct_raw"] for r in faults) if faults else None,
                     "overall_accuracy": mean(r["correct_raw"] for r in rows),
                     "refusal_rate": mean(r["correct_raw"] for r in antihall) if antihall else None,
                     "false_positive_rate": mean(not r["correct_raw"] for r in normal) if normal else None,
                     "json_rate": mean(r["json_ok"] for r in rows),
                     "faith_rate": mean(r["faith_ok"] for r in valid_hallu) if valid_hallu else None,
                     "hallu_rate": mean(r["hallu_ok"] for r in valid_hallu) if valid_hallu else None,
                     "n_wrong": sum(not r["correct_raw"] for r in rows),
                     "unique_input_accuracy": mean(r["correct_raw"] for r in dedup),
                     "brier": calibration(rows)["brier"], "ece_10": calibration(rows)["ece_10"],
                     "calibration_n": calibration(rows)["n"],
                     "unique_input_brier": calibration(dedup)["brier"]}
            results.append(stats)
        output["arms"][name] = {"directory": str(directory), "gpu": manifest["gpu"],
                                "runs": results, "mean": {}, "std": {}}
        for metric in results[0]:
            values = [r[metric] for r in results if r[metric] is not None]
            output["arms"][name]["mean"][metric] = mean(values) if values else None
            output["arms"][name]["std"][metric] = pstdev(values) if values else None
    output["inference"] = reference_inference
    return output


def markdown(result):
    lines = ["# SFT 与 RL 后训练行为对比", "",
             f"测试集：`{result['dataset']}`；{result['n']} 条，{result['unique_inputs']} 个唯一输入。",
             f"SHA256：`{result['dataset_sha256']}`。", "",
             "各列为重复推理均值；重复推理不代表多个独立训练种子。缺少的子集指标显示为 —。", "",
             "| 方法 | 复测次数 | 故障准确率 | 总体准确率 | 拒诊率 | 误报率 | JSON | 证据忠实 | Brier ↓ | ECE ↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, arm in result["arms"].items():
        m = arm["mean"]
        pct = lambda key: "—" if m[key] is None else f"{100*m[key]:.2f}%"
        dec = lambda key: "—" if m[key] is None else f"{m[key]:.4f}"
        lines.append(f"| {name} | {len(arm['runs'])} | " + " | ".join(
            [pct(k) for k in ("fault_accuracy", "overall_accuracy", "refusal_rate", "false_positive_rate",
                              "json_rate", "faith_rate")] + [dec("brier"), dec("ece_10")]) + " |")
    lines += ["", "Brier/ECE 使用模型自报置信度与原始根因是否正确计算；10 个等宽置信度区间。",
              "原始明细、去重结果和标准差见同名 JSON。训练数据来源、预算和成本须结合实验记录解释。", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/sft/eval.jsonl")
    ap.add_argument("--arm", action="append", required=True, help="NAME=EVALUATION_DIRECTORY")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    arms = dict(item.split("=", 1) for item in args.arm)
    if len(arms) != len(args.arm):
        ap.error("duplicate arm names")
    result = summarize(args.dataset, arms)
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    out.with_suffix(".md").write_text(markdown(result))
    print(markdown(result))


if __name__ == "__main__":
    main()
