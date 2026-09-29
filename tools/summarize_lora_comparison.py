#!/usr/bin/env python3
"""Check completed paired evaluations and summarize errors and duplicate inputs."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="logs/lora_minicpm5_20260923")
    parser.add_argument("--dataset", default="data/sft/eval.jsonl")
    parser.add_argument("--out", default="logs/lora_minicpm5_20260923/comparison_summary.json")
    args = parser.parse_args()
    dataset = Path(args.dataset)
    samples = read_rows(dataset)
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    unique = {}
    for index, row in enumerate(samples):
        key = json.dumps(row["messages"][:-1], ensure_ascii=False, sort_keys=True)
        unique.setdefault(key, index)
    result = {"dataset_sha256": digest, "rows": len(samples), "unique_inputs": len(unique)}
    all_rows, manifests = {}, {}
    for name in ("full", "lora"):
        directory = Path(args.run_dir) / f"{name}_eval"
        manifest = read_json(directory / "evaluation_manifest.json")
        if manifest["status"] != "completed" or not manifest["temporary_service_stopped"]:
            raise ValueError(f"evaluation not completed and stopped: {directory}")
        if manifest["dataset_sha256"] != digest or manifest["runs"] != 3:
            raise ValueError(f"dataset or repetition count mismatch: {directory}")
        manifests[name] = manifest
        all_rows[name] = []
        summaries, errors, deduplicated = [], [], []
        for repeat in range(1, 4):
            rows = read_rows(directory / f"run{repeat}.jsonl")
            summary = read_json(directory / f"run{repeat}.summary.json")
            if len(rows) != len(samples) or summary["errors"]:
                raise ValueError(f"incomplete/failed run: {name} {repeat}")
            for index, row in enumerate(rows):
                if row["idx"] != index or row["gold"] != samples[index]["meta"]["gold"]:
                    raise ValueError(f"row alignment mismatch: {name} {repeat} {index}")
            all_rows[name].append(rows)
            summaries.append(summary)
            errors.append([{key: row[key] for key in ("idx", "kind", "gold", "rc_raw", "conf_raw")}
                           for row in rows if not row["correct_raw"]])
            selected = [rows[index] for index in unique.values()]
            faults = [row for row in selected if row["kind"] in ("clear", "boundary", "composite", "misleading")]
            deduplicated.append({"n": len(selected), "correct": sum(row["correct_raw"] for row in selected),
                                 "n_fault": len(faults), "fault_correct": sum(row["correct_raw"] for row in faults)})
        result[name] = {
            "runs": summaries, "errors_by_run": errors, "deduplicated_by_run": deduplicated,
            "predictions_identical_across_runs": all(
                [row["rc_raw"] for row in rows] == [row["rc_raw"] for row in all_rows[name][0]]
                for rows in all_rows[name][1:]),
            "mean_fault_accuracy": mean(item["accuracy"]["fault_subset_raw"] for item in summaries),
            "mean_p50_seconds": mean(item["latency"]["p50"] for item in summaries),
            "mean_p95_seconds": mean(item["latency"]["p95"] for item in summaries),
            "mean_eval_wall_seconds": mean(item["wall_s"] for item in summaries),
            "startup_seconds": manifest["startup_seconds"], "server_gpu_state": manifest["server_gpu_state"],
        }
    a, b = manifests["full"], manifests["lora"]
    if a["gpu"] != b["gpu"] or a["inference"] != b["inference"] or a["server_command"][5:] != b["server_command"][5:]:
        raise ValueError("inference settings differ between the two models")
    result["paired_disagreements_by_run"] = []
    for full, lora in zip(all_rows["full"], all_rows["lora"]):
        result["paired_disagreements_by_run"].append([
            {"idx": x["idx"], "kind": x["kind"], "gold": x["gold"],
             "full": x["rc_raw"], "lora": y["rc_raw"],
             "full_correct": x["correct_raw"], "lora_correct": y["correct_raw"]}
            for x, y in zip(full, lora) if x["rc_raw"] != y["rc_raw"]])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    for name in ("full", "lora"):
        print(name, json.dumps({k: v for k, v in result[name].items() if k != "runs"}, ensure_ascii=False))
    print("paired_disagreements", json.dumps(result["paired_disagreements_by_run"], ensure_ascii=False))


if __name__ == "__main__":
    main()
