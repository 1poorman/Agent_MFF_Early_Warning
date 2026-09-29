#!/usr/bin/env python3
"""Read-only validation of hard_faults_v1, with optional simulation replay.

Checks rendered observations independently of the builder's audit. Writes only
the requested report; never modifies dataset files or historical fingerprints.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(text):
    return json.loads(text, parse_constant=reject_constant)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section(prompt, start, end):
    return read_json(prompt.split(start, 1)[1].split(end, 1)[0].strip())


def observation(prompt):
    return section(prompt, "【当前窗口统计】", "【当前工况】")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/sft_hard_v1")
    parser.add_argument("--historical-manifest", type=Path,
                        default=ROOT / "logs/rl_minicpm5_20260928/data_v2/manifest.json")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--replay", action="store_true", help="replay one episode per injected fault set per split")
    args = parser.parse_args()
    manifest = read_json((args.data_dir / "manifest.json").read_text())
    ontology = read_json((args.data_dir / "ontology.json").read_text())
    require(manifest["version"] == ontology["version"] == "hard_faults_v1", "unsupported version")
    for name, expected in manifest["source_sha256"].items():
        require(digest(args.data_dir / "source" / name) == expected, f"snapshot changed: {name}")
        require(digest(ROOT / name) == expected, f"current source differs: {name}")
    historical = read_json(args.historical_manifest.read_text())
    historical_inputs, historical_seeds = set(), set()
    for name, expected in historical["inputs_sha256"].items():
        path = ROOT / name
        require(digest(path) == expected, f"frozen data changed: {name}")
        for line in path.read_text().splitlines():
            row = read_json(line)
            historical_inputs.add(json.dumps(row["messages"][:-1], ensure_ascii=False, sort_keys=True))
            historical_seeds.add(row["meta"]["seed"])

    datasets, identities, counts = {}, {}, {}
    for split in ("train", "val", "eval"):
        path = args.data_dir / f"{split}.jsonl"
        require(digest(path) == manifest["sha256"][split], f"data hash mismatch: {split}")
        rows = [read_json(line) for line in path.read_text().splitlines()]
        datasets[split] = rows
        ids = {key: set() for key in ("input", "seed", "episode_id", "equipment_id")}
        episode_windows = {}
        for row in rows:
            meta, messages = row["meta"], row["messages"]
            require([m["role"] for m in messages] == ["system", "user", "assistant"], "message roles")
            require(meta["split"] == split and meta["version"] == manifest["version"], "split/version mismatch")
            prompt = messages[1]["content"]
            answer = read_json(messages[-1]["content"])
            observed = observation(prompt)
            reference = section(prompt, "【本炉次早期参考观测】", "【当前窗口统计】")
            require(set(observed) == set(reference), "reference channel mismatch")
            require(not set(observed) & set(meta["hidden_channels"]), "hidden channel exposed")
            for values in (observed, reference):
                for stats in values.values():
                    for field in ("均值", "去趋势标准差", "每分钟变化"):
                        require(type(stats[field]) in (int, float) and math.isfinite(stats[field]), "invalid statistic")
                    require(stats["去趋势标准差"] >= 0, "negative standard deviation")
            strengths = meta["counterfactual_support"]
            require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in strengths.values()), "invalid support")
            supported = sorted(ontology["fault_names"][name] for name, value in strengths.items() if value >= 1)
            require(answer["root_causes"] == meta["gold_causes"] == supported, "cause mismatch")
            require(type(answer["needs_more_data"]) is bool, "uncertainty must be boolean")
            require(answer["needs_more_data"] == meta["needs_more_data"], "uncertainty mismatch")
            expected_gold = " + ".join(supported) if supported else ontology[
                "uncertain_label" if answer["needs_more_data"] else "normal_label"]
            require(answer["root_cause"] == meta["gold"] == expected_gold, "set serialization mismatch")
            confidence = answer["confidence"]
            require(type(confidence) in (float, int) and math.isfinite(confidence) and 0 <= confidence <= 1, "invalid confidence")
            expected_evidence = {
                f"{v['传感器']}均值 {v['均值']}{v['单位']}，去趋势标准差 {v['去趋势标准差']}{v['单位']}，"
                f"每分钟变化 {v['每分钟变化']}{v['单位']}/min" for v in observed.values()}
            if meta["hidden_channels"]:
                unavailable = prompt.split("【不可用通道】", 1)[1].split("【维修工单】", 1)[0].strip()
                expected_evidence.add("缺失通道：" + unavailable)
            require(bool(answer["evidence"]) and all(e in expected_evidence for e in answer["evidence"]), "untraceable evidence")
            require(bool(answer["sop"]) and all(isinstance(s, str) and s.strip() for s in answer["sop"]), "invalid SOP")
            require(not any(key in prompt for key in ("counterfactual_support", "injected_faults", "simulation_config", "episode_id")), "metadata in prompt")
            key = json.dumps(messages[:-1], ensure_ascii=False, sort_keys=True)
            require(key not in ids["input"], "duplicate input")
            ids["input"].add(key)
            for name in ("seed", "episode_id", "equipment_id"):
                ids[name].add(meta[name])
            episode_windows.setdefault(meta["episode_id"], []).append(meta["window_end"])
        require(all(sorted(ends) == [1200, 1800] for ends in episode_windows.values()), "episode windows incomplete")
        require(not ids["input"] & historical_inputs and not ids["seed"] & historical_seeds, "historical overlap")
        basic = {"samples": len(rows), "episodes": len(ids["episode_id"]),
                 "by_kind": dict(Counter(r["meta"]["kind"] for r in rows)),
                 "by_cause": dict(Counter(c for r in rows for c in r["meta"]["gold_causes"])),
                 "shifted_samples": sum(r["meta"]["shifted_envelope"] for r in rows)}
        require(basic == manifest["counts"][split], "manifest counts differ")
        counts[split] = {**basic,
            "injected_compound_samples": sum(len(r["meta"]["injected_faults"]) > 1 for r in rows),
            "multi_cause_targets": sum(len(r["meta"]["gold_causes"]) > 1 for r in rows),
            "partial_compound_targets": sum(len(r["meta"]["injected_faults"]) > len(r["meta"]["gold_causes"]) > 0 for r in rows),
            "hidden_channel_samples": sum(bool(r["meta"]["hidden_channels"]) for r in rows),
            "injected_normal_samples": sum(not r["meta"]["injected_faults"] for r in rows)}
        identities[split] = ids
    for a, b in (("train", "val"), ("train", "eval"), ("val", "eval")):
        require(all(not identities[a][key] & identities[b][key] for key in identities[a]), f"split overlap: {a}/{b}")

    replayed = []
    versions = {"python": platform.python_version()}
    if args.replay:
        import numpy as np
        import pandas as pd
        from simulator import DataSimulator, FaultSpec, SimConfig
        versions.update(numpy=np.__version__, pandas=pd.__version__)
        for split, rows in datasets.items():
            seen = set()
            for row in sorted(rows, key=lambda r: (r["meta"]["episode_id"], r["meta"]["window_end"])):
                meta = row["meta"]
                fault_set = tuple(sorted(s["name"] for s in meta["injected_faults"]))
                if fault_set in seen:
                    continue
                seen.add(fault_set)
                frame = DataSimulator(SimConfig(**meta["simulation_config"]), schedule=meta["schedule"],
                                      faults=[FaultSpec(**s) for s in meta["injected_faults"]]).run(1800)
                require(np.isfinite(frame.select_dtypes(include="number").to_numpy()).all(), "non-finite replay")
                views = [r for r in rows if r["meta"]["episode_id"] == meta["episode_id"]]
                for view in views:
                    end = view["meta"]["window_end"]
                    prompt = view["messages"][1]["content"]
                    for start, stop, rendered in (
                        (end - 600, end, observation(prompt)),
                        (120, 300, section(prompt, "【本炉次早期参考观测】", "【当前窗口统计】")),
                    ):
                        for name, values in rendered.items():
                            x = frame[name].iloc[start:stop].to_numpy(dtype=float)
                            line = np.polyfit(np.arange(len(x)), x, 1)
                            expected = (round(float(x[-120:].mean()), 3),
                                        round(float((x - np.polyval(line, np.arange(len(x)))).std()), 3),
                                        round(float(line[0] * 60), 3))
                            actual = tuple(values[k] for k in ("均值", "去趋势标准差", "每分钟变化"))
                            require(actual == expected, f"replay mismatch: {meta['episode_id']}/{name}")
                replayed.append(meta["episode_id"])
            print(f"{split}: replayed {len(seen)} fault sets", flush=True)
    report = {"status": "passed", "dataset": str(args.data_dir.resolve()), "counts": counts,
              "data_sha256": manifest["sha256"], "source_sha256": manifest["source_sha256"],
              "historical_sha256": historical["inputs_sha256"],
              "checks": ["file and source fingerprints", "finite statistics", "exact evidence traceability",
                         "set-valued labels and uncertainty", "input/seed/episode/equipment isolation",
                         "historical fingerprints and input/seed isolation"],
              "replayed_episodes": replayed, "environment": versions}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
