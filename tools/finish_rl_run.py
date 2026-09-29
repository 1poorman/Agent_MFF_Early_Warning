#!/usr/bin/env python3
"""Wait for a measured RL run, merge its adapter, then evaluate both frozen sets.

Each stage has its own refusing-to-overwrite output directory. A failed training
run cannot accidentally be scored as a completed experiment.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="logs/rl_minicpm5_20260928")
    ap.add_argument("--method", required=True, choices=["dpo", "grpo"])
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--train-python", default="/home/huachenghao/.conda/envs/mff_rl/bin/python")
    ap.add_argument("--timeout-hours", type=float, default=8)
    args = ap.parse_args()
    root = Path(args.run_dir) / args.method
    status_path = root / "pipeline.json"
    if status_path.exists():
        raise FileExistsError(f"refusing to overwrite {status_path}")
    state = {"method": args.method, "started_utc": datetime.now(timezone.utc).isoformat(),
             "command": [sys.executable, *sys.argv], "stages": []}

    def status(value):
        state["status"] = value
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(state, indent=2) + "\n")
        print(value, flush=True)

    def run(label, command, env=None):
        status(label)
        start = time.perf_counter()
        with (root / f"{label}.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        state["stages"].append({"label": label, "command": command,
                                "wall_seconds": time.perf_counter() - start,
                                "exit_code": result.returncode})
        result.check_returncode()

    try:
        status("waiting_for_training")
        deadline = time.monotonic() + args.timeout_hours * 3600
        while True:
            measured = json.loads((root / "train/resource_summary.json").read_text())
            if measured["status"] == "failed":
                raise RuntimeError("training failed; inspect train/command.log")
            if measured["status"] == "completed":
                break
            if time.monotonic() > deadline:
                raise TimeoutError("training wait timed out")
            time.sleep(10)
        manifest = json.loads((root / "adapter/run_manifest.json").read_text())
        if manifest["status"] != "completed":
            raise RuntimeError("training measurement and model manifest disagree")
        base = manifest["policy_start_and_reference"]
        run("merge", [args.train_python, "tools/merge_lora.py", "--base-model", base,
                       "--adapter", str(root / "adapter"), "--output", str(root / "merged")],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8"))
        for label, dataset, repeats in (("eval", "data/sft/eval.jsonl", 3),
                                         ("stress_eval", "data/rl/eval.jsonl", 1)):
            run(label, [sys.executable, "tools/run_adapter_eval.py", "--model", str(root / "merged"),
                        "--name", f"mff-{args.method}", "--gpu", args.gpu, "--port", str(args.port),
                        "--runs", str(repeats), "--dataset", dataset, "--out-dir", str(root / label)])
        status("completed")
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        status("failed")
        raise


if __name__ == "__main__":
    main()
