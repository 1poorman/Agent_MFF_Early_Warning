#!/usr/bin/env python3
"""Run a command and record wall time plus sampled GPU/process memory.

Example: python tools/measure_run.py --gpu 1 --out-dir logs/experiment/train -- \
    /path/to/python tools/train_lora.py --config tools/mff_lora_minicpm5.yaml
The child runs with CUDA_VISIBLE_DEVICES set to --gpu. GPU memory uses MiB.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_sample(gpu: str, pid: int | None) -> dict:
    result = subprocess.run(
        ["nvidia-smi", "-i", gpu,
         "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    values = next(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    apps = subprocess.run(
        ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    processes = {}
    for row in csv.reader(apps.stdout.splitlines(), skipinitialspace=True):
        if len(row) == 2 and row[0].isdigit():
            try:
                processes[row[0]] = float(row[1])
            except ValueError:
                processes[row[0]] = None
    return {
        "time_utc": utc_now(), "gpu_name": values[0], "gpu_uuid": values[1],
        "driver_version": values[2], "gpu_total_mib": float(values[3]),
        "gpu_used_mib": float(values[4]), "gpu_util_percent": float(values[5]),
        "power_w": float(values[6]), "compute_process_memory_mib": processes,
        "child_gpu_memory_mib": processes.get(str(pid), 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0:
        parser.error("provide a command after -- and a positive interval")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite measurement directory: {out}")
    baseline = gpu_sample(args.gpu, None)
    meta = {
        "command": command, "cwd": str(Path.cwd()), "gpu_selector": args.gpu,
        "sample_interval_seconds": args.interval, "baseline": baseline,
        "memory_units": "MiB (2^20 bytes)",
        "process_memory_scope": "direct child PID; use a Python executable directly, not conda run",
        "started_utc": utc_now(), "status": "running",
    }
    summary_path = out / "resource_summary.json"

    def save_summary():
        temporary = out / "resource_summary.tmp"
        temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(summary_path)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false")
    samples = 0
    sample_errors = []
    peak_board, peak_child = 0.0, 0.0
    started = time.perf_counter()
    with (out / "command.log").open("w", encoding="utf-8") as log, \
            (out / "gpu_samples.jsonl").open("w", encoding="utf-8") as trace:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        meta["child_pid"] = child.pid
        save_summary()
        print(f"[measure] child_pid={child.pid} log={out / 'command.log'}", flush=True)
        try:
            while child.poll() is None:
                try:
                    row = gpu_sample(args.gpu, child.pid)
                    row["elapsed_seconds"] = time.perf_counter() - started
                    trace.write(json.dumps(row, ensure_ascii=False) + "\n")
                    trace.flush()
                    samples += 1
                    peak_board = max(peak_board, row["gpu_used_mib"])
                    peak_child = max(peak_child, row["child_gpu_memory_mib"] or 0.0)
                except (subprocess.SubprocessError, ValueError, StopIteration) as exc:
                    sample_errors.append({"time_utc": utc_now(), "error": str(exc)})
                try:
                    child.wait(timeout=args.interval)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        finally:
            meta.update({
                "status": "completed" if child.returncode == 0 else "failed",
                "exit_code": child.returncode, "finished_utc": utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "gpu_samples": samples, "sample_errors": sample_errors,
                "peak_gpu_used_mib": peak_board, "peak_child_gpu_memory_mib": peak_child,
                "children_max_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            })
            save_summary()
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
