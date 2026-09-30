#!/usr/bin/env python3
"""Serve one frozen model temporarily and repeat the existing SFT evaluation.

Runs with the mff_agent Python. Only the vLLM child uses the selected GPU;
the evaluation stack stays on CPU. The temporary service is stopped on exit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def request_json(url: str, payload: dict | None = None, timeout: float = 5):
    request = Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=3763)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="data/sft/eval.jsonl")
    parser.add_argument("--schema", choices=["legacy", "hard_faults_v1"], default="legacy")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--vllm", default="/home/huachenghao/.conda/envs/mff_sft_serve/bin/vllm")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    # Fail before starting a model if another service owns this port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    dataset = Path(args.dataset)
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    command = [args.vllm, "serve", str(Path(args.model).resolve()),
               "--served-model-name", args.name, "--dtype", "bfloat16",
               "--host", "127.0.0.1", "--port", str(args.port),
               "--max-model-len", "8192", "--max-num-seqs", "8",
               "--gpu-memory-utilization", "0.5", "--seed", "42"]
    manifest = {
        "model": str(Path(args.model).resolve()), "served_name": args.name,
        "gpu": args.gpu, "server_command": command, "evaluation_commands": [],
        "dataset": str(dataset.resolve()), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "n": len(rows), "runs": args.runs, "schema": args.schema,
        "inference": {"enable_thinking": False, "temperature": 0.1, "max_tokens": args.max_tokens,
                      "stop": ["<|im_end|>"] if args.schema == "hard_faults_v1" else None,
                      "server_seed": 42, "top_p": 0.95,
                      "top_p_source": "local model generation_config.json",
                      "arbitration": "off", "warmup_requests": 3},
        "repeat_note": ("eval_hard_model --seed is passed as the per-request sampling seed"
                        if args.schema == "hard_faults_v1" else
                        "eval_sft_model --seed is a run label, not a per-request sampling seed"),
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "starting",
    }
    generation_config = json.loads((Path(args.model) / "generation_config.json").read_text())
    if generation_config.get("top_p", 1.0) != 0.95:
        raise ValueError("expected top_p=0.95 for this comparison")
    manifest["model_generation_config"] = generation_config

    def save_manifest():
        (output / "evaluation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    started = time.perf_counter()
    server_env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
                      VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    url = f"http://127.0.0.1:{args.port}/v1"
    with (output / "server.log").open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(command, env=server_env, stdout=server_log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        manifest["server_pid"] = server.pid
        save_manifest()
        print(f"[serve] pid={server.pid} model={args.name} GPU={args.gpu}", flush=True)
        try:
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited with {server.returncode}; see {output / 'server.log'}")
                if time.perf_counter() - started > 900:
                    raise TimeoutError("vLLM readiness timed out after 900 seconds")
                try:
                    models = request_json(url + "/models")
                    if args.name in [item["id"] for item in models["data"]]:
                        manifest["models_response"] = models
                        break
                except (URLError, TimeoutError, ConnectionError):
                    pass
                time.sleep(1)
            manifest["startup_seconds"] = time.perf_counter() - started
            manifest["server_gpu_state"] = subprocess.run(
                ["nvidia-smi", "-i", args.gpu,
                 "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True).stdout.strip()
            print(f"[ready] startup={manifest['startup_seconds']:.1f}s; warmup=3", flush=True)
            for row in rows[:3]:
                request_json(url + "/chat/completions", {
                    "model": args.name, "messages": row["messages"][:-1],
                    "temperature": 0.1, "max_tokens": args.max_tokens,
                    **({"stop": ["<|im_end|>"]} if args.schema == "hard_faults_v1" else {}),
                    "chat_template_kwargs": {"enable_thinking": False},
                }, timeout=180)
            manifest["status"] = "evaluating"
            save_manifest()
            for repeat in range(1, args.runs + 1):
                result_path = output / f"run{repeat}.jsonl"
                evaluator = "tools/eval_hard_model.py" if args.schema == "hard_faults_v1" else "tests/eval_sft_model.py"
                eval_command = [sys.executable, evaluator, "--url", url,
                                "--model", args.name, "--dataset", args.dataset,
                                "--temperature", "0.1",
                                "--max-tokens", str(args.max_tokens), "--seed", str(repeat),
                                "--out", str(result_path), "--report", str(output / "report.md")]
                if args.schema == "legacy":
                    eval_command += ["--arbitration", "off"]
                manifest["evaluation_commands"].append(eval_command)
                save_manifest()
                print(f"[eval] repeat={repeat}/{args.runs}", flush=True)
                with (output / f"run{repeat}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(eval_command, env=cpu_env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                summary = json.loads(result_path.with_suffix(".summary.json").read_text())
                if summary["n"] != len(rows) or summary["errors"]:
                    raise RuntimeError(f"evaluation incomplete or endpoint errors: {result_path}")
                accuracy = summary["set_exact"] if args.schema == "hard_faults_v1" else summary["accuracy"]["fault_subset"]
                print(f"[result] repeat={repeat} accuracy={accuracy:.4%} "
                      f"wall={summary['wall_s']}s", flush=True)
            manifest["status"] = "completed"
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["wall_seconds"] = time.perf_counter() - started
            manifest["temporary_service_stopped"] = server.poll() is not None
            save_manifest()
    print(f"[done] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
