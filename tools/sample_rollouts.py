#!/usr/bin/env python3
"""在 GPU0 临时起 vLLM，对 RL prompt 池做 k 采样并用统一奖励逐条打分。

产出 data/rl/rollouts.jsonl，是 DPO 偏好对与 RM 训练集的共同上游。

口径要点
--------
* **服务同构**：与 tools/run_adapter_eval.py:113-117 一样走 /chat/completions +
  `chat_template_kwargs={"enable_thinking": false}`，由 vLLM 套模板。训练侧的
  纯文本 prompt 由 tools/check_rl_template.py 断言与此逐 token 相同。
* **采样温度 1.0（不是评估用的 0.1）**：偏好学习需要组内方差。温度太低会让
  8 条 rollout 几乎一致，产不出 margin ≥ 0.15 的对子，G1 就会卡住。
* 一次请求取 `n=k`，比发 k 次请求省掉 k-1 次 prefill。
* 逐条追加写盘，中途崩溃不丢已采样本。
* 服务由本脚本启停，`finally` 里 killpg 保证不残留（与既有评估脚本同模式）。

**不碰 svc.diagnose()**：`.env` 的 `cascade_enabled: true` 会让 27B 兜底静默
污染结果（docs/SFT_STATUS.md:34 记载过基线因此作废）。本脚本只打 vLLM 原生
HTTP 接口，不经过 server.service。

用法：
    conda run --no-capture-output -n mff_agent python tools/sample_rollouts.py \
        --model saves/mff-lora-minicpm5-2b-20260923-merged \
        --name mff-lora-sampler --gpu 0 --port 3764 \
        --prompts data/rl/prompts.jsonl \
        --out-dir logs/rl_minicpm5_20260924/rollouts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.rl_reward import RewardScorer  # noqa: E402


def request_json(url: str, payload: dict | None = None, timeout: float = 5):
    req = Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="saves/mff-lora-minicpm5-2b-20260923-merged")
    ap.add_argument("--name", default="mff-lora-sampler")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=3764)
    ap.add_argument("--prompts", default="data/rl/prompts.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("-k", "--num-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="冒烟用：只采前 N 条")
    ap.add_argument("--vllm", default="/home/huachenghao/.conda/envs/mff_sft_serve/bin/vllm")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有采样目录：{out}")
    with socket.socket() as probe:          # 端口被占先失败，别白等模型加载
        probe.bind(("127.0.0.1", args.port))

    prompts_path = Path(args.prompts)
    rows = [json.loads(l) for l in prompts_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print(f"prompt 池为空：{prompts_path}", file=sys.stderr)
        return 2

    command = [args.vllm, "serve", str(Path(args.model).resolve()),
               "--served-model-name", args.name, "--dtype", "bfloat16",
               "--host", "127.0.0.1", "--port", str(args.port),
               "--max-model-len", "8192", "--max-num-seqs", str(args.max_num_seqs),
               "--gpu-memory-utilization", "0.5", "--seed", "42"]
    sampling = {"n": args.num_samples, "temperature": args.temperature,
                "top_p": args.top_p, "max_tokens": args.max_tokens,
                "enable_thinking": False}
    manifest = {
        "model": str(Path(args.model).resolve()), "served_name": args.name,
        "gpu": args.gpu, "server_command": command,
        "prompts": str(prompts_path.resolve()),
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "n_prompts": len(rows), "sampling": sampling,
        "concurrency": args.concurrency,
        "sampling_note": "温度 1.0 远高于评估用 0.1：偏好学习需要组内方差",
        "cascade": "off (直连 vLLM HTTP，不经 server.service)",
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "starting",
    }

    def save_manifest():
        (out / "sampling_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 每线程一个 scorer：AntiHallucinationChecker 未声明线程安全，16 路并发共享
    # 一个实例是在赌它没有可变状态 —— 奖励算错不会报错，只会静默训错，不值得赌。
    _local = threading.local()

    def get_scorer() -> RewardScorer:
        if not hasattr(_local, "scorer"):
            _local.scorer = RewardScorer()
        return _local.scorer

    url = f"http://127.0.0.1:{args.port}/v1"
    server_env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
                      VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
    started = time.perf_counter()
    done = 0
    rollout_path = out / "rollouts.jsonl"

    def sample_one(item):
        i, row = item
        msgs = [m for m in row["messages"] if m["role"] != "assistant"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        gold = row["meta"]["gold"]
        payload = {"model": args.name, "messages": msgs,
                   "n": args.num_samples, "temperature": args.temperature,
                   "top_p": args.top_p, "max_tokens": args.max_tokens,
                   "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.perf_counter()
        try:
            resp = request_json(url + "/chat/completions", payload, timeout=600)
        except Exception as exc:
            return {"idx": i, "error": f"{type(exc).__name__}: {exc}",
                    "kind": row["meta"].get("kind"), "gold": gold}
        latency = time.perf_counter() - t0

        samples = []
        scorer = get_scorer()
        for choice in resp.get("choices", []):
            text = (choice.get("message") or {}).get("content") or ""
            s = scorer.score(user, text, gold)
            samples.append({
                "text": text, "reward": round(s.total, 6), "parts": s.parts,
                "root_cause": s.root_cause, "confidence": s.confidence,
                "correct": s.correct, "json_ok": s.json_ok,
                "faith_ok": s.faith_ok, "hallu_ok": s.hallu_ok,
                "finish_reason": choice.get("finish_reason"),
            })
        rewards = [x["reward"] for x in samples]
        stats = {
            "n": len(rewards),
            "mean": round(statistics.fmean(rewards), 6) if rewards else None,
            "std": round(statistics.pstdev(rewards), 6) if len(rewards) > 1 else 0.0,
            "min": min(rewards) if rewards else None,
            "max": max(rewards) if rewards else None,
            "span": round(max(rewards) - min(rewards), 6) if rewards else None,
            "n_correct": sum(1 for x in samples if x["correct"]),
            "n_json_ok": sum(1 for x in samples if x["json_ok"]),
            "latency_s": round(latency, 3),
        }
        return {"idx": i, "kind": row["meta"].get("kind"), "gold": gold,
                "seed": row["meta"].get("seed"), "messages": msgs,
                "samples": samples, "stats": stats}

    with (out / "server.log").open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(command, env=server_env, stdout=server_log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        manifest["server_pid"] = server.pid
        save_manifest()
        print(f"[serve] pid={server.pid} model={args.name} GPU={args.gpu}", flush=True)
        try:
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM 退出码 {server.returncode}，见 {out/'server.log'}")
                if time.perf_counter() - started > 900:
                    raise TimeoutError("vLLM 就绪超时 900s")
                try:
                    models = request_json(url + "/models")
                    if args.name in [m["id"] for m in models["data"]]:
                        break
                except (URLError, TimeoutError, ConnectionError):
                    pass
                time.sleep(1)
            manifest["startup_seconds"] = round(time.perf_counter() - started, 2)
            manifest["status"] = "sampling"
            save_manifest()
            print(f"[ready] startup={manifest['startup_seconds']:.1f}s "
                  f"n={len(rows)} k={args.num_samples}", flush=True)

            results: list[dict] = []
            with rollout_path.open("w", encoding="utf-8") as sink, \
                    ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for res in pool.map(sample_one, enumerate(rows)):
                    results.append(res)
                    sink.write(json.dumps(res, ensure_ascii=False) + "\n")
                    sink.flush()
                    done += 1
                    if done % 25 == 0 or done == len(rows):
                        el = time.perf_counter() - started
                        spans = [r["stats"]["span"] for r in results
                                 if "error" not in r and r["stats"]["span"] is not None]
                        tail = (f" mean_span={statistics.fmean(spans):.4f}"
                                f" varied={sum(1 for s in spans if s >= 0.15)}"
                                if spans else "")
                        print(f"  [{done}/{len(rows)}] {el:.0f}s{tail}", flush=True)

            ok = [r for r in results if "error" not in r]
            errs = [r for r in results if "error" in r]
            spans = [r["stats"]["span"] for r in ok]
            all_rewards = [x["reward"] for r in ok for x in r["samples"]]
            summary = {
                "n_prompts": len(results), "n_ok": len(ok), "n_error": len(errs),
                "n_samples_total": len(all_rewards),
                "reward_mean": round(statistics.fmean(all_rewards), 6) if all_rewards else None,
                "reward_std": round(statistics.pstdev(all_rewards), 6) if len(all_rewards) > 1 else None,
                "span_mean": round(statistics.fmean(spans), 6) if spans else None,
                # G1 的先行指标：只有 span ≥ 0.15 的组才产得出偏好对
                "groups_span_ge_0.15": sum(1 for s in spans if s is not None and s >= 0.15),
                "groups_zero_variance": sum(1 for s in spans if s == 0.0),
                "by_kind": {},
            }
            for kind in sorted({r.get("kind") for r in ok}):
                sub = [r for r in ok if r.get("kind") == kind]
                ss = [r["stats"]["span"] for r in sub]
                summary["by_kind"][kind] = {
                    "n": len(sub),
                    "span_mean": round(statistics.fmean(ss), 6) if ss else None,
                    "span_ge_0.15": sum(1 for s in ss if s >= 0.15),
                    "mean_correct_rate": round(
                        statistics.fmean([r["stats"]["n_correct"] / max(r["stats"]["n"], 1)
                                          for r in sub]), 4) if sub else None,
                }
            manifest["summary"] = summary
            manifest["status"] = "completed" if not errs else "completed_with_errors"
            print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
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
            manifest["wall_seconds"] = round(time.perf_counter() - started, 2)
            manifest["temporary_service_stopped"] = server.poll() is not None
            save_manifest()

    print(f"\n[done] {rollout_path}")
    print(f"下一步：conda run -n mff_agent python tools/build_preference_data.py "
          f"--rollouts {rollout_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
