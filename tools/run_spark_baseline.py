"""S1：Spark-X2.5-4B 零成本原生基线的 3 次复跑驱动。

口径与 MiniCPM5-2B 基线**严格同源**，保证可比：
  - 数据集 `data/sft/eval.jsonl`（退化版 prompt = 训练分布，200 条）
  - `--arbitration off`（关仲裁，主口径）
  - `tests/eval_sft_model.py` 原样复用，一行不改

对照锚点（`data/sft/baseline_degraded_results.summary.json`）：
  MiniCPM5-2B 原生 故障子集 20.6% / 混淆对 30.0%

用法：
    conda run -n mff_agent python tools/run_spark_baseline.py \
        --url http://localhost:3762/v1 --model Spark-X2.5-4B
    conda run -n mff_agent python tools/run_spark_baseline.py --check   # 只探活
"""
import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "tests" / "eval_sft_model.py"
REPORT = ROOT / "design" / "SFT_SPARK_BASELINE.md"


def alive(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/models", timeout=10) as r:
            body = json.loads(r.read())
        names = [m.get("id") for m in body.get("data", [])]
        print(f"端点存活 ✅ {url} -> {names}")
        return True
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"端点不可用 ❌ {url}: {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3762/v1")
    ap.add_argument("--model", default="Spark-X2.5-4B")
    ap.add_argument("--dataset", default=str(ROOT / "data" / "sft" / "eval.jsonl"))
    ap.add_argument("--runs", type=int, default=3, help="复跑次数（测试方案 §8 要求 3 次）")
    ap.add_argument("--check", action="store_true", help="只探活，不跑评估")
    args = ap.parse_args()

    if not alive(args.url):
        return 1
    if args.check:
        return 0

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if not REPORT.exists():
        REPORT.write_text(
            "# Spark-X2.5-4B 零成本原生基线（S1）\n\n"
            f"> 数据集 `data/sft/eval.jsonl`（200 条，退化版 prompt）· 关仲裁 · "
            f"3 次复跑\n"
            f"> 对照锚点：MiniCPM5-2B 原生 **故障子集 20.6% / 混淆对 30.0%**\n"
            f"> （`data/sft/baseline_degraded_results.summary.json`，MS3 基线复算）\n",
            encoding="utf-8")

    rc_all = 0
    for seed in range(1, args.runs + 1):
        out = ROOT / "logs" / f"spark_baseline_{seed}.jsonl"
        cmd = [
            sys.executable, str(EVAL),
            "--url", args.url, "--model", args.model,
            "--dataset", args.dataset, "--arbitration", "off",
            "--seed", str(seed), "--out", str(out),
            "--report", str(REPORT),
        ]
        print(f"\n===== 复跑 {seed}/{args.runs} =====", flush=True)
        print(" ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        rc_all |= rc
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
