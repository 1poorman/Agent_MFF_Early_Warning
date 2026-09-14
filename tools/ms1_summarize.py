"""MS1 基线明细汇总（可复用于任意 models 的 ms1_baseline*.jsonl）。

用法：
    conda run -n mff_agent python tools/ms1_summarize.py data/sft/ms1_baseline_results.jsonl
    conda run -n mff_agent python tools/ms1_summarize.py \
        data/sft/ms1_baseline_results.jsonl data/sft/ms1_baseline_spark.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

GOLDS = ["水泵气蚀", "管道泄漏", "线圈结垢", "过滤器堵塞"]


def summarize(path: str) -> dict:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    faults = [r for r in rows if r.get("kind") == "fault"]
    llm = [r for r in faults if r.get("llm_called")]
    normals = [r for r in rows if r.get("kind") == "normal"]
    lats = [r["llm_latency_s"] for r in llm if r.get("llm_latency_s")]
    return {
        "path": path,
        "n_total": len(rows),
        "n_fault": len(faults),
        "n_llm": len(llm),
        "acc_raw": sum(r.get("rc_raw") == r["gold"] for r in llm) / max(len(llm), 1),
        "acc_arb": sum(r.get("rc_arbitrated") == r["gold"] for r in llm) / max(len(llm), 1),
        "json_rate": sum(bool(r.get("json_ok")) for r in llm) / max(len(llm), 1),
        "lat_mean": (sum(lats) / len(lats)) if lats else None,
        "by_gold": {g: (lambda s: (len(s), sum(
            r.get("rc_raw") == g for r in s) / max(len(s), 1)))([
                r for r in llm if r["gold"] == g]) for g in GOLDS},
        "confusion": {g: dict(Counter(
            r.get("rc_raw") for r in llm if r["gold"] == g)) for g in GOLDS},
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for p in sys.argv[1:]:
        if not Path(p).exists():
            print(f"!! 文件不存在: {p}")
            continue
        m = summarize(p)
        print("=" * 72)
        print(f"{m['path']}  (总 {m['n_total']} 行 / 故障 {m['n_fault']} / 有LLM输出 {m['n_llm']})")
        print(f"  关仲裁根因准确率: {m['acc_raw']:.1%}")
        print(f"  含仲裁准确率(参考): {m['acc_arb']:.1%}")
        print(f"  JSON 合法率: {m['json_rate']:.1%}")
        if m["lat_mean"] is not None:
            print(f"  LLM 平均时延: {m['lat_mean']:.2f}s")
        print("  分故障类别（关仲裁）:")
        for g, (n, a) in m["by_gold"].items():
            if n:
                print(f"    {g}: {a:.1%} (n={n})")
        print("  混淆矩阵（关仲裁 raw → gold）:")
        for g in GOLDS:
            if m["confusion"][g]:
                print(f"    {g} ← {m['confusion'][g]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
