"""构建 L3 回归集（MS6 硬门槛 #9：历史线上 case 零退化）。

数据来源两路：
  1. **真实线上反馈**：data/feedback/*.jsonl 中**含 prompt** 的诊断快照
     （由 MS6 修复的 action/feedback.py 快照注入产生；历史数据无 prompt 故跳过）；
  2. **curated 硬 case 种子**：data/sft/eval.jsonl 的难例子集（boundary 混淆对 /
     transition 误报 / antihall 拒诊 / composite 主次因），作为线上样本积累前的
     基线守卫（含报告点名的两条 v3 判错 seed 901/951）。

产出 data/sft/regression.jsonl（与 eval.jsonl 同构：messages + meta），
供 tests/run_regression.py 对基线做零退化校验。

用法：
    conda run -n mff_agent python tools/build_regression_set.py \
        --eval data/sft/eval.jsonl --feedback data/feedback \
        --out data/sft/regression.jsonl
"""

import argparse
import json
from pathlib import Path

# curated 种子选取的难例类别（clear/misleading 属易/已充分覆盖，不入回归集）
HARD_KINDS = {"boundary", "transition", "antihall", "composite"}


def load_jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def from_feedback(fb_dir: Path):
    """从反馈库提取含 prompt 的真实线上 case -> 回归样本。"""
    cases = []
    for fp in sorted(fb_dir.glob("*.jsonl")):
        for rec in load_jsonl(fp):
            diag = rec.get("diagnosis", {}) or {}
            fb = rec.get("feedback", {}) or {}
            prompt = diag.get("prompt")
            gold = fb.get("actual_root_cause")
            if not prompt or not gold or not fb.get("is_true_fault"):
                continue
            cases.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": diag.get("output", "")},
                ],
                "meta": {"kind": "feedback", "gold": gold,
                         "source": f"feedback:{fp.name}",
                         "order_id": fb.get("order_id")},
            })
    return cases


def from_eval_seed(eval_path: Path):
    """从 eval.jsonl 选 curated 硬 case 作为基线守卫（确定性、可复现）。"""
    rows = load_jsonl(eval_path)
    seeds = []
    for r in rows:
        m = r.get("meta", {})
        if m.get("kind") in HARD_KINDS:
            r2 = dict(r)
            m2 = dict(m)
            m2["source"] = "curated_eval"
            r2["meta"] = m2
            seeds.append(r2)
    return seeds


def main():
    ap = argparse.ArgumentParser(description="构建 L3 回归集（MS6 #9）")
    ap.add_argument("--eval", default="data/sft/eval.jsonl")
    ap.add_argument("--feedback", default="data/feedback")
    ap.add_argument("--out", default="data/sft/regression.jsonl")
    args = ap.parse_args()

    fb_cases = from_feedback(Path(args.feedback))
    seed_cases = from_eval_seed(Path(args.eval))
    cases = fb_cases + seed_cases

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    from collections import Counter
    kinds = Counter(c["meta"]["kind"] for c in cases)
    print(f"回归集已写出: {out} 共 {len(cases)} 条")
    print(f"  真实反馈(含prompt): {len(fb_cases)} 条")
    print(f"  curated 种子: {len(seed_cases)} 条")
    print(f"  类别分布: {dict(kinds)}")
    if not fb_cases:
        print("  注：历史反馈无 prompt（MS6 前的快照缺陷），线上 case 将随新反馈自动积累。")


if __name__ == "__main__":
    main()
