"""L3 回归集零退化校验（MS6 硬门槛 #9）。

对 data/sft/regression.jsonl 逐条请求端点（**关仲裁**，对齐 SFT 主口径），
与金标签比对，并与已记录基线做差：
  - 首次运行：建立基线 data/sft/regression.baseline.json；
  - 后续运行：任一 case 由「通过」翻转为「不通过」即判为**退化**（退出码 1）。

零退化定义：相对基线无 pass->fail 翻转（新增 case 仅报告、不算退化）。

用法：
    conda run -n mff_agent python tests/run_regression.py \
        --url http://localhost:3761/v1 --model mff-sft-minicpm5-2b
    # 重置基线（模型/数据升级后）：加 --update-baseline
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from reasoning.llm_client import LLMClient  # noqa: E402
from reasoning.root_cause import RootCauseReasoner  # noqa: E402


def load_jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def case_id(idx, meta):
    return meta.get("order_id") or f"{meta.get('source', 'seed')}:{idx}"


def run(args):
    rows = load_jsonl(Path(args.dataset))
    if not rows:
        print(f"[SKIP] 回归集为空或不存在: {args.dataset}（先跑 tools/build_regression_set.py）")
        return 0

    llm = LLMClient(config={"url": args.url, "key": "empty",
                            "big_model_name": args.model,
                            "enable_thinking": False, "timeout": args.timeout})
    parse = RootCauseReasoner(llm=llm)._parse

    results = {}          # case_id -> {"pass": bool, "gold":.., "pred":.., "kind":..}
    for i, r in enumerate(rows):
        msgs = {m["role"]: m["content"] for m in r["messages"]}
        system, user = msgs.get("system", ""), msgs.get("user", "")
        meta = r.get("meta", {})
        gold = meta.get("gold")
        try:
            raw = llm.chat(user, system=system, max_tokens=args.max_tokens,
                           temperature=args.temperature)
        except Exception as e:
            raw = ""
            print(f"  [{i}] 请求异常: {type(e).__name__}: {e}")
        parsed = parse(raw) if raw else None
        pred = parsed.get("root_cause") if parsed else None
        ok = pred == gold
        results[case_id(i, meta)] = {"pass": ok, "gold": gold, "pred": pred,
                                     "kind": meta.get("kind")}

    n = len(results)
    passed = sum(1 for v in results.values() if v["pass"])
    print(f"\n回归集: {args.dataset} | {n} 条 | 通过 {passed}/{n} = {passed/max(n,1):.1%}")

    base_path = Path(args.baseline)
    baseline = json.loads(base_path.read_text(encoding="utf-8")) if base_path.is_file() else None

    if baseline is None or args.update_baseline:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(
            {k: v["pass"] for k, v in results.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        action = "已更新" if args.update_baseline else "首次建立"
        print(f"[BASELINE {action}] {base_path}（{passed}/{n} 通过）")
        print("回归集基线就绪，后续运行将做零退化校验 ✔")
        return 0

    # 与基线做差：pass -> fail 翻转即退化
    regressions = [(k, results[k]) for k, was in baseline.items()
                   if was and k in results and not results[k]["pass"]]
    new_cases = [k for k in results if k not in baseline]
    if regressions:
        print(f"\n[FAIL] 检测到 {len(regressions)} 条退化（基线通过→现不通过）:")
        for k, v in regressions:
            print(f"  - {k} [{v['kind']}] gold={v['gold']} pred={v['pred']}")
        return 1
    print(f"[PASS] 零退化：相对基线无 pass→fail 翻转"
          + (f"；新增 {len(new_cases)} 条 case（未纳入退化判定）" if new_cases else ""))
    return 0


def main():
    ap = argparse.ArgumentParser(description="L3 回归集零退化校验（MS6 #9）")
    ap.add_argument("--url", default="http://localhost:3761/v1")
    ap.add_argument("--model", default="mff-sft-minicpm5-2b")
    ap.add_argument("--dataset", default="data/sft/regression.jsonl")
    ap.add_argument("--baseline", default="data/sft/regression.baseline.json")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
