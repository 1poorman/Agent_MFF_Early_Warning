"""S1 附加诊断：Spark 在「开思考」下的表现（对照关思考的塌缩现象）。

动机：S1 主口径关仲裁 + **关思考**（对齐线上 ≤5s 契约与 2B 基线口径），
但 Spark-X2.5 是思考型模型，若关闭思考导致它塌缩到单一标签，
则"非思考"这一前提对 Spark 不成立，级联契约需要重新评估。

本脚本在同 eval 集前 N 条上跑 thinking=True / thinking=False 对照，
输出逐条是否判对、输出分布、时延——用于判定塌缩是否为模式问题。

用法：
    conda run -n mff_agent python tools/spark_thinking_probe.py --n 30
"""
import argparse
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def post(ep, payload, timeout=300.0):
    req = urllib.request.Request(
        ep, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_rc(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0)).get("root_cause")
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3762/v1")
    ap.add_argument("--model", default="Spark-X2.5-4B")
    ap.add_argument("--n", type=int, default=0,
                    help=">0 时取前 N 条；=0 时按类别分层抽样")
    ap.add_argument("--per-kind", type=int, default=8,
                    help="分层抽样时每类取几条")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--modes", default="off,on",
                    help="测哪些模式：off=关思考 / on=开思考，逗号分隔")
    args = ap.parse_args()
    ep = args.url.rstrip("/") + "/chat/completions"
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    rows = [json.loads(l) for l in
            open(ROOT / "data" / "sft" / "eval.jsonl", encoding="utf-8") if l.strip()]
    if args.n > 0:
        samples = rows[:args.n]
    else:
        # 分层抽样：覆盖 transition（应判无故障）与 antihall（应判数据异常），
        # 这两类是判断"塌缩是否为模式问题"最敏感的。
        by_kind = {}
        for r in rows:
            by_kind.setdefault(r["meta"]["kind"], []).append(r)
        samples = [r for k in sorted(by_kind) for r in by_kind[k][:args.per_kind]]

    for mode in modes:
        et = (mode == "on")
        ok = 0
        dist, lat, kinds = Counter(), [], Counter()
        print(f"\n===== enable_thinking={et} (n={len(samples)}) =====")
        for i, s in enumerate(samples):
            msgs = {m["role"]: m["content"] for m in s["messages"]}
            gold = s.get("meta", {}).get("gold")
            t0 = time.time()
            try:
                out = post(ep, {
                    "model": args.model,
                    "messages": [{"role": "system", "content": msgs.get("system", "")},
                                 {"role": "user", "content": msgs["user"]}],
                    "max_tokens": args.max_tokens, "temperature": 0.1,
                    "chat_template_kwargs": {"enable_thinking": et},
                })
            except Exception as e:
                print(f"  [{i}] 端点异常 {type(e).__name__}: {e}")
                continue
            dt = time.time() - t0
            lat.append(dt)
            text = out["choices"][0]["message"].get("content") or ""
            usage = out.get("usage", {})
            rc = parse_rc(text)
            dist[rc] += 1
            kinds[s.get("meta", {}).get("kind")] += 1
            hit = rc == gold
            ok += hit
            if i % 5 == 0 or not hit:
                print(f"  [{i:3d}] kind={s['meta'].get('kind'):10s} gold={gold} "
                      f"raw={rc} {'✅' if hit else '❌'} {dt:.1f}s "
                      f"({usage.get('completion_tokens')} tok)")
        n = len(lat)
        print(f"  --- 准确率 {ok}/{n} = {ok / max(n, 1):.1%} | "
              f"时延 均值{sum(lat) / max(n, 1):.1f}s 最大{max(lat) if lat else 0:.1f}s")
        print(f"  --- 输出分布: {dict(dist.most_common(8))}")


if __name__ == "__main__":
    main()
