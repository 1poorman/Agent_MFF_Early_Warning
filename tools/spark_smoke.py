"""S0.2 验收：Spark 服务的 raw 冒烟（探活 / enable_thinking 透传 / 一次真实诊断样本）。

用法：
    conda run -n mff_agent python tools/spark_smoke.py
    conda run -n mff_agent python tools/spark_smoke.py --url http://localhost:3762/v1

只依赖标准库 urllib，避免把 mff_agent 的环境差异掺进来。
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3762/v1")
    ap.add_argument("--model", default="Spark-X2.5-4B")
    args = ap.parse_args()
    ep = args.url.rstrip("/") + "/chat/completions"

    # --- 1. 基础问答 + 非思考模式透传 ---
    for et in (False, True):
        t0 = time.time()
        try:
            out = post(ep, {
                "model": args.model,
                "messages": [{"role": "user", "content": "用一句话说明什么是水泵气蚀。"}],
                "max_tokens": 128, "temperature": 0.1,
                "chat_template_kwargs": {"enable_thinking": et},
            })
        except Exception as e:
            print(f"[FAIL] enable_thinking={et}: {type(e).__name__}: {e}")
            continue
        ch = out["choices"][0]["message"]
        content = (ch.get("content") or "").strip()
        reasoning = (ch.get("reasoning_content") or "").strip()
        dt = time.time() - t0
        print(f"[OK] enable_thinking={et} {dt:.1f}s | "
              f"reasoning={len(reasoning)}字 content={len(content)}字")
        print(f"     content: {content[:160]!r}")
        if not et:
            # 非思考模式：不应产出 reasoning_content
            print(f"     非思考模式无思考链: {'✅' if not reasoning else '❌ ' + reasoning[:80]!r}")

    # --- 2. 真实诊断样本（走退化版 prompt，同 eval 集） ---
    import re
    with open(ROOT / "data" / "sft" / "eval.jsonl", encoding="utf-8") as f:
        sample = json.loads(f.readline())
    msgs = {m["role"]: m["content"] for m in sample["messages"]}
    gold = sample.get("meta", {}).get("gold")
    t0 = time.time()
    try:
        out = post(ep, {
            "model": args.model,
            "messages": [{"role": "system", "content": msgs.get("system", "")},
                         {"role": "user", "content": msgs["user"]}],
            "max_tokens": 1000, "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=300.0)
    except Exception as e:
        print(f"[FAIL] 诊断样本: {type(e).__name__}: {e}")
        return 1
    dt = time.time() - t0
    text = out["choices"][0]["message"].get("content") or ""
    usage = out.get("usage", {})
    print(f"\n[OK] 诊断样本 {dt:.1f}s | gold={gold} | "
          f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")

    m = re.search(r"\{.*\}", text, re.S)
    parsed = None
    if m:
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            print(f"     JSON 解析失败: {e}")
    print(f"     JSON 可解析: {'✅' if parsed else '❌'}")
    if parsed:
        print(f"     root_cause={parsed.get('root_cause')!r} "
              f"confidence={parsed.get('confidence')} "
              f"evidence={len(parsed.get('evidence') or [])}条 "
              f"sop={len(parsed.get('sop') or [])}条")
        print(f"     判对: {'✅' if parsed.get('root_cause') == gold else '❌'}")
    print(f"     输出前 300 字: {text[:300]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
