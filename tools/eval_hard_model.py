#!/usr/bin/env python3
"""Evaluate an eight-fault model directly against a local OpenAI-style endpoint."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hard_fault_metrics import HardFaultScorer, summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="data/sft_hard_v1/eval.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=640)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    dataset = Path(args.dataset)
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    scorer = HardFaultScorer(dataset.parent / "ontology.json")
    results = []
    start = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as out:
        for index, row in enumerate(rows):
            user = row["messages"][1]["content"]
            answer = json.loads(row["messages"][-1]["content"])
            gold = {key: answer[key] for key in ("root_causes", "needs_more_data")}
            payload = {"model": args.model, "messages": row["messages"][:-1],
                       "temperature": args.temperature, "top_p": 0.95, "seed": args.seed,
                       "max_tokens": args.max_tokens, "stop": ["<|im_end|>"],
                       "chat_template_kwargs": {"enable_thinking": False}}
            request = Request(args.url.rstrip("/") + "/chat/completions",
                              data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            response, error, text = {}, None, ""
            try:
                with urlopen(request, timeout=180) as stream:
                    response = json.load(stream)
                text = response["choices"][0]["message"]["content"] or ""
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            score = scorer.score(user, text, gold)
            result = {"index": index, "gold": gold, "raw_output": text, "score": asdict(score),
                      "latency_s": time.perf_counter() - t0, "error": error,
                      "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                      "usage": response.get("usage"), "meta": row["meta"],
                      "input_sha256": hashlib.sha256(json.dumps(row["messages"][:-1], ensure_ascii=False).encode()).hexdigest()}
            results.append(result)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            if (index + 1) % 20 == 0:
                print(f"{index + 1}/{len(rows)} evaluated", flush=True)
    summary = summarize(results, scorer.names)
    groups = {
        "compound": [r for r in results if len(r["meta"]["injected_faults"]) > 1],
        "missing_sensors": [r for r in results if r["meta"]["hidden_channels"]],
        "shifted": [r for r in results if r["meta"]["shifted_envelope"]],
        "unshifted": [r for r in results if not r["meta"]["shifted_envelope"]],
        "weak_signal": [r for r in results if r["meta"]["injected_faults"] and max(s["severity"] for s in r["meta"]["injected_faults"]) < 0.3],
    }
    summary.update({"groups": {k: summarize(v, scorer.names) for k, v in groups.items()},
                    "model": args.model, "seed": args.seed, "wall_s": time.perf_counter() - start,
                    "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    "metric_script_sha256": hashlib.sha256((ROOT / "tools/hard_fault_metrics.py").read_bytes()).hexdigest(),
                    "truncated": sum(r["finish_reason"] == "length" for r in results)})
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if args.report:
        args.report.write_text("# Eight-fault evaluation\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("groups", "per_class")}, ensure_ascii=False, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
