"""Spark-X2.5-4B 冒烟：模板渲染 + 真机加载 + 一次 greedy 生成。

验证三件事：
  1. chat_template 在 enable_thinking=False 下渲染正常（对齐线上 vLLM 口径）；
  2. 权重能被 transformers 5.8.0（trust_remote_code）真实加载；
  3. 非思考模式能产出「推理 + 单个 JSON」。

用法：
    conda run -n mff_sft python tools/probe_spark.py --device cuda:1
    conda run -n mff_sft python tools/probe_spark.py --dry    # 只渲染模板，不加载权重
"""
import argparse
import json
import sys

MODEL = "/models/hch/Spark-X2.5-4B"
ROOT = "/home/huachenghao/codes/Agent_MFF_Early_Warning"


def to_ids(out) -> list:
    """兼容 transformers 5.x：apply_chat_template 返回类型不固定。"""
    if hasattr(out, "ids"):                      # Encoding
        return list(out.ids)
    if hasattr(out, "encoding"):                 # BatchEncoding(单条)
        return list(out.encoding.ids)
    try:                                         # BatchEncoding（UserDict）
        v = out["input_ids"]
        return list(v) if not (v and hasattr(v[0], "ids")) else list(v[0].ids)
    except (TypeError, KeyError, IndexError):
        pass
    if isinstance(out, list):
        if out and hasattr(out[0], "ids"):
            return list(out[0].ids)
        return list(out)
    raise TypeError(f"无法解析 apply_chat_template 返回类型: {type(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dry", action="store_true", help="只渲染模板，不加载权重")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f"tokenizer ✅ len={len(tok)} eos={tok.eos_token!r}({tok.eos_token_id})")

    with open(f"{ROOT}/data/sft/eval.jsonl", encoding="utf-8") as f:
        sample = json.loads(f.readline())
    msgs = sample["messages"]
    meta = sample.get("meta", {})
    print(f"样本 kind={meta.get('kind')} gold={meta.get('gold')}")

    pre = [m for m in msgs if m["role"] != "assistant"]
    for et in (False, True):
        ids = to_ids(tok.apply_chat_template(pre, tokenize=True,
                                             add_generation_prompt=True,
                                             enable_thinking=et))
        print(f"  enable_thinking={et}: prompt tokens={len(ids)} "
              f"尾4={tok.convert_ids_to_tokens(ids[-4:])}")

    infer_ids = to_ids(tok.apply_chat_template(pre, tokenize=True,
                                               add_generation_prompt=True,
                                               enable_thinking=False))
    print(f"线上口径 prompt 渲染 ✅ {len(infer_ids)} token "
          f"(MiniCPM5-2B 同口径约 1049)")

    if args.dry:
        return 0

    import torch
    from transformers import AutoModelForCausalLM

    print(f"加载权重（{args.device}）…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.bfloat16)
    model = model.to(args.device).eval()
    n = sum(p.numel() for p in model.parameters())
    vram = torch.cuda.memory_allocated(args.device) / 1e9
    print(f"模型 ✅ {type(model).__name__} params={n / 1e9:.3f}B "
          f"显存={vram:.2f}GB")

    prompt = tok.apply_chat_template(pre, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=False)
    enc = tok(prompt, return_tensors="pt").to(args.device)
    import time
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    dt = time.time() - t0
    n_new = out.shape[1] - enc["input_ids"].shape[1]
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"生成 ✅ {n_new} token / {dt:.1f}s ({n_new / max(dt, 1e-9):.1f} tok/s)")
    print("-" * 60)
    print(text[:1500])
    print("-" * 60)
    print(f"峰值显存 {torch.cuda.max_memory_allocated(args.device) / 1e9:.2f}GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
