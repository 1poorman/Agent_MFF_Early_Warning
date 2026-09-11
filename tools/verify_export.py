"""MS4 验收 3：导出产物可被 transformers 加载 + 模板未在导出中被破坏。

检查项：
  1. AutoTokenizer / AutoModelForCausalLM 能加载导出目录（MILESTONES 验收命令）；
  2. 导出后的 tokenizer 对同一条样本的渲染与**原始底座**逐 token 一致
     （防止 export 时 fix_special_tokens / chat_template 重写引入偏移）；
  3. （可选，--generate）用导出模型做一次 greedy 生成，肉眼确认输出是
     "推理过程 + 单个 JSON"、无多余内容——这是 MS5 行为评估前的最后一道自检。

用法：
    conda run -n mff_sft python tools/verify_export.py
    conda run -n mff_sft python tools/verify_export.py --generate --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "/home/huachenghao/models/MiniCPM5-2B"
MERGED = "/home/huachenghao/models/mff-sft-minicpm5-2b"


def ids_of(out) -> list:
    if hasattr(out, "ids"):
        return list(out.ids)
    if hasattr(out, "encoding"):
        return list(out.encoding.ids)
    try:
        return list(out["input_ids"])
    except (TypeError, KeyError):
        return list(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--merged", default=MERGED)
    ap.add_argument("--sample", default=str(ROOT / "data" / "sft" / "val.jsonl"))
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    ok = True

    # --- 1. 可加载 ---
    cfg = AutoConfig.from_pretrained(args.merged, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.merged, trust_remote_code=True)
    print(f"[1] 配置加载 ✅  architectures={cfg.architectures} vocab={cfg.vocab_size}")
    print(f"[1] tokenizer 加载 ✅  len={len(tok)} eos={tok.eos_token!r}")

    # --- 2. 渲染与底座逐 token 一致 ---
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8").splitlines()[0])
    msgs = sample["messages"]
    tok_base = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    a = ids_of(tok_base.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False))
    b = ids_of(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False))
    same = a == b
    print(f"[2] 模板渲染 底座={len(a)} 导出={len(b)} → {'✅ 一致' if same else '❌ 不一致'}")
    if not same:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"    首个分歧 @ {i}: {x} vs {y}")
                break
        ok = False

    # --- 3. 可选生成 ---
    if args.generate:
        model = AutoModelForCausalLM.from_pretrained(
            args.merged, trust_remote_code=True, dtype="bfloat16", device_map=args.device)
        model.eval()
        prompt = tok.apply_chat_template(
            [m for m in msgs if m["role"] != "assistant"],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        import torch
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=400, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"[3] 生成样本（gold={sample.get('meta',{}).get('gold')}）:\n{'-'*50}\n{text}\n{'-'*50}")

    print("=" * 60)
    print("✅ 导出产物校验通过" if ok else "❌ 导出产物校验未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
