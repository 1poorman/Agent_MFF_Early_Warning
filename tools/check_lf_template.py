"""MS4 L1 模板 sanity check：LLaMA-Factory 渲染 vs 模型自带 chat_template 字节级比对。

为什么必须做（docs/SFT_MILESTONES.md MS4 验收标准 2）：
    模板错配时 loss 照样下降、照样出模型，但输出格式与线上不一致，评估才暴露。
    LLaMA-Factory cookbook 明示禁止 empty/llama3 模板即此故。

结论口径：
    1) `template: minicpm5` 的完整 token 序列应与 `tokenizer.apply_chat_template(
       messages, add_generation_prompt=False)` 逐 token 相同；
    2) `enable_thinking: false` 时，空思考块 `<think>\\n\\n</think>\\n\\n` 落在
       **prompt**（不计 loss），与线上 LLMClient(enable_thinking=False)
       → vLLM chat_template_kwargs={enable_thinking:false} 的推理 prompt 一致；
    3) 因此无需 `minicpm5_nothink`（该模板名在 LLaMA-Factory 中未注册）。

用法：
    conda run -n mff_sft python tools/check_lf_template.py
    conda run -n mff_sft python tools/check_lf_template.py --n 5
"""

import argparse
import json
import sys
from pathlib import Path

MODEL = "/home/huachenghao/models/MiniCPM5-2B"
ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "sft" / "train.jsonl"
LF_REPO = Path.home() / "codes" / "LLaMA-Factory"

sys.path.insert(0, str(LF_REPO / "src"))


def to_ids(out) -> list:
    """兼容 transformers 5.x：apply_chat_template(tokenize=True) 可能返回
    BatchEncoding / Encoding / list[int] / list[Encoding]。"""
    if hasattr(out, "ids"):                     # Encoding
        return list(out.ids)
    if hasattr(out, "encoding"):                # BatchEncoding(单条)
        return list(out.encoding.ids)
    try:                                        # BatchEncoding（UserDict，非 dict 子类）
        return list(out["input_ids"])
    except (TypeError, KeyError):
        pass
    if isinstance(out, list):
        if out and hasattr(out[0], "ids"):
            return list(out[0].ids)
        return list(out)
    raise TypeError(f"无法解析 apply_chat_template 返回类型: {type(out)}")


def compare(name, a, b, tok):
    same = a == b
    print(f"  [{name}] token 数 LF={len(a)} 原生={len(b)} → {'✅ 完全一致' if same else '❌ 不一致'}")
    if not same:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"    首个分歧 @ {i}: LF={x}({tok.convert_ids_to_tokens(x)!r}) "
                      f"原生={y}({tok.convert_ids_to_tokens(y)!r})")
                break
        print(f"    LF  尾20: {tok.convert_ids_to_tokens(a[-20:])}")
        print(f"    原生尾20: {tok.convert_ids_to_tokens(b[-20:])}")
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="抽检样本数")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--train", default=str(TRAIN))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments

    samples = []
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            if len(samples) >= args.n:
                break
            samples.append(json.loads(line))

    # --- 原生 tokenizer（pristine，先渲染，避免被 LF 的 fix_special_tokens 改写 eos） ---
    tok_native = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    native = []
    for s in samples:
        msgs = s["messages"]
        full = to_ids(tok_native.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False))
        # 线上推理口径：只在 system+user 上开 generation prompt，并关闭思考
        pre = [m for m in msgs if m["role"] != "assistant"]
        infer = to_ids(tok_native.apply_chat_template(
            pre, tokenize=True, add_generation_prompt=True, enable_thinking=False))
        native.append((full, infer))

    # --- LLaMA-Factory 口径（另起一个 tokenizer 实例，隔离 fix_special_tokens 副作用） ---
    tok_lf = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    results = {}
    for et in (False, True):
        data_args = DataArguments(template="minicpm5", enable_thinking=et)
        template = get_template_and_fix_tokenizer(tok_lf, data_args)
        pairs = []
        for s in samples:
            msgs = s["messages"]
            system = ""
            rest = msgs
            if msgs and msgs[0]["role"] == "system":   # 与 SharegptDatasetConverter 同款抽取
                system, rest = msgs[0]["content"], msgs[1:]
            prompt_ids, response_ids = template.encode_oneturn(tok_lf, rest, system=system)
            pairs.append((prompt_ids, response_ids))
        results[et] = pairs

    print(f"模型: {args.model}")
    print(f"样本: {args.train} 前 {len(samples)} 条")
    print(f"LF 模板: minicpm5 (ReasoningTemplate)  thought_words={template.thought_words}")
    print(f"eos 处理后: {tok_lf.eos_token!r} (id={tok_lf.eos_token_id})\n")

    ok_all = True
    for i, s in enumerate(samples):
        full, infer = native[i]
        kind = s.get("meta", {}).get("kind")
        print(f"--- 样本 {i} (kind={kind}) ---")
        for et in (False, True):
            prompt_ids, response_ids = results[et][i]
            lf_full = prompt_ids + response_ids
            tag = f"enable_thinking={et}"
            ok = compare(f"完整序列 {tag}", lf_full, full, tok_native)
            ok_all &= ok
            if et is False:
                print(f"    loss 边界: prompt={len(prompt_ids)} response={len(response_ids)} "
                      f"(自 response 起算 loss)")
                # 线上推理 prompt 应与 LF 的 prompt 完全一致（含空思考块）
                same_prompt = prompt_ids == infer
                print(f"    与线上推理 prompt 一致: {'✅' if same_prompt else '❌'} "
                      f"(LF={len(prompt_ids)} 线上={len(infer)})")
                ok_all &= same_prompt
        print(f"    尾 6 token: {tok_native.convert_ids_to_tokens(full[-6:])}")
        print()

    print("=" * 60)
    print("✅ 模板 sanity check 通过：LLaMA-Factory 渲染与模型原生模板逐 token 一致，"
          "且 enable_thinking=false 与线上推理 prompt 等价" if ok_all else
          "❌ 模板 sanity check 未通过——禁止开训（模板错配会静默产出坏模型）")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
