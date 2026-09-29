#!/usr/bin/env python3
"""G0 决策门第 2 项：RL 训练侧 prompt 与线上服务侧 prompt **逐 token 相同**。

为什么必须查
------------
线上（tools/run_adapter_eval.py:113-117）发的是 messages +
`chat_template_kwargs={"enable_thinking": false}`，由 vLLM 套模板。
而 TRL 的 `maybe_apply_chat_template` 只认 `tools`/`documents` 等固定 kwargs，
**不会透传 `enable_thinking`**。若把会话格式直接喂给 TRL，渲染出的 prompt
会与线上不同（MiniCPM5 模板在 enable_thinking 开关下会插入不同的思考前缀），
训练与服务就此错位 —— 这正是本项目 MS4 吃过的同类亏
（docs/SFT_STATUS.md:52/77）。

对策：**预渲染成纯文本** `prompt` 列喂 TRL，由本脚本断言：

  1. 预渲染文本 tokenize（add_special_tokens=False）== 模板 tokenize=True 的 ids
     —— 这是「训练看到的 token」与「模板真源」的逐 token 一致性。
  2. add_special_tokens=True 会不会多插 BOS（双 BOS 是最常见的静默错位）。
     若会，训练侧必须用 add_special_tokens=False，本脚本把结论打出来。
  3. enable_thinking True/False 渲染结果**确实不同** —— 反证该 kwarg 真被模板
     读取；若两者相同，说明模板没实现该开关，前面的断言就失去意义。

用法（在 mff_rl 里跑，只读模型目录）：
    /home/huachenghao/.conda/envs/mff_rl/bin/python tools/check_rl_template.py \
        --dataset data/rl/prompts.jsonl --n 0
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

BASE_MODEL = "/home/huachenghao/models/MiniCPM5-2B"


def _ids(value) -> list[int]:
    """把 apply_chat_template(tokenize=True) 的返回值归一成扁平 id 列表。

    transformers 5.x 默认返回 BatchEncoding（含 input_ids/attention_mask），
    直接 len() 得到的是 **键的个数 2**，会把 1200 条全判成不一致。
    与 tools/train_lora.py:58 `_ids()` 同一套归一逻辑。
    """
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "ids"):
        value = value.ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def render(tokenizer, messages, enable_thinking: bool = False) -> str:
    """与线上同口径的 prompt 渲染（add_generation_prompt=True）。"""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/rl/prompts.jsonl")
    ap.add_argument("--gold-dataset", default="data/rl/prompts_gold.jsonl",
                    help="带金标 assistant 的同批样本，用于校验 completion 侧收尾 token")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--n", type=int, default=0, help="0 = 全量")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(l) for l in Path(args.dataset).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if args.n:
        rows = rows[:args.n]
    if not rows:
        print(f"数据集为空：{args.dataset}")
        return 2

    fails: list[str] = []
    double_bos = 0
    thinking_differs = 0

    for i, row in enumerate(rows):
        msgs = row.get("messages")
        if msgs is None:
            fails.append(f"row{i}: 缺 messages 字段")
            continue
        # 只取 system+user（去掉金标 assistant），与采样/训练时一致
        msgs = [m for m in msgs if m["role"] != "assistant"]

        text = render(tok, msgs, enable_thinking=False)
        ref_ids = _ids(tok.apply_chat_template(msgs, tokenize=True,
                                               add_generation_prompt=True,
                                               enable_thinking=False))
        ids_plain = _ids(tok(text, add_special_tokens=False)["input_ids"])
        ids_special = _ids(tok(text, add_special_tokens=True)["input_ids"])

        # 断言 1：预渲染纯文本的 token 与模板真源逐 token 相同
        if ids_plain != ref_ids:
            n = min(len(ids_plain), len(ref_ids))
            at = next((k for k in range(n) if ids_plain[k] != ref_ids[k]), n)
            fails.append(f"row{i}: token 不一致 len={len(ids_plain)}/{len(ref_ids)} 首个差异@{at}")

        # 断言 2：add_special_tokens=True 是否多插前缀（双 BOS）
        if ids_special != ids_plain:
            double_bos += 1

        # 断言 3：enable_thinking 开关确实影响渲染
        if render(tok, msgs, enable_thinking=True) != text:
            thinking_differs += 1

        # 预渲染字段若已存在，必须与现算结果一致（防数据集与模板漂移）
        if "prompt" in row and row["prompt"] != text:
            fails.append(f"row{i}: 数据集内 prompt 字段与现渲染不一致")

    n = len(rows)
    # ---- completion 侧：TRL 给 chosen/rejected 无条件补 eos（dpo_trainer.py:734）----
    # 线上 assistant 轮的收尾 token 由模板决定。若两者不是同一个 token，DPO 学到的
    # 收尾方式就和服务时不同 —— 与 prompt 侧错位同一类隐性 bug，必须验。
    completion_check: dict = {"checked": 0, "suffix_set": [], "eos_token": tok.eos_token,
                              "suffix_equals_eos": None, "note": ""}
    gold_path = Path(args.gold_dataset)
    if gold_path.exists():
        gold_rows = [json.loads(l) for l in
                     gold_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.n:
            gold_rows = gold_rows[:args.n]
        suffixes = set()
        for i, row in enumerate(gold_rows):
            msgs_all = row.get("messages") or []
            ans = next((m["content"] for m in msgs_all if m["role"] == "assistant"), None)
            if ans is None:
                continue
            msgs_p = [m for m in msgs_all if m["role"] != "assistant"]
            prompt_text = render(tok, msgs_p, enable_thinking=False)
            full_text = tok.apply_chat_template(msgs_all, tokenize=False,
                                                add_generation_prompt=False,
                                                enable_thinking=False)
            if not full_text.startswith(prompt_text):
                fails.append(f"gold{i}: 完整会话渲染不以 prompt 渲染为前缀，"
                             f"无法切出 completion 边界")
                continue
            tail = full_text[len(prompt_text):]
            if not tail.startswith(ans):
                fails.append(f"gold{i}: completion 段不以 assistant 原文开头")
                continue
            suffixes.add(tail[len(ans):])
            completion_check["checked"] += 1
        completion_check["suffix_set"] = sorted(suffixes)
        if len(suffixes) == 1:
            suffix = next(iter(suffixes))
            eq = suffix == (tok.eos_token or "")
            completion_check["suffix_equals_eos"] = eq
            completion_check["note"] = (
                "模板收尾 == eos_token，TRL 补的 eos 与线上一致"
                if eq else
                f"模板收尾 {suffix!r} != eos_token {tok.eos_token!r}："
                f"DPO 的 chosen/rejected 需自带该收尾串，否则训练与服务收尾错位")
        elif len(suffixes) > 1:
            completion_check["note"] = "模板收尾串不唯一，需逐条处理"
            fails.append(f"completion 收尾串不唯一：{sorted(suffixes)}")
    else:
        completion_check["note"] = f"未找到 {gold_path}，跳过 completion 侧校验"

    result = {
        "dataset": args.dataset, "model": args.model, "n": n,
        "token_identity_failures": len([f for f in fails if "token 不一致" in f]),
        "rows_where_add_special_tokens_changes_ids": double_bos,
        "rows_where_enable_thinking_changes_render": thinking_differs,
        "add_special_tokens_must_be_false": double_bos > 0,
        "trl_tokenizes_with_add_special_tokens_false": "dpo_trainer.py:674 硬编码 False",
        "completion_suffix": completion_check,
        "failures": fails[:20], "n_failures": len(fails),
        "passed": not fails and thinking_differs == n,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")

    if fails:
        print(f"\n✗ 模板一致性失败 {len(fails)} 条（G0 第 2 项不通过）")
        return 1
    if thinking_differs != n:
        print(f"\n✗ enable_thinking 在 {n - thinking_differs}/{n} 条上不改变渲染："
              f"模板可能未实现该开关，一致性断言失去意义。")
        return 1
    print(f"\n✓ {n}/{n} 条 prompt 与模板真源逐 token 相同；"
          f"enable_thinking 开关在全部样本上生效。")
    if double_bos:
        print(f"⚠ 有 {double_bos} 条在 add_special_tokens=True 时 token 会变化："
              f"训练/采样侧 **必须** 用 add_special_tokens=False，否则双 BOS。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
