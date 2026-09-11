"""MS4 验收补充：验证集 loss 轨迹（底座 vs 各 checkpoint）。

背景：训练时 `eval_steps: 200` 未触发评估（transformers 5.x 需显式 eval_strategy，
LLaMA-Factory 的 eval_dataset 不会自动开评估），日志里 "No metric eval_loss to plot"。
本脚本事后补算，且比训练内评估更完整——能直接看出 val loss 是否随 checkpoint 回升。

口径与训练完全一致：
  - 用 LLaMA-Factory 的 minicpm5 模板 + enable_thinking=false 编码；
  - loss 只算 response 段（prompt 用 -100 屏蔽），与训练目标同口径。

用法：
  conda run -n mff_sft python tools/eval_loss.py --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
LF_REPO = Path.home() / "codes" / "LLaMA-Factory"
sys.path.insert(0, str(LF_REPO / "src"))

BASE = "/home/huachenghao/models/MiniCPM5-2B"
SAVES = ROOT / "saves" / "mff-sft-full"


def build_batches(template, tok, samples, device):
    """按 response 为目标构造 (input_ids, labels) 列表。"""
    out = []
    for s in samples:
        msgs = s["messages"]
        system, rest = "", msgs
        if msgs and msgs[0]["role"] == "system":
            system, rest = msgs[0]["content"], msgs[1:]
        prompt_ids, response_ids = template.encode_oneturn(tok, rest, system=system)
        if not response_ids:
            continue
        out.append((prompt_ids + response_ids, [-100] * len(prompt_ids) + list(response_ids)))
    return out


@torch.no_grad()
def eval_loss(model, batches, device, batch_size=8):
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(batches), batch_size):
        chunk = batches[i:i + batch_size]
        maxlen = max(len(x) for x, _ in chunk)
        input_ids, labels, attn = [], [], []
        for ids, lab in chunk:
            pad = maxlen - len(ids)
            input_ids.append(ids + [0] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        input_ids = torch.tensor(input_ids, device=device)
        labels = torch.tensor(labels, device=device)
        attn = torch.tensor(attn, device=device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        # 逐 token 交叉熵，只统计 labels != -100
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), ignore_index=-100, reduction="sum")
        n = int((shift_labels != -100).sum())
        total_nll += float(loss)
        total_tok += n
    return total_nll / max(total_tok, 1), total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--splits", default="val,train",
                    help="逗号分隔：val/train（train 默认抽样 300 条）")
    ap.add_argument("--train-limit", type=int, default=300)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    template = get_template_and_fix_tokenizer(
        tok, DataArguments(template="minicpm5", enable_thinking=False))

    data = {}
    for split in args.splits.split(","):
        lines = [json.loads(l) for l in
                 open(ROOT / "data" / "sft" / f"{split}.jsonl", encoding="utf-8")]
        if split == "train" and args.train_limit:
            lines = lines[:args.train_limit]
        data[split] = build_batches(template, tok, lines, args.device)
        print(f"{split}: {len(lines)} 条 → {len(data[split])} 个 batch 样本")

    models = [("base（未微调）", BASE)]
    for ck in ("checkpoint-200", "checkpoint-400", "checkpoint-441"):
        p = SAVES / ck
        if p.exists():
            models.append((ck, str(p)))
    models.append(("final（=441）", str(SAVES)))

    print(f"\n{'模型':<22} " + " ".join(f"{s:>10}" for s in data))
    print("-" * 60)
    results = {}
    for name, path in models:
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, dtype="bfloat16").to(args.device)
        row = {}
        for split, batches in data.items():
            l, n = eval_loss(model, batches, args.device)
            row[split] = (l, n)
        results[name] = row
        print(f"{name:<22} " + " ".join(f"{row[s][0]:>10.4f}" for s in data))
        del model
        torch.cuda.empty_cache()

    print("\n（数值=逐 token 交叉熵 NLL，仅 response 段；越小越好）")
    base_val = results.get("base（未微调）", {}).get("val", (None,))[0]
    final_val = results.get("final（=441）", results.get("checkpoint-441", {})).get("val", (None,))[0]
    if base_val and final_val:
        print(f"val loss: base {base_val:.4f} → SFT {final_val:.4f} "
              f"(降低 {(1 - final_val / base_val) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
