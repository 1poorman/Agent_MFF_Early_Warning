"""MiniCPM5-2B SFT 产物 INT4 量化（MS5 L3）。

量化后端：**vLLM 官方推荐的 llmcompressor**（compressed-tensors W4A16 格式，
vLLM 0.28 原生加载）。选择理由：
  - `autoawq 0.2.9` 要求 transformers≤4.x，与本机 transformers 5.x 不兼容；
  - llmcompressor 产出的 compressed-tensors 与 serve 环境已装的
    `compressed-tensors 0.17` 同族，无需额外运行时依赖。

环境隔离（勿污染训练环境 `mff_sft`）：
    conda create -y -n mff_sft_quant python=3.11
    $HOME/.conda/envs/mff_sft_quant/bin/pip install llmcompressor==0.13.0 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

量化：
    $HOME/.conda/envs/mff_sft_quant/bin/python tools/mff_quantize.py \
        --model ~/models/mff-sft-minicpm5-2b \
        --out   ~/models/mff-sft-minicpm5-2b-int4 \
        --calib data/sft/train.jsonl --calib-n 128 --scheme W4A16

L3 复评（决策门：INT4 vs bf16 混淆对差 ≤3pp，否则弃量化用 bf16）：
    CUDA_VISIBLE_DEVICES=0 $HOME/.conda/envs/mff_sft_serve/bin/vllm serve \
        ~/models/mff-sft-minicpm5-2b-int4 --served-model-name mff-sft-int4 \
        --max-model-len 32768 --port 3762 &
    conda run -n mff_agent python tests/eval_sft_model.py \
        --url http://localhost:3762/v1 --model mff-sft-int4 --arbitration off
    conda run -n mff_agent python tests/run_regression.py \
        --url http://localhost:3762/v1 --model mff-sft-int4
"""

import argparse
import json
import sys
import traceback
from pathlib import Path


def build_calib(train_path: Path, n: int) -> list:
    """用 SFT 训练集的退化版 user prompt 作校准语料（与部署分布一致）。"""
    rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = []
    for r in rows[:n]:
        for m in r.get("messages", []):
            if m.get("role") == "user":
                texts.append(m["content"])
                break
    return texts


def build_recipe(scheme: str, ignore=("lm_head",)):
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    # W4A16 = 4bit 权重、16bit 激活（vLLM 支持）；GPTQ 比 RTN 精度更好
    if scheme.upper() == "W4A16_GPTQ":
        return GPTQModifier(targets="Linear", scheme="W4A16", ignore=list(ignore))
    return QuantizationModifier(targets="Linear", scheme="W4A16", ignore=list(ignore))


def main():
    ap = argparse.ArgumentParser(description="MiniCPM5-2B SFT INT4 量化（llmcompressor）")
    ap.add_argument("--model", default=str(Path.home() / "models/mff-sft-minicpm5-2b"))
    ap.add_argument("--out", default=str(Path.home() / "models/mff-sft-minicpm5-2b-int4"))
    ap.add_argument("--calib", default="data/sft/train.jsonl")
    ap.add_argument("--calib-n", type=int, default=128)
    ap.add_argument("--scheme", default="W4A16", choices=["W4A16", "W4A16_GPTQ"])
    ap.add_argument("--max-seq-length", type=int, default=2048)
    args = ap.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import Dataset
        from llmcompressor import oneshot
    except ImportError:
        print("[缺依赖] 未安装 llmcompressor。请先建独立环境：\n"
              "  conda create -y -n mff_sft_quant python=3.11\n"
              "  $HOME/.conda/envs/mff_sft_quant/bin/pip install llmcompressor==0.13.0 "
              "-i https://pypi.tuna.tsinghua.edu.cn/simple")
        sys.exit(2)

    calib = build_calib(Path(args.calib), args.calib_n)
    print(f"校准语料: {len(calib)} 条（来自 {args.calib}）")
    print(f"量化方案: {args.scheme} · 后端 llmcompressor/compressed-tensors")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    recipe = build_recipe(args.scheme)
    ds = Dataset.from_list([{"text": t} for t in calib])

    oneshot(
        model=model,
        tokenizer=tok,
        dataset=ds,
        recipe=recipe,
        output_dir=args.out,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(calib),
        text_column="text",
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(args.out)
    print(f"INT4 产物已保存: {args.out}\n下一步：3762 起服务 + 3 seed 全表 + 回归（混淆对差 ≤3pp）")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
