"""MiniCPM5-2B SFT 产物 INT4(AWQ) 量化（MS5 L3 / 可与 MS6 一并做）。

⚠️ 本机当前**未安装** autoawq，且离线网络受限；本脚本为 ready-to-run recipe，
   安装工具后即可执行，产出 ~/models/mff-sft-minicpm5-2b-awq，再按下方口径复评。
   MS5 报告已将量化复评列为可与 MS6 并行的后续项（bf16 已可部署）。

依赖安装（清华镜像可达）：
    $HOME/.conda/envs/mff_sft/bin/pip install autoawq \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

量化：
    conda run -n mff_sft python tools/mff_quantize.py \
        --model ~/models/mff-sft-minicpm5-2b --out ~/models/mff-sft-minicpm5-2b-awq

复评（L3 决策门：INT4 vs bf16 混淆对差 ≤3pp，否则弃量化用 bf16）：
    # 起 AWQ 服务
    CUDA_VISIBLE_DEVICES=0 $HOME/.conda/envs/mff_sft_serve/bin/vllm serve \
        ~/models/mff-sft-minicpm5-2b-awq --served-model-name mff-sft-awq \
        --quantization awq --max-model-len 8192 --port 3762 &
    # 同 3 seed 全表复评 + 回归零退化
    conda run -n mff_agent python tests/eval_sft_model.py \
        --url http://localhost:3762/v1 --model mff-sft-awq --arbitration off
    conda run -n mff_agent python tests/run_regression.py \
        --url http://localhost:3762/v1 --model mff-sft-awq
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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


def main():
    ap = argparse.ArgumentParser(description="MiniCPM5-2B SFT INT4(AWQ) 量化")
    ap.add_argument("--model", default=str(Path.home() / "models/mff-sft-minicpm5-2b"))
    ap.add_argument("--out", default=str(Path.home() / "models/mff-sft-minicpm5-2b-awq"))
    ap.add_argument("--calib", default="data/sft/train.jsonl")
    ap.add_argument("--calib-n", type=int, default=128)
    args = ap.parse_args()

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("[缺依赖] 未安装 autoawq。安装后重试：\n"
              "  $HOME/.conda/envs/mff_sft/bin/pip install autoawq "
              "-i https://pypi.tuna.tsinghua.edu.cn/simple")
        sys.exit(2)

    calib = build_calib(Path(args.calib), args.calib_n)
    print(f"校准语料: {len(calib)} 条（来自 {args.calib}）")

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    model = AutoAWQForCausalLM.from_pretrained(args.model, safetensors=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.quantize(tok, quant_config=quant_config, calib_data=calib)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_quantized(args.out)
    tok.save_pretrained(args.out)
    print(f"INT4(AWQ) 产物已保存: {args.out}\n下一步：起 3762 服务 + 3 seed 复评（差 ≤3pp）")


if __name__ == "__main__":
    main()
