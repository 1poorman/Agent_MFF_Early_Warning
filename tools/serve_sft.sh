#!/usr/bin/env bash
# 启动 vLLM OpenAI 兼容服务（MS0 基线复算 / MS5 SFT 产物评估共用）
#
# 端口统一 3761（2026-09-10 由 9998 改，见 docs/SFT_STATUS.md 第五节）。
# 用法：
#   bash tools/serve_sft.sh                                  # 默认 SFT 产物，GPU0:3761
#   bash tools/serve_sft.sh /home/huachenghao/models/MiniCPM5-2B MiniCPM5-2B
#   GPU=1 PORT=3761 bash tools/serve_sft.sh <model_dir> <served_name>
#
# 后台常驻并写日志：
#   nohup bash tools/serve_sft.sh > logs/vllm_serve.log 2>&1 &
set -e

SERVE=/home/huachenghao/.conda/envs/mff_sft_serve
MODEL=${1:-/home/huachenghao/models/mff-sft-minicpm5-2b}
NAME=${2:-$(basename "$MODEL")}
PORT=${PORT:-3761}
GPU=${GPU:-0}
MAXLEN=${MAXLEN:-32768}
UTIL=${UTIL:-0.5}

echo "serving $MODEL as '$NAME' on GPU$GPU:$PORT (max-model-len=$MAXLEN util=$UTIL)"
CUDA_VISIBLE_DEVICES=$GPU exec "$SERVE/bin/vllm" serve "$MODEL" \
  --served-model-name "$NAME" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" \
  --port "$PORT"
