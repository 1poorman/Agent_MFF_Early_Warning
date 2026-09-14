#!/usr/bin/env bash
# 启动 Spark-X2.5-4B 的 vLLM OpenAI 兼容服务（S0/S1/S3 共用）
#
# 端口 3762（3761 是 2B 生产端口，不占用）。
# 环境 mff_sft_spark_serve = mff_sft_serve 的克隆 + XHToken/Spark-plugin，
# 与 2B 生产服务隔离（插件是 vllm.general_plugins 入口，装了会在该环境所有
# vLLM 进程生效）。
#
# 用法：
#   bash tools/serve_spark.sh                                   # 底座，GPU1:3762
#   MODEL=~/models/mff-sft-spark-x25-4b bash tools/serve_spark.sh   # SFT 产物
#   GPU=0 PORT=3762 bash tools/serve_spark.sh
#
# 后台常驻：
#   nohup bash tools/serve_spark.sh > logs/vllm_spark_3762.log 2>&1 &
set -e

SERVE=/home/huachenghao/.conda/envs/mff_sft_spark_serve
MODEL=${MODEL:-/models/hch/Spark-X2.5-4B}
NAME=${NAME:-$(basename "$MODEL")}
PORT=${PORT:-3762}
GPU=${GPU:-1}
MAXLEN=${MAXLEN:-8192}
UTIL=${UTIL:-0.5}

echo "serving $MODEL as '$NAME' on GPU$GPU:$PORT (max-model-len=$MAXLEN util=$UTIL)"
echo "env: $SERVE （含 vllm-spark2_5-plugin）"

CUDA_VISIBLE_DEVICES=$GPU exec "$SERVE/bin/vllm" serve "$MODEL" \
  --served-model-name "$NAME" \
  --trust-remote-code \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" \
  --port "$PORT" \
  --chat-template "$MODEL/chat_template.jinja" \
  --enable-prefix-caching
