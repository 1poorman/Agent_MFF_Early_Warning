#!/usr/bin/env bash
# 启动 vLLM OpenAI 兼容服务（MS0 基线复算 / MS5 SFT 产物评估 / MS6 级联前置共用）
#
# 端口统一 3761（2026-09-10 由 9998 改，见 docs/SFT_STATUS.md 第五节）。
# 用法：
#   bash tools/serve_sft.sh                                  # 默认 SFT 产物，GPU0:3761
#   bash tools/serve_sft.sh /home/huachenghao/models/MiniCPM5-2B MiniCPM5-2B
#   GPU=1 PORT=3761 bash tools/serve_sft.sh <model_dir> <served_name>
#
# 可选运维开关（默认编译模式，性能最佳）：
#   EAGER=1      禁用 torch.compile/cudagraphs（--enforce-eager），**默认关闭**。
#                实测 2B 时延 0.99s -> 3.8~6.1s（约 4x），仅在特殊排障时使用。
#   CLEAN_CACHE=1 启动前删除 ~/.cache/vllm/torch_compile_cache。
#                缓存加载告警（`ptxas Bad address`，非致命，会自动重编译）时可清理。
#
# 说明：日志中 `EngineDeadError` 出现在手动 `kill` 的 SIGTERM 之后，是正常关闭
#       的收尾日志，并非服务崩溃；vLLM 稳定性无实质问题。
#
# 后台常驻并写日志：
#   nohup bash tools/serve_sft.sh > logs/vllm_sft_3761.log 2>&1 &
set -e

SERVE=/home/huachenghao/.conda/envs/mff_sft_serve
MODEL=${1:-/home/huachenghao/models/mff-sft-minicpm5-2b}
NAME=${2:-$(basename "$MODEL")}
PORT=${PORT:-3761}
GPU=${GPU:-0}
MAXLEN=${MAXLEN:-32768}
UTIL=${UTIL:-0.5}
EAGER=${EAGER:-0}
CLEAN_CACHE=${CLEAN_CACHE:-0}

if [ "$CLEAN_CACHE" = "1" ]; then
  rm -rf "$HOME/.cache/vllm/torch_compile_cache"
  echo "[serve_sft] 已清理 torch_compile_cache"
fi

EXTRA=""
if [ "$EAGER" = "1" ]; then
  EXTRA="--enforce-eager"
fi

echo "serving $MODEL as '$NAME' on GPU$GPU:$PORT (max-model-len=$MAXLEN util=$UTIL eager=$EAGER)"
# exec 让 vLLM 成为本进程，便于 systemd/supervisor 直接管控与自动重启
CUDA_VISIBLE_DEVICES=$GPU exec "$SERVE/bin/vllm" serve "$MODEL" \
  --served-model-name "$NAME" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" \
  --port "$PORT" \
  $EXTRA
