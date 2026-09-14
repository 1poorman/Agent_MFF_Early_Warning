#!/usr/bin/env bash
# SFT 前置服务监督器：进程退出后自动重启 + 退避（可选运维工具）。
#
# 用途：让 2B vLLM（GPU0:3761）常驻——正常退出/被停服后按退避自动拉起；
#       短时间连续退出则顺带清理 `~/.cache/vllm/torch_compile_cache`（缓存异常时）。
# 说明：日志中的 `EngineDeadError` 为手动 SIGTERM 关闭的收尾日志，非服务崩溃；
#       本脚本仅提供"常驻/自愈"便利，不用于修复稳定性问题。
#
# 用法（后台常驻）：
#   nohup bash tools/serve_sft_supervised.sh > logs/vllm_sft_supervisor.log 2>&1 &
# 停止：
#   pkill -f serve_sft_supervised ; pkill -f 'vllm serve.*minicpm5'
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT=${PORT:-3761}
BACKOFF=${BACKOFF:-10}          # 基础退避秒数
FAILS=0
FAIL_WINDOW=180                 # 该窗口内计连续退出

echo "[supervisor] watching SFT front on port $PORT (backoff=${BACKOFF}s)"
while true; do
  START=$(date +%s)
  echo "[supervisor] $(date '+%F %T') starting serve_sft.sh"
  bash "$ROOT/tools/serve_sft.sh"
  RC=$?
  UP=$(( $(date +%s) - START ))
  echo "[supervisor] serve_sft.sh exited rc=$RC after ${UP}s"
  # 存活超过窗口视为稳定运行，重置失败计数
  if [ "$UP" -ge "$FAIL_WINDOW" ]; then
    FAILS=0
  else
    FAILS=$((FAILS + 1))
  fi
  # 连续短命退出（>=2）疑似编译缓存异常 -> 清缓存
  if [ "$FAILS" -ge 2 ]; then
    echo "[supervisor] 连续短命退出 ${FAILS} 次，清理 torch_compile_cache"
    rm -rf "$HOME/.cache/vllm/torch_compile_cache"
    FAILS=0
  fi
  SLEEP=$(( BACKOFF * (FAILS + 1) ))
  echo "[supervisor] restart in ${SLEEP}s"
  sleep "$SLEEP"
done
