#!/usr/bin/env bash
# S0：为 Spark-X2.5-4B 建独立 vLLM 服务环境（不动 mff_sft_serve）
#
# 做法：clone 现有 mff_sft_serve（保住 vLLM 0.28 + torch 2.13 + cu130 的已验证组合），
#       再把 XHToken/Spark-plugin 装进克隆体。
# 理由：插件 pyproject 的 dependencies 只有 "vllm"（无版本 pin），不会改 vLLM 本体；
#       它是 vllm.general_plugins 入口，装了会在该环境下**所有** vLLM 进程生效，
#       故必须与 2B 生产服务隔离。
set -u

SRC=mff_sft_serve
DST=mff_sft_spark_serve
PLUGIN=/tmp/Spark-plugin
PY=/home/huachenghao/.conda/envs/$DST/bin/python

echo "=========== 1. 检查插件源码 ==========="
if [ ! -d "$PLUGIN" ]; then
  git clone --depth 1 https://github.com/XHToken/Spark-plugin.git "$PLUGIN"
fi
git -C "$PLUGIN" log --oneline -1

echo "=========== 2. clone conda 环境 $SRC -> $DST ==========="
if [ -x "$PY" ]; then
  echo "已存在 $DST，跳过 clone"
else
  conda create --clone "$SRC" -n "$DST" -y
fi

echo "=========== 3. 装插件（--no-deps，绝不动 vLLM） ==========="
"$PY" -m pip install --no-deps -e "$PLUGIN" 2>&1 | tail -5

echo "=========== 4. 验证 ==========="
"$PY" - <<'EOF'
import importlib.metadata as md
import vllm, torch, transformers
print("vllm", vllm.__version__, "| torch", torch.__version__, "| tf", transformers.__version__)

# 插件的 entry point 是否注册
eps = md.entry_points()
try:
    group = eps.select(group="vllm.general_plugins")
except AttributeError:
    group = eps.get("vllm.general_plugins", [])
print("vllm.general_plugins entry points:")
for ep in group:
    print("   ", ep.name, "->", ep.value)

# 插件模块与 vLLM 内部 API 是否可导入（真正的兼容性判据）
import vllm_spark2_5_plugin as p
print("plugin import OK | ARCHITECTURE =", p.ARCHITECTURE, "| TESTED_VLLM =", p.TESTED_VLLM)

from vllm.transformers_utils.config import _CONFIG_REGISTRY
print("_CONFIG_REGISTRY 可访问 OK")

from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
print("ToolParserManager 可访问 OK")

from vllm import ModelRegistry
print("ModelRegistry 可访问 OK")

# 关键：插件自带的模型实现能否导入（依赖 vLLM 内部算子层）
import vllm_spark2_5_plugin.spark2_5 as s
print("spark2_5 模型实现 import OK ->", s.Spark2_5ForCausalLM)

p.register()
print("register() 调用成功")
print("已支持架构含 Spark2_5ForCausalLM:",
      "Spark2_5ForCausalLM" in ModelRegistry.get_supported_archs())
EOF

echo "=========== 5. 训练环境依赖现状（供 S2 参考） ==========="
for e in mff_sft; do
  echo "--- $e ---"
  /home/huachenghao/.conda/envs/$e/bin/python - <<'EOF'
import importlib.util as u
import torch, transformers, sys
print("  py", sys.version.split()[0], "| torch", torch.__version__, "| tf", transformers.__version__)
for m in ["deepspeed", "peft", "fastapi", "uvicorn", "datasets", "accelerate"]:
    print(f"  {m}: " + ("OK" if u.find_spec(m) else "-"))
EOF
done

echo "=========== 6. 磁盘 ==========="
df -h /home | tail -1
