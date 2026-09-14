"""Spark-X2.5-4B 落地勘察（一次性脚本，只读 + clone 到 /tmp，不安装、不改现有环境）。

用 `conda run -n mff_sft python tools/spark_recon.py` 运行。
"""
import os
import shutil
import subprocess
import sys

SB = "/home/huachenghao/.conda/envs"
LF_UPSTREAM = "/home/huachenghao/codes/LLaMA-Factory"
PLUGIN = "/tmp/Spark-plugin"
LF_FORK = "/tmp/lf_spark_probe"


def run(cmd, timeout=300, cwd=None):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd)
        return (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return "!! TIMEOUT"
    except Exception as e:
        return f"!! {type(e).__name__}: {e}"


def head(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


head("0. 环境")
print(run("hostname; date; whoami"))

head("1. 磁盘")
print(run("df -h /home | tail -2"))
for e in ["mff_sft", "mff_sft_serve", "mff_sft_quant"]:
    print(run(f"du -sh {SB}/{e} 2>/dev/null"))

head("2. GPU")
print(run("nvidia-smi --query-gpu=index,name,memory.used,memory.total "
          "--format=csv"))
print("--- 端口 3761/3762/8000/30000 ---")
print(run("ss -ltn 2>/dev/null | grep -E '3761|3762|8000|30000' || echo '均未监听'"))

head("3. 各环境关键包")
probe = ("import importlib.util as u, sys\n"
         "import torch, transformers\n"
         "print('py', sys.version.split()[0], '| torch', torch.__version__,"
         " '| tf', transformers.__version__)\n"
         "for m in ['deepspeed','peft','fastapi','uvicorn','vllm','datasets']:\n"
         "    print(f'  {m}: ' + ('OK' if u.find_spec(m) else '-'))\n")
for e in ["mff_sft", "mff_sft_serve", "mff_sft_quant", "mff_agent"]:
    print(f"--- {e} ---")
    print(run(f"{SB}/{e}/bin/python -c {probe!r}"))

head("4. 上游 LLaMA-Factory（现有训练环境）")
print(run(f"git -C {LF_UPSTREAM} log --oneline -1; "
          f"git -C {LF_UPSTREAM} remote -v | head -2"))
print("--- 是否注册 spark 模板 ---")
print(run(f"grep -rn -i spark {LF_UPSTREAM}/src/llamafactory/ | head -10 "
          f"|| echo '(无)'"))

head("5. clone XHToken/Spark-plugin")
if os.path.isdir(PLUGIN):
    print(f"已存在 {PLUGIN}")
else:
    print(run(f"git clone --depth 1 https://github.com/XHToken/Spark-plugin.git "
              f"{PLUGIN}", timeout=300))
if os.path.isdir(PLUGIN):
    print("--- commit ---")
    print(run(f"git -C {PLUGIN} log --oneline -1"))
    print("--- 顶层文件 ---")
    print(run(f"ls -la {PLUGIN}"))
    for f in ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"]:
        p = os.path.join(PLUGIN, f)
        if os.path.exists(p):
            print(f"--- {f} ---")
            print(open(p, encoding="utf-8", errors="replace").read())
    print("--- .py 文件树 ---")
    print(run(f"find {PLUGIN} -name '*.py' -not -path '*/.git/*' | head -40"))
    print("--- 是否声明 vLLM 插件入口 / 版本约束 ---")
    print(run(f"grep -rn -i -E 'vllm|entry.points|transformers' {PLUGIN} "
              f"--include='*.toml' --include='*.py' --include='*.cfg' | head -30"))
    print("--- README 前 60 行 ---")
    rp = os.path.join(PLUGIN, "README.md")
    if os.path.exists(rp):
        print("\n".join(open(rp, encoding="utf-8",
                             errors="replace").read().splitlines()[:60]))

head("6. clone XHToken/LlamaFactory fork")
if os.path.isdir(LF_FORK):
    print(f"已存在 {LF_FORK}")
else:
    print(run(f"git clone --depth 1 https://github.com/XHToken/LlamaFactory.git "
              f"{LF_FORK}", timeout=600))
if os.path.isdir(LF_FORK):
    print("--- commit ---")
    print(run(f"git -C {LF_FORK} log --oneline -1"))
    print("--- 模板注册中的 spark ---")
    print(run(f"grep -n -i spark {LF_FORK}/src/llamafactory/data/template.py "
              f"|| echo '(template.py 无 spark)'"))
    print("--- 全仓 spark 引用 ---")
    print(run(f"grep -rn -i spark {LF_FORK}/src/llamafactory/ | head -25 "
              f"|| echo '(无)'"))
    print("--- 是否有 tied_weights 相关修补 ---")
    print(run(f"grep -rn tied_weights {LF_FORK}/src/llamafactory/ | head -10 "
              f"|| echo '(无)'"))
    print("--- 与上游差异（仅文件名） ---")
    print(run(f"diff -rq {LF_UPSTREAM}/src/llamafactory "
              f"{LF_FORK}/src/llamafactory 2>/dev/null | head -40"))

head("7. eval.jsonl 规模")
print(run(f"""{SB}/mff_agent/bin/python -c "
import json
rows=[json.loads(l) for l in open('/home/huachenghao/codes/Agent_MFF_Early_Warning/data/sft/eval.jsonl',encoding='utf-8')]
u=[len(m['content']) for r in rows for m in r['messages'] if m['role']=='user']
a=[len(m['content']) for r in rows for m in r['messages'] if m['role']=='assistant']
from collections import Counter
print('n=',len(rows))
print('kind:',dict(Counter(r['meta']['kind'] for r in rows)))
print(f'user chars min/mean/max={min(u)}/{sum(u)//len(u)}/{max(u)}')
print(f'asst chars min/mean/max={min(a)}/{sum(a)//len(a)}/{max(a)}')
"
"""))

head("8. 缓存中的 remote code 目录")
cache = ("/home/huachenghao/.cache/huggingface/modules/transformers_modules/"
         "Spark_hyphen_X2_dot_5_hyphen_4B")
print(run(f"find {cache} -maxdepth 2 -printf '%y %10s  %p\\n' 2>/dev/null | head -20 "
          f"|| echo '(无)'"))

print("\n\n勘察结束。")
sys.exit(0)
