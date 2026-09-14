"""S0 起服务入口（被 conda run 调用，内部转调 shell 脚本）。

与 spark_setup_serve.py 同款壳：`bash tools/xxx.sh` 直调在本机命令门控下不稳定，
`conda run -n <env> python tools/xxx.py` 稳定通过。

用法：
    conda run -n mff_sft python tools/spark_serve_launcher.py
环境变量透传：MODEL / NAME / PORT / GPU / MAXLEN / UTIL
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SH = ROOT / "tools" / "serve_spark.sh"

print(f"[launcher] 执行 {SH}", flush=True)
print(f"[launcher] MODEL={os.environ.get('MODEL', '(默认底座)')} "
      f"GPU={os.environ.get('GPU', '1')} PORT={os.environ.get('PORT', '3762')}",
      flush=True)
p = subprocess.run(["bash", str(SH)], cwd=str(ROOT))
print(f"[launcher] 退出码 {p.returncode}", flush=True)
sys.exit(p.returncode)
