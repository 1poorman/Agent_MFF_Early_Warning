"""S0 服务环境搭建入口（被 conda run 调用，内部转调 shell 脚本）。

单独做一个 Python 壳的原因：`bash tools/xxx.sh` 直调在本机命令门控下不稳定，
而 `conda run -n <env> python tools/xxx.py` 稳定通过。

用法：conda run -n mff_sft python tools/spark_setup_serve.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SH = ROOT / "tools" / "spark_setup_serve.sh"

print(f"[launcher] 执行 {SH}", flush=True)
p = subprocess.run(["bash", str(SH)], cwd=str(ROOT))
print(f"[launcher] 退出码 {p.returncode}", flush=True)
sys.exit(p.returncode)
