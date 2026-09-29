#!/usr/bin/env python3
"""修复 TRL 0.24.0 × transformers 5.x 的可用性探测 bug（只打在 mff_rl 环境上）。

背景
----
`trl/import_utils.py` 从 transformers 借用 `_is_package_available`：

    from transformers.utils.import_utils import _is_package_available
    _mergekit_available = _is_package_available("mergekit")

TRL 0.24.0 假定该函数在 `return_version=False` 时返回 **bool**。但
transformers 5.8.0 的实现在两条分支上都返回**元组**：

    if return_version: return package_exists, package_version
    return package_exists, None          # ← 未装时是 (False, None)

`(False, None)` 是非空元组，**恒为真**。于是 12 个可用性开关全部恒真，
TRL 会把所有可选依赖都当成已安装：

    is_deepspeed/fastapi/joblib/llm_blender/math_verify/mergekit/
    pydantic/requests/unsloth/uvicorn/vllm_ascend/weave_available

最先炸的是 mergekit —— `trl/mergekit_utils.py` 的
`if is_mergekit_available():` 恒真，于是无条件 `from mergekit.config import ...`，
直接 ModuleNotFoundError，DPO/GRPO/RLOO/Reward 四个 Trainer 全部取不到。

为什么不装 mergekit
------------------
装 mergekit 只能拆掉 12 颗雷里的 1 颗，且引入新冲突：mergekit 0.1.4 硬钉
`pydantic~=2.10.6`，而本环境是 pydantic 2.12.3，`create_model` 不再接受
`torch.Tensor` 字段，import 时报 "Unable to generate pydantic-core schema
for <class 'torch.Tensor'>"。降级 pydantic 又会波及 deepspeed/fastapi/
gradio/llamafactory。因此改为修正上游解包 bug：mff_rl 的包版本与 mff_sft
**逐行相同**，只多这一处带 diff 留痕的源码补丁。

用法（幂等，重复执行只会报告 already patched）：
    python tools/patch_trl_import_utils.py --env mff_rl [--check]
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# 形如：_xxx_available = _is_package_available("xxx")   （行尾无解包）
PATTERN = re.compile(r'^(_[a-z0-9_]+_available) = (_is_package_available\("[a-z0-9_]+"\))$')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="mff_rl", help="只允许打在非 mff_sft 的环境上")
    ap.add_argument("--check", action="store_true", help="只报告，不落盘")
    ap.add_argument("--out-dir", default="logs/rl_minicpm5_20260924/env")
    args = ap.parse_args()

    if args.env == "mff_sft":
        print("拒绝修改 mff_sft：已发表 SFT 基线的环境必须逐字节不变。", file=sys.stderr)
        return 2

    root = Path(f"/home/huachenghao/.conda/envs/{args.env}/lib/python3.11/site-packages")
    target = root / "trl" / "import_utils.py"
    if not target.exists():
        print(f"找不到 {target}", file=sys.stderr)
        return 2

    original = target.read_text(encoding="utf-8")
    before_sha = sha256(target)
    lines = original.splitlines(keepends=True)

    patched_lines, touched = [], []
    for line in lines:
        m = PATTERN.match(line.rstrip("\n"))
        if m:
            patched_lines.append(f"{m.group(1)} = {m.group(2)}[0]\n")
            touched.append(m.group(1))
        else:
            patched_lines.append(line)
    patched = "".join(patched_lines)

    if not touched:
        print(f"already patched（或版本不匹配）：{target}")
        print(f"sha256={before_sha}")
        return 0

    diff = "".join(difflib.unified_diff(
        lines, patched_lines,
        fromfile=f"a/trl/import_utils.py ({args.env}, original)",
        tofile=f"b/trl/import_utils.py ({args.env}, patched)"))
    print(f"将修正 {len(touched)} 个恒真开关：")
    for name in touched:
        print(f"  - {name}")
    if args.check:
        print("\n--check：未落盘。")
        print(diff)
        return 0

    backup = target.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    after_sha = sha256(target)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trl_import_utils.patch").write_text(diff, encoding="utf-8")

    try:
        trl_version = subprocess.run(
            [f"/home/huachenghao/.conda/envs/{args.env}/bin/python", "-c",
             "import importlib.metadata as m; print(m.version('trl'))"],
            capture_output=True, text=True, check=True).stdout.strip()
        tf_version = subprocess.run(
            [f"/home/huachenghao/.conda/envs/{args.env}/bin/python", "-c",
             "import importlib.metadata as m; print(m.version('transformers'))"],
            capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        trl_version = tf_version = "unknown"

    manifest = {
        "env": args.env, "file": str(target), "backup": str(backup),
        "reason": "TRL 0.24.0 assumes _is_package_available returns bool; "
                  "transformers 5.x returns a tuple, making 12 flags always-true",
        "trl_version": trl_version, "transformers_version": tf_version,
        "flags_fixed": touched, "n_flags_fixed": len(touched),
        "sha256_before": before_sha, "sha256_after": after_sha,
        "patched_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "trl_patch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已落盘。备份 {backup}")
    print(f"sha256 {before_sha[:12]} -> {after_sha[:12]}")
    print(f"留痕 {out_dir}/trl_import_utils.patch, trl_patch_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
