#!/usr/bin/env python3
"""G0 决策门第 1 项：mff_rl 里四个 Trainer 可导入，且可用性开关全为真 bool。

背景见 tools/patch_trl_import_utils.py 的模块文档：TRL 0.24.0 借用
transformers 的 `_is_package_available`，而 transformers 5.x 改成返回元组，
导致 12 个开关恒真、mergekit 被无条件 import。本脚本验证补丁生效。

用法：
    /home/huachenghao/.conda/envs/mff_rl/bin/python tools/check_rl_env.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("logs/rl_minicpm5_20260924/env/trl_env_check.json")


def main() -> int:
    import trl.import_utils as iu

    flags = {n: getattr(iu, n)() for n in dir(iu)
             if n.startswith("is_") and n.endswith("_available")}
    non_bool = {n: repr(v) for n, v in flags.items() if not isinstance(v, bool)}

    trainers, errors = [], {}
    for name in ("DPOTrainer", "GRPOTrainer", "RLOOTrainer", "RewardTrainer",
                 "DPOConfig", "GRPOConfig", "RLOOConfig", "RewardConfig"):
        try:
            getattr(__import__("trl", fromlist=[name]), name)
            trainers.append(name)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    import accelerate
    import peft
    import torch
    import transformers
    import trl

    import inspect
    rloo_params = []
    if "RLOOTrainer" in trainers:
        from trl import RLOOTrainer
        rloo_params = [p for p in inspect.signature(RLOOTrainer.__init__).parameters
                       if p != "self"]

    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"trl": trl.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__, "torch": torch.__version__,
                     "accelerate": accelerate.__version__},
        "n_flags": len(flags), "flags_non_bool": non_bool,
        "is_mergekit_available": flags.get("is_mergekit_available"),
        "trainers_imported": trainers, "trainer_errors": errors,
        "rloo_init_params": rloo_params,
        "passed": not non_bool and not errors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    v = result["versions"]
    print(f"trl={v['trl']} transformers={v['transformers']} peft={v['peft']} "
          f"torch={v['torch']} accelerate={v['accelerate']}")
    print(f"可用性开关 {len(flags)} 个，非 bool {len(non_bool)} 个；"
          f"is_mergekit_available()={flags.get('is_mergekit_available')}")
    if non_bool:
        print("✗ 仍有恒真元组开关：", non_bool)
    if errors:
        print("✗ Trainer 导入失败：")
        for k, e in errors.items():
            print(f"   {k}: {e}")
    if result["passed"]:
        print(f"✓ 四个 Trainer + 四个 Config 全部导入成功：{', '.join(trainers)}")
        print(f"✓ RLOOTrainer 参数：{rloo_params}")
    print(f"留痕 {OUT}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
