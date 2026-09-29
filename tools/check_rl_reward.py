#!/usr/bin/env python3
"""G0 决策门第 3 项：tools/rl_reward.py 的移植副本与 tests/eval_sft_model.py 逐条同分。

为什么需要
----------
`tools/rl_reward.py` **不能** import 评估器 —— 后者在 import 时执行
`os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")`（tests/eval_sft_model.py:41），
训练进程 import 它就会把 GPU 藏掉。因此奖励函数里 `numbers_in` /
`extract_features` / `evidence_faithful` / `THRESHOLD_CONSTANTS` 都是**手抄副本**。
手抄就有漂移风险：常数少一个、容差写错、正则丢负号，奖励和评估指标就会
指向两个不同的目标，而训练看不出任何异常。本脚本用金标答案当 completion，
在冻结 eval 集 200 条上逐条比对两边。

本脚本**是** CPU-only 的检查器，所以可以放心 import 评估器。

一处**故意**的不一致（断言其成立，而非断言其不存在）
--------------------------------------------------
评估器算 `pv = check_physics(features, evidence)`，再把 antihall 整类从
防幻觉分母里剔掉（tests/eval_sft_model.py:278）。奖励函数没有"分母"可剔，
于是改为只惩罚**模型自己造成**的违规：

    physics = check_physics(features, evidence) - check_physics(features, [])

antihall 的输入本来就是故意越界的，把输入固有违规记到模型头上，会训练模型
**不要**正确拒诊 —— 正好与本轮 RL 的靶点相反。本脚本断言：
  * 非 antihall 行：`check_physics(features, [])` 为空 ⇒ 两种定义完全等价；
  * antihall 行：`check_physics(features, [])` 非空 ⇒ 减法确实是必要的。

用法：
    conda run -n mff_agent python tools/check_rl_reward.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(path: Path, name: str):
    """按文件路径加载模块（tests/ 与 tools/ 都没有 __init__.py）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/sft/eval.jsonl")
    ap.add_argument("--out", default="logs/rl_minicpm5_20260924/env/reward_consistency.json")
    args = ap.parse_args()

    ev = load(ROOT / "tests" / "eval_sft_model.py", "_ev_sft")
    rr = load(ROOT / "tools" / "rl_reward.py", "_rl_reward")

    kg = ev.KnowledgeGraph()
    checker = ev.AntiHallucinationChecker(kg)
    fault_names = kg.fault_names()
    scorer = rr.RewardScorer()

    rows = [json.loads(l) for l in Path(args.dataset).read_text(encoding="utf-8").splitlines()
            if l.strip()]

    fails: list[str] = []
    # 常数表漂移：最隐蔽的一类抄错
    if set(rr.THRESHOLD_CONSTANTS) != set(ev.THRESHOLD_CONSTANTS):
        only_rr = set(rr.THRESHOLD_CONSTANTS) - set(ev.THRESHOLD_CONSTANTS)
        only_ev = set(ev.THRESHOLD_CONSTANTS) - set(rr.THRESHOLD_CONSTANTS)
        fails.append(f"THRESHOLD_CONSTANTS 不一致 only_rl={only_rr} only_eval={only_ev}")
    for const in ("NO_FAULT", "DATA_BAD"):
        if getattr(rr, const) != getattr(ev, const):
            fails.append(f"{const} 字面量不一致")

    counts = {"n": len(rows), "antihall": 0, "non_antihall": 0,
              "input_physics_nonempty_antihall": 0,
              "input_physics_empty_non_antihall": 0,
              "physics_equal_non_antihall": 0}

    for i, row in enumerate(rows):
        msgs = {m["role"]: m["content"] for m in row["messages"]}
        user, assistant = msgs["user"], msgs["assistant"]
        meta = row.get("meta", {})
        kind, gold = meta.get("kind"), meta.get("gold")

        # ---- 评估器一侧（权威口径，逐字用它自己的函数） ----
        feats_ev = ev.extract_features(user)
        nums_ev = ev.numbers_in(user)
        parsed_ev = ev.RootCauseReasoner._parse(assistant)
        rc_ev = parsed_ev.get("root_cause") if parsed_ev else None
        evid_ev = (parsed_ev.get("evidence") or []) if parsed_ev else []
        sop_ev = (parsed_ev.get("sop") or []) if parsed_ev else []
        faith_ev, bad_ev = ev.evidence_faithful(evid_ev, nums_ev)
        pv_ev = checker.check_physics(feats_ev, evid_ev)
        kv_ev = checker.check_kg(rc_ev, evid_ev) if rc_ev in fault_names else []

        # ---- 奖励函数一侧（移植副本） ----
        feats_rr = rr.extract_features(user)
        nums_rr = rr.numbers_in(user)
        parsed_rr = rr.parse_output(assistant)
        rc_rr = parsed_rr.get("root_cause") if parsed_rr else None
        evid_rr = (parsed_rr.get("evidence") or []) if parsed_rr else []
        sop_rr = (parsed_rr.get("sop") or []) if parsed_rr else []
        faith_rr, bad_rr = rr.evidence_faithful(evid_rr, nums_rr)
        score = scorer.score(user, assistant, gold)

        tag = f"row{i}({kind})"
        if feats_ev != feats_rr:
            fails.append(f"{tag}: extract_features 不一致")
        if nums_ev != nums_rr:
            fails.append(f"{tag}: numbers_in 不一致 n={len(nums_ev)}/{len(nums_rr)}")
        if rc_ev != rc_rr or evid_ev != evid_rr or sop_ev != sop_rr:
            fails.append(f"{tag}: JSON 解析结果不一致 rc={rc_ev!r}/{rc_rr!r}")
        if faith_ev != faith_rr or sorted(bad_ev) != sorted(bad_rr):
            fails.append(f"{tag}: evidence_faithful 不一致 ok={faith_ev}/{faith_rr} "
                         f"bad={bad_ev}/{bad_rr}")
        if faith_rr != score.faith_ok:
            fails.append(f"{tag}: Score.faith_ok 与分项函数自相矛盾")
        if sorted(map(str, kv_ev)) != sorted(map(str, checker.check_kg(rc_rr, evid_rr)
                                                 if rc_rr in fault_names else [])):
            fails.append(f"{tag}: check_kg 不一致")

        # r_gold 必须等于评估器主指标 correct_raw 的定义
        if score.correct != (rc_ev == gold):
            fails.append(f"{tag}: r_gold 与 correct_raw 定义不一致")

        # ---- 物理违规：故意不一致的那一处 ----
        input_pv = checker.check_physics(feats_ev, [])
        model_pv = [v for v in pv_ev if v not in set(input_pv)]
        if kind == "antihall":
            counts["antihall"] += 1
            if input_pv:
                counts["input_physics_nonempty_antihall"] += 1
            else:
                fails.append(f"{tag}: antihall 输入却无固有物理违规 —— "
                             f"数据可能又被 GVIO 类 bug 毒化")
        else:
            counts["non_antihall"] += 1
            if input_pv:
                fails.append(f"{tag}: 非 antihall 输入却自带物理违规 {input_pv}；"
                             f"两种 physics 定义在该行不等价")
            else:
                counts["input_physics_empty_non_antihall"] += 1
            if model_pv == pv_ev:
                counts["physics_equal_non_antihall"] += 1
            else:
                fails.append(f"{tag}: 非 antihall 行 physics 竟不等价")

    ok = not fails
    result = {
        "dataset": args.dataset, "passed": ok, "n_failures": len(fails),
        "failures": fails[:30], "counts": counts,
        "note": "antihall 行的 physics 口径为故意不一致，已单独断言其前提成立",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(counts, ensure_ascii=False, indent=2))
    if not ok:
        print(f"\n✗ 一致性失败 {len(fails)} 条（G0 第 3 项不通过）：")
        for f in fails[:15]:
            print("  -", f)
        return 1
    print(f"\n✓ {counts['n']} 条逐条一致："
          f"extract_features / numbers_in / JSON 解析 / evidence_faithful / "
          f"check_kg / r_gold 全等；")
    print(f"  非 antihall {counts['non_antihall']} 行输入无固有物理违规 ⇒ physics 两种定义等价；")
    print(f"  antihall {counts['antihall']} 行输入均自带违规 ⇒ 减法是必要的。")
    print(f"留痕 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
