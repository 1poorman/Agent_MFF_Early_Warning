#!/usr/bin/env python3
"""构造 RL 用 prompt 池与独立压力集（全新种子段，与 SFT train/eval 零重叠）。

种子分段（三段互不相交，这是本脚本最重要的不变量）
--------------------------------------------------
    SFT train/val : 故障 101–106      正常/异常 201–208   （已冻结）
    SFT eval      : 故障 901–902      正常/异常 951–952   （已冻结，本轮主对比面）
    RL  池        : 故障 301–306      正常/异常 401–408   ← 本脚本
    RL  压力集    : 故障 307–308      正常/异常 409–411   ← 本脚本

RL 池与压力集**也**互不相交：压力集若与训练 prompt 同种子，就等于在训练数据上
自测，辅助证据立刻失效。

产物
----
    data/rl/train.jsonl          RL 原始池（与 SFT train 同规模，供 audit 七项）
    data/rl/val.jsonl            池内 10% 留出（RM 训练/调参用）
    data/rl/eval.jsonl           压力集合并（供 audit 的 LEAK 检查比对两段种子）
    data/rl/prompts.jsonl        去重 + 难例加权后的 RL prompt 池（训练实际输入）
    data/rl/prompts_gold.jsonl   同批样本带金标 assistant（合成负例取 chosen 用）
    data/rl/stress_antihall.jsonl  n≈100，量程/物理越界拒诊
    data/rl/stress_calib.jsonl     n≈200，混合难例，专测校准
    data/rl/build_manifest.json    全部计数、比例、SHA256、反泄漏断言结果

为什么要加权：eval 集 antihall 只有 10 条，1 条 = 10pp，撑不起"拒诊改善"的结论。
prompt 池刻意向 RL 靶点（拒诊、校准）倾斜，同时留 clear/transition 作防遗忘护栏。

用法（生成走物理仿真，约数十分钟）：
    conda run -n mff_agent python tools/build_rl_prompts.py --out-dir data/rl
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 新种子段（与 SFT 的 101-106/201-208 与 901-902/951-952 均不相交）
POOL_FAULT_SEEDS = [301, 302, 303, 304, 305, 306]
POOL_NORMAL_SEEDS = [401, 402, 403, 404, 405, 406, 407, 408]
STRESS_FAULT_SEEDS = [307, 308]
STRESS_NORMAL_SEEDS = [409, 410, 411]

# RL prompt 池的类别配比（刻意偏向拒诊与校准）
POOL_PROPORTIONS = {
    "antihall": 0.20,     # 主靶点：数据质量拒诊
    "boundary": 0.30,     # 校准信号最丰富（模型真会犹豫的区间）
    "composite": 0.15,    # 现存唯一稳定失败模式（主根因口径）
    "misleading": 0.15,   # 抗工单误导
    "transition": 0.10,   # 防遗忘护栏
    "clear": 0.10,        # 防遗忘护栏
}
STRESS_CALIB_MIX = {"boundary": 90, "composite": 40, "misleading": 30,
                    "clear": 25, "transition": 15}
STRESS_ANTIHALL_N = 100


def load_generator():
    """按路径加载 tools/gen_sft_data.py（不修改它：已发表报告的复现命令引用该文件）。"""
    spec = importlib.util.spec_from_file_location(
        "_gen_sft_data", ROOT / "tools" / "gen_sft_data.py")
    gsd = importlib.util.module_from_spec(spec)
    sys.modules["_gen_sft_data"] = gsd
    spec.loader.exec_module(gsd)
    # main() 里设置的两个模块级全局，build_sample 依赖它们
    gsd.MAINT_TEXT = gsd.default_maintenance_log().to_prompt_text(days=60)
    gsd.SCHED_TEXT = gsd.default_operating_schedule().to_prompt_text()
    return gsd


def input_key(sample: dict) -> str:
    """完整 system+user 作为输入指纹（与 tools/summarize_lora_comparison.py:30 同口径）。"""
    msgs = [m for m in sample["messages"] if m["role"] != "assistant"]
    return json.dumps(msgs, ensure_ascii=False, sort_keys=True)


def dedupe(samples: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in samples:
        k = input_key(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_segment(gsd, kg, fault_seeds, normal_seeds, label):
    """生成一个种子段的六类样本（返回 {kind: [sample...]}）。"""
    print(f"\n=== [{label}] 故障池 seeds={fault_seeds} ===", flush=True)
    fault_pool = gsd.gen_fault_pool(kg, fault_seeds)
    print(f"  clear={len(fault_pool['clear'])} boundary={len(fault_pool['boundary'])}",
          flush=True)

    print(f"=== [{label}] 正常窗口 seeds={normal_seeds} ===", flush=True)
    nwin = gsd.gen_normal_windows(normal_seeds)
    transitions = gsd.gen_transition(kg, nwin)
    print(f"  normal_windows={len(nwin)} transition={len(transitions)}", flush=True)

    print(f"=== [{label}] 复合故障 ===", flush=True)
    composites = gsd.gen_composite_pool(kg, fault_seeds)
    print(f"  composite={len(composites)}", flush=True)

    print(f"=== [{label}] 误导工单 / 防幻觉注入 ===", flush=True)
    # gen_misleading 会 shuffle 传入的 list，给副本
    misleading = gsd.gen_misleading(kg, list(fault_pool["clear"]), 10**6)
    antihall = gsd.gen_antihall(kg, nwin, 10**6)
    print(f"  misleading={len(misleading)} antihall={len(antihall)}", flush=True)

    return {"clear": fault_pool["clear"], "boundary": fault_pool["boundary"],
            "transition": transitions, "composite": composites,
            "misleading": misleading, "antihall": antihall}


def scaled_take(avail: dict[str, list], proportions: dict[str, float],
                target: int, rng: random.Random):
    """在保持配比的前提下取尽可能接近 target 的样本量（供给不足则整体等比缩小）。"""
    feasible = min(len(avail.get(k, [])) / p for k, p in proportions.items() if p > 0)
    n = int(min(target, feasible))
    out, taken = [], {}
    for kind, p in proportions.items():
        want = int(round(p * n))
        pool = list(avail.get(kind, []))
        rng.shuffle(pool)
        got = pool[:want]
        taken[kind] = len(got)
        out += got
    rng.shuffle(out)
    return out, taken, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/rl")
    ap.add_argument("--target", type=int, default=1200, help="prompt 池目标条数")
    ap.add_argument("--sft-dir", default="data/sft")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.glob("*.jsonl")):
        print(f"拒绝覆盖已有数据：{out_dir}（换一个 --out-dir）", file=sys.stderr)
        return 2

    gsd = load_generator()
    kg = gsd.KnowledgeGraph()
    rng = random.Random(args.seed)

    pool = build_segment(gsd, kg, POOL_FAULT_SEEDS, POOL_NORMAL_SEEDS, "RL 池")
    stress = build_segment(gsd, kg, STRESS_FAULT_SEEDS, STRESS_NORMAL_SEEDS, "压力集")

    # ---- RL 原始池：与 SFT train 同规模同配比，让 audit 的 QUOTA 项有意义 ----
    # 配额取 max(SFT 同规模配额, prompt 池按配比所需 × 1.15 余量)。
    # 第一次构建踩的坑：只按 gsd.QUOTA 取，antihall 上限 100，于是 prompt 池被
    # 卡在 100/0.20 = 500 条，不到目标的一半 —— 而生成阶段明明产出了 296 条
    # antihall，是被配额扔掉的。QUOTA 对 audit 是**下限**（audit_sft_data.py 的
    # QUOTA_MIN），取多不违规；取少却直接压垮本轮主靶点的样本量。
    pool_quota = {kind: max(quota, math.ceil(POOL_PROPORTIONS.get(kind, 0.0)
                                             * args.target * 1.15))
                  for kind, quota in gsd.QUOTA.items()}
    raw = []
    for kind, quota in pool_quota.items():
        got = gsd.take(list(pool[kind]), quota, rng)
        raw += got
        short = "" if len(got) >= quota else "   (受生成供给上限所限)"
        print(f"  原始池 {kind}: {len(got)}/{quota}{short}", flush=True)
    rng.shuffle(raw)
    n_val = max(len(raw) // 10, 1)
    rl_val, rl_train = raw[:n_val], raw[n_val:]

    # ---- 压力集（独立种子段） ----
    ah = dedupe(list(stress["antihall"]))
    rng.shuffle(ah)
    stress_antihall = ah[:STRESS_ANTIHALL_N]

    calib = []
    for kind, want in STRESS_CALIB_MIX.items():
        cand = dedupe(list(stress[kind]))
        rng.shuffle(cand)
        calib += cand[:want]
    rng.shuffle(calib)

    # ---- RL prompt 池：池内去重 + 难例加权 ----
    pool_dedup = {k: dedupe(v) for k, v in pool.items()}
    # 只保留真正进入 train/val 的输入（避免用到未纳入原始池的窗口）
    in_raw = {input_key(s) for s in raw}
    pool_dedup = {k: [s for s in v if input_key(s) in in_raw] for k, v in pool_dedup.items()}
    prompts_samples, taken, n_pool = scaled_take(pool_dedup, POOL_PROPORTIONS,
                                                 args.target, rng)
    prompts = [{"messages": [m for m in s["messages"] if m["role"] != "assistant"],
                "meta": s["meta"]} for s in prompts_samples]
    # 同一批样本**带金标 assistant** 的副本：合成负例在组内无正确样本时要用它当
    # chosen。放这里而不是回 train.jsonl 里查，是因为 prompt 池会横跨 train/val
    # 切分，跨文件查必然漏一部分。
    prompts_gold = [{"messages": s["messages"], "meta": s["meta"]}
                    for s in prompts_samples]

    # ---- 反泄漏断言（任一不过即退出） ----
    def keys_of(path: Path) -> set[str]:
        if not path.exists():
            return set()
        return {input_key(json.loads(l))
                for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}

    sft = Path(args.sft_dir)
    sft_eval_keys = keys_of(sft / "eval.jsonl")
    sft_train_keys = keys_of(sft / "train.jsonl") | keys_of(sft / "val.jsonl")
    prompt_keys = {input_key(s) for s in prompts}
    stress_keys = {input_key(s) for s in stress_antihall + calib}

    checks = {
        "prompts_vs_sft_eval": len(prompt_keys & sft_eval_keys),
        "stress_vs_sft_eval": len(stress_keys & sft_eval_keys),
        "stress_vs_prompts": len(stress_keys & prompt_keys),
        "prompts_vs_sft_train_val": len(prompt_keys & sft_train_keys),
    }
    pool_seeds = {s["meta"]["seed"] for s in raw}
    stress_seeds = {s["meta"]["seed"] for s in stress_antihall + calib}
    sft_seeds = set()
    for name in ("train", "val", "eval"):
        p = sft / f"{name}.jsonl"
        if p.exists():
            sft_seeds |= {json.loads(l)["meta"]["seed"]
                          for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    checks["pool_seeds_vs_sft_seeds"] = len(pool_seeds & sft_seeds)
    checks["stress_seeds_vs_sft_seeds"] = len(stress_seeds & sft_seeds)
    checks["stress_seeds_vs_pool_seeds"] = len(stress_seeds & pool_seeds)

    print("\n=== 反泄漏断言（全部必须为 0） ===")
    for k, v in checks.items():
        print(f"  {k}: {v}{'' if v == 0 else '   ✗'}")
    if any(checks.values()):
        print("\n✗ 反泄漏断言失败，已中止且未写盘。", file=sys.stderr)
        return 1

    # ---- 落盘 ----
    digests = {
        "train": write_jsonl(out_dir / "train.jsonl", rl_train),
        "val": write_jsonl(out_dir / "val.jsonl", rl_val),
        "eval": write_jsonl(out_dir / "eval.jsonl", stress_antihall + calib),
        "prompts": write_jsonl(out_dir / "prompts.jsonl", prompts),
        "prompts_gold": write_jsonl(out_dir / "prompts_gold.jsonl", prompts_gold),
        "stress_antihall": write_jsonl(out_dir / "stress_antihall.jsonl", stress_antihall),
        "stress_calib": write_jsonl(out_dir / "stress_calib.jsonl", calib),
    }

    manifest = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": {"pool_fault": POOL_FAULT_SEEDS, "pool_normal": POOL_NORMAL_SEEDS,
                  "stress_fault": STRESS_FAULT_SEEDS, "stress_normal": STRESS_NORMAL_SEEDS,
                  "pool_seeds_actual": sorted(pool_seeds),
                  "stress_seeds_actual": sorted(stress_seeds)},
        "generated_available": {k: len(v) for k, v in pool.items()},
        "generated_available_stress": {k: len(v) for k, v in stress.items()},
        "rl_raw_pool": {"n": len(raw), "kinds": dict(Counter(s["meta"]["kind"] for s in raw)),
                        "train": len(rl_train), "val": len(rl_val),
                        "quota_used": pool_quota, "quota_sft_reference": dict(gsd.QUOTA)},
        "prompt_pool": {"n": len(prompts), "target": args.target, "feasible_n": n_pool,
                        "kinds": dict(Counter(s["meta"]["kind"] for s in prompts)),
                        "taken_by_proportion": taken,
                        "proportions_requested": POOL_PROPORTIONS,
                        "proportions_actual": {
                            k: round(v / max(len(prompts), 1), 4)
                            for k, v in Counter(s["meta"]["kind"] for s in prompts).items()}},
        "stress": {"antihall_n": len(stress_antihall), "calib_n": len(calib),
                   "calib_kinds": dict(Counter(s["meta"]["kind"] for s in calib))},
        "leakage_checks": checks,
        "sha256": digests,
        "note": "压力集为辅助证据；主对比面仍是冻结的 data/sft/eval.jsonl",
    }
    (out_dir / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n✓ RL 原始池 train={len(rl_train)} val={len(rl_val)}")
    print(f"✓ prompt 池 {len(prompts)} 条 "
          f"{dict(Counter(s['meta']['kind'] for s in prompts))}")
    print(f"✓ 压力集 antihall={len(stress_antihall)} calib={len(calib)}")
    print(f"✓ 反泄漏全部为 0；清单 {out_dir / 'build_manifest.json'}")
    print(f"\n下一步：conda run -n mff_agent python tools/audit_sft_data.py --dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
