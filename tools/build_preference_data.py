#!/usr/bin/env python3
"""rollouts → DPO 偏好对 / RM 训练集。

schema 与 action/feedback.py:64 `build_preference_pairs()` 兼容
-----------------------------------------------------------
那里返回 `order_id / prompt / rejected / actual_root_cause / model_root_cause`，
docstring 明写「chosen 具体 JSON 由 MS7 组装」—— 本脚本就是那个 MS7 组装步骤。
保持同名字段，将来真实产线反馈接进来时，同一套 DPO 脚本可直接吃两种来源。
`prompt` 沿用 feedback.py 的语义：**模型输入的 user 文本**，不是套过模板的串；
套模板在 tools/train_rl.py 里做（G0 第 2 项断言其逐 token 正确），渲染只发生在
有 tokenizer 的那一侧。

三类对子的取舍
--------------
1. **组内有对有错**：chosen=argmax R，rejected=argmin R，要求 margin ≥ 0.15。
2. **组内全错**：这类组恰好是「该降置信度」的最强校准信号，丢掉可惜。但若两
   侧根因不同，DPO 会顺带把某个**错根因**的似然推上去。故只在
   `chosen.root_cause == rejected.root_cause` 时保留 —— 两侧根因相同，梯度就
   只能作用在置信度/证据/SOP 上，不会改变根因偏好。
3. **组内全对且无方差**（多为 clear）：自然不产出对子，符合预期，不强凑。

合成负例（`--synthetic`，兜底）
------------------------------
四种构造：错根因 / 判错却高置信 / 编造证据数值 / 破坏 JSON。**每条都用同一个
奖励函数重新打分验证** margin 确实达标，不达标就丢 —— 合成数据也必须过奖励
这道唯一真源，不能靠"想当然更差"。信息量低（模型本来不产出这种输出），按
方案只用于补 antihall 与校准覆盖，不作主力，统计时与 on-policy 分开。

RLAIF（`--judge-url/--judge-model`）本轮**不跑**，仅预留参数位并写进清单。

用法：
    conda run -n mff_agent python tools/build_preference_data.py \
        --rollouts logs/rl_minicpm5_20260924/rollouts/rollouts.jsonl \
        --out-dir data/rl --min-margin 0.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.rl_reward import (  # noqa: E402
    DATA_BAD, REQUIRED_KEYS, THRESHOLD_CONSTANTS, RewardScorer, numbers_in, parse_output,
)

G1_MIN_PAIRS = 500          # docs/SFT_STATUS.md:26 记载的 MS7 前置条件
G1_MIN_ON_POLICY = 300


def input_key(msgs: list[dict]) -> str:
    """与 tools/build_rl_prompts.py:79 同口径的输入指纹。"""
    keep = [m for m in msgs if m["role"] != "assistant"]
    return json.dumps(keep, ensure_ascii=False, sort_keys=True)


def splice_json(text: str, obj: dict) -> str:
    """把改写后的 JSON 塞回原文本的 JSON 位置，保留 JSON 之前的自然语言部分。"""
    m = re.search(r"\{.*\}", text or "", re.S)
    body = json.dumps(obj, ensure_ascii=False, indent=2)
    if not m:
        return body
    return text[:m.start()] + body + text[m.end():]


def untraceable(value: float, input_nums: list[float]) -> float:
    """造一个在输入里溯源不到的数值（跳开阈值常量与 ±5%/0.15 容差）。"""
    cand = round(abs(value) * 1.37 + 7.3, 2) if value else 137.3
    for _ in range(50):
        ok = cand not in THRESHOLD_CONSTANTS and all(
            abs(cand - n) > max(0.15, abs(n) * 0.05) for n in input_nums)
        if ok:
            return cand
        cand = round(cand * 1.31 + 3.7, 2)
    return cand


class SyntheticMaker:
    """从一条高分 chosen 派生四类劣化负例，逐条经奖励函数验证后才采用。"""

    KINDS = ("wrong_root_cause", "overconfident_wrong", "fabricated_evidence", "broken_json")

    def __init__(self, scorer: RewardScorer, rng: random.Random):
        self.scorer = scorer
        self.rng = rng
        self.faults = list(scorer._fault_names)

    def _other_fault(self, gold: str | None) -> str:
        cands = [f for f in self.faults if f != gold]
        return self.rng.choice(cands) if cands else DATA_BAD

    def make(self, kind: str, chosen_text: str, gold: str, prompt: str) -> str | None:
        obj = parse_output(chosen_text)
        if obj is None:
            return None
        obj = json.loads(json.dumps(obj))       # 深拷贝，别改到 chosen

        if kind == "wrong_root_cause":
            obj["root_cause"] = self._other_fault(gold)
        elif kind == "overconfident_wrong":
            obj["root_cause"] = self._other_fault(gold)
            obj["confidence"] = 0.95
        elif kind == "fabricated_evidence":
            ev = obj.get("evidence")
            if not isinstance(ev, list) or not ev:
                return None
            input_nums = numbers_in(prompt)
            i = self.rng.randrange(len(ev))
            item = str(ev[i])
            m = re.search(r"-?\d+(?:\.\d+)?", item)
            if not m:
                return None
            fake = untraceable(float(m.group(0)), input_nums)
            ev[i] = item[:m.start()] + str(fake) + item[m.end():]
            obj["evidence"] = ev
        elif kind == "broken_json":
            obj.pop(self.rng.choice(list(REQUIRED_KEYS[1:])), None)   # 保留 root_cause 便于统计
        else:
            raise ValueError(kind)

        return splice_json(chosen_text, obj)


def build_pairs_for_group(group: dict, min_margin: float, pairs_per_group: int,
                          keep_all_wrong: bool, stats: Counter) -> list[dict]:
    """一个 prompt 组 → 0..pairs_per_group 个 on-policy 对子。"""
    samples = [s for s in group.get("samples", [])]
    if len(samples) < 2:
        stats["group_too_few_samples"] += 1
        return []
    order = sorted(range(len(samples)), key=lambda i: samples[i]["reward"])
    out: list[dict] = []
    lo, hi = 0, len(order) - 1
    while lo < hi and len(out) < pairs_per_group:
        rej, cho = samples[order[lo]], samples[order[hi]]
        lo, hi = lo + 1, hi - 1
        margin = cho["reward"] - rej["reward"]
        if margin < min_margin:
            stats["pair_margin_too_small"] += 1
            continue
        if not cho["correct"]:
            # 组内（这一对里）chosen 也判错：只在两侧根因相同时保留
            if not keep_all_wrong:
                stats["pair_dropped_chosen_wrong"] += 1
                continue
            if cho["root_cause"] != rej["root_cause"]:
                stats["pair_dropped_all_wrong_diff_rc"] += 1
                continue
            stats["pair_all_wrong_same_rc_kept"] += 1
        out.append({"chosen": cho, "rejected": rej, "margin": margin,
                    "chosen_origin": "on_policy", "rejected_origin": "on_policy"})
    return out


def as_record(group: dict, chosen_text: str, rejected_text: str, margin: float,
              chosen_origin: str, rejected_origin: str, chosen_s: dict | None,
              rejected_s: dict | None, synthetic_kind: str | None = None) -> dict:
    prompt = next(m["content"] for m in group["messages"] if m["role"] == "user")
    src = ("on_policy" if chosen_origin == "on_policy" and rejected_origin == "on_policy"
           else "synthetic")
    return {
        # --- 与 action/feedback.py:64 同名字段 ---
        "order_id": f"rl-{group['idx']}-{src}-{synthetic_kind or 'argmaxmin'}",
        "prompt": prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "actual_root_cause": group["gold"],
        "model_root_cause": (rejected_s or {}).get("root_cause"),
        # --- 本轮新增留痕 ---
        "source": src,
        "chosen_origin": chosen_origin,
        "rejected_origin": rejected_origin,
        "synthetic_kind": synthetic_kind,
        "margin": round(margin, 6),
        "chosen_reward": (chosen_s or {}).get("reward"),
        "rejected_reward": (rejected_s or {}).get("reward"),
        "chosen_parts": (chosen_s or {}).get("parts"),
        "rejected_parts": (rejected_s or {}).get("parts"),
        "chosen_confidence": (chosen_s or {}).get("confidence"),
        "rejected_confidence": (rejected_s or {}).get("confidence"),
        "chosen_correct": (chosen_s or {}).get("correct"),
        "rejected_correct": (rejected_s or {}).get("correct"),
        "kind": group.get("kind"),
        "seed": group.get("seed"),
        "prompt_idx": group["idx"],
        "messages": group["messages"],      # 供 train_rl.py 套模板
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out-dir", default="data/rl")
    ap.add_argument("--prefix", default="prefs")
    ap.add_argument("--min-margin", type=float, default=0.15)
    ap.add_argument("--pairs-per-group", type=int, default=1)
    ap.add_argument("--holdout", type=float, default=0.10,
                    help="RM 准确率留出比例，**按 prompt 组**切分（防同 prompt 泄漏）")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="合成负例上限条数（0=不合成）；仅在 on-policy 不足时使用")
    ap.add_argument("--synthetic-kinds", default="antihall,boundary,composite",
                    help="只给这些 kind 补合成负例")
    ap.add_argument("--gold-dataset", default="data/rl/prompts_gold.jsonl",
                    help="组内无正确样本时，用金标 assistant 当 chosen")
    ap.add_argument("--no-keep-all-wrong", action="store_true",
                    help="丢弃 chosen 也判错的对子（默认保留，但要求两侧根因相同）")
    ap.add_argument("--seed", type=int, default=42)
    # RLAIF 预留位（本轮不跑）
    ap.add_argument("--judge-url", default="")
    ap.add_argument("--judge-model", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    scorer = RewardScorer()
    roll_path = Path(args.rollouts)
    groups = [json.loads(l) for l in roll_path.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    groups = [g for g in groups if "error" not in g]
    if not groups:
        print(f"没有可用 rollout：{roll_path}", file=sys.stderr)
        return 2

    stats: Counter = Counter()
    records: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    # ---------- 1. on-policy 对子 ----------
    for g in groups:
        for p in build_pairs_for_group(g, args.min_margin, args.pairs_per_group,
                                       not args.no_keep_all_wrong, stats):
            rec = as_record(g, p["chosen"]["text"], p["rejected"]["text"], p["margin"],
                            "on_policy", "on_policy", p["chosen"], p["rejected"])
            k = (rec["prompt"], rec["chosen"], rec["rejected"])
            if k in seen or rec["chosen"] == rec["rejected"]:
                stats["pair_duplicate"] += 1
                continue
            seen.add(k)
            records.append(rec)

    n_on_policy = len(records)
    print(f"on-policy 对子 {n_on_policy}（组 {len(groups)}，"
          f"margin<{args.min_margin} 丢弃 {stats['pair_margin_too_small']}）")

    # ---------- 2. 合成负例（兜底） ----------
    n_synth = 0
    if args.synthetic > 0:
        want_kinds = {k.strip() for k in args.synthetic_kinds.split(",") if k.strip()}
        gold_map: dict[str, str] = {}
        gp = Path(args.gold_dataset)
        if gp.exists():
            for line in gp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                ga = next((m["content"] for m in row["messages"]
                           if m["role"] == "assistant"), None)
                if ga:
                    gold_map[input_key(row["messages"])] = ga
        maker = SyntheticMaker(scorer, rng)
        cands = [g for g in groups if g.get("kind") in want_kinds]
        rng.shuffle(cands)
        for g in cands:
            if n_synth >= args.synthetic:
                break
            prompt = next(m["content"] for m in g["messages"] if m["role"] == "user")
            best = max(g["samples"], key=lambda s: s["reward"]) if g["samples"] else None
            chosen_origin, chosen_text, chosen_s = "on_policy", None, None
            if best and best["correct"] and best["json_ok"]:
                chosen_text, chosen_s = best["text"], best
            else:
                ga = gold_map.get(input_key(g["messages"]))
                if not ga:
                    stats["synth_no_chosen_available"] += 1
                    continue
                s = scorer.score(prompt, ga, g["gold"])
                chosen_origin = "gold"
                chosen_text = ga
                chosen_s = {"text": ga, "reward": round(s.total, 6), "parts": s.parts,
                            "root_cause": s.root_cause, "confidence": s.confidence,
                            "correct": s.correct, "json_ok": s.json_ok}
            for kind in maker.KINDS:
                if n_synth >= args.synthetic:
                    break
                rej_text = maker.make(kind, chosen_text, g["gold"], prompt)
                if rej_text is None or rej_text == chosen_text:
                    stats[f"synth_build_failed_{kind}"] += 1
                    continue
                rs = scorer.score(prompt, rej_text, g["gold"])
                margin = chosen_s["reward"] - rs.total
                if margin < args.min_margin:          # 合成也得过奖励这道真源
                    stats[f"synth_margin_too_small_{kind}"] += 1
                    continue
                rej_s = {"reward": round(rs.total, 6), "parts": rs.parts,
                         "root_cause": rs.root_cause, "confidence": rs.confidence,
                         "correct": rs.correct, "json_ok": rs.json_ok}
                rec = as_record(g, chosen_text, rej_text, margin, chosen_origin,
                                "synthetic", chosen_s, rej_s, synthetic_kind=kind)
                k = (rec["prompt"], rec["chosen"], rec["rejected"])
                if k in seen:
                    stats["pair_duplicate"] += 1
                    continue
                seen.add(k)
                records.append(rec)
                n_synth += 1
                stats[f"synth_kept_{kind}"] += 1
        print(f"合成负例 {n_synth}")

    if not records:
        print("✗ 一个对子都没产出：回到 §4 提温/加 k/加权难例。", file=sys.stderr)
        return 1

    # ---------- 3. 按 prompt 组切留出集（不按对子，防同 prompt 泄漏到 RM 评估） ----------
    idxs = sorted({r["prompt_idx"] for r in records})
    rng.shuffle(idxs)
    n_hold = int(round(len(idxs) * args.holdout))
    hold_idx = set(idxs[:n_hold])
    train = [r for r in records if r["prompt_idx"] not in hold_idx]
    held = [r for r in records if r["prompt_idx"] in hold_idx]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, rows: list[dict]) -> str:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    p_train = out_dir / f"{args.prefix}_train.jsonl"
    p_held = out_dir / f"{args.prefix}_heldout.jsonl"
    digests = {p_train.name: dump(p_train, train), p_held.name: dump(p_held, held)}

    margins = [r["margin"] for r in records]
    by_source = Counter(r["source"] for r in records)
    g1_pass = len(records) >= G1_MIN_PAIRS and n_on_policy >= G1_MIN_ON_POLICY
    manifest = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "rollouts": str(roll_path.resolve()),
        "rollouts_sha256": hashlib.sha256(roll_path.read_bytes()).hexdigest(),
        "n_groups": len(groups),
        "params": {"min_margin": args.min_margin, "pairs_per_group": args.pairs_per_group,
                   "holdout": args.holdout, "synthetic_cap": args.synthetic,
                   "synthetic_kinds": args.synthetic_kinds,
                   "keep_all_wrong_same_rc": not args.no_keep_all_wrong,
                   "seed": args.seed},
        "rlaif": {"enabled": False, "judge_url": args.judge_url or None,
                  "judge_model": args.judge_model or None,
                  "note": "RLAIF 本轮不在范围内，参数位仅预留，未参与任何打标"},
        "counts": {"total": len(records), "on_policy": n_on_policy, "synthetic": n_synth,
                   "by_source": dict(by_source),
                   "by_kind": dict(Counter(r["kind"] for r in records)),
                   "by_synthetic_kind": dict(Counter(
                       r["synthetic_kind"] for r in records if r["synthetic_kind"])),
                   "chosen_origin": dict(Counter(r["chosen_origin"] for r in records)),
                   "train": len(train), "heldout": len(held),
                   "heldout_prompt_groups": len(hold_idx)},
        "margin": {"mean": round(statistics.fmean(margins), 6),
                   "min": round(min(margins), 6), "max": round(max(margins), 6),
                   "median": round(statistics.median(margins), 6)},
        "filter_stats": dict(stats),
        "g1_gate": {"min_pairs": G1_MIN_PAIRS, "min_on_policy": G1_MIN_ON_POLICY,
                    "pairs": len(records), "on_policy": n_on_policy, "passed": g1_pass},
        "sha256": digests,
        "schema_note": "字段与 action/feedback.py:64 build_preference_pairs() 兼容，"
                       "prompt 为 user 文本（套模板在 tools/train_rl.py）",
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"margin mean={manifest['margin']['mean']:.4f} "
          f"min={manifest['margin']['min']:.4f} max={manifest['margin']['max']:.4f}")
    print(f"\n{'✓' if g1_pass else '✗'} 决策门 G1：对子 {len(records)}/{G1_MIN_PAIRS}，"
          f"on-policy {n_on_policy}/{G1_MIN_ON_POLICY}")
    print(f"  {p_train} ({len(train)})  |  {p_held} ({len(held)}，供 RM 准确率)")
    if not g1_pass:
        print("  G1 不过：按方案回到 §4 提温 / 加 k / 加权难例，不硬上合成数据。",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
