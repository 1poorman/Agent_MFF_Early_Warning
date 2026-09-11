"""MS2 SFT 数据审计：L0.1 七项自动审计（docs/SFT_TEST_PLAN.md）。

用例：
  1. JSON 可解析（复用 root_cause._parse 同款正则+loads）
  2. root_cause == 金标签（meta.gold）
  3. SOP ⊆ 知识图谱 actions_for_fault(金标签)（无故障/数据异常类除外）
  4. 证据数值忠实：evidence 中的数值均可在输入 stats 中找到（±10% 容差）
  5. 配比达标：各类别数量 >= 阈值（--min 检查 train 集）
  6. 防泄漏：train/val 与 eval 的 seed 集合不相交
  7. GVIO 违规落地：antihall 声称的物理违规必须真出现在输入特征中且确实越界
     （MS5 v1 漏检项：20 条 antihall 声称"出水温度=5.0 越界"但输入里根本没这个值）

用法：
  conda run -n mff_agent python tools/audit_sft_data.py [--dir data/sft] [--sample 0.05]
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning.knowledge_graph import KnowledgeGraph

QUOTA_MIN = {"clear": 1000, "boundary": 500, "transition": 250,
             "composite": 150, "misleading": 150, "antihall": 80}

PHYS_LIMITS = {"湿度": (0.0, 100.0), "压力": (0.0, 1000.0), "流量": (0.0, 50.0),
               "出水温度": (0.0, 100.0), "进水温度": (0.0, 60.0), "水箱液位": (0.0, 500.0)}


def parse_output(text: str):
    """与 root_cause._parse 同款：提取 assistant 输出尾部 JSON。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def numbers_in(text):
    """提取带符号数值（'PQ偏移 -0.1%' 的负号不能丢，否则与真值 -0.1 对不上）。"""
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]


def audit_file(kg, path, label):
    fails = []
    kinds = Counter()
    n = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            n += 1
            s = json.loads(line)
            meta = s["meta"]
            kinds[meta["kind"]] += 1
            assistant = s["messages"][2]["content"]
            user = s["messages"][1]["content"]
            where = f"{label}#L{i+1}"

            # 1. JSON 可解析
            out = parse_output(assistant)
            if out is None:
                fails.append(f"{where} [JSON] assistant 输出无合法 JSON")
                continue
            # 2. 根因 == 金标签
            if out.get("root_cause") != meta["gold"]:
                fails.append(f"{where} [GOLD] root_cause={out.get('root_cause')} != {meta['gold']}")
            # 3. SOP 一致（图谱域故障才检查）
            if meta["gold"] in kg.fault_names():
                allowed = set(kg.actions_for_fault(meta["gold"]))
                sop = out.get("sop") or []
                bad = [x for x in sop if x not in allowed]
                if bad:
                    fails.append(f"{where} [SOP] 越界处置项: {bad}")
            # 4. 证据数值忠实：evidence 数值须在输入 stats 值域内（±10% 容差）
            #    antihall 类引用的是设计内的违规值（已注入 prompt），豁免
            stat_vals = [float(v) for v in meta["stats"].values()
                         if isinstance(v, (int, float))]
            if meta["kind"] != "antihall":
                for ev in out.get("evidence") or []:
                    for num in numbers_in(ev):
                        if not any(abs(num - v) <= 0.1 * max(abs(v), 1.0) for v in stat_vals):
                            fails.append(f"{where} [FAITH] evidence 数值 {num} 未见于输入 stats")
                # 4b. CoT 里的数值同样忠实（推理链前半段）；置信度是模型自校准输出，豁免
                cot = assistant.rsplit("{", 1)[0]
                conf_val = out.get("confidence")
                for num in numbers_in(cot):
                    # 阈值常数（3/10/6.4/70/65/100 等）豁免：来自知识而非输入
                    if num in (3, 10, 6.4, 70, 65, 4, 50, 55, 100, 120, 2.2):
                        continue
                    if conf_val is not None and abs(num - float(conf_val)) < 1e-9:
                        continue
                    if not any(abs(num - v) <= 0.1 * max(abs(v), 1.0) for v in stat_vals):
                        fails.append(f"{where} [FAITH] CoT 数值 {num} 未见于输入 stats")

            # 7. [GVIO] antihall 声称的违规必须真出现在输入的实时特征中，且确实越界
            if meta["kind"] == "antihall":
                mf = re.search(r"【实时异常特征】\s*(\{.*?\})", user, re.S)
                feats = {}
                if mf:
                    try:
                        feats = json.loads(mf.group(1))
                    except json.JSONDecodeError:
                        feats = {}
                nums = [float(v) for v in feats.values() if isinstance(v, (int, float))]
                claimed = [float(x) for x in re.findall(r"=(-?\d+\.?\d*)", assistant)]
                if not claimed or not any(any(abs(c - n) < 1e-6 for n in nums) for c in claimed):
                    fails.append(f"{where} [GVIO] antihall 声称违规值 {claimed} 未见于输入特征")
                elif not (
                    any(k in feats and not (lo <= float(feats[k]) <= hi)
                        for k, (lo, hi) in PHYS_LIMITS.items())
                    or float(feats.get("出水温度", 99)) < float(feats.get("进水温度", 0))
                ):
                    fails.append(f"{where} [GVIO] antihall 输入无物理越界却标'数据质量异常'")
    return n, kinds, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/sft")
    args = ap.parse_args()
    d = Path(args.dir)
    kg = KnowledgeGraph()

    all_fails = []
    summaries = {}
    for name in ["train", "val", "eval"]:
        p = d / f"{name}.jsonl"
        if not p.exists():
            print(f"跳过缺失文件: {p}")
            continue
        n, kinds, fails = audit_file(kg, p, name)
        summaries[name] = (n, kinds)
        all_fails += fails

    for name, (n, kinds) in summaries.items():
        print(f"{name}: {n} 条 {dict(kinds)}")

    # 5. 配比达标（train）
    if "train" in summaries:
        _, kinds = summaries["train"]
        for k, vmin in QUOTA_MIN.items():
            if kinds.get(k, 0) < vmin:
                all_fails.append(f"[QUOTA] train.{k}={kinds.get(k, 0)} < {vmin}")

    # 6. seed 防泄漏
    def seeds_of(name):
        p = d / f"{name}.jsonl"
        if not p.exists():
            return set()
        return {json.loads(l)["meta"]["seed"] for l in open(p, encoding="utf-8")}
    tr, ev = seeds_of("train") | seeds_of("val"), seeds_of("eval")
    if tr & ev:
        all_fails.append(f"[LEAK] train/val 与 eval seed 相交: {sorted(tr & ev)}")

    if all_fails:
        print(f"\n审计失败 {len(all_fails)} 项：")
        for f in all_fails[:50]:
            print(" ", f)
        sys.exit(1)
    print("\n审计通过：JSON/GOLD/SOP/FAITH/QUOTA/LEAK 全绿 ✅")


if __name__ == "__main__":
    main()
