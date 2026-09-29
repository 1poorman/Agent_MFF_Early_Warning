"""MS3 离线评估基建：对 data/sft/eval.jsonl 一键出 L2 全表（docs/SFT_TEST_PLAN.md 第四节）。

与 MS1 基线脚本（tools/ms1_baseline.py）的分工：
  - MS1 走线上实时流组包（_build_prompt 原样，**含**判定规则），测"原模型+提示工程"上限；
  - 本脚本直接使用 SFT 评估集的退化版 prompt（**不含**判定规则）逐条请求任意
    OpenAI 兼容端点，即 MS4/MS5 训练分布，是 SFT 产物的**同分布**评估口径。

指标（测试方案 L2，第 1/2/3/4/5/6/7/8/9 项）：
  1 根因准确率（主口径由 --arbitration 选择，默认关仲裁；故障子集对齐 MS1 口径）/ 总体准确率
  2 混淆对准确率（boundary 子集）
  3 JSON 合法率            ┐
  4 证据忠实率            ├ 硬门槛（任一不过禁止进入 L3/L4）
  5 防幻觉通过率          ┘
  6 置信度校准（判对−判错 / 判错均值与虚高比例）
  7 端到端时延（P50/P95）
  8 无故障误报率（transition 子集）
  9 回归集（历史 case；数据未接入时显式标注）

用法：
  conda run -n mff_agent python tests/eval_sft_model.py \
      --url http://localhost:3761/v1 --model MiniCPM5-2B \
      --dataset data/sft/eval.jsonl --arbitration off
  # 3 次复跑（对齐测试方案复跑规程）：
  for i in 1 2 3; do ... --seed $i >> logs/sft_eval_$i.log; done
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- 环境隔离：评估栈全跑 CPU（GPU0 留给 vLLM），必须在 torch 导入前设置 ----
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

_torch_load_orig = torch.load
def _cpu_torch_load(*a, **k):
    k.setdefault("map_location", "cpu")
    return _torch_load_orig(*a, **k)
torch.load = _cpu_torch_load

from reasoning.knowledge_graph import KnowledgeGraph  # noqa: E402
from reasoning.llm_client import LLMClient  # noqa: E402
from reasoning.root_cause import RootCauseReasoner  # noqa: E402
from reasoning.anti_hallucination import AntiHallucinationChecker  # noqa: E402

NO_FAULT = "无故障（工况切换正常波动）"
DATA_BAD = "数据质量异常（建议人工核查）"
FAULT_KINDS = {"clear", "boundary", "composite", "misleading"}
CONFUSION_KINDS = {"boundary"}          # 气蚀/泄漏 + 堵塞/结垢 混淆带

# 判对/判错样本的置信度安全线
CAL_GAP_MIN = 0.15
CAL_WRONG_MAX = 0.8


# ---------------- 输入解析（prompt 内客观数据） ----------------

def extract_features(prompt: str) -> dict:
    """从退化版 prompt 的【实时异常特征】段还原 features（物理/图谱校验用）。"""
    m = re.search(r"【实时异常特征】\s*(\{.*?\})", prompt, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


def extract_candidates(prompt: str) -> list:
    """还原 prompt 内【知识图谱候选根因】列表（仲裁用）。"""
    m = re.search(r"【知识图谱候选根因】(.+)", prompt)
    if not m:
        return []
    txt = m.group(1).strip()
    if txt.startswith("（"):
        return []
    return [x.strip() for x in txt.split(",") if x.strip()]


def numbers_in(text: str) -> list:
    """提取带符号数值（负号不能丢，与 audit_sft_data.numbers_in 同款）。"""
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]


# 判定阈值常量：模型需内化的领域知识（退化版 prompt 已删"判定规则"节），
# 引用这些整数/常数不计为幻觉；与 tools/audit_sft_data.py 同源并补充 20/7.8。
THRESHOLD_CONSTANTS = {2.2, 3, 4, 6.4, 7.8, 10, 20, 50, 55, 60, 65, 70, 100, 120}


def evidence_faithful(evidence, input_nums):
    """证据中每个数值须能在**输入 prompt**找到（±5% 或 0.15 容差，吸收四舍五入）。

    输入 prompt 含 stats + 实时特征 + 工单日期 + 工况表，即模型全部可见数值；
    阈值常量（THRESHOLD_CONSTANTS）为应内化的领域知识，豁免。
    返回 (ok, bad_nums)。
    """
    bad = []
    for ev in evidence or []:
        for num in numbers_in(str(ev)):
            if num in THRESHOLD_CONSTANTS:
                continue
            if not any(abs(num - v) <= max(0.05 * abs(v), 0.15) for v in input_nums):
                bad.append(num)
    return (not bad), bad


# ---------------- 统计仲裁（镜像 root_cause.diagnose 155~219） ----------------

def apply_arbitration(kg, precheck, rc_llm, conf_llm, features, stats, prompt_cands):
    """含仲裁口径：图谱 Top1 一致性 + 统计强先验 + 三条硬校验。

    与 reasoning/root_cause.py 的 diagnose() 保持同序，仅省略"重试"部分
    （评估只对首次输出做一次确定性修正，便于 on/off 对比）。
    """
    sensors = list(features.keys())
    ranked = kg.faults_for_sensors(sensors)
    candidates = [kg.nodes[fid].name for fid, _ in ranked][:3]
    for extra in list(precheck) + list(prompt_cands):
        if extra in kg.fault_names() and extra not in candidates:
            candidates.append(extra)
    kg_top1 = candidates[0] if candidates else "未知"

    rc, conf = rc_llm, conf_llm
    if rc == kg_top1:
        conf = min(conf + 0.1, 0.98)
    elif rc in candidates and conf >= 0.8:
        pass
    else:
        rc, conf = kg_top1, max(conf - 0.1, 0.5)

    if precheck and rc not in precheck:
        rc = precheck[0]
        conf = max(min(conf_llm, 0.85), 0.7)

    press_std = float(stats.get("压力波动幅度_std_kPa", 0.0) or 0.0)
    hum_mean = float(stats.get("湿度均值_pctRH", 50.0) or 50.0)
    hum_delta = float(stats.get("湿度上升量_pctRH", 0.0) or 0.0)
    pq_off = float(stats.get("PQ特性偏移_pct", 0.0) or 0.0)

    if rc == "管道泄漏" and hum_mean <= 65.0 and hum_delta <= 4.0:
        if press_std > 3.0:
            rc = "水泵气蚀"
        conf = min(conf, 0.75)
    elif rc in ("过滤器堵塞", "线圈结垢") and hum_delta > 4.0:
        rc = "管道泄漏"
        conf = max(min(conf, 0.85), 0.7)
    if rc == "过滤器堵塞" and hum_delta <= 4.0 and press_std <= 3.0 and pq_off >= 10.0:
        rc = "线圈结垢"
        conf = max(min(conf, 0.85), 0.7)
    return rc, conf


# ---------------- 评估主循环 ----------------

def percentile(vals, p):
    if not vals:
        return None
    vs = sorted(vals)
    k = min(len(vs) - 1, int(round((p / 100.0) * (len(vs) - 1))))
    return vs[k]


def run(args):
    ds_path = Path(args.dataset)
    samples = [json.loads(l) for l in open(ds_path, encoding="utf-8") if l.strip()]
    if args.limit:
        samples = samples[:args.limit]

    kg = KnowledgeGraph()
    checker = AntiHallucinationChecker(kg)
    llm = LLMClient(config={
        "url": args.url, "key": "empty", "big_model_name": args.model,
        "enable_thinking": False, "timeout": args.timeout})
    reasoner = RootCauseReasoner(llm=llm, kg=kg)  # 复用 _parse

    # 延迟导入统计预鉴别（server.agents -> server.service，随后者导入 detection）
    from server.agents import WarningAnalysisAgent

    records = []
    t_start = time.time()
    for i, s in enumerate(samples):
        msgs = {m["role"]: m["content"] for m in s["messages"]}
        system, user = msgs.get("system", ""), msgs["user"]
        meta = s.get("meta", {})
        stats = meta.get("stats", {})
        features = extract_features(user)
        prompt_cands = extract_candidates(user)
        precheck = WarningAnalysisAgent._stat_precheck(features, stats)

        t0 = time.time()
        raw, err = "", None
        try:
            raw = llm.chat(user, system=system, max_tokens=args.max_tokens,
                           temperature=args.temperature)
        except Exception as e:  # 网络/服务异常：记录后继续，保证全表可出
            err = f"{type(e).__name__}: {e}"
        latency = time.time() - t0

        parsed = reasoner._parse(raw) if raw else None
        rc_raw = parsed.get("root_cause") if parsed else None
        conf_raw = float(parsed.get("confidence", 0.5)) if parsed else None
        evidence = (parsed.get("evidence") or []) if parsed else []
        sop = (parsed.get("sop") or []) if parsed else []

        # 含仲裁
        rc_arb, conf_arb = (None, None)
        if parsed and rc_raw is not None:
            rc_arb, conf_arb = apply_arbitration(
                kg, precheck, rc_raw, conf_raw, features, stats, prompt_cands)

        # 主口径由 --arbitration 选择（off=关仲裁，测试方案主指标）
        if args.arbitration == "on":
            rc_p, conf_p = rc_arb, conf_arb
        else:
            rc_p, conf_p = rc_raw, conf_raw

        # 证据忠实（输入 stats 值域）
        faith_ok, bad_nums = evidence_faithful(evidence, numbers_in(user))

        # 防幻觉：物理违规零容忍；图谱校验仅对图谱域故障根因
        pv = checker.check_physics(features, evidence)
        kv = checker.check_kg(rc_raw, evidence) if rc_raw in kg.fault_names() else []
        hallu_ok = (not pv) and (not kv)

        gold = meta.get("gold")
        records.append({
            "idx": i, "kind": meta.get("kind"), "seed": meta.get("seed"),
            "gold": gold, "error": err,
            "json_ok": parsed is not None,
            "rc_raw": rc_raw, "conf_raw": conf_raw,
            "rc_arb": rc_arb, "conf_arb": conf_arb,
            "rc": rc_p, "conf": conf_p, "correct": rc_p == gold,
            "correct_raw": rc_raw == gold, "correct_arb": rc_arb == gold,
            "evidence": evidence, "sop": sop,
            "faith_ok": faith_ok, "bad_nums": bad_nums,
            "physics_ok": not pv, "kg_ok": not kv, "hallu_ok": hallu_ok,
            "latency_s": round(latency, 3),
        })
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(samples)}] {meta.get('kind')} "
                  f"gold={gold} raw={rc_raw} ok={rc_raw == gold}", flush=True)

    metrics = summarize(records, kg)
    metrics["wall_s"] = round(time.time() - t_start, 1)
    metrics["n"] = len(records)
    return samples, records, metrics


def summarize(records, kg):
    n = len(records)
    ok = [r for r in records if r["json_ok"]]
    json_rate = len(ok) / max(n, 1)

    fault = [r for r in records if r["kind"] in FAULT_KINDS]
    acc_fault = sum(r["correct"] for r in fault) / max(len(fault), 1)
    acc_fault_raw = sum(r["correct_raw"] for r in fault) / max(len(fault), 1)
    acc_fault_arb = sum(r["correct_arb"] for r in fault) / max(len(fault), 1)
    acc_overall = sum(r["correct"] for r in records) / max(n, 1)

    conf_pair = [r for r in records if r["kind"] in CONFUSION_KINDS]
    acc_confusion = sum(r["correct"] for r in conf_pair) / max(len(conf_pair), 1)

    # 证据忠实：仅统计 JSON 合法且非 antihall（其"输入"本身为故意注入的越界值）
    faith_pool = [r for r in ok if r["kind"] != "antihall"]
    faith_rate = sum(r["faith_ok"] for r in faith_pool) / max(len(faith_pool), 1)

    # 防幻觉：antihall 输入故意物理越界，不计入通过率分母（正确拒诊另行统计）
    hallu_pool = [r for r in ok if r["kind"] != "antihall"]
    hallu_rate = sum(r["hallu_ok"] for r in hallu_pool) / max(len(hallu_pool), 1)

    # 置信度校准（当前主口径，仅 JSON 合法）
    corr = [r["conf"] for r in ok if r["correct"] and r["conf"] is not None]
    wrong = [r["conf"] for r in ok if not r["correct"] and r["conf"] is not None]
    cal = {
        "mean_correct": mean(corr) if corr else None,
        "mean_wrong": mean(wrong) if wrong else None,
        "gap": (mean(corr) - mean(wrong)) if (corr and wrong) else None,
        "wrong_ge_0.8": (sum(1 for c in wrong if c >= CAL_WRONG_MAX) / len(wrong)) if wrong else None,
    }

    lats = [r["latency_s"] for r in ok]
    latency = {
        "p50": percentile(lats, 50), "p95": percentile(lats, 95),
        "mean": mean(lats) if lats else None, "max": max(lats) if lats else None,
    }

    trans = [r for r in records if r["kind"] == "transition"]
    fp = sum(1 for r in trans if r["rc"] != NO_FAULT) / max(len(trans), 1)

    ant = [r for r in records if r["kind"] == "antihall"]
    ant_refuse = sum(1 for r in ant if r["rc"] == DATA_BAD) / max(len(ant), 1)

    # 分层
    by_kind = defaultdict(lambda: [0, 0])
    for r in records:
        by_kind[r["kind"]][0] += 1
        by_kind[r["kind"]][1] += int(r["correct"])
    by_kind = {k: (v[0], v[1] / max(v[0], 1)) for k, v in by_kind.items()}

    by_gold = defaultdict(lambda: [0, 0])
    for r in fault:
        by_gold[r["gold"]][0] += 1
        by_gold[r["gold"]][1] += int(r["correct"])
    by_gold = {k: (v[0], v[1] / max(v[0], 1)) for k, v in by_gold.items()}

    confusion = defaultdict(Counter)
    for r in fault:
        confusion[r["gold"]][r["rc"]] += 1

    hard = {
        "json": json_rate >= 0.99,
        "faith": faith_rate >= 0.98,
        "hallu": hallu_rate >= 1.0,
    }
    return {
        "accuracy": {
            "fault_subset": acc_fault,
            "fault_subset_raw": acc_fault_raw, "fault_subset_arb": acc_fault_arb,
            "overall": acc_overall,
            "overall_correct": sum(r["correct"] for r in records),
            "n_fault": len(fault), "n_overall": n,
        },
        "confusion_pair": {"acc": acc_confusion, "n": len(conf_pair)},
        "json": {"rate": json_rate, "ok": len(ok), "n": n},
        "faith": {"rate": faith_rate, "n": len(faith_pool),
                  "bad": sum(1 for r in faith_pool if not r["faith_ok"])},
        "hallu": {"rate": hallu_rate, "n": len(hallu_pool),
                  "failed": [r["idx"] for r in hallu_pool if not r["hallu_ok"]]},
        "calibration": cal,
        "latency": latency,
        "false_positive": {"rate": fp, "n": len(trans)},
        "antihall": {"refuse_rate": ant_refuse, "n": len(ant)},
        "hard": hard,
        "by_kind": by_kind,
        "by_gold": by_gold,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "errors": [{"idx": r["idx"], "error": r["error"]}
                   for r in records if r["error"]],
    }


# ---------------- 报告渲染 ----------------

def _pct(x):
    return "—" if x is None else f"{x:.1%}"


def _num(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def _mark(cond):
    return "✅" if cond else "❌"


def render(samples, metrics, args):
    m = metrics
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "unknown"
    lines = []
    lines.append(f"# SFT 离线评估全表（MS3/MS5）")
    lines.append("")
    lines.append(f"- 端点: `{args.url}` · 模型: `{args.model}` · "
                 f"arbitration: `{args.arbitration}` · seed: `{args.seed}`")
    lines.append(f"- 数据集: `{args.dataset}` ({m['n']} 条) · "
                 f"git: `{git}` · 耗时: {m['wall_s']}s")
    lines.append("")
    lines.append("| # | 指标 | 实测 | 门槛 | 判定 |")
    lines.append("|---|---|---|---|---|")
    acc = m["accuracy"]
    mode_cn = "含仲裁" if args.arbitration == "on" else "关仲裁"
    lines.append(f"| 1 | 根因准确率（{mode_cn}，故障子集 n={acc['n_fault']}） | "
                 f"{_pct(acc['fault_subset'])} | ≥90% | {_mark(acc['fault_subset'] >= 0.90)} |")
    lines.append(f"| 1b | 根因准确率·关仲裁 / 含仲裁 | {_pct(acc['fault_subset_raw'])} / "
                 f"{_pct(acc['fault_subset_arb'])} | — | — |")
    lines.append(f"| 1c | 总体准确率（含无故障/数据异常 n={acc['n_overall']}） | "
                 f"{_pct(acc['overall'])} | — | — |")
    cp = m["confusion_pair"]
    lines.append(f"| 2 | 混淆对准确率（n={cp['n']}） | {_pct(cp['acc'])} | ≥90% | "
                 f"{_mark(cp['acc'] >= 0.90)} |")
    j = m["json"]
    lines.append(f"| 3 | JSON 合法率 | {_pct(j['rate'])} | ≥99% | {_mark(m['hard']['json'])} |")
    fa = m["faith"]
    lines.append(f"| 4 | 证据忠实率（n={fa['n']}，违规 {fa['bad']}） | {_pct(fa['rate'])} | ≥98% | "
                 f"{_mark(m['hard']['faith'])} |")
    ha = m["hallu"]
    lines.append(f"| 5 | 防幻觉通过率（n={ha['n']}，不含数据异常样本） | {_pct(ha['rate'])} | 100% | "
                 f"{_mark(m['hard']['hallu'])} |")
    c = m["calibration"]
    lines.append(f"| 6 | 置信度校准：判对−判错 | {_num(c['gap'])} | >0.15 | "
                 f"{_mark(c['gap'] is not None and c['gap'] > CAL_GAP_MIN)} |")
    lines.append(f"| 6b | 判错样本平均 conf / ≥0.8 比例 | {_num(c['mean_wrong'])} / "
                 f"{_pct(c['wrong_ge_0.8'])} | <0.8 且不虚高 | — |")
    lat = m["latency"]
    lines.append(f"| 7 | 时延 P50 / P95 / max (s) | {_num(lat['p50'],2)} / {_num(lat['p95'],2)} / "
                 f"{_num(lat['max'],2)} | P95 ≤5s | {_mark(lat['p95'] is not None and lat['p95'] <= 5.0)} |")
    fp = m["false_positive"]
    lines.append(f"| 8 | 无故障误报率（n={fp['n']}） | {_pct(fp['rate'])} | ≤5% | "
                 f"{_mark(fp['rate'] <= 0.05)} |")
    lines.append(f"| 9 | 回归集（历史线上 case） | 未接入 | 零退化 | ⚠️ |")
    lines.append("")
    ah = m["antihall"]
    lines.append(f"- 数据质量拒诊率：{_pct(ah['refuse_rate'])} (n={ah['n']})")
    lines.append(f"- 服务异常：{len(m['errors'])} 条" +
                 (f"（如 {m['errors'][0]}）" if m['errors'] else ""))
    lines.append("")
    lines.append(f"### 分层（{mode_cn}）")
    lines.append("")
    lines.append("| 类别 | n | 准确率 |")
    lines.append("|---|---|---|")
    for k, (cnt, a) in sorted(m["by_kind"].items()):
        lines.append(f"| {k} | {cnt} | {_pct(a)} |")
    lines.append("")
    lines.append("| 金标签（故障子集） | n | 准确率 | 模型输出分布 |")
    lines.append("|---|---|---|---|")
    for k, (cnt, a) in sorted(m["by_gold"].items()):
        dist = m["confusion"].get(k, {})
        lines.append(f"| {k} | {cnt} | {_pct(a)} | {dist} |")
    lines.append("")
    all_hard = all(m["hard"].values())
    lines.append(f"**硬门槛汇总**：{'✅ 全绿' if all_hard else '❌ 不过关（禁止进入 L3/L4）'} "
                 f"(JSON={m['hard']['json']} · FAITH={m['hard']['faith']} · HALLU={m['hard']['hallu']})")
    return "\n".join(lines), all_hard


def main():
    ap = argparse.ArgumentParser(description="SFT 离线评估全表（L2）")
    ap.add_argument("--url", default="http://localhost:3761/v1")
    ap.add_argument("--model", default="MiniCPM5-2B")
    ap.add_argument("--dataset", default="data/sft/eval.jsonl")
    ap.add_argument("--arbitration", choices=["on", "off"], default="off")
    ap.add_argument("--seed", type=int, default=0, help="复跑标识（记录用）")
    ap.add_argument("--limit", type=int, default=0, help="仅评估前 N 条（冒烟）")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default="logs/sft_eval_latest.jsonl",
                    help="逐条明细输出（jsonl）")
    ap.add_argument("--report", default="", help="追加 markdown 报告到该文件")
    args = ap.parse_args()

    samples, records, metrics = run(args)
    table, _ = render(samples, metrics, args)
    print("\n" + table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n明细: {out} · 汇总: {out.with_suffix('.summary.json')}")

    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "a", encoding="utf-8") as f:
            f.write("\n\n" + table + "\n")


if __name__ == "__main__":
    main()
