"""MS1 零成本基线试验：MiniCPM5-2B 原模型（未训练）根因诊断基线。

目的：回答"MiniCPM5-2B 原模型的窄域鉴别上限够不够"（决策门见 docs/SFT_MILESTONES.md MS1）。

方法（与线上链路严格同构）：
  1. 复用 AgentService + WarningAnalysisAgent.analyze() 完整组包
     （L1 规则 / stats 窗口统计 / extra_candidates / 工况表 / 维修工单）；
  2. 仅替换 reasoner.llm 为本地 MiniCPM5-2B vLLM 端点（:3761），
     并用录制包装器捕获每次 LLM 原始输出；
  3. 关仲裁准确率 = LLM 首次输出解析出的 root_cause 与仿真金标签的命中率
     （不含 root_cause.py 的图谱/统计仲裁，那是 SFT 要替代的对象）。

用法：
  conda run -n mff_agent python tools/ms1_baseline.py [--n-per-fault 40] [--url http://localhost:3761/v1]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# 仓库根入 sys.path（python tools/ms1_baseline.py 直跑时 sys.path[0] 是 tools/）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- 环境隔离：评估栈全跑 CPU（GPU0 留给 vLLM 服务，GPU1~3 为他人占用）----
# 必须在 torch 导入前设置；detection/fast_track.py:47 默认 cuda:1 会撞满卡
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pandas as pd

import torch
_torch_load_orig = torch.load
def _cpu_torch_load(*a, **k):
    k.setdefault("map_location", "cpu")
    return _torch_load_orig(*a, **k)
torch.load = _cpu_torch_load

# ---------------- 评估集定义 ----------------
# 单一故障场景（决策门口径）；composite 仅观察不入准确率统计
SCENARIOS = [
    # (csv, 中文金标签, 目标窗口数)
    ("data/simulated/cavitation_4h.csv", "水泵气蚀", 40),
    ("data/simulated/leak_4h.csv", "管道泄漏", 40),
    ("data/simulated/scale_12h.csv", "线圈结垢", 40),
    ("data/simulated/scale_severe_12h.csv", "线圈结垢", 20),
    ("data/simulated/fault_demo_6h.csv", "过滤器堵塞", 40),  # filter_clog 单故障段
]
COMPOSITE_SCENARIOS = [  # 复合故障（观察项：LLM 能否给出主/次根因）
    ("data/simulated/mixed_faults_8h.csv", 10),
]
NORMAL_CSV = "data/simulated/normal_24h.csv"
NORMAL_WINDOWS = 40

LABEL_MAP = {  # 仿真 fault_label -> 知识图谱故障名
    "pump_cavitation": "水泵气蚀",
    "pipe_leak": "管道泄漏",
    "scale_buildup": "线圈结垢",
    "filter_clog": "过滤器堵塞",
    "none": "无故障",
    # 复合故障（观察项用，不计入决策门准确率）
    "filter_clog+pipe_leak": "过滤器堵塞+管道泄漏",
    "filter_clog+pump_cavitation": "过滤器堵塞+水泵气蚀",
    "filter_clog+pump_cavitation+pipe_leak": "过滤器堵塞+水泵气蚀+管道泄漏",
}
COMPOSITE_LABELS = {"filter_clog+pipe_leak", "filter_clog+pump_cavitation",
                    "filter_clog+pump_cavitation+pipe_leak"}

WINDOW_S = 1200    # 窗口 20min（覆盖 stats 需要的 240s + 工况上下文）
STRIDE_S = 600     # 滑动步长 10min


def sample_windows(csv: str, n_target: int, single_only: bool = True):
    """取尾段 120s 为单一故障标签的滑窗，均匀抽取 n_target 个。"""
    df = pd.read_csv(csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    n = len(df)
    wins = []
    for start in range(0, n - WINDOW_S + 1, STRIDE_S):
        win = df.iloc[start:start + WINDOW_S]
        tail_labels = win.iloc[-120:]["fault_label"].unique()
        if len(tail_labels) != 1:
            continue
        lab = tail_labels[0]
        if lab == "none":
            continue
        if single_only and lab in COMPOSITE_LABELS:
            continue
        if not single_only and lab not in COMPOSITE_LABELS:
            continue
        wins.append((win, LABEL_MAP[lab]))
    if not wins:
        return []
    step = max(len(wins) // n_target, 1)
    picked = wins[::step][:n_target]
    return picked


def sample_normal_windows(csv: str, n_target: int):
    df = pd.read_csv(csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    n = len(df)
    # 覆盖不同时段（该 CSV 24h，取 40 个窗口步长 ~35min）
    stride = max((n - WINDOW_S) // max(n_target, 1), 1)
    wins = []
    for start in range(0, n - WINDOW_S + 1, stride):
        win = df.iloc[start:start + WINDOW_S]
        if len(win) < WINDOW_S:
            break
        wins.append(win)
        if len(wins) >= n_target:
            break
    return wins


class RecordingLLM:
    """LLMClient 兼容包装：录制每次原始输出，用于关仲裁指标。"""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []  # [(prompt_head, raw, latency_s)]

    def chat(self, prompt, system="", max_tokens=1500, temperature=0.2):
        t0 = time.time()
        raw = self.inner.chat(prompt, system=system, max_tokens=max_tokens,
                              temperature=temperature)
        self.calls.append((prompt[:80], raw, time.time() - t0))
        return raw


def parse_root_cause(raw: str):
    """与 root_cause._parse 同款逻辑（MS1 独立实现，避免 import 私有方法漂移）。"""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d.get("root_cause"), True
    except json.JSONDecodeError:
        return None, False


def build_stream_report(df: pd.DataFrame):
    """镜像 service._diagnose_async 的组包（实时流路径，demo 主链路）。

    与批量路径（WarningAnalysisAgent.analyze）的关键差异：
      - 不注入 l1_alerts/l2_forecast（实时流只传 stats + 工况上下文，
        L1/L2 仅事后挂在 context；批量路径全量注入 L1 会撑爆上下文）；
      - 注入 diag_context（工况切换提示，与线上一致）。
    返回 (features, condition, sensors, stats, extra_cands, diag_ctx)。
    """
    from server.agents import WarningAnalysisAgent
    last = df.iloc[-1]
    col_map = {"出水温度": "outlet_temp", "进水温度": "inlet_temp", "压力": "pressure",
               "流量": "flow_rate", "水箱液位": "tank_level", "湿度": "cabinet_humidity",
               "电导率": "conductivity"}
    features = {}
    for s, c in col_map.items():
        if c in df.columns and not pd.isna(last[c]):
            features[s] = float(last[c])
    stats = {}
    WarningAnalysisAgent.compute_diag_stats(df, stats, features)
    extra_cands = WarningAnalysisAgent._stat_precheck(features, stats)
    cond = last.get("operating_condition")
    condition = "unknown" if cond is None or (
        isinstance(cond, float) and pd.isna(cond)) else str(cond)
    cond_change = False
    if "operating_condition" in df.columns:
        cs = df["operating_condition"].dropna()
        if len(cs) >= 60:
            recent = cs.iloc[-60:]
            cond_change = len(recent.unique()) > 1 or \
                (len(recent) > 1 and (recent != recent.iloc[0]).sum() > 0)
    diag_ctx = {"condition_transition": cond_change,
                "hint": ("当前正处于工况切换/升降温阶段，L2 异常分升高可能为工况变化导致的正常现象；"
                         "请优先依据 L1 规则与统计特征判断，若 L1 未命中且特征正常，请判为正常工况，不要误报故障。"
                         if cond_change else "当前工况稳定。")}
    sensors = ["出水温度", "压力", "流量", "湿度"]
    return features, condition, sensors, stats, extra_cands, diag_ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3761/v1")
    ap.add_argument("--model", default="MiniCPM5-2B")
    ap.add_argument("--out", default="data/sft/ms1_baseline_results.jsonl")
    args = ap.parse_args()

    # ---- 服务栈（读线上同款配置，模型权重已在 models/）----
    from server.service import AgentService
    from reasoning.llm_client import LLMClient

    svc = AgentService(use_llm=True)
    local_llm = LLMClient(config={
        "url": args.url, "key": "empty", "big_model_name": args.model,
        "enable_thinking": False, "timeout": 180.0})
    rec = RecordingLLM(local_llm)
    svc.pipeline.reasoner.llm = rec  # 仅换端点，组包/仲裁逻辑不动

    results = []

    def run_one(win, gold, scenario, kind):
        """按实时流路径（_diagnose_async 同构）组包并诊断，显式记录异常。"""
        df = win.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        from perception.ingest import ALL_COLUMNS
        for c in ALL_COLUMNS:
            if c not in df.columns:
                df[c] = float("nan")
        df = df[ALL_COLUMNS]
        feats, cond, sensors, stats, extra, dctx = build_stream_report(df)
        n_calls_before = len(rec.calls)
        t0 = time.time()
        err = None
        try:
            diag = svc.diagnose(feats, cond, sensors, stats=stats,
                                extra_candidates=extra, diag_context=dctx)
            rc_arb = diag.root_cause
        except Exception as e:
            diag = None
            rc_arb = None
            err = f"{type(e).__name__}: {e}"
        wall = time.time() - t0
        rc_raw, json_ok, llm_lat = None, None, None
        if len(rec.calls) > n_calls_before:
            _, raw, llm_lat = rec.calls[n_calls_before]
            rc_raw, json_ok = parse_root_cause(raw)
        results.append({
            "scenario": scenario, "kind": kind, "gold": gold,
            "error": err,
            "llm_called": len(rec.calls) > n_calls_before,
            "rc_raw": rc_raw,            # 关仲裁根因（首次 LLM 输出）
            "json_ok": json_ok,
            "rc_arbitrated": rc_arb,     # 含仲裁（参考）
            "retries": diag.retries if diag else None,
            "llm_latency_s": round(llm_lat, 2) if llm_lat else None,
            "wall_s": round(wall, 2),
        })

    # ---- 单一故障窗口（决策门口径）----
    total = sum(n for _, _, n in SCENARIOS)
    done = 0
    for csv, gold, n_target in SCENARIOS:
        wins = sample_windows(csv, n_target)
        print(f"[{csv}] 金标签={gold} 采样 {len(wins)}/{n_target} 窗口", flush=True)
        for win, lab in wins:
            run_one(win, lab, Path(csv).stem, "fault")
            done += 1
            if done % 10 == 0:
                print(f"  进度 {done}/{total}", flush=True)

    # ---- 复合故障（观察项）----
    for csv, n_target in COMPOSITE_SCENARIOS:
        wins = sample_windows(csv, n_target, single_only=False)
        print(f"[{csv}] 复合故障观察窗口 {len(wins)}", flush=True)
        for win, lab in wins:
            run_one(win, lab, Path(csv).stem, "composite")

    # ---- 正常窗口（上游误触发率：L1 规则 + L2 异常分，不调 LLM——
    #      实时流路径下未触发即不诊断，LLM 层无"无故障"选项，强测无意义）----
    det = svc.pipeline.detector
    for win in sample_normal_windows(NORMAL_CSV, NORMAL_WINDOWS):
        df = win.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        from perception.ingest import ALL_COLUMNS
        for c in ALL_COLUMNS:
            if c not in df.columns:
                df[c] = float("nan")
        df = df[ALL_COLUMNS]
        l1 = svc.rule_engine.evaluate(df)
        l2_trig = False
        if det is not None and len(df) >= det.window:
            try:
                l2_trig = float(det.score(df).iloc[-1]) > det.threshold
            except Exception:
                pass
        results.append({
            "scenario": "normal_24h", "kind": "normal", "gold": "无故障",
            "error": None, "llm_called": False,
            "triggered_upstream": bool(len(l1)) or l2_trig,
            "l1_count": len(l1), "l2_triggered": l2_trig,
        })

    # ---- 汇总 ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    faults = [r for r in results if r["kind"] == "fault"]
    llm_faults = [r for r in faults if r["llm_called"]]
    errors = [r for r in faults if r.get("error")]
    acc_raw = sum(r["rc_raw"] == r["gold"] for r in llm_faults) / max(len(llm_faults), 1)
    acc_arb = sum(r["rc_arbitrated"] == r["gold"] for r in llm_faults) / max(len(llm_faults), 1)
    json_ok_rate = sum(bool(r["json_ok"]) for r in llm_faults) / max(len(llm_faults), 1)
    lats = [r["llm_latency_s"] for r in llm_faults if r["llm_latency_s"]]
    retries = [r["retries"] for r in llm_faults if r["retries"] is not None]
    normals = [r for r in results if r["kind"] == "normal"]
    false_trigger = sum(r.get("triggered_upstream") for r in normals) / max(len(normals), 1)

    print("\n" + "=" * 60)
    print(f"LLM 调用成功: {len(llm_faults)}/{len(faults)} （异常 {len(errors)}，详见明细 error 字段）")
    print(f"关仲裁根因准确率: {acc_raw:.1%}  （决策门: ≥60% 投入 / 40~60% 降级 / <40% 终止）")
    print(f"含仲裁准确率(参考): {acc_arb:.1%}")
    print(f"JSON 合法率: {json_ok_rate:.1%}")
    print(f"防幻觉重试率: {sum(1 for x in retries if x > 0)}/{len(retries)}")
    print(f"LLM 平均时延: {sum(lats)/max(len(lats),1):.1f}s (n={len(lats)})")
    print(f"正常窗口上游误触发率: {false_trigger:.1%} (n={len(normals)})")
    print("分故障类别（关仲裁）:")
    for gold in ["水泵气蚀", "管道泄漏", "线圈结垢", "过滤器堵塞"]:
        sub = [r for r in llm_faults if r["gold"] == gold]
        if sub:
            a = sum(r["rc_raw"] == r["gold"] for r in sub) / len(sub)
            print(f"  {gold}: {a:.1%} (n={len(sub)})")
    print("混淆矩阵（关仲裁 raw → gold）:")
    from collections import Counter
    for gold in ["水泵气蚀", "管道泄漏", "线圈结垢", "过滤器堵塞"]:
        sub = Counter(r["rc_raw"] for r in llm_faults if r["gold"] == gold)
        print(f"  {gold} ← {dict(sub)}")
    print(f"复合故障观察 (n={sum(1 for r in results if r['kind']=='composite')}):")
    for r in results:
        if r["kind"] == "composite":
            print(f"  gold={r['gold']} raw={r['rc_raw']}")
    print(f"\n明细已存 {out_path}")


if __name__ == "__main__":
    main()
