"""MS2 SFT 数据生成：仿真场景 → 线上同构 report → 退化版提示 + 金标签推理链。

设计依据：docs/LLM_SFT后训练实施指南.md 第 2 节 + MS1 基线结论
（气蚀/泄漏全军覆没于"默认堵塞"偏置 → 数据配比向混淆对倾斜）。

样本六类（总数 ~2600）：
  clear      4 类故障清晰签名 ~1200（≥300/类）
  boundary   混淆对边界 ~600（湿度 62~76%RH / 压力std 2~4.5kPa / PQ 7~14%）
  transition 工况切换正常波动 ~300（输出"无故障"）
  composite  复合故障 ~200（主因+次因排序）
  misleading 工单误导 ~200（维修记录指向错误根因，金标签仍按实时 stats）
  antihall   防幻觉负样本 ~100（特征物理违规 → "数据质量异常"）

输入 = 退化版提示：镜像 root_cause._build_prompt 结构，删除"判定规则"节与 3 条"注意"
（要内化的知识），保留客观数据（stats/工况/工单/图谱候选）。
输出 = 推理链（引用真实 stats 数值）+ 严格 JSON（与 DiagnosisResult 对齐，
置信度按证据强度分层校准）。

用法：
  conda run -n mff_agent python tools/gen_sft_data.py [--out-dir data/sft]
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ms1_baseline import build_stream_report  # 复用线上同构组包（CPU 隔离补丁随导入生效）

from simulator import DataSimulator, FaultSpec, SimConfig
from context.maintenance import default_maintenance_log
from context.operating import default_operating_schedule
from reasoning.knowledge_graph import KnowledgeGraph

# ---------------- 配比与采样参数 ----------------
FAULTS = ["pump_cavitation", "pipe_leak", "scale_buildup", "filter_clog"]
FAULT_CN = {"pump_cavitation": "水泵气蚀", "pipe_leak": "管道泄漏",
            "scale_buildup": "线圈结垢", "filter_clog": "过滤器堵塞"}
SEVERITIES = [0.12, 0.2, 0.3, 0.4, 0.55, 0.75, 1.0]  # 细网格：低段命中边界带（气蚀 std∝sev）
RAMPS = [300, 900, 1800]
SEEDS_TRAIN = [101, 102, 103, 104, 105, 106]
SEEDS_EVAL = [901, 902]          # 评估集独立 seed 段（审计断言与训练集不相交）
NORMAL_SEEDS_TRAIN = [201, 202, 203, 204, 205, 206, 207, 208]
NORMAL_SEEDS_EVAL = [951, 952]
SIM_DUR = 5400                    # 90min：故障 600s 起注，窗口取故障充分发展段
FAULT_START = 600
WINDOW = 1200
WINDOW_ENDS = [2700, 3300, 3900, 4500, 5100]   # 窗口末尾时刻（尾 120s 即 stats 窗口）
NORMAL_DUR = 21600               # 快循环 8 个周期，切换窗口挖掘充分
# 工况切换挖掘用快循环（真实炉次周期因炉而异，属合理增强；
# 默认 8400s 循环的切换点恰好落在 600 对齐窗口边界上，采不到）
FAST_SCHEDULE = [("startup", 900, 1650.0), ("melting", 600, 1650.0),
                 ("holding", 600, 1550.0), ("tapping", 300, 900.0), ("idle", 300, 35.0)]

QUOTA = {  # 训练集目标（另有 val=10% 从 train 池再切）
    "clear": 1200, "boundary": 600, "transition": 300,
    "composite": 200, "misleading": 200, "antihall": 100,
}
EVAL_QUOTA = {  # 评估集（独立 seed）
    "clear": 80, "boundary": 60, "transition": 20,
    "composite": 20, "misleading": 10, "antihall": 10,
}

# ---------------- 边界（混淆区）判定 ----------------

def is_boundary(gold_cn: str, stats: dict) -> bool:
    """样本 stats 是否落在该故障的混淆阈值带内（决定 boundary 类别）。"""
    hum = float(stats.get("湿度均值_pctRH", 50))
    hd = float(stats.get("湿度上升量_pctRH", 0) or 0)
    pstd = float(stats.get("压力波动幅度_std_kPa", 0))
    pq = float(stats.get("PQ特性偏移_pct", 0) or 0)
    flow = float(stats.get("流量均值_Lps", 8.0))
    if gold_cn == "管道泄漏":
        # 60~70 为"湿度上升但未达铁证"的争议带（>70 即泄漏铁证，属 clear）
        return 60 <= hum <= 70 or 2.5 <= hd <= 6.0
    if gold_cn == "水泵气蚀":
        return 2.0 <= pstd <= 4.5
    if gold_cn == "线圈结垢":
        return 7.0 <= pq <= 14.0
    if gold_cn == "过滤器堵塞":
        return 6.0 <= flow <= 7.2 and pq < 10.0
    return False


def dominant_fault(stats: dict) -> str:
    """复合故障主因判定（与 root_cause 统计强先验同序：泄漏>气蚀>结垢>堵塞）。"""
    hum = float(stats.get("湿度均值_pctRH", 50))
    hd = float(stats.get("湿度上升量_pctRH", 0) or 0)
    pstd = float(stats.get("压力波动幅度_std_kPa", 0))
    pq = float(stats.get("PQ特性偏移_pct", 0) or 0)
    flow = float(stats.get("流量均值_Lps", 8.0))
    if hum > 70 or (hd > 4.0 and flow < 7.8):
        return "管道泄漏"
    if pstd > 3.0:
        return "水泵气蚀"
    if pq >= 10.0:
        return "线圈结垢"
    if flow < 6.4:
        return "过滤器堵塞"
    return ""  # 签名太弱，丢弃该窗口

# ---------------- 提示构造（退化版 _build_prompt） ----------------
SYSTEM_PROMPT = ("你是中频炉水冷系统的故障诊断专家。输出先给推理过程（引用统计特征数值做鉴别），"
                 "最后输出唯一一个JSON对象，不要输出多余内容。")

_OUTPUT_SCHEMA = """请严格按以下 JSON 输出（不要输出多余内容）：
{{
  "root_cause": "根因名称（来自候选根因或图谱故障域；若为工况切换正常波动输出'无故障（工况切换正常波动）'；若数据存在物理违规输出'数据质量异常（建议人工核查）'）",
  "confidence": 0.0到1.0的置信度,
  "evidence": ["证据1", "证据2"],
  "sop": ["处置步骤1", "处置步骤2"]
}}"""


def build_candidates(kg: KnowledgeGraph, sensors, extra_cands):
    """镜像 root_cause.diagnose 的候选集组装（图谱召回 Top3 + 统计预鉴别）。"""
    ranked = kg.faults_for_sensors(sensors)
    cands = [kg.nodes[fid].name for fid, _ in ranked][:3]
    for extra in extra_cands:
        if extra in kg.fault_names() and extra not in cands:
            cands.append(extra)
    return cands


def build_degraded_prompt(kg: KnowledgeGraph, report: dict, candidates: list) -> str:
    """镜像 _build_prompt 结构；删除"判定规则"节与 3 条"注意"（SFT 要内化的知识），
    保留全部客观数据段；schema 扩展支持无故障/数据异常输出。"""
    kg_facts = "；".join(
        f"{c}的关联处置: {','.join(kg.actions_for_fault(c)) or '无'}" for c in candidates)
    feats = report["features"]
    stats = report["stats"]
    stats_text = "\n".join(f"- {k}: {v}" for k, v in stats.items()) or "无"
    diag_ctx = report.get("diag_context") or {}
    cond_transition = ("【工况状态】当前处于工况切换/升降温阶段，L2 异常分升高可能是工况变化导致的"
                       "正常现象，请依据统计特征判断是否真实故障。" if diag_ctx.get("condition_transition") else "")
    cond_stable = "【工况状态】当前工况稳定，异常更可能是真实故障。" \
        if diag_ctx.get("condition_transition") is False else ""
    return f"""你是中频炉水冷系统的故障诊断专家。综合统计鉴别特征、近期维修工单与工况运行表，做多跳因果推理，定位最可能的物理根因。

【统计鉴别特征（窗口计算，判定依据）】
{stats_text}

【实时异常特征】{json.dumps(feats, ensure_ascii=False)}

【当前工况】{report.get('condition', 'unknown')}
{cond_transition}{cond_stable}
【工况运行表】
{report.get('operating_schedule', '无')}

【近期维修工单】
{report.get('maintenance_log', '无')}

【知识图谱候选根因】{', '.join(candidates) if candidates else '（无先验候选）'}
【图谱事实】{kg_facts or '无'}

{_OUTPUT_SCHEMA}"""

# ---------------- 推理链模板（数值全部来自 stats，保证证据忠实） ----------------

def cot_leak(s, conf):
    return (f"尾段湿度均值 {s.get('湿度均值_pctRH', '?')}%RH，湿度上升量 {s.get('湿度上升量_pctRH', 0)}%RH——"
            f"水汽逸散特征明确（湿度持续上升是泄漏区别于堵塞/结垢的关键：后两者不改变湿度）；"
            f"伴随流量均值 {s.get('流量均值_Lps', '?')}L/s 与压力下降。"
            f"压力去趋势 std {s.get('压力波动幅度_std_kPa', '?')}kPa 未达 3kPa 气蚀震荡阈值，排除气蚀；"
            f"PQ 特性偏移 {s.get('PQ特性偏移_pct', 0)}% 未显著抬升，排除线圈结垢。判定管道泄漏，置信度 {conf}。")

def cot_cavitation(s, conf):
    return (f"压力去趋势波动 std {s.get('压力波动幅度_std_kPa', '?')}kPa > 3kPa，存在压力震荡——NPSH 不足的典型签名；"
            f"湿度均值 {s.get('湿度均值_pctRH', '?')}%RH、上升量 {s.get('湿度上升量_pctRH', 0)}%RH，无水汽逸散证据，排除泄漏；"
            f"PQ 特性偏移 {s.get('PQ特性偏移_pct', 0)}% < +10%，线圈侧阻抗未抬升，排除结垢。判定水泵气蚀，置信度 {conf}。")

def cot_scale(s, conf):
    return (f"PQ 特性偏移 +{s.get('PQ特性偏移_pct', 0)}% ≥ +10%——实测压力显著高于水力模型预测（P_model=120+2.2·Q²），"
            f"线圈管路阻抗抬升的确定性证据；进出水温差 {s.get('进出水温差_℃', '?')}℃ 同功率下拉大（线圈热阻增大）；"
            f"流量均值 {s.get('流量均值_Lps', '?')}L/s 伴随下降。湿度均值 {s.get('湿度均值_pctRH', '?')}%RH 无上升趋势，排除泄漏；"
            f"压力 std {s.get('压力波动幅度_std_kPa', '?')}kPa 无震荡，排除气蚀。判定线圈结垢，置信度 {conf}。")

def cot_clog(s, conf):
    return (f"流量均值 {s.get('流量均值_Lps', '?')}L/s 低于 6.4L/s 下限——过滤器段阻抗升高使流量衰减；"
            f"PQ 特性偏移 {s.get('PQ特性偏移_pct', 0)}% < +10%，线圈侧压力仍贴近水力模型，排除线圈结垢；"
            f"湿度均值 {s.get('湿度均值_pctRH', '?')}%RH 无变化（堵塞不改变湿度），排除泄漏；"
            f"压力去趋势 std {s.get('压力波动幅度_std_kPa', '?')}kPa 无震荡，排除气蚀。判定过滤器堵塞，置信度 {conf}。")

def cot_transition(s, conf, cond):
    return (f"当前处于工况切换/升降温阶段（{cond}），压力与温度的波动符合升降温物理规律；"
            f"湿度均值 {s.get('湿度均值_pctRH', '?')}%RH 稳定、上升量 {s.get('湿度上升量_pctRH', 0)}%RH 无泄漏迹象；"
            f"压力去趋势 std {s.get('压力波动幅度_std_kPa', '?')}kPa 正常（<3kPa）；"
            f"流量均值 {s.get('流量均值_Lps', '?')}L/s 与 PQ 偏移 {s.get('PQ特性偏移_pct', 0)}% 正常。"
            f"统计特征均无故障签名命中，判定为工况切换正常波动，未发生故障，置信度 {conf}。")

def cot_composite(s, conf, primary_cn, secondary_cn):
    base = {"管道泄漏": cot_leak, "水泵气蚀": cot_cavitation,
            "线圈结垢": cot_scale, "过滤器堵塞": cot_clog}[primary_cn](s, conf)
    sec_sign = {"管道泄漏": "湿度上升", "水泵气蚀": "压力震荡",
                "线圈结垢": "PQ 特性偏移", "过滤器堵塞": "流量下降"}[secondary_cn]
    return (f"{base[:-len('。')]}。"
            f"同时检测到{secondary_cn}的伴随特征（{sec_sign}），作为次因一并排查处置。")

def cot_antihall(violations, conf):
    return ("输入特征存在物理违规：" + "；".join(violations) +
            "。各物理量存在合理界限，数据疑似传感器故障或录入错误，"
            "不宜强行诊断根因，建议人工核查数据链路，置信度 " + f"{conf}。")

COT = {"管道泄漏": cot_leak, "水泵气蚀": cot_cavitation,
       "线圈结垢": cot_scale, "过滤器堵塞": cot_clog}

# 置信度分层（2.4 节校准原则）
#
# 注：v2 曾尝试把置信度改为"随签名余量连续变化"，**实测为净负面**并被撤销：
#   准确率 99.4% → 97.2%（composite 95% → 80%），校准 gap −0.045 → −0.088。
# 原因是"判对−判错 > 0.15"这一判据与高准确率记忆型 SFT 结构性冲突
# （判对集合含按设计就低置信的类：antihall/boundary；判错样本反而高置信）。
# 详见 design/SFT_EVAL_REPORT.md 第六节。故回到按类别给常数的简单方案。
def calibrate(kind: str, stats: dict, gold_cn: str = None) -> float:
    if kind == "boundary":
        return 0.78
    if kind == "composite":
        return 0.85
    if kind == "transition":
        return 0.90
    if kind == "antihall":
        return 0.50
    return 0.88          # clear / misleading


def build_sample(kg, window_df, gold_cn, kind, seed, misleading_log=None,
                 corrupt=None, secondary_cn=None):
    """组装一条 SFT 样本。返回 (sample_dict, stats, ok)。"""
    feats, cond, sensors, stats, extra, dctx = build_stream_report(window_df)
    report = {"features": feats, "condition": cond, "stats": stats,
              "extra_candidates": extra, "diag_context": dctx,
              "operating_schedule": SCHED_TEXT,
              "maintenance_log": (misleading_log or MAINT_TEXT)}
    if corrupt:  # 防幻觉负样本：注入物理违规（features + 对应 stats 同步，保持输入自洽）
        report["features"][corrupt["feat"]] = corrupt["bad"]
        stat_key, mode = corrupt.get("stat"), corrupt.get("stat_mode")
        if stat_key and mode == "delta_out_minus_in":
            # 出水温度无独立 stat → 用温差 stat 联动，保持 stats/features 相互自洽
            in_t = float(report["features"].get("进水温度", 0.0) or 0.0)
            report["stats"][stat_key] = round(corrupt["bad"] - in_t, 1)
        elif stat_key:
            report["stats"][stat_key] = corrupt["bad"]
    candidates = build_candidates(kg, sensors, extra)

    if kind == "antihall":
        # 违规说明必须与**真正注入到输入里的值**一致
        # （MS5 v1 的 20 条错标签正是"说明的字段/值在输入中并不存在"）
        ft = report["features"]
        viol = [corrupt["reason"]]
        if corrupt["kind"] != "second_law" and ft.get("出水温度", 99) < ft.get("进水温度", 0):
            viol.append(f"出水温度{ft['出水温度']}℃ 低于进水温度{ft['进水温度']}℃，"
                        f"违背热力学第二定律")
        conf = calibrate(kind, stats)
        gold = "数据质量异常（建议人工核查）"
        cot = cot_antihall(viol, conf)
        sop = ["复核传感器读数与数据链路", "排查变送器/采集模块", "数据恢复后再行诊断"]
    elif kind == "transition":
        conf = calibrate(kind, stats)
        gold = "无故障（工况切换正常波动）"
        cot = cot_transition(stats, conf, cond)
        sop = ["持续监测", "关注工况稳定后的特征复归"]
    else:
        conf = calibrate(kind, stats, gold_cn)
        gold = gold_cn
        if kind == "composite" and secondary_cn:
            cot = cot_composite(stats, conf, gold_cn, secondary_cn)
        else:
            cot = COT[gold_cn](stats, conf)
        sop = kg.actions_for_fault(gold_cn) or ["按运维手册排查"]

    answer = cot + "\n" + json.dumps(
        {"root_cause": gold, "confidence": conf,
         "evidence": extract_evidence(kind, gold_cn, stats, secondary_cn),
         "sop": sop}, ensure_ascii=False)
    sample = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_degraded_prompt(kg, report, candidates)},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"kind": kind, "gold": gold, "seed": seed,
                 "secondary": secondary_cn, "stats": stats},
    }
    return sample, stats, True


def extract_evidence(kind, gold_cn, stats, secondary_cn):
    """证据条目：引用 stats 真实键值（供审计数值忠实性）。"""
    ev = []
    if kind == "antihall":
        return ["特征值超出物理界限", "疑似传感器/数据链路故障"]
    if kind == "transition":
        return [f"湿度均值 {stats.get('湿度均值_pctRH')}%RH 无上升趋势",
                f"压力波动 std {stats.get('压力波动幅度_std_kPa')}kPa 正常",
                "工况切换阶段特征波动符合物理规律"]
    if gold_cn in ("管道泄漏", "水泵气蚀"):
        ev.append(f"压力波动幅度_std {stats.get('压力波动幅度_std_kPa')}kPa")
        ev.append(f"湿度均值 {stats.get('湿度均值_pctRH')}%RH，上升量 {stats.get('湿度上升量_pctRH', 0)}%RH")
    if gold_cn in ("线圈结垢", "过滤器堵塞"):
        ev.append(f"PQ特性偏移 {stats.get('PQ特性偏移_pct', 0)}%")
        ev.append(f"流量均值 {stats.get('流量均值_Lps')}L/s")
    if secondary_cn:
        ev.append(f"伴随{secondary_cn}特征，建议一并排查")
    return ev

# ---------------- 场景池生成 ----------------

def run_sim(faults, seed, duration=SIM_DUR):
    sim = DataSimulator(config=SimConfig(seed=seed), faults=faults)
    df = sim.run(duration)
    df["timestamp"] = sim_df_timestamps(duration)
    return df


def sim_df_timestamps(duration):
    import pandas as pd
    return pd.date_range("2026-08-20 00:00:00", periods=int(duration), freq="1s")


def fault_windows(df, label_cn):
    """按窗口末尾切窗；返回 (window_df, 尾段金标签) 列表。"""
    out = []
    for end in WINDOW_ENDS:
        win = df.iloc[end - WINDOW:end]
        if len(win) < WINDOW:
            continue
        tail_labels = win.iloc[-120:]["fault_label"].unique()
        if len(tail_labels) != 1:
            continue
        lab = tail_labels[0]
        cn = "+".join(FAULT_CN.get(x, x) for x in lab.split("+")) if lab != "none" else "none"
        out.append((win, cn))
    return out

# ---------------- 主流程 ----------------

MAINT_TEXT = None
SCHED_TEXT = None
PHYS_LIMITS = {"湿度": (0.0, 100.0), "压力": (0.0, 1000.0), "流量": (0.0, 50.0),
               "出水温度": (0.0, 100.0), "进水温度": (0.0, 60.0), "水箱液位": (0.0, 500.0)}


def gen_fault_pool(kg, seeds, n_max=None):
    """生成 4 类故障窗口池：返回 {kind: [sample...]}（clear/boundary 自动分流）。"""
    pool = {"clear": [], "boundary": []}
    for fault in FAULTS:
        got = 0
        for sev in SEVERITIES:
            for ramp in RAMPS:
                for seed in seeds:
                    if n_max and got >= n_max:
                        break
                    fs = FaultSpec(name=fault, start=FAULT_START, ramp=ramp, severity=sev)
                    df = run_sim([fs], seed)
                    for win, cn in fault_windows(df, FAULT_CN[fault]):
                        if cn != FAULT_CN[fault]:
                            continue
                        # 先判类别再组包：boundary 才能拿到 0.78 的校准置信度
                        _, _, _, stats, _, _ = build_stream_report(win)
                        kind = "boundary" if is_boundary(cn, stats) else "clear"
                        sample, _, _ = build_sample(kg, win, cn, kind, seed)
                        pool[kind].append(sample)
                        got += 1
        print(f"  {FAULT_CN[fault]}: 累计 {got} 窗口", flush=True)
    return pool


def gen_normal_windows(seeds):
    """无故障仿真（快工况循环）：返回 [(window_df, seed)]。

    筛选条件：窗口尾段 120s 内发生工况切换（diag_ctx 的 60s 判定随之命中），
    步长 60 保证切换点不与网格对齐漏采。返回原始窗口，供 transition 与
    antihall（需在结构上注入违规）两路共用。
    """
    out = []
    for seed in seeds:
        sim = DataSimulator(config=SimConfig(seed=seed), schedule=FAST_SCHEDULE, faults=[])
        df = sim.run(NORMAL_DUR)
        df["timestamp"] = sim_df_timestamps(NORMAL_DUR)
        for start in range(600, len(df) - WINDOW + 1, 60):
            win = df.iloc[start:start + WINDOW]
            tail = win.iloc[-120:]
            cond = tail["operating_condition"]
            if (cond != cond.iloc[0]).sum() > 0:  # 尾 120s 内有切换
                out.append((win, seed))
    return out


def gen_transition(kg, normal_windows):
    """由正常窗口构建 transition 样本（输出"无故障（工况切换正常波动）"）。"""
    return [build_sample(kg, win, None, "transition", seed)[0]
            for win, seed in normal_windows]


def gen_composite_pool(kg, seeds):
    """复合故障：主因按统计签名主导性判定，次因注入 CoT。"""
    pairs = [("filter_clog", "pipe_leak"), ("filter_clog", "pump_cavitation"),
             ("pipe_leak", "pump_cavitation")]
    out = []
    for f1, f2 in pairs:
        for s1, s2 in [(0.7, 0.5), (0.5, 0.7), (0.8, 0.8)]:
            for seed in seeds:
                fs = [FaultSpec(name=f1, start=FAULT_START, ramp=900, severity=s1),
                      FaultSpec(name=f2, start=FAULT_START + 900, ramp=900, severity=s2)]
                df = run_sim(fs, seed)
                for win, cn in fault_windows(df, None):
                    if "+" not in cn or cn == "none":
                        continue
                    # 先取 stats 判主因
                    _, _, _, stats, _, _ = build_stream_report(win)
                    primary = dominant_fault(stats)
                    if not primary or primary not in cn:
                        continue
                    secondary = FAULT_CN[f2] if primary == FAULT_CN[f1] else FAULT_CN[f1]
                    sample, _, _ = build_sample(kg, win, primary, "composite", seed,
                                                 secondary_cn=secondary)
                    out.append(sample)
    return out


MISLEADING_LOGS = {
    "管道泄漏": "【维修记录（近60天）】\n- 2026-09-01 WO-0001 更换过滤器滤芯，冲洗管路（堵塞已处理）\n- 2026-08-15 WO-0002 水泵叶轮动平衡校验",
    "水泵气蚀": "【维修记录（近60天）】\n- 2026-09-03 WO-0003 更换管道法兰垫片，处理渗漏点\n- 2026-08-20 WO-0004 水箱补水阀检修",
    "线圈结垢": "【维修记录（近60天）】\n- 2026-09-05 WO-0005 更换过滤器滤芯（堵塞已处理）\n- 2026-08-28 WO-0006 水泵密封件更换",
    "过滤器堵塞": "【维修记录（近60天）】\n- 2026-09-02 WO-0007 管道渗漏点补焊（泄漏已处理）\n- 2026-08-18 WO-0008 电气柜除湿维护",
}


def gen_misleading(kg, clear_samples, quota):
    """复用 clear 样本，替换维修工单为指向错误根因的记录（金标签不变，教模型'工单仅辅助'）。"""
    out = []
    random.Random(7).shuffle(clear_samples)
    for s in clear_samples:
        if len(out) >= quota:
            break
        gold = s["meta"]["gold"]
        prompt = s["messages"][1]["content"]
        # 将默认工单段替换为误导工单
        import re
        new_prompt = re.sub(r"【近期维修工单】\n.*?(?=\n\n【知识图谱候选根因】)",
                            f"【近期维修工单】\n{MISLEADING_LOGS[gold]}\n", prompt, flags=re.S)
        if new_prompt == prompt:
            continue
        s2 = {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": new_prompt},
                           {"role": "assistant", "content": s["messages"][2]["content"]}],
              "meta": {**s["meta"], "kind": "misleading"}}
        out.append(s2)
    return out


# 防幻觉负样本的注入方案：字段 / 注入值 / 违规类型 / 对应 stats 键 / 违规说明。
# 违规说明必须能由**输入直接验证**——MS5 v1 的 20 条错标签正是"说明与注入不一致"所致。
CORRUPTIONS = [
    {"feat": "出水温度", "bad": 5.0, "kind": "second_law",
     "stat": "进出水温差_℃", "stat_mode": "delta_out_minus_in",
     "reason": "出水温度=5.0℃ 低于进水温度，违背热力学第二定律"
               "（冷却水吸热，出水温度必然高于进水温度）"},
    {"feat": "湿度", "bad": 130.0, "kind": "bounds",
     "stat": "湿度均值_pctRH", "stat_mode": "same",
     "reason": "湿度=130.0%RH 超出物理界限[0.0,100.0]"},
    {"feat": "压力", "bad": -50.0, "kind": "bounds",
     "stat": "压力均值_kPa", "stat_mode": "same",
     "reason": "压力=-50.0kPa 超出物理界限[0.0,1000.0]"},
    {"feat": "流量", "bad": 80.0, "kind": "bounds",
     "stat": "流量均值_Lps", "stat_mode": "same",
     "reason": "流量=80.0L/s 超出物理界限[0.0,50.0]"},
]


def antihall_grounded(sample) -> bool:
    """自检：CoT 声称的违规值必须真出现在 user prompt 的【实时异常特征】里，且确实越界。

    不满足即丢弃该样本。MS5 v1 的错标签（输入全正常却声称"出水温度=5.0 越界"）
    就是缺这道自检造成的。
    """
    user = sample["messages"][1]["content"]
    m = re.search(r"【实时异常特征】\s*(\{.*?\})", user, re.S)
    if not m:
        return False
    try:
        feats = json.loads(m.group(1))
    except json.JSONDecodeError:
        return False
    nums = [float(v) for v in feats.values() if isinstance(v, (int, float))]
    claimed = [float(x) for x in re.findall(r"=(-?\d+\.?\d*)", sample["messages"][2]["content"])]
    if not claimed or not any(any(abs(c - n) < 1e-6 for n in nums) for c in claimed):
        return False                                   # 声称的违规值不在输入中
    for k, (lo, hi) in PHYS_LIMITS.items():
        if k in feats and not (lo <= float(feats[k]) <= hi):
            return True                                # 确有越界
    return float(feats.get("出水温度", 99)) < float(feats.get("进水温度", 0))  # 或热二律违规


def gen_antihall(kg, normal_windows, quota):
    """复用正常窗口，注入物理违规特征 → 训练"数据异常拒诊"。

    走 build_sample(corrupt=...) 的结构化注入（直接改 report dict），不再对已渲染的
    prompt 做正则手术——v1 正是后者漏改了 features JSON 的**首个键**（出水温度），
    产出"输入无越界却标数据质量异常"的毒样本。出口再过 antihall_grounded 自检。
    """
    out, skipped = [], 0
    for i, (win, seed) in enumerate(normal_windows):
        if len(out) >= quota:
            break
        corrupt = CORRUPTIONS[i % len(CORRUPTIONS)]
        sample, _, _ = build_sample(kg, win, None, "antihall", seed, corrupt=corrupt)
        if not antihall_grounded(sample):
            skipped += 1
            continue
        out.append(sample)
    if skipped:
        print(f"  [antihall] 自检丢弃 {skipped} 条（声称违规未被输入支撑）", flush=True)
    return out


def take(pool, quota, rng):
    rng.shuffle(pool)
    return pool[:quota]


def main():
    global MAINT_TEXT, SCHED_TEXT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/sft")
    args = ap.parse_args()

    kg = KnowledgeGraph()
    MAINT_TEXT = default_maintenance_log().to_prompt_text(days=60)
    SCHED_TEXT = default_operating_schedule().to_prompt_text()
    rng = random.Random(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 训练池 ----
    print("[1/4] 生成 4 类故障窗口池（清晰/边界自动分流）...", flush=True)
    fault_pool = gen_fault_pool(kg, SEEDS_TRAIN)
    print(f"  池存: clear={len(fault_pool['clear'])} boundary={len(fault_pool['boundary'])}", flush=True)

    print("[2/4] 工况切换正常窗口...", flush=True)
    nwin = gen_normal_windows(NORMAL_SEEDS_TRAIN)
    transitions = gen_transition(kg, nwin)
    print(f"  transition={len(transitions)}", flush=True)

    print("[3/4] 复合故障...", flush=True)
    composites = gen_composite_pool(kg, SEEDS_TRAIN)
    print(f"  composite={len(composites)}", flush=True)

    clear = take(fault_pool["clear"], QUOTA["clear"], rng)
    boundary = take(fault_pool["boundary"], QUOTA["boundary"], rng)
    transition = take(transitions, QUOTA["transition"], rng)
    composite = take(composites, QUOTA["composite"], rng)
    misleading = gen_misleading(kg, fault_pool["clear"], QUOTA["misleading"])
    antihall = gen_antihall(kg, nwin, QUOTA["antihall"])
    train_all = clear + boundary + transition + composite + misleading + antihall
    rng.shuffle(train_all)
    n_val = max(len(train_all) // 10, 1)
    val, train = train_all[:n_val], train_all[n_val:]

    # ---- 评估池（独立 seed）----
    print("[4/4] 评估集（独立 seed）...", flush=True)
    epool = gen_fault_pool(kg, SEEDS_EVAL)
    enwin = gen_normal_windows(NORMAL_SEEDS_EVAL)
    etrans = gen_transition(kg, enwin)
    ecomp = gen_composite_pool(kg, SEEDS_EVAL)
    eclear = take(epool["clear"], EVAL_QUOTA["clear"], rng)
    ebound = take(epool["boundary"], EVAL_QUOTA["boundary"], rng)
    etr = take(etrans, EVAL_QUOTA["transition"], rng)
    ecp = take(ecomp, EVAL_QUOTA["composite"], rng)
    emis = gen_misleading(kg, epool["clear"], EVAL_QUOTA["misleading"])
    eah = gen_antihall(kg, enwin, EVAL_QUOTA["antihall"])
    evalset = eclear + ebound + etr + ecp + emis + eah
    rng.shuffle(evalset)

    for name, rows in [("train", train), ("val", val), ("eval", evalset)]:
        p = out_dir / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for s in rows:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        from collections import Counter
        kinds = Counter(s["meta"]["kind"] for s in rows)
        print(f"  {p}: {len(rows)} 条 {dict(kinds)}", flush=True)

    # seed 防泄漏断言
    train_seeds = {s["meta"]["seed"] for s in train + val}
    eval_seeds = {s["meta"]["seed"] for s in evalset}
    assert not (train_seeds & eval_seeds), "train/eval seed 相交！"
    print(f"seed 防泄漏断言通过（train={sorted(train_seeds)} eval={sorted(eval_seeds)}）")


if __name__ == "__main__":
    main()
