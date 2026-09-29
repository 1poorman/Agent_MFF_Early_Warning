#!/usr/bin/env python3
"""Versioned eight-fault simulator dataset; never overwrites the frozen four-fault data.

Splits use disjoint episode seeds and equipment IDs. Labels come from fault
injection, with paired counterfactual runs used only to audit observability.
Counterfactual values and injection metadata never enter model prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator import DataSimulator, FaultSpec, SimConfig
from simulator.faults import FAULT_NAMES

VERSION = "hard_faults_v1"
NORMAL = "无故障（工况变化或正常波动）"
UNCERTAIN = "证据不足（建议补充测量）"
SENSORS = {
    "inlet_temp": ("进水温度", "℃", 0.3), "outlet_temp": ("出水温度", "℃", 0.3),
    "flow_rate": ("流量", "L/s", 0.12), "pressure": ("压力", "kPa", 1.5),
    "tank_level": ("水箱液位", "cm", 1.5), "conductivity": ("电导率", "µS/cm", 8.0),
    "cabinet_temp": ("电气柜温度", "℃", 0.6), "cabinet_humidity": ("湿度", "%RH", 1.5),
    "electric_power": ("电功率", "kW", 20.0), "furnace_temp": ("炉内温度", "℃", 2.0),
}
CHANNELS = {
    "filter_clog": ["flow_rate", "pressure", "outlet_temp"],
    "pump_cavitation": ["pressure", "flow_rate", "outlet_temp"],
    "pipe_leak": ["tank_level", "cabinet_humidity", "flow_rate", "pressure", "outlet_temp", "conductivity"],
    "scale_buildup": ["pressure", "flow_rate", "outlet_temp"],
    "heat_exchanger_degradation": ["inlet_temp", "outlet_temp"],
    "water_quality": ["conductivity"],
    "cabinet_fan_failure": ["cabinet_temp"],
    "makeup_valve_stuck_open": ["tank_level", "conductivity"],
}
SOPS = {
    "filter_clog": ["检查过滤器压差并清洗过滤器"],
    "pump_cavitation": ["检查吸入口与液位，必要时切换备用泵"],
    "pipe_leak": ["定位漏点并检查管路密封"],
    "scale_buildup": ["检查线圈水路并安排除垢"],
    "heat_exchanger_degradation": ["检查换热器冷侧流量及散热能力"],
    "water_quality": ["复测电导率并检查离子交换及补水水质"],
    "cabinet_fan_failure": ["检查电气柜风机与风道，按温升情况降载"],
    "makeup_valve_stuck_open": ["核对补水指令与实际流量，检查补水阀关闭状态"],
}
PAIRS = [
    ("filter_clog", "heat_exchanger_degradation"), ("pipe_leak", "water_quality"),
    ("scale_buildup", "cabinet_fan_failure"), ("pump_cavitation", "makeup_valve_stuck_open"),
    ("water_quality", "makeup_valve_stuck_open"), ("filter_clog", "pipe_leak"),
    ("scale_buildup", "heat_exchanger_degradation"), ("cabinet_fan_failure", "water_quality"),
]
SYSTEM = ("你是中频炉水冷与电气柜系统的诊断助手。根据观测识别所有有证据支持的故障，"
          "不要根据维修工单直接下结论。关键证据缺失或弱于噪声时应保留不确定性。只输出一个 JSON。")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_key(row):
    return json.dumps(row["messages"][:-1], ensure_ascii=False, sort_keys=True)


def stats(frame):
    result = {}
    for key, (name, unit, _) in SENSORS.items():
        x = frame[key].to_numpy(dtype=float)
        slope = np.polyfit(np.arange(len(x)), x, 1)
        residual = x - np.polyval(slope, np.arange(len(x)))
        result[key] = {"传感器": name, "单位": unit, "均值": round(float(x[-120:].mean()), 3),
                       "去趋势标准差": round(float(residual.std()), 3),
                       "每分钟变化": round(float(slope[0] * 60), 3)}
    return result


def support(actual, alternative, fault, hidden):
    scores = []
    for key in CHANNELS[fault]:
        if key in hidden:
            continue
        threshold = SENSORS[key][2]
        scores.append(abs(actual[key]["均值"] - alternative[key]["均值"]) / threshold)
        if key == "pressure" and fault == "pump_cavitation":
            scores.append(abs(actual[key]["去趋势标准差"] - alternative[key]["去趋势标准差"]) / 0.25)
    return max(scores, default=0.0)


def episode(split, index, faults, severity, ramp, view_offset):
    seed = {"train": 11000, "val": 21000, "eval": 31000}[split] + index
    rng = random.Random(seed)
    # Evaluation includes a shifted operating envelope; all classes use the same distribution.
    shifted = split == "eval" and rng.random() < 0.5
    cfg = SimConfig(seed=seed, ambient_temp=rng.uniform(31, 38) if shifted else rng.uniform(24, 31),
                    inlet_temp_base=rng.uniform(31, 35) if shifted else rng.uniform(25, 30),
                    rated_power=rng.uniform(1800, 2500) if shifted else rng.uniform(2500, 3200),
                    coil_resistance=rng.uniform(0.0019, 0.0025),
                    filter_resistance=rng.uniform(0.0005, 0.0007),
                    tank_area=rng.uniform(3.5, 4.5), conductivity_init=rng.uniform(400, 650))
    schedule = [("startup", 900, 1650.0), ("holding", 300, 500.0), ("melting", 600, 1650.0)]
    specs = [FaultSpec(name, 300, ramp, severity) for name in faults]
    frame = DataSimulator(cfg, schedule=schedule, faults=specs).run(1800)
    # Each cause must change at least one visible sensor relative to removing that cause.
    alternatives = {name: DataSimulator(cfg, schedule=schedule, faults=[s for s in specs if s.name != name]).run(1800)
                    for name in faults}
    reference = stats(frame.iloc[120:300])
    rows = []
    for view, end in enumerate((1200, 1800)):
        current = stats(frame.iloc[end-600:end])
        mode = (view_offset + view) % 4
        hidden = set()
        if mode == 3:
            # Missing sensors also occur on healthy episodes, so missingness is not a fault label.
            candidate = rng.choice(faults) if faults else rng.choice(list(FAULT_NAMES))
            hidden = set(CHANNELS[candidate])
        strengths = {name: support(current, stats(alternatives[name].iloc[end-600:end]), name, hidden)
                     for name in faults}
        observable = [name for name in faults if strengths[name] >= 1.0]
        unresolved = len(observable) != len(faults) or (not faults and bool(hidden))
        # For partially observed compound faults, retain supported causes and flag remaining uncertainty.
        names = sorted(FAULT_NAMES[name] for name in observable)
        gold = " + ".join(names) if names else UNCERTAIN if unresolved else NORMAL
        kind = "insufficient" if unresolved else "composite" if len(faults)>1 else "normal" if not faults else "boundary" if severity<0.3 else "clear"
        wrong_name = rng.choice([name for name in FAULT_NAMES.values() if name not in names])
        maintenance = f"历史工单曾记录{wrong_name}，尚未确认本次是否复发。" if mode == 2 else "近期工单未提供可确认的本次根因。"
        if mode == 2 and not unresolved:
            kind = "misleading"
        visible = {key: value for key, value in current.items() if key not in hidden}
        observed_reference = {key: value for key, value in reference.items() if key not in hidden}
        prompt = ("【可选故障域】" + "、".join(FAULT_NAMES.values()) + "\n"
                  "【设备标定】" + json.dumps({"环境温度_℃": round(cfg.ambient_temp, 2),
                      "线圈水阻_kPa每流量平方": round(cfg.coil_resistance*1000, 3),
                      "静压_kPa": cfg.static_pressure*1000,
                      "正常柜体温升系数_℃每kW": cfg.cabinet_temp_coef}, ensure_ascii=False) + "\n"
                  "【本炉次早期参考观测】" + json.dumps(observed_reference, ensure_ascii=False) + "\n"
                  "【当前窗口统计】" + json.dumps(visible, ensure_ascii=False) + "\n"
                  "【当前工况】" + str(frame.iloc[end-1]["operating_condition"]) + "\n"
                  "【不可用通道】" + "、".join(SENSORS[key][0] for key in sorted(hidden)) + "\n"
                  "【维修工单】" + maintenance + "\n"
                  f"输出 root_cause、root_causes（所有可辨识故障名称数组）、needs_more_data、confidence、evidence、sop。"
                  f"无故障填“{NORMAL}”；证据不足且无法确认任何故障填“{UNCERTAIN}”。"
                  "复合故障不得编造主次顺序，root_cause 用已确认故障名称排序后以 ' + ' 连接。")
        evidence_keys = sorted({key for name in observable for key in CHANNELS[name] if key not in hidden})
        if not evidence_keys:
            evidence_keys = list(visible)[:3]
        evidence = [f"{SENSORS[key][0]}均值 {current[key]['均值']}{SENSORS[key][1]}，"
                    f"去趋势标准差 {current[key]['去趋势标准差']}{SENSORS[key][1]}，"
                    f"每分钟变化 {current[key]['每分钟变化']}{SENSORS[key][1]}/min"
                    for key in evidence_keys]
        if hidden:
            evidence.append("缺失通道：" + "、".join(SENSORS[key][0] for key in sorted(hidden)))
        actions = list(dict.fromkeys(action for name in observable for action in SOPS[name]))
        if unresolved:
            actions.append("补充缺失通道或延长观测后再确认其余根因")
        if not actions:
            actions = ["持续监测并核对工况变化"]
        answer = {"root_cause": gold, "root_causes": names, "needs_more_data": unresolved,
                  "confidence": 0.45 if unresolved else 0.7 if kind in ("boundary", "composite") else 0.85,
                  "evidence": evidence, "sop": actions}
        rows.append({"messages": [{"role":"system","content":SYSTEM}, {"role":"user","content":prompt},
                                  {"role":"assistant","content":json.dumps(answer,ensure_ascii=False)}],
                     "meta": {"version":VERSION, "split":split, "kind":kind, "gold":gold,
                              "gold_causes":names,"needs_more_data":unresolved,"seed":seed,
                              "episode_id":f"{split}-{index:05d}","equipment_id":f"equipment-{seed}",
                              "shifted_envelope":shifted,"window_end":end,"window_seconds":600,
                              "injected_faults": [asdict(s) for s in specs], "hidden_channels":sorted(hidden),
                              "counterfactual_support":strengths,"simulation_config":asdict(cfg),"schedule":schedule}})
    return rows


def audit(datasets):
    keys, seeds, episodes = {}, {}, {}
    counts = {}
    for split, rows in datasets.items():
        keys[split] = {input_key(r) for r in rows}
        seeds[split] = {r["meta"]["seed"] for r in rows}
        episodes[split] = {r["meta"]["episode_id"] for r in rows}
        if len(keys[split]) != len(rows):
            raise ValueError(f"duplicate inputs in {split}")
        for r in rows:
            answer=json.loads(r["messages"][-1]["content"])
            if answer["root_cause"] != r["meta"]["gold"] or answer["root_causes"] != r["meta"]["gold_causes"]:
                raise ValueError("gold mismatch")
            if answer["needs_more_data"] != r["meta"]["needs_more_data"]:
                raise ValueError("uncertainty mismatch")
            supported=sorted(FAULT_NAMES[name] for name,value in r["meta"]["counterfactual_support"].items() if value>=1.0)
            if supported != answer["root_causes"]:
                raise ValueError("answer includes an unobservable cause or omits a supported cause")
            prompt=r["messages"][1]["content"]
            if any(key in prompt for key in ("injected_faults", "counterfactual_support", "episode_id", "simulation_config")):
                raise ValueError("hidden simulation metadata leaked into prompt")
            # Every numeric evidence item quotes the rendered observation exactly.
            import re
            for ev in answer["evidence"]:
                for number in re.findall(r"-?\d+(?:\.\d+)?", ev):
                    if number not in prompt:
                        raise ValueError(f"untraceable evidence {number}")
        counts[split] = {"samples":len(rows),"episodes":len(episodes[split]),
                         "by_kind":dict(Counter(r["meta"]["kind"] for r in rows)),
                         "by_cause":dict(Counter(c for r in rows for c in r["meta"]["gold_causes"])),
                         "shifted_samples":sum(r["meta"]["shifted_envelope"] for r in rows)}
    for a,b in (("train","val"),("train","eval"),("val","eval")):
        if keys[a]&keys[b] or seeds[a]&seeds[b] or episodes[a]&episodes[b]:
            raise ValueError(f"split leakage: {a}/{b}")
    return counts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/sft_hard_v1")
    ap.add_argument("--train-per-fault",type=int,default=48)
    ap.add_argument("--val-per-fault",type=int,default=8)
    ap.add_argument("--eval-per-fault",type=int,default=12)
    ap.add_argument("--seed",type=int,default=42,help="output shuffling seed; simulation seeds are split-specific")
    args=ap.parse_args()
    if min(args.train_per_fault,args.val_per_fault,args.eval_per_fault)<1:
        ap.error("per-fault counts must be positive")
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    datasets={}
    for split,n in (("train",args.train_per_fault),("val",args.val_per_fault),("eval",args.eval_per_fault)):
        rows=[];index=0
        specs=[([fault],j) for fault in FAULT_NAMES for j in range(n)]
        specs += [(list(pair),j) for pair in PAIRS for j in range(max(1,min(n,4),n//8))]
        specs += [([],j) for j in range(max(4,n//2))]
        for faults,j in specs:
            severity=(0.08,0.2,0.5,0.85)[j%4] if len(faults)<2 else (0.5,0.85)[j%2]
            rows.extend(episode(split,index,faults,severity,(180,600,900)[j%3],j))
            index+=1
            if index%16==0 or index==len(specs):
                print(f"{split}: {index}/{len(specs)} episodes, {len(rows)} samples",flush=True)
        random.Random(args.seed).shuffle(rows)
        datasets[split]=rows
    counts=audit(datasets)
    # Keep all old data frozen and verify seed/input separation from both previous experiments.
    historical=[]
    for folder,names in (("data/sft",("train","val","eval")),("data/rl",("prompts_gold","eval"))):
        for name in names:
            path=ROOT/folder/f"{name}.jsonl"
            if path.exists(): historical += [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    old_keys={input_key(r) for r in historical};old_seeds={r["meta"]["seed"] for r in historical}
    for rows in datasets.values():
        if old_keys & {input_key(r) for r in rows} or old_seeds & {r["meta"]["seed"] for r in rows}:
            raise ValueError("overlap with frozen experiments")
    out.mkdir(parents=True,exist_ok=True)
    hashes={}
    for split,rows in datasets.items():
        p=out/f"{split}.jsonl"
        p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
        hashes[split]=sha(p)
    ontology={"version":VERSION,"fault_names":FAULT_NAMES,"sops":SOPS,"channels":CHANNELS,
              "normal_label":NORMAL,"uncertain_label":UNCERTAIN,
              "schema_note":"root_causes is a set-valued target; evaluate exact set match and macro F1, not a fixed-priority primary cause"}
    (out/'ontology.json').write_text(json.dumps(ontology,ensure_ascii=False,indent=2)+'\n')
    manifest={"version":VERSION,"counts":counts,"sha256":hashes,"params":vars(args),
              "leakage":{"inputs":0,"seeds":0,"episodes":0,"vs_historical":0},
              "confidence_note":"synthetic instructional confidence, not empirically calibrated probability",
              "label_note":"injection labels gated by visible paired-counterfactual signal; simulator-specific, not real industrial annotations",
              "source_sha256":{p:sha(ROOT/p) for p in ['tools/build_hard_sft_data.py','simulator/faults.py',
                                                        'simulator/physics.py','simulator/generator.py','simulator/config.py']}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for name in manifest["source_sha256"]:
        destination=out/'source'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    print(json.dumps(counts,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
