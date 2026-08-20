"""MS2 验收测试：L1 规则引擎 + 工具层（含露点/凝露预测）。

验收标准（design/MILESTONES.md）：
- 单条判定 <10ms
- 故障召回率 100%（filter_clog / pipe_leak 必触发）
- 正常误报 <5%
- 衍生特征正确性（与物理重建误差 <5%）
- 季节修正生效
- 露点计算误差 <0.5℃，凝露预测提前量
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection import RuleEngine
from perception import DataIngestor, QualityController
from tools.dew_point import CondensationPredictor, dew_point
from tools.ts_features import pq_offset, seasonal_inlet_threshold

ROOT = Path(__file__).resolve().parent.parent
NORMAL = ROOT / "data" / "simulated" / "normal_24h.csv"
FAULT = ROOT / "data" / "simulated" / "fault_demo_6h.csv"


def load(path):
    df = DataIngestor().load_csv(path)
    clean, _ = QualityController().process(df)
    return clean


def test_dew_point_accuracy():
    """Magnus-Tetens 对照标准气象露点数据（误差 <0.5℃）。"""
    # (温度, 湿度, 标准露点) 来自 Magnus 标准表
    cases = [(30.0, 50.0, 18.4), (25.0, 60.0, 16.7), (20.0, 80.0, 16.4), (35.0, 40.0, 19.4)]
    for t, rh, td_ref in cases:
        td = dew_point(t, rh)
        assert abs(td - td_ref) < 0.5, f"露点误差过大: T={t} RH={rh} -> {td:.2f} (ref {td_ref})"
    print("[PASS] 露点计算: 4 组标准气象点误差 <0.5℃")


def test_condensation_prediction():
    """构造湿度爬升场景，验证凝露预测提前量。"""
    n = 1800  # 30min
    ts = pd.date_range("2026-08-20", periods=n, freq="1s")
    # 柜温缓降 + 湿度爬升 -> 裕度单调收缩
    temp = 35.0 - np.linspace(0, 3.0, n)
    hum = 55.0 + np.linspace(0, 30.0, n)
    df = pd.DataFrame({"timestamp": ts, "cabinet_temp": temp, "cabinet_humidity": hum})

    pred = CondensationPredictor(window_s=600, horizon_s=600)
    first_eta = None
    for end in range(600, n, 60):
        win = df.iloc[end - 600:end]
        risk = pred.assess(win["timestamp"], win["cabinet_temp"], win["cabinet_humidity"])
        if risk.predicted_risk and first_eta is None:
            first_eta = risk.eta_s
            break
    assert first_eta is not None, "未预测到凝露风险"
    assert 0 < first_eta <= 600, f"预测提前量异常: {first_eta}s"
    print(f"[PASS] 凝露预测: 提前 {first_eta/60:.1f}min 预警 (裕度趋势外推)")


def test_latency(normal: pd.DataFrame):
    engine = RuleEngine()
    rows = normal.head(100000 if len(normal) >= 100000 else len(normal))
    t0 = time.perf_counter()
    n_alerts = 0
    for row in rows.itertuples(index=False):
        n_alerts += len(engine.evaluate_row(row._asdict()))
    elapsed = time.perf_counter() - t0
    per_row_us = elapsed / len(rows) * 1e6
    assert per_row_us < 10_000, f"单条判定 {per_row_us:.0f}µs 超 10ms"
    print(f"[PASS] 响应时延: 单条判定 {per_row_us:.1f}µs (<10ms), {len(rows)} 条产生 {n_alerts} 条预警")


def test_recall_on_faults(fault: pd.DataFrame):
    """filter_clog 爬升完成后流量<80% 必触发；pipe_leak 段压力低必触发。"""
    engine = RuleEngine()
    alerts = engine.evaluate(fault)
    clog_seg = fault[(fault.index > 12600) & (fault.fault_label.str.contains("filter_clog"))]
    leak_seg = fault[fault.fault_label.str.contains("pipe_leak")]

    a_clog = alerts[(alerts.timestamp >= clog_seg.timestamp.min()) & (alerts.timestamp <= clog_seg.timestamp.max())]
    a_leak = alerts[(alerts.timestamp >= leak_seg.timestamp.min()) & (alerts.timestamp <= leak_seg.timestamp.max())]

    assert (a_clog.rule_id == "FLOW_LOW").any(), "堵塞段未触发 FLOW_LOW"
    assert (a_leak.rule_id == "PRESSURE_LOW").any(), "泄漏段未触发 PRESSURE_LOW"
    # 组合规则：泄漏 + 湿度 → 疑似泄漏
    assert (a_leak.rule_id == "COMBO_LEAK_SUSPECT").any(), "泄漏段未触发组合泄漏规则"
    print(f"[PASS] 故障召回: 堵塞段 FLOW_LOW {int((a_clog.rule_id=='FLOW_LOW').sum())} 次, "
          f"泄漏段 PRESSURE_LOW {int((a_leak.rule_id=='PRESSURE_LOW').sum())} 次 + COMBO_LEAK_SUSPECT 触发")


def test_false_alarm_rate(normal: pd.DataFrame):
    engine = RuleEngine()
    alerts = engine.evaluate(normal)
    far = len(alerts) / len(normal)
    assert far < 0.05, f"正常误报率 {far:.2%} >= 5%"
    print(f"[PASS] 正常误报: {len(alerts)} 条 / {len(normal)} 条 = {far:.4%} (<5%)")
    if len(alerts):
        print(alerts.rule_id.value_counts().to_dict())


def test_derived_features(normal: pd.DataFrame):
    """P-Q 偏移度与物理重建一致性：正常数据偏移应 <5%（含测量噪声）。"""
    offset = pq_offset(normal["pressure"], normal["flow_rate"])
    assert offset.abs().median() < 0.05, f"P-Q 偏移中位 {offset.abs().median():.2%} 超 5%"
    print(f"[PASS] 衍生特征: P-Q 偏移度中位 {offset.abs().median():.2%} (<5%, 与水力模型一致)")


def test_seasonal_correction():
    """夏季基线上浮时进水温度阈值应动态上调，避免误报。"""
    ts = pd.date_range("2026-08-20", periods=86400, freq="1s")
    summer = pd.Series(33.0 + np.sin(np.arange(86400) / 86400 * 2 * np.pi) * 1.5)  # 夏季基线 33℃
    th = seasonal_inlet_threshold(summer, base_threshold=35.0, baseline_c=28.0)
    assert th.iloc[-1] > 35.0, "夏季阈值未上浮"
    # 夏季进水 36℃：固定 35℃ 阈值会误报，动态阈值（上浮后）不应误报
    engine = RuleEngine()
    row = {"timestamp": ts[-1], "inlet_temp": 36.0, "outlet_temp": 40.0, "pressure": 260.0,
           "flow_rate": 8.0, "conductivity": 550.0, "cabinet_temp": 40.0, "cabinet_humidity": 50.0}
    fixed = engine.evaluate_row(row)  # 固定阈值
    dyn = engine.evaluate_row(row, inlet_threshold=float(th.iloc[-1]))  # 动态阈值
    assert any(a.rule_id == "INLET_TEMP_HIGH" for a in fixed), "固定阈值应触发（对照）"
    assert not any(a.rule_id == "INLET_TEMP_HIGH" for a in dyn), "动态阈值不应误报"
    print(f"[PASS] 季节修正: 夏季阈值 35.0 -> {th.iloc[-1]:.1f}℃, 36.0℃ 进水不再误报")


def main():
    normal = load(NORMAL)
    fault = load(FAULT)
    test_dew_point_accuracy()
    test_condensation_prediction()
    test_latency(normal)
    test_recall_on_faults(fault)
    test_false_alarm_rate(normal)
    test_derived_features(normal)
    test_seasonal_correction()
    print("\nMS2 全部验收通过 ✔")


if __name__ == "__main__":
    main()
