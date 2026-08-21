"""凝露风险预测验证：露点计算 + 缓变趋势预警。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.service import AgentService
from tools.dew_point import CondensationPredictor, dew_point


def test_dew_point_basic():
    # 已知: 温度 25℃, 湿度 60% -> 露点约 16.7℃
    td = dew_point(25.0, 60.0)
    assert abs(td - 16.7) < 0.5, f"露点误差: {td}"
    # 温度 35℃, 湿度 80% -> 露点约 31.1℃
    td2 = dew_point(35.0, 80.0)
    assert abs(td2 - 31.1) < 0.5, f"露点误差: {td2}"
    print(f"[PASS] 露点计算: 25℃/60%->{td:.1f}℃, 35℃/80%->{td2:.1f}℃")


def test_condensation_trend():
    """柜温缓降 + 湿度缓升 -> 裕度收缩 -> 趋势预警。"""
    n = 1800
    ts = pd.date_range("2026-08-21", periods=n, freq="1s")
    temp = 30.0 - np.linspace(0, 6.0, n)      # 柜温 30->24℃
    hum = 60.0 + np.linspace(0, 35.0, n)      # 湿度 60->95%
    df = pd.DataFrame({"timestamp": ts, "cabinet_temp": temp, "cabinet_humidity": hum})

    pred = CondensationPredictor(window_s=600, horizon_s=600)
    first_eta = None
    for end in range(600, n, 60):
        win = df.iloc[end - 600:end]
        risk = pred.assess(win["timestamp"], win["cabinet_temp"], win["cabinet_humidity"])
        if risk.predicted_risk:
            first_eta = risk.eta_s
            break
    assert first_eta is not None, "未预测到凝露"
    print(f"[PASS] 凝露趋势: 提前 {first_eta/60:.0f}min 预警凝露")


def test_stream_dew_output():
    """stream_step 输出 dew 字段（实时流集成验证）。"""
    s = AgentService(use_llm=False)
    s.reset_stream()
    # 模拟柜温低+湿度高（凝露风险场景）
    rows = []
    for i in range(700):
        row = {"inlet_temp": 28.0, "outlet_temp": 40.0, "pressure": 250.0, "flow_rate": 8.0,
               "flow_velocity": 2.4, "tank_level": 200.0, "conductivity": 550.0,
               "cabinet_temp": 24.0 + np.sin(i / 50) * 0.2,
               "cabinet_humidity": 75.0 + (i / 700) * 20.0,
               "furnace_temp": 1600.0, "electric_power": 2800.0, "electric_current": 1757.0,
               "operating_condition": "melting", "fault_label": "none",
               "timestamp": f"2026-08-21 13:00:{i%60:02d}"}
        rows.append(row)
    # 预加载 600 条 + 实时 100 条
    for r in rows[:600]:
        s.preload_row(r)
    out = None
    for r in rows[600:]:
        out = s.stream_step(r)
    assert out is not None and out.get("dew") is not None, "dew 字段缺失"
    d = out["dew"]
    print(f"[PASS] stream dew: 露点 {d['dew_point']}℃ 裕度 {d['margin']}℃ "
          f"at_risk={d['at_risk']} eta={d['eta_s']}")
    assert d["margin"] < 3.0, "凝露裕度应接近危险"


def main():
    test_dew_point_basic()
    test_condensation_trend()
    test_stream_dew_output()
    print("\n凝露风险全部验证通过 ✔")


if __name__ == "__main__":
    main()
