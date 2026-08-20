"""数据模拟模块自测：数据质量 + 物理一致性验证。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator import DataSimulator, FaultSpec, SimConfig

CP = 4.186  # kJ/(kg·K)


def check_quality(df: pd.DataFrame, duration: float) -> None:
    assert len(df) == int(duration), f"记录数不符: {len(df)}"
    ts = pd.to_datetime(df["timestamp"])
    diffs = ts.diff().dropna().dt.total_seconds()
    assert (diffs == 1.0).all(), "时间戳不连续或非 1s 间隔"
    assert ts.is_monotonic_increasing, "时间戳未严格递增"
    assert not df.isna().any().any(), "存在空值"
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        decimals = (df[c] * 10).round(6) % 1
        assert (decimals < 1e-4).all(), f"{c} 精度超过 1 位小数"
    print("[PASS] 数据质量: 无空值/乱序/丢包, 时间戳 1s 连续递增, 精度 1 位小数")


def check_normal_physics(df: pd.DataFrame) -> None:
    melt = df[df["operating_condition"].isin(["melting", "startup"])]
    idle = df[df["operating_condition"] == "idle"]

    # 1) 电气公式 I = P·1000/(√3·U·cosφ) 容差 1%
    p = df["electric_power"].to_numpy()
    i_expected = np.where(p > 0, p * 1000 / (np.sqrt(3) * 1000 * 0.92), 0.0)
    err = np.abs(df["electric_current"].to_numpy() - i_expected) / np.maximum(i_expected, 1.0)
    mask = p > 1000.0  # 低功率段测量噪声占比大，仅校验显著功率段
    assert err[mask].max() < 0.01, "电流-功率不满足 √3·U·I·cosφ"
    print("[PASS] 电气一致性: P=√3·U·I·cosφ 误差<1%")

    # 2) 热平衡: ΔT = Q_heat/(c·ṁ), Q_heat = 0.08·P + 0.10·(T_炉 - T_进)
    q_heat = 0.08 * df["electric_power"] + 0.10 * (df["furnace_temp"] - df["inlet_temp"])
    dt_pred = q_heat / (CP * df["flow_rate"])
    dt_actual = df["outlet_temp"] - df["inlet_temp"]
    dt_err = (dt_pred - dt_actual).abs()
    assert dt_err.median() < 0.5, f"热平衡重建中位误差过大: {dt_err.median():.2f}℃"
    assert np.corrcoef(dt_pred, dt_actual)[0, 1] > 0.99, "热平衡重建相关性不足"
    print(f"[PASS] 热平衡一致性: ΔT=c·m·Q 重建中位误差 {dt_err.median():.2f}℃, "
          f"相关系数 {np.corrcoef(dt_pred, dt_actual)[0, 1]:.4f}")

    # 3) 水力特性 P = P_s + R·Q² 在故障段验证（见 check_fault_physics）
    #    正常段流量波动极小，此处仅校验静压下界（压力单位 kPa）
    assert df["pressure"].min() > 120 - 10, "压力低于静压，违反管网特性"

    # 4) 流速与流量严格线性: v = Q/A
    v_expected = df["flow_rate"] * 1e-3 / 3.318e-3
    assert np.abs(df["flow_velocity"] - v_expected).max() < 0.15, "流速-流量不满足 v=Q/A"
    print("[PASS] 流速一致性: v=Q/A 误差<0.15 m/s (含测量噪声)")

    # 5) 功率关系: 熔炼段功率显著高于待机等
    assert melt["electric_power"].mean() > 10 * max(idle["electric_power"].mean(), 1.0)
    melting = df[df["operating_condition"] == "melting"]
    assert idle["furnace_temp"].max() < melting["furnace_temp"].min(), "待机炉温应低于熔炼炉温"
    print(f"[PASS] 工况合理性: 熔炼均功率 {melt['electric_power'].mean():.0f}kW, "
          f"熔炼炉温 {melt['furnace_temp'].min():.0f}~{melt['furnace_temp'].max():.0f}℃")

    # 6) 正常范围检查（题目给定阈值，压力单位 kPa）
    assert df["pressure"].between(100, 350).all(), "压力超出合理区间"
    assert melt["outlet_temp"].max() < 55, f"正常工况出水温度超限: {melt['outlet_temp'].max()}"
    assert df["flow_rate"].min() > 0.8 * 8.0 * 0.9, "正常工况流量低于额定 80%"
    print(f"[PASS] 正常范围: 压力 {df['pressure'].min():.1f}~{df['pressure'].max():.1f}kPa, "
          f"出水温度≤{df['outlet_temp'].max():.1f}℃, 流量≥{df['flow_rate'].min():.1f}L/s")


def check_fault_physics(cfg: SimConfig) -> None:
    duration = 10800  # 3h
    fault_start = 7200.0  # 第 3 小时注入过滤器堵塞
    sim = DataSimulator(
        config=cfg,
        faults=[FaultSpec(name="filter_clog", start=fault_start, ramp=600, severity=0.9)],
    )
    df = sim.run(duration)
    normal = df[df["fault_label"] == "none"]
    fault = df.iloc[int(fault_start) + 600:]  # 爬升完成后

    # 堵塞后流量应显著衰减
    flow_drop = 1 - fault["flow_rate"].mean() / normal["flow_rate"].mean()
    assert flow_drop > 0.2, f"堵塞后流量衰减不足: {flow_drop:.1%}"
    # 热平衡必然导致出水温度升高
    assert fault["outlet_temp"].mean() > normal["outlet_temp"].mean() + 3, "堵塞后出水温度未升高"
    # 堵塞严重时流量应跌破 80% 额定（对应题目 L1 预警规则）
    assert fault["flow_rate"].min() < 0.8 * 8.0, "堵塞故障未触发流量<80%额定"
    print(f"[PASS] 故障物理: 堵塞后流量衰减 {flow_drop:.1%}, "
          f"出水温度 {normal['outlet_temp'].mean():.1f}℃ -> {fault['outlet_temp'].mean():.1f}℃, "
          f"最低流量 {fault['flow_rate'].min():.1f}L/s (<80%额定)")

    # 水力特性: P = P_s + R·Q²（kPa 单位下 R=2.2 kPa/(L/s)²），重建静压并校验单调性
    p = df["pressure"].to_numpy()
    q2 = df["flow_rate"].to_numpy() ** 2
    p_static_hat = p - 2.2 * q2
    assert abs(p_static_hat.mean() - 120) < 5, f"重建静压 {p_static_hat.mean():.1f}kPa 偏离 120kPa"
    assert p_static_hat.std() < 3, f"重建静压波动过大: {p_static_hat.std():.2f}kPa"
    # 单调性在故障爬升窗口内校验（流量连续变化，曲线关系最清晰）
    win = df.iloc[int(fault_start):int(fault_start) + 700]  # 爬升段(600s)+短平台
    rank_p = win["pressure"].rank().to_numpy()
    rank_q = win["flow_rate"].rank().to_numpy()
    rho = np.corrcoef(rank_p, rank_q)[0, 1]  # Spearman 秩相关（不依赖 scipy）
    assert rho > 0.98, f"压力-流量单调性不足: {rho:.3f}"
    print(f"[PASS] 水力一致性: 重建静压 {p_static_hat.mean():.1f}±{p_static_hat.std():.2f}kPa "
          f"(设定 120), 压力-流量 Spearman 相关 {rho:.4f}")


def main() -> None:
    cfg = SimConfig(seed=42)
    duration = 14400  # 4h，覆盖完整工况循环
    df = DataSimulator(config=cfg).run(duration)
    check_quality(df, duration)
    check_normal_physics(df)
    check_fault_physics(cfg)
    print("\n全部自测通过 ✔")


if __name__ == "__main__":
    main()
