"""时序衍生特征工具（对齐题目"特征提取与状态识别"要求）。

- heating_rate_per_power: 单位电耗温升率（换热效率衰减前兆）
- pq_offset:              压力-流量特性曲线偏移度（管网阻抗变化前兆）
- seasonal_inlet_threshold: 进水温度阈值季节动态修正
"""

import numpy as np
import pandas as pd

# 与 simulator 管网模型一致的水力参数（实测辨识可更新）
P_STATIC_KPA = 120.0        # 静压 (kPa)
R_COIL = 2.2                # 线圈管路阻抗 (kPa/(L/s)^2)
RATED_FLOW_LPS = 8.0        # 额定流量 (L/s)


def heating_rate_per_power(
    furnace_temp: pd.Series,
    electric_power: pd.Series,
    window_s: int = 600,
) -> pd.Series:
    """单位电耗温升率 = 炉温变化率(℃/min) / 电功率(kW)。

    在功率 >500kW 时有效；其余返回 NaN。换热效率衰减时该比值下降。
    """
    dt_min = window_s / 60.0
    rate = (furnace_temp - furnace_temp.shift(window_s)) / dt_min  # ℃/min
    ratio = rate / electric_power.replace(0, np.nan)
    ratio[electric_power < 500.0] = np.nan
    return ratio


def pq_offset(
    pressure_kpa: pd.Series,
    flow_lps: pd.Series,
    p_static: float = P_STATIC_KPA,
    r_coil: float = R_COIL,
) -> pd.Series:
    """压力-流量特性曲线偏移度 = (P - P_model) / P_model，P_model = P_s + R·Q²。"""
    p_model = p_static + r_coil * flow_lps ** 2
    return (pressure_kpa - p_model) / p_model


def seasonal_inlet_threshold(
    inlet_temp: pd.Series,
    base_threshold: float = 35.0,
    baseline_c: float = 28.0,
    window_s: int = 86400,
) -> pd.Series:
    """进水温度预警阈值季节修正。

    以 24h 滚动均值估计环境/季节影响：均值高于基线时阈值等幅上浮
    （夏季冷却塔散热受限，进水温度基线整体抬升，固定阈值会误报）。
    """
    seasonal_mean = inlet_temp.rolling(window_s, min_periods=3600).mean()
    offset = (seasonal_mean - baseline_c).clip(lower=0.0)
    return base_threshold + offset
