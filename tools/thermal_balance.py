"""热平衡计算工具：Q = c · ṁ · ΔT。"""

import numpy as np
import pandas as pd

WATER_CP = 4.186  # kJ/(kg·K)


def expected_heat_kw(flow_lps: pd.Series, delta_t: pd.Series) -> pd.Series:
    """由冷却水侧计算带走的热功率 Q = c·ṁ·ΔT (kW)。"""
    return WATER_CP * flow_lps * delta_t


def expected_delta_t(q_heat_kw: pd.Series, flow_lps: pd.Series) -> pd.Series:
    """由热功率与流量反推进出水温差 ΔT = Q / (c·ṁ)。流量过低时温差封顶 60℃。"""
    flow = flow_lps.clip(lower=0.3)
    return (q_heat_kw / (WATER_CP * flow)).clip(upper=60.0)


def heat_balance_check(
    outlet_temp: pd.Series,
    inlet_temp: pd.Series,
    flow_lps: pd.Series,
    q_heat_kw: pd.Series,
    tol_ratio: float = 0.10,
) -> pd.Series:
    """热平衡校验：实际温差与理论温差的相对偏差是否超限。

    返回布尔 Series，True 表示违反热平衡（供 L3 防幻觉物理校验使用）。
    """
    actual = outlet_temp - inlet_temp
    expect = expected_delta_t(q_heat_kw, flow_lps)
    rel = (actual - expect).abs() / expect.clip(lower=1.0)
    return rel > tol_ratio
