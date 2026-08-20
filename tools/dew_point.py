"""露点计算与电气柜凝露预测工具（Magnus-Tetens 公式）。

γ(T, RH) = ln(RH/100) + b·T/(c+T)
T_dew   = c·γ / (b-γ)

凝露判据：柜体表面温度 < 露点温度 → 水汽凝结 → 绝缘下降/短路风险。
裕度 margin = T_surface - T_dew，裕度 <3℃ 即预警；
同时基于近期裕度线性趋势外推，预测 horizon 内是否跌破阈值。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

MAGNUS_B = 17.62
MAGNUS_C = 243.12  # ℃

MARGIN_WARN_C = 3.0      # 凝露裕度预警阈值 (℃)
PREDICT_WINDOW_S = 600   # 趋势拟合窗口 (s)
PREDICT_HORIZON_S = 600  # 预测时域 (s)


def dew_point(temp_c, rh_percent):
    """Magnus-Tetens 露点温度 (℃)。支持标量与 Series。RH 截断到 [1, 100]。"""
    rh = np.clip(np.asarray(rh_percent, dtype=float), 1.0, 100.0)
    t = np.asarray(temp_c, dtype=float)
    gamma = np.log(rh / 100.0) + MAGNUS_B * t / (MAGNUS_C + t)
    td = MAGNUS_C * gamma / (MAGNUS_B - gamma)
    if np.isscalar(temp_c) and np.isscalar(rh_percent):
        return float(td)
    return td


def dew_point_margin(cabinet_temp_c, humidity_rh):
    """凝露裕度 = 柜体表面温度 - 露点温度 (℃)。裕度越小风险越高。"""
    return np.asarray(cabinet_temp_c, dtype=float) - dew_point(cabinet_temp_c, humidity_rh)


@dataclass
class CondensationRisk:
    """凝露风险评估结果。"""
    margin_now: float          # 当前裕度 ℃
    at_risk_now: bool          # 当前裕度已低于阈值
    predicted_risk: bool       # horizon 内预测将跌破阈值
    eta_s: Optional[float]     # 预计跌破时间（秒），无风险为 None
    slope_c_per_s: float       # 裕度变化速率 ℃/s


class CondensationPredictor:
    """电气柜凝露预测器：当前裕度判定 + 线性趋势外推。"""

    def __init__(
        self,
        margin_warn_c: float = MARGIN_WARN_C,
        window_s: int = PREDICT_WINDOW_S,
        horizon_s: int = PREDICT_HORIZON_S,
    ):
        self.margin_warn = margin_warn_c
        self.window_s = window_s
        self.horizon_s = horizon_s

    def assess(self, timestamps: pd.Series, cabinet_temp: pd.Series, humidity: pd.Series) -> CondensationRisk:
        """输入最近 window 内的柜温与湿度序列，输出凝露风险。"""
        margin = dew_point_margin(cabinet_temp.to_numpy(), humidity.to_numpy())
        margin_now = float(margin[-1])
        at_risk = margin_now < self.margin_warn

        ts = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()
        win = ts >= ts[-1] - self.window_s
        t_w, m_w = ts[win], margin[win]
        slope, eta, predicted = 0.0, None, False
        if len(t_w) >= 10:
            slope = float(np.polyfit(t_w, m_w, 1)[0])
            if slope < 0 and not at_risk:
                eta = (margin_now - self.margin_warn) / (-slope)
                predicted = eta <= self.horizon_s

        return CondensationRisk(
            margin_now=round(margin_now, 2),
            at_risk_now=at_risk,
            predicted_risk=predicted,
            eta_s=round(eta, 1) if eta is not None else None,
            slope_c_per_s=slope,
        )
