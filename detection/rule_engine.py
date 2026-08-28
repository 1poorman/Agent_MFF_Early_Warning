"""L1 规则预警引擎（快/浅层，<10ms）。

规则体系：
- 绝对阈值规则：题目阈值表（出水温度/温差/压力/流量）+ 知识库（电导率）
- 进水温度季节动态修正（滚动 24h 基线上浮）
- 衍生特征规则：单位电耗温升率漂移、P-Q 特性偏移度
- 凝露规则：露点裕度实时判定（趋势预测由 CondensationPredictor 承担）
- 组合逻辑：多参数联合（如 流量<80% AND 温度>55℃ → 积热风险升级）

单条判定走 evaluate_row()（无状态、微秒级）；
批量/带状态判定走 evaluate()（季节阈值、衍生特征、凝露趋势）。
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tools.dew_point import CondensationPredictor, dew_point_margin
from tools.ts_features import (
    RATED_FLOW_LPS,
    heating_rate_per_power,
    pq_offset,
    seasonal_inlet_threshold,
)

LEVEL_YELLOW = "yellow"  # L1 触发黄色预警（题目约定）


@dataclass
class RuleThresholds:
    """L1 阈值字典（可配置，默认对齐题目与知识库模块二）。"""
    outlet_temp_high: float = 55.0        # ℃
    inlet_temp_high_base: float = 35.0    # ℃（季节修正基线）
    delta_t_high: float = 25.0            # ℃
    pressure_low_kpa: float = 150.0       # kPa
    pressure_high_kpa: float = 300.0      # kPa
    flow_low_ratio: float = 0.8           # 额定流量比例
    flow_high_ratio: float = 1.2
    rated_flow_lps: float = RATED_FLOW_LPS
    conductivity_high: float = 800.0      # µS/cm（外循环/软化水；内循环纯水应为 10）
    dew_margin_warn_c: float = 3.0        # 凝露裕度 ℃
    heat_rate_drift: float = 0.15         # 单位电耗温升率漂移 15%
    pq_offset_limit: float = 0.10         # P-Q 特性偏移 10%
    pressure_osc_std_kpa: float = 3.0     # 压力震荡阈值（去趋势 std kPa，气蚀特征）
    pressure_osc_window: int = 60         # 压力震荡检测滑动窗口（秒）

    @classmethod
    def from_settings(cls) -> "RuleThresholds":
        """从集中配置（config/settings.yaml rules 段）构造。"""
        from config import get_settings
        return get_settings().to_rule_thresholds()


@dataclass
class Alert:
    """L1 预警输出。"""
    timestamp: pd.Timestamp
    rule_id: str
    level: str
    message: str
    value: float

    def as_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "rule_id": self.rule_id,
            "level": self.level,
            "message": self.message,
            "value": round(float(self.value), 2),
        }


class RuleEngine:
    """L1 规则引擎。"""

    def __init__(self, thresholds: Optional[RuleThresholds] = None):
        self.th = thresholds or RuleThresholds()
        self.dew = CondensationPredictor(margin_warn_c=self.th.dew_margin_warn_c)
        # 压力震荡检测的流式滑动窗口（气蚀特征：去趋势 std，不依赖系统级状态）
        self._press_hist: deque = deque(maxlen=max(int(self.th.pressure_osc_window), 10))

    def reset(self) -> None:
        """清空带状态的滑动窗口（新一轮监测开始时调用）。"""
        self._press_hist.clear()

    @staticmethod
    def _detrended_std(values: List[float]) -> Optional[float]:
        """剔除线性趋势后残差 std（专测震荡幅度，爬升趋势不计入）。"""
        v = np.asarray(values, dtype=float)
        if len(v) < 10:
            return None
        t = np.arange(len(v))
        slope, intercept = np.polyfit(t, v, 1)
        resid = v - (slope * t + intercept)
        return float(np.std(resid))

    def _pressure_oscillation(self, pressure: Optional[float]) -> Optional[float]:
        """更新压力滑窗并返回当前窗口的去趋势 std（样本不足返回 None）。"""
        if pressure is None:
            return None
        try:
            p = float(pressure)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(p):
            return None
        self._press_hist.append(p)
        return self._detrended_std(list(self._press_hist))

    # ---------------- 单条判定（<10ms） ----------------

    def evaluate_row(self, row: Dict, inlet_threshold: Optional[float] = None) -> List[Alert]:
        """无状态瞬时规则判定。inlet_threshold 可由季节修正外部注入。"""
        th = self.th
        ts = row.get("timestamp", pd.NaT)
        alerts: List[Alert] = []

        def add(rule_id, message, value):
            alerts.append(Alert(ts, rule_id, LEVEL_YELLOW, message, value))

        if row["outlet_temp"] > th.outlet_temp_high:
            add("OUTLET_TEMP_HIGH", f"出水温度 {row['outlet_temp']}℃ 超限(>{th.outlet_temp_high}℃)", row["outlet_temp"])

        in_th = inlet_threshold if inlet_threshold is not None else th.inlet_temp_high_base
        if row["inlet_temp"] > in_th:
            add("INLET_TEMP_HIGH", f"进水温度 {row['inlet_temp']}℃ 超动态阈值({in_th:.1f}℃)", row["inlet_temp"])

        delta_t = row["outlet_temp"] - row["inlet_temp"]
        if delta_t > th.delta_t_high:
            add("DELTA_T_HIGH", f"进出水温差 {delta_t:.1f}℃ 超限(>{th.delta_t_high}℃)", delta_t)

        if row["pressure"] < th.pressure_low_kpa:
            add("PRESSURE_LOW", f"压力 {row['pressure']}kPa 低于下限(<{th.pressure_low_kpa}kPa)", row["pressure"])
        elif row["pressure"] > th.pressure_high_kpa:
            add("PRESSURE_HIGH", f"压力 {row['pressure']}kPa 高于上限(>{th.pressure_high_kpa}kPa)", row["pressure"])

        flow_ratio = row["flow_rate"] / th.rated_flow_lps
        if flow_ratio < th.flow_low_ratio:
            add("FLOW_LOW", f"流量 {row['flow_rate']}L/s 低于额定 80%", row["flow_rate"])
        elif flow_ratio > th.flow_high_ratio:
            add("FLOW_HIGH", f"流量 {row['flow_rate']}L/s 高于额定 120%", row["flow_rate"])

        if row["conductivity"] > th.conductivity_high:
            add("CONDUCTIVITY_HIGH", f"电导率 {row['conductivity']}µS/cm 超限", row["conductivity"])

        margin = float(dew_point_margin(row["cabinet_temp"], row["cabinet_humidity"]))
        if margin < th.dew_margin_warn_c:
            add("DEW_MARGIN_LOW", f"电气柜凝露裕度 {margin:.1f}℃ 低于 {th.dew_margin_warn_c}℃", margin)

        # 湿度显著升高 -> 疑似泄漏水汽（知识库：泄漏使环境湿度升高）
        if row["cabinet_humidity"] > 70.0:
            add("HUMIDITY_HIGH", f"湿度 {row['cabinet_humidity']:.0f}%RH 超 70%RH：疑似泄漏水汽", row["cabinet_humidity"])

        # 组合逻辑：流量不足 + 出水高温 → 积热风险
        if flow_ratio < th.flow_low_ratio and row["outlet_temp"] > th.outlet_temp_high:
            add("COMBO_HEAT_BUILDUP", "流量<80%额定 且 出水温度超限：线圈积热风险", row["outlet_temp"])
        # 组合逻辑：压力偏低 + 湿度高 → 疑似泄漏（压力阈值 180kPa 覆盖泄漏中期）
        if row["pressure"] < 180.0 and row["cabinet_humidity"] > 70.0:
            add("COMBO_LEAK_SUSPECT", "压力偏低 且 湿度>70%RH：疑似管路泄漏", row["pressure"])

        # 压力震荡（气蚀特征）：滑动窗口去趋势 std 超阈（绝对阈值规则对围绕
        # 正常带的低频振荡失明，气蚀典型振幅 6~12kPa、正常≈0.5kPa，区分度充分）
        osc = self._pressure_oscillation(row.get("pressure"))
        if osc is not None and osc > th.pressure_osc_std_kpa:
            add("PRESSURE_OSC",
                f"压力波动 {osc:.1f}kPa 超 {th.pressure_osc_std_kpa:.0f}kPa（去趋势）：疑似水泵气蚀",
                round(osc, 2))

        return alerts

    # ---------------- 批量判定（含状态规则） ----------------

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """批量评估：瞬时规则（季节修正阈值）+ 衍生特征规则 + 凝露趋势预测。"""
        records: List[Dict] = []
        th = self.th

        # 1) 季节动态阈值
        in_th_series = seasonal_inlet_threshold(df["inlet_temp"], th.inlet_temp_high_base)

        # 2) 逐行瞬时规则
        for i, row in enumerate(df.itertuples(index=False)):
            row_d = row._asdict() if hasattr(row, "_asdict") else dict(row)
            in_th = float(in_th_series.iloc[i]) if not np.isnan(in_th_series.iloc[i]) else None
            for a in self.evaluate_row(row_d, inlet_threshold=in_th):
                records.append(a.as_dict())

        # 3) 衍生特征规则
        records += self._heat_rate_rule(df)
        records += self._pq_offset_rule(df)

        # 4) 凝露趋势预测（滑动窗口评估，末段输出）
        records += self._condensation_rule(df)

        if not records:
            return pd.DataFrame(columns=["timestamp", "rule_id", "level", "message", "value"])
        return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

    # ---------------- 衍生特征规则 ----------------

    def _heat_rate_rule(self, df: pd.DataFrame) -> List[Dict]:
        ratio = heating_rate_per_power(df["furnace_temp"], df["electric_power"])
        out = []
        # 基线：熔炼/保温稳态段的滚动中位数
        steady = df["operating_condition"].isin(["melting", "holding"])
        baseline = ratio[steady & ratio.notna()].median()
        if not np.isnan(baseline) and abs(baseline) > 1e-6:
            drift = (ratio - baseline).abs() / abs(baseline)
            hit = (drift > self.th.heat_rate_drift) & steady & ratio.notna()
            for i in np.where(hit)[0]:
                out.append(Alert(
                    df["timestamp"].iloc[i], "HEAT_RATE_DRIFT", LEVEL_YELLOW,
                    f"单位电耗温升率漂移 {drift.iloc[i]:.0%}(>{self.th.heat_rate_drift:.0%})：换热效率衰减前兆",
                    float(ratio.iloc[i]),
                ).as_dict())
        return out

    def _pq_offset_rule(self, df: pd.DataFrame) -> List[Dict]:
        offset = pq_offset(df["pressure"], df["flow_rate"])
        hit = offset.abs() > self.th.pq_offset_limit
        return [
            Alert(
                df["timestamp"].iloc[i], "PQ_OFFSET", LEVEL_YELLOW,
                f"压力-流量特性偏移 {offset.iloc[i]:+.0%}(|限|{self.th.pq_offset_limit:.0%})：管网阻抗变化前兆",
                float(offset.iloc[i]),
            ).as_dict()
            for i in np.where(hit)[0]
        ]

    def _condensation_rule(self, df: pd.DataFrame) -> List[Dict]:
        """每 60s 评估一次凝露趋势，命中即输出预测预警。"""
        out = []
        step = 60
        for end in range(self.dew.window_s, len(df), step):
            win = df.iloc[end - self.dew.window_s:end]
            risk = self.dew.assess(win["timestamp"], win["cabinet_temp"], win["cabinet_humidity"])
            if risk.predicted_risk and not risk.at_risk_now:
                out.append(Alert(
                    df["timestamp"].iloc[end - 1], "DEW_PREDICT", LEVEL_YELLOW,
                    f"预测 {risk.eta_s/60:.0f}min 后电气柜凝露裕度跌破 {self.dew.margin_warn}℃"
                    f"（当前 {risk.margin_now}℃）",
                    risk.margin_now,
                ).as_dict())
        return out
