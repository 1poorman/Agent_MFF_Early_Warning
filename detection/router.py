"""模型路由器：快轨 / 精轨 分流 + 降级保护。

路由规则（design/BLUEPRINT.md M2）：
1. 默认快轨（单参数 ≤3、步长 ≤10min）
2. 升级精轨：①参数 ≥4 ②步长 >10min ③异常残差超阈复核 —— 任一满足
3. 异常并行轨始终在线（由 FastAnomalyDetector 承载）
4. 降级保护：精轨不可用/超时回退快轨并标注置信度降级
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .fast_track import FastTrackForecaster
from .precise_track import PreciseTrackForecaster

FAST_MAX_PARAMS = 3
FAST_MAX_HORIZON_S = 600  # 10min


@dataclass
class RouteResult:
    """路由输出。"""
    track: str                      # "fast" / "precise" / "fast_degraded"
    predictions: Dict[str, np.ndarray]
    horizon_s: int
    degraded: bool = False
    reason: str = ""


class ModelRouter:
    """快/精轨路由器。"""

    def __init__(self,
                 fast_models: Optional[Dict[str, FastTrackForecaster]] = None,
                 precise_model: Optional[PreciseTrackForecaster] = None,
                 precise_available: bool = True):
        self.fast_models = fast_models or {}
        self.precise_model = precise_model
        self.precise_available = precise_available

    def decide_track(self, columns: List[str], horizon_s: int,
                     anomaly_score: float = 0.0, anomaly_threshold: float = 0.6) -> str:
        """路由判定（纯规则，<1ms）。"""
        if anomaly_score > anomaly_threshold:
            return "precise"                      # 规则③残差超阈复核
        if len(columns) > FAST_MAX_PARAMS:        # 规则①多参数
            return "precise"
        if horizon_s > FAST_MAX_HORIZON_S:        # 规则②长步长
            return "precise"
        return "fast"

    def forecast(self, df: pd.DataFrame, columns: List[str], horizon_s: int,
                 anomaly_score: float = 0.0) -> RouteResult:
        """执行路由预测，含降级保护。"""
        track = self.decide_track(columns, horizon_s, anomaly_score)

        if track == "precise":
            if self.precise_available and self.precise_model is not None:
                try:
                    preds = self.precise_model.predict(df)
                    # 截取请求的列与步长
                    preds = {c: preds[c][:horizon_s] for c in columns if c in preds}
                    return RouteResult("precise", preds, horizon_s, reason="多参数/长步长/残差复核")
                except Exception as e:  # 精轨异常 -> 降级
                    track = "fast_degraded"
            else:
                track = "fast_degraded"

        if track == "fast_degraded":
            # 回退快轨：逐列预测，步长截断到快轨上限并外推标注
            preds = {}
            for c in columns:
                if c in self.fast_models:
                    p = self.fast_models[c].predict(df[c])
                    preds[c] = p[:min(horizon_s, len(p))]
            return RouteResult("fast_degraded", preds, min(horizon_s, FAST_MAX_HORIZON_S),
                               degraded=True, reason="精轨不可用，快轨降级接管(置信度降级)")

        # 快轨
        preds = {}
        for c in columns:
            if c in self.fast_models:
                preds[c] = self.fast_models[c].predict(df[c])[:horizon_s]
        return RouteResult("fast", preds, horizon_s, reason="单参数短步长")
