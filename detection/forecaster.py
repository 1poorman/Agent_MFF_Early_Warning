"""L2 时序预测模型后端：统一接口，支持 GRU / PatchTST / TimesFM-2.5 可切换。

设计：
- `ForecastBackend`：统一预测接口（输入出水温度历史序列 -> 未来 horizon 预测数组）；
- 三个实现：
  - `GRUBackend`      ：现有快轨 FastTrackForecaster（STL 分解 + GRU+attention，残差学习）；
  - `PatchTSTBackend` ：现有精轨 PreciseTrackForecaster（多参数通道独立 PatchTST）；
  - `TimesFMBackend`  ：TimesFM-2.5 基础模型（transformers 移植版，本地权重 timesfm/weights）；
- `ForecastEngine`   ：按 config.detection.forecast_model 加载对应后端并做统一预测，返回
  {horizon_s, end_value, max_value, min_value, series, method} 结构（对齐 demo 前端契约）。

配置（config/settings.yaml detection 段）：
    forecast_model: gru | patchtst | timesfm
    forecast_horizon_s: 600      # 预测步长（秒），GRU/PatchTST 按窗口采样间隔=1s
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("mff_agent.detection.forecaster")

# 各后端预测输出步长（秒）：demo 前端约定 600s（10min）
DEFAULT_HORIZON_S = 600
# TimesFM 单次预测步数（config.horizon_length，不可改）
TIMESFM_HORIZON_STEPS = 128


# ---------------- 统一接口 ----------------

class ForecastBackend:
    """预测后端统一接口。"""

    name: str = "base"

    def predict(self, recent: pd.Series, horizon_s: int = DEFAULT_HORIZON_S) -> np.ndarray:
        """输入最近窗口序列（出水温度），返回未来 horizon_s 步预测（原尺度）。"""
        raise NotImplementedError


# ---------------- GRU 后端 ----------------

class GRUBackend(ForecastBackend):
    """快轨 GRU+attention（STL 分解残差学习）。"""

    name = "gru"

    def __init__(self, model=None, model_path: Optional[str] = None):
        from detection.fast_track import FastTrackForecaster
        if model is not None:
            self.model = model
        elif model_path:
            self.model = FastTrackForecaster.load(model_path)
        else:
            raise ValueError("GRUBackend 需要 model 或 model_path")

    def predict(self, recent: pd.Series, horizon_s: int = DEFAULT_HORIZON_S) -> np.ndarray:
        m = self.model
        horizon = min(horizon_s, m.horizon)
        pred = m.predict(recent.iloc[-m.window:])[:horizon]
        return np.asarray(pred, dtype=float)


# ---------------- PatchTST 后端 ----------------

class PatchTSTBackend(ForecastBackend):
    """精轨 PatchTST（多参数通道独立 Transformer）。"""

    name = "patchtst"

    def __init__(self, model=None, model_path: Optional[str] = None):
        from detection.precise_track import PreciseTrackForecaster
        if model is not None:
            self.model = model
        elif model_path:
            self.model = PreciseTrackForecaster.load(model_path)
        else:
            raise ValueError("PatchTSTBackend 需要 model 或 model_path")

    def predict(self, recent: pd.Series, horizon_s: int = DEFAULT_HORIZON_S) -> np.ndarray:
        m = self.model
        horizon = min(horizon_s, m.horizon)
        # 精轨为多参数预测：以出水温度列构造 DataFrame，仅取 outlet_temp
        df = pd.DataFrame({m.columns[0]: recent.iloc[-m.window:]})
        # 补足全部列（缺失列用 0，仅用于 shape 对齐，结果取 outlet_temp）
        for c in m.columns[1:]:
            if c not in df.columns:
                df[c] = 0.0
        preds = m.predict(df)
        pred = preds.get("outlet_temp", preds.get(m.columns[0]))[:horizon]
        return np.asarray(pred, dtype=float)


# ---------------- TimesFM 后端 ----------------

class TimesFMBackend(ForecastBackend):
    """TimesFM-2.5 基础模型（transformers 移植版）。

    使用本地权重 timesfm/weights（config.json + model.safetensors，925MB），
    无需联网下载。单次预测固定输出 128 步（config.horizon_length）。
    """

    name = "timesfm"

    def __init__(self, weights_dir: str = "timesfm/weights",
                 context_len: int = 1024, device: Optional[str] = None):
        from transformers.models.timesfm2_5.modeling_timesfm2_5 import (
            TimesFm2_5ModelForPrediction,
        )
        self.context_len = context_len
        self.device = device or ("cuda:1" if torch.cuda.device_count() > 1
                                 else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info("TimesFM-2.5 加载本地权重: %s (device=%s)", weights_dir, self.device)
        self.model = TimesFm2_5ModelForPrediction.from_pretrained(
            weights_dir, local_files_only=True)
        self.model = self.model.to(torch.float32).to(self.device).eval()
        logger.info("TimesFM-2.5 加载完成")

    def predict(self, recent: pd.Series, horizon_s: int = DEFAULT_HORIZON_S) -> np.ndarray:
        # 取最近 context_len 个点（不足则前补），转 tensor 后调用模型
        ctx = int(min(self.context_len, 1024))
        series = recent.to_numpy(dtype=float)
        if len(series) < ctx:
            series = np.concatenate([np.zeros(ctx - len(series)), series])
        else:
            series = series[-ctx:]
        t = torch.tensor(series, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(past_values=[t], forecast_context_len=ctx)
        pred = out.mean_predictions[0].detach().cpu().numpy()  # (128,)
        # 若请求步长小于 128，截断；大于则末值外推补齐（TimesFM 固定 128 步）
        n = len(pred)
        if horizon_s <= n:
            return pred[:horizon_s]
        ext = np.full(horizon_s - n, pred[-1], dtype=float)
        return np.concatenate([pred, ext])


# ---------------- 预测引擎 ----------------

class ForecastEngine:
    """按配置加载预测后端并统一输出。"""

    BACKENDS: Dict[str, type] = {
        "gru": GRUBackend,
        "patchtst": PatchTSTBackend,
        "timesfm": TimesFMBackend,
    }

    def __init__(self, backend_name: str = "gru",
                 fast_models: Optional[Dict] = None,
                 precise_model=None,
                 timesfm_weights_dir: str = "timesfm/weights",
                 horizon_s: int = DEFAULT_HORIZON_S,
                 models_dir: str = "models"):
        self.name = (backend_name or "gru").lower()
        if self.name not in self.BACKENDS:
            logger.warning("未知预测模型 %s，回退 gru", backend_name)
            self.name = "gru"
        self.horizon_s = horizon_s
        self.backend: Optional[ForecastBackend] = None
        self._fast_models = fast_models or {}
        self._precise_model = precise_model
        self._timesfm_weights = timesfm_weights_dir
        self._models_dir = models_dir
        self._load()

    def _load(self):
        """惰性加载对应后端（失败降级 GRU，仍失败则 None）。"""
        try:
            if self.name == "gru":
                m = self._fast_models.get("outlet_temp")
                if m is None:
                    from detection.fast_track import FastTrackForecaster
                    p = f"{self._models_dir}/fast_outlet_temp.pt"
                    m = FastTrackForecaster.load(p)
                self.backend = GRUBackend(model=m)
            elif self.name == "patchtst":
                if self._precise_model is not None:
                    self.backend = PatchTSTBackend(model=self._precise_model)
                else:
                    p = f"{self._models_dir}/precise.pt"
                    self.backend = PatchTSTBackend(model_path=p)
            elif self.name == "timesfm":
                self.backend = TimesFMBackend(weights_dir=self._timesfm_weights)
            logger.info("L2 预测后端就绪: %s", self.name)
        except Exception as e:
            logger.warning("L2 预测后端 %s 加载失败(%s)，降级 GRU", self.name, e)
            try:
                if self._fast_models.get("outlet_temp") is None:
                    from detection.fast_track import FastTrackForecaster
                    m = FastTrackForecaster.load(f"{self._models_dir}/fast_outlet_temp.pt")
                    self._fast_models["outlet_temp"] = m
                self.backend = GRUBackend(model=self._fast_models["outlet_temp"])
                self.name = "gru"
            except Exception as e2:
                logger.warning("GRU 降级也失败: %s", e2)
                self.backend = None

    def available(self) -> bool:
        return self.backend is not None

    def switch(self, backend_name: str, fast_models: Optional[Dict] = None,
               precise_model=None, models_dir: Optional[str] = None) -> bool:
        """运行时切换预测后端（失败保持原后端，返回是否成功）。"""
        name = (backend_name or "gru").lower()
        if name not in self.BACKENDS:
            logger.warning("未知预测模型 %s，忽略切换", backend_name)
            return False
        if name == self.name and self.backend is not None:
            return True
        self._fast_models = fast_models or self._fast_models
        if precise_model is not None:
            self._precise_model = precise_model
        if models_dir:
            self._models_dir = models_dir
        self.name = name
        try:
            self._load()
            return self.backend is not None
        except Exception as e:
            logger.warning("切换预测后端 %s 失败(%s)，保留 %s", name, e, self.name)
            return False

    def backend_window(self) -> int:
        """返回当前后端所需的最小输入窗口（条）。"""
        b = self.backend
        if b is None:
            return 0
        if isinstance(b, GRUBackend):
            return int(b.model.window)
        if isinstance(b, PatchTSTBackend):
            return int(b.model.window)
        if isinstance(b, TimesFMBackend):
            return int(min(b.context_len, 1024))
        return 0

    def exceedance_eta(self, series: pd.Series, threshold: float,
                       lookback_s: int = 600, horizon_s: int = 600) -> Optional[float]:
        """趋势越限预测：近 lookback 线性斜率外推，返回预计多少秒后越限（不越限返回 None）。

        GRU 后端复用 FastTrackForecaster.predict_exceedance；其余后端用通用线性外推。
        """
        b = self.backend
        if b is None:
            return None
        if isinstance(b, GRUBackend):
            return b.model.predict_exceedance(series, threshold,
                                              lookback_s=lookback_s, horizon_s=horizon_s)
        seg = series.iloc[-lookback_s:].to_numpy(dtype=float)
        if len(seg) < 60:
            return None
        slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0])
        cur = float(seg[-1])
        if slope <= 1e-6 or cur >= threshold:
            return 0.0 if cur >= threshold else None
        eta = (threshold - cur) / slope
        return float(eta) if 0 < eta <= horizon_s else None

    def forecast(self, recent: pd.Series) -> Optional[Dict]:
        """统一预测入口。返回 {horizon_s, end_value, max_value, min_value, series, method}。"""
        if self.backend is None:
            return None
        try:
            pred = self.backend.predict(recent, self.horizon_s)
            pred = np.asarray(pred, dtype=float)
            if len(pred) == 0:
                return None
            return {
                "horizon_s": self.horizon_s,
                "end_value": round(float(pred[-1]), 2),
                "max_value": round(float(pred.max()), 2),
                "min_value": round(float(pred.min()), 2),
                "series": [round(float(v), 2) for v in pred[:: max(1, len(pred) // 30)]],
                "method": self.name.upper(),
            }
        except Exception as e:
            logger.warning("预测失败(%s): %s", self.name, e)
            return None
