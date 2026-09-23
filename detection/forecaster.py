"""L2 时序预测模型后端：统一接口，支持 GRU / PatchTST / TimesFM-2.5 / Chronos-2 可切换。

设计：
- `ForecastBackend`：统一预测接口（输入出水温度历史序列 -> 未来 horizon 预测数组）；
- 四个实现：
  - `GRUBackend`      ：现有快轨 FastTrackForecaster（STL 分解 + GRU+attention，残差学习）；
  - `PatchTSTBackend` ：现有精轨 PreciseTrackForecaster（多参数通道独立 PatchTST）；
  - `TimesFMBackend`  ：TimesFM-2.5 基础模型（transformers 移植版，本地权重 timesfm/weights）；
  - `ChronosBackend`  ：Chronos-2 基础模型（本地源码 chronos-forecasting/src，权重 models/chronos-2）；
- `ForecastEngine`   ：按 config.detection.forecast_model 加载对应后端并做统一预测，返回
  {horizon_s, end_value, max_value, min_value, series, method} 结构（对齐 demo 前端契约）。

配置（config/settings.yaml detection 段）：
    forecast_model: gru | patchtst | timesfm | chronos
    forecast_horizon_s: 600      # 预测步长（秒），GRU/PatchTST 按窗口采样间隔=1s
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("mff_agent.detection.forecaster")

# 各后端预测输出步长（秒）：demo 前端约定 600s（10min）
DEFAULT_HORIZON_S = 600
# TimesFM 单次预测步数（config.horizon_length，不可改）
TIMESFM_HORIZON_STEPS = 128

# ---------------- 多变量输入契约 ----------------
# 物理依据（simulator/ 机理方程，覆盖「热源→换热→水力」链路）：
#   outlet_temp     目标变量；
#   inlet_temp      热平衡边界 T_out = T_in + Q/(c·ṁ)；
#   flow_rate       冷却能力 ṁ（流量下降→温升增大，堵塞/泄漏直接特征）；
#   electric_power  热输入 P（P=√3·U·I·cosφ，加热功率比例调节）；
#   furnace_temp    炉体蓄热状态（炉体-水换热的温度梯度来源）；
#   pressure        管网阻抗 P=P_s+R·Q²（与流量强耦合，堵塞/气蚀判别依据）。
MV_FEATURES = ["outlet_temp", "inlet_temp", "flow_rate",
               "electric_power", "furnace_temp", "pressure"]
MV_TARGET = "outlet_temp"


# ---------------- 统一接口 ----------------

class ForecastBackend:
    """预测后端统一接口。

    predict 的 recent 支持 pd.Series（单变量，目标列）或 pd.DataFrame
    （多变量，含 MV_FEATURES 列）；各按后端能力取用。
    """

    name: str = "base"
    multivariate: bool = False     # 是否利用多变量输入

    def predict(self, recent, horizon_s: int = DEFAULT_HORIZON_S) -> np.ndarray:
        """输入最近窗口（Series/DataFrame），返回未来 horizon_s 步目标列预测（原尺度）。"""
        raise NotImplementedError


def _as_series(recent) -> pd.Series:
    """从 Series/DataFrame 输入中提取目标列序列。"""
    if isinstance(recent, pd.DataFrame):
        return recent[MV_TARGET]
    return recent


# ---------------- GRU 后端 ----------------

class GRUBackend(ForecastBackend):
    """快轨 GRU+attention（残差学习）。

    model 可为单变量 FastTrackForecaster 或多变量 FastTrackMultivarForecaster
    （多变量模型需要 DataFrame 输入，含 MV_FEATURES 全部列）。
    """

    def __init__(self, model=None, model_path: Optional[str] = None):
        if model is not None:
            self.model = model
        elif model_path:
            self.model = _load_fast_model(model_path)
        else:
            raise ValueError("GRUBackend 需要 model 或 model_path")
        from detection.fast_track_mv import FastTrackMultivarForecaster
        self.multivariate = isinstance(self.model, FastTrackMultivarForecaster)

    def predict(self, recent, horizon_s: int = DEFAULT_HORIZON_S) -> np.ndarray:
        m = self.model
        horizon = min(horizon_s, m.horizon)
        if self.multivariate:
            if not isinstance(recent, pd.DataFrame):
                raise ValueError("多变量快轨需要 DataFrame 输入（含辅助通道列）")
            pred = m.predict(recent.iloc[-m.window:])
        else:
            pred = m.predict(_as_series(recent).iloc[-m.window:])
        return np.asarray(pred, dtype=float)[:horizon]


def _load_fast_model(model_path: str):
    """按文件内容自动识别单变量/多变量快轨模型。"""
    import torch as _t
    ckpt = _t.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "aux_cols" in ckpt:
        from detection.fast_track_mv import FastTrackMultivarForecaster
        return FastTrackMultivarForecaster.load(model_path)
    from detection.fast_track import FastTrackForecaster
    return FastTrackForecaster.load(model_path)


# ---------------- PatchTST 后端 ----------------

class PatchTSTBackend(ForecastBackend):
    """精轨 PatchTST（多参数通道独立 Transformer，天然多变量）。"""

    name = "patchtst"
    multivariate = True

    def __init__(self, model=None, model_path: Optional[str] = None):
        from detection.precise_track import PreciseTrackForecaster
        if model is not None:
            self.model = model
        elif model_path:
            self.model = PreciseTrackForecaster.load(model_path)
        else:
            raise ValueError("PatchTSTBackend 需要 model 或 model_path")

    def predict(self, recent, horizon_s: int = DEFAULT_HORIZON_S) -> np.ndarray:
        m = self.model
        horizon = min(horizon_s, m.horizon)
        if isinstance(recent, pd.DataFrame) and all(c in recent.columns for c in m.columns):
            # DataFrame 输入：直接取真实多参数列（比零填充精度高）
            df = recent.iloc[-m.window:][list(m.columns)]
        else:
            # Series 输入：以出水温度列构造 DataFrame，补足全部列（缺失列用 0）
            df = pd.DataFrame({m.columns[0]: _as_series(recent).iloc[-m.window:]})
            for c in m.columns[1:]:
                if c not in df.columns:
                    df[c] = 0.0
        preds = m.predict(df)
        pred = preds.get(MV_TARGET, preds.get(m.columns[0]))[:horizon]
        return np.asarray(pred, dtype=float)


# ---------------- TimesFM 后端 ----------------

class TimesFMBackend(ForecastBackend):
    """TimesFM-2.5 基础模型（transformers 移植版）。

    使用本地权重 timesfm/weights（config.json + model.safetensors，925MB），
    无需联网下载。单次预测固定输出 128 步（config.horizon_length）。

    单变量限制：transformers 移植版 past_values 仅接受 (batch, seq_len)
    张量，不支持跨通道信息共享的多变量输入（多通道只能当独立 batch 序列，
    无物理关联建模），故仅使用目标列。
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

    def predict(self, recent, horizon_s: int = DEFAULT_HORIZON_S) -> np.ndarray:
        # 取最近 context_len 个点（不足则前补），转 tensor 后调用模型
        ctx = int(min(self.context_len, 1024))
        series = _as_series(recent).to_numpy(dtype=float)
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


# ---------------- Chronos-2 后端 ----------------

class ChronosBackend(ForecastBackend):
    """Chronos-2 基础模型（amazon，120M，T5 架构，零样本多分位预测）。

    源码：项目本地 chronos-forecasting/src/chronos（sys.path 注入，与 timesfm/
    同等离线策略）；权重：models/chronos-2（safetensors 456MB）。
    底层 predict 返回 (batch, 21 quantiles, pred_len)，取 0.5 分位为中位预测。

    多变量：DataFrame 输入（含 MV_FEATURES 列）时按 (1, C, L) 张量送入，
    Chronos-2 原生支持跨变体信息共享的多变量预测，输出 (C, 21, pred_len)，
    取第 0 通道（目标列 outlet_temp）；Series 输入退化为单变量。
    """

    name = "chronos"
    multivariate = True
    MEDIAN_IDX = 10          # quantiles 列表中 0.5 的下标（0.01..0.99 共 21 个）

    def __init__(self, weights_dir: str = "models/chronos-2",
                 context_len: int = 1024, device: Optional[str] = None,
                 multivariate: bool = True):
        import sys
        src = str(Path(__file__).resolve().parent.parent / "chronos-forecasting" / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from chronos import Chronos2Pipeline
        self.context_len = int(context_len)
        self.force_univariate = not multivariate
        self.device = device or ("cuda:1" if torch.cuda.device_count() > 1
                                 else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info("Chronos-2 加载本地权重: %s (device=%s)", weights_dir, self.device)
        self.pipeline = Chronos2Pipeline.from_pretrained(
            weights_dir, device_map=self.device, dtype=torch.float32)
        logger.info("Chronos-2 加载完成")

    def predict(self, recent, horizon_s: int = DEFAULT_HORIZON_S) -> np.ndarray:
        if isinstance(recent, pd.DataFrame) and not self.force_univariate \
                and all(c in recent.columns for c in MV_FEATURES):
            X = recent[MV_FEATURES].iloc[-self.context_len:].to_numpy(dtype=np.float32)  # (L, C)
            t = torch.tensor(X.T).view(1, len(MV_FEATURES), -1)      # (1, C, ctx)
            ch = 0                                                   # 目标列通道
        else:
            series = _as_series(recent).to_numpy(dtype=np.float32)[-self.context_len:]
            t = torch.tensor(series).view(1, 1, -1)                  # (1, 1, ctx)
            ch = 0
        with torch.no_grad():
            out = self.pipeline.predict(t, prediction_length=int(horizon_s))
        pred = out[0][ch, self.MEDIAN_IDX].detach().cpu().numpy()    # (pred_len,)
        return np.asarray(pred, dtype=float)


# ---------------- 预测引擎 ----------------

class ForecastEngine:
    """按配置加载预测后端并统一输出。"""

    BACKENDS: Dict[str, type] = {
        "gru": GRUBackend,
        "patchtst": PatchTSTBackend,
        "timesfm": TimesFMBackend,
        "chronos": ChronosBackend,
    }

    def __init__(self, backend_name: str = "gru",
                 fast_models: Optional[Dict] = None,
                 precise_model=None,
                 timesfm_weights_dir: str = "timesfm/weights",
                 chronos_weights_dir: str = "models/chronos-2",
                 chronos_context: int = 1024,
                 fast_mv_model: str = "models/fast_outlet_temp_mv.pt",
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
        self._chronos_weights = chronos_weights_dir
        self._chronos_context = chronos_context
        self._fast_mv_model = fast_mv_model
        self._models_dir = models_dir
        self._load()

    def _load(self):
        """惰性加载对应后端（失败降级 GRU，仍失败则 None）。

        GRU 后端自动选型：多变量快轨权重（fast_outlet_temp_mv.pt）存在则优先
        （多变量输入），否则回退单变量快轨。
        """
        try:
            if self.name == "gru":
                m = None
                # 多变量快轨优先（若权重存在），否则用注入的单变量模型/文件
                mv_p = Path(self._fast_mv_model)
                if mv_p.exists():
                    from detection.fast_track_mv import FastTrackMultivarForecaster
                    m = FastTrackMultivarForecaster.load(str(mv_p))
                    logger.info("快轨多变量模型: %s (aux=%s)", mv_p, m.aux_cols)
                else:
                    m = self._fast_models.get("outlet_temp")
                    if m is None:
                        from detection.fast_track import FastTrackForecaster
                        m = FastTrackForecaster.load(f"{self._models_dir}/fast_outlet_temp.pt")
                self.backend = GRUBackend(model=m)
            elif self.name == "patchtst":
                if self._precise_model is not None:
                    self.backend = PatchTSTBackend(model=self._precise_model)
                else:
                    p = f"{self._models_dir}/precise.pt"
                    self.backend = PatchTSTBackend(model_path=p)
            elif self.name == "timesfm":
                self.backend = TimesFMBackend(weights_dir=self._timesfm_weights)
            elif self.name == "chronos":
                self.backend = ChronosBackend(weights_dir=self._chronos_weights,
                                              context_len=self._chronos_context)
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
        if isinstance(b, ChronosBackend):
            return int(b.context_len)
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

    def backend_features(self) -> Optional[List[str]]:
        """当前后端实际使用的输入特征列（多变量后端返回列名列表）。"""
        b = self.backend
        if b is None:
            return None
        if isinstance(b, ChronosBackend) and not b.force_univariate:
            return list(MV_FEATURES)
        if isinstance(b, PatchTSTBackend):
            return list(b.model.columns)
        if isinstance(b, GRUBackend) and b.multivariate:
            return [b.model.target] + list(b.model.aux_cols)
        return [MV_TARGET]

    def forecast(self, recent) -> Optional[Dict]:
        """统一预测入口。

        recent 支持 pd.Series（单变量：目标列历史）或 pd.DataFrame（多变量：
        含 MV_FEATURES 列，多变量后端自动利用辅助通道）。返回
        {horizon_s, end_value, max_value, min_value, series, method}。
        """
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
